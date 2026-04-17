"""Lightweight SMPL motion lib for the Python inference pipeline.

Mirrors the training-code preprocessing in
`gear_sonic/utils/motion_lib/motion_lib_base.py` — but only the slice needed
to drive `SonicSMPLInference`, i.e. per-frame SMPL joints + SMPL root quat
(Z-up, base-rot removed) + retargeted robot DOF (IsaacLab order) at target_fps.

Paired files loaded from disk:
    smpl_pkl   : `{smpl_dir}/{name}.pkl`    keys: pose_aa (T, 72), transl, smpl_joints (T, 24, 3), fps
    robot_pkl  : `{robot_dir}/{dataset}/{name}.pkl` → {name: {dof (T', 29, MuJoCo), fps, root_rot, ...}}

Time alignment: SMPL pkl is already at target_fps=50 (2002 frames for ~40 s).
Robot pkl is at 30 fps (1202 frames) and gets linearly interpolated to 50 Hz
to match SMPL frame count. Joint order permuted MuJoCo → IsaacLab via
`G1_MUJOCO_TO_ISAACLAB_DOF` (see gear_sonic/envs/manager_env/robots/g1.py:93).

Root-quat preprocessing (matches commands.py:1324-1340):
    root_aa     = pose_aa[:, :3]
    root_quat   = angle_axis_to_quaternion(root_aa)    # wxyz
    if smpl_y_up: root_quat = R_xrot_90 @ root_quat    # Y-up → Z-up
    root_quat   = quat_mul(root_quat, [0.5,-0.5,-0.5,-0.5])  # remove SMPL base rest rot

This keeps obs semantics identical to training; the VR3PT/SMPL encoders were
trained to this exact contract and cannot be retrained.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch


G1_MUJOCO_TO_ISAACLAB_DOF = [
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24,
    18, 25, 19, 26, 20, 27, 21, 28,
]

SMPL_NUM_JOINTS = 24
TARGET_FPS = 50
SMPL_DT_FUTURE_REF_FRAMES = 0.02  # 20 ms → frame_skip=1 at 50 Hz
DEFAULT_WRIST_DOF_IDX = (23, 24, 25, 26, 27, 28)


def _angle_axis_to_quat_wxyz(aa: torch.Tensor) -> torch.Tensor:
    """Axis-angle → unit quaternion [w, x, y, z]. Matches
    torch_transform.angle_axis_to_quaternion (the convention used in commands.py)."""
    theta = aa.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    half = theta * 0.5
    w = torch.cos(half)
    xyz = aa * (torch.sin(half) / theta)
    return torch.cat([w, xyz], dim=-1)


def _quat_mul_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product for wxyz quaternions."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack([ow, ox, oy, oz], dim=-1)


def _ytoz_up_wxyz(q: torch.Tensor) -> torch.Tensor:
    """Rotate SMPL root quat from Y-up to Z-up by left-multiplying 90° about X."""
    # angle_axis_to_quat([π/2, 0, 0]) = [cos(π/4), sin(π/4), 0, 0]
    c = float(np.cos(np.pi / 4))
    s = float(np.sin(np.pi / 4))
    base = torch.tensor([c, s, 0.0, 0.0], dtype=q.dtype, device=q.device)
    base = base.expand_as(q)
    return _quat_mul_wxyz(base, q)


def _remove_smpl_base_rot_wxyz(q: torch.Tensor) -> torch.Tensor:
    """Conjugate out the SMPL [0.5, 0.5, 0.5, 0.5] rest-pose rotation."""
    inv = torch.tensor([0.5, -0.5, -0.5, -0.5], dtype=q.dtype, device=q.device)
    inv = inv.expand_as(q)
    return _quat_mul_wxyz(q, inv)


def _linear_resample(data: np.ndarray, fps_src: float, fps_tgt: float) -> np.ndarray:
    """Linearly resample a [T, ...] array from fps_src to fps_tgt.

    Uses the same arange(0, duration, 1/target_fps) formula as
    torch_humanoid_batch._compute_frame_blend so the resulting frame count
    matches the training-code interpolation (and matches the pre-resampled
    SMPL pkl produced by the training pipeline).
    """
    if abs(fps_src - fps_tgt) < 1e-6:
        return data
    n_src = data.shape[0]
    duration = (n_src - 1) / fps_src
    tgt_times = np.arange(0.0, duration, 1.0 / fps_tgt, dtype=np.float32)
    phase = tgt_times / duration
    idx0 = np.floor(phase * (n_src - 1)).astype(np.int64)
    idx1 = np.minimum(idx0 + 1, n_src - 1)
    blend = (phase * (n_src - 1) - idx0).astype(np.float32)
    for _ in range(data.ndim - 1):
        blend = blend[..., None]
    return data[idx0] * (1.0 - blend) + data[idx1] * blend


@dataclass
class _MotionEntry:
    name: str
    num_frames: int
    # All at TARGET_FPS, CPU torch tensors for flexible later device transfer.
    smpl_joints: torch.Tensor           # [T, 24, 3]
    smpl_root_quat_w: torch.Tensor      # [T, 4] wxyz, Z-up, base-rot removed
    dof_pos_il: torch.Tensor            # [T, 29] IsaacLab order
    robot_root_pos_w: torch.Tensor      # [T, 3]
    robot_root_quat_w_wxyz: torch.Tensor  # [T, 4] converted from pkl xyzw


class SmplMotionLib:
    """Batched SMPL+robot motion source for `SonicSMPLInference`.

    Select N motions (one per env) at construction. `sample_future(time_steps)`
    returns (smpl_joints_future, smpl_root_quat_future_wxyz, wrist_dof_future)
    for a 10-frame window at dt=0.02 s.
    """

    def __init__(
        self,
        smpl_dir: str | Path,
        robot_dir: str | Path,
        motion_names: list[str],
        device: str | torch.device = "cuda",
        smpl_y_up: bool = True,
        wrist_dof_idx: tuple[int, ...] = DEFAULT_WRIST_DOF_IDX,
        num_future_frames: int = 10,
    ):
        self.smpl_dir = Path(smpl_dir)
        self.robot_dir = Path(robot_dir)
        self.device = torch.device(device)
        self.smpl_y_up = smpl_y_up
        self.wrist_dof_idx = wrist_dof_idx
        self.num_future_frames = num_future_frames
        self.target_fps = TARGET_FPS
        # frame_skip = smpl_dt_future_ref_frames * target_fps = 0.02 * 50 = 1
        self.frame_skip = int(round(SMPL_DT_FUTURE_REF_FRAMES * TARGET_FPS))

        self.motions: list[_MotionEntry] = [self._load_pair(n) for n in motion_names]
        self.motion_num_steps = torch.tensor(
            [m.num_frames for m in self.motions], device=self.device, dtype=torch.long
        )
        self.max_step = int(self.motion_num_steps.min().item())

        # future-frame offsets [F] shared across envs
        self.future_offsets = (
            torch.arange(num_future_frames, device=self.device, dtype=torch.long)
            * self.frame_skip
        )

    def _load_pair(self, name: str) -> _MotionEntry:
        smpl_path = self.smpl_dir / f"{name}.pkl"
        robot_candidates = sorted(self.robot_dir.rglob(f"{name}.pkl"))
        if not robot_candidates:
            raise FileNotFoundError(f"No robot pkl for motion '{name}' under {self.robot_dir}")
        robot_path = robot_candidates[0]

        smpl = joblib.load(smpl_path)
        robot_outer = joblib.load(robot_path)
        # Robot pkl is wrapped as {name: {...}}
        if isinstance(robot_outer, dict) and name in robot_outer:
            robot = robot_outer[name]
        else:
            robot = next(iter(robot_outer.values())) if isinstance(robot_outer, dict) else robot_outer

        pose_aa = np.asarray(smpl["pose_aa"], dtype=np.float32)  # [T_smpl, 72]
        smpl_joints = np.asarray(smpl["smpl_joints"], dtype=np.float32)  # [T_smpl, 24, 3]
        smpl_fps = float(smpl["fps"])
        if abs(smpl_fps - TARGET_FPS) > 1e-6:
            pose_aa = _linear_resample(pose_aa, smpl_fps, TARGET_FPS)
            smpl_joints = _linear_resample(smpl_joints, smpl_fps, TARGET_FPS)

        dof_mj = np.asarray(robot["dof"], dtype=np.float32)  # [T_rob, 29] MuJoCo order
        robot_root_pos = np.asarray(robot["root_trans_offset"], dtype=np.float32)  # [T_rob, 3]
        # Robot pkl stores root_rot as xyzw; Isaac Lab uses wxyz
        robot_root_quat_xyzw = np.asarray(robot["root_rot"], dtype=np.float32)  # [T_rob, 4]
        robot_root_quat_wxyz = np.concatenate(
            [robot_root_quat_xyzw[:, 3:4], robot_root_quat_xyzw[:, :3]], axis=-1
        )
        robot_fps = float(robot["fps"])
        if abs(robot_fps - TARGET_FPS) > 1e-6:
            dof_mj = _linear_resample(dof_mj, robot_fps, TARGET_FPS)
            robot_root_pos = _linear_resample(robot_root_pos, robot_fps, TARGET_FPS)
            robot_root_quat_wxyz = _linear_resample(robot_root_quat_wxyz, robot_fps, TARGET_FPS)
            # renormalize after linear interp
            robot_root_quat_wxyz /= np.linalg.norm(
                robot_root_quat_wxyz, axis=-1, keepdims=True
            ).clip(min=1e-9)
        dof_il = dof_mj[:, G1_MUJOCO_TO_ISAACLAB_DOF]

        # Align frame counts (they should match after interpolation; pick min)
        T = min(pose_aa.shape[0], smpl_joints.shape[0], dof_il.shape[0], robot_root_pos.shape[0])
        pose_aa = pose_aa[:T]
        smpl_joints = smpl_joints[:T]
        dof_il = dof_il[:T]
        robot_root_pos = robot_root_pos[:T]
        robot_root_quat_wxyz = robot_root_quat_wxyz[:T]

        # Root quat: pose_aa[:, :3] → quat wxyz, Y→Z up, remove_smpl_base_rot
        root_aa_t = torch.from_numpy(pose_aa[:, :3])
        root_quat = _angle_axis_to_quat_wxyz(root_aa_t)
        if self.smpl_y_up:
            root_quat = _ytoz_up_wxyz(root_quat)
        root_quat = _remove_smpl_base_rot_wxyz(root_quat)

        return _MotionEntry(
            name=name,
            num_frames=T,
            smpl_joints=torch.from_numpy(smpl_joints).to(self.device),
            smpl_root_quat_w=root_quat.to(self.device),
            dof_pos_il=torch.from_numpy(dof_il).to(self.device),
            robot_root_pos_w=torch.from_numpy(robot_root_pos).to(self.device),
            robot_root_quat_w_wxyz=torch.from_numpy(robot_root_quat_wxyz).to(self.device),
        )

    def get_initial_state(self) -> dict[str, torch.Tensor]:
        """Frame-0 robot state for all N motions, used to pose the sim at reset."""
        N = len(self.motions)
        root_pos = torch.stack([m.robot_root_pos_w[0] for m in self.motions], dim=0)
        root_quat = torch.stack([m.robot_root_quat_w_wxyz[0] for m in self.motions], dim=0)
        dof_pos = torch.stack([m.dof_pos_il[0] for m in self.motions], dim=0)
        return {
            "root_pos_w": root_pos,          # [N, 3]
            "root_quat_w_wxyz": root_quat,   # [N, 4]
            "dof_pos_il": dof_pos,           # [N, 29]
        }

    def sample_future(
        self,
        time_steps: torch.Tensor,  # [N] long, current policy tick per env
    ) -> dict[str, torch.Tensor]:
        """Return a dict of future-frame tensors for a 10-frame window.

        Indices are clipped to each motion's last valid frame (same as
        commands.py `smpl_future_time_steps`).

        Returns:
            smpl_joints_future_w:       [N, F, 24, 3]
            smpl_root_quat_future_w:    [N, F, 4]        wxyz Z-up base-rot-removed
            wrist_dof_future:           [N, F, 6]         IL indices wrist_dof_idx
            dof_pos_ref:                [N, 29]           IL order, current frame (for debugging)
        """
        assert time_steps.shape == (len(self.motions),)
        time_steps = time_steps.to(self.device).long()
        F = self.num_future_frames
        N = time_steps.shape[0]

        # [N, F] absolute indices, clipped per-motion
        abs_idx = time_steps[:, None] + self.future_offsets[None, :]
        abs_idx = torch.minimum(abs_idx, self.motion_num_steps[:, None] - 1)

        smpl_joints_future = torch.empty(
            N, F, SMPL_NUM_JOINTS, 3, device=self.device, dtype=torch.float32
        )
        smpl_root_quat_future = torch.empty(
            N, F, 4, device=self.device, dtype=torch.float32
        )
        wrist_dof_future = torch.empty(
            N, F, len(self.wrist_dof_idx), device=self.device, dtype=torch.float32
        )
        dof_pos_ref = torch.empty(N, 29, device=self.device, dtype=torch.float32)

        for i, motion in enumerate(self.motions):
            idx = abs_idx[i]
            smpl_joints_future[i] = motion.smpl_joints[idx]
            smpl_root_quat_future[i] = motion.smpl_root_quat_w[idx]
            wrist_dof_future[i] = motion.dof_pos_il[idx][:, list(self.wrist_dof_idx)]
            dof_pos_ref[i] = motion.dof_pos_il[time_steps[i].clamp(max=motion.num_frames - 1)]

        return {
            "smpl_joints_future_w": smpl_joints_future,
            "smpl_root_quat_future_w_wxyz": smpl_root_quat_future,
            "wrist_dof_future": wrist_dof_future,
            "dof_pos_ref": dof_pos_ref,
        }
