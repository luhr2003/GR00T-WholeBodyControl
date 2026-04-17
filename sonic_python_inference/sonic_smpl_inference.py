"""Python inference pipeline for SONIC SMPL-mode tracking, N envs in parallel.

Semantics follow the **training code** (gear_sonic Isaac Lab obs functions), not
the C++ deploy — the deploy has no SMPL path. Specifically:

    - Policy 50 Hz, decimation 4 (same as VR3PT).
    - No planner. SMPL motion data (retargeted DOF + raw SMPL joints + SMPL root
      pose) directly drives the encoder obs.
    - SMPL encoder: `[N, 840] → [N, 64]`, input packs 10 future frames at
      dt=0.02 s (frame_skip=1 at target_fps=50), each frame being
      [smpl_joints_local(72), smpl_root_ori_b_6d(6), wrist_dof_future(6)].
    - Decoder: identical to VR3PT — `[N, 994] = token(64) ‖ proprio(930)` → `[N, 29]`.
    - motor_target_il = default_il + action_scale_il * action_il.

Obs details cross-referenced to training code
(gear_sonic/envs/manager_env/mdp/observations.py and commands.py):

    smpl_joints_multi_future_local_nonflat  (observations.py:1716)
        ref_joints = motion_lib.smpl_joints (N, 10, 24, 3) — raw from the SMPL
            dataset, in SMPL world frame.
        ref_root_quat = smpl_root_quat_w_multi_future (wxyz, Z-up, after
            remove_smpl_base_rot). PER-FRAME canonicalisation — NOT the first
            frame, each of the 10 frames gets canonicalised by its own root.
        local = quat_apply(quat_inv(ref_root_quat), ref_joints)        → 72 per frame
    smpl_root_ori_b_multi_future  (observations.py:1625, commands.py:1343)
        6D of quat_mul(quat_inv(robot_anchor_quat_w), smpl_root_quat_w)  → 6 per frame
    joint_pos_multi_future_wrist_for_smpl  (observations.py:1567)
        motion_lib.get_dof_pos(...)[:, :, [23,24,25,26,27,28]]           → 6 per frame

Proprioception (930 D) layout is **IDENTICAL** to the VR3PT module — same
history ring buffers, same concat order (base_ang_vel, joint_pos_rel,
joint_vel_rel, last_action, gravity_dir). The decoder is also identical
(`decoder_dyn.onnx` reused).

Quaternion convention everywhere: wxyz (matches training / VR3PT pipeline).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from .sonic_inference import (
    ACTION_DIM,
    HIST_LEN,
    NUM_JOINTS,
    quat_conjugate,
    quat_mul,
    quat_rotate,
    quat_to_6d,
)


SMPL_NUM_FUTURE_FRAMES = 10  # config.yaml: manager_env.commands.motion.smpl_num_future_frames
SMPL_NUM_JOINTS = 24  # SMPL body joint count — shape (T, 24, 3)
SMPL_WRIST_DOF_IDX = (23, 24, 25, 26, 27, 28)  # IsaacLab DOF indices for wrist obs
SMPL_ENCODER_INPUT_DIM = (
    SMPL_NUM_FUTURE_FRAMES * (SMPL_NUM_JOINTS * 3 + 6 + len(SMPL_WRIST_DOF_IDX))
)  # 10 * (72 + 6 + 6) = 840

DECODER_INPUT_DIM = 994  # 64 token + 930 proprio (reuses VR3PT decoder ONNX)


def _quat_inv_wxyz(q: torch.Tensor) -> torch.Tensor:
    """Inverse of a unit quaternion == its conjugate."""
    return quat_conjugate(q)


def _smpl_joints_local_per_frame(
    smpl_joints: torch.Tensor,  # [N, F, J, 3] world frame (raw from pkl; per-frame root-canonical input)
    smpl_root_quat_w_wxyz: torch.Tensor,  # [N, F, 4] per-frame SMPL root quat in Z-up world frame
) -> torch.Tensor:
    """Rotate each frame's joints into that frame's own root frame.

    Mirrors gear_sonic/envs/manager_env/mdp/observations.py:1716
    (smpl_joints_multi_future_local):
        ref_root_quat_exp = smpl_root_quat_w.unsqueeze(-2).repeat(1, 1, J, 1)
        local = quat_apply(quat_inv(ref_root_quat_exp), ref_joints)

    Returns [N, F, J*3] flattened.
    """
    N, F, J, _ = smpl_joints.shape
    q_exp = smpl_root_quat_w_wxyz.unsqueeze(-2).expand(N, F, J, 4)
    q_inv = _quat_inv_wxyz(q_exp)  # [N, F, J, 4]
    local = quat_rotate(q_inv, smpl_joints)  # [N, F, J, 3]
    return local.reshape(N, F, J * 3)


def _smpl_root_ori_b_6d_per_frame(
    robot_anchor_quat_wxyz: torch.Tensor,  # [N, 4] current robot pelvis quat
    smpl_root_quat_w_wxyz: torch.Tensor,  # [N, F, 4] per-frame SMPL root quat
) -> torch.Tensor:
    """6D rotation of SMPL root relative to robot anchor, per future frame.

    Mirrors commands.py:1343 (smpl_root_quat_w_dif_l_multi_future):
        root_rot_dif = quat_mul(quat_inv(robot_anchor_quat).expand(F),
                                smpl_root_quat_w)
        6d = matrix_from_quat(...)[..., :2].reshape(-1, 6)   # row-wise flatten

    Returns [N, F, 6].
    """
    N, F, _ = smpl_root_quat_w_wxyz.shape
    anchor_inv = _quat_inv_wxyz(robot_anchor_quat_wxyz).unsqueeze(1).expand(N, F, 4)
    rel = quat_mul(anchor_inv, smpl_root_quat_w_wxyz)  # [N, F, 4]
    return quat_to_6d(rel)  # [N, F, 6]


class SonicSMPLInference:
    """Batched SONIC SMPL-mode inference for N Isaac Lab envs.

    Usage:
        infer = SonicSMPLInference(
            num_envs=4,
            smpl_encoder_onnx="sonic_python_inference/assets/smpl_encoder_dyn.onnx",
            decoder_onnx="sonic_python_inference/assets/decoder_dyn.onnx",
            default_angles=default_il,
            action_scale=scale_il,
            device="cuda",
        )
        infer.reset(joint_pos=q_il)
        for policy_tick in range(T):
            targets = infer.step(
                smpl_joints_future_w=...,       # [N, 10, 24, 3]
                smpl_root_quat_future_w_wxyz=..., # [N, 10, 4]  (already Y→Z, base-rot removed)
                wrist_dof_future=...,           # [N, 10, 6]  IL indices [23..28]
                joint_pos=..., joint_vel=...,
                base_ang_vel=..., gravity_in_base=...,
                root_quat_wxyz=...,             # [N, 4] robot pelvis quat
            )
    """

    def __init__(
        self,
        num_envs: int,
        smpl_encoder_onnx: str | Path,
        decoder_onnx: str | Path,
        default_angles: np.ndarray,  # [29] IsaacLab DOF order
        action_scale: np.ndarray,  # [29] IsaacLab DOF order
        device: str = "cuda",
        device_id: int = 0,
    ):
        self.N = num_envs
        self.device = torch.device(device)
        self.default_angles = torch.as_tensor(
            default_angles, dtype=torch.float32, device=self.device
        )
        self.action_scale = torch.as_tensor(
            action_scale, dtype=torch.float32, device=self.device
        )
        assert self.default_angles.shape == (ACTION_DIM,)
        assert self.action_scale.shape == (ACTION_DIM,)

        providers = [
            (
                "CUDAExecutionProvider",
                {"device_id": device_id, "cudnn_conv_algo_search": "DEFAULT"},
            ),
            "CPUExecutionProvider",
        ]
        self.enc_sess = ort.InferenceSession(str(smpl_encoder_onnx), providers=providers)
        self.dec_sess = ort.InferenceSession(str(decoder_onnx), providers=providers)

        # Proprioception ring buffers — IDENTICAL layout to VR3PT path
        # (see sonic_inference.py:269-276 and concat order at :516-525).
        self.his_ang_vel = torch.zeros(self.N, HIST_LEN, 3, device=self.device)
        self.his_joint_pos = torch.zeros(self.N, HIST_LEN, NUM_JOINTS, device=self.device)
        self.his_joint_vel = torch.zeros(self.N, HIST_LEN, NUM_JOINTS, device=self.device)
        self.his_last_action = torch.zeros(self.N, HIST_LEN, ACTION_DIM, device=self.device)
        self.his_gravity = torch.zeros(self.N, HIST_LEN, 3, device=self.device)
        self.his_gravity[..., 2] = -1.0

        self.step_counter = 0

    def reset(
        self,
        joint_pos: torch.Tensor,  # [N, 29] IsaacLab order
    ):
        """Seed histories with current pose, zeros elsewhere."""
        jp_il = joint_pos.to(self.device)
        self.his_joint_pos[:] = (jp_il - self.default_angles.unsqueeze(0)).unsqueeze(1)
        self.his_joint_vel.zero_()
        self.his_last_action.zero_()
        self.his_ang_vel.zero_()
        self.his_gravity.zero_()
        self.his_gravity[..., 2] = -1.0
        self.step_counter = 0

    def _build_encoder_obs(
        self,
        smpl_joints_future_w: torch.Tensor,  # [N, 10, 24, 3]
        smpl_root_quat_future_w_wxyz: torch.Tensor,  # [N, 10, 4]
        wrist_dof_future: torch.Tensor,  # [N, 10, 6]
        robot_anchor_quat_wxyz: torch.Tensor,  # [N, 4]
    ) -> np.ndarray:
        """Pack (N, 10, 84) → (N, 840). Per-frame concat order: [joints, root6d, wrist]."""
        assert smpl_joints_future_w.shape == (
            self.N,
            SMPL_NUM_FUTURE_FRAMES,
            SMPL_NUM_JOINTS,
            3,
        ), f"smpl_joints_future_w shape: {smpl_joints_future_w.shape}"
        assert smpl_root_quat_future_w_wxyz.shape == (self.N, SMPL_NUM_FUTURE_FRAMES, 4)
        assert wrist_dof_future.shape == (self.N, SMPL_NUM_FUTURE_FRAMES, len(SMPL_WRIST_DOF_IDX))
        assert robot_anchor_quat_wxyz.shape == (self.N, 4)

        joints_local = _smpl_joints_local_per_frame(
            smpl_joints_future_w.to(self.device),
            smpl_root_quat_future_w_wxyz.to(self.device),
        )  # [N, 10, 72]
        root_ori_6d = _smpl_root_ori_b_6d_per_frame(
            robot_anchor_quat_wxyz.to(self.device),
            smpl_root_quat_future_w_wxyz.to(self.device),
        )  # [N, 10, 6]
        wrist = wrist_dof_future.to(self.device)  # [N, 10, 6]

        # Per-frame order: [joints_local(72), root_ori_6d(6), wrist(6)] = 84
        # Order MUST match the ONNX encoder, which was exported with input packed
        # in config.yaml's encoders.smpl.inputs order — see
        # sonic_python_inference/scripts/export_dynamic_batch_onnx.py.
        per_frame = torch.cat([joints_local, root_ori_6d, wrist], dim=-1)  # [N, 10, 84]
        flat = per_frame.reshape(self.N, -1)  # [N, 840]
        assert flat.shape[-1] == SMPL_ENCODER_INPUT_DIM
        return flat.cpu().numpy().astype(np.float32)

    def step(
        self,
        smpl_joints_future_w: torch.Tensor,  # [N, 10, 24, 3] raw SMPL joints (world-ish, from pkl)
        smpl_root_quat_future_w_wxyz: torch.Tensor,  # [N, 10, 4] Z-up root quat, base-rot removed
        wrist_dof_future: torch.Tensor,  # [N, 10, 6] retargeted robot DOF @ IL [23..28]
        joint_pos: torch.Tensor,  # [N, 29] IL-order measured
        joint_vel: torch.Tensor,  # [N, 29] IL-order measured
        base_ang_vel: torch.Tensor,  # [N, 3] base frame
        gravity_in_base: torch.Tensor,  # [N, 3]
        root_quat_wxyz: torch.Tensor,  # [N, 4] robot pelvis quat (world)
    ) -> torch.Tensor:
        """One 50 Hz policy tick. Returns motor targets [N, 29] in IsaacLab order."""
        joint_pos_il = joint_pos.to(self.device)
        joint_vel_il = joint_vel.to(self.device)

        # Push CURRENT obs into histories BEFORE building proprio. In training
        # (IsaacLab ObsManager + CircularBuffer), obs functions push new values
        # first, then the policy reads `buffer[:, -1]` which is the current-tick
        # measurement. `last_action`, however, reflects the PREVIOUS tick's
        # action at obs-collection time — so it is pushed AFTER the policy runs
        # (below), which naturally makes it "previous" on the next tick.
        self.his_ang_vel = torch.roll(self.his_ang_vel, -1, dims=1)
        self.his_ang_vel[:, -1] = base_ang_vel.to(self.device)
        self.his_joint_pos = torch.roll(self.his_joint_pos, -1, dims=1)
        self.his_joint_pos[:, -1] = joint_pos_il - self.default_angles.unsqueeze(0)
        self.his_joint_vel = torch.roll(self.his_joint_vel, -1, dims=1)
        self.his_joint_vel[:, -1] = joint_vel_il
        self.his_gravity = torch.roll(self.his_gravity, -1, dims=1)
        self.his_gravity[:, -1] = gravity_in_base.to(self.device)

        smpl_obs = self._build_encoder_obs(
            smpl_joints_future_w,
            smpl_root_quat_future_w_wxyz,
            wrist_dof_future,
            root_quat_wxyz,
        )
        token_flat = self.enc_sess.run(None, {"smpl_obs": smpl_obs})[0]  # [N, 64]

        proprio = torch.cat(
            [
                self.his_ang_vel.reshape(self.N, -1),  # 30
                self.his_joint_pos.reshape(self.N, -1),  # 290
                self.his_joint_vel.reshape(self.N, -1),  # 290
                self.his_last_action.reshape(self.N, -1),  # 290 (action_{t-1})
                self.his_gravity.reshape(self.N, -1),  # 30
            ],
            dim=-1,
        )  # [N, 930]
        dec_in = np.concatenate(
            [token_flat, proprio.cpu().numpy().astype(np.float32)], axis=-1
        )  # [N, 994]
        assert dec_in.shape[-1] == DECODER_INPUT_DIM
        action_np = self.dec_sess.run(None, {"decoder_input": dec_in})[0]  # [N, 29]
        action = torch.as_tensor(action_np, device=self.device, dtype=torch.float32)

        # last_action lags by one tick: stored here so next tick sees it as "previous".
        self.his_last_action = torch.roll(self.his_last_action, -1, dims=1)
        self.his_last_action[:, -1] = action

        self.step_counter += 1

        return self.default_angles.unsqueeze(0) + self.action_scale.unsqueeze(0) * action
