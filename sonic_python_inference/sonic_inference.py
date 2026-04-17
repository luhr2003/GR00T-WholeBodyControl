"""Python inference pipeline for SONIC VR-3pt tracking, N envs in parallel.

Semantics follow the C++ deploy (gear_sonic_deploy/src/g1/g1_deploy_onnx_ref):
    - Sim 200 Hz, policy 50 Hz (decimation 4), planner 10 Hz (every 5th policy step).
    - Planner output is 64 frames @ 30 Hz → resampled to 50 Hz and cached.
    - Encoder (teleop mode, id=1) reads 10 frames @ step=5 from that cache.
    - Decoder reads token_flattened (64) ‖ proprioception history (930) → action [29].
    - motor_target_il[i] = default_il[i] + action_il[i] * action_scale_il[i].

DOF order — important, the pipeline crosses two orderings and we keep the API
boundary in IsaacLab/URDF order:
    * Planner `context_mujoco_qpos` input + output trajectory → **MuJoCo** order
    * Encoder `motion_joint_positions_lowerbody_10frame_step5` → **MuJoCo** order
      (first 12 entries of the planner trajectory's 29 joints, already MJ ordered)
    * Decoder proprio (`his_body_joint_positions|velocities|last_actions`) → **IsaacLab** order
    * Decoder action output → **IsaacLab** order
    * `default_angles` + `action_scale` passed to __init__ → **IsaacLab** order
      (the returned target = default_il + scale_il * action_il, directly feedable
       into Isaac Lab's `robot.set_joint_position_target`)

Inputs from the caller (joint_pos, joint_vel to step()/reset()) are in
**IsaacLab** order. Internally we convert to MuJoCo only for the planner context.

Scope: G1 29 DOF body only. Dex3 fingers (7+7) bypass the policy in deploy, so they
are a separate concern handled outside this class.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from .sonic_planner_pool import (
    PLANNER_FRAME_DIM,
    PLANNER_OUTPUT_FRAMES,
    PlannerSessionPool,
)

# Opt-in telemetry: SONIC_DEBUG=N dumps per-tick diagnostics for the first N ticks
# (env 0 only). Unset / 0 → silent. Used to diagnose fall-over modes in stage1.
_SONIC_DEBUG_TICKS = int(os.environ.get("SONIC_DEBUG", "0"))

# Proprio feature order MUST match the Isaac Lab ObservationManager's concat
# order for the `policy` group, which iterates terms in **@configclass dataclass
# field declaration order** (NOT YAML dict order). For SONIC the relevant
# PolicyCfg fields (gear_sonic/envs/manager_env/mdp/observations.py:107-128) are:
#     base_ang_vel (line 107)
#     joint_pos    (line 108)   → joint_pos_rel
#     joint_vel    (line 109)   → joint_vel_rel
#     actions      (line 110)   → last_action
#     gravity_dir  (line 128)   ← last, far down the class body
# Explicit confirmation at observations.py:171 for PolicyAtmCfg:
#   "# Order matches PolicyCfg: base_ang_vel, joint_pos, joint_vel, actions, gravity_dir"
# Each term has history_length=10 → per-frame 93, total 930.
# joint_pos uses `joint_pos_rel = joint_pos - default_joint_pos` (isaaclab MDP
# builtin). joint_vel uses `joint_vel_rel = joint_vel - default_joint_vel`
# (default_joint_vel == 0 for G1 standing, so joint_vel_rel == joint_vel).
PROPRIO_KEYS_ORDER = (
    "his_base_angular_velocity_10frame_step1",  # 30
    "his_body_joint_positions_10frame_step1",  # 290  (joint_pos_rel)
    "his_body_joint_velocities_10frame_step1",  # 290  (joint_vel_rel)
    "his_last_actions_10frame_step1",  # 290
    "his_gravity_dir_10frame_step1",  # 30
)
PROPRIO_DIMS = {
    "his_base_angular_velocity_10frame_step1": 30,
    "his_body_joint_positions_10frame_step1": 290,
    "his_body_joint_velocities_10frame_step1": 290,
    "his_last_actions_10frame_step1": 290,
    "his_gravity_dir_10frame_step1": 30,
}
PROPRIO_TOTAL = sum(PROPRIO_DIMS.values())  # 930

ACTION_DIM = 29
NUM_JOINTS = 29
HIST_LEN = 10
NUM_LOWER_BODY_JOINTS = 12  # G1: 2×(hip_yaw,hip_roll,hip_pitch,knee,ankle_pitch,ankle_roll)

PLANNER_NATIVE_HZ = 30.0
POLICY_HZ = 50.0
PLANNER_EVERY_K_POLICY_STEPS = 5  # 50Hz / 10Hz = 5
RESAMPLED_FRAMES = int(math.ceil(PLANNER_OUTPUT_FRAMES * POLICY_HZ / PLANNER_NATIVE_HZ)) + 1

# Planner locomotion mode ids. Match deploy C++
# (localmotion_kplanner.hpp:527 "0=IDLE, 1=SLOW_WALK, 2=WALK, 3=RUN").
PLANNER_MODE_IDLE = 0
PLANNER_MODE_SLOW_WALK = 1
PLANNER_MODE_WALK = 2
PLANNER_MODE_RUN = 3

# `allowed_pred_num_tokens` mask — first 6 tokens allowed, last 5 masked.
# Matches deploy default (localmotion_kplanner_onnx.hpp:155-163).
ALLOWED_PRED_NUM_TOKENS = np.array(
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=np.int64
)
# Sentinel — passing -1 to the planner means "use default height, don't constrain".
PLANNER_HEIGHT_DEFAULT = -1.0

# Default standing height the planner was trained with. Matches the deploy
# config `config_.default_height` used by `InitializeContext`
# (localmotion_kplanner.hpp:591-624, :216 `default_height = 0.788740`).
PLANNER_CONTEXT_DEFAULT_HEIGHT = 0.788740


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """wxyz quaternion conjugate."""
    return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by wxyz quaternion q. Shapes: q[..., 4], v[..., 3]."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    # t = 2 * (q.xyz × v)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    # rotated = v + w*t + q.xyz × t
    rx = vx + w * tx + (y * tz - z * ty)
    ry = vy + w * ty + (z * tx - x * tz)
    rz = vz + w * tz + (x * ty - y * tx)
    return torch.stack([rx, ry, rz], dim=-1)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """wxyz quaternion multiply."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def quat_to_6d(q: torch.Tensor) -> torch.Tensor:
    """wxyz quaternion → 6D rot.

    Matches training/deploy convention: `R[..., :2].reshape(-1)` where R is the
    3x3 rotation matrix — first 2 columns, flattened ROW-WISE:
        [R00, R01, R10, R11, R20, R21].

    Cross-checked against:
      - training: gear_sonic/envs/manager_env/mdp/commands.py:1961-1962
        (`mat[..., :2].reshape(mat.shape[0], -1)`)
      - deploy:   g1_deploy_onnx_ref.cpp:679-683 (row-wise flatten comment).
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - z * w)
    r10 = 2 * (x * y + z * w)
    r11 = 1 - 2 * (x * x + z * z)
    r20 = 2 * (x * z - y * w)
    r21 = 2 * (y * z + x * w)
    return torch.stack([r00, r01, r10, r11, r20, r21], dim=-1)


def slerp_torch(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Batched wxyz slerp. q0, q1: [..., 4], t: [...] or broadcastable."""
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs().clamp(max=1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    small = sin_theta < 1e-6
    w0 = torch.where(small, 1.0 - t.unsqueeze(-1), torch.sin((1 - t.unsqueeze(-1)) * theta) / sin_theta)
    w1 = torch.where(small, t.unsqueeze(-1), torch.sin(t.unsqueeze(-1) * theta) / sin_theta)
    return w0 * q0 + w1 * q1


def resample_traj_30_to_50hz(
    traj_30hz: torch.Tensor, num_output: int = RESAMPLED_FRAMES
) -> torch.Tensor:
    """traj_30hz: [N, 64, 36] (root_pos 3 | root_quat_wxyz 4 | joints 29).
    Returns [N, num_output, 36] resampled to 50 Hz via linear + slerp.
    """
    N, T_in, D = traj_30hz.shape
    assert D == PLANNER_FRAME_DIM
    device = traj_30hz.device
    t_out = torch.arange(num_output, device=device, dtype=torch.float32) * (
        PLANNER_NATIVE_HZ / POLICY_HZ
    )
    idx0 = torch.clamp(t_out.long(), 0, T_in - 1)
    idx1 = torch.clamp(idx0 + 1, 0, T_in - 1)
    alpha = (t_out - idx0.float()).clamp(0.0, 1.0)  # [num_output]

    pos0 = traj_30hz[:, idx0, 0:3]
    pos1 = traj_30hz[:, idx1, 0:3]
    pos = pos0 + (pos1 - pos0) * alpha.view(1, -1, 1)

    q0 = traj_30hz[:, idx0, 3:7]
    q1 = traj_30hz[:, idx1, 3:7]
    a_q = alpha.view(1, -1).expand(N, -1)
    quat = slerp_torch(q0, q1, a_q)

    j0 = traj_30hz[:, idx0, 7:]
    j1 = traj_30hz[:, idx1, 7:]
    joints = j0 + (j1 - j0) * alpha.view(1, -1, 1)

    return torch.cat([pos, quat, joints], dim=-1)


class SonicVR3PTInference:
    """Batched SONIC VR-3pt inference for N Isaac Lab envs.

    Usage:
        infer = SonicVR3PTInference(num_envs=4, default_angles=..., action_scale=...)
        infer.reset(...)  # initialize histories + planner context from sim state
        for policy_tick in range(T):
            targets = infer.step(vr_pos_local, vr_orn_local, mode, move_dir, face_dir,
                                 robot_state)
    """

    def __init__(
        self,
        num_envs: int,
        encoder_onnx: str | Path,
        decoder_onnx: str | Path,
        planner_onnx: str | Path,
        default_angles: np.ndarray,  # [29] in IsaacLab DOF order
        action_scale: np.ndarray,  # [29] in IsaacLab DOF order
        isaaclab_to_mujoco_dof: list[int] | np.ndarray,  # [29], mj[i] = il[ix[i]]
        lower_body_joint_idx: list[int] | None = None,
        device: str = "cuda",
        device_id: int = 0,
        planner_serial: bool = False,
    ):
        self.N = num_envs
        self.device = torch.device(device)
        self.default_angles = torch.as_tensor(default_angles, dtype=torch.float32, device=self.device)
        self.action_scale = torch.as_tensor(action_scale, dtype=torch.float32, device=self.device)
        assert self.default_angles.shape == (ACTION_DIM,)
        assert self.action_scale.shape == (ACTION_DIM,)

        self.il_to_mj = torch.as_tensor(
            isaaclab_to_mujoco_dof, dtype=torch.long, device=self.device
        )
        assert self.il_to_mj.shape == (NUM_JOINTS,), (
            f"isaaclab_to_mujoco_dof must be length {NUM_JOINTS}"
        )

        # On the planner's MJ-ordered 29-joint array, the first 12 entries are
        # already the lower body in MJ order (L leg 0-5, R leg 6-11) — matching
        # the encoder's `motion_joint_positions_lowerbody_10frame_step5` spec.
        self.lower_body_joint_idx = (
            torch.arange(NUM_LOWER_BODY_JOINTS, device=self.device)
            if lower_body_joint_idx is None
            else torch.as_tensor(lower_body_joint_idx, dtype=torch.long, device=self.device)
        )
        assert self.lower_body_joint_idx.numel() == NUM_LOWER_BODY_JOINTS

        providers = [
            (
                "CUDAExecutionProvider",
                {"device_id": device_id, "cudnn_conv_algo_search": "DEFAULT"},
            ),
            "CPUExecutionProvider",
        ]
        self.enc_sess = ort.InferenceSession(str(encoder_onnx), providers=providers)
        self.dec_sess = ort.InferenceSession(str(decoder_onnx), providers=providers)
        self.planner_pool = PlannerSessionPool(
            planner_onnx, pool_size=num_envs, device_id=device_id, serial=planner_serial
        )

        # Histories: ring buffers, index 0 = oldest, index HIST_LEN-1 = newest
        self.his_ang_vel = torch.zeros(self.N, HIST_LEN, 3, device=self.device)
        self.his_joint_pos = torch.zeros(self.N, HIST_LEN, NUM_JOINTS, device=self.device)
        self.his_joint_vel = torch.zeros(self.N, HIST_LEN, NUM_JOINTS, device=self.device)
        self.his_last_action = torch.zeros(self.N, HIST_LEN, ACTION_DIM, device=self.device)
        self.his_gravity = torch.zeros(self.N, HIST_LEN, 3, device=self.device)
        # Gravity direction in base frame for upright robot = [0, 0, -1]
        self.his_gravity[..., 2] = -1.0

        # Planner context: 4 latest mujoco_qpos frames [N, 4, 36]
        self.planner_context = torch.zeros(
            self.N, 4, PLANNER_FRAME_DIM, device=self.device
        )
        self.planner_cache = torch.zeros(
            self.N, RESAMPLED_FRAMES, PLANNER_FRAME_DIM, device=self.device
        )
        self.planner_playback_idx = torch.zeros(self.N, device=self.device, dtype=torch.long)
        self.step_counter = 0

        # Encoder mode one-hot (teleop = id 1, 4 modes total)
        self.encoder_mode_4 = torch.zeros(self.N, 4, device=self.device)
        self.encoder_mode_4[:, 1] = 1.0

    # ------------------------------------------------------------
    # State initialization
    # ------------------------------------------------------------
    def reset(
        self,
        joint_pos: torch.Tensor,  # [N, 29] IsaacLab order
        root_pos: torch.Tensor,  # [N, 3]  (kept for signature compat, ignored)
        root_quat_wxyz: torch.Tensor,  # [N, 4] (kept for signature compat, ignored)
    ):
        """Fill ring buffers with the initial standing pose and seed planner context.

        Planner context follows the deploy's `InitializeContext`
        (localmotion_kplanner.hpp:591-624): all 4 frames set to a yaw-normalized
        standing pose — `[0, 0, default_height, 1, 0, 0, 0, joints_mj]` — NOT the
        robot's live world pose. The planner was trained to treat its context as
        an egocentric, zero-heading history; feeding live world xyz/yaw biases
        the output and on replan causes the arms to latch in odd poses. Live
        `root_pos` / `root_quat_wxyz` are accepted for API compatibility with
        stage1/stage2 but deliberately dropped here.

        Histories store IL order (decoder expects IL). Planner context stores MJ
        order (planner expects MJ). We convert once here.
        """
        del root_pos, root_quat_wxyz  # silently ignored; see docstring
        jp_il = joint_pos.to(self.device)
        jp_mj = jp_il[:, self.il_to_mj]
        # Histories store joint_pos_rel = joint_pos - default_joint_pos (IsaacLab MDP
        # builtin `joint_pos_rel`, observations.py:212-219). At reset joint_pos ≈
        # default_joint_pos so joint_pos_rel ≈ 0.
        self.his_joint_pos[:] = (jp_il - self.default_angles.unsqueeze(0)).unsqueeze(1)
        self.his_joint_vel.zero_()
        self.his_last_action.zero_()
        self.his_ang_vel.zero_()
        self.his_gravity.zero_()
        self.his_gravity[..., 2] = -1.0

        # Yaw-normalized init frame: xyz=[0,0,default_h], quat=identity, joints_mj.
        init_frame = torch.zeros(self.N, PLANNER_FRAME_DIM, device=self.device)
        init_frame[:, 2] = PLANNER_CONTEXT_DEFAULT_HEIGHT
        init_frame[:, 3] = 1.0  # quat w
        init_frame[:, 7:] = jp_mj
        self.planner_context[:] = init_frame.unsqueeze(1).expand(-1, 4, -1).clone()
        self.planner_cache.zero_()
        self.planner_playback_idx.zero_()
        self.step_counter = 0

    # ------------------------------------------------------------
    # Planner
    # ------------------------------------------------------------
    def _context_from_cache(self) -> torch.Tensor:
        """Build the next planner context from the planner's own 50 Hz cache.

        Mirrors deploy `UpdateContextFromMotion` (localmotion_kplanner.hpp:628-678):
            gen_time = playback_idx / 50
            for n in 0..3:  sample at t = gen_time + n / 30

        Cache joints are stored MJ-ordered (planner output is MJ; the deploy
        reorders MJ→IL when caching, then IL→MJ when feeding the context — net
        identity on the joint slice, so we skip the round-trip). Returns
        [N, 4, 36] MJ-ordered mujoco_qpos context.
        """
        L = self.planner_cache.shape[1]
        device = self.device
        t_offsets = torch.arange(4, device=device, dtype=torch.float32) * (50.0 / 30.0)
        idx_f = self.planner_playback_idx.view(-1, 1).float() + t_offsets.view(1, -1)
        idx0 = idx_f.long().clamp(0, L - 1)
        idx1 = (idx0 + 1).clamp(0, L - 1)
        alpha = (idx_f - idx0.float()).clamp(0.0, 1.0)
        n_range = torch.arange(self.N, device=device).view(-1, 1).expand(-1, 4)
        f0 = self.planner_cache[n_range, idx0]
        f1 = self.planner_cache[n_range, idx1]
        pos = f0[..., 0:3] + (f1[..., 0:3] - f0[..., 0:3]) * alpha.unsqueeze(-1)
        quat = slerp_torch(f0[..., 3:7], f1[..., 3:7], alpha)
        joints = f0[..., 7:] + (f1[..., 7:] - f0[..., 7:]) * alpha.unsqueeze(-1)
        return torch.cat([pos, quat, joints], dim=-1)

    def _run_planner(
        self,
        mode: torch.Tensor,  # [N] int64
        movement_direction: torch.Tensor,  # [N, 3]
        facing_direction: torch.Tensor,  # [N, 3]
        target_vel: torch.Tensor,  # [N]
        height: torch.Tensor,  # [N]
    ):
        ctx = self.planner_context.cpu().numpy().astype(np.float32)
        mv = movement_direction.cpu().numpy().astype(np.float32)
        fd = facing_direction.cpu().numpy().astype(np.float32)
        md = mode.cpu().numpy().astype(np.int64)
        tv = target_vel.cpu().numpy().astype(np.float32)
        ht = height.cpu().numpy().astype(np.float32)

        feeds = []
        # Planner ONNX is batch=1: all non-scalar inputs carry a leading [1, ...]
        # dim. Scalars (`target_vel`, `mode`, `random_seed`, `height`) are [1].
        # Shapes verified via `onnx.load(...).graph.input`.
        allowed_mask = ALLOWED_PRED_NUM_TOKENS.reshape(1, 11)
        for i in range(self.N):
            feeds.append(
                {
                    "context_mujoco_qpos": ctx[i : i + 1],  # [1, 4, 36]
                    "target_vel": tv[i : i + 1].reshape(1),
                    "mode": md[i : i + 1].reshape(1),
                    "movement_direction": mv[i : i + 1],  # [1, 3]
                    "facing_direction": fd[i : i + 1],  # [1, 3]
                    "random_seed": np.array([0], dtype=np.int64),
                    "has_specific_target": np.zeros((1, 1), dtype=np.int64),
                    "specific_target_positions": np.zeros((1, 4, 3), dtype=np.float32),
                    "specific_target_headings": np.zeros((1, 4), dtype=np.float32),
                    "allowed_pred_num_tokens": allowed_mask,
                    "height": ht[i : i + 1].reshape(1),
                }
            )
        traj, _ = self.planner_pool.run_batched(feeds)  # [N, 64, 36]
        traj_t = torch.as_tensor(traj, device=self.device, dtype=torch.float32)
        self.planner_cache[:] = resample_traj_30_to_50hz(traj_t, RESAMPLED_FRAMES)
        self.planner_playback_idx.zero_()

    def _sample_planner_obs(self, root_quat_wxyz: torch.Tensor):
        """Gather 10-frame (step=5) lowerbody pos/vel + motion_anchor_orientation.

        Velocity semantics MUST match training's `joint_vel_lower_body_multi_future`
        (commands.py:1754): motion_lib stores `dof_vel[t] = (pos[t+1]-pos[t]) / (1/50)`
        — an **instantaneous finite-diff at the motion's native 50 Hz rate**
        (torch_humanoid_batch.py:449) — then `get_dof_vel` samples THAT velocity
        tensor at `future_time_steps` which are step=5 apart. So each of the 10
        sampled velocities is a 50-Hz instantaneous velocity, NOT the average
        velocity over the 0.1s gap between sampled frames. We replicate this
        by sampling `cache[idx]` AND `cache[idx+1]` and diffing over 1/50 s.

        `motion_anchor_ori_b` is the RELATIVE rotation from the robot's current
        base to the planner's target root — not the robot's absolute orientation.
        Mirrors training `subtract_frame_transforms(robot, ref) → conj(robot) * ref`
        (observations.py:943-966).

        Returns:
            lower_pos [N, 120], lower_vel [N, 120], anchor_6d [N, 6]
        """
        # 10 sample points at step=5 starting from playback idx (10 Hz sampling
        # over the 50 Hz cache)
        idx = self.planner_playback_idx.view(-1, 1) + torch.arange(
            0, 10 * 5, 5, device=self.device
        ).view(1, -1)
        idx = idx.clamp(max=RESAMPLED_FRAMES - 1)  # [N, 10]
        idx_next = (idx + 1).clamp(max=RESAMPLED_FRAMES - 1)  # [N, 10]

        n_range = torch.arange(self.N, device=self.device).view(-1, 1).expand(-1, 10)
        frames = self.planner_cache[n_range, idx]  # [N, 10, 36]
        frames_next = self.planner_cache[n_range, idx_next]  # [N, 10, 36]

        joints = frames[..., 7:]  # [N, 10, 29]
        lower = joints[..., self.lower_body_joint_idx]  # [N, 10, 12]
        lower_pos = lower.reshape(self.N, -1)  # [N, 120]

        # Instantaneous 50 Hz finite-diff velocity at each sampled frame.
        lower_next = frames_next[..., 7:][..., self.lower_body_joint_idx]
        lower_vel_frames = (lower_next - lower) * POLICY_HZ  # dt = 1/50 s
        lower_vel = lower_vel_frames.reshape(self.N, -1)  # [N, 120]

        # motion_anchor_orientation: relative rotation base → planner target
        base_q = root_quat_wxyz.to(self.device)  # [N, 4]
        ref_q = frames[:, 0, 3:7]  # [N, 4]  planner target root quat at current playback frame
        rel_q = quat_mul(quat_conjugate(base_q), ref_q)
        anchor_6d = quat_to_6d(rel_q)  # [N, 6]
        return lower_pos, lower_vel, anchor_6d

    # ------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------
    def step(
        self,
        vr_3pt_position: torch.Tensor,  # [N, 9]  root-local (neck, L-wrist, R-wrist)
        vr_3pt_orientation: torch.Tensor,  # [N, 12] root-local wxyz for 3 points
        mode: torch.Tensor,  # [N] int
        movement_direction: torch.Tensor,  # [N, 3]
        facing_direction: torch.Tensor,  # [N, 3]
        target_vel: torch.Tensor,  # [N]
        height: torch.Tensor,  # [N]
        joint_pos: torch.Tensor,  # [N, 29] current measured joint pos, IsaacLab order
        joint_vel: torch.Tensor,  # [N, 29] current measured joint vel, IsaacLab order
        base_ang_vel: torch.Tensor,  # [N, 3] base frame angular velocity
        gravity_in_base: torch.Tensor,  # [N, 3] gravity direction in base frame
        root_pos: torch.Tensor,  # [N, 3] world pos
        root_quat_wxyz: torch.Tensor,  # [N, 4] world quat
    ) -> torch.Tensor:
        """Run one 50 Hz policy tick. Returns joint position targets [N, 29] in IsaacLab order.

        NOTE: `root_pos` is unused (planner context is self-consistent via cache
        feedback, not live world state). `root_quat_wxyz` is still consumed by
        `_sample_planner_obs` to compute `motion_anchor_orientation` = relative
        rotation from robot's base to planner target root.
        """
        del root_pos  # planner context no longer uses live world pos
        joint_pos_il = joint_pos.to(self.device)
        joint_vel_il = joint_vel.to(self.device)
        # 1. Planner every 5th tick — planner context is MJ-ordered.
        #    First call (step_counter == 0) uses the yaw-normalized init context
        #    seeded by reset(). Subsequent replans rebuild the 4-frame context
        #    from the planner's own 50 Hz cache (mirroring deploy
        #    `UpdateContextFromMotion`, localmotion_kplanner.hpp:628-678),
        #    NEVER from live world state.
        if self.step_counter % PLANNER_EVERY_K_POLICY_STEPS == 0:
            if self.step_counter > 0:
                self.planner_context = self._context_from_cache()
            self._run_planner(mode, movement_direction, facing_direction, target_vel, height)

        # 1.5. Update STATE histories with current values BEFORE running policy.
        # Training's Isaac Lab ObservationManager appends the current obs to the
        # CircularBuffer during obs gather, so at tick t the state history
        # INCLUDES state_t. The `last_action` term reads env.action_manager.action
        # which still holds action_{t-1} at that point, so action history is
        # updated AFTER the decoder below.
        self.his_ang_vel = torch.roll(self.his_ang_vel, -1, dims=1)
        self.his_ang_vel[:, -1] = base_ang_vel.to(self.device)
        self.his_joint_pos = torch.roll(self.his_joint_pos, -1, dims=1)
        self.his_joint_pos[:, -1] = joint_pos_il - self.default_angles.unsqueeze(0)
        self.his_joint_vel = torch.roll(self.his_joint_vel, -1, dims=1)
        self.his_joint_vel[:, -1] = joint_vel_il
        self.his_gravity = torch.roll(self.his_gravity, -1, dims=1)
        self.his_gravity[:, -1] = gravity_in_base.to(self.device)

        # 2. Gather encoder obs from planner cache + VR inputs
        lower_pos, lower_vel, anchor_6d = self._sample_planner_obs(root_quat_wxyz)
        teleop_obs = torch.cat(
            [
                lower_pos,  # 120
                lower_vel,  # 120
                vr_3pt_position.to(self.device),  # 9
                vr_3pt_orientation.to(self.device),  # 12
                anchor_6d,  # 6
            ],
            dim=-1,
        )  # [N, 267]

        _dbg = self.step_counter < _SONIC_DEBUG_TICKS
        if _dbg:
            pb = int(self.planner_playback_idx[0].item())
            pf0 = self.planner_cache[0, pb].cpu().numpy()
            rq = root_quat_wxyz[0].cpu().numpy()
            print(
                f"[SONIC_DEBUG t={self.step_counter}] pb_idx={pb}  "
                f"planner_frame0 xyz={pf0[:3].round(4).tolist()} "
                f"quat={pf0[3:7].round(4).tolist()} "
                f"joints_mj[0:6]={pf0[7:13].round(3).tolist()}"
            )
            print(
                f"[SONIC_DEBUG t={self.step_counter}] robot_quat={rq.round(4).tolist()}  "
                f"anchor_6d={anchor_6d[0].cpu().numpy().round(4).tolist()}  "
                f"lower_pos[0:6]={lower_pos[0, :6].cpu().numpy().round(3).tolist()}"
            )
            print(
                f"[SONIC_DEBUG t={self.step_counter}] "
                f"joint_pos_rel[0:6]={self.his_joint_pos[0, -1, :6].cpu().numpy().round(3).tolist()}  "
                f"joint_vel[0:6]={self.his_joint_vel[0, -1, :6].cpu().numpy().round(3).tolist()}  "
                f"gravity={self.his_gravity[0, -1].cpu().numpy().round(3).tolist()}"
            )

        # 3. Encoder ONNX
        token_flat = self.enc_sess.run(
            None, {"teleop_obs": teleop_obs.cpu().numpy().astype(np.float32)}
        )[0]  # [N, 64]

        # 4. Decoder ONNX. Proprio concat order MUST match PolicyCfg @configclass
        # field declaration order in gear_sonic/envs/manager_env/mdp/observations.py:70-157:
        #   base_ang_vel → joint_pos_rel → joint_vel_rel → last_action → gravity_dir.
        # Isaac Lab's ObservationManager concatenates by dataclass field order,
        # NOT by YAML dict order. Getting this wrong silently permutes the
        # decoder's input MLP.
        proprio = torch.cat(
            [
                self.his_ang_vel.reshape(self.N, -1),  # 30   base_ang_vel
                self.his_joint_pos.reshape(self.N, -1),  # 290  joint_pos_rel
                self.his_joint_vel.reshape(self.N, -1),  # 290  joint_vel_rel
                self.his_last_action.reshape(self.N, -1),  # 290  last_action
                self.his_gravity.reshape(self.N, -1),  # 30   gravity_dir
            ],
            dim=-1,
        )  # [N, 930]
        dec_in = np.concatenate(
            [token_flat, proprio.cpu().numpy().astype(np.float32)], axis=-1
        )  # [N, 994]
        action_np = self.dec_sess.run(None, {"decoder_input": dec_in})[0]  # [N, 29]
        action = torch.as_tensor(action_np, device=self.device, dtype=torch.float32)

        if _dbg:
            a0 = action[0].cpu().numpy()
            tok = token_flat[0]
            target = (self.default_angles + self.action_scale * action[0]).cpu().numpy()
            print(
                f"[SONIC_DEBUG t={self.step_counter}] "
                f"token_flat[0:6]={tok[:6].round(3).tolist()}  "
                f"action[0:6]={a0[:6].round(3).tolist()}  "
                f"action_norm={np.linalg.norm(a0):.3f}"
            )
            print(
                f"[SONIC_DEBUG t={self.step_counter}] "
                f"target[0:6]={target[:6].round(3).tolist()}  "
                f"default[0:6]={self.default_angles[:6].cpu().numpy().round(3).tolist()}"
            )

        # 5. Update ACTION history AFTER decoder so that at tick t+1 the last
        # entry is action_t (matches training's env.action_manager.action).
        self.his_last_action = torch.roll(self.his_last_action, -1, dims=1)
        self.his_last_action[:, -1] = action  # decoder action is IL-ordered

        # 6. Advance planner playback
        self.planner_playback_idx = (self.planner_playback_idx + 1).clamp(
            max=RESAMPLED_FRAMES - 1
        )
        self.step_counter += 1

        # 7. target_il = default_il + scale_il * action_il  (all IL-ordered)
        return self.default_angles.unsqueeze(0) + self.action_scale.unsqueeze(0) * action
