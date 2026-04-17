"""SONIC VR-3pt inference using the deploy release ONNX (batch=1 encoder + decoder).

Parallel alternative to `sonic_inference.SonicVR3PTInference` (which depends on
our own dynamic-batch re-exported ONNX with per-field named inputs). This module
uses the *shipped* `gear_sonic_deploy/policy/release/{model_encoder,model_decoder}.onnx`
AS-IS — both batch=1, single flat `obs_dict` input — via an ORT session pool
(mirror of `PlannerSessionPool`). Same public API as the dynamic-batch class so
the Isaac Lab runner just swaps the implementation.

Frequencies (unchanged, identical to deploy):
    sim      200 Hz   (Isaac Lab dt=0.005)
    policy    50 Hz   (decimation=4)
    planner   10 Hz   (every 5th policy tick; output cached then resampled to 50 Hz)

Joint-ordering contract (verified against deploy C++ ref, see inline comments):
    * kinematic planner output qpos[:, :, 7:]                            MJ order
    * lowerbody encoder obs (first 12 of planner 29)                     MJ order
    * Isaac Lab joint_pos / joint_vel (caller input)                     IL order
    * decoder proprio `his_body_joint_positions_*`                       IL order, rel to default
    * decoder proprio `his_body_joint_velocities_*` / `his_last_actions_*`   IL order
    * decoder action output                                              IL order
    * PD target returned to Isaac Lab                                    IL order

The ONE difference from `sonic_inference.py` concat layout is the decoder:
deploy wants
    [token(64), ang_vel(30), joint_pos_rel(290), joint_vel(290), last_action(290), gravity(30)]
(= 994 total, per observation_config.yaml order), whereas our dynamic-batch
code placed gravity first. Here we match deploy.

Encoder obs_dict[1,1762] is built by packing fields in the order declared in
`gear_sonic_deploy/policy/release/observation_config.yaml`. For teleop mode
only 6 fields carry data; the rest stay zero (the encoder learned a mode-gate
from `encoder_mode_4`).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

# Opt-in telemetry: SONIC_DEBUG=N dumps per-tick diagnostics for the first N ticks
# (env 0 only). Unset / 0 → silent. Used to diagnose fall-over modes in stage1.
_SONIC_DEBUG_TICKS = int(os.environ.get("SONIC_DEBUG", "0"))

from .sonic_inference import (
    ACTION_DIM,
    ALLOWED_PRED_NUM_TOKENS,
    HIST_LEN,
    NUM_JOINTS,
    NUM_LOWER_BODY_JOINTS,
    PLANNER_CONTEXT_DEFAULT_HEIGHT,
    PLANNER_EVERY_K_POLICY_STEPS,
    POLICY_HZ,
    RESAMPLED_FRAMES,
    quat_conjugate,
    quat_mul,
    quat_to_6d,
    resample_traj_30_to_50hz,
    slerp_torch,
)
from .sonic_onnx_pool import (
    DECODER_ACTION_DIM,
    DECODER_OBS_DIM,
    ENCODER_OBS_DIM,
    ENCODER_TOKEN_DIM,
    SonicDecoderPool,
    SonicEncoderPool,
)
from .sonic_planner_pool import PLANNER_FRAME_DIM, PlannerSessionPool


# ---------------------------------------------------------------------------
# Encoder obs_dict[1, 1762] layout.
#
# DO NOT trust the deploy `observation_config.yaml` order (`encoder_mode_4(4) |
# motion_joint_positions_10frame_step5(290) | ...`) — that is documentation for
# a DIFFERENT export path and disagrees with the release ONNX. The release
# encoder was exported by `gear_sonic/utils/inference_helpers.py:200
# export_universal_token_encoders_as_onnx` whose wrapper packs:
#     obs_dict[:, 0]    = scalar float encoder_index  (gather → long → one-hot inside)
#     obs_dict[:, 1:]   = tokenizer_obs concat in `module.tokenizer_obs_names` order
#                         (only features in `features_needed` get usable slots,
#                          but the encoder_index TOKENIZER feature is *always*
#                          prepended as a 3-dim placeholder, followed by every
#                          feature consumed by g1/teleop/smpl encoders)
#
# Verified against the release ONNX by dumping all axis-2 `Slice` ops on
# `/Unsqueeze_2_output_0` (tokenizer_input). Sliced ranges + reshape targets:
#    tokenizer_input[  3: 583] dim 580  shape[10,58]  → command_multi_future_nonflat        (g1)
#    tokenizer_input[594: 600] dim  6   shape[6]      → motion_anchor_ori_b                  (teleop)
#    tokenizer_input[600: 660] dim 60   shape[10,6]   → motion_anchor_ori_b_mf_nonflat       (g1)
#    tokenizer_input[660: 900] dim 240  shape[240]    → command_multi_future_lower_body      (teleop)
#    tokenizer_input[900: 909] dim  9   shape[9]      → vr_3point_local_target               (teleop)
#    tokenizer_input[909: 921] dim 12   shape[12]     → vr_3point_local_orn_target           (teleop)
#    tokenizer_input[921:1641] dim 720  shape[10,72]  → smpl_joints_multi_future_local_nonflat (smpl)
#    tokenizer_input[1641:1701] dim 60  shape[10,6]   → smpl_root_ori_b_multi_future         (smpl)
#    tokenizer_input[1701:1761] dim 60  shape[10,6]   → joint_pos_multi_future_wrist_for_smpl (smpl)
# Gaps [0:3]=3 and [583:594]=11 are `encoder_index` and
# `command_z_multi_future_nonflat` placeholder slots (no encoder slices them).
#
# obs_dict = [1 | 1761] = 1762, so layout offsets below are in `obs_dict` space
# (tokenizer_input offset + 1).
# ---------------------------------------------------------------------------
ENCODER_FIELDS: tuple[tuple[str, int], ...] = (
    ("encoder_index_scalar", 1),              # obs_dict[0]   teleop = 1.0
    ("encoder_index_tokenizer_pad", 3),       # obs_dict[1:4]  unused 3-dim
    ("command_multi_future_nonflat", 580),    # obs_dict[4:584]   g1
    ("command_z_multi_future_nonflat", 11),   # obs_dict[584:595] unused (placeholder)
    ("motion_anchor_ori_b", 6),               # obs_dict[595:601] TELEOP
    ("motion_anchor_ori_b_mf_nonflat", 60),   # obs_dict[601:661] g1
    ("command_multi_future_lower_body", 240), # obs_dict[661:901] TELEOP  (pos120 | vel120)
    ("vr_3point_local_target", 9),            # obs_dict[901:910] TELEOP
    ("vr_3point_local_orn_target", 12),       # obs_dict[910:922] TELEOP
    ("smpl_joints_multi_future_local_nonflat", 720),  # obs_dict[922:1642] smpl
    ("smpl_root_ori_b_multi_future", 60),     # obs_dict[1642:1702] smpl
    ("joint_pos_multi_future_wrist_for_smpl", 60),    # obs_dict[1702:1762] smpl
)
ENCODER_OFFSETS: dict[str, int] = {}
_acc = 0
for _name, _dim in ENCODER_FIELDS:
    ENCODER_OFFSETS[_name] = _acc
    _acc += _dim
assert _acc == ENCODER_OBS_DIM, f"encoder layout total {_acc} != {ENCODER_OBS_DIM}"


# ---------------------------------------------------------------------------
# Decoder concat layout.
#
# `g1_dyn` decoder (inputs: ["token_flattened", "proprioception"], no tokenizer
# obs — see `gear_sonic/config/actor_critic/decoders/g1_dyn_mlp.yaml`) expects:
#   obs_dict = [encoded_tokens(64) | proprioception(930)] = 994.
#
# `proprioception = actor_obs` (flat). `actor_obs` is built by Isaac Lab's
# ObservationManager which iterates the terms in **@configclass dataclass field
# declaration order**, NOT YAML dict order. The relevant PolicyCfg fields
# (observations.py:107-128) appear in class-declaration order:
#     base_ang_vel (line 107)
#     joint_pos    (line 108)   → joint_pos_rel
#     joint_vel    (line 109)   → joint_vel_rel
#     actions      (line 110)   → last_action
#     gravity_dir  (line 128)   ← last, much further down the class body
# This matches the explicit PolicyAtmCfg comment at line 171:
#     "# Order matches PolicyCfg: base_ang_vel, joint_pos, joint_vel, actions, gravity_dir"
# So the true actor_obs concat is
#   [ang_vel(30) | jp_rel(290) | jv_rel(290) | last_action(290) | gravity(30)] = 930,
# and the decoder obs_dict is
#   [token(64) | ang_vel(30) | jp_rel(290) | jv_rel(290) | last_action(290) | gravity(30)].
# (Earlier this code placed gravity at the head; empirical test showed that is
#  ALSO wrong — the correct position is the tail per the dataclass order above.)
# ---------------------------------------------------------------------------
DECODER_FIELDS: tuple[tuple[str, int], ...] = (
    ("token_state", 64),
    ("his_base_angular_velocity_10frame_step1", 30),
    ("his_body_joint_positions_10frame_step1", 290),
    ("his_body_joint_velocities_10frame_step1", 290),
    ("his_last_actions_10frame_step1", 290),
    ("his_gravity_dir_10frame_step1", 30),
)
DECODER_OFFSETS: dict[str, int] = {}
_acc = 0
for _name, _dim in DECODER_FIELDS:
    DECODER_OFFSETS[_name] = _acc
    _acc += _dim
assert _acc == DECODER_OBS_DIM, f"decoder layout total {_acc} != {DECODER_OBS_DIM}"


class SonicVR3PTInferenceDeploy:
    """Batched SONIC VR-3pt inference using the deploy release ONNX."""

    def __init__(
        self,
        num_envs: int,
        encoder_onnx: str | Path,
        decoder_onnx: str | Path,
        planner_onnx: str | Path,
        default_angles: np.ndarray,  # [29] IsaacLab DOF order
        action_scale: np.ndarray,  # [29] IsaacLab DOF order
        isaaclab_to_mujoco_dof: list[int] | np.ndarray,  # [29], mj[i] = il[ix[i]]
        lower_body_joint_idx: list[int] | None = None,
        device: str = "cuda",
        device_id: int = 0,
        planner_serial: bool = False,
        onnx_serial: bool = False,
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

        self.il_to_mj = torch.as_tensor(
            isaaclab_to_mujoco_dof, dtype=torch.long, device=self.device
        )
        assert self.il_to_mj.shape == (NUM_JOINTS,)

        # Planner qpos joint slice is already MJ-ordered — first 12 entries are
        # the lower body in MJ order (matches encoder obs spec).
        self.lower_body_joint_idx = (
            torch.arange(NUM_LOWER_BODY_JOINTS, device=self.device)
            if lower_body_joint_idx is None
            else torch.as_tensor(
                lower_body_joint_idx, dtype=torch.long, device=self.device
            )
        )
        assert self.lower_body_joint_idx.numel() == NUM_LOWER_BODY_JOINTS

        self.encoder_pool = SonicEncoderPool(
            encoder_onnx, pool_size=num_envs, device_id=device_id, serial=onnx_serial
        )
        self.decoder_pool = SonicDecoderPool(
            decoder_onnx, pool_size=num_envs, device_id=device_id, serial=onnx_serial
        )
        self.planner_pool = PlannerSessionPool(
            planner_onnx,
            pool_size=num_envs,
            device_id=device_id,
            serial=planner_serial,
        )

        # Histories (ring buffers, index 0 oldest, index HIST_LEN-1 newest)
        self.his_ang_vel = torch.zeros(self.N, HIST_LEN, 3, device=self.device)
        self.his_joint_pos = torch.zeros(
            self.N, HIST_LEN, NUM_JOINTS, device=self.device
        )
        self.his_joint_vel = torch.zeros(
            self.N, HIST_LEN, NUM_JOINTS, device=self.device
        )
        self.his_last_action = torch.zeros(
            self.N, HIST_LEN, ACTION_DIM, device=self.device
        )
        self.his_gravity = torch.zeros(self.N, HIST_LEN, 3, device=self.device)
        self.his_gravity[..., 2] = -1.0  # upright default

        # Planner context — 4 latest mujoco_qpos frames (MJ order)
        self.planner_context = torch.zeros(
            self.N, 4, PLANNER_FRAME_DIM, device=self.device
        )
        self.planner_cache = torch.zeros(
            self.N, RESAMPLED_FRAMES, PLANNER_FRAME_DIM, device=self.device
        )
        self.planner_playback_idx = torch.zeros(
            self.N, device=self.device, dtype=torch.long
        )
        self.step_counter = 0

        # Encoder mode one-hot (teleop = id 1, 4 modes total)
        self.encoder_mode_4 = torch.zeros(self.N, 4, device=self.device)
        self.encoder_mode_4[:, 1] = 1.0

        # Pre-allocated pinned buffers for encoder/decoder concat (CPU → GPU)
        self._enc_buf = np.zeros((self.N, ENCODER_OBS_DIM), dtype=np.float32)
        self._dec_buf = np.zeros((self.N, DECODER_OBS_DIM), dtype=np.float32)

    # ------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------
    def reset(
        self,
        joint_pos: torch.Tensor,  # [N, 29] IL order
        root_pos: torch.Tensor,  # [N, 3]   (planner context uses yaw-normalized)
        root_quat_wxyz: torch.Tensor,  # [N, 4]
    ):
        """Seed histories + planner context. Mirrors `SonicVR3PTInference.reset`."""
        del root_pos, root_quat_wxyz  # planner context is yaw-normalized, see docstring of sonic_inference.reset
        jp_il = joint_pos.to(self.device)
        jp_mj = jp_il[:, self.il_to_mj]

        # joint_pos_rel = joint_pos - default_joint_pos (IsaacLab MDP builtin)
        self.his_joint_pos[:] = (
            jp_il - self.default_angles.unsqueeze(0)
        ).unsqueeze(1)
        self.his_joint_vel.zero_()
        self.his_last_action.zero_()
        self.his_ang_vel.zero_()
        self.his_gravity.zero_()
        self.his_gravity[..., 2] = -1.0

        init_frame = torch.zeros(self.N, PLANNER_FRAME_DIM, device=self.device)
        init_frame[:, 2] = PLANNER_CONTEXT_DEFAULT_HEIGHT
        init_frame[:, 3] = 1.0  # quat w
        init_frame[:, 7:] = jp_mj
        self.planner_context[:] = init_frame.unsqueeze(1).expand(-1, 4, -1).clone()
        self.planner_cache.zero_()
        self.planner_playback_idx.zero_()
        self.step_counter = 0

    # ------------------------------------------------------------
    # Planner @ 10 Hz
    # ------------------------------------------------------------
    def _context_from_cache(self) -> torch.Tensor:
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
        mode: torch.Tensor,
        movement_direction: torch.Tensor,
        facing_direction: torch.Tensor,
        target_vel: torch.Tensor,
        height: torch.Tensor,
    ):
        ctx = self.planner_context.cpu().numpy().astype(np.float32)
        mv = movement_direction.cpu().numpy().astype(np.float32)
        fd = facing_direction.cpu().numpy().astype(np.float32)
        md = mode.cpu().numpy().astype(np.int64)
        tv = target_vel.cpu().numpy().astype(np.float32)
        ht = height.cpu().numpy().astype(np.float32)
        allowed_mask = ALLOWED_PRED_NUM_TOKENS.reshape(1, 11)

        feeds = []
        for i in range(self.N):
            feeds.append(
                {
                    "context_mujoco_qpos": ctx[i : i + 1],
                    "target_vel": tv[i : i + 1].reshape(1),
                    "mode": md[i : i + 1].reshape(1),
                    "movement_direction": mv[i : i + 1],
                    "facing_direction": fd[i : i + 1],
                    "random_seed": np.array([0], dtype=np.int64),
                    "has_specific_target": np.zeros((1, 1), dtype=np.int64),
                    "specific_target_positions": np.zeros((1, 4, 3), dtype=np.float32),
                    "specific_target_headings": np.zeros((1, 4), dtype=np.float32),
                    "allowed_pred_num_tokens": allowed_mask,
                    "height": ht[i : i + 1].reshape(1),
                }
            )
        traj, _ = self.planner_pool.run_batched(feeds)  # [N, 64, 36] MJ-ordered
        traj_t = torch.as_tensor(traj, device=self.device, dtype=torch.float32)
        self.planner_cache[:] = resample_traj_30_to_50hz(traj_t, RESAMPLED_FRAMES)
        self.planner_playback_idx.zero_()

    def _sample_planner_obs(self, root_quat_wxyz: torch.Tensor):
        """Return (lower_pos[N,120], lower_vel[N,120], anchor_6d[N,6]), MJ-ordered joints.

        Velocity semantics mirror training: `motion_lib.get_dof_vel` returns the
        instantaneous 50-Hz finite-difference velocity at each queried frame
        (torch_humanoid_batch.py:449). Sampled at step=5 future frames, each of
        the 10 velocities is still a 50-Hz instantaneous velocity, NOT the
        average over the 100 ms gap between samples. Match that by diffing
        `cache[idx]` against `cache[idx+1]` (one 50-Hz step apart), not against
        `cache[idx-5]`.
        """
        idx = self.planner_playback_idx.view(-1, 1) + torch.arange(
            0, 10 * 5, 5, device=self.device
        ).view(1, -1)
        idx = idx.clamp(max=RESAMPLED_FRAMES - 1)
        idx_next = (idx + 1).clamp(max=RESAMPLED_FRAMES - 1)

        n_range = torch.arange(self.N, device=self.device).view(-1, 1).expand(-1, 10)
        frames = self.planner_cache[n_range, idx]  # [N, 10, 36]
        frames_next = self.planner_cache[n_range, idx_next]  # [N, 10, 36]

        joints = frames[..., 7:]  # [N, 10, 29] MJ order
        lower = joints[..., self.lower_body_joint_idx]  # [N, 10, 12] MJ lowerbody
        lower_pos = lower.reshape(self.N, -1)  # [N, 120]

        lower_next = frames_next[..., 7:][..., self.lower_body_joint_idx]  # [N, 10, 12]
        lower_vel_frames = (lower_next - lower) * POLICY_HZ  # dt = 1/50 s
        lower_vel = lower_vel_frames.reshape(self.N, -1)  # [N, 120]

        base_q = root_quat_wxyz.to(self.device)
        ref_q = frames[:, 0, 3:7]
        rel_q = quat_mul(quat_conjugate(base_q), ref_q)
        anchor_6d = quat_to_6d(rel_q)  # [N, 6]
        return lower_pos, lower_vel, anchor_6d

    # ------------------------------------------------------------
    # Concat builders (fill pre-allocated buffers, keep zero-fill implicit)
    # ------------------------------------------------------------
    def _build_encoder_obs(
        self,
        lower_pos: torch.Tensor,  # [N, 120]  MJ lowerbody order × 10 frames
        lower_vel: torch.Tensor,  # [N, 120]  MJ lowerbody order × 10 frames
        anchor_6d: torch.Tensor,  # [N, 6]    quat_to_6d(q_robot^-1 · q_ref)
        vr_3pt_position: torch.Tensor,  # [N, 9]  in REF-anchor local frame
        vr_3pt_orientation: torch.Tensor,  # [N, 12] wxyz, in REF-anchor local frame
    ) -> np.ndarray:
        """Fill the [N, 1762] `obs_dict` buffer for the deploy encoder.

        Matches TRAINING export semantics exactly (not deploy YAML). Teleop only
        needs 4 tokenizer fields populated; everything else is left at 0 (the
        FSQ-selected teleop encoder never reads them due to the encoder_index
        one-hot gate inside the ONNX).

        Training encoder (teleop) input features (sonic_release/config.yaml:94-99):
            - command_multi_future_lower_body  (cat[pos10×12 | vel10×12] = 240)
            - vr_3point_local_target            (9)
            - vr_3point_local_orn_target        (12)
            - motion_anchor_ori_b               (6)
        """
        buf = self._enc_buf
        buf.fill(0.0)  # reset every tick; unfilled fields stay zero

        # obs_dict[0] = scalar encoder_index (long-cast inside graph → one-hot).
        # Training wrapper layout: teleop is encoder_index = 1
        # (encoders_to_iterate = [g1, teleop, smpl] per config.yaml declaration).
        buf[:, 0] = 1.0

        # motion_anchor_ori_b  (training: subtract_frame_transforms(robot, ref) 6D)
        off = ENCODER_OFFSETS["motion_anchor_ori_b"]
        buf[:, off : off + 6] = anchor_6d.detach().cpu().numpy().astype(np.float32)

        # command_multi_future_lower_body = cat([pos_mf, vel_mf], dim=1)
        # Training: each is shape [N, num_future=10, num_lower=12]; cat on dim=1
        # → [N, 20, 12]; flatten → [pos_f0…pos_f9 | vel_f0…vel_f9] * 12 joints.
        # lower_pos/lower_vel are already flattened to that layout (MJ lowerbody
        # order, oldest-future-first per `_sample_planner_obs`).
        off = ENCODER_OFFSETS["command_multi_future_lower_body"]
        cmd_lb = torch.cat([lower_pos, lower_vel], dim=1)  # [N, 240]
        buf[:, off : off + 240] = cmd_lb.detach().cpu().numpy().astype(np.float32)

        # vr_3point_local_target  (9)   — caller provides values already in
        # REF-anchor local frame (see stage1 runner's reset-time normalization).
        off = ENCODER_OFFSETS["vr_3point_local_target"]
        buf[:, off : off + 9] = (
            vr_3pt_position.to(self.device).detach().cpu().numpy().astype(np.float32)
        )

        # vr_3point_local_orn_target  (12)
        off = ENCODER_OFFSETS["vr_3point_local_orn_target"]
        buf[:, off : off + 12] = (
            vr_3pt_orientation.to(self.device).detach().cpu().numpy().astype(np.float32)
        )
        return buf

    def _build_decoder_obs(self, token: np.ndarray) -> np.ndarray:
        """Fill the [N, 994] obs_dict for the deploy decoder.

        Field order (deploy): [token, ang_vel, joint_pos_rel, joint_vel, last_action, gravity].
        All proprio fields in IL order; joint_pos is relative to `default_angles`.
        """
        buf = self._dec_buf
        off = DECODER_OFFSETS

        buf[:, off["token_state"] : off["token_state"] + ENCODER_TOKEN_DIM] = token
        buf[
            :,
            off["his_base_angular_velocity_10frame_step1"] : off[
                "his_base_angular_velocity_10frame_step1"
            ]
            + 30,
        ] = self.his_ang_vel.reshape(self.N, -1).detach().cpu().numpy()
        buf[
            :,
            off["his_body_joint_positions_10frame_step1"] : off[
                "his_body_joint_positions_10frame_step1"
            ]
            + 290,
        ] = self.his_joint_pos.reshape(self.N, -1).detach().cpu().numpy()
        buf[
            :,
            off["his_body_joint_velocities_10frame_step1"] : off[
                "his_body_joint_velocities_10frame_step1"
            ]
            + 290,
        ] = self.his_joint_vel.reshape(self.N, -1).detach().cpu().numpy()
        buf[
            :,
            off["his_last_actions_10frame_step1"] : off["his_last_actions_10frame_step1"]
            + 290,
        ] = self.his_last_action.reshape(self.N, -1).detach().cpu().numpy()
        buf[
            :,
            off["his_gravity_dir_10frame_step1"] : off["his_gravity_dir_10frame_step1"]
            + 30,
        ] = self.his_gravity.reshape(self.N, -1).detach().cpu().numpy()
        return buf

    # ------------------------------------------------------------
    # Main 50 Hz tick
    # ------------------------------------------------------------
    def step(
        self,
        vr_3pt_position: torch.Tensor,  # [N, 9]  root-local (L-wrist, R-wrist, head)
        vr_3pt_orientation: torch.Tensor,  # [N, 12]
        mode: torch.Tensor,  # [N] int
        movement_direction: torch.Tensor,  # [N, 3]
        facing_direction: torch.Tensor,  # [N, 3]
        target_vel: torch.Tensor,  # [N]
        height: torch.Tensor,  # [N]
        joint_pos: torch.Tensor,  # [N, 29] IL order
        joint_vel: torch.Tensor,  # [N, 29] IL order
        base_ang_vel: torch.Tensor,  # [N, 3]  base frame
        gravity_in_base: torch.Tensor,  # [N, 3]
        root_pos: torch.Tensor,  # [N, 3]  world
        root_quat_wxyz: torch.Tensor,  # [N, 4]  world
    ) -> torch.Tensor:
        """Return joint position targets [N, 29] in IsaacLab order."""
        del root_pos  # planner context is self-consistent via cache feedback
        joint_pos_il = joint_pos.to(self.device)
        joint_vel_il = joint_vel.to(self.device)

        # 1. Planner @ 10 Hz (every 5th tick)
        if self.step_counter % PLANNER_EVERY_K_POLICY_STEPS == 0:
            if self.step_counter > 0:
                self.planner_context = self._context_from_cache()
            self._run_planner(
                mode, movement_direction, facing_direction, target_vel, height
            )

        # 2. Update STATE histories with current values BEFORE running policy
        # (training: Isaac Lab ObservationManager appends current-state obs to
        # circular buffer during obs gather, so at tick t the state history
        # INCLUDES state_t. The action history, however, holds action_{t-1}
        # since env.action_manager.action has not yet been updated with the
        # new action — we update it after decoder below.)
        self.his_ang_vel = torch.roll(self.his_ang_vel, -1, dims=1)
        self.his_ang_vel[:, -1] = base_ang_vel.to(self.device)
        self.his_joint_pos = torch.roll(self.his_joint_pos, -1, dims=1)
        self.his_joint_pos[:, -1] = joint_pos_il - self.default_angles.unsqueeze(0)
        self.his_joint_vel = torch.roll(self.his_joint_vel, -1, dims=1)
        self.his_joint_vel[:, -1] = joint_vel_il
        self.his_gravity = torch.roll(self.his_gravity, -1, dims=1)
        self.his_gravity[:, -1] = gravity_in_base.to(self.device)

        # 3. Gather planner-derived encoder obs
        lower_pos, lower_vel, anchor_6d = self._sample_planner_obs(root_quat_wxyz)

        _dbg = self.step_counter < _SONIC_DEBUG_TICKS
        if _dbg:
            pb = int(self.planner_playback_idx[0].item())
            pf0 = self.planner_cache[0, pb].cpu().numpy()
            rq = root_quat_wxyz[0].cpu().numpy()
            vrp = vr_3pt_position[0].cpu().numpy() if torch.is_tensor(vr_3pt_position) else vr_3pt_position[0]
            vro = vr_3pt_orientation[0].cpu().numpy() if torch.is_tensor(vr_3pt_orientation) else vr_3pt_orientation[0]
            print(
                f"[SONIC_DEBUG t={self.step_counter}] vr_pos_local[L]={vrp[0:3].round(3).tolist()} "
                f"[R]={vrp[3:6].round(3).tolist()} [H]={vrp[6:9].round(3).tolist()} "
                f"mode={int(mode[0].item())}"
            )
            print(
                f"[SONIC_DEBUG t={self.step_counter}] pb_idx={pb}  "
                f"planner_frame0 xyz={pf0[:3].round(4).tolist()} "
                f"quat={pf0[3:7].round(4).tolist()} "
                f"joints_mj[0:6]={pf0[7:13].round(3).tolist()}"
            )
            print(
                f"[SONIC_DEBUG t={self.step_counter}] robot_quat={rq.round(4).tolist()}  "
                f"anchor_6d={anchor_6d[0].cpu().numpy().round(4).tolist()}  "
                f"lower_pos[0:6]={lower_pos[0, :6].cpu().numpy().round(3).tolist()}  "
                f"lower_vel[0:6]={lower_vel[0, :6].cpu().numpy().round(3).tolist()}"
            )
            print(
                f"[SONIC_DEBUG t={self.step_counter}] "
                f"joint_pos_rel[0:6]={self.his_joint_pos[0, -1, :6].cpu().numpy().round(3).tolist()}  "
                f"joint_vel[0:6]={self.his_joint_vel[0, -1, :6].cpu().numpy().round(3).tolist()}  "
                f"ang_vel={self.his_ang_vel[0, -1].cpu().numpy().round(3).tolist()}  "
                f"gravity={self.his_gravity[0, -1].cpu().numpy().round(3).tolist()}"
            )

        # 4. Encoder ONNX (deploy, batch=1 per session, N sessions)
        enc_obs = self._build_encoder_obs(
            lower_pos, lower_vel, anchor_6d, vr_3pt_position, vr_3pt_orientation
        )
        token = self.encoder_pool.run_batched(enc_obs)  # [N, 64]

        # 5. Decoder ONNX
        dec_obs = self._build_decoder_obs(token)
        action_np = self.decoder_pool.run_batched(dec_obs)  # [N, 29]
        action = torch.as_tensor(action_np, device=self.device, dtype=torch.float32)

        if _dbg:
            a0 = action[0].cpu().numpy()
            target = (self.default_angles + self.action_scale * action[0]).cpu().numpy()
            print(
                f"[SONIC_DEBUG t={self.step_counter}] "
                f"token[0:6]={token[0, :6].round(3).tolist()}  "
                f"action[0:6]={a0[:6].round(3).tolist()}  "
                f"action_norm={np.linalg.norm(a0):.3f}  "
                f"action_max={np.abs(a0).max():.3f}"
            )
            print(
                f"[SONIC_DEBUG t={self.step_counter}] "
                f"target[0:6]={target[:6].round(3).tolist()}  "
                f"default[0:6]={self.default_angles[:6].cpu().numpy().round(3).tolist()}  "
                f"scale[0:6]={self.action_scale[:6].cpu().numpy().round(3).tolist()}"
            )

        # 6. Update ACTION history AFTER decoder (so at tick t+1 the history's
        # last entry is action_t, matching training's env.action_manager.action).
        self.his_last_action = torch.roll(self.his_last_action, -1, dims=1)
        self.his_last_action[:, -1] = action

        # 7. Advance planner playback cursor
        self.planner_playback_idx = (self.planner_playback_idx + 1).clamp(
            max=RESAMPLED_FRAMES - 1
        )
        self.step_counter += 1

        # 8. target_il = default_il + scale_il * action_il  (all IL-ordered)
        return self.default_angles.unsqueeze(0) + self.action_scale.unsqueeze(0) * action
