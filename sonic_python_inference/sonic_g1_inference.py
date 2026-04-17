"""Python inference pipeline for SONIC G1-mode (teacher) tracking, N envs in parallel.

The G1 encoder is the **teacher** encoder in training — SMPL/teleop distill to
it. It consumes the retargeted robot trajectory directly, so this inference
path only needs `robot_filtered/*.pkl` (no SMPL pkl required). Architecturally
it differs from `SonicSMPLInference` only in `_build_encoder_obs` — the
proprio layout, history buffers, decoder call, and action scaling are all
identical (decoder_dyn.onnx is shared).

Encoder obs layout — EXACTLY matches training code memory layout, which is
semantically weird but deterministic:

    command_multi_future (commands.py:897-903) =
        cat([joint_pos_multi_future, joint_vel_multi_future], dim=1)
    where each child is already flat of shape [N, F*29] with frame-major layout
    [pos_f0(29), pos_f1(29), ..., pos_fF-1(29)]. So cmd_flat per env =
        [pos_f0, pos_f1, ..., pos_fF-1, vel_f0, vel_f1, ..., vel_fF-1]  (2*F*29 elems)

    command_multi_future_nonflat (observations.py:584-587) reshapes the flat
    tensor to [N, F, 2*29=58], which crosses the pos/vel boundary and mixes
    frames — e.g. slot-0 of the reshape holds [pos_f0(29), pos_f1(29)], and
    slot-5 of the reshape holds [vel_f0(29), vel_f1(29)]. This is NOT a
    logical per-frame packing; it's an artefact of the cat(dim=1)+reshape
    order. But the trained MLP has learned this exact mangled input, so we
    must reproduce it bit-for-bit.

    motion_anchor_ori_b_mf_nonflat (observations.py:1022-1043, commands.py:1942-1963)
        flat layout [f0_6d, f1_6d, ..., fF-1_6d] → reshape [N, F, 6]
        6D( quat_mul(quat_inv(robot_anchor_quat_w), ref_root_quat_future) )
        ref_root_quat is the ROBOT RETARGET root (from robot pkl `root_rot`,
        xyzw→wxyz). No Y→Z, no base-rot removal.

    cat([cmd_nonflat(58), anchor_ori_nonflat(6)], dim=-1) → [N, F, 64]
    reshape(N, -1) → [N, 640] → g1_encoder_dyn.onnx → [N, 64] token

Quaternion convention everywhere: wxyz.
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
    quat_to_6d,
)


G1_NUM_FUTURE_FRAMES = 10  # config.yaml: num_future_frames
G1_ENCODER_INPUT_DIM = G1_NUM_FUTURE_FRAMES * (NUM_JOINTS + NUM_JOINTS + 6)  # 10 * 64 = 640
DECODER_INPUT_DIM = 994  # 64 token + 930 proprio (shared with SMPL/VR3PT paths)


def _root_ori_b_6d_per_frame(
    robot_anchor_quat_wxyz: torch.Tensor,  # [N, 4] current robot pelvis quat
    ref_root_quat_future_wxyz: torch.Tensor,  # [N, F, 4] retarget root quat (from robot pkl)
) -> torch.Tensor:
    """6D rotation of retarget root relative to robot anchor, per future frame.

    Mirrors commands.py:1942-1963 (root_rot_dif_l_multi_future):
        root_rot_dif = quat_mul(quat_inv(robot_anchor_quat).expand(F),
                                ref_root_quat_future)
        6d = matrix_from_quat(...)[..., :2].reshape(-1, 6)

    Returns [N, F, 6].
    """
    N, F, _ = ref_root_quat_future_wxyz.shape
    anchor_inv = quat_conjugate(robot_anchor_quat_wxyz).unsqueeze(1).expand(N, F, 4)
    rel = quat_mul(anchor_inv, ref_root_quat_future_wxyz)  # [N, F, 4]
    return quat_to_6d(rel)  # [N, F, 6]


class SonicG1Inference:
    """Batched SONIC G1-mode (teacher) inference for N Isaac Lab envs.

    Usage:
        infer = SonicG1Inference(
            num_envs=4,
            g1_encoder_onnx="sonic_python_inference/assets/g1_encoder_dyn.onnx",
            decoder_onnx="sonic_python_inference/assets/decoder_dyn.onnx",
            default_angles=default_il,
            action_scale=scale_il,
            device="cuda",
        )
        infer.reset(joint_pos=q_il)
        for policy_tick in range(T):
            targets = infer.step(
                joint_pos_future=...,            # [N, 10, 29] IL-order ref dof
                joint_vel_future=...,            # [N, 10, 29] IL-order ref dof_vel
                ref_root_quat_future_wxyz=...,   # [N, 10, 4]  robot pkl root_rot (wxyz)
                joint_pos=..., joint_vel=...,
                base_ang_vel=..., gravity_in_base=...,
                root_quat_wxyz=...,              # [N, 4] robot pelvis quat
            )
    """

    def __init__(
        self,
        num_envs: int,
        g1_encoder_onnx: str | Path,
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
        self.enc_sess = ort.InferenceSession(str(g1_encoder_onnx), providers=providers)
        self.dec_sess = ort.InferenceSession(str(decoder_onnx), providers=providers)

        # Proprioception ring buffers — IDENTICAL layout to SMPL/VR3PT path.
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
        joint_pos_future: torch.Tensor,  # [N, 10, 29]
        joint_vel_future: torch.Tensor,  # [N, 10, 29]
        ref_root_quat_future_wxyz: torch.Tensor,  # [N, 10, 4]
        robot_anchor_quat_wxyz: torch.Tensor,  # [N, 4]
    ) -> np.ndarray:
        """Pack per training-code layout (see module docstring). Result: [N, 640]."""
        assert joint_pos_future.shape == (self.N, G1_NUM_FUTURE_FRAMES, NUM_JOINTS)
        assert joint_vel_future.shape == (self.N, G1_NUM_FUTURE_FRAMES, NUM_JOINTS)
        assert ref_root_quat_future_wxyz.shape == (self.N, G1_NUM_FUTURE_FRAMES, 4)
        assert robot_anchor_quat_wxyz.shape == (self.N, 4)

        jp = joint_pos_future.to(self.device)  # [N, F, 29]
        jv = joint_vel_future.to(self.device)  # [N, F, 29]
        root_ori_6d = _root_ori_b_6d_per_frame(
            robot_anchor_quat_wxyz.to(self.device),
            ref_root_quat_future_wxyz.to(self.device),
        )  # [N, F, 6]

        # Reproduce commands.py:903 exactly: cat of already-flat tensors.
        # Each .reshape(N, -1) produces frame-major flat [f0(29), f1(29), ...].
        jp_flat = jp.reshape(self.N, -1)  # [N, F*29]
        jv_flat = jv.reshape(self.N, -1)  # [N, F*29]
        cmd_flat = torch.cat([jp_flat, jv_flat], dim=1)  # [N, 2*F*29]

        # Reproduce observations.py:584-587 non_flatten=True reshape. This is
        # the mangled per-"slot" reshape (slot i of 58 bytes is NOT [pos_fi, vel_fi]).
        cmd_nonflat = cmd_flat.reshape(
            self.N, G1_NUM_FUTURE_FRAMES, -1
        )  # [N, F, 58]

        # Per config.yaml encoders.g1.inputs: command_multi_future_nonflat first,
        # motion_anchor_ori_b_mf_nonflat second — concatenated on last dim.
        slot = torch.cat([cmd_nonflat, root_ori_6d], dim=-1)  # [N, F, 64]
        flat = slot.reshape(self.N, -1)  # [N, 640]
        assert flat.shape[-1] == G1_ENCODER_INPUT_DIM
        return flat.cpu().numpy().astype(np.float32)

    def step(
        self,
        joint_pos_future: torch.Tensor,  # [N, 10, 29] retarget ref dof
        joint_vel_future: torch.Tensor,  # [N, 10, 29] retarget ref dof_vel
        ref_root_quat_future_wxyz: torch.Tensor,  # [N, 10, 4] retarget root quat (wxyz)
        joint_pos: torch.Tensor,  # [N, 29] IL-order measured
        joint_vel: torch.Tensor,  # [N, 29] IL-order measured
        base_ang_vel: torch.Tensor,  # [N, 3] base frame
        gravity_in_base: torch.Tensor,  # [N, 3]
        root_quat_wxyz: torch.Tensor,  # [N, 4] robot pelvis quat (world)
    ) -> torch.Tensor:
        """One 50 Hz policy tick. Returns motor targets [N, 29] in IsaacLab order."""
        joint_pos_il = joint_pos.to(self.device)
        joint_vel_il = joint_vel.to(self.device)

        # Push current obs into histories before building proprio. `last_action`
        # is pushed AFTER the policy runs so next tick sees it as "previous".
        self.his_ang_vel = torch.roll(self.his_ang_vel, -1, dims=1)
        self.his_ang_vel[:, -1] = base_ang_vel.to(self.device)
        self.his_joint_pos = torch.roll(self.his_joint_pos, -1, dims=1)
        self.his_joint_pos[:, -1] = joint_pos_il - self.default_angles.unsqueeze(0)
        self.his_joint_vel = torch.roll(self.his_joint_vel, -1, dims=1)
        self.his_joint_vel[:, -1] = joint_vel_il
        self.his_gravity = torch.roll(self.his_gravity, -1, dims=1)
        self.his_gravity[:, -1] = gravity_in_base.to(self.device)

        g1_obs = self._build_encoder_obs(
            joint_pos_future,
            joint_vel_future,
            ref_root_quat_future_wxyz,
            root_quat_wxyz,
        )
        token_flat = self.enc_sess.run(None, {"g1_obs": g1_obs})[0]  # [N, 64]

        proprio = torch.cat(
            [
                self.his_ang_vel.reshape(self.N, -1),  # 30
                self.his_joint_pos.reshape(self.N, -1),  # 290
                self.his_joint_vel.reshape(self.N, -1),  # 290
                self.his_last_action.reshape(self.N, -1),  # 290
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

        self.his_last_action = torch.roll(self.his_last_action, -1, dims=1)
        self.his_last_action[:, -1] = action

        self.step_counter += 1

        return self.default_angles.unsqueeze(0) + self.action_scale.unsqueeze(0) * action
