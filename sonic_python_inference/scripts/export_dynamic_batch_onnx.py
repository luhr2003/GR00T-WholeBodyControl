"""Re-export SONIC teleop encoder + g1_dyn decoder from sonic_release/last.pt with dynamic batch.

Usage:
    uv run --active python -m sonic_python_inference.scripts.export_dynamic_batch_onnx \
        --checkpoint /path/to/sonic_release/last.pt \
        --out-dir    /path/to/sonic_python_inference/assets

Rationale: the shipped ONNX in gear_sonic_deploy/policy is batch=1 hardcoded (internal
Reshape nodes bake in [1, T, D] shapes that cannot be patched post-hoc). This script
rebuilds the encoder MLP + FSQ + decoder MLP directly from the state_dict weights and
exports each with `dynamic_axes={...: {0: "batch"}}`, so one ONNX model serves any N.

We intentionally avoid constructing the full gear_sonic Actor via custom_instantiate
because that requires an Isaac Lab env_config with populated obs dims (only exists
after env boot). The weight shapes alone fully specify the MLP + FSQ pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn


TELEOP_HIDDEN = [2048, 1024, 512, 512]
TELEOP_INPUT_DIM = 267
TELEOP_TOKEN_NUM = 2
TELEOP_TOKEN_DIM = 32
TELEOP_FSQ_LEVELS = [32] * 32

# SMPL encoder: 10 future frames × (24 joints×3 local + 6D root rel + 6 wrist dof) = 840
SMPL_HIDDEN = [2048, 1024, 512, 512]
SMPL_INPUT_DIM = 840
SMPL_TOKEN_NUM = 2
SMPL_TOKEN_DIM = 32
SMPL_FSQ_LEVELS = [32] * 32

DECODER_HIDDEN = [2048, 2048, 1024, 1024, 512, 512]
DECODER_INPUT_DIM = 994
DECODER_ACTION_DIM = 29
DECODER_TOKEN_FLAT = 64


def build_mlp(input_dim: int, hidden_dims: list[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dims[0]), nn.SiLU()]
    for i in range(len(hidden_dims)):
        if i == len(hidden_dims) - 1:
            layers.append(nn.Linear(hidden_dims[i], output_dim))
        else:
            layers.extend([nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.SiLU()])
    return nn.Sequential(*layers)


def load_state_dict_with_shim(ckpt_path: Path) -> dict:
    import trl.trainer.utils as tu

    if not hasattr(tu, "OnlineTrainerState"):
        tu.OnlineTrainerState = type(
            "OnlineTrainerState",
            (),
            {"__reduce__": lambda self: (dict, ())},
        )
    full = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = full["policy_state_dict"]
    out = {}
    for k, v in sd.items():
        if k.startswith("actor_module.encoders.teleop.module."):
            out[f"teleop.{k.removeprefix('actor_module.encoders.teleop.module.')}"] = v
        elif k.startswith("actor_module.encoders.smpl.module."):
            out[f"smpl.{k.removeprefix('actor_module.encoders.smpl.module.')}"] = v
        elif k.startswith("actor_module.decoders.g1_dyn.module."):
            out[f"g1_dyn.{k.removeprefix('actor_module.decoders.g1_dyn.module.')}"] = v
    return out


class TeleopEncoderWithFSQ(nn.Module):
    """Teleop obs [N, 267] → MLP → [N, 64] → reshape → FSQ → [N, 64] token_flattened."""

    def __init__(self):
        super().__init__()
        self.mlp = build_mlp(TELEOP_INPUT_DIM, TELEOP_HIDDEN, TELEOP_TOKEN_NUM * TELEOP_TOKEN_DIM)
        from vector_quantize_pytorch import FSQ

        self.fsq = FSQ(levels=TELEOP_FSQ_LEVELS)

    def load_from(self, sd: dict):
        teleop_sd = {k.removeprefix("teleop."): v for k, v in sd.items() if k.startswith("teleop.")}
        # mlp Sequential layer indices match exactly: 0,2,4,6,8
        self.mlp.load_state_dict(teleop_sd, strict=True)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        latent = self.mlp(obs)  # [N, 64]
        latent = latent.view(-1, TELEOP_TOKEN_NUM, TELEOP_TOKEN_DIM)  # [N, 2, 32]
        quantized, _ = self.fsq(latent)  # [N, 2, 32]
        return quantized.view(-1, TELEOP_TOKEN_NUM * TELEOP_TOKEN_DIM)  # [N, 64]


class SmplEncoderWithFSQ(nn.Module):
    """SMPL obs [N, 840] → MLP → [N, 64] → reshape (N,2,32) → FSQ → flatten → [N, 64].

    The 840-D obs packs 10 future frames (dt=0.02 s) of:
      * smpl_joints_multi_future_local_nonflat: 24 SMPL joints × 3, per-frame canonicalised
        by each frame's own SMPL root quaternion (72 per frame).
      * smpl_root_ori_b_multi_future: 6D rotation of SMPL root relative to robot anchor
        (quat_inv(robot_anchor) * smpl_root → first 2 columns of rotation matrix, 6 per frame).
      * joint_pos_multi_future_wrist_for_smpl: robot dof future ref at IL indices 23-28
        (6 wrist dofs per frame).

    Concatenation order inside a frame must be [joints(72), root6d(6), wrist(6)] then
    flatten over 10 frames to 840. See gear_sonic training code (config.yaml encoders.smpl.inputs).
    """

    def __init__(self):
        super().__init__()
        self.mlp = build_mlp(SMPL_INPUT_DIM, SMPL_HIDDEN, SMPL_TOKEN_NUM * SMPL_TOKEN_DIM)
        from vector_quantize_pytorch import FSQ

        self.fsq = FSQ(levels=SMPL_FSQ_LEVELS)

    def load_from(self, sd: dict):
        smpl_sd = {k.removeprefix("smpl."): v for k, v in sd.items() if k.startswith("smpl.")}
        self.mlp.load_state_dict(smpl_sd, strict=True)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        latent = self.mlp(obs)
        latent = latent.view(-1, SMPL_TOKEN_NUM, SMPL_TOKEN_DIM)
        quantized, _ = self.fsq(latent)
        return quantized.view(-1, SMPL_TOKEN_NUM * SMPL_TOKEN_DIM)


class G1DynDecoder(nn.Module):
    """[N, 994] (token_flat 64 ‖ proprio 930) → MLP → [N, 29] action."""

    def __init__(self):
        super().__init__()
        self.mlp = build_mlp(DECODER_INPUT_DIM, DECODER_HIDDEN, DECODER_ACTION_DIM)

    def load_from(self, sd: dict):
        dec_sd = {k.removeprefix("g1_dyn."): v for k, v in sd.items() if k.startswith("g1_dyn.")}
        self.mlp.load_state_dict(dec_sd, strict=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def export_encoder(model: TeleopEncoderWithFSQ, out_path: Path):
    model.eval()
    dummy = torch.randn(1, TELEOP_INPUT_DIM)
    torch.onnx.export(
        model,
        (dummy,),
        str(out_path),
        input_names=["teleop_obs"],
        output_names=["token_flattened"],
        dynamic_axes={"teleop_obs": {0: "batch"}, "token_flattened": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )


def _export_smpl_encoder(model: SmplEncoderWithFSQ, out_path: Path):
    model.eval()
    dummy = torch.randn(1, SMPL_INPUT_DIM)
    torch.onnx.export(
        model,
        (dummy,),
        str(out_path),
        input_names=["smpl_obs"],
        output_names=["token_flattened"],
        dynamic_axes={"smpl_obs": {0: "batch"}, "token_flattened": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )


def export_decoder(model: G1DynDecoder, out_path: Path):
    model.eval()
    dummy = torch.randn(1, DECODER_INPUT_DIM)
    torch.onnx.export(
        model,
        (dummy,),
        str(out_path),
        input_names=["decoder_input"],
        output_names=["action"],
        dynamic_axes={"decoder_input": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )


def verify_dynamic(path: Path, input_name: str, input_dim: int, out_dim: int):
    import numpy as np
    import onnx
    import onnxruntime as ort

    m = onnx.load(str(path))
    in_shape = m.graph.input[0].type.tensor_type.shape.dim
    out_shape = m.graph.output[0].type.tensor_type.shape.dim
    assert in_shape[0].dim_param == "batch", f"{path}: input[0].dim 0 not dynamic"
    assert out_shape[0].dim_param == "batch", f"{path}: output[0].dim 0 not dynamic"
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for n in (1, 4, 64):
        x = np.random.randn(n, input_dim).astype("float32")
        y = sess.run(None, {input_name: x})[0]
        assert y.shape == (n, out_dim), f"{path}: N={n} got {y.shape}"
    print(f"  OK  {path.name}: dynamic batch validated for N=1,4,64")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sd = load_state_dict_with_shim(args.checkpoint)
    print(f"loaded {len(sd)} teleop+g1_dyn tensors from {args.checkpoint}")

    enc = TeleopEncoderWithFSQ()
    enc.load_from(sd)
    smpl_enc = SmplEncoderWithFSQ()
    smpl_enc.load_from(sd)
    dec = G1DynDecoder()
    dec.load_from(sd)

    enc_path = args.out_dir / "encoder_dyn.onnx"
    smpl_enc_path = args.out_dir / "smpl_encoder_dyn.onnx"
    dec_path = args.out_dir / "decoder_dyn.onnx"
    export_encoder(enc, enc_path)
    _export_smpl_encoder(smpl_enc, smpl_enc_path)
    export_decoder(dec, dec_path)

    verify_dynamic(enc_path, "teleop_obs", TELEOP_INPUT_DIM, DECODER_TOKEN_FLAT)
    verify_dynamic(smpl_enc_path, "smpl_obs", SMPL_INPUT_DIM, DECODER_TOKEN_FLAT)
    verify_dynamic(dec_path, "decoder_input", DECODER_INPUT_DIM, DECODER_ACTION_DIM)


if __name__ == "__main__":
    main()
