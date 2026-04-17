"""Stage 1.5 — isolated subtests for the SONIC encoder/decoder pipeline.

Pure Python: no Isaac Lab, no planner ONNX. Runs in seconds. Each subtest is
self-contained and prints PASS/FAIL. If everything passes here, the bug in
stage1 is NOT in our math utilities / history buffer / obs concat — it's in
how we FEED them from Isaac Lab (e.g. root_quat convention, VR 3pt frame,
joint ordering between sim and API).

Covers (from the verification plan):
    A. ONNX I/O schema dump — confirm encoder expects `teleop_obs[N,267]` and
       decoder expects `decoder_input[N,994]` in the order we feed.
    C. quat_to_6d, slerp, motion_anchor_orientation formula on known rotations.
    D. History ring buffer roll direction.
    E. Finite-diff velocity on a constant-joint trajectory.
    F. Encoder token / decoder action sanity at a standing pose.

Run:
    uv run --active python -m sonic_python_inference.scripts.stage1_5_subtests
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from sonic_python_inference.sonic_inference import (
    HIST_LEN,
    NUM_JOINTS,
    POLICY_HZ,
    PROPRIO_DIMS,
    PROPRIO_KEYS_ORDER,
    PROPRIO_TOTAL,
    RESAMPLED_FRAMES,
    quat_conjugate,
    quat_mul,
    quat_to_6d,
    resample_traj_30_to_50hz,
    slerp_torch,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENCODER_ONNX = PROJECT_ROOT / "sonic_python_inference/assets/encoder_dyn.onnx"
DECODER_ONNX = PROJECT_ROOT / "sonic_python_inference/assets/decoder_dyn.onnx"

TELEOP_INPUT_DIM = 267
DECODER_INPUT_DIM = 994
TOKEN_DIM = 64
ACTION_DIM = 29

ATOL = 1e-5


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


class Reporter:
    def __init__(self) -> None:
        self.n_pass = 0
        self.n_fail = 0
        self.failures: list[str] = []

    def ok(self, name: str, extra: str = "") -> None:
        self.n_pass += 1
        tail = f"  ({extra})" if extra else ""
        print(f"  [PASS] {name}{tail}")

    def bad(self, name: str, detail: str) -> None:
        self.n_fail += 1
        self.failures.append(f"{name}: {detail}")
        print(f"  [FAIL] {name}  — {detail}")

    def check_allclose(
        self, name: str, got: torch.Tensor | np.ndarray, want: torch.Tensor | np.ndarray, atol: float = ATOL
    ) -> None:
        g = got.detach().cpu().numpy() if isinstance(got, torch.Tensor) else np.asarray(got)
        w = want.detach().cpu().numpy() if isinstance(want, torch.Tensor) else np.asarray(want)
        if g.shape != w.shape:
            return self.bad(name, f"shape mismatch: got {g.shape}, want {w.shape}")
        diff = float(np.max(np.abs(g - w)))
        if diff > atol:
            return self.bad(name, f"max|Δ|={diff:.2e} > {atol:.1e}\n      got  {g.ravel()}\n      want {w.ravel()}")
        self.ok(name, f"max|Δ|={diff:.2e}")

    def finish(self) -> int:
        print()
        print("-" * 60)
        print(f"  total: {self.n_pass} passed, {self.n_fail} failed")
        if self.n_fail:
            print("  failed subtests:")
            for f in self.failures:
                print(f"    - {f}")
        return 1 if self.n_fail else 0


def section(title: str) -> None:
    print()
    print(f"=== {title} ===")


# ---------------------------------------------------------------------------
# A. ONNX schema dump
# ---------------------------------------------------------------------------


def test_onnx_schema(r: Reporter) -> None:
    section("A. ONNX I/O schema")

    for label, path, in_name, in_dim, out_name, out_dim in [
        ("encoder", ENCODER_ONNX, "teleop_obs", TELEOP_INPUT_DIM, "token_flattened", TOKEN_DIM),
        ("decoder", DECODER_ONNX, "decoder_input", DECODER_INPUT_DIM, "action", ACTION_DIM),
    ]:
        if not path.exists():
            r.bad(f"{label}: file exists", f"missing {path}")
            continue
        m = onnx.load(str(path))
        inputs = [(i.name, [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input]
        outputs = [(o.name, [d.dim_param or d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output]
        print(f"  {label}: {path.name}")
        print(f"    inputs  = {inputs}")
        print(f"    outputs = {outputs}")

        if len(inputs) != 1 or inputs[0][0] != in_name:
            r.bad(f"{label}: single input named {in_name!r}", f"got {[i[0] for i in inputs]}")
            continue
        if inputs[0][1] != ["batch", in_dim]:
            r.bad(f"{label}: input shape", f"want ['batch', {in_dim}], got {inputs[0][1]}")
            continue
        if len(outputs) != 1 or outputs[0][0] != out_name:
            r.bad(f"{label}: single output named {out_name!r}", f"got {[o[0] for o in outputs]}")
            continue
        # Output dim 0 must be "batch". Dim 1 is usually the literal int, but the
        # encoder's final `view(-1, 64)` after FSQ makes torch.onnx emit a
        # symbolic name (e.g. "Reshapetoken_flattened_dim_1"). Runtime shape is
        # still (N, out_dim) — F1 smoke test catches a real regression.
        out_shape_ok = outputs[0][1][0] == "batch" and (
            outputs[0][1][1] == out_dim or isinstance(outputs[0][1][1], str)
        )
        if not out_shape_ok:
            r.bad(f"{label}: output shape", f"want ['batch', {out_dim}|symbolic], got {outputs[0][1]}")
            continue
        if outputs[0][1][1] != out_dim:
            r.ok(f"{label}: schema matches wiring", f"dim1 symbolic={outputs[0][1][1]!r} — runtime shape checked in F1")
        else:
            r.ok(f"{label}: schema matches wiring")


# ---------------------------------------------------------------------------
# C. Math utilities — quat_to_6d, slerp, motion_anchor_orientation
# ---------------------------------------------------------------------------


def _yaw_quat(angle: float) -> torch.Tensor:
    c, s = math.cos(angle / 2), math.sin(angle / 2)
    return torch.tensor([c, 0.0, 0.0, s], dtype=torch.float32)


def _rot_6d_row_major(R: np.ndarray) -> np.ndarray:
    """First 2 cols of R, flattened ROW-WISE (matches quat_to_6d spec)."""
    return R[..., :2].reshape(*R.shape[:-2], -1)


def test_quat_to_6d(r: Reporter) -> None:
    section("C1. quat_to_6d")

    # identity → R = I → first 2 cols row-flat = [1,0,0,1,0,0]
    q_id = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    r.check_allclose("identity", quat_to_6d(q_id)[0], torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]))

    # yaw(+π/2): Rz = [[0,-1,0],[1,0,0],[0,0,1]] → first 2 cols row-flat = [0,-1,1,0,0,0]
    q_yaw = _yaw_quat(math.pi / 2).unsqueeze(0)
    r.check_allclose("yaw(+π/2)", quat_to_6d(q_yaw)[0], torch.tensor([0.0, -1.0, 1.0, 0.0, 0.0, 0.0]))

    # yaw(-π/2): Rz = [[0,1,0],[-1,0,0],[0,0,1]] → [0,1,-1,0,0,0]
    q_yaw_n = _yaw_quat(-math.pi / 2).unsqueeze(0)
    r.check_allclose("yaw(-π/2)", quat_to_6d(q_yaw_n)[0], torch.tensor([0.0, 1.0, -1.0, 0.0, 0.0, 0.0]))

    # Cross-check against scipy.Rotation on a random batch
    from scipy.spatial.transform import Rotation as R_scipy

    rng = np.random.default_rng(0)
    q_rand_xyzw = R_scipy.random(8, random_state=rng).as_quat()  # xyzw
    q_rand_wxyz = q_rand_xyzw[:, [3, 0, 1, 2]]
    R_mat = R_scipy.from_quat(q_rand_xyzw).as_matrix()  # [B, 3, 3]
    want = _rot_6d_row_major(R_mat)
    got = quat_to_6d(torch.as_tensor(q_rand_wxyz, dtype=torch.float32)).numpy()
    r.check_allclose("vs scipy random batch (8)", got, want, atol=1e-5)


def test_slerp(r: Reporter) -> None:
    section("C2. slerp_torch")

    q0 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    q1 = _yaw_quat(math.pi).unsqueeze(0)  # [0, 0, 0, 1]

    # t=0 → q0
    r.check_allclose("t=0", slerp_torch(q0, q1, torch.tensor([0.0]))[0], q0[0], atol=1e-5)
    # t=1 → q1
    r.check_allclose("t=1", slerp_torch(q0, q1, torch.tensor([1.0]))[0], q1[0], atol=1e-5)
    # t=0.5 → yaw(π/2)
    r.check_allclose(
        "t=0.5 (identity→yaw π)", slerp_torch(q0, q1, torch.tensor([0.5]))[0], _yaw_quat(math.pi / 2), atol=1e-5
    )


def test_anchor_formula(r: Reporter) -> None:
    section("C3. motion_anchor_orientation = quat_to_6d(conj(base) * ref)")

    # Case 1: base = ref = identity → anchor = identity's 6D = [1,0,0,1,0,0]
    base = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    ref = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    rel = quat_mul(quat_conjugate(base), ref)
    r.check_allclose("base=ref=identity", quat_to_6d(rel)[0], torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]))

    # Case 2: base = identity, ref = yaw(+π/2) → anchor = yaw(+π/2)'s 6D
    base = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    ref = _yaw_quat(math.pi / 2).unsqueeze(0)
    rel = quat_mul(quat_conjugate(base), ref)
    r.check_allclose("base=I, ref=yaw(π/2)", quat_to_6d(rel)[0], torch.tensor([0.0, -1.0, 1.0, 0.0, 0.0, 0.0]))

    # Case 3: base = ref = yaw(+π/2) → rel should be identity, not yaw(π/2)!
    # This is the critical semantic: anchor is RELATIVE, so same yaw → identity.
    base = _yaw_quat(math.pi / 2).unsqueeze(0)
    ref = _yaw_quat(math.pi / 2).unsqueeze(0)
    rel = quat_mul(quat_conjugate(base), ref)
    r.check_allclose(
        "base=ref=yaw(π/2) → identity", quat_to_6d(rel)[0], torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    )

    # Case 4: base = yaw(+π/2), ref = identity (robot turned CCW while planner still faces +X)
    # rel = yaw(-π/2) → [0,1,-1,0,0,0]
    base = _yaw_quat(math.pi / 2).unsqueeze(0)
    ref = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    rel = quat_mul(quat_conjugate(base), ref)
    r.check_allclose("base=yaw(π/2), ref=I", quat_to_6d(rel)[0], torch.tensor([0.0, 1.0, -1.0, 0.0, 0.0, 0.0]))


# ---------------------------------------------------------------------------
# E. Trajectory resampling (30 → 50 Hz) sanity
# ---------------------------------------------------------------------------


def test_resample(r: Reporter) -> None:
    section("E. resample_traj_30_to_50hz")

    # Construct a constant-velocity linear trajectory in 30Hz grid, then resample
    # and check values at specific 50Hz indices match the linear interp.
    N, T, D = 1, 64, 36
    frames = torch.zeros(N, T, D)
    # pos ramps: pos_x(k) = k * 0.1 (so at 30 Hz, 0.1 m per 1/30 s)
    frames[..., 0] = torch.arange(T, dtype=torch.float32) * 0.1
    # identity quat (w=1)
    frames[..., 3] = 1.0
    # joint[0] ramps: j0(k) = k * 0.01
    frames[..., 7] = torch.arange(T, dtype=torch.float32) * 0.01

    out = resample_traj_30_to_50hz(frames, num_output=RESAMPLED_FRAMES)

    # At 50Hz idx 0 → 30Hz idx 0 → pos_x = 0.0, j0 = 0.0
    r.check_allclose("idx=0 pos_x", out[0, 0, 0], torch.tensor(0.0))
    r.check_allclose("idx=0 j0", out[0, 0, 7], torch.tensor(0.0))

    # At 50Hz idx 5 → 30Hz t = 5 * (30/50) = 3.0 → pos_x = 0.3, j0 = 0.03
    r.check_allclose("idx=5 pos_x", out[0, 5, 0], torch.tensor(0.3), atol=1e-5)
    r.check_allclose("idx=5 j0", out[0, 5, 7], torch.tensor(0.03), atol=1e-5)

    # At 50Hz idx 3 → t=1.8 → pos_x = 0.18 (lerp between 0.1 and 0.2)
    r.check_allclose("idx=3 pos_x", out[0, 3, 0], torch.tensor(0.18), atol=1e-5)

    # Identity quat preserved everywhere (no rotation)
    r.check_allclose(
        "all quats stay identity",
        out[0, :, 3:7],
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(RESAMPLED_FRAMES, 4),
        atol=1e-5,
    )


# ---------------------------------------------------------------------------
# D. History ring-buffer roll direction
# ---------------------------------------------------------------------------


def test_history_roll(r: Reporter) -> None:
    section("D. history roll direction (torch.roll(-1, dim=1), write at [:, -1])")

    N = 2
    his = torch.zeros(N, HIST_LEN, NUM_JOINTS)
    for t in range(HIST_LEN + 2):
        his = torch.roll(his, -1, dims=1)
        his[:, -1, 0] = float(t)

    # After HIST_LEN+2 writes (t = 0..HIST_LEN+1), newest (idx -1) should be
    # HIST_LEN+1 and the frame at idx 0 should be (HIST_LEN+2 - HIST_LEN) = 2.
    expected_newest = float(HIST_LEN + 1)
    expected_oldest = float(2)
    r.check_allclose("newest at [-1, 0]", his[0, -1, 0], torch.tensor(expected_newest))
    r.check_allclose("oldest at [0, 0]", his[0, 0, 0], torch.tensor(expected_oldest))

    # Full time-ordered sequence should be t_oldest .. t_newest in idx 0..-1
    expected_seq = torch.tensor(
        [float(t) for t in range(2, 2 + HIST_LEN)], dtype=torch.float32
    )
    r.check_allclose("monotonic window", his[0, :, 0], expected_seq)


# ---------------------------------------------------------------------------
# F. Encoder + decoder smoke at standing pose
# ---------------------------------------------------------------------------


def _make_standing_teleop_obs(N: int) -> np.ndarray:
    """267-D teleop obs: zero lower body motion + fixed VR 3pt + identity anchor.

    Layout (from sonic_inference.py step()):
        lower_pos(120) + lower_vel(120) + vr_pos(9) + vr_orn(12) + anchor_6d(6)
    """
    obs = np.zeros((N, TELEOP_INPUT_DIM), dtype=np.float32)
    # vr_pos slice: ALL zero means 3 VR points at root origin (crude proxy for "hands at chest").
    # vr_orn slice (12): 3 × wxyz identity
    vr_orn_off = 120 + 120 + 9
    for k in range(3):
        obs[:, vr_orn_off + 4 * k + 0] = 1.0  # w
    # anchor_6d: identity rotation → [1,0,0,1,0,0]
    anc = 120 + 120 + 9 + 12
    obs[:, anc + 0] = 1.0
    obs[:, anc + 3] = 1.0
    return obs


def test_encoder_smoke(r: Reporter) -> None:
    section("F1. encoder smoke at standing / identity VR")

    if not ENCODER_ONNX.exists():
        r.bad("encoder smoke", f"missing {ENCODER_ONNX}")
        return
    sess = ort.InferenceSession(str(ENCODER_ONNX), providers=["CPUExecutionProvider"])

    for N in (1, 4):
        obs = _make_standing_teleop_obs(N)
        token = sess.run(None, {"teleop_obs": obs})[0]
        print(f"    N={N}: token shape={token.shape}, "
              f"range=[{token.min():.3f}, {token.max():.3f}], "
              f"std={token.std():.3f}, nnz={(token != 0).sum()}/{token.size}")

        if token.shape != (N, TOKEN_DIM):
            r.bad(f"N={N} shape", f"got {token.shape}, want ({N}, {TOKEN_DIM})")
            continue

        if not np.isfinite(token).all():
            r.bad(f"N={N} finite", "token contains NaN or Inf")
            continue

        # FSQ output is a discrete codebook: every latent dim is quantized to one
        # of 32 levels in [-limit, limit]. Values should be bounded and not all
        # zero. We don't know the exact codebook magnitudes, so just sanity-check
        # bounds and non-triviality.
        if float(np.max(np.abs(token))) > 10.0:
            r.bad(f"N={N} magnitude", f"|token|>10 at standing pose ({float(np.max(np.abs(token))):.3f})")
            continue
        if float(np.std(token)) < 1e-4:
            r.bad(f"N={N} non-trivial", f"token has ~zero std ({float(np.std(token)):.2e})")
            continue

        # Determinism: same input → same token across two calls
        token2 = sess.run(None, {"teleop_obs": obs})[0]
        if not np.allclose(token, token2):
            r.bad(f"N={N} determinism", "different outputs on repeat call")
            continue

        # Batch independence: row i of token[N] == token of obs[i:i+1] run alone
        if N > 1:
            solo = sess.run(None, {"teleop_obs": obs[0:1]})[0]
            if not np.allclose(token[0:1], solo, atol=1e-5):
                r.bad(f"N={N} batch independence", "row 0 of batch != solo run of obs[0]")
                continue

        r.ok(f"N={N} standing token")


def test_decoder_smoke(r: Reporter) -> None:
    section("F2. decoder smoke at zero proprio")

    if not DECODER_ONNX.exists():
        r.bad("decoder smoke", f"missing {DECODER_ONNX}")
        return
    sess = ort.InferenceSession(str(DECODER_ONNX), providers=["CPUExecutionProvider"])

    for N in (1, 4):
        # Token = zeros (ATM has never actually seen this exactly — but should be
        # a "neutral-ish" token; we're just checking shape/finiteness here).
        # Proprio = zeros: gravity=0 (wrong but well-behaved), ang_vel=0,
        # joint_pos_rel=0 (= default pose), joint_vel=0, last_action=0.
        dec_in = np.zeros((N, DECODER_INPUT_DIM), dtype=np.float32)
        action = sess.run(None, {"decoder_input": dec_in})[0]
        print(f"    N={N}: action shape={action.shape}, "
              f"range=[{action.min():.3f}, {action.max():.3f}], "
              f"|mean|={np.abs(action).mean():.3f}")

        if action.shape != (N, ACTION_DIM):
            r.bad(f"N={N} shape", f"got {action.shape}, want ({N}, {ACTION_DIM})")
            continue

        if not np.isfinite(action).all():
            r.bad(f"N={N} finite", "action contains NaN or Inf")
            continue
        if float(np.max(np.abs(action))) > 20.0:
            r.bad(f"N={N} magnitude", f"|action| unreasonably large ({float(np.max(np.abs(action))):.3f})")
            continue

        # Determinism
        action2 = sess.run(None, {"decoder_input": dec_in})[0]
        if not np.allclose(action, action2):
            r.bad(f"N={N} determinism", "different outputs on repeat call")
            continue

        r.ok(f"N={N} zero-input action")


def test_proprio_layout(r: Reporter) -> None:
    section("F3. proprio layout — total 930, key order = sonic_release config")

    total = sum(PROPRIO_DIMS[k] for k in PROPRIO_KEYS_ORDER)
    if total != PROPRIO_TOTAL:
        r.bad("PROPRIO_TOTAL", f"sum of PROPRIO_DIMS = {total} != {PROPRIO_TOTAL}")
        return
    if PROPRIO_TOTAL + TOKEN_DIM != DECODER_INPUT_DIM:
        r.bad("decoder width", f"{TOKEN_DIM}+{PROPRIO_TOTAL} != {DECODER_INPUT_DIM}")
        return
    # Expected order = Isaac Lab ObservationManager iteration order on
    # PolicyCfg (@configclass = stdlib dataclass → __dict__ follows class
    # declaration order). Training's PolicyCfg declares base_ang_vel, joint_pos,
    # joint_vel, actions first, then gravity_dir last. Deploy observation_config
    # agrees (gravity LAST). sonic_release/config.yaml:437-475 lists gravity
    # first but that's a documentation re-ordering, not what the runtime sees.
    expected = (
        "his_base_angular_velocity_10frame_step1",
        "his_body_joint_positions_10frame_step1",
        "his_body_joint_velocities_10frame_step1",
        "his_last_actions_10frame_step1",
        "his_gravity_dir_10frame_step1",
    )
    if PROPRIO_KEYS_ORDER != expected:
        r.bad("key order", f"{PROPRIO_KEYS_ORDER} vs expected {expected}")
        return
    r.ok(f"proprio keys {len(expected)}, total {PROPRIO_TOTAL} + token {TOKEN_DIM} = {DECODER_INPUT_DIM}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    r = Reporter()

    test_onnx_schema(r)
    test_quat_to_6d(r)
    test_slerp(r)
    test_anchor_formula(r)
    test_resample(r)
    test_history_roll(r)
    test_proprio_layout(r)
    test_encoder_smoke(r)
    test_decoder_smoke(r)

    return r.finish()


if __name__ == "__main__":
    sys.exit(main())
