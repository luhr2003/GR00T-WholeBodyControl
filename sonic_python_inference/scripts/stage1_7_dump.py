"""Stage 1.7 — dump the full pipeline's intermediate tensors for the first K policy ticks.

Boots Isaac Lab the same way stage1_loco_only does, but:
    - Only runs K policy ticks (default 3).
    - Intercepts encoder/decoder ONNX calls to capture the 267-D teleop_obs and
      994-D decoder_input, plus their outputs (64-D token, 29-D action).
    - Prints every observation field broken out by name, with shape/range/stats.
    - Cross-checks each field against a standing-pose expectation. Anything
      outside the expected range is flagged with `[!!]`.

Use this when stage1 goes sideways but stage1.5 subtests all pass — the bug is
somewhere in how Isaac Lab is feeding the pipeline (DOF order, coord frame,
missing offset, etc). The dump tells you which feature group is wrong.

Run:
    uv run --active python -m sonic_python_inference.scripts.stage1_7_dump \
        --num-envs 2 --num-policy-steps 3 --headless
"""

from __future__ import annotations

import argparse


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=2)
    ap.add_argument("--num-policy-steps", type=int, default=3)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument(
        "--encoder-onnx",
        type=str,
        default="sonic_python_inference/assets/encoder_dyn.onnx",
    )
    ap.add_argument(
        "--decoder-onnx",
        type=str,
        default="sonic_python_inference/assets/decoder_dyn.onnx",
    )
    ap.add_argument(
        "--planner-onnx",
        type=str,
        default="gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx",
    )
    return ap.parse_args()


args = _parse_args()

# ---------------------------------------------------------------------------
# AppLauncher MUST come before any other isaaclab import.
# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.terrains import TerrainImporterCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from gear_sonic.envs.manager_env.robots.g1 import (  # noqa: E402
    G1_CYLINDER_MODEL_12_DEX_CFG,
    G1_ISAACLAB_TO_MUJOCO_DOF,
    G1_MODEL_12_ACTION_SCALE,
)

from sonic_python_inference.sonic_inference import (  # noqa: E402
    HIST_LEN,
    NUM_JOINTS,
    PLANNER_HEIGHT_DEFAULT,
    PLANNER_MODE_IDLE,
    PROPRIO_DIMS,
    PROPRIO_KEYS_ORDER,
    SonicVR3PTInference,
    quat_conjugate,
    quat_mul,
    quat_rotate,
)


SIM_DT = 0.005
DECIMATION = 4

VR_3PT_BODY_NAMES = ("left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link")
VR_3PT_BODY_OFFSETS = (
    (0.18, -0.025, 0.0),
    (0.18, 0.025, 0.0),
    (0.0, 0.0, 0.35),
)

# Expected ranges at standing pose (first tick, before physics diverges).
# Tighter = more useful. These are what the policy was trained to see at rest.
EXPECT = {
    "gravity_in_base":        {"near": [0.0, 0.0, -1.0], "atol": 0.05},
    "base_ang_vel":           {"near": [0.0, 0.0, 0.0], "atol": 0.1},
    "joint_pos_rel":          {"near_zero": True, "atol": 0.05},  # joint_pos ≈ default
    "joint_vel":              {"near_zero": True, "atol": 0.5},
    "last_action":            {"near_zero": True, "atol": 1e-6},  # first tick only
    "vr_pos_local":           {"finite": True, "abs_max": 1.5},  # wrists/head within 1.5m of root
    "vr_orn_local":           {"unit_quat": True},
    "anchor_6d":              {"near_identity_6d": True, "atol": 0.2},
    "lower_pos_planner_dev":  {"near_zero": True, "atol": 0.5},  # planner lower body ≈ default
    "lower_vel_planner":      {"finite": True, "abs_max": 20.0},
    "token":                  {"finite": True, "abs_max": 5.0},  # FSQ bounded
    "action":                 {"finite": True, "abs_max": 3.0},  # decoder output ≈ [-1,1] scale
}


@configclass
class SonicStage17SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
    )
    robot = G1_CYLINDER_MODEL_12_DEX_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(1.0, 1.0, 1.0), intensity=2000.0),
    )


# ---------------------------------------------------------------------------
# Printing / checking helpers
# ---------------------------------------------------------------------------


def _stats(name: str, x, expect: dict | None = None) -> str:
    a = np.asarray(x)
    line = (
        f"    {name:<30s} shape={str(a.shape):<18s} "
        f"range=[{a.min():+8.4f}, {a.max():+8.4f}] "
        f"|mean|={np.abs(a).mean():7.4f} std={a.std():7.4f}"
    )
    flags = []
    if expect is not None:
        if expect.get("finite", False):
            if not np.isfinite(a).all():
                flags.append("!!NaN/Inf")
        if "abs_max" in expect:
            if float(np.max(np.abs(a))) > expect["abs_max"]:
                flags.append(f"!!|x|>{expect['abs_max']}")
        if expect.get("near_zero", False):
            atol = expect.get("atol", 0.1)
            if float(np.max(np.abs(a))) > atol:
                flags.append(f"!!not≈0 (atol={atol})")
        if "near" in expect:
            near = np.broadcast_to(np.asarray(expect["near"], dtype=a.dtype), a.shape)
            atol = expect.get("atol", 0.1)
            if float(np.max(np.abs(a - near))) > atol:
                flags.append(f"!!not≈{expect['near']} (atol={atol})")
        if expect.get("unit_quat", False):
            # reshape last axis into 4 and check ||q||≈1
            if a.shape[-1] % 4 != 0:
                flags.append("!!not quat-shaped")
            else:
                q = a.reshape(*a.shape[:-1], -1, 4)
                norms = np.linalg.norm(q, axis=-1)
                if float(np.max(np.abs(norms - 1.0))) > 1e-3:
                    flags.append(f"!!|q|≠1 max_err={float(np.max(np.abs(norms - 1.0))):.2e}")
        if expect.get("near_identity_6d", False):
            # 6-D = [R00, R01, R10, R11, R20, R21]; identity = [1,0,0,1,0,0]
            ident = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=a.dtype)
            if a.shape[-1] != 6:
                flags.append("!!not 6D")
            else:
                atol = expect.get("atol", 0.1)
                per_env_err = np.max(np.abs(a - ident), axis=-1)
                if float(np.max(per_env_err)) > atol:
                    flags.append(f"!!6D≠identity max|Δ|={float(np.max(per_env_err)):.3f}")
    if flags:
        line += "   " + " ".join(flags)
    return line


def _short(x, k: int = 6) -> str:
    a = np.asarray(x).ravel()
    head = ", ".join(f"{v:+.3f}" for v in a[:k])
    tail = "…" if a.size > k else ""
    return f"[{head}{tail}]"


def _initial_vr_3pt_root_local(
    body_pos_w: torch.Tensor,
    body_quat_w: torch.Tensor,
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    body_offsets: tuple[tuple[float, float, float], ...],
):
    """Same as stage1_loco_only._initial_vr_3pt_root_local — copied to avoid a cross-script import."""
    N = body_pos_w.shape[0]
    device = body_pos_w.device
    root_quat_inv = quat_conjugate(root_quat_w)
    pos_list, orn_list = [], []
    for i, off in enumerate(body_offsets):
        bp = body_pos_w[:, i]
        bq = body_quat_w[:, i]
        off_b = torch.tensor(off, dtype=torch.float32, device=device).expand(N, 3)
        off_w = quat_rotate(bq, off_b)
        vr_w = bp + off_w
        rel = vr_w - root_pos_w
        pos_list.append(quat_rotate(root_quat_inv, rel))
        orn_list.append(quat_mul(root_quat_inv, bq))
    return torch.cat(pos_list, dim=-1), torch.cat(orn_list, dim=-1)


def _action_scale_il(joint_names: list[str]) -> np.ndarray:
    scale = np.zeros(NUM_JOINTS, dtype=np.float32)
    for i, name in enumerate(joint_names):
        scale[i] = G1_MODEL_12_ACTION_SCALE.get(name, 1.0)
    return scale


# ---------------------------------------------------------------------------
# ONNX call interception
# ---------------------------------------------------------------------------


class OnnxTap:
    """Wraps an ort.InferenceSession.run to record the last {inputs, outputs}."""

    def __init__(self, sess):
        self.sess = sess
        self._orig_run = sess.run
        self.last_feeds: dict | None = None
        self.last_outputs = None
        sess.run = self._run  # type: ignore[assignment]

    def _run(self, output_names, feeds):
        self.last_feeds = {k: np.asarray(v).copy() for k, v in feeds.items()}
        out = self._orig_run(output_names, feeds)
        self.last_outputs = [np.asarray(o).copy() for o in out]
        return out


# ---------------------------------------------------------------------------
# Per-tick dump
# ---------------------------------------------------------------------------


def dump_tick(
    t: int,
    infer: SonicVR3PTInference,
    enc_tap: OnnxTap,
    dec_tap: OnnxTap,
    robot_state_before: dict,
    target_il: torch.Tensor,
) -> None:
    print()
    print(f"=== tick t={t} (step_counter_after={infer.step_counter}) ===")

    # --- Raw Isaac Lab state passed in ------------------------------------
    print("  [inputs from Isaac Lab]")
    for k, v in robot_state_before.items():
        print(_stats(k, v.cpu().numpy(), EXPECT.get(k)))

    # --- Teleop obs broken out --------------------------------------------
    teleop = enc_tap.last_feeds["teleop_obs"]  # [N, 267]
    lower_pos = teleop[:, 0:120].reshape(teleop.shape[0], HIST_LEN, -1)  # [N, 10, 12]
    lower_vel = teleop[:, 120:240].reshape(teleop.shape[0], HIST_LEN, -1)
    vr_pos = teleop[:, 240:249]
    vr_orn = teleop[:, 249:261]
    anchor_6d = teleop[:, 261:267]
    print("  [teleop_obs breakdown]")
    print(_stats("lower_pos (planner, 10×12)", lower_pos))
    # `planner lower body - default` would need the MJ default; here just check
    # that the current-frame lower pos isn't wildly off.
    print(_stats("lower_pos[now] (frame 0)", lower_pos[:, 0, :]))
    print(_stats("lower_vel (planner, 10×12)", lower_vel, EXPECT["lower_vel_planner"]))
    print(_stats("vr_pos_local (3×3)", vr_pos, EXPECT["vr_pos_local"]))
    print(_stats("vr_orn_local (3×4)", vr_orn, EXPECT["vr_orn_local"]))
    print(_stats("anchor_6d", anchor_6d, EXPECT["anchor_6d"]))
    # Print the anchor row-by-row to see if e.g. yaw is drifting
    for i in range(teleop.shape[0]):
        print(f"      env{i} anchor_6d = {_short(anchor_6d[i], 6)}")

    # --- Token output ------------------------------------------------------
    token = enc_tap.last_outputs[0]  # [N, 64]
    print("  [encoder token]")
    print(_stats("token", token, EXPECT["token"]))
    print(f"      env0 token[:12] = {_short(token[0], 12)}")

    # --- Decoder input (proprio + token) -----------------------------------
    dec_in = dec_tap.last_feeds["decoder_input"]  # [N, 994]
    proprio = dec_in[:, 64:]  # [N, 930]
    print("  [decoder proprio breakdown]")
    # Fields in order: PROPRIO_KEYS_ORDER, widths from PROPRIO_DIMS
    off = 0
    for key in PROPRIO_KEYS_ORDER:
        w = PROPRIO_DIMS[key]
        blk = proprio[:, off : off + w]
        # Find which feature this is
        if "gravity" in key:
            exp = EXPECT["gravity_in_base"]  # each row is gravity
            blk_current = blk.reshape(blk.shape[0], HIST_LEN, -1)[:, -1]  # newest
            print(_stats(key + " [newest]", blk_current, exp))
        elif "base_angular_velocity" in key:
            exp = EXPECT["base_ang_vel"]
            blk_current = blk.reshape(blk.shape[0], HIST_LEN, -1)[:, -1]
            print(_stats(key + " [newest]", blk_current, exp))
        elif "joint_positions" in key:
            # stored as joint_pos_rel = joint_pos - default
            blk_current = blk.reshape(blk.shape[0], HIST_LEN, -1)[:, -1]
            print(_stats(key + " [newest, joint_pos_rel]", blk_current, EXPECT["joint_pos_rel"]))
        elif "joint_velocities" in key:
            blk_current = blk.reshape(blk.shape[0], HIST_LEN, -1)[:, -1]
            print(_stats(key + " [newest]", blk_current, EXPECT["joint_vel"]))
        elif "last_actions" in key:
            blk_current = blk.reshape(blk.shape[0], HIST_LEN, -1)[:, -1]
            exp = EXPECT["last_action"] if t == 0 else {"finite": True, "abs_max": 3.0}
            print(_stats(key + " [newest]", blk_current, exp))
        else:
            print(_stats(key, blk, {"finite": True}))
        off += w

    # --- Action ------------------------------------------------------------
    action = dec_tap.last_outputs[0]  # [N, 29]
    print("  [decoder action (pre-scale)]")
    print(_stats("action", action, EXPECT["action"]))
    print(f"      env0 action = {_short(action[0], 8)}")

    # --- Final target ------------------------------------------------------
    target_np = target_il.detach().cpu().numpy()
    print("  [motor targets (IL order) = default + scale * action]")
    print(_stats("target_il", target_np, {"finite": True}))
    print(f"      env0 target[:8] = {_short(target_np[0], 8)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    device = "cuda"

    sim_cfg = sim_utils.SimulationCfg(device=device, dt=SIM_DT)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.75])

    scene_cfg = SonicStage17SceneCfg(num_envs=args.num_envs, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    robot = scene["robot"]
    scene.update(dt=0.0)

    il_joint_names = list(robot.data.joint_names)
    il_body_names = list(robot.data.body_names)
    vr_body_idx = torch.as_tensor(
        [il_body_names.index(n) for n in VR_3PT_BODY_NAMES], dtype=torch.long, device=device
    )

    default_joint_pos_il = robot.data.default_joint_pos[0].cpu().numpy().astype(np.float32)
    action_scale_il = _action_scale_il(il_joint_names)

    print("=== static info ===")
    print(f"  IL joint names ({len(il_joint_names)}): {il_joint_names}")
    print(f"  VR body names → IL indices: {list(zip(VR_3PT_BODY_NAMES, vr_body_idx.tolist()))}")
    print(f"  default_joint_pos_il = {_short(default_joint_pos_il, 29)}")
    print(f"  action_scale_il      = {_short(action_scale_il, 29)}")

    infer = SonicVR3PTInference(
        num_envs=args.num_envs,
        encoder_onnx=args.encoder_onnx,
        decoder_onnx=args.decoder_onnx,
        planner_onnx=args.planner_onnx,
        default_angles=default_joint_pos_il,
        action_scale=action_scale_il,
        isaaclab_to_mujoco_dof=G1_ISAACLAB_TO_MUJOCO_DOF,
        device=device,
    )

    enc_tap = OnnxTap(infer.enc_sess)
    dec_tap = OnnxTap(infer.dec_sess)

    # `sim.reset()` populates `robot.data.default_*` from the config but
    # does NOT push those values into PhysX — the articulation boots at
    # joint_pos=0 unless we write it. Canonical Isaac Lab reset (same
    # pattern the GR00T training env uses — `write_joint_state_to_sim` +
    # `write_root_state_to_sim`).
    default_joint_pos_t = robot.data.default_joint_pos.clone()
    default_joint_vel_t = robot.data.default_joint_vel.clone()
    default_root_state = robot.data.default_root_state.clone()
    default_root_state[:, 0:3] += scene.env_origins
    robot.write_root_state_to_sim(default_root_state)
    robot.write_joint_state_to_sim(default_joint_pos_t, default_joint_vel_t)
    robot.set_joint_position_target(default_joint_pos_t)
    scene.write_data_to_sim()
    scene.update(dt=0.0)

    # Initial VR 3pt (frozen for the whole episode, same as stage1)
    joint_pos_il = robot.data.joint_pos.clone()
    root_state = robot.data.root_state_w.clone()
    root_pos_w = root_state[:, 0:3] - scene.env_origins
    root_quat_w = root_state[:, 3:7]
    body_state = robot.data.body_state_w.clone()
    body_pos_w = body_state[..., 0:3] - scene.env_origins[:, None, :]
    body_quat_w = body_state[..., 3:7]
    vr_pos_w_3 = body_pos_w[:, vr_body_idx]
    vr_quat_w_3 = body_quat_w[:, vr_body_idx]

    vr_pos_local, vr_orn_local = _initial_vr_3pt_root_local(
        body_pos_w=vr_pos_w_3,
        body_quat_w=vr_quat_w_3,
        root_pos_w=root_pos_w,
        root_quat_w=root_quat_w,
        body_offsets=VR_3PT_BODY_OFFSETS,
    )

    print()
    print("=== one-shot state before first tick ===")
    print(_stats("root_pos_w", root_pos_w.cpu().numpy()))
    print(f"      env0 root_pos_w = {_short(root_pos_w[0].cpu().numpy(), 3)}")
    print(_stats("root_quat_wxyz", root_quat_w.cpu().numpy(), {"unit_quat": True}))
    print(f"      env0 root_quat_wxyz = {_short(root_quat_w[0].cpu().numpy(), 4)}")
    print(_stats("vr_pos_local (frozen)", vr_pos_local.cpu().numpy(), EXPECT["vr_pos_local"]))
    print(f"      env0 vr_pos_local = {_short(vr_pos_local[0].cpu().numpy(), 9)}")
    print(_stats("vr_orn_local (frozen)", vr_orn_local.cpu().numpy(), EXPECT["vr_orn_local"]))

    infer.reset(joint_pos=joint_pos_il, root_pos=root_pos_w, root_quat_wxyz=root_quat_w)

    # Command — IDLE stand-still (no gait, no movement). Isolates whether
    # the pipeline can just hold the default pose before we trust any gait.
    N = args.num_envs
    mode = torch.full((N,), PLANNER_MODE_IDLE, dtype=torch.long, device=device)
    movement = torch.tensor([[0.0, 0.0, 0.0]] * N, dtype=torch.float32, device=device)
    facing = torch.tensor([[1.0, 0.0, 0.0]] * N, dtype=torch.float32, device=device)
    target_vel = torch.full((N,), 0.0, dtype=torch.float32, device=device)
    height = torch.full((N,), PLANNER_HEIGHT_DEFAULT, dtype=torch.float32, device=device)

    for t in range(args.num_policy_steps):
        joint_pos_il = robot.data.joint_pos.clone()
        joint_vel_il = robot.data.joint_vel.clone()
        root_state = robot.data.root_state_w.clone()
        root_pos_w = root_state[:, 0:3] - scene.env_origins
        root_quat_w = root_state[:, 3:7]
        base_ang_vel_b = robot.data.root_ang_vel_b.clone()
        gravity_b = robot.data.projected_gravity_b.clone()

        robot_state_before = {
            "joint_pos_rel": joint_pos_il - torch.as_tensor(default_joint_pos_il, device=device),
            "joint_vel": joint_vel_il,
            "root_pos_w": root_pos_w,
            "root_quat_wxyz": root_quat_w,
            "base_ang_vel": base_ang_vel_b,
            "gravity_in_base": gravity_b,
        }

        target_il = infer.step(
            vr_3pt_position=vr_pos_local,
            vr_3pt_orientation=vr_orn_local,
            mode=mode,
            movement_direction=movement,
            facing_direction=facing,
            target_vel=target_vel,
            height=height,
            joint_pos=joint_pos_il,
            joint_vel=joint_vel_il,
            base_ang_vel=base_ang_vel_b,
            gravity_in_base=gravity_b,
            root_pos=root_pos_w,
            root_quat_wxyz=root_quat_w,
        )

        dump_tick(t, infer, enc_tap, dec_tap, robot_state_before, target_il)

        # Step the sim forward so tick t+1 sees evolved state
        for _ in range(DECIMATION):
            robot.set_joint_position_target(target_il)
            scene.write_data_to_sim()
            sim.step()
            scene.update(dt=SIM_DT)

    print("\n=== done ===")
    simulation_app.close()


if __name__ == "__main__":
    main()
