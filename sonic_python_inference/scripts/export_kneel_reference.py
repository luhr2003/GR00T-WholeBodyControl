"""Export G1 one-leg kneel joint references across pelvis heights, + video.

Derived from ``stage_hybrid_kneel.py`` and the MagicSim-USD variant
(``stage_hybrid_ik_magicsim.py``). Differences:

- **Legs only, no upper-body IK.** The original hybrid pipeline drives the
  upper 17 DOF with Pink IK (needs ``pinocchio``). The kneel *reference* we
  want is the lower body, so we drop Pink IK and hold the upper 17 joints at
  their default rest angles. Removes the pinocchio dependency; the leg kneel
  pose is unaffected (the original demo holds the arms near rest anyway).
- **MagicSim USD robot by default.** Spawns from ``assets/g1_magicsim.usd``
  (copy of ``MagicSim/Assets/Robots/g1_new.usd``) instead of converting
  SONIC's URDF on every boot — far faster startup, no LFS mesh dependency.
  The USD exposes 43 joints (29 body + 14 dex fingers); we map to the 29
  SONIC body DOFs via ``g1_body_indices_in_training_order`` and hold the
  fingers at default. ``--robot urdf`` falls back to the gear_sonic URDF.
- **Sweep + capture.** Boots Isaac once, then loops over
  ``heights × seeds`` one-leg kneels. Per config: run the closed loop
  (planner mode 6 ``kneelOneLeg`` → G1 encoder → decoder → PD), hold the
  crouch, record a video, and capture two LEG references:
    * ``planner_reference``  — the planner's kinematic kneel target (12 leg DOF)
    * ``settled_achieved``   — the pose the robot physically settles into
- **Side detection.** Mode 6 picks the grounded knee from ``random_seed``.
  We detect the grounded side from knee-link world heights (kneeling knee is
  lowest) and label by the detected side, trying seeds until both sides land.

Outputs (under ``--out-dir``):
    kneel_left.json   — left-knee-down references, one entry per height
    kneel_right.json  — right-knee-down references, one entry per height
    videos/kneel_<side>_h<height>.mp4

Usage:
    OMNI_KIT_ACCEPT_EULA=YES uv run --active python -m \
        sonic_python_inference.scripts.export_kneel_reference \
        --heights 0.20 0.25 0.30 0.35 0.40 --out-dir kneel_reference_out
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np


PLANNER_MODE_KNEEL_ONE_LEG = 6  # docs/source/references/planner_onnx.md


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heights", type=float, nargs="+",
                    default=[0.20, 0.25, 0.30, 0.35, 0.40])
    ap.add_argument("--side", choices=["both", "right", "left"], default="both",
                    help="kneelOneLeg deterministically grounds the RIGHT knee; "
                         "'left' is produced by sagittally mirroring the planner "
                         "reference each tick and simulating it.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Planner seed (no effect on side in V2 planner).")
    ap.add_argument("--crouch-steps", type=int, default=250,
                    help="Policy steps to hold the crouch before capture (50Hz).")
    ap.add_argument("--out-dir", type=str, default="kneel_reference_out")
    ap.add_argument("--robot", choices=["magicsim", "urdf"], default="magicsim")
    ap.add_argument("--magicsim-usd-path", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--cam-width", type=int, default=640)
    ap.add_argument("--cam-height", type=int, default=480)
    ap.add_argument("--render-every", type=int, default=2,
                    help="Capture a video frame every N policy steps.")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--headless", action="store_true", default=True)
    ap.add_argument("--gui", dest="headless", action="store_false")
    ap.add_argument("--g1-encoder-onnx", type=str,
                    default="sonic_python_inference/assets/g1_encoder_dyn.onnx")
    ap.add_argument("--decoder-onnx", type=str,
                    default="sonic_python_inference/assets/decoder_dyn.onnx")
    ap.add_argument("--planner-onnx", type=str,
                    default="gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx")
    return ap.parse_args()


args = _parse_args()

from isaaclab.app import AppLauncher  # noqa: E402

# enable_cameras must be on whenever the scene contains a CameraCfg (it always
# does here); --no-video only skips frame *capture*, not camera creation.
app_launcher = AppLauncher(headless=args.headless, enable_cameras=True)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.terrains import TerrainImporterCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from gear_sonic.envs.manager_env.robots.g1 import (  # noqa: E402
    G1_CYLINDER_MODEL_12_DEX_CFG,
    G1_ISAACLAB_TO_MUJOCO_DOF,
    G1_MODEL_12_ACTION_SCALE,
    G1_MUJOCO_TO_ISAACLAB_DOF,
)
from sonic_python_inference.g1_magicsim_cfg import (  # noqa: E402
    DEFAULT_MAGICSIM_USD_PATH,
    make_g1_magicsim_cfg,
)
from sonic_python_inference.sonic_inference import (  # noqa: E402
    ALLOWED_PRED_NUM_TOKENS,
    NUM_JOINTS,
    PLANNER_CONTEXT_DEFAULT_HEIGHT,
    PLANNER_EVERY_K_POLICY_STEPS,
    POLICY_HZ,
    RESAMPLED_FRAMES,
    resample_traj_30_to_50hz,
    slerp_torch,
)
from sonic_python_inference.sonic_planner_pool import (  # noqa: E402
    PLANNER_FRAME_DIM,
    PlannerSessionPool,
)
from sonic_python_inference.sonic_g1_inference import (  # noqa: E402
    G1_NUM_FUTURE_FRAMES,
    SonicG1Inference,
)
from sonic_python_inference.joint_order import (  # noqa: E402
    g1_body_indices_in_training_order,
)

import imageio.v2 as imageio  # noqa: E402  (per-height stills always; video optional)


SIM_DT = 0.005
DECIMATION = 4
LEG_NAME_PATTERNS = (
    r".*_hip_.*_joint",
    r".*_knee_joint",
    r".*_ankle_.*_joint",
)
HAND_RE = re.compile(r".*_hand_.*_joint")

if args.robot == "magicsim":
    _usd = args.magicsim_usd_path or DEFAULT_MAGICSIM_USD_PATH
    print(f"[info] robot=magicsim USD: {_usd}")
    _ROBOT_CFG = make_g1_magicsim_cfg(_usd)
else:
    print("[info] robot=urdf (gear_sonic main.urdf, on-the-fly URDF->USD)")
    _ROBOT_CFG = G1_CYLINDER_MODEL_12_DEX_CFG


@configclass
class KneelRefSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="plane", collision_group=-1)
    robot = _ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(1.0, 1.0, 1.0), intensity=2000.0))
    camera = CameraCfg(
        prim_path="/World/RefCam",
        update_period=0.0,
        height=args.cam_height,
        width=args.cam_width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 100.0)),
    )


def _action_scale(joint_names: list[str]) -> np.ndarray:
    scale = np.ones(len(joint_names), dtype=np.float32)
    patterns = [(re.compile(p), v) for p, v in G1_MODEL_12_ACTION_SCALE.items()]
    for i, name in enumerate(joint_names):
        for pat, v in patterns:
            if pat.fullmatch(name):
                scale[i] = float(v)
                break
        else:
            raise RuntimeError(f"No scale pattern matches joint '{name}'.")
    return scale


def _leg_indices(joint_names: list[str]) -> list[int]:
    leg_re = [re.compile(p) for p in LEG_NAME_PATTERNS]
    return [i for i, n in enumerate(joint_names) if any(r.fullmatch(n) for r in leg_re)]


def _build_planner_feeds(ctx_np, height, movement_dir, facing_dir, seed):
    allowed_mask = ALLOWED_PRED_NUM_TOKENS.reshape(1, 11)
    return [{
        "context_mujoco_qpos": ctx_np[0:1].astype(np.float32),
        "target_vel": np.array([0.0], dtype=np.float32),
        "mode": np.array([PLANNER_MODE_KNEEL_ONE_LEG], dtype=np.int64),
        "movement_direction": movement_dir.reshape(1, 3).astype(np.float32),
        "facing_direction": facing_dir.reshape(1, 3).astype(np.float32),
        "random_seed": np.array([seed], dtype=np.int64),
        "has_specific_target": np.zeros((1, 1), dtype=np.int64),
        "specific_target_positions": np.zeros((1, 4, 3), dtype=np.float32),
        "specific_target_headings": np.zeros((1, 4), dtype=np.float32),
        "allowed_pred_num_tokens": allowed_mask,
        "height": np.array([height], dtype=np.float32),
    }]


def _context_from_cache(cache, playback_idx):
    N, L, _ = cache.shape
    device = cache.device
    t_offsets = torch.arange(4, device=device, dtype=torch.float32) * (50.0 / 30.0)
    idx_f = playback_idx.view(-1, 1).float() + t_offsets.view(1, -1)
    idx0 = idx_f.long().clamp(0, L - 1)
    idx1 = (idx0 + 1).clamp(0, L - 1)
    alpha = (idx_f - idx0.float()).clamp(0.0, 1.0)
    n_range = torch.arange(N, device=device).view(-1, 1).expand(-1, 4)
    f0 = cache[n_range, idx0]
    f1 = cache[n_range, idx1]
    pos = f0[..., 0:3] + (f1[..., 0:3] - f0[..., 0:3]) * alpha.unsqueeze(-1)
    quat = slerp_torch(f0[..., 3:7], f1[..., 3:7], alpha)
    joints = f0[..., 7:] + (f1[..., 7:] - f0[..., 7:]) * alpha.unsqueeze(-1)
    return torch.cat([pos, quat, joints], dim=-1)


def main():
    device = args.device
    out_dir = Path(args.out_dir)
    video_dir = out_dir / "videos"
    image_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_video:
        video_dir.mkdir(parents=True, exist_ok=True)

    sim_cfg = sim_utils.SimulationCfg(device=device, dt=SIM_DT)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.0, 3.0, 2.0], [0.0, 0.0, 0.75])

    scene_cfg = KneelRefSceneCfg(num_envs=1, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    robot = scene["robot"]
    camera = scene["camera"]
    scene.update(dt=0.0)

    camera.set_world_poses_from_view(
        eyes=torch.tensor([[2.4, 2.0, 1.1]], device=camera.device),
        targets=torch.tensor([[0.0, 0.0, 0.45]], device=camera.device))

    # --- 43->29 body-DOF mapping (identity for the 29-DOF urdf robot) -----
    full_joint_names = list(robot.data.joint_names)
    body_idx_full, body_joint_names = g1_body_indices_in_training_order(
        full_joint_names, HAND_RE)
    assert len(body_idx_full) == NUM_JOINTS, \
        f"expected {NUM_JOINTS} body joints, got {len(body_idx_full)}"
    body_idx_t = torch.as_tensor(body_idx_full, dtype=torch.long, device=device)
    print(f"[info] articulation has {len(full_joint_names)} joints -> "
          f"{NUM_JOINTS} SONIC body DOFs")

    default_jp_full = robot.data.default_joint_pos[0].cpu().numpy().astype(np.float32)
    default_joint_pos_il = default_jp_full[body_idx_full]
    action_scale_il = _action_scale(body_joint_names)

    leg_idx_il = _leg_indices(body_joint_names)
    assert len(leg_idx_il) == 12, f"expected 12 leg joints, got {len(leg_idx_il)}"
    leg_joint_names = [body_joint_names[i] for i in leg_idx_il]
    upper_idx_il = [i for i in range(NUM_JOINTS) if i not in set(leg_idx_il)]
    leg_idx_il_t = torch.as_tensor(leg_idx_il, dtype=torch.long, device=device)
    upper_idx_il_t = torch.as_tensor(upper_idx_il, dtype=torch.long, device=device)

    # Sagittal-mirror map over the 12-leg vector. kneelOneLeg only grounds the
    # right knee; the left-knee kneel is its mirror. Swap L<->R partners and
    # negate the lateral/rotational DOFs (roll, yaw) that flip under a sagittal
    # mirror; keep pitch/knee. Simulation validates this sign convention.
    def _partner(name: str) -> str:
        if name.startswith("left"):
            return "right" + name[len("left"):]
        if name.startswith("right"):
            return "left" + name[len("right"):]
        return name
    leg_mirror_perm = [leg_joint_names.index(_partner(n)) for n in leg_joint_names]
    leg_mirror_sign = np.array(
        [-1.0 if ("_roll_" in n or "_yaw_" in n) else 1.0 for n in leg_joint_names],
        dtype=np.float32)
    leg_mirror_perm_t = torch.as_tensor(leg_mirror_perm, dtype=torch.long, device=device)
    leg_mirror_sign_t = torch.as_tensor(leg_mirror_sign, device=device)

    il_to_mj = torch.as_tensor(G1_ISAACLAB_TO_MUJOCO_DOF, dtype=torch.long, device=device)
    mj_to_il = torch.as_tensor(G1_MUJOCO_TO_ISAACLAB_DOF, dtype=torch.long, device=device)
    leg_mj_slots = mj_to_il[leg_idx_il_t]

    # knee links for grounded-side detection (kneeling knee sits lowest)
    body_names = list(robot.data.body_names)
    knee_idx = {}
    for side in ("left", "right"):
        m = [i for i, n in enumerate(body_names) if n.startswith(side) and "knee" in n]
        assert m, f"no {side} knee link in {body_names}"
        knee_idx[side] = m[0]
    print(f"[info] leg joints (IL order): {leg_joint_names}")

    planner_pool = PlannerSessionPool(args.planner_onnx, pool_size=1, device_id=0, serial=True)
    infer = SonicG1Inference(
        num_envs=1, g1_encoder_onnx=args.g1_encoder_onnx, decoder_onnx=args.decoder_onnx,
        default_angles=default_joint_pos_il, action_scale=action_scale_il, device=device)

    default_jp_il_t = torch.as_tensor(default_joint_pos_il, device=device).unsqueeze(0)
    default_jp_mj_t = default_jp_il_t[:, il_to_mj]
    init_frame = torch.zeros(1, PLANNER_FRAME_DIM, device=device)
    init_frame[:, 2] = PLANNER_CONTEXT_DEFAULT_HEIGHT
    init_frame[:, 3] = 1.0
    init_frame[:, 7:] = default_jp_mj_t

    movement_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    facing_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    future_offsets = torch.arange(0, G1_NUM_FUTURE_FRAMES * 5, 5, dtype=torch.long, device=device)
    env_origins = scene.env_origins
    upper_rest = default_jp_il_t[:, upper_idx_il_t]                 # [1,17]
    default_jp_full_t = robot.data.default_joint_pos.clone()        # [1, full]

    def reset_to_stand():
        root_state = robot.data.default_root_state.clone()
        root_state[:, 0:3] += env_origins
        root_state[:, 7:13] = 0.0
        jp = robot.data.default_joint_pos.clone()
        jv = torch.zeros_like(jp)
        robot.write_root_state_to_sim(root_state)
        robot.write_joint_state_to_sim(jp, jv)
        robot.set_joint_position_target(jp)
        scene.write_data_to_sim()
        scene.update(dt=0.0)
        infer.reset(joint_pos=jp[:, body_idx_t])

    def detect_grounded_side() -> str:
        z = robot.data.body_pos_w[0, :, 2]
        return "left" if float(z[knee_idx["left"]]) < float(z[knee_idx["right"]]) else "right"

    def grab_frame():
        sim.render()
        camera.update(dt=SIM_DT * DECIMATION, force_recompute=True)
        rgb = camera.data.output["rgb"][0].detach().cpu().numpy()
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
        return np.ascontiguousarray(rgb[..., :3])

    def run_config(height: float, mirror: bool):
        seed = args.seed
        reset_to_stand()
        planner_context = init_frame.unsqueeze(1).expand(1, 4, -1).clone()
        planner_cache = torch.zeros(1, RESAMPLED_FRAMES, PLANNER_FRAME_DIM, device=device)
        playback_idx = torch.zeros(1, dtype=torch.long, device=device)
        frames_rgb = []
        planner_ref_legs = planner_ref_pelvis_z = planner_ref_quat = None

        for t in range(args.crouch_steps):
            joint_pos_il = robot.data.joint_pos[:, body_idx_t].clone()
            joint_vel_il = robot.data.joint_vel[:, body_idx_t].clone()
            root_quat_w = robot.data.root_quat_w.clone()
            base_ang_vel_b = robot.data.root_ang_vel_b.clone()
            gravity_b = robot.data.projected_gravity_b.clone()

            if t % PLANNER_EVERY_K_POLICY_STEPS == 0:
                if t > 0:
                    planner_context = _context_from_cache(planner_cache, playback_idx)
                feeds = _build_planner_feeds(
                    planner_context.cpu().numpy(), height, movement_dir, facing_dir, seed)
                traj_np, _ = planner_pool.run_batched(feeds)
                traj_t = torch.as_tensor(traj_np, device=device, dtype=torch.float32)
                planner_cache[:] = resample_traj_30_to_50hz(traj_t, RESAMPLED_FRAMES)
                playback_idx.zero_()

            idx = (playback_idx.view(-1, 1) + future_offsets.view(1, -1)).clamp(max=RESAMPLED_FRAMES - 1)
            idx_next = (idx + 1).clamp(max=RESAMPLED_FRAMES - 1)
            n_range = torch.zeros(1, G1_NUM_FUTURE_FRAMES, dtype=torch.long, device=device)
            frames = planner_cache[n_range, idx]
            frames_next = planner_cache[n_range, idx_next]

            joints_mj_future = frames[..., 7:]
            leg_pos_mj = joints_mj_future[..., leg_mj_slots]
            leg_vel_mj = (frames_next[..., 7:][..., leg_mj_slots] - leg_pos_mj) * POLICY_HZ
            if mirror:
                leg_pos_mj = leg_mirror_sign_t * leg_pos_mj[..., leg_mirror_perm_t]
                leg_vel_mj = leg_mirror_sign_t * leg_vel_mj[..., leg_mirror_perm_t]

            joint_pos_future = torch.zeros(1, G1_NUM_FUTURE_FRAMES, NUM_JOINTS,
                                           dtype=torch.float32, device=device)
            joint_vel_future = torch.zeros_like(joint_pos_future)
            joint_pos_future[..., leg_idx_il_t] = leg_pos_mj
            joint_vel_future[..., leg_idx_il_t] = leg_vel_mj
            joint_pos_future[..., upper_idx_il_t] = upper_rest.unsqueeze(1).expand(
                -1, G1_NUM_FUTURE_FRAMES, -1)

            ref_root_quat_future_wxyz = frames[..., 3:7]
            if mirror:
                q = ref_root_quat_future_wxyz
                ref_root_quat_future_wxyz = torch.stack(
                    [q[..., 0], -q[..., 1], q[..., 2], -q[..., 3]], dim=-1)

            target_body_il = infer.step(
                joint_pos_future=joint_pos_future, joint_vel_future=joint_vel_future,
                ref_root_quat_future_wxyz=ref_root_quat_future_wxyz,
                joint_pos=joint_pos_il, joint_vel=joint_vel_il,
                base_ang_vel=base_ang_vel_b, gravity_in_base=gravity_b,
                root_quat_wxyz=root_quat_w)

            target_full = default_jp_full_t.clone()
            target_full[:, body_idx_t] = target_body_il

            for _ in range(DECIMATION):
                robot.set_joint_position_target(target_full)
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(dt=SIM_DT)

            planner_ref_legs = leg_pos_mj[0, 0].cpu().numpy().copy()
            planner_ref_pelvis_z = float(frames[0, 0, 2])
            planner_ref_quat = ref_root_quat_future_wxyz[0, 0].cpu().numpy().copy()
            playback_idx = (playback_idx + 1).clamp(max=RESAMPLED_FRAMES - 1)

            if not args.no_video and (t % args.render_every == 0):
                frames_rgb.append(grab_frame())

        body29 = robot.data.joint_pos[0, body_idx_t].cpu().numpy()
        settled_legs = body29[leg_idx_il]
        root_pos_w = (robot.data.root_pos_w[0] - env_origins[0]).cpu().numpy()
        settled_quat = robot.data.root_quat_w[0].cpu().numpy()
        side = detect_grounded_side()
        fallen = bool(root_pos_w[2] < 0.1)

        # Per-height settled still (robot has settled at end of the crouch hold).
        still = grab_frame()
        img = image_dir / f"kneel_{side}_h{height:.2f}.png"
        imageio.imwrite(str(img), still)

        entry = {
            "requested_height_m": round(float(height), 4),
            "seed": int(seed),
            "grounded_knee": side,
            "fallen": fallen,
            "image": str(img.relative_to(out_dir)),
            "planner_reference": {
                "pelvis_z_m": round(planner_ref_pelvis_z, 5),
                "root_quat_wxyz": [round(float(x), 6) for x in planner_ref_quat],
                "leg_joint_pos_rad": [round(float(x), 6) for x in planner_ref_legs],
            },
            "settled_achieved": {
                "pelvis_z_m": round(float(root_pos_w[2]), 5),
                "root_quat_wxyz": [round(float(x), 6) for x in settled_quat],
                "leg_joint_pos_rad": [round(float(x), 6) for x in settled_legs],
            },
        }
        if not args.no_video and frames_rgb:
            vid = video_dir / f"kneel_{side}_h{height:.2f}.mp4"
            imageio.mimsave(str(vid), frames_rgb, fps=args.fps, macro_block_size=None)
            entry["video"] = str(vid.relative_to(out_dir))
        return entry, side

    # --- sweep -----------------------------------------------------------
    sides = ["right", "left"] if args.side == "both" else [args.side]
    results = {"left": {}, "right": {}}
    for want_side in sides:
        mirror = (want_side == "left")
        for height in args.heights:
            t0 = time.time()
            entry, det_side = run_config(height, mirror)
            dt = time.time() - t0
            if entry["fallen"]:
                tag = "FALLEN"
            elif det_side != want_side:
                tag = f"WRONG-SIDE(got {det_side})"
            else:
                tag = "ok"
            print(f"[run] side={want_side} h={height:.2f} -> grounded={det_side} "
                  f"settled_z={entry['settled_achieved']['pelvis_z_m']} "
                  f"[{tag}] ({dt:.1f}s)", flush=True)
            results[want_side][f"{height:.2f}"] = entry

    for side in sides:
        refs = {f"{h:.2f}": results[side][f"{h:.2f}"]
                for h in args.heights if f"{h:.2f}" in results[side]}
        doc = {
            "mode": "kneelOneLeg", "mode_id": PLANNER_MODE_KNEEL_ONE_LEG,
            "grounded_knee": side, "dof_order": "isaaclab",
            "leg_joint_names": leg_joint_names,
            "notes": ("Upper body held at default rest (no Pink IK). "
                      "planner_reference = planner kinematic kneel target; "
                      "settled_achieved = closed-loop SONIC tracked pose. "
                      "Left knee = sagittal mirror of the planner's right-only kneel."),
            "heights_m": [round(float(h), 4) for h in args.heights],
            "references": refs,
        }
        path = out_dir / f"kneel_{side}.json"
        path.write_text(json.dumps(doc, indent=2))
        print(f"[out] {path}  ({len(refs)} heights, dict keyed by height)")

    planner_pool.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
