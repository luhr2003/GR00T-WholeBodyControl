"""Export G1 one-leg kneel references at different pelvis heights, KINEMATICALLY.

Direct kinematic-planner path — **no SONIC tracking** (no encoder/decoder/PD).
The planner's `kneelOneLeg` produces a single fixed END pose (height-invariant),
but its kneel-DOWN trajectory descends through every pelvis height ~0.79->0.43.
So we sample that descent at target heights and pose the robot DIRECTLY at the
planner qpos (write_root_state + write_joint_state), then render.

Left knee = sagittal mirror of the planner's right-only kneel (validated).
One-leg kneel bottoms out at ~0.43 m (cannot reach 0.2-0.4 on one knee).

Outputs (under --out-dir):
    kneel_kin_left.json / kneel_kin_right.json   (dict keyed by height)
    images/kneel_kin_<side>_h<H>.png

Usage:
    OMNI_KIT_ACCEPT_EULA=YES .venv_isaac/bin/python -m \
        sonic_python_inference.scripts.export_kneel_kinematic \
        --heights 0.70 0.60 0.50 0.45 0.43 --side both --out-dir kneel_kinematic_out
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heights", type=float, nargs="+",
                    default=[0.70, 0.60, 0.50, 0.45, 0.43],
                    help="Target pelvis heights to sample from the kneel descent.")
    ap.add_argument("--side", choices=["both", "right", "left"], default="both")
    ap.add_argument("--planner-height", type=float, default=0.2,
                    help="Height input to the planner to trigger a clean kneel descent.")
    ap.add_argument("--out-dir", type=str, default="kneel_kinematic_out")
    ap.add_argument("--magicsim-usd-path", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--cam-width", type=int, default=640)
    ap.add_argument("--cam-height", type=int, default=480)
    ap.add_argument("--headless", action="store_true", default=True)
    ap.add_argument("--gui", dest="headless", action="store_false")
    ap.add_argument("--planner-onnx", type=str,
                    default="gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx")
    return ap.parse_args()


args = _parse_args()

from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher(headless=args.headless, enable_cameras=True)
simulation_app = app_launcher.app

import torch  # noqa: E402
import imageio.v2 as imageio  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.terrains import TerrainImporterCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from gear_sonic.envs.manager_env.robots.g1 import (  # noqa: E402
    G1_ISAACLAB_TO_MUJOCO_DOF,
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
)
from sonic_python_inference.sonic_planner_pool import PlannerSessionPool  # noqa: E402
from sonic_python_inference.joint_order import g1_body_indices_in_training_order  # noqa: E402


HAND_RE = re.compile(r".*_hand_.*_joint")
LEG_NAME_PATTERNS = (r".*_hip_.*_joint", r".*_knee_joint", r".*_ankle_.*_joint")
PLANNER_MODE_KNEEL_ONE_LEG = 6

_USD = args.magicsim_usd_path or DEFAULT_MAGICSIM_USD_PATH
_ROBOT_CFG = make_g1_magicsim_cfg(_USD)


@configclass
class SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", collision_group=-1)
    robot = _ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    dome_light = AssetBaseCfg(prim_path="/World/DomeLight",
                              spawn=sim_utils.DomeLightCfg(color=(1.0, 1.0, 1.0), intensity=2000.0))
    camera = CameraCfg(prim_path="/World/RefCam", update_period=0.0,
                       height=args.cam_height, width=args.cam_width, data_types=["rgb"],
                       spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 100.0)))


def _leg_indices(names):
    leg_re = [re.compile(p) for p in LEG_NAME_PATTERNS]
    return [i for i, n in enumerate(names) if any(r.fullmatch(n) for r in leg_re)]


def main():
    device = args.device
    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=device, dt=0.005))
    sim.set_camera_view([3.0, 3.0, 2.0], [0.0, 0.0, 0.6])
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=3.0))
    sim.reset()
    robot = scene["robot"]
    camera = scene["camera"]
    scene.update(dt=0.0)
    camera.set_world_poses_from_view(eyes=torch.tensor([[2.4, 2.0, 1.0]], device=camera.device),
                                     targets=torch.tensor([[0.0, 0.0, 0.4]], device=camera.device))

    full_names = list(robot.data.joint_names)
    body_idx, body_names = g1_body_indices_in_training_order(full_names, HAND_RE)
    assert len(body_idx) == NUM_JOINTS
    body_idx_t = torch.as_tensor(body_idx, dtype=torch.long, device=device)
    leg_idx = _leg_indices(body_names)
    assert len(leg_idx) == 12
    leg_names = [body_names[i] for i in leg_idx]

    default_jp_full = robot.data.default_joint_pos[0].cpu().numpy().astype(np.float32)
    default_jp_full_t = robot.data.default_joint_pos.clone()
    default_body_il = default_jp_full[body_idx]

    il_to_mj = np.asarray(G1_ISAACLAB_TO_MUJOCO_DOF)
    mj_to_il = np.asarray(G1_MUJOCO_TO_ISAACLAB_DOF)

    # full-29 sagittal mirror map (swap L<->R, negate roll/yaw)
    def partner(n):
        if n.startswith("left"):
            return "right" + n[4:]
        if n.startswith("right"):
            return "left" + n[5:]
        return n
    mir_perm = np.array([body_names.index(partner(n)) for n in body_names])
    mir_sign = np.array([-1.0 if ("_roll" in n or "_yaw" in n) else 1.0 for n in body_names], np.float32)

    # --- one planner call: the kneelOneLeg descent (MJ qpos) -------------
    pool = PlannerSessionPool(args.planner_onnx, pool_size=1, device_id=0, serial=True)
    ctx = np.zeros((1, 4, 36), np.float32)
    ctx[:, :, 2] = PLANNER_CONTEXT_DEFAULT_HEIGHT
    ctx[:, :, 3] = 1.0
    ctx[:, :, 7:] = default_body_il[il_to_mj]  # standing joints in MJ order
    feed = [{
        "context_mujoco_qpos": ctx, "target_vel": np.array([0.], np.float32),
        "mode": np.array([PLANNER_MODE_KNEEL_ONE_LEG], np.int64),
        "movement_direction": np.array([[1., 0, 0]], np.float32),
        "facing_direction": np.array([[1., 0, 0]], np.float32),
        "random_seed": np.array([0], np.int64), "has_specific_target": np.zeros((1, 1), np.int64),
        "specific_target_positions": np.zeros((1, 4, 3), np.float32),
        "specific_target_headings": np.zeros((1, 4), np.float32),
        "allowed_pred_num_tokens": ALLOWED_PRED_NUM_TOKENS.reshape(1, 11),
        "height": np.array([args.planner_height], np.float32),
    }]
    traj, npred = pool.run_batched(feed)        # traj [1, N, 36]
    traj = traj[0]                              # [N, 36]
    zs = traj[:, 2]
    # The trajectory continues past the kneel bottom (pelvis rises again), so only
    # the frames up to the lowest pelvis are the actual descent — sample within those.
    kneel_frame = int(np.argmin(zs))
    z_desc = zs[:kneel_frame + 1]
    print(f"[info] kneel descent: {traj.shape[0]} frames, kneel bottom @frame {kneel_frame} "
          f"(z={zs[kneel_frame]:.3f}); descent z {z_desc.min():.3f}..{z_desc.max():.3f}")

    def pose_and_capture(root_pos, root_quat, joints_il, side, H):
        full_jp = default_jp_full_t.clone()
        full_jp[0, body_idx_t] = torch.as_tensor(joints_il, dtype=torch.float32, device=device)
        root_state = robot.data.default_root_state.clone()
        root_state[0, 0:3] = torch.as_tensor(root_pos, dtype=torch.float32, device=device)
        root_state[0, 3:7] = torch.as_tensor(root_quat, dtype=torch.float32, device=device)
        root_state[0, 7:13] = 0.0
        robot.write_root_state_to_sim(root_state)
        robot.write_joint_state_to_sim(full_jp, torch.zeros_like(full_jp))
        robot.set_joint_position_target(full_jp)
        scene.write_data_to_sim()
        scene.update(dt=0.0)
        for _ in range(3):
            sim.render()
        camera.update(dt=0.0, force_recompute=True)
        rgb = camera.data.output["rgb"][0].detach().cpu().numpy()
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0., 1.) * 255).astype(np.uint8)
        p = img_dir / f"kneel_kin_{side}_h{H:.2f}.png"
        imageio.imwrite(str(p), np.ascontiguousarray(rgb[..., :3]))
        return str(p.relative_to(out_dir))

    sides = ["right", "left"] if args.side == "both" else [args.side]
    for side in sides:
        refs = {}
        for H in args.heights:
            i = int(np.argmin(np.abs(z_desc - H)))  # descent frames only
            root_pos = traj[i, 0:3].copy()
            root_quat = traj[i, 3:7].copy()          # wxyz
            joints_il = traj[i, 7:][mj_to_il].copy()  # 29 IL
            if side == "left":
                joints_il = mir_sign * joints_il[mir_perm]
                root_pos[1] = -root_pos[1]
                root_quat = np.array([root_quat[0], -root_quat[1], root_quat[2], -root_quat[3]], np.float32)
            img = pose_and_capture(root_pos, root_quat, joints_il, side, H)
            legs = joints_il[leg_idx]
            refs[f"{H:.2f}"] = {
                "target_height_m": round(float(H), 4),
                "matched_pelvis_z_m": round(float(zs[i]), 5),
                "descent_frame": i,
                "root_quat_wxyz": [round(float(x), 6) for x in root_quat],
                "leg_joint_pos_rad": [round(float(x), 6) for x in legs],
                "image": img,
            }
            print(f"[{side}] H={H:.2f} -> frame {i} z={zs[i]:.3f}  {img}")
        doc = {
            "mode": "kneelOneLeg", "mode_id": PLANNER_MODE_KNEEL_ONE_LEG,
            "grounded_knee": side, "source": "kinematic_planner_descent (no tracking)",
            "dof_order": "isaaclab", "leg_joint_names": leg_names,
            "note": ("Leg refs sampled from the planner's kneel-DOWN trajectory at target "
                     "pelvis heights (one-leg kneel endpoint is fixed ~0.43m; heights >0.43 are "
                     "descent/transition poses). Robot posed directly at planner qpos, NO tracking. "
                     + ("Left = sagittal mirror." if side == "left" else "Right = native planner.")),
            "references": refs,
        }
        (out_dir / f"kneel_kin_{side}.json").write_text(json.dumps(doc, indent=2))
        print(f"[out] {out_dir}/kneel_kin_{side}.json ({len(refs)} heights)")

    pool.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
