"""Stage G1 eval: closed-loop SONIC G1-mode (teacher) tracking in Isaac Lab.

Mirrors `stage_smpl_eval.py` but swaps SMPL encoder → G1 (teacher) encoder.
Only needs `robot_filtered/*.pkl` — the retargeted robot trajectory drives the
encoder directly (no SMPL pkl, no kinematic planner). Serves as the cleanest
sanity baseline: if the g1 path can't track, decoder/proprio is broken; if it
tracks but SMPL doesn't, the issue is in SMPL obs construction.

Obs semantics follow the **training code** (gear_sonic Isaac Lab obs
functions). The g1 encoder sees 10 future frames @ dt=0.1 s (frame_skip=5 at
TARGET_FPS=50) of [joint_pos(29) ‖ joint_vel(29) ‖ anchor_ori_6d(6)].

Runs for exactly one pass of the motion (`max_step - 1` policy ticks); prints
a summary at the end. Ctrl+C or close the viewer to stop early.

Usage:
    python -m sonic_python_inference.scripts.stage_g1_eval \
        --motion walk_forward_amateur_001__A001 --num-envs 4

Flags:
    --motion        motion basename (robot pkl only — no SMPL pkl needed)
    --num-envs      parallel env count
    --headless      run without viewer
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", type=str, default="walk_forward_amateur_001__A001")
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument(
        "--robot-dir", type=str, default="sample_data/robot_filtered"
    )
    ap.add_argument(
        "--g1-encoder-onnx",
        type=str,
        default="sonic_python_inference/assets/g1_encoder_dyn.onnx",
    )
    ap.add_argument(
        "--decoder-onnx",
        type=str,
        default="sonic_python_inference/assets/decoder_dyn.onnx",
    )
    return ap.parse_args()


args = _parse_args()

# ---------------------------------------------------------------------------
# AppLauncher must come before any other isaaclab import.
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
    G1_MODEL_12_ACTION_SCALE,
)

from sonic_python_inference.sonic_inference import NUM_JOINTS  # noqa: E402
from sonic_python_inference.sonic_g1_inference import SonicG1Inference  # noqa: E402
from sonic_python_inference.sonic_smpl_motion_lib import SmplMotionLib  # noqa: E402


SIM_DT = 0.005  # 200 Hz
DECIMATION = 4  # policy @ 50 Hz
G1_DT_FUTURE_REF_FRAMES = 0.1  # matches config.yaml dt_future_ref_frames


@configclass
class RobotEvalSceneCfg(InteractiveSceneCfg):
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


def _action_scale_il(joint_names: list[str]) -> np.ndarray:
    """Resolve per-joint action scales. G1_MODEL_12_ACTION_SCALE keys are regex
    patterns (same format as IsaacLab's JointPositionActionCfg.scale): pick the
    first regex key that fully matches the joint name."""
    scale = np.ones(NUM_JOINTS, dtype=np.float32)
    patterns = [(re.compile(p), v) for p, v in G1_MODEL_12_ACTION_SCALE.items()]
    for i, name in enumerate(joint_names):
        for pat, v in patterns:
            if pat.fullmatch(name):
                scale[i] = float(v)
                break
        else:
            raise RuntimeError(
                f"No G1_MODEL_12_ACTION_SCALE pattern matches joint '{name}'."
            )
    return scale


def main():
    device = "cuda"

    sim_cfg = sim_utils.SimulationCfg(device=device, dt=SIM_DT)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.0, 3.0, 2.0], [0.0, 0.0, 0.75])

    scene_cfg = RobotEvalSceneCfg(num_envs=args.num_envs, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    robot = scene["robot"]
    scene.update(dt=0.0)

    il_joint_names = list(robot.data.joint_names)
    print(f"[info] joints (IL order): {il_joint_names}")

    default_joint_pos_il = robot.data.default_joint_pos[0].cpu().numpy().astype(np.float32)
    action_scale_il = _action_scale_il(il_joint_names)
    print(f"[info] action_scale_il (min/max/mean): "
          f"{action_scale_il.min():.4f} / {action_scale_il.max():.4f} / "
          f"{action_scale_il.mean():.4f}")

    # --- Motion lib: robot-only, dt=0.1s future window ---------------------
    motion_names = [args.motion] * args.num_envs
    motion_lib = SmplMotionLib(
        smpl_dir=None,
        robot_dir=args.robot_dir,
        motion_names=motion_names,
        device=device,
        dt_future_ref_frames=G1_DT_FUTURE_REF_FRAMES,
        need_smpl=False,
    )
    max_step = motion_lib.max_step
    print(f"[info] motion '{args.motion}' frames={max_step} @ {motion_lib.target_fps} Hz "
          f"(frame_skip={motion_lib.frame_skip})")

    # --- Policy ------------------------------------------------------------
    infer = SonicG1Inference(
        num_envs=args.num_envs,
        g1_encoder_onnx=args.g1_encoder_onnx,
        decoder_onnx=args.decoder_onnx,
        default_angles=default_joint_pos_il,
        action_scale=action_scale_il,
        device=device,
    )

    # --- Seed the sim from the retargeted motion's frame 0 -----------------
    init = motion_lib.get_initial_state()
    root_state = robot.data.default_root_state.clone()
    root_state[:, 0:3] = init["root_pos_w"] + scene.env_origins
    root_state[:, 3:7] = init["root_quat_w_wxyz"]
    root_state[:, 7:13] = 0.0

    joint_pos_init = init["dof_pos_il"].clone()
    joint_vel_init = torch.zeros_like(joint_pos_init)

    robot.write_root_state_to_sim(root_state)
    robot.write_joint_state_to_sim(joint_pos_init, joint_vel_init)
    robot.set_joint_position_target(joint_pos_init)
    scene.write_data_to_sim()
    scene.update(dt=0.0)

    infer.reset(joint_pos=joint_pos_init)

    # --- Loop (one pass of the motion, clamped to max_step - 1) -----------
    num_policy_steps = max(max_step - 1, 1)
    mpjpe_accum = torch.zeros(args.num_envs, device=device)
    step_counter = 0

    for t in range(num_policy_steps):
        if not simulation_app.is_running():
            break

        joint_pos_il = robot.data.joint_pos.clone()
        joint_vel_il = robot.data.joint_vel.clone()
        root_state = robot.data.root_state_w.clone()
        root_quat_w = root_state[:, 3:7]
        base_ang_vel_b = robot.data.root_ang_vel_b.clone()
        gravity_b = robot.data.projected_gravity_b.clone()

        time_steps = torch.full(
            (args.num_envs,), t, dtype=torch.long, device=device
        )
        fut = motion_lib.sample_future_robot(time_steps)

        target_il = infer.step(
            joint_pos_future=fut["joint_pos_future"],
            joint_vel_future=fut["joint_vel_future"],
            ref_root_quat_future_wxyz=fut["robot_root_quat_future_wxyz"],
            joint_pos=joint_pos_il,
            joint_vel=joint_vel_il,
            base_ang_vel=base_ang_vel_b,
            gravity_in_base=gravity_b,
            root_quat_wxyz=root_quat_w,
        )

        for _ in range(DECIMATION):
            robot.set_joint_position_target(target_il)
            scene.write_data_to_sim()
            sim.step()
            scene.update(dt=SIM_DT)

        mpjpe = (robot.data.joint_pos - fut["dof_pos_ref"]).abs().mean(dim=-1)
        mpjpe_accum += mpjpe
        step_counter += 1

        if t % 50 == 0:
            z = robot.data.root_pos_w[:, 2] - scene.env_origins[:, 2]
            # Per-joint diff for env 0: lower body only (first 12 of IL order,
            # but ordered LHP,RHP,waist_yaw,LHR,RHR,... so pick explicit hip+knee+ankle)
            il_names = il_joint_names
            diff = (robot.data.joint_pos - fut["dof_pos_ref"])[0].cpu().numpy()
            legs = ["left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint","left_knee_joint","left_ankle_pitch_joint","left_ankle_roll_joint","right_hip_pitch_joint","right_hip_roll_joint","right_hip_yaw_joint","right_knee_joint","right_ankle_pitch_joint","right_ankle_roll_joint"]
            leg_str = " ".join(f"{n.replace('_joint','').replace('_','')[:8]}={diff[il_names.index(n)]:+.3f}" for n in legs)
            print(
                f"[t={t / 50:5.2f}s] z={z.cpu().tolist()}  "
                f"joint_mae={mpjpe.cpu().tolist()}  "
                f"fallen={(z < 0.4).cpu().tolist()}"
            )
            print(f"  lower-body err (rob-ref): {leg_str}")

    mean_mpjpe = (mpjpe_accum / max(step_counter, 1)).cpu().tolist()
    final_z = (robot.data.root_pos_w[:, 2] - scene.env_origins[:, 2]).cpu().tolist()
    fallen = [z < 0.4 for z in final_z]
    print("\n=== Robot (G1-teacher) eval summary ===")
    for i in range(args.num_envs):
        print(
            f"env {i}: mean_joint_mae={mean_mpjpe[i]:.4f} rad  "
            f"final_z={final_z[i]:.3f}m  fallen={fallen[i]}"
        )

    simulation_app.close()


if __name__ == "__main__":
    main()
