"""Stage hybrid PLANNER-ONLY on a fixed-base G1.

Strips the SONIC encoder/decoder from `stage_hybrid_eval.py` — just drives
Isaac Lab's PD directly with the hybrid-assembled 29-DoF IL target (lower 12
from the kinematic planner MJ→IL, upper 17 from Pink IK at frozen pelvis-frame
rest poses, padded). Fixed-base so a bad gait can't knock the robot over.

Purpose: isolate whether the joint-order / splice / IK assembly is correct,
BEFORE adding the encoder/decoder on top.

Usage:
    uv run --active python -m sonic_python_inference.scripts.stage_hybrid_planner_only \\
        --num-envs 1 --headless --episode-sec 5
"""

from __future__ import annotations

import argparse
import itertools
import re

import numpy as np


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--episode-sec", type=float, default=10.0)
    ap.add_argument("--spawn-height", type=float, default=1.2)
    ap.add_argument("--target-vel", type=float, default=0.3)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument(
        "--planner-onnx",
        type=str,
        default="gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx",
    )
    ap.add_argument(
        "--urdf-path",
        type=str,
        default="sonic_python_inference/assets/g1_pink_ik.urdf",
    )
    return ap.parse_args()


args = _parse_args()

# ---------------------------------------------------------------------------
# AppLauncher must come before any other isaaclab import.
# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.terrains import TerrainImporterCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from gear_sonic.envs.manager_env.robots.g1 import (  # noqa: E402
    G1_CYLINDER_MODEL_12_DEX_CFG,
    G1_ISAACLAB_TO_MUJOCO_DOF,
    G1_MUJOCO_TO_ISAACLAB_DOF,
)

from sonic_python_inference.sonic_inference import (  # noqa: E402
    ALLOWED_PRED_NUM_TOKENS,
    NUM_JOINTS,
    PLANNER_CONTEXT_DEFAULT_HEIGHT,
    PLANNER_EVERY_K_POLICY_STEPS,
    PLANNER_HEIGHT_DEFAULT,
    PLANNER_MODE_SLOW_WALK,
    POLICY_HZ,
    RESAMPLED_FRAMES,
    resample_traj_30_to_50hz,
    slerp_torch,
)
from sonic_python_inference.sonic_planner_pool import (  # noqa: E402
    PLANNER_FRAME_DIM,
    PlannerSessionPool,
)
from sonic_python_inference.pink_ik_driver import PinkIKDriver  # noqa: E402
from sonic_python_inference.g1_pink_ik_cfg import (  # noqa: E402
    LEFT_WRIST_REST_POSE_PELVIS,
    PINK_CONTROLLED_JOINTS_IL,
    RIGHT_WRIST_REST_POSE_PELVIS,
)


SIM_DT = 0.005  # 200 Hz
DECIMATION = 4  # policy @ 50 Hz
LEG_NAME_PATTERNS = (
    r".*_hip_.*_joint",
    r".*_knee_joint",
    r".*_ankle_.*_joint",
)


def _fixed_base_robot_cfg(spawn_height: float):
    cfg = G1_CYLINDER_MODEL_12_DEX_CFG.copy()
    cfg.spawn = cfg.spawn.replace(fix_base=True)
    cfg.init_state = cfg.init_state.replace(pos=(0.0, 0.0, spawn_height))
    return cfg


@configclass
class HybridPlannerOnlySceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
    )
    robot = _fixed_base_robot_cfg(1.2).replace(prim_path="{ENV_REGEX_NS}/Robot")
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(1.0, 1.0, 1.0), intensity=2000.0),
    )


def _leg_indices_il(joint_names: list[str]) -> list[int]:
    leg_re = [re.compile(p) for p in LEG_NAME_PATTERNS]
    return [i for i, n in enumerate(joint_names) if any(r.fullmatch(n) for r in leg_re)]


def _build_planner_feeds(
    ctx_np: np.ndarray,
    mode: int,
    movement_dir: np.ndarray,
    facing_dir: np.ndarray,
    target_vel: float,
    height: float,
    num_envs: int,
) -> list[dict[str, np.ndarray]]:
    feeds: list[dict[str, np.ndarray]] = []
    allowed_mask = ALLOWED_PRED_NUM_TOKENS.reshape(1, 11)
    for i in range(num_envs):
        feeds.append({
            "context_mujoco_qpos": ctx_np[i : i + 1].astype(np.float32),
            "target_vel": np.array([target_vel], dtype=np.float32),
            "mode": np.array([mode], dtype=np.int64),
            "movement_direction": movement_dir.reshape(1, 3).astype(np.float32),
            "facing_direction": facing_dir.reshape(1, 3).astype(np.float32),
            "random_seed": np.array([0], dtype=np.int64),
            "has_specific_target": np.zeros((1, 1), dtype=np.int64),
            "specific_target_positions": np.zeros((1, 4, 3), dtype=np.float32),
            "specific_target_headings": np.zeros((1, 4), dtype=np.float32),
            "allowed_pred_num_tokens": allowed_mask,
            "height": np.array([height], dtype=np.float32),
        })
    return feeds


def _context_from_cache(
    cache: torch.Tensor, playback_idx: torch.Tensor
) -> torch.Tensor:
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
    device = "cuda"

    sim_cfg = sim_utils.SimulationCfg(device=device, dt=SIM_DT)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.0, 3.0, 2.0], [0.0, 0.0, args.spawn_height])

    scene_cfg = HybridPlannerOnlySceneCfg(num_envs=args.num_envs, env_spacing=3.0)
    scene_cfg.robot = _fixed_base_robot_cfg(args.spawn_height).replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    robot = scene["robot"]
    scene.update(dt=0.0)

    il_joint_names = list(robot.data.joint_names)
    print(f"[info] joints (IL order): {il_joint_names}")

    default_jp_il = robot.data.default_joint_pos[0].cpu().numpy().astype(np.float32)

    # Leg IL indices + upper IL indices
    leg_idx_il = _leg_indices_il(il_joint_names)
    assert len(leg_idx_il) == 12, f"expected 12 leg joints, got {len(leg_idx_il)}"
    upper_idx_il = [il_joint_names.index(n) for n in PINK_CONTROLLED_JOINTS_IL]
    assert set(leg_idx_il) | set(upper_idx_il) == set(range(NUM_JOINTS))
    leg_idx_il_t = torch.as_tensor(leg_idx_il, dtype=torch.long, device=device)
    upper_idx_il_t = torch.as_tensor(upper_idx_il, dtype=torch.long, device=device)

    il_to_mj = torch.as_tensor(
        G1_ISAACLAB_TO_MUJOCO_DOF, dtype=torch.long, device=device
    )
    mj_to_il = torch.as_tensor(
        G1_MUJOCO_TO_ISAACLAB_DOF, dtype=torch.long, device=device
    )
    leg_mj_slots = mj_to_il[leg_idx_il_t]  # [12]

    # --- Planner --------------------------------------------------------
    planner_pool = PlannerSessionPool(
        args.planner_onnx, pool_size=args.num_envs, device_id=0, serial=False
    )
    default_jp_il_t = torch.as_tensor(default_jp_il, device=device).unsqueeze(0).expand(
        args.num_envs, -1
    )
    default_jp_mj_t = default_jp_il_t[:, il_to_mj]  # IL→MJ gather
    init_frame = torch.zeros(args.num_envs, PLANNER_FRAME_DIM, device=device)
    init_frame[:, 2] = PLANNER_CONTEXT_DEFAULT_HEIGHT
    init_frame[:, 3] = 1.0
    init_frame[:, 7:] = default_jp_mj_t
    planner_context = init_frame.unsqueeze(1).expand(-1, 4, -1).clone()
    planner_cache = torch.zeros(
        args.num_envs, RESAMPLED_FRAMES, PLANNER_FRAME_DIM, device=device
    )
    playback_idx = torch.zeros(args.num_envs, dtype=torch.long, device=device)

    movement_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    facing_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # --- Pink IK -------------------------------------------------------
    pink_driver = PinkIKDriver(
        num_envs=args.num_envs,
        robot_cfg=G1_CYLINDER_MODEL_12_DEX_CFG,
        urdf_path=args.urdf_path,
        all_joint_names_il=il_joint_names,
        device=device,
        dt=1.0 / POLICY_HZ,
    )
    left_target_pelvis = torch.tensor(
        LEFT_WRIST_REST_POSE_PELVIS, dtype=torch.float32, device=device
    ).unsqueeze(0).expand(args.num_envs, -1).contiguous()
    right_target_pelvis = torch.tensor(
        RIGHT_WRIST_REST_POSE_PELVIS, dtype=torch.float32, device=device
    ).unsqueeze(0).expand(args.num_envs, -1).contiguous()

    # --- Seed robot at default ----------------------------------------
    joint_pos_init = robot.data.default_joint_pos.clone()
    joint_vel_init = torch.zeros_like(joint_pos_init)
    robot.write_joint_state_to_sim(joint_pos_init, joint_vel_init)
    robot.set_joint_position_target(joint_pos_init)
    scene.write_data_to_sim()
    scene.update(dt=0.0)

    # --- Main loop -----------------------------------------------------
    print(
        f"[info] hybrid-planner-only: fixed_base, mode=SLOW_WALK, vel={args.target_vel}"
    )

    for t in itertools.count():
        if not simulation_app.is_running():
            break

        joint_pos_il = robot.data.joint_pos.clone()

        if t % PLANNER_EVERY_K_POLICY_STEPS == 0:
            if t > 0:
                planner_context = _context_from_cache(planner_cache, playback_idx)
            feeds = _build_planner_feeds(
                planner_context.cpu().numpy(),
                mode=PLANNER_MODE_SLOW_WALK,
                movement_dir=movement_dir,
                facing_dir=facing_dir,
                target_vel=args.target_vel,
                height=PLANNER_HEIGHT_DEFAULT,
                num_envs=args.num_envs,
            )
            traj_np, _ = planner_pool.run_batched(feeds)
            traj_t = torch.as_tensor(traj_np, device=device, dtype=torch.float32)
            planner_cache[:] = resample_traj_30_to_50hz(traj_t, RESAMPLED_FRAMES)
            playback_idx.zero_()

        # Current planner frame → legs in MJ, gather IL-order leg values
        n_range = torch.arange(args.num_envs, device=device)
        idx = playback_idx.clamp(max=RESAMPLED_FRAMES - 1)
        frame = planner_cache[n_range, idx]  # [N, 36]
        joints_mj_now = frame[..., 7:]       # [N, 29] MJ
        leg_pos_il_vals = joints_mj_now[..., leg_mj_slots]  # [N, 12] IL leg order

        # Pink IK for upper
        upper_pos_il = pink_driver.solve(
            curr_joint_pos_il=joint_pos_il,
            left_target_pelvis=left_target_pelvis,
            right_target_pelvis=right_target_pelvis,
        )  # [N, 17]

        # Splice into [N, 29] IL — default as a sanity fallback (should be
        # overwritten by all 29 scatters below)
        target_il = default_jp_il_t.clone()
        target_il[..., leg_idx_il_t] = leg_pos_il_vals
        target_il[..., upper_idx_il_t] = upper_pos_il

        for _ in range(DECIMATION):
            robot.set_joint_position_target(target_il)
            scene.write_data_to_sim()
            sim.step()
            scene.update(dt=SIM_DT)

        playback_idx = (playback_idx + 1).clamp(max=RESAMPLED_FRAMES - 1)

        if t % 50 == 0:
            err = (robot.data.joint_pos - target_il).abs().mean(dim=-1)
            print(
                f"[t={t / 50:5.2f}s] tracking_mae={err.cpu().tolist()}  "
                f"leg_il_target[0,:6]={target_il[0, leg_idx_il_t[:6]].cpu().tolist()}"
            )

    planner_pool.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
