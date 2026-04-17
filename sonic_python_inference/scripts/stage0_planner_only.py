"""Stage 0 (planner-only): test the SONIC kinematic planner on a fixed-base G1.

No encoder, no decoder. Just:
    planner (10 Hz, batch=1) → mujoco_qpos [1, 64, 36] @ 30 Hz
    → resample to 50 Hz → slice current-frame joints (MJ order)
    → MJ→IL reorder → robot.set_joint_position_target → Isaac Lab PD steps

The planner is fed context in its own *yaw-normalized* frame, never live world
state. Mirrors the C++ deploy (localmotion_kplanner.hpp:InitializeContext /
UpdateContextFromMotion):
    - Init:  4 identical context frames = [0, 0, default_height, 1,0,0,0, joints_mj]
    - Replan: slice 4 frames from the planner's OWN 50 Hz cache at
              gen_time + n/30 s; joints stay MJ (planner output is MJ).

If the pipeline's planner wiring is correct, the legs should show a smooth
walking gait. If the legs twitch or lock, the bug is in the planner feeds
(context, mode, height, etc.) — encoder/decoder are eliminated from the loop.

Usage:
    uv run --active python -m sonic_python_inference.scripts.stage0_planner_only
"""

from __future__ import annotations

import argparse


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--episode-sec", type=float, default=20.0)
    ap.add_argument("--spawn-height", type=float, default=1.2)
    ap.add_argument("--target-vel", type=float, default=0.3)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument(
        "--planner-onnx",
        type=str,
        default="gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx",
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
    G1_ISAACLAB_JOINTS,
    G1_ISAACLAB_TO_MUJOCO_DOF,
    G1_MUJOCO_TO_ISAACLAB_DOF,
)

from sonic_python_inference.sonic_inference import (  # noqa: E402
    ALLOWED_PRED_NUM_TOKENS,
    PLANNER_HEIGHT_DEFAULT,
    PLANNER_MODE_SLOW_WALK,
    RESAMPLED_FRAMES,
    resample_traj_30_to_50hz,
    slerp_torch,
)
from sonic_python_inference.sonic_planner_pool import (  # noqa: E402
    PLANNER_FRAME_DIM,
    PlannerSessionPool,
)


SIM_DT = 0.005  # 200 Hz
DECIMATION = 4  # policy @ 50 Hz
PLANNER_EVERY = 5  # replan every 5 policy steps = 10 Hz
PLANNER_DEFAULT_HEIGHT = 0.788740  # matches deploy config.default_height
NUM_JOINTS = 29

# Gather indices (verified against policy_parameters.hpp:
#   lower_body_joint_mujoco_order_in_isaaclab_index = {0,3,6,9,13,17,...}):
#   G1_ISAACLAB_TO_MUJOCO_DOF is MJ-ordered and holds IL indices.
#     → mj = il[G1_ISAACLAB_TO_MUJOCO_DOF]  (IL→MJ gather)
#   G1_MUJOCO_TO_ISAACLAB_DOF is its argsort-inverse, IL-ordered, holds MJ
#   indices.
#     → il = mj[G1_MUJOCO_TO_ISAACLAB_DOF]  (MJ→IL gather)
_IL_TO_MJ = np.asarray(G1_ISAACLAB_TO_MUJOCO_DOF, dtype=np.int64)  # IL→MJ gather
_MJ_TO_IL = np.asarray(G1_MUJOCO_TO_ISAACLAB_DOF, dtype=np.int64)  # MJ→IL gather
assert np.array_equal(_MJ_TO_IL, np.argsort(_IL_TO_MJ)), "MJ/IL DOF arrays must be argsort inverses"


def _fixed_base_robot_cfg(spawn_height: float):
    cfg = G1_CYLINDER_MODEL_12_DEX_CFG.copy()
    cfg.spawn = cfg.spawn.replace(fix_base=True)
    cfg.init_state = cfg.init_state.replace(pos=(0.0, 0.0, spawn_height))
    return cfg


@configclass
class Stage0PlannerOnlySceneCfg(InteractiveSceneCfg):
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


def _build_planner_feeds(
    context_np: np.ndarray,  # [N, 4, 36] MJ order
    mode: int,
    target_vel: float,
    height: float,
    movement_dir_np: np.ndarray,  # [N, 3]
    facing_dir_np: np.ndarray,  # [N, 3]
) -> list[dict[str, np.ndarray]]:
    """Build per-env batch=1 feed dicts matching planner_sonic.onnx shapes."""
    N = context_np.shape[0]
    allowed_mask = ALLOWED_PRED_NUM_TOKENS.reshape(1, 11)
    feeds = []
    for i in range(N):
        feeds.append(
            {
                "context_mujoco_qpos": context_np[i : i + 1],  # [1, 4, 36]
                "target_vel": np.array([target_vel], dtype=np.float32),
                "mode": np.array([mode], dtype=np.int64),
                "movement_direction": movement_dir_np[i : i + 1],  # [1, 3]
                "facing_direction": facing_dir_np[i : i + 1],  # [1, 3]
                "random_seed": np.array([0], dtype=np.int64),
                "has_specific_target": np.zeros((1, 1), dtype=np.int64),
                "specific_target_positions": np.zeros((1, 4, 3), dtype=np.float32),
                "specific_target_headings": np.zeros((1, 4), dtype=np.float32),
                "allowed_pred_num_tokens": allowed_mask,
                "height": np.array([height], dtype=np.float32),
            }
        )
    return feeds


def _context_from_cache(
    cache_50hz: torch.Tensor,  # [N, L, 36] planner output (MJ order)
    playback_idx: torch.Tensor,  # [N] current playback pos in 50 Hz cache
) -> torch.Tensor:
    """Build 4-frame planner context from the planner's own 50 Hz cache.

    Mirrors UpdateContextFromMotion (localmotion_kplanner.hpp:628-678):
        gen_time = playback_idx / 50
        for n in 0..3:  sample at t = gen_time + n / 30

    Planner output joints are already MJ-ordered; the deploy reorders MJ→IL when
    storing in `planner_motion_50hz_`, then IL→MJ when feeding the context —
    net identity, so we can keep the cache in MJ and slice directly.
    """
    N, L, _ = cache_50hz.shape
    device = cache_50hz.device
    # In 50 Hz cache indices, step = 50/30 per 30 Hz tick.
    t_offsets = torch.arange(4, device=device, dtype=torch.float32) * (50.0 / 30.0)
    idx_f = playback_idx.view(-1, 1).float() + t_offsets.view(1, -1)  # [N, 4]
    idx0 = idx_f.long().clamp(0, L - 1)
    idx1 = (idx0 + 1).clamp(0, L - 1)
    alpha = (idx_f - idx0.float()).clamp(0.0, 1.0)

    n_range = torch.arange(N, device=device).view(-1, 1).expand(-1, 4)
    f0 = cache_50hz[n_range, idx0]  # [N, 4, 36]
    f1 = cache_50hz[n_range, idx1]  # [N, 4, 36]
    pos = f0[..., 0:3] + (f1[..., 0:3] - f0[..., 0:3]) * alpha.unsqueeze(-1)
    quat = slerp_torch(f0[..., 3:7], f1[..., 3:7], alpha)
    joints = f0[..., 7:] + (f1[..., 7:] - f0[..., 7:]) * alpha.unsqueeze(-1)
    return torch.cat([pos, quat, joints], dim=-1)  # [N, 4, 36]


def main():
    device = "cuda"

    sim_cfg = sim_utils.SimulationCfg(device=device, dt=SIM_DT)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.0, 3.0, 2.0], [0.0, 0.0, args.spawn_height])

    scene_cfg = Stage0PlannerOnlySceneCfg(num_envs=args.num_envs, env_spacing=3.0)
    scene_cfg.robot = _fixed_base_robot_cfg(args.spawn_height).replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    robot = scene["robot"]
    scene.update(dt=0.0)

    il_joint_names = list(robot.data.joint_names)
    print(f"[info] runtime IL joints ({len(il_joint_names)}): {il_joint_names}")
    # G1_ISAACLAB_JOINTS stores child-link names; joint names end in "_joint".
    # Strip both suffixes for order comparison.
    def _stem(n: str) -> str:
        for suf in ("_joint", "_link"):
            if n.endswith(suf):
                return n[: -len(suf)]
        return n
    got_stems = [_stem(n) for n in il_joint_names]
    want_stems = [_stem(n) for n in G1_ISAACLAB_JOINTS[1:]]  # strip pelvis root
    if got_stems != want_stems:
        print("[warn] runtime IL joint order != gear_sonic.G1_ISAACLAB_JOINTS[1:]")
        for i, (got, want) in enumerate(zip(got_stems, want_stems)):
            if got != want:
                print(f"  il[{i:2d}]: got={got}  want={want}")

    default_jp_il = robot.data.default_joint_pos[0].cpu().numpy().astype(np.float32)
    default_jp_mj = default_jp_il[_IL_TO_MJ]  # MJ order for planner context init

    mj_to_il_t = torch.as_tensor(_MJ_TO_IL, dtype=torch.long, device=device)
    N = args.num_envs

    planner_pool = PlannerSessionPool(
        args.planner_onnx, pool_size=N, device_id=0, serial=False
    )

    # --- Yaw-normalized initial context (deploy InitializeContext) ---------
    init_frame = np.zeros((PLANNER_FRAME_DIM,), dtype=np.float32)
    init_frame[2] = PLANNER_DEFAULT_HEIGHT  # z
    init_frame[3] = 1.0  # quat w (identity)
    init_frame[7:] = default_jp_mj
    context_np = np.broadcast_to(init_frame, (N, 4, PLANNER_FRAME_DIM)).copy()

    # --- Fixed per-env commands for the whole run -----------------------
    mode = PLANNER_MODE_SLOW_WALK  # id=1, 0.1–0.8 m/s range
    target_vel = args.target_vel
    height = PLANNER_HEIGHT_DEFAULT
    movement_np = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (N, 1))
    facing_np = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (N, 1))

    # --- Initial planner inference --------------------------------------
    feeds = _build_planner_feeds(
        context_np, mode, target_vel, height, movement_np, facing_np
    )
    traj_30hz, num_pred = planner_pool.run_batched(feeds)  # [N, 64, 36], [N]
    print(f"[info] init planner: num_pred_frames={num_pred.tolist()}")
    traj_50hz = resample_traj_30_to_50hz(
        torch.as_tensor(traj_30hz, device=device, dtype=torch.float32),
        RESAMPLED_FRAMES,
    )
    playback_idx = torch.zeros(N, device=device, dtype=torch.long)

    num_policy_steps = int(args.episode_sec * 50)
    print(
        f"[info] planner-only probe: mode=WALK, move=[1,0,0], facing=[1,0,0], "
        f"target_vel={target_vel}, spawn_z={args.spawn_height:.2f}"
    )

    for t in range(num_policy_steps):
        # Replan at 10 Hz using planner's own cache as context feedback
        if t > 0 and t % PLANNER_EVERY == 0:
            new_ctx = _context_from_cache(traj_50hz, playback_idx)
            context_np = new_ctx.cpu().numpy().astype(np.float32)
            feeds = _build_planner_feeds(
                context_np, mode, target_vel, height, movement_np, facing_np
            )
            traj_30hz, _ = planner_pool.run_batched(feeds)
            traj_50hz = resample_traj_30_to_50hz(
                torch.as_tensor(traj_30hz, device=device, dtype=torch.float32),
                RESAMPLED_FRAMES,
            )
            playback_idx.zero_()

        # Pull current frame's joints (MJ) from the cache, convert to IL
        idx = playback_idx.clamp(0, RESAMPLED_FRAMES - 1)
        n_range = torch.arange(N, device=device)
        frame = traj_50hz[n_range, idx]  # [N, 36]
        joints_mj = frame[:, 7:]  # [N, 29] MJ
        joints_il = joints_mj[:, mj_to_il_t]  # [N, 29] IL

        # Directly drive Isaac Lab PD with the planner's joint target
        robot.set_joint_position_target(joints_il)
        scene.write_data_to_sim()
        for _ in range(DECIMATION):
            sim.step()
            scene.update(dt=SIM_DT)

        playback_idx = (playback_idx + 1).clamp(max=RESAMPLED_FRAMES - 1)

        if t % 50 == 0:
            print(
                f"[t={t / 50:5.2f}s] planner_joints_il[0, :6]="
                f"{joints_il[0, :6].cpu().tolist()}"
            )

    planner_pool.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
