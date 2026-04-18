"""Stage hybrid kneel: hybrid planner+PinkIK → G1 encoder, planner in
`kneelOneLeg` mode, with two VisualCuboid wrist targets that auto-track the
pelvis as the robot drops.

Pipeline identical to `stage_hybrid_eval.py`:
    planner (lower 12) + Pink IK (upper 17) → G1 encoder → decoder → PD

What differs:
- Planner `mode = 6` (kneelOneLeg), `target_vel = 0`, `height` from CLI
  (default 0.2m — valid range 0.2–0.4m per `docs/source/references/planner_onnx.md`).
  Left vs right is chosen internally by the planner — nudge `--random-seed`
  to flip sides.
- Two cubes (red = right hand target, blue = left hand target) spawn at the
  world position of the standing rest pose. Each tick the script **rewrites
  their world Z** to `pelvis_z + rest_z_pelvis_scaled`, where the pelvis-frame
  rest Z is scaled proportionally with `target_height / PLANNER_CONTEXT_DEFAULT_HEIGHT`.
  As the robot kneels down, the cubes drop with the pelvis — user sees the
  Pink IK targets tracking the lowered body instead of floating above the head.
- X/Y of each cube are still user-draggable (we only overwrite Z).

Usage:
    uv run --active python -m sonic_python_inference.scripts.stage_hybrid_kneel \\
        --num-envs 1 --kneel-height 0.2
"""

from __future__ import annotations

import argparse
import re

import numpy as np


# Mode indices from docs/source/references/planner_onnx.md — not exported from
# sonic_inference.py so we inline them here.
PLANNER_MODE_KNEEL_TWO_LEG = 5
PLANNER_MODE_KNEEL_ONE_LEG = 6
KNEEL_MODE_TABLE = {
    "one_leg": PLANNER_MODE_KNEEL_ONE_LEG,
    "two_leg": PLANNER_MODE_KNEEL_TWO_LEG,
}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--episode-sec", type=float, default=15.0)
    ap.add_argument("--headless", action="store_true")
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
    ap.add_argument(
        "--kneel-mode", type=str, default="one_leg",
        choices=list(KNEEL_MODE_TABLE.keys()),
        help="one_leg = mode 6 (kneelOneLeg); two_leg = mode 5 (kneelTwoLeg).",
    )
    ap.add_argument(
        "--kneel-height", type=float, default=0.2,
        help="Target pelvis height. Valid range 0.2–0.4m for both kneel modes.",
    )
    ap.add_argument(
        "--random-seed", type=int, default=0,
        help="Planner random seed — flip to swap which knee goes down (one_leg only).",
    )
    return ap.parse_args()


args = _parse_args()

from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.terrains import TerrainImporterCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from isaacsim.core.api.objects import VisualCuboid  # noqa: E402

from gear_sonic.envs.manager_env.robots.g1 import (  # noqa: E402
    G1_CYLINDER_MODEL_12_DEX_CFG,
    G1_ISAACLAB_TO_MUJOCO_DOF,
    G1_MODEL_12_ACTION_SCALE,
    G1_MUJOCO_TO_ISAACLAB_DOF,
)

from sonic_python_inference.sonic_inference import (  # noqa: E402
    ALLOWED_PRED_NUM_TOKENS,
    NUM_JOINTS,
    PLANNER_CONTEXT_DEFAULT_HEIGHT,
    PLANNER_EVERY_K_POLICY_STEPS,
    POLICY_HZ,
    RESAMPLED_FRAMES,
    quat_conjugate,
    quat_mul,
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
from sonic_python_inference.pink_ik_driver import PinkIKDriver  # noqa: E402
from sonic_python_inference.g1_pink_ik_cfg import (  # noqa: E402
    LEFT_WRIST_REST_POSE_PELVIS,
    PINK_CONTROLLED_JOINTS_IL,
    RIGHT_WRIST_REST_POSE_PELVIS,
)


SIM_DT = 0.005
DECIMATION = 4
LEG_NAME_PATTERNS = (
    r".*_hip_.*_joint",
    r".*_knee_joint",
    r".*_ankle_.*_joint",
)


@configclass
class HybridKneelSceneCfg(InteractiveSceneCfg):
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
    scale = np.ones(NUM_JOINTS, dtype=np.float32)
    patterns = [(re.compile(p), v) for p, v in G1_MODEL_12_ACTION_SCALE.items()]
    for i, name in enumerate(joint_names):
        for pat, v in patterns:
            if pat.fullmatch(name):
                scale[i] = float(v)
                break
        else:
            raise RuntimeError(f"No scale pattern matches joint '{name}'.")
    return scale


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
    random_seed: int = 0,
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
            "random_seed": np.array([random_seed], dtype=np.int64),
            "has_specific_target": np.zeros((1, 1), dtype=np.int64),
            "specific_target_positions": np.zeros((1, 4, 3), dtype=np.float32),
            "specific_target_headings": np.zeros((1, 4), dtype=np.float32),
            "allowed_pred_num_tokens": allowed_mask,
            "height": np.array([height], dtype=np.float32),
        })
    return feeds


def _context_from_cache(cache: torch.Tensor, playback_idx: torch.Tensor) -> torch.Tensor:
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


def _lowered_rest_pose(
    base_rest: tuple[float, ...], target_height: float,
) -> tuple[float, ...]:
    """Scale the pelvis-frame Z of the standing rest pose proportionally
    with the planner's target pelvis height. X/Y and the identity quat stay.

    Example: standing rest Z = 0.14523 (hand ~15 cm above pelvis at 0.789
    standing height). At kneel height 0.3 the scale becomes 0.38 → adjusted
    rest Z = 0.055 (hand just above the lowered pelvis). Prevents the
    standing rest target from floating way above the head once the pelvis
    drops during a kneel/squat.
    """
    if target_height <= 0.0:
        return base_rest
    scale = target_height / PLANNER_CONTEXT_DEFAULT_HEIGHT
    x, y, z = base_rest[0], base_rest[1], base_rest[2] * scale
    return (x, y, z, base_rest[3], base_rest[4], base_rest[5], base_rest[6])


def main():
    device = "cpu"

    sim_cfg = sim_utils.SimulationCfg(device=device, dt=SIM_DT)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.0, 3.0, 2.0], [0.0, 0.0, 0.75])

    scene_cfg = HybridKneelSceneCfg(num_envs=args.num_envs, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    robot = scene["robot"]
    scene.update(dt=0.0)

    il_joint_names = list(robot.data.joint_names)
    default_joint_pos_il = robot.data.default_joint_pos[0].cpu().numpy().astype(np.float32)
    action_scale_il = _action_scale_il(il_joint_names)

    leg_idx_il = _leg_indices_il(il_joint_names)
    assert len(leg_idx_il) == 12
    upper_idx_il = [il_joint_names.index(n) for n in PINK_CONTROLLED_JOINTS_IL]
    assert set(leg_idx_il) | set(upper_idx_il) == set(range(NUM_JOINTS))
    leg_idx_il_t = torch.as_tensor(leg_idx_il, dtype=torch.long, device=device)
    upper_idx_il_t = torch.as_tensor(upper_idx_il, dtype=torch.long, device=device)

    il_to_mj = torch.as_tensor(G1_ISAACLAB_TO_MUJOCO_DOF, dtype=torch.long, device=device)
    mj_to_il = torch.as_tensor(G1_MUJOCO_TO_ISAACLAB_DOF, dtype=torch.long, device=device)
    leg_mj_slots = mj_to_il[leg_idx_il_t]

    # --- Planner infra ---------------------------------------------------
    planner_pool = PlannerSessionPool(
        args.planner_onnx, pool_size=args.num_envs, device_id=0, serial=False
    )
    planner_context = torch.zeros(args.num_envs, 4, PLANNER_FRAME_DIM, device=device)
    planner_cache = torch.zeros(
        args.num_envs, RESAMPLED_FRAMES, PLANNER_FRAME_DIM, device=device
    )
    playback_idx = torch.zeros(args.num_envs, dtype=torch.long, device=device)

    default_jp_il_t = torch.as_tensor(default_joint_pos_il, device=device).unsqueeze(0).expand(
        args.num_envs, -1
    )
    default_jp_mj_t = default_jp_il_t[:, il_to_mj]
    init_frame = torch.zeros(args.num_envs, PLANNER_FRAME_DIM, device=device)
    init_frame[:, 2] = PLANNER_CONTEXT_DEFAULT_HEIGHT
    init_frame[:, 3] = 1.0
    init_frame[:, 7:] = default_jp_mj_t
    planner_context[:] = init_frame.unsqueeze(1).expand(-1, 4, -1).clone()

    movement_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    facing_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # --- G1 encoder + Pink IK --------------------------------------------
    infer = SonicG1Inference(
        num_envs=args.num_envs,
        g1_encoder_onnx=args.g1_encoder_onnx,
        decoder_onnx=args.decoder_onnx,
        default_angles=default_joint_pos_il,
        action_scale=action_scale_il,
        device=device,
    )
    pink_driver = PinkIKDriver(
        num_envs=args.num_envs,
        robot_cfg=G1_CYLINDER_MODEL_12_DEX_CFG,
        urdf_path=args.urdf_path,
        all_joint_names_il=il_joint_names,
        device=device,
        dt=1.0 / POLICY_HZ,
    )

    # Seed sim at default stand.
    root_state = robot.data.default_root_state.clone()
    root_state[:, 0:3] += scene.env_origins
    root_state[:, 7:13] = 0.0
    joint_pos_init = robot.data.default_joint_pos.clone()
    joint_vel_init = torch.zeros_like(joint_pos_init)
    robot.write_root_state_to_sim(root_state)
    robot.write_joint_state_to_sim(joint_pos_init, joint_vel_init)
    robot.set_joint_position_target(joint_pos_init)
    scene.write_data_to_sim()
    scene.update(dt=0.0)

    infer.reset(joint_pos=joint_pos_init)

    # Rest Z scaled proportionally with target pelvis height.
    left_rest = _lowered_rest_pose(LEFT_WRIST_REST_POSE_PELVIS, args.kneel_height)
    right_rest = _lowered_rest_pose(RIGHT_WRIST_REST_POSE_PELVIS, args.kneel_height)
    rest_z_scaled = left_rest[2]  # same scale for both sides
    kneel_mode_id = KNEEL_MODE_TABLE[args.kneel_mode]
    print(
        f"[info] kneel mode={kneel_mode_id} ({args.kneel_mode}), "
        f"target_vel=0, height={args.kneel_height}m, seed={args.random_seed}. "
        f"Rest Z scaled = {rest_z_scaled:.4f} "
        f"(scale={args.kneel_height / PLANNER_CONTEXT_DEFAULT_HEIGHT:.3f})."
    )

    # Spawn two VisualCuboids at the standing rest world position; script
    # rewrites their Z each tick to track the pelvis as it drops.
    root_pos_w_init = (
        robot.data.root_state_w[:, 0:3] - scene.env_origins
    ).cpu().numpy()
    env_origins_np = scene.env_origins.cpu().numpy()
    right_cubes: list[VisualCuboid] = []
    left_cubes: list[VisualCuboid] = []
    for i in range(args.num_envs):
        origin = env_origins_np[i]
        # Spawn at standing world pose (pelvis_z_default + rest_z_standing).
        right_world_init = (
            float(root_pos_w_init[i, 0]) + RIGHT_WRIST_REST_POSE_PELVIS[0] + origin[0],
            float(root_pos_w_init[i, 1]) + RIGHT_WRIST_REST_POSE_PELVIS[1] + origin[1],
            float(root_pos_w_init[i, 2]) + RIGHT_WRIST_REST_POSE_PELVIS[2] + origin[2],
        )
        left_world_init = (
            float(root_pos_w_init[i, 0]) + LEFT_WRIST_REST_POSE_PELVIS[0] + origin[0],
            float(root_pos_w_init[i, 1]) + LEFT_WRIST_REST_POSE_PELVIS[1] + origin[1],
            float(root_pos_w_init[i, 2]) + LEFT_WRIST_REST_POSE_PELVIS[2] + origin[2],
        )
        right_cubes.append(
            VisualCuboid(
                prim_path=f"/World/envs/env_{i}/RightTargetCube",
                name=f"right_target_cube_{i}",
                position=np.array(right_world_init, dtype=np.float32),
                size=0.06,
                color=np.array([0.9, 0.1, 0.1], dtype=np.float32),
            )
        )
        left_cubes.append(
            VisualCuboid(
                prim_path=f"/World/envs/env_{i}/LeftTargetCube",
                name=f"left_target_cube_{i}",
                position=np.array(left_world_init, dtype=np.float32),
                size=0.06,
                color=np.array([0.1, 0.3, 0.9], dtype=np.float32),
            )
        )

    def _quat_rotate_inv(q_wxyz: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q_inv = quat_conjugate(q_wxyz)
        w = q_inv[..., 0:1]
        xyz = q_inv[..., 1:4]
        t = 2.0 * torch.cross(xyz, v, dim=-1)
        return v + w * t + torch.cross(xyz, t, dim=-1)

    # --- Main loop -------------------------------------------------------
    num_policy_steps = int(args.episode_sec * POLICY_HZ)
    future_offsets = torch.arange(
        0, G1_NUM_FUTURE_FRAMES * 5, 5, dtype=torch.long, device=device
    )

    for t in range(num_policy_steps):
        if not simulation_app.is_running():
            break

        joint_pos_il = robot.data.joint_pos.clone()
        joint_vel_il = robot.data.joint_vel.clone()
        root_pos_w = (robot.data.root_state_w[:, 0:3] - scene.env_origins).clone()
        root_quat_w = robot.data.root_state_w[:, 3:7].clone()
        base_ang_vel_b = robot.data.root_ang_vel_b.clone()
        gravity_b = robot.data.projected_gravity_b.clone()

        if t % PLANNER_EVERY_K_POLICY_STEPS == 0:
            if t > 0:
                planner_context = _context_from_cache(planner_cache, playback_idx)
            feeds = _build_planner_feeds(
                planner_context.cpu().numpy(),
                mode=kneel_mode_id,
                movement_dir=movement_dir,
                facing_dir=facing_dir,
                target_vel=0.0,
                height=args.kneel_height,
                num_envs=args.num_envs,
                random_seed=args.random_seed,
            )
            traj_np, _ = planner_pool.run_batched(feeds)
            traj_t = torch.as_tensor(traj_np, device=device, dtype=torch.float32)
            planner_cache[:] = resample_traj_30_to_50hz(traj_t, RESAMPLED_FRAMES)
            playback_idx.zero_()

        idx = (playback_idx.view(-1, 1) + future_offsets.view(1, -1)).clamp(
            max=RESAMPLED_FRAMES - 1
        )
        idx_next = (idx + 1).clamp(max=RESAMPLED_FRAMES - 1)
        n_range = torch.arange(args.num_envs, device=device).view(-1, 1).expand(
            -1, G1_NUM_FUTURE_FRAMES
        )
        frames = planner_cache[n_range, idx]
        frames_next = planner_cache[n_range, idx_next]

        joints_mj_future = frames[..., 7:]
        joints_mj_future_next = frames_next[..., 7:]
        leg_pos_mj = joints_mj_future[..., leg_mj_slots]
        leg_vel_mj = (joints_mj_future_next[..., leg_mj_slots] - leg_pos_mj) * POLICY_HZ

        # --- Auto-update cube Z to track pelvis; user keeps X/Y dragging --
        # cube_world_z := pelvis_z + rest_z_scaled. Written back each tick so
        # the cube is always anchored to the pelvis's current height.
        for i in range(args.num_envs):
            origin = env_origins_np[i]
            pelvis_z_w = float(root_pos_w[i, 2].item()) + origin[2]
            target_z = pelvis_z_w + rest_z_scaled
            rp, rq = right_cubes[i].get_world_pose()
            lp, lq = left_cubes[i].get_world_pose()
            rp = np.asarray(rp, dtype=np.float32); rp[2] = target_z
            lp = np.asarray(lp, dtype=np.float32); lp[2] = target_z
            right_cubes[i].set_world_pose(position=rp, orientation=np.asarray(rq, dtype=np.float32))
            left_cubes[i].set_world_pose(position=lp, orientation=np.asarray(lq, dtype=np.float32))

        # Read cube world poses (after Z rewrite), transform to pelvis frame.
        right_pos_list, right_quat_list = [], []
        left_pos_list, left_quat_list = [], []
        for i in range(args.num_envs):
            rp, rq = right_cubes[i].get_world_pose()
            lp, lq = left_cubes[i].get_world_pose()
            right_pos_list.append(np.asarray(rp, dtype=np.float32))
            right_quat_list.append(np.asarray(rq, dtype=np.float32))
            left_pos_list.append(np.asarray(lp, dtype=np.float32))
            left_quat_list.append(np.asarray(lq, dtype=np.float32))
        right_pos_w = torch.as_tensor(
            np.stack(right_pos_list), device=device
        ) - scene.env_origins
        right_quat_w = torch.as_tensor(np.stack(right_quat_list), device=device)
        left_pos_w = torch.as_tensor(
            np.stack(left_pos_list), device=device
        ) - scene.env_origins
        left_quat_w = torch.as_tensor(np.stack(left_quat_list), device=device)

        right_pos_p = _quat_rotate_inv(root_quat_w, right_pos_w - root_pos_w)
        right_quat_p = quat_mul(quat_conjugate(root_quat_w), right_quat_w)
        left_pos_p = _quat_rotate_inv(root_quat_w, left_pos_w - root_pos_w)
        left_quat_p = quat_mul(quat_conjugate(root_quat_w), left_quat_w)
        right_target_pelvis = torch.cat([right_pos_p, right_quat_p], dim=-1)
        left_target_pelvis = torch.cat([left_pos_p, left_quat_p], dim=-1)

        upper_pos_il = pink_driver.solve(
            curr_joint_pos_il=joint_pos_il,
            left_target_pelvis=left_target_pelvis,
            right_target_pelvis=right_target_pelvis,
        )

        joint_pos_future = torch.zeros(
            args.num_envs, G1_NUM_FUTURE_FRAMES, NUM_JOINTS,
            dtype=torch.float32, device=device,
        )
        joint_vel_future = torch.zeros_like(joint_pos_future)
        joint_pos_future[..., leg_idx_il_t] = leg_pos_mj
        joint_vel_future[..., leg_idx_il_t] = leg_vel_mj
        joint_pos_future[..., upper_idx_il_t] = upper_pos_il.unsqueeze(1).expand(
            -1, G1_NUM_FUTURE_FRAMES, -1
        )

        ref_root_quat_future_wxyz = frames[..., 3:7]

        target_il = infer.step(
            joint_pos_future=joint_pos_future,
            joint_vel_future=joint_vel_future,
            ref_root_quat_future_wxyz=ref_root_quat_future_wxyz,
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

        playback_idx = (playback_idx + 1).clamp(max=RESAMPLED_FRAMES - 1)

        if t % 50 == 0:
            z = (robot.data.root_pos_w[:, 2] - scene.env_origins[:, 2]).cpu().tolist()
            ref_z = frames[:, 0, 2].cpu().tolist()
            print(f"[t={t / POLICY_HZ:5.2f}s] z={z}  planner_ref_z={ref_z}")

    planner_pool.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
