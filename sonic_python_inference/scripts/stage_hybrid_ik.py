"""Stage hybrid IK: stage_hybrid_eval with cube-driven Pink IK wrist targets.

Identical to `stage_hybrid_eval.py` — full SONIC G1 teacher closed-loop:
kinematic planner → lower 12, Pink IK → upper 17, encoder + decoder → motor
targets, free-floating robot tracked by Isaac Lab PD — with ONE change:

    frozen LEFT/RIGHT_WRIST_REST_POSE_PELVIS → live cube world poses
    transformed into pelvis_contour_link frame each tick.

Scene:
    Red cube  → right_hand_palm_link target.
    Blue cube → left_hand_palm_link  target.
    Cubes spawn at the pelvis-frame rest poses (world = pelvis at init); they
    have gravity and collisions disabled so they stay put until moved.

Usage:
    uv run --active python -m sonic_python_inference.scripts.stage_hybrid_ik \\
        --num-envs 1 --headless
"""

from __future__ import annotations

import argparse
import itertools
import re

import numpy as np


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--episode-sec", type=float, default=30.0)
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
    ap.add_argument("--target-vel", type=float, default=0.0)
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
    PLANNER_HEIGHT_DEFAULT,
    PLANNER_MODE_IDLE,
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

# World-frame spawn for the cubes: pelvis-frame rest + default robot z.
# The robot spawns at z ≈ 0.793 with identity pelvis quat, so world = pelvis.
_ROBOT_DEFAULT_Z = 0.793
_RIGHT_CUBE_POS = (
    RIGHT_WRIST_REST_POSE_PELVIS[0],
    RIGHT_WRIST_REST_POSE_PELVIS[1],
    RIGHT_WRIST_REST_POSE_PELVIS[2] + _ROBOT_DEFAULT_Z,
)
_LEFT_CUBE_POS = (
    LEFT_WRIST_REST_POSE_PELVIS[0],
    LEFT_WRIST_REST_POSE_PELVIS[1],
    LEFT_WRIST_REST_POSE_PELVIS[2] + _ROBOT_DEFAULT_Z,
)


@configclass
class HybridIKSceneCfg(InteractiveSceneCfg):
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


def _quat_rotate_inv(q_wxyz: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate v [N,3] by q^-1 [N,4] (wxyz). Expresses a world-frame vector in
    the frame whose world orientation is q (here: the pelvis frame)."""
    q_inv = quat_conjugate(q_wxyz)
    w = q_inv[..., 0:1]
    xyz = q_inv[..., 1:4]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


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


def main():
    device = "cpu"

    sim_cfg = sim_utils.SimulationCfg(device=device, dt=SIM_DT)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.0, 3.0, 2.0], [0.0, 0.0, 0.75])

    scene_cfg = HybridIKSceneCfg(num_envs=args.num_envs, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    robot = scene["robot"]
    scene.update(dt=0.0)

    # Spawn two VisualCuboids per env directly via isaacsim API (no IsaacLab
    # asset wrapping). VisualCuboid has no collider/rigid body, so it just
    # sits where set_world_pose places it — move in the viewport to drive IK.
    env_origins_np = scene.env_origins.cpu().numpy()
    right_cubes: list[VisualCuboid] = []
    left_cubes: list[VisualCuboid] = []
    for i in range(args.num_envs):
        origin = env_origins_np[i]
        right_cubes.append(
            VisualCuboid(
                prim_path=f"/World/envs/env_{i}/RightTargetCube",
                name=f"right_target_cube_{i}",
                position=np.array(_RIGHT_CUBE_POS, dtype=np.float32) + origin,
                size=0.06,
                color=np.array([0.9, 0.1, 0.1], dtype=np.float32),
            )
        )
        left_cubes.append(
            VisualCuboid(
                prim_path=f"/World/envs/env_{i}/LeftTargetCube",
                name=f"left_target_cube_{i}",
                position=np.array(_LEFT_CUBE_POS, dtype=np.float32) + origin,
                size=0.06,
                color=np.array([0.1, 0.3, 0.9], dtype=np.float32),
            )
        )

    il_joint_names = list(robot.data.joint_names)
    print(f"[info] joints (IL order): {il_joint_names}")

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

    # --- Planner ---------------------------------------------------------
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

    # --- G1 encoder + Pink IK -------------------------------------------
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

    # --- Seed sim at default --------------------------------------------
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

    # --- Main loop -------------------------------------------------------
    future_offsets = torch.arange(
        0, G1_NUM_FUTURE_FRAMES * 5, 5, dtype=torch.long, device=device
    )

    print(
        f"[info] cubes spawned at pelvis-frame rest → "
        f"right={_RIGHT_CUBE_POS}, left={_LEFT_CUBE_POS}. Move them to redirect hands."
    )

    for t in itertools.count():
        if not simulation_app.is_running():
            break

        joint_pos_il = robot.data.joint_pos.clone()
        joint_vel_il = robot.data.joint_vel.clone()
        root_pos_w = robot.data.root_pos_w.clone()
        root_quat_w = robot.data.root_quat_w.clone()
        base_ang_vel_b = robot.data.root_ang_vel_b.clone()
        gravity_b = robot.data.projected_gravity_b.clone()

        # 1. Planner every 5th tick
        if t % PLANNER_EVERY_K_POLICY_STEPS == 0:
            if t > 0:
                planner_context = _context_from_cache(planner_cache, playback_idx)
            feeds = _build_planner_feeds(
                planner_context.cpu().numpy(),
                mode=PLANNER_MODE_IDLE,
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

        # 2. Sample future frames
        idx = (playback_idx.view(-1, 1) + future_offsets.view(1, -1)).clamp(
            max=RESAMPLED_FRAMES - 1
        )
        idx_next = (idx + 1).clamp(max=RESAMPLED_FRAMES - 1)
        n_range = torch.arange(args.num_envs, device=device).view(-1, 1).expand(
            -1, G1_NUM_FUTURE_FRAMES
        )
        frames = planner_cache[n_range, idx]
        frames_next = planner_cache[n_range, idx_next]

        # 3. Lower-body future from planner
        joints_mj_future = frames[..., 7:]
        joints_mj_future_next = frames_next[..., 7:]
        leg_pos_mj = joints_mj_future[..., leg_mj_slots]
        leg_vel_mj = (joints_mj_future_next[..., leg_mj_slots] - leg_pos_mj) * POLICY_HZ

        # 4. Pink IK wrist targets — LIVE from cubes, world → pelvis frame
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
        # robot root_pos_w also includes env_origins; subtract for consistency
        robot_root_pos_local = root_pos_w - scene.env_origins

        right_pos_p = _quat_rotate_inv(root_quat_w, right_pos_w - robot_root_pos_local)
        right_quat_p = quat_mul(quat_conjugate(root_quat_w), right_quat_w)
        left_pos_p = _quat_rotate_inv(root_quat_w, left_pos_w - robot_root_pos_local)
        left_quat_p = quat_mul(quat_conjugate(root_quat_w), left_quat_w)

        right_target_pelvis = torch.cat([right_pos_p, right_quat_p], dim=-1)
        left_target_pelvis = torch.cat([left_pos_p, left_quat_p], dim=-1)

        upper_pos_il = pink_driver.solve(
            curr_joint_pos_il=joint_pos_il,
            left_target_pelvis=left_target_pelvis,
            right_target_pelvis=right_target_pelvis,
        )

        # 5. Splice [N, 10, 29] IL
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

        # 6. ref_root_quat_future (world-frame wxyz from planner cache)
        ref_root_quat_future_wxyz = frames[..., 3:7]

        # 7. Encoder/Decoder → motor targets
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
            print(
                f"[t={t / POLICY_HZ:5.2f}s] z={z}  "
                f"right_pelvis={right_pos_p[0].cpu().tolist()}  "
                f"left_pelvis={left_pos_p[0].cpu().tolist()}"
            )

    planner_pool.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
