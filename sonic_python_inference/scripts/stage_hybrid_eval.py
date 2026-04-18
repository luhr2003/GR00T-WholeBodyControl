"""Stage hybrid eval: closed-loop SONIC G1-teacher with a hybrid future.

Combines two upstream signals into the 29-DoF future trajectory the G1
teacher encoder expects:

  lower 12 (legs)        ←  kinematic planner (MJ→IL)
  upper 17 (waist+arms)  ←  Pink IK (IL), padded across 10 future frames,
                             upper joint_vel = 0
  ref_root_quat_future   ←  planner cache root quat at [0,5,...,45]

Obs semantics follow the training code via SonicG1Inference.step(). Pink IK
wrist targets are expressed in **pelvis_contour_link (torso) frame** — v1
freezes them at the pelvis-frame rest poses copied from MagicSim.

This is a sanity experiment, not a production path. We expect OOD behavior
because the G1 encoder trained on smooth retargeted robot pkls, not
planner+IK hybrids.

Usage:
    uv run --active python -m sonic_python_inference.scripts.stage_hybrid_eval \
        --num-envs 4 --headless
"""

from __future__ import annotations

import argparse
import math
import re

import numpy as np


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--episode-sec", type=float, default=10.0)
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
    ap.add_argument("--target-vel", type=float, default=0.3)
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
    G1_MODEL_12_ACTION_SCALE,
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


SIM_DT = 0.005  # 200 Hz
DECIMATION = 4  # policy @ 50 Hz
LEG_NAME_PATTERNS = (
    r".*_hip_.*_joint",
    r".*_knee_joint",
    r".*_ankle_.*_joint",
)


@configclass
class HybridEvalSceneCfg(InteractiveSceneCfg):
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
    ctx_np: np.ndarray,        # [N, 4, 36] float32
    mode: int,
    movement_dir: np.ndarray,  # [3]
    facing_dir: np.ndarray,    # [3]
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
    cache: torch.Tensor,           # [N, L, 36]
    playback_idx: torch.Tensor,    # [N] long
) -> torch.Tensor:
    """Rebuild the 4-frame planner context from the 50 Hz cache.

    Mirrors deploy UpdateContextFromMotion: sample at [pb, pb+50/30, pb+100/30,
    pb+150/30] cache indices, interpolated. Output: [N, 4, 36] MJ.
    """
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

    scene_cfg = HybridEvalSceneCfg(num_envs=args.num_envs, env_spacing=3.0)
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

    # --- Index splice maps -------------------------------------------------
    # Legs in IL order (12 indices), upper in IL order (17 indices). Their
    # union must cover 0..28.
    leg_idx_il = _leg_indices_il(il_joint_names)
    assert len(leg_idx_il) == 12, f"expected 12 leg joints, got {len(leg_idx_il)}"
    upper_idx_il = [il_joint_names.index(n) for n in PINK_CONTROLLED_JOINTS_IL]
    assert set(leg_idx_il) | set(upper_idx_il) == set(range(NUM_JOINTS)), (
        "leg + pink-controlled indices do not cover all 29 joints"
    )
    leg_idx_il_t = torch.as_tensor(leg_idx_il, dtype=torch.long, device=device)
    upper_idx_il_t = torch.as_tensor(upper_idx_il, dtype=torch.long, device=device)

    # Planner cache stores 29 joints MJ-ordered at `cache[..., 7:36]`.
    # Semantics (verified in stage0_planner_only.py):
    #   mj_data = il_data[G1_ISAACLAB_TO_MUJOCO_DOF]  (IL→MJ gather)
    #   il_data = mj_data[G1_MUJOCO_TO_ISAACLAB_DOF]  (MJ→IL gather)
    # To extract an MJ-indexed value for IL joint k, use MJ slot
    # `G1_MUJOCO_TO_ISAACLAB_DOF[k]` (the argsort-inverse of IL→MJ).
    il_to_mj = torch.as_tensor(
        G1_ISAACLAB_TO_MUJOCO_DOF, dtype=torch.long, device=device
    )
    mj_to_il = torch.as_tensor(
        G1_MUJOCO_TO_ISAACLAB_DOF, dtype=torch.long, device=device
    )
    leg_mj_slots = mj_to_il[leg_idx_il_t]  # [12] MJ slot holding each leg IL joint's value

    # --- Planner infra -----------------------------------------------------
    planner_pool = PlannerSessionPool(
        args.planner_onnx, pool_size=args.num_envs, device_id=0, serial=False
    )
    planner_context = torch.zeros(
        args.num_envs, 4, PLANNER_FRAME_DIM, device=device
    )
    planner_cache = torch.zeros(
        args.num_envs, RESAMPLED_FRAMES, PLANNER_FRAME_DIM, device=device
    )
    playback_idx = torch.zeros(args.num_envs, dtype=torch.long, device=device)

    # Seed planner context: yaw-normalized stand, joints @ default (IL→MJ).
    default_jp_il_t = torch.as_tensor(default_joint_pos_il, device=device).unsqueeze(0).expand(args.num_envs, -1)
    default_jp_mj_t = default_jp_il_t[:, il_to_mj]
    init_frame = torch.zeros(args.num_envs, PLANNER_FRAME_DIM, device=device)
    init_frame[:, 2] = PLANNER_CONTEXT_DEFAULT_HEIGHT
    init_frame[:, 3] = 1.0  # quat w
    init_frame[:, 7:] = default_jp_mj_t
    planner_context[:] = init_frame.unsqueeze(1).expand(-1, 4, -1).clone()

    # Frozen locomotion preset (SLOW_WALK +x).
    movement_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    facing_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # --- G1 encoder policy + Pink IK driver --------------------------------
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

    # --- Seed sim at default stand ----------------------------------------
    root_state = robot.data.default_root_state.clone()
    root_state[:, 0:3] += scene.env_origins  # z = 0.76 from cfg
    root_state[:, 7:13] = 0.0
    joint_pos_init = robot.data.default_joint_pos.clone()
    joint_vel_init = torch.zeros_like(joint_pos_init)

    robot.write_root_state_to_sim(root_state)
    robot.write_joint_state_to_sim(joint_pos_init, joint_vel_init)
    robot.set_joint_position_target(joint_pos_init)
    scene.write_data_to_sim()
    scene.update(dt=0.0)

    infer.reset(joint_pos=joint_pos_init)

    # Frozen Pink IK targets (pelvis_contour_link frame).
    left_target_pelvis = torch.tensor(
        LEFT_WRIST_REST_POSE_PELVIS, dtype=torch.float32, device=device
    ).unsqueeze(0).expand(args.num_envs, -1).contiguous()
    right_target_pelvis = torch.tensor(
        RIGHT_WRIST_REST_POSE_PELVIS, dtype=torch.float32, device=device
    ).unsqueeze(0).expand(args.num_envs, -1).contiguous()

    # --- Main loop --------------------------------------------------------
    num_policy_steps = int(args.episode_sec * POLICY_HZ)
    mpjpe_accum = torch.zeros(args.num_envs, device=device)
    step_counter = 0
    future_offsets = torch.arange(
        0, G1_NUM_FUTURE_FRAMES * 5, 5, dtype=torch.long, device=device
    )  # [10]

    for t in range(num_policy_steps):
        if not simulation_app.is_running():
            break

        joint_pos_il = robot.data.joint_pos.clone()
        joint_vel_il = robot.data.joint_vel.clone()
        root_quat_w = robot.data.root_state_w[:, 3:7].clone()
        base_ang_vel_b = robot.data.root_ang_vel_b.clone()
        gravity_b = robot.data.projected_gravity_b.clone()

        # 1. Planner every 5th tick
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
            traj_np, _ = planner_pool.run_batched(feeds)  # [N, 64, 36]
            traj_t = torch.as_tensor(traj_np, device=device, dtype=torch.float32)
            planner_cache[:] = resample_traj_30_to_50hz(traj_t, RESAMPLED_FRAMES)
            playback_idx.zero_()

        # 2. Sample future frames from cache at [pb, pb+5, ..., pb+45]
        idx = (playback_idx.view(-1, 1) + future_offsets.view(1, -1)).clamp(
            max=RESAMPLED_FRAMES - 1
        )  # [N, 10]
        idx_next = (idx + 1).clamp(max=RESAMPLED_FRAMES - 1)
        n_range = torch.arange(args.num_envs, device=device).view(-1, 1).expand(-1, G1_NUM_FUTURE_FRAMES)
        frames = planner_cache[n_range, idx]       # [N, 10, 36]
        frames_next = planner_cache[n_range, idx_next]  # [N, 10, 36]

        # 3. Lower-body future: legs from planner (MJ→IL), vel = finite diff @ 50 Hz
        joints_mj_future = frames[..., 7:]            # [N, 10, 29] MJ
        joints_mj_future_next = frames_next[..., 7:]
        leg_pos_mj = joints_mj_future[..., leg_mj_slots]  # [N, 10, 12]
        leg_vel_mj = (joints_mj_future_next[..., leg_mj_slots] - leg_pos_mj) * POLICY_HZ

        # 4. Upper-body future: Pink IK at current IL joint state, padded
        upper_pos_il = pink_driver.solve(
            curr_joint_pos_il=joint_pos_il,
            left_target_pelvis=left_target_pelvis,
            right_target_pelvis=right_target_pelvis,
        )  # [N, 17]

        # 5. Splice into [N, 10, 29] IL
        joint_pos_future = torch.zeros(
            args.num_envs, G1_NUM_FUTURE_FRAMES, NUM_JOINTS,
            dtype=torch.float32, device=device,
        )
        joint_vel_future = torch.zeros_like(joint_pos_future)
        # Legs (IL scatter)
        joint_pos_future[..., leg_idx_il_t] = leg_pos_mj
        joint_vel_future[..., leg_idx_il_t] = leg_vel_mj
        # Upper: pad one Pink IK solution across all 10 future frames; vel = 0
        joint_pos_future[..., upper_idx_il_t] = upper_pos_il.unsqueeze(1).expand(
            -1, G1_NUM_FUTURE_FRAMES, -1
        )
        # (joint_vel_future upper stays zero)

        # 6. ref_root_quat_future (world-frame wxyz from planner cache)
        ref_root_quat_future_wxyz = frames[..., 3:7]  # [N, 10, 4]

        # 7. Encoder/decoder step → motor targets
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

        # MPJPE vs frame-0 of the hybrid reference
        mpjpe = (robot.data.joint_pos - joint_pos_future[:, 0]).abs().mean(dim=-1)
        mpjpe_accum += mpjpe
        step_counter += 1

        playback_idx = (playback_idx + 1).clamp(max=RESAMPLED_FRAMES - 1)

        if t % 50 == 0:
            z = robot.data.root_pos_w[:, 2] - scene.env_origins[:, 2]
            print(
                f"[t={t / 50:5.2f}s] z={z.cpu().tolist()}  "
                f"joint_mae={mpjpe.cpu().tolist()}  "
                f"fallen={(z < 0.4).cpu().tolist()}"
            )

    mean_mpjpe = (mpjpe_accum / max(step_counter, 1)).cpu().tolist()
    final_z = (robot.data.root_pos_w[:, 2] - scene.env_origins[:, 2]).cpu().tolist()
    fallen = [z < 0.4 for z in final_z]
    print("\n=== Hybrid (planner + Pink IK) eval summary ===")
    for i in range(args.num_envs):
        print(
            f"env {i}: mean_joint_mae={mean_mpjpe[i]:.4f} rad  "
            f"final_z={final_z[i]:.3f}m  fallen={fallen[i]}"
        )

    simulation_app.close()


if __name__ == "__main__":
    main()
