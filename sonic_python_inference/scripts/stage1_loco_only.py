"""Stage 1 verification: SONIC full-pipeline smoke test in Isaac Lab.

Goal: run planner → encoder → decoder → Isaac Lab end-to-end under the same
command the stage0 planner-only probe validated, so any remaining issue is
isolated to the encoder/decoder path (not the planner). All envs use the
*same* command; N is purely for parallelism / statistical signal.

Fixed per-env command (all envs identical, mirrors stage0_planner_only):
    mode=SLOW_WALK (id=1)   movement=[1,0,0]   facing=[1,0,0]   target_vel=0.3

VR 3pt is frozen at the value computed at reset (root-local from the
default-pose wrist/torso world poses). It's still an encoder input here, but
does not vary over the episode.

Usage:
    uv run --active python -m sonic_python_inference.scripts.stage1_loco_only \
        --num-envs 4 --episode-sec 10

Flags:
    --headless       run without viewer (default: viewer on)
    --num-envs       number of parallel envs (default 4)
    --episode-sec    seconds to simulate (default 10)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--episode-sec", type=float, default=10.0)
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
# AppLauncher must come before any other isaaclab import. Keep this block at
# the top of the script.
# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

# Safe to import the rest now
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.terrains import TerrainImporterCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

# gear_sonic is importable (installed earlier) — use its G1 cfg / mapping
from gear_sonic.envs.manager_env.robots.g1 import (  # noqa: E402
    G1_CYLINDER_MODEL_12_DEX_CFG,
    G1_ISAACLAB_TO_MUJOCO_DOF,
)

from sonic_python_inference.sonic_inference import (  # noqa: E402
    NUM_JOINTS,
    PLANNER_HEIGHT_DEFAULT,
    PLANNER_MODE_IDLE,
    SonicVR3PTInference,
    quat_conjugate,
    quat_mul,
    quat_rotate,
)


SIM_DT = 0.005  # 200 Hz
DECIMATION = 4  # policy @ 50 Hz

# VR 3pt order MUST match training (sonic_release/config.yaml:378-391) and C++
# deploy (g1_deploy_onnx_ref.cpp:1133 "Order: left_wrist (0), right_wrist (1),
# head/torso (2)"). Offsets are in body-local frame, applied before root
# normalization — wrist +0.18 along body x (hand forward), torso +0.35 along
# body z (head height).
VR_3PT_BODY_NAMES = ("left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link")
VR_3PT_BODY_OFFSETS = (
    (0.18, -0.025, 0.0),  # left wrist → hand
    (0.18, 0.025, 0.0),  # right wrist → hand (y mirrored)
    (0.0, 0.0, 0.35),  # torso → head
)


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------
@configclass
class SonicStage1SceneCfg(InteractiveSceneCfg):
    """Flat ground + G1 per env. No motion dataset, no observation manager."""

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
    """Training scale = 1.0 (isaaclab JointPositionActionCfg default, unset in
    sonic_release/config.yaml:429-434). Deploy-style `0.25 * effort/stiffness`
    attenuates the policy output and is wrong here — training wrote `target =
    action * 1.0 + default_joint_pos`, so inference must feed the same.
    """
    del joint_names  # kept for API compat
    return np.ones(NUM_JOINTS, dtype=np.float32)


def _initial_vr_3pt_root_local(
    body_pos_w: torch.Tensor,  # [N, 3, 3]  world xyz for (l_wrist, r_wrist, torso)
    body_quat_w: torch.Tensor,  # [N, 3, 4]  world quat_wxyz, same order
    root_pos_w: torch.Tensor,  # [N, 3]
    root_quat_w: torch.Tensor,  # [N, 4] wxyz
    body_offsets: tuple[tuple[float, float, float], ...],  # per-body local offset
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror of C++ GatherVR3PointPosition (g1_deploy_onnx_ref.cpp:1102-1187).

    1. Rotate each body's local offset into world frame via body_quat, add to
       body_pos → world-frame VR point.
    2. Subtract root_pos, rotate by conj(root_quat) → root-local position.
    3. Orientation = conj(root_quat) * body_quat (orientations don't get the
       offset treatment — offsets only translate, they don't rotate).
    """
    N = body_pos_w.shape[0]
    device = body_pos_w.device
    root_quat_inv = quat_conjugate(root_quat_w)  # [N, 4]
    positions = []
    orientations = []
    for i, off in enumerate(body_offsets):
        bp = body_pos_w[:, i]  # [N, 3]
        bq = body_quat_w[:, i]  # [N, 4]
        off_b = torch.tensor(off, dtype=torch.float32, device=device).expand(N, 3)
        off_w = quat_rotate(bq, off_b)  # rotate offset from body frame to world
        vr_w = bp + off_w  # [N, 3] world-frame VR point
        rel = vr_w - root_pos_w
        positions.append(quat_rotate(root_quat_inv, rel))
        orientations.append(quat_mul(root_quat_inv, bq))
    pos = torch.cat(positions, dim=-1)  # [N, 9]
    orn = torch.cat(orientations, dim=-1)  # [N, 12]
    return pos, orn


def _projected_gravity_to_base(base_quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Gravity direction [0,0,-1] rotated into base frame."""
    gw = torch.zeros(base_quat_wxyz.shape[0], 3, device=base_quat_wxyz.device)
    gw[:, 2] = -1.0
    return quat_rotate(quat_conjugate(base_quat_wxyz), gw)


def main():
    device = "cuda"

    # --- Sim + scene -------------------------------------------------------
    sim_cfg = sim_utils.SimulationCfg(device=device, dt=SIM_DT)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.75])

    scene_cfg = SonicStage1SceneCfg(num_envs=args.num_envs, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    robot = scene["robot"]
    scene.update(dt=0.0)

    # --- Resolve indices ---------------------------------------------------
    # Isaac Lab joint order and body order (names as configured by the URDF importer).
    il_joint_names = list(robot.data.joint_names)
    il_body_names = list(robot.data.body_names)

    # Resolve 3 VR body indices (by link name) in Isaac Lab body order
    vr_body_idx = torch.as_tensor(
        [il_body_names.index(n) for n in VR_3PT_BODY_NAMES], dtype=torch.long, device=device
    )
    print(f"[info] joints (IL order): {il_joint_names}")
    print(f"[info] VR 3pt bodies (IL indices): {VR_3PT_BODY_NAMES} → {vr_body_idx.tolist()}")

    # default_angles + action_scale in IsaacLab order — matches the IL-ordered
    # action output that the decoder produces (see C++ deploy ref line 3098).
    default_joint_pos_il = robot.data.default_joint_pos[0].cpu().numpy().astype(np.float32)  # [29]
    action_scale_il = _action_scale_il(il_joint_names)

    # --- Policy ------------------------------------------------------------
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

    # --- Reset: initial joint pos → planner + history, compute fixed VR 3pt
    # `sim.reset()` populates `robot.data.default_*` from the config but does
    # NOT push those values into PhysX — the articulation boots at joint_pos=0.
    # Canonical Isaac Lab reset (same pattern the GR00T training env uses):
    # explicitly write default joint and root state into the sim.
    default_joint_pos_t = robot.data.default_joint_pos.clone()
    default_joint_vel_t = robot.data.default_joint_vel.clone()
    default_root_state = robot.data.default_root_state.clone()
    default_root_state[:, 0:3] += scene.env_origins
    robot.write_root_state_to_sim(default_root_state)
    robot.write_joint_state_to_sim(default_joint_pos_t, default_joint_vel_t)
    robot.set_joint_position_target(default_joint_pos_t)
    scene.write_data_to_sim()
    scene.update(dt=0.0)

    joint_pos_il = robot.data.joint_pos.clone()
    root_state = robot.data.root_state_w.clone()  # [N, 13] pos+quat_wxyz+lin+ang
    root_pos_w = root_state[:, 0:3] - scene.env_origins  # undo per-env offset → common frame
    root_quat_w = root_state[:, 3:7]

    body_state = robot.data.body_state_w.clone()  # [N, B, 13]
    body_pos_w = body_state[..., 0:3] - scene.env_origins[:, None, :]
    body_quat_w = body_state[..., 3:7]

    vr_pos_w = body_pos_w[:, vr_body_idx]  # [N, 3, 3]  order: (l_wrist, r_wrist, torso)
    vr_quat_w = body_quat_w[:, vr_body_idx]  # [N, 3, 4]

    vr_pos_local, vr_orn_local = _initial_vr_3pt_root_local(
        body_pos_w=vr_pos_w,
        body_quat_w=vr_quat_w,
        root_pos_w=root_pos_w,
        root_quat_w=root_quat_w,
        body_offsets=VR_3PT_BODY_OFFSETS,
    )
    print(f"[info] initial vr_pos_local[0] = {vr_pos_local[0].tolist()}")

    infer.reset(joint_pos=joint_pos_il, root_pos=root_pos_w, root_quat_wxyz=root_quat_w)

    # --- Per-env command schedule (frozen for episode) ---------------------
    # All envs run IDLE stand-still: zero movement, target_vel=0. This strips
    # the gait out of the test so any residual motion comes from the
    # pipeline's attempt to just hold the default pose. If this fails the
    # bug is in proprio / action_scale / physics setup, not the gait path.
    N = args.num_envs
    mode = torch.full((N,), PLANNER_MODE_IDLE, dtype=torch.long, device=device)
    movement = torch.tensor([[0.0, 0.0, 0.0]] * N, dtype=torch.float32, device=device)
    facing = torch.tensor([[1.0, 0.0, 0.0]] * N, dtype=torch.float32, device=device)
    target_vel = torch.full((N,), 0.0, dtype=torch.float32, device=device)
    height = torch.full((N,), PLANNER_HEIGHT_DEFAULT, dtype=torch.float32, device=device)

    presets = [
        {"mode": PLANNER_MODE_IDLE, "move": [0, 0, 0], "face": [1, 0, 0], "vel": 0.0}
        for _ in range(N)
    ]

    # --- Main sim loop -----------------------------------------------------
    num_policy_steps = int(args.episode_sec * 50)
    start_root_xy = root_pos_w[:, 0:2].clone()
    last_action_target_il = robot.data.default_joint_pos.clone()  # fallback for first tick

    for t in range(num_policy_steps):
        # Gather state
        joint_pos_il = robot.data.joint_pos.clone()
        joint_vel_il = robot.data.joint_vel.clone()
        root_state = robot.data.root_state_w.clone()
        root_pos_w = root_state[:, 0:3] - scene.env_origins
        root_quat_w = root_state[:, 3:7]
        base_ang_vel_b = robot.data.root_ang_vel_b.clone()
        gravity_b = robot.data.projected_gravity_b.clone()

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
        last_action_target_il = target_il

        # Apply target for `decimation` sim steps at 200 Hz
        for _ in range(DECIMATION):
            robot.set_joint_position_target(last_action_target_il)
            scene.write_data_to_sim()
            sim.step()
            scene.update(dt=SIM_DT)

        if t % 50 == 0:
            travelled = (root_pos_w[:, 0:2] - start_root_xy).norm(dim=-1)
            z = root_pos_w[:, 2]
            print(
                f"[t={t / 50:5.2f}s] travelled={travelled.tolist()}, "
                f"z={z.tolist()}, any_fallen={(z < 0.4).any().item()}"
            )

    # --- Pass/fail summary -------------------------------------------------
    travelled = (root_pos_w[:, 0:2] - start_root_xy).norm(dim=-1).cpu().tolist()
    fallen = (root_pos_w[:, 2] < 0.4).cpu().tolist()
    print("\n=== Stage 1 summary ===")
    for i in range(N):
        preset = presets[i] if i < len(presets) else {}
        print(f"env {i}: travelled={travelled[i]:.2f}m  fallen={fallen[i]}  preset={preset}")

    simulation_app.close()


if __name__ == "__main__":
    main()
