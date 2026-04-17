"""Stage 1 (deploy ONNX variant): SONIC full-pipeline smoke test in Isaac Lab.

Identical to `stage1_loco_only.py` except it drives `SonicVR3PTInferenceDeploy`
(deploy release ONNX `model_encoder.onnx` / `model_decoder.onnx`, batch=1 per
ORT session, N sessions concurrent) instead of our dynamic-batch re-export.

Frequencies match deploy exactly:
    sim 200 Hz  →  decimation 4  →  policy 50 Hz  →  planner every 5th tick = 10 Hz

Fixed per-env command (all envs identical, IDLE stand-still):
    mode=IDLE   movement=[0,0,0]   facing=[1,0,0]   target_vel=0.0

VR 3pt is frozen at reset-time root-local values so the test isolates the
locomotion path.

Usage:
    uv run --active python -m sonic_python_inference.scripts.stage1_loco_only_deploy \
        --num-envs 4 --episode-sec 10
"""

from __future__ import annotations

import argparse


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--episode-sec", type=float, default=10.0)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument(
        "--encoder-onnx",
        type=str,
        default="gear_sonic_deploy/policy/release/model_encoder.onnx",
    )
    ap.add_argument(
        "--decoder-onnx",
        type=str,
        default="gear_sonic_deploy/policy/release/model_decoder.onnx",
    )
    ap.add_argument(
        "--planner-onnx",
        type=str,
        default="gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx",
    )
    return ap.parse_args()


args = _parse_args()

# ---------------------------------------------------------------------------
# AppLauncher first, as always for Isaac Lab scripts.
# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import ContactSensorCfg  # noqa: E402
from isaaclab.terrains import TerrainImporterCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from gear_sonic.envs.manager_env.robots.g1 import (  # noqa: E402
    G1_CYLINDER_MODEL_12_DEX_CFG,
    G1_ISAACLAB_TO_MUJOCO_DOF,
    G1_MODEL_12_ACTION_SCALE,
)

from sonic_python_inference.sonic_inference import (  # noqa: E402
    NUM_JOINTS,
    PLANNER_HEIGHT_DEFAULT,
    PLANNER_MODE_IDLE,
    quat_conjugate,
    quat_mul,
    quat_rotate,
)
from sonic_python_inference.sonic_inference_deploy import (  # noqa: E402
    SonicVR3PTInferenceDeploy,
)


SIM_DT = 0.005  # 200 Hz
DECIMATION = 4  # policy @ 50 Hz

VR_3PT_BODY_NAMES = ("left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link")
VR_3PT_BODY_OFFSETS = (
    (0.18, -0.025, 0.0),  # left wrist → hand
    (0.18, 0.025, 0.0),  # right wrist → hand (y mirrored)
    (0.0, 0.0, 0.35),  # torso → head
)


@configclass
class SonicStage1SceneCfg(InteractiveSceneCfg):
    """Mirrors training MySceneCfg (plane variant) from
    gear_sonic/envs/manager_env/modular_tracking_env_cfg.py:268-390.
    """

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )
    robot = G1_CYLINDER_MODEL_12_DEX_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=10.0,
        debug_vis=False,
    )


def _action_scale_il(joint_names: list[str]) -> np.ndarray:
    """Resolve `G1_MODEL_12_ACTION_SCALE` — its keys are regex patterns
    (`.*_hip_pitch_joint`), NOT concrete joint names — by matching each IL joint
    name against all patterns. Training's `JointPositionActionCfg` does the same
    internally via `string_utils.resolve_matching_names`.
    """
    import re

    scale = np.zeros(NUM_JOINTS, dtype=np.float32)
    unresolved: list[str] = []
    for i, name in enumerate(joint_names):
        matched = None
        for pattern, value in G1_MODEL_12_ACTION_SCALE.items():
            if re.fullmatch(pattern, name):
                matched = value
                break
        if matched is None:
            unresolved.append(name)
            scale[i] = 1.0
        else:
            scale[i] = matched
    if unresolved:
        raise RuntimeError(
            f"Could not resolve action_scale for joints: {unresolved}. "
            f"Available patterns: {list(G1_MODEL_12_ACTION_SCALE.keys())}"
        )
    return scale


def _initial_vr_3pt_root_local(
    body_pos_w: torch.Tensor,
    body_quat_w: torch.Tensor,
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    body_offsets: tuple[tuple[float, float, float], ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    N = body_pos_w.shape[0]
    device = body_pos_w.device
    root_quat_inv = quat_conjugate(root_quat_w)
    positions, orientations = [], []
    for i, off in enumerate(body_offsets):
        bp = body_pos_w[:, i]
        bq = body_quat_w[:, i]
        off_b = torch.tensor(off, dtype=torch.float32, device=device).expand(N, 3)
        off_w = quat_rotate(bq, off_b)
        vr_w = bp + off_w
        rel = vr_w - root_pos_w
        positions.append(quat_rotate(root_quat_inv, rel))
        orientations.append(quat_mul(root_quat_inv, bq))
    return torch.cat(positions, dim=-1), torch.cat(orientations, dim=-1)


def main():
    device = "cuda"

    # Mirror training sim setup (modular_tracking_env_cfg.py:964-976):
    #   sim.dt=0.005, render_interval=decimation, physics_material bound to
    #   terrain, physx gpu buffers bumped up. This is what the policy trained
    #   against; any mismatch (default friction, smaller physx buffers, …)
    #   changes contact dynamics and causes the robot to slip / fall.
    sim_cfg = sim_utils.SimulationCfg(device=device, dt=SIM_DT)
    sim_cfg.render_interval = DECIMATION
    scene_cfg = SonicStage1SceneCfg(num_envs=args.num_envs, env_spacing=2.5)
    sim_cfg.physics_material = scene_cfg.terrain.physics_material
    sim_cfg.physx.gpu_max_rigid_patch_count = 10 * 2**15
    sim_cfg.physx.gpu_collision_stack_size = 2**26
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.75])

    scene = InteractiveScene(scene_cfg)

    sim.reset()
    robot = scene["robot"]
    scene.update(dt=0.0)

    il_joint_names = list(robot.data.joint_names)
    il_body_names = list(robot.data.body_names)

    vr_body_idx = torch.as_tensor(
        [il_body_names.index(n) for n in VR_3PT_BODY_NAMES],
        dtype=torch.long,
        device=device,
    )
    print(f"[info] joints (IL order): {il_joint_names}")
    print(
        f"[info] VR 3pt bodies (IL indices): {VR_3PT_BODY_NAMES} → {vr_body_idx.tolist()}"
    )

    default_joint_pos_il = (
        robot.data.default_joint_pos[0].cpu().numpy().astype(np.float32)
    )
    action_scale_il = _action_scale_il(il_joint_names)

    infer = SonicVR3PTInferenceDeploy(
        num_envs=args.num_envs,
        encoder_onnx=args.encoder_onnx,
        decoder_onnx=args.decoder_onnx,
        planner_onnx=args.planner_onnx,
        default_angles=default_joint_pos_il,
        action_scale=action_scale_il,
        isaaclab_to_mujoco_dof=G1_ISAACLAB_TO_MUJOCO_DOF,
        device=device,
    )

    # Canonical Isaac Lab reset — sim.reset() populates robot.data.default_* but
    # does NOT push into PhysX. Mirror the training manager env's reset pattern.
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
    root_state = robot.data.root_state_w.clone()
    root_pos_w = root_state[:, 0:3] - scene.env_origins
    root_quat_w = root_state[:, 3:7]

    body_state = robot.data.body_state_w.clone()
    body_pos_w = body_state[..., 0:3] - scene.env_origins[:, None, :]
    body_quat_w = body_state[..., 3:7]

    vr_pos_w = body_pos_w[:, vr_body_idx]
    vr_quat_w = body_quat_w[:, vr_body_idx]

    vr_pos_local, vr_orn_local = _initial_vr_3pt_root_local(
        body_pos_w=vr_pos_w,
        body_quat_w=vr_quat_w,
        root_pos_w=root_pos_w,
        root_quat_w=root_quat_w,
        body_offsets=VR_3PT_BODY_OFFSETS,
    )
    print(f"[info] initial vr_pos_local[0] = {vr_pos_local[0].tolist()}")

    infer.reset(joint_pos=joint_pos_il, root_pos=root_pos_w, root_quat_wxyz=root_quat_w)

    N = args.num_envs
    mode = torch.full((N,), PLANNER_MODE_IDLE, dtype=torch.long, device=device)
    movement = torch.tensor([[0.0, 0.0, 0.0]] * N, dtype=torch.float32, device=device)
    facing = torch.tensor([[1.0, 0.0, 0.0]] * N, dtype=torch.float32, device=device)
    target_vel = torch.full((N,), 0.0, dtype=torch.float32, device=device)
    height = torch.full((N,), PLANNER_HEIGHT_DEFAULT, dtype=torch.float32, device=device)

    num_policy_steps = int(args.episode_sec * 50)
    start_root_xy = root_pos_w[:, 0:2].clone()
    last_action_target_il = robot.data.default_joint_pos.clone()

    for t in range(num_policy_steps):
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

    travelled = (root_pos_w[:, 0:2] - start_root_xy).norm(dim=-1).cpu().tolist()
    fallen = (root_pos_w[:, 2] < 0.4).cpu().tolist()
    print("\n=== Stage 1 (deploy ONNX) summary ===")
    for i in range(N):
        print(f"env {i}: travelled={travelled[i]:.2f}m  fallen={fallen[i]}")

    simulation_app.close()


if __name__ == "__main__":
    main()
