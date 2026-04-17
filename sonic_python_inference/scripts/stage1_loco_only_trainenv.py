"""Stage 1 (training-env variant): SONIC deploy pipeline inside training's EXACT scene.

Same policy & step loop as `stage1_loco_only_deploy.py`, but the scene /
physics setup is built by reusing training's own classes from
`gear_sonic/envs/manager_env/modular_tracking_env_cfg.py`:

    * `MySceneCfg(config=minimal_config)` → IDENTICAL terrain plane with
      `RigidBodyMaterialCfg(friction=1.0, multiply)`, DistantLight + dim
      DomeLight, ContactSensorCfg on `/Robot/.*`, env_spacing / replicate_physics.
    * Robot slot (`MISSING` in `MySceneCfg`) filled with
      `G1_CYLINDER_MODEL_12_DEX_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")`,
      mirroring `override_settings()` exactly (modular_tracking_env_cfg.py:998–1020).
    * Sim overrides copied from `override_settings()`: dt=0.005, decimation=4,
      `render_interval=decimation`, `physics_material = terrain.physics_material`,
      `physx.gpu_max_rigid_patch_count = 10 * 2**15`, `gpu_collision_stack_size = 2**26`.

We DO NOT instantiate commands / rewards / terminations / events / recorders:
those depend on `motion_lib` datasets (`data/motion_lib_bones_seed/...`) that
are not part of the release. Inference doesn't need them — we drive actions
directly via `robot.set_joint_position_target` in the sim step loop.

If the robot still falls with this setup, any remaining mismatch is NOT on the
scene/physics side — it's obs values or policy frames.

Usage:
    cd /home/magics/magicsim/GR00T-WholeBodyControl
    .venv_isaac/bin/python -m sonic_python_inference.scripts.stage1_loco_only_trainenv \
        --num-envs 4 --episode-sec 10 --headless
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
# AppLauncher must come first (Isaac Lab extension bootstrap).
# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402

from gear_sonic.envs.manager_env.modular_tracking_env_cfg import MySceneCfg  # noqa: E402
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


SIM_DT = 0.005  # 200 Hz, matches training sim_dt (modular_tracking_env_cfg.py:969)
DECIMATION = 4  # policy @ 50 Hz, matches training decimation (:965)

VR_3PT_BODY_NAMES = ("left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link")
VR_3PT_BODY_OFFSETS = (
    (0.18, -0.025, 0.0),  # left wrist → hand
    (0.18, 0.025, 0.0),  # right wrist → hand (y mirrored)
    (0.0, 0.0, 0.35),  # torso → head
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


def _build_scene_cfg(num_envs: int) -> MySceneCfg:
    """Reuse training's MySceneCfg (plane variant) with a minimal config dict.

    Only the keys MySceneCfg actually reads are provided — all motion_lib / object
    / curriculum / ego-render paths are left out. Robot slot is filled after
    __init__ because MySceneCfg declares `self.robot = dataclasses.MISSING`
    (training fills it in `override_settings`).
    """
    cfg_dict = {
        "num_envs": num_envs,
        "env_spacing": 2.5,  # matches MySceneCfg default (modular_tracking_env_cfg.py:276)
        "replicate_physics": True,  # matches default
        "terrain_type": "plane",
        "render_results": False,
        "overview_camera": False,
        "render_ego_random": False,
        "add_object": False,
        "robot": {"type": "g1_model_12_dex"},
    }
    scene_cfg = MySceneCfg(config=cfg_dict)
    # Training's override_settings (:1011-1014) assigns robot via:
    #     robot_mapping["g1_model_12_dex"]["robot_cfg"].replace(prim_path=...)
    scene_cfg.robot = G1_CYLINDER_MODEL_12_DEX_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )
    return scene_cfg


def _build_sim_cfg(scene_cfg: MySceneCfg, device: str) -> sim_utils.SimulationCfg:
    """Mirror training `override_settings()` sim customisations.

    modular_tracking_env_cfg.py:964–976:
        sim.dt = config.sim_dt = 0.005
        sim.render_interval = decimation = 4
        sim.physics_material = scene.terrain.physics_material
        sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        sim.physx.gpu_collision_stack_size = 2**26
    """
    sim_cfg = sim_utils.SimulationCfg(device=device, dt=SIM_DT)
    sim_cfg.render_interval = DECIMATION
    sim_cfg.physics_material = scene_cfg.terrain.physics_material
    sim_cfg.physx.gpu_max_rigid_patch_count = 10 * 2**15
    sim_cfg.physx.gpu_collision_stack_size = 2**26
    return sim_cfg


def main():
    device = "cuda"

    scene_cfg = _build_scene_cfg(args.num_envs)
    sim_cfg = _build_sim_cfg(scene_cfg, device)

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
    # does NOT push into PhysX. Mirror training's ResetRobotState event
    # (commands.py:3054-3063 in TrackingCommand._resample_command).
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
    print("\n=== Stage 1 (training-env scene) summary ===")
    for i in range(N):
        print(f"env {i}: travelled={travelled[i]:.2f}m  fallen={fallen[i]}")

    simulation_app.close()


if __name__ == "__main__":
    main()
