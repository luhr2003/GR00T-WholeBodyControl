"""Stage 1: SONIC VR-3PT closed-loop smoke test in Isaac Lab.

Mirrors stage_smpl_eval.py's env / obs / reset plumbing; the only intended
divergence is the motion source. SMPL eval feeds the encoder directly from a
retargeted motion clip; this script feeds it from the kinematic planner
(deploy's planner_sonic.onnx) plus a frozen VR 3pt target captured at reset.

Obs semantics follow the **training code** (gear_sonic Isaac Lab obs
functions). Proprioception history, action scale, scene, sim dt/decimation,
robot reset are identical to stage_smpl_eval.

Default command preset: IDLE stand-still (mode=0, zero movement, target_vel=0).
VR 3pt is frozen at reset — computed from default-pose wrist / torso world
poses, normalised into root-local frame (mirror of C++
GatherVR3PointPosition).

Usage:
    uv run --active python -m sonic_python_inference.scripts.stage1_loco_only \
        --num-envs 4 --episode-sec 10
"""

from __future__ import annotations

import argparse
import re
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
    G1_ISAACLAB_TO_MUJOCO_DOF,
    G1_MODEL_12_ACTION_SCALE,
)

from sonic_python_inference.sonic_inference import (  # noqa: E402
    NUM_JOINTS,
    PLANNER_HEIGHT_DEFAULT,
    PLANNER_MODE_SLOW_WALK,
    SonicVR3PTInference,
    quat_conjugate,
    quat_mul,
    quat_rotate,
)


SIM_DT = 0.005  # 200 Hz
DECIMATION = 4  # policy @ 50 Hz

# VR 3pt order: left wrist (0), right wrist (1), head/torso (2). Offsets are
# body-local, applied before root-local normalization (wrist +0.18 x, torso
# +0.35 z). Matches C++ GatherVR3PointPosition in g1_deploy_onnx_ref.cpp.
VR_3PT_BODY_NAMES = ("left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link")
VR_3PT_BODY_OFFSETS = (
    (0.18, -0.025, 0.0),
    (0.18, 0.025, 0.0),
    (0.0, 0.0, 0.35),
)


@configclass
class Stage1SceneCfg(InteractiveSceneCfg):
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
    patterns (e.g. `.*_hip_pitch_joint`) — same format as IsaacLab's
    JointPositionActionCfg.scale. We replicate IsaacLab's dict-regex resolution
    here: pick the first regex key that fully matches the joint name.
    """
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


def _initial_vr_3pt_root_local(
    body_pos_w: torch.Tensor,  # [N, 3, 3]  world xyz (l_wrist, r_wrist, torso)
    body_quat_w: torch.Tensor,  # [N, 3, 4]  world quat_wxyz
    root_pos_w: torch.Tensor,  # [N, 3]
    root_quat_w: torch.Tensor,  # [N, 4] wxyz
    body_offsets: tuple[tuple[float, float, float], ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror of C++ GatherVR3PointPosition (g1_deploy_onnx_ref.cpp:1102-1187).

    1. Rotate body-local offset into world via body_quat, add to body_pos.
    2. Subtract root_pos, rotate by conj(root_quat) → root-local position.
    3. Orientation = conj(root_quat) * body_quat (offsets only translate).
    """
    N = body_pos_w.shape[0]
    device = body_pos_w.device
    root_quat_inv = quat_conjugate(root_quat_w)
    positions = []
    orientations = []
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

    sim_cfg = sim_utils.SimulationCfg(device=device, dt=SIM_DT)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.0, 3.0, 2.0], [0.0, 0.0, 0.75])

    scene_cfg = Stage1SceneCfg(num_envs=args.num_envs, env_spacing=3.0)
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
    print(f"[info] VR 3pt bodies: {VR_3PT_BODY_NAMES} → {vr_body_idx.tolist()}")

    default_joint_pos_il = (
        robot.data.default_joint_pos[0].cpu().numpy().astype(np.float32)
    )
    action_scale_il = _action_scale_il(il_joint_names)
    print(f"[info] action_scale_il (min/max/mean): "
          f"{action_scale_il.min():.4f} / {action_scale_il.max():.4f} / "
          f"{action_scale_il.mean():.4f}")

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

    # Canonical Isaac Lab reset: push default pose into PhysX (sim.reset() populates
    # robot.data.default_* but doesn't write into the articulation).
    default_joint_pos_t = robot.data.default_joint_pos.clone()
    default_joint_vel_t = robot.data.default_joint_vel.clone()
    default_root_state = robot.data.default_root_state.clone()
    default_root_state[:, 0:3] += scene.env_origins
    robot.write_root_state_to_sim(default_root_state)
    robot.write_joint_state_to_sim(default_joint_pos_t, default_joint_vel_t)
    robot.set_joint_position_target(default_joint_pos_t)
    scene.write_data_to_sim()
    scene.update(dt=0.0)

    joint_pos_init = robot.data.joint_pos.clone()
    root_state = robot.data.root_state_w.clone()
    root_pos_w = root_state[:, 0:3] - scene.env_origins
    root_quat_w = root_state[:, 3:7]

    body_state = robot.data.body_state_w.clone()
    body_pos_w = body_state[..., 0:3] - scene.env_origins[:, None, :]
    body_quat_w = body_state[..., 3:7]

    vr_pos_local, vr_orn_local = _initial_vr_3pt_root_local(
        body_pos_w=body_pos_w[:, vr_body_idx],
        body_quat_w=body_quat_w[:, vr_body_idx],
        root_pos_w=root_pos_w,
        root_quat_w=root_quat_w,
        body_offsets=VR_3PT_BODY_OFFSETS,
    )
    print(f"[info] initial vr_pos_local[0] = {vr_pos_local[0].tolist()}")

    infer.reset(joint_pos=joint_pos_init, root_pos=root_pos_w, root_quat_wxyz=root_quat_w)

    # Frozen command: SLOW_WALK forward at 0.3 m/s for all envs.
    N = args.num_envs
    mode = torch.full((N,), PLANNER_MODE_SLOW_WALK, dtype=torch.long, device=device)
    movement = torch.tensor([[1.0, 0.0, 0.0]] * N, dtype=torch.float32, device=device)
    facing = torch.tensor([[1.0, 0.0, 0.0]] * N, dtype=torch.float32, device=device)
    target_vel = torch.full((N,), 0.3, dtype=torch.float32, device=device)
    height = torch.full((N,), PLANNER_HEIGHT_DEFAULT, dtype=torch.float32, device=device)

    num_policy_steps = int(args.episode_sec * 50)
    start_root_xy = root_pos_w[:, 0:2].clone()

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

        for _ in range(DECIMATION):
            robot.set_joint_position_target(target_il)
            scene.write_data_to_sim()
            sim.step()
            scene.update(dt=SIM_DT)

        if t % 50 == 0:
            z = robot.data.root_pos_w[:, 2] - scene.env_origins[:, 2]
            travelled = (root_pos_w[:, 0:2] - start_root_xy).norm(dim=-1)
            print(
                f"[t={t / 50:5.2f}s] travelled={travelled.cpu().tolist()}  "
                f"z={z.cpu().tolist()}  fallen={(z < 0.4).cpu().tolist()}"
            )

    final_z = (robot.data.root_pos_w[:, 2] - scene.env_origins[:, 2]).cpu().tolist()
    travelled = (root_pos_w[:, 0:2] - start_root_xy).norm(dim=-1).cpu().tolist()
    fallen = [z < 0.4 for z in final_z]
    print("\n=== Stage 1 summary ===")
    for i in range(N):
        print(
            f"env {i}: travelled={travelled[i]:.2f}m  "
            f"final_z={final_z[i]:.3f}m  fallen={fallen[i]}"
        )

    simulation_app.close()


if __name__ == "__main__":
    main()
