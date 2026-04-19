"""VR 3pt eval with two interactive cubes driving the wrist targets.

Same pipeline as `stage_vr3pt_eval.py` — `SonicVR3PTInference` (teleop
encoder + shared decoder + kplanner). The only differences:

- Planner is pinned to **IDLE mode, target_vel = 0** (stand in place).
- Left wrist target and right wrist target are no longer the frozen
  reset-time poses; they are read per-tick from two interactive VisualCuboids:
      red  cube → right_wrist_yaw_link target
      blue cube → left_wrist_yaw_link  target
- Torso (neck/head) target stays frozen at the reset-time pose — just like
  the original script.

No Pink IK. Targets go straight into `vr_3pt_position / vr_3pt_orientation`
(root-local), mirroring C++ `GatherVR3PointPosition`.

Cubes are `isaacsim.core.api.objects.VisualCuboid` — no collider, no rigid
body — so dragging one in the viewer doesn't touch the robot.

Usage:
    uv run --active python -m sonic_python_inference.scripts.stage_vr3pt_cube_eval \\
        --num-envs 1
"""

from __future__ import annotations

import argparse
import re


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--episode-sec", type=float, default=600.0)
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

from isaacsim.core.api.objects import VisualCuboid  # noqa: E402

from gear_sonic.envs.manager_env.robots.g1 import (  # noqa: E402
    G1_CYLINDER_MODEL_12_DEX_CFG,
    G1_ISAACLAB_TO_MUJOCO_DOF,
    G1_MODEL_12_ACTION_SCALE,
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


SIM_DT = 0.005
DECIMATION = 4

# VR 3pt slot order: left wrist (0), right wrist (1), head/torso (2).
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


def _initial_vr_3pt_root_local(
    body_pos_w: torch.Tensor,   # [N, 3, 3]  world xyz (l_wrist, r_wrist, torso)
    body_quat_w: torch.Tensor,  # [N, 3, 4]  world quat_wxyz
    root_pos_w: torch.Tensor,   # [N, 3]
    root_quat_w: torch.Tensor,  # [N, 4] wxyz
    body_offsets: tuple[tuple[float, float, float], ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bit-for-bit copy of stage_vr3pt_eval._initial_vr_3pt_root_local — keeps
    the cube spawn position identical to the frozen target vr3pt_eval uses.
    Returns ([N, 9], [N, 12]) flattened over three slots."""
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


def _root_local_to_world(
    pos_local: torch.Tensor,
    quat_local: torch.Tensor,
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of _world_to_root_local."""
    return (
        root_pos_w + quat_rotate(root_quat_w, pos_local),
        quat_mul(root_quat_w, quat_local),
    )


def _world_to_root_local(
    target_pos_w: torch.Tensor,
    target_quat_w: torch.Tensor,
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """World-frame pose → root-local frame. Mirror of C++
    GatherVR3PointPosition for the second half (after the offset has been
    folded into target_pos_w)."""
    root_quat_inv = quat_conjugate(root_quat_w)
    rel = target_pos_w - root_pos_w
    pos_local = quat_rotate(root_quat_inv, rel)
    quat_local = quat_mul(root_quat_inv, target_quat_w)
    return pos_local, quat_local


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
        dtype=torch.long, device=device,
    )
    print(f"[info] joints (IL order): {il_joint_names}")
    print(f"[info] VR 3pt bodies: {VR_3PT_BODY_NAMES} → {vr_body_idx.tolist()}")

    default_joint_pos_il = robot.data.default_joint_pos[0].cpu().numpy().astype(np.float32)
    action_scale_il = _action_scale_il(il_joint_names)

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

    # --- Canonical Isaac Lab reset -----------------------------------------
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

    # --- Compute frozen VR 3pt targets, exactly like stage_vr3pt_eval -----
    # vr_pos_local: [N, 9] = cat(left, right, torso); vr_orn_local: [N, 12].
    vr_pos_local_init, vr_orn_local_init = _initial_vr_3pt_root_local(
        body_pos_w=body_pos_w[:, vr_body_idx],
        body_quat_w=body_quat_w[:, vr_body_idx],
        root_pos_w=root_pos_w,
        root_quat_w=root_quat_w,
        body_offsets=VR_3PT_BODY_OFFSETS,
    )
    l_pos_local_init = vr_pos_local_init[:, 0:3]
    r_pos_local_init = vr_pos_local_init[:, 3:6]
    torso_pos_local = vr_pos_local_init[:, 6:9]
    l_quat_local_init = vr_orn_local_init[:, 0:4]
    r_quat_local_init = vr_orn_local_init[:, 4:8]
    torso_quat_local = vr_orn_local_init[:, 8:12]
    print(f"[info] initial vr_pos_local[0] = {vr_pos_local_init[0].tolist()}")

    # Cube world-frame spawn: back-transform the frozen root-local wrist
    # targets into world so the cubes start exactly at vr3pt_eval's frozen
    # positions. Torso cube not spawned — torso stays frozen in root-local.
    l_cube_pos_w, _ = _root_local_to_world(
        l_pos_local_init, l_quat_local_init, root_pos_w, root_quat_w,
    )
    r_cube_pos_w, _ = _root_local_to_world(
        r_pos_local_init, r_quat_local_init, root_pos_w, root_quat_w,
    )

    right_cubes: list[VisualCuboid] = []
    left_cubes: list[VisualCuboid] = []
    r_wrist_w_np = (r_cube_pos_w + scene.env_origins).cpu().numpy()
    l_wrist_w_np = (l_cube_pos_w + scene.env_origins).cpu().numpy()
    for i in range(args.num_envs):
        right_cubes.append(
            VisualCuboid(
                prim_path=f"/World/envs/env_{i}/RightWristCube",
                name=f"right_wrist_cube_{i}",
                position=r_wrist_w_np[i].astype(np.float32),
                size=0.06,
                color=np.array([0.9, 0.1, 0.1], dtype=np.float32),
            )
        )
        left_cubes.append(
            VisualCuboid(
                prim_path=f"/World/envs/env_{i}/LeftWristCube",
                name=f"left_wrist_cube_{i}",
                position=l_wrist_w_np[i].astype(np.float32),
                size=0.06,
                color=np.array([0.1, 0.3, 0.9], dtype=np.float32),
            )
        )

    infer.reset(joint_pos=joint_pos_init, root_pos=root_pos_w, root_quat_wxyz=root_quat_w)

    # --- Frozen command: IDLE, target_vel = 0 -----------------------------
    N = args.num_envs
    mode = torch.full((N,), PLANNER_MODE_IDLE, dtype=torch.long, device=device)
    movement = torch.tensor([[1.0, 0.0, 0.0]] * N, dtype=torch.float32, device=device)
    facing = torch.tensor([[1.0, 0.0, 0.0]] * N, dtype=torch.float32, device=device)
    target_vel = torch.zeros(N, dtype=torch.float32, device=device)
    height = torch.full((N,), PLANNER_HEIGHT_DEFAULT, dtype=torch.float32, device=device)

    print("[info] IDLE @ 0 m/s. Drag the red (right) / blue (left) cubes to "
          "move the wrist targets. Torso target stays frozen at the reset pose.")

    t = 0
    while simulation_app.is_running():
        joint_pos_il = robot.data.joint_pos.clone()
        joint_vel_il = robot.data.joint_vel.clone()
        root_state = robot.data.root_state_w.clone()
        root_pos_w = root_state[:, 0:3] - scene.env_origins
        root_quat_w = root_state[:, 3:7]
        base_ang_vel_b = robot.data.root_ang_vel_b.clone()
        gravity_b = robot.data.projected_gravity_b.clone()

        # --- Read live wrist targets from cubes ---------------------------
        r_pos_list, r_quat_list, l_pos_list, l_quat_list = [], [], [], []
        for i in range(N):
            rp, rq = right_cubes[i].get_world_pose()
            lp, lq = left_cubes[i].get_world_pose()
            r_pos_list.append(np.asarray(rp, dtype=np.float32))
            r_quat_list.append(np.asarray(rq, dtype=np.float32))
            l_pos_list.append(np.asarray(lp, dtype=np.float32))
            l_quat_list.append(np.asarray(lq, dtype=np.float32))
        r_pos_w = torch.as_tensor(np.stack(r_pos_list), device=device) - scene.env_origins
        r_quat_w = torch.as_tensor(np.stack(r_quat_list), device=device)
        l_pos_w = torch.as_tensor(np.stack(l_pos_list), device=device) - scene.env_origins
        l_quat_w = torch.as_tensor(np.stack(l_quat_list), device=device)

        l_pos_local, l_quat_local = _world_to_root_local(
            l_pos_w, l_quat_w, root_pos_w, root_quat_w,
        )
        r_pos_local, r_quat_local = _world_to_root_local(
            r_pos_w, r_quat_w, root_pos_w, root_quat_w,
        )

        # Slot order: left wrist, right wrist, torso.
        vr_pos_local = torch.cat([l_pos_local, r_pos_local, torso_pos_local], dim=-1)
        vr_orn_local = torch.cat([l_quat_local, r_quat_local, torso_quat_local], dim=-1)

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
            print(
                f"[t={t / 50:5.2f}s] z={z.cpu().tolist()}  "
                f"r_local[0]={r_pos_local[0].cpu().tolist()}  "
                f"l_local[0]={l_pos_local[0].cpu().tolist()}"
            )

        t += 1

    simulation_app.close()


if __name__ == "__main__":
    main()
