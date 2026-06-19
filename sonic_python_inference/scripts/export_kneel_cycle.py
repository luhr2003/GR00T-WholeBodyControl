"""Export the FULL one-leg kneel-down -> hold -> stand-up cycle, WITH tracking.

Closed-loop SONIC (planner -> G1 encoder -> decoder -> PD), two-phase like
stage_hybrid_kneel.py:
  - crouch phase (t < --crouch-steps): planner mode 6 kneelOneLeg, height set
  - stand  phase (t >= --crouch-steps): planner mode 0 idle, height disabled
Records the ACHIEVED leg-joint trajectory at EVERY policy step (one JSON entry
per step) plus the whole-process video. Right = native planner; left = sagittal
mirror of the planner reference (validated). Legs only (upper held at rest).

Outputs (under --out-dir):
    kneel_cycle_left.json / kneel_cycle_right.json   ("steps": one entry per step)
    videos/kneel_cycle_<side>.mp4

Usage:
    OMNI_KIT_ACCEPT_EULA=YES .venv_isaac/bin/python -m \
        sonic_python_inference.scripts.export_kneel_cycle \
        --side both --crouch-steps 180 --stand-steps 180 --out-dir kneel_cycle_out
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


PLANNER_MODE_KNEEL_ONE_LEG = 6


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=["both", "right", "left"], default="both")
    ap.add_argument("--down-steps", type=int, default=140, help="policy steps descending into the kneel (50Hz)")
    ap.add_argument("--hold-steps", type=int, default=150, help="policy steps balancing in the kneel")
    ap.add_argument("--stand-steps", type=int, default=160, help="policy steps standing back up")
    ap.add_argument("--kneel-height", type=float, default=0.2, help="planner height to trigger kneel")
    ap.add_argument("--out-dir", type=str, default="kneel_cycle_out")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--cam-width", type=int, default=640)
    ap.add_argument("--cam-height", type=int, default=480)
    ap.add_argument("--magicsim-usd-path", type=str, default=None)
    ap.add_argument("--headless", action="store_true", default=True)
    ap.add_argument("--gui", dest="headless", action="store_false")
    ap.add_argument("--g1-encoder-onnx", type=str,
                    default="sonic_python_inference/assets/g1_encoder_dyn.onnx")
    ap.add_argument("--decoder-onnx", type=str,
                    default="sonic_python_inference/assets/decoder_dyn.onnx")
    ap.add_argument("--planner-onnx", type=str,
                    default="gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx")
    return ap.parse_args()


args = _parse_args()

from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher(headless=args.headless, enable_cameras=True)
simulation_app = app_launcher.app

import torch  # noqa: E402
import imageio.v2 as imageio  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.terrains import TerrainImporterCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from gear_sonic.envs.manager_env.robots.g1 import (  # noqa: E402
    G1_ISAACLAB_TO_MUJOCO_DOF,
    G1_MODEL_12_ACTION_SCALE,
    G1_MUJOCO_TO_ISAACLAB_DOF,
)
from sonic_python_inference.g1_magicsim_cfg import (  # noqa: E402
    DEFAULT_MAGICSIM_USD_PATH, make_g1_magicsim_cfg)
from sonic_python_inference.sonic_inference import (  # noqa: E402
    ALLOWED_PRED_NUM_TOKENS, NUM_JOINTS, PLANNER_CONTEXT_DEFAULT_HEIGHT,
    PLANNER_EVERY_K_POLICY_STEPS, PLANNER_HEIGHT_DEFAULT, PLANNER_MODE_IDLE,
    POLICY_HZ, RESAMPLED_FRAMES, resample_traj_30_to_50hz, slerp_torch)
from sonic_python_inference.sonic_planner_pool import PLANNER_FRAME_DIM, PlannerSessionPool  # noqa: E402
from sonic_python_inference.sonic_g1_inference import G1_NUM_FUTURE_FRAMES, SonicG1Inference  # noqa: E402
from sonic_python_inference.joint_order import g1_body_indices_in_training_order  # noqa: E402


SIM_DT = 0.005
DECIMATION = 4
LEG_NAME_PATTERNS = (r".*_hip_.*_joint", r".*_knee_joint", r".*_ankle_.*_joint")
HAND_RE = re.compile(r".*_hand_.*_joint")
_USD = args.magicsim_usd_path or DEFAULT_MAGICSIM_USD_PATH
_ROBOT_CFG = make_g1_magicsim_cfg(_USD)


@configclass
class SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", collision_group=-1)
    robot = _ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    dome_light = AssetBaseCfg(prim_path="/World/DomeLight",
                              spawn=sim_utils.DomeLightCfg(color=(1.0, 1.0, 1.0), intensity=2000.0))
    camera = CameraCfg(prim_path="/World/RefCam", update_period=0.0,
                       height=args.cam_height, width=args.cam_width, data_types=["rgb"],
                       spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 100.0)))


def _action_scale(names):
    scale = np.ones(len(names), dtype=np.float32)
    pats = [(re.compile(p), v) for p, v in G1_MODEL_12_ACTION_SCALE.items()]
    for i, n in enumerate(names):
        for pat, v in pats:
            if pat.fullmatch(n):
                scale[i] = float(v); break
        else:
            raise RuntimeError(f"no scale for {n}")
    return scale


def _leg_indices(names):
    leg_re = [re.compile(p) for p in LEG_NAME_PATTERNS]
    return [i for i, n in enumerate(names) if any(r.fullmatch(n) for r in leg_re)]


def _build_feeds(ctx_np, mode, height, mvd, fcd):
    return [{
        "context_mujoco_qpos": ctx_np[0:1].astype(np.float32), "target_vel": np.array([0.], np.float32),
        "mode": np.array([mode], np.int64), "movement_direction": mvd.reshape(1, 3).astype(np.float32),
        "facing_direction": fcd.reshape(1, 3).astype(np.float32), "random_seed": np.array([0], np.int64),
        "has_specific_target": np.zeros((1, 1), np.int64), "specific_target_positions": np.zeros((1, 4, 3), np.float32),
        "specific_target_headings": np.zeros((1, 4), np.float32),
        "allowed_pred_num_tokens": ALLOWED_PRED_NUM_TOKENS.reshape(1, 11), "height": np.array([height], np.float32)}]


def _ctx_from_cache(cache, pidx):
    N, L, _ = cache.shape; dev = cache.device
    to = torch.arange(4, device=dev, dtype=torch.float32) * (50.0 / 30.0)
    idxf = pidx.view(-1, 1).float() + to.view(1, -1)
    i0 = idxf.long().clamp(0, L - 1); i1 = (i0 + 1).clamp(0, L - 1)
    a = (idxf - i0.float()).clamp(0, 1); nr = torch.arange(N, device=dev).view(-1, 1).expand(-1, 4)
    f0 = cache[nr, i0]; f1 = cache[nr, i1]
    pos = f0[..., 0:3] + (f1[..., 0:3] - f0[..., 0:3]) * a.unsqueeze(-1)
    q = slerp_torch(f0[..., 3:7], f1[..., 3:7], a)
    j = f0[..., 7:] + (f1[..., 7:] - f0[..., 7:]) * a.unsqueeze(-1)
    return torch.cat([pos, q, j], dim=-1)


def main():
    dev = args.device
    out_dir = Path(args.out_dir); vid_dir = out_dir / "videos"
    out_dir.mkdir(parents=True, exist_ok=True); vid_dir.mkdir(parents=True, exist_ok=True)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=dev, dt=SIM_DT))
    sim.set_camera_view([3.0, 3.0, 2.0], [0.0, 0.0, 0.6])
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=3.0))
    sim.reset(); robot = scene["robot"]; camera = scene["camera"]; scene.update(dt=0.0)
    camera.set_world_poses_from_view(eyes=torch.tensor([[2.4, 2.0, 1.0]], device=camera.device),
                                     targets=torch.tensor([[0.0, 0.0, 0.45]], device=camera.device))

    full_names = list(robot.data.joint_names)
    body_idx, body_names = g1_body_indices_in_training_order(full_names, HAND_RE)
    assert len(body_idx) == NUM_JOINTS
    body_idx_t = torch.as_tensor(body_idx, dtype=torch.long, device=dev)
    leg_idx = _leg_indices(body_names); assert len(leg_idx) == 12
    leg_names = [body_names[i] for i in leg_idx]
    upper_idx = [i for i in range(NUM_JOINTS) if i not in set(leg_idx)]
    leg_idx_t = torch.as_tensor(leg_idx, dtype=torch.long, device=dev)
    upper_idx_t = torch.as_tensor(upper_idx, dtype=torch.long, device=dev)

    default_full = robot.data.default_joint_pos[0].cpu().numpy().astype(np.float32)
    default_body = default_full[body_idx]
    action_scale = _action_scale(body_names)

    il_to_mj = torch.as_tensor(G1_ISAACLAB_TO_MUJOCO_DOF, dtype=torch.long, device=dev)
    mj_to_il = torch.as_tensor(G1_MUJOCO_TO_ISAACLAB_DOF, dtype=torch.long, device=dev)
    leg_mj_slots = mj_to_il[leg_idx_t]

    def partner(n):
        return ("right" + n[4:]) if n.startswith("left") else ("left" + n[5:]) if n.startswith("right") else n
    mir_perm = torch.as_tensor([leg_names.index(partner(n)) for n in leg_names], dtype=torch.long, device=dev)
    mir_sign = torch.as_tensor([-1.0 if ("_roll" in n or "_yaw" in n) else 1.0 for n in leg_names],
                               dtype=torch.float32, device=dev)

    pool = PlannerSessionPool(args.planner_onnx, pool_size=1, device_id=0, serial=True)
    infer = SonicG1Inference(num_envs=1, g1_encoder_onnx=args.g1_encoder_onnx, decoder_onnx=args.decoder_onnx,
                             default_angles=default_body, action_scale=action_scale, device=dev)

    default_body_t = torch.as_tensor(default_body, device=dev).unsqueeze(0)
    default_mj = default_body_t[:, il_to_mj]
    init_frame = torch.zeros(1, PLANNER_FRAME_DIM, device=dev)
    init_frame[:, 2] = PLANNER_CONTEXT_DEFAULT_HEIGHT; init_frame[:, 3] = 1.0; init_frame[:, 7:] = default_mj
    upper_rest = default_body_t[:, upper_idx_t]
    default_full_t = robot.data.default_joint_pos.clone()
    mvd = np.array([1., 0, 0], np.float32); fcd = np.array([1., 0, 0], np.float32)
    fut = torch.arange(0, G1_NUM_FUTURE_FRAMES * 5, 5, dtype=torch.long, device=dev)
    knee_idx = {s: [i for i, n in enumerate(robot.data.body_names) if n.startswith(s) and "knee" in n][0]
                for s in ("left", "right")}

    def reset_stand():
        rs = robot.data.default_root_state.clone(); rs[:, 0:3] += scene.env_origins; rs[:, 7:13] = 0.0
        jp = robot.data.default_joint_pos.clone()
        robot.write_root_state_to_sim(rs); robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
        robot.set_joint_position_target(jp); scene.write_data_to_sim(); scene.update(dt=0.0)
        infer.reset(joint_pos=jp[:, body_idx_t])

    def grab():
        sim.render(); camera.update(dt=SIM_DT * DECIMATION, force_recompute=True)
        rgb = camera.data.output["rgb"][0].detach().cpu().numpy()
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        return np.ascontiguousarray(rgb[..., :3])

    def run_cycle(mirror):
        reset_stand()
        pctx = init_frame.unsqueeze(1).expand(1, 4, -1).clone()
        cache = torch.zeros(1, RESAMPLED_FRAMES, PLANNER_FRAME_DIM, device=dev)
        pidx = torch.zeros(1, dtype=torch.long, device=dev)
        kneel_steps = args.down_steps + args.hold_steps
        total = kneel_steps + args.stand_steps
        frames, steps = [], []
        min_knee_z = {"left": 1e9, "right": 1e9}
        for t in range(total):
            in_kneel = t < kneel_steps
            mode = PLANNER_MODE_KNEEL_ONE_LEG if in_kneel else PLANNER_MODE_IDLE
            height = args.kneel_height if in_kneel else PLANNER_HEIGHT_DEFAULT
            phase = "descend" if t < args.down_steps else ("hold" if in_kneel else "stand")
            jp_il = robot.data.joint_pos[:, body_idx_t].clone()
            jv_il = robot.data.joint_vel[:, body_idx_t].clone()
            rq = robot.data.root_quat_w.clone()
            bav = robot.data.root_ang_vel_b.clone(); gb = robot.data.projected_gravity_b.clone()

            if t % PLANNER_EVERY_K_POLICY_STEPS == 0:
                if t > 0:
                    pctx = _ctx_from_cache(cache, pidx)
                traj_np, _ = pool.run_batched(_build_feeds(pctx.cpu().numpy(), mode, height, mvd, fcd))
                cache[:] = resample_traj_30_to_50hz(torch.as_tensor(traj_np, device=dev, dtype=torch.float32),
                                                    RESAMPLED_FRAMES)
                pidx.zero_()
            idx = (pidx.view(-1, 1) + fut.view(1, -1)).clamp(max=RESAMPLED_FRAMES - 1)
            idxn = (idx + 1).clamp(max=RESAMPLED_FRAMES - 1)
            nr = torch.zeros(1, G1_NUM_FUTURE_FRAMES, dtype=torch.long, device=dev)
            fr = cache[nr, idx]; frn = cache[nr, idxn]
            leg_pos = fr[..., 7:][..., leg_mj_slots]
            leg_vel = (frn[..., 7:][..., leg_mj_slots] - leg_pos) * POLICY_HZ
            rqf = fr[..., 3:7]
            if mirror:
                leg_pos = mir_sign * leg_pos[..., mir_perm]
                leg_vel = mir_sign * leg_vel[..., mir_perm]
                rqf = torch.stack([rqf[..., 0], -rqf[..., 1], rqf[..., 2], -rqf[..., 3]], dim=-1)
            jpf = torch.zeros(1, G1_NUM_FUTURE_FRAMES, NUM_JOINTS, dtype=torch.float32, device=dev)
            jvf = torch.zeros_like(jpf)
            jpf[..., leg_idx_t] = leg_pos; jvf[..., leg_idx_t] = leg_vel
            jpf[..., upper_idx_t] = upper_rest.unsqueeze(1).expand(-1, G1_NUM_FUTURE_FRAMES, -1)
            tgt_body = infer.step(joint_pos_future=jpf, joint_vel_future=jvf, ref_root_quat_future_wxyz=rqf,
                                  joint_pos=jp_il, joint_vel=jv_il, base_ang_vel=bav, gravity_in_base=gb,
                                  root_quat_wxyz=rq)
            tgt_full = default_full_t.clone(); tgt_full[:, body_idx_t] = tgt_body
            for _ in range(DECIMATION):
                robot.set_joint_position_target(tgt_full); scene.write_data_to_sim()
                sim.step(render=False); scene.update(dt=SIM_DT)
            pidx = (pidx + 1).clamp(max=RESAMPLED_FRAMES - 1)

            # record achieved leg trajectory this step
            ach_body = robot.data.joint_pos[0, body_idx_t].cpu().numpy()
            rpz = float((robot.data.root_pos_w[0, 2] - scene.env_origins[0, 2]).item())
            rqw = robot.data.root_quat_w[0].cpu().numpy()
            steps.append({"step": t, "t_s": round(t / POLICY_HZ, 3),
                          "phase": phase, "pelvis_z_m": round(rpz, 5),
                          "root_quat_wxyz": [round(float(x), 6) for x in rqw],
                          "leg_joint_pos_rad": [round(float(ach_body[i]), 6) for i in leg_idx]})
            bz = robot.data.body_pos_w[0, :, 2]
            for s in ("left", "right"):
                min_knee_z[s] = min(min_knee_z[s], float(bz[knee_idx[s]]))
            frames.append(grab())
        side = "left" if min_knee_z["left"] < min_knee_z["right"] else "right"
        vid = vid_dir / f"kneel_cycle_{side}.mp4"
        imageio.mimsave(str(vid), frames, fps=args.fps, macro_block_size=None)
        return steps, side, str(vid.relative_to(out_dir))

    sides = ["right", "left"] if args.side == "both" else [args.side]
    for want in sides:
        steps, det, vid = run_cycle(mirror=(want == "left"))
        z0 = steps[0]["pelvis_z_m"]; zmin = min(s["pelvis_z_m"] for s in steps); zend = steps[-1]["pelvis_z_m"]
        tag = "ok" if det == want else f"WRONG-SIDE({det})"
        print(f"[run] side={want} grounded={det} [{tag}] steps={len(steps)} "
              f"z: start={z0:.3f} min={zmin:.3f} end={zend:.3f}", flush=True)
        doc = {"mode": "kneelOneLeg_cycle", "grounded_knee": want, "with_tracking": True,
               "dof_order": "isaaclab", "policy_hz": POLICY_HZ, "leg_joint_names": leg_names,
               "phases": {"down_steps": args.down_steps, "hold_steps": args.hold_steps,
                          "stand_steps": args.stand_steps},
               "note": ("Full kneel-down->hold->stand-up cycle, closed-loop SONIC tracking. "
                        "Per-step ACHIEVED leg trajectory (12 DOF, IsaacLab order). Legs only. "
                        + ("Left = sagittal mirror." if want == "left" else "Right = native planner.")),
               "steps": steps}
        (out_dir / f"kneel_cycle_{want}.json").write_text(json.dumps(doc, indent=2))
        print(f"[out] {out_dir}/kneel_cycle_{want}.json ({len(steps)} steps) + {vid}")

    pool.close(); simulation_app.close()


if __name__ == "__main__":
    main()
