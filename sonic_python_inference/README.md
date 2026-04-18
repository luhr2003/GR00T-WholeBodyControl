# SONIC Python Inference (Isaac Lab)

Pure-Python inference pipeline for SONIC in Isaac Lab. Runs N envs in parallel
from a single `sonic_release/last.pt` checkpoint, closed-loop through Isaac
Lab physics — no DDS, no C++, no real-robot deploy.

Three encoder paths are supported against the **shared** decoder
(`decoder_dyn.onnx`, `[N, 994] → [N, 29]`):

| Pipeline | Encoder (ONNX) | Command source | Needs on disk | Eval entry point |
|----------|----------------|----------------|---------------|------------------|
| **VR 3pt** teleop | `encoder_dyn.onnx` (`[N, 267] → [N, 64]`) | kplanner + 3-point sparse targets | `planner_sonic.onnx` | `stage_vr3pt_eval.py` |
| **SMPL tracking** | `smpl_encoder_dyn.onnx` (`[N, 840] → [N, 64]`) | per-frame SMPL joints + root quat + wrist DOF | SMPL pkl + robot pkl | `stage_smpl_eval.py` |
| **G1 (teacher) tracking** | `g1_encoder_dyn.onnx` (`[N, 640] → [N, 64]`) | retargeted robot DOF + DOF velocity + root anchor 6D | robot pkl **only** | `stage_g1_eval.py` |
| **Hybrid (planner + Pink IK) → G1** | `g1_encoder_dyn.onnx` | kplanner lower 12 + Pink IK upper 17 (padded); no pkl | `planner_sonic.onnx` + URDF | `stage_hybrid_eval.py` |

G1 is the training-time teacher; SMPL and teleop distill to it. Running G1 as
inference is the cleanest sanity baseline — if G1 can't track but SMPL can,
the bug is in SMPL obs construction; if SMPL can track but VR 3pt can't, the
bug is in the planner wiring. Dex3 fingers are out of scope.

---

## Layout

```
sonic_python_inference/
├── pyproject.toml                    # uv deps (Isaac Lab 2.3.2, torch cu128, onnxruntime-gpu, trl 0.28)
├── sonic_inference.py                # VR 3pt: SonicVR3PTInference (encoder + decoder + planner)
├── sonic_planner_pool.py             # Concurrent ORT session pool (batch=1 planner × N streams)
├── sonic_smpl_inference.py           # SMPL:    SonicSMPLInference
├── sonic_g1_inference.py             # G1:      SonicG1Inference (teacher)
├── sonic_smpl_motion_lib.py          # Shared motion lib (SMPL + robot pkl loading, future-frame sampling)
├── pink_ik_driver.py                 # N parallel Pink IK solvers (IsaacLab PinkIKController wrapper)
├── g1_pink_ik_cfg.py                 # G1 Pink IK config (task recipe + pelvis-frame rest poses)
├── assets/
│   ├── encoder_dyn.onnx              # GENERATED (Export), dynamic batch — teleop
│   ├── smpl_encoder_dyn.onnx         # GENERATED (Export), dynamic batch — SMPL
│   ├── g1_encoder_dyn.onnx           # GENERATED (Export), dynamic batch — G1
│   ├── decoder_dyn.onnx              # GENERATED (Export), dynamic batch — shared
│   └── g1_pink_ik.urdf               # G1 URDF with visual/collision stripped (Pink IK kinematics only)
└── scripts/
    ├── export_dynamic_batch_onnx.py  # Export: rebuild all 3 encoders + decoder from last.pt
    ├── stage_vr3pt_eval.py           # VR 3pt eval
    ├── stage_vr3pt_cube_eval.py      # VR 3pt eval: IDLE@0, two VisualCuboids drive L/R wrist targets (torso frozen)
    ├── stage_smpl_eval.py            # SMPL eval
    ├── stage_g1_eval.py              # G1 (teacher) eval
    ├── stage_hybrid_eval.py          # Hybrid (planner + Pink IK) → G1 encoder
    ├── stage_hybrid_planner_only.py  # Fixed-base debug: drive PD directly with hybrid-assembled target
    ├── stage_hybrid_ik.py            # Interactive two-cube Pink IK demo (CPU sim, IDLE@0, VisualCuboid targets)
    └── stage_hybrid_kneel.py         # Hybrid, planner mode=kneelOneLeg, auto-lowered rest pose
```

The VR 3pt **planner** ONNX stays batch=1 and is loaded from its original
location at `gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx`. It
can't be re-exported with a dynamic batch dim (the graph has 486 `Reshape`
nodes whose shapes come from `Concat` over ~612 `Constant(1)` tensors —
batch=1 is a structural invariant). We work around it with
`PlannerSessionPool`: N independent ORT sessions, each on its own CUDA stream,
dispatched in parallel via `ThreadPoolExecutor`.

---

## Setup

Run everything from the repo root `GR00T-WholeBodyControl/`.

### 1. One-shot install (`.venv_isaac`, Python 3.11)

All indices, pins, and the workspace member are declared in the root
`pyproject.toml`. A single `uv sync` populates `.venv_isaac` (Python 3.11)
with Isaac Lab 2.3.2 (from NVIDIA index), torch cu128, ONNX runtime, and
editable `sonic_python_inference` + `gear_sonic`:

```bash
uv sync
```

This is the env used by all three eval pipelines below. `uv` auto-creates
`.venv_isaac` if it doesn't exist; subsequent syncs are incremental.

### 2. Auto-activate with direnv

`.envrc` exports `UV_PROJECT_ENVIRONMENT=.venv_isaac`, activates the venv,
and sources `gear_sonic_deploy/scripts/setup_env.sh` on `cd` into the repo:

```bash
direnv allow
```

Without direnv: `source .venv_isaac/bin/activate` manually each shell.

### 3. (Optional)Separate env for MuJoCo sim side (`.venv_sim`, Python 3.10)

The `gear_sonic[sim]` / `[teleop]` stack (MuJoCo `run_sim_loop.py`, pinocchio
teleop, pyvista viz) uses a **separate** Python-3.10 venv because pinocchio's
wheels are 3.10-only. It is NOT managed by the root `uv sync` above. Set it
up once with:

```bash
uv venv .venv_sim --python 3.10
source .venv_sim/bin/activate
uv pip install -e "gear_sonic[sim,teleop]"
```

The three SONIC eval pipelines in this README only need `.venv_isaac`. Keep
`.venv_sim` around only if you also run MuJoCo sim / ZMQ teleop.

### 4. Isaac Sim EULA

The first-ever `isaacsim` / `isaaclab` launch prompts for EULA. Accept it once
interactively, then subsequent runs are non-interactive.

### 5. Checkpoint + sample data

Required:
- `sonic_release/last.pt` + `sonic_release/config.yaml` — the policy checkpoint
- `sample_data/robot_filtered/` — retargeted robot trajectories (used by SMPL + G1 evals)
- `sample_data/smpl_filtered/` — SMPL pkls (used by SMPL eval)

Pull with two separate commands:

```bash
python download_from_hf.py --training --no-smpl   # checkpoint only (~469 MB)
python download_from_hf.py --sample               # sample_data/ (robot + smpl pkls, small)
```

Note on `--no-smpl`: the `--training` flag bundles a ~30 GB `bones_seed_smpl/`
**training** dataset (extracted to `data/smpl_filtered/`). That's only needed
if you're re-training the policy. For inference / eval, always pass
`--no-smpl` — the SMPL eval pipeline reads the small
`sample_data/smpl_filtered/` pkls that come from `--sample`, not the full
training bundle.

---

## Export ONNX (one-time per checkpoint)

```bash
python -m sonic_python_inference.scripts.export_dynamic_batch_onnx \
    --checkpoint sonic_release/last.pt \
    --out-dir    sonic_python_inference/assets
```

Outputs (all with dynamic batch on dim 0):

| File | Role | Shape |
|------|------|-------|
| `encoder_dyn.onnx`     | teleop encoder + FSQ         | `[N, 267] → [N, 64]` |
| `smpl_encoder_dyn.onnx`| SMPL encoder + FSQ           | `[N, 840] → [N, 64]` |
| `g1_encoder_dyn.onnx`  | G1 (teacher) encoder + FSQ   | `[N, 640] → [N, 64]` |
| `decoder_dyn.onnx`     | g1_dyn decoder (shared)      | `[N, 994] → [N, 29]` |

The script validates `dim[0] == "batch"` and runs each model at N=1, 4, 64 to
confirm the dynamic axis propagates correctly.

---

## Pipeline 1 — VR 3pt (teleop, `stage_vr3pt_eval.py`)

Closed-loop locomotion with a kplanner driving lower-body targets + frozen
3-point VR sparse targets. Verifies **planner → encoder → decoder → Isaac Lab
→ locomotion**. The VR 3pt targets are frozen at reset, so this pipeline does
**not** exercise upper-body tracking on its own.

### Run

```bash
python -m sonic_python_inference.scripts.stage_vr3pt_eval --num-envs 4
```

Add `--headless` to suppress the viewer. Runs indefinitely — Ctrl+C or close
the Isaac Lab window to stop.

All envs share the same frozen command: SLOW_WALK forward at 0.3 m/s, facing
`[1, 0, 0]`, VR 3pt targets captured from the default pose.

### CLI flags

| flag | default | meaning |
|------|---------|---------|
| `--num-envs` | `4` | parallel envs |
| `--headless` | off | no viewer |
| `--encoder-onnx` | `sonic_python_inference/assets/encoder_dyn.onnx` | |
| `--decoder-onnx` | `sonic_python_inference/assets/decoder_dyn.onnx` | |
| `--planner-onnx` | `gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx` | batch=1, pooled |

### Passing criteria

- all envs: no `fallen=True` (root z stays > 0.4 m)
- all envs: `travelled` grows at ~0.3 m/s (per-tick log every 1 s)
- visually: robot walks forward steadily without drifting or spinning

If any env falls, it's a wiring bug — do not move on to the tracking
pipelines below until VR 3pt is clean.

### Interactive cube variant (`stage_vr3pt_cube_eval.py`)

Same `SonicVR3PTInference` pipeline, but pinned to planner **IDLE @ 0 m/s**
(stand in place), with two `isaacsim.core.api.objects.VisualCuboid` spawned
at the reset-time wrist targets:

- red  cube → right wrist target
- blue cube → left  wrist target
- torso/head target stays frozen at reset (same as base script)

Each tick the cubes' world poses are read via `get_world_pose()`,
transformed into root-local frame (mirror of C++ `GatherVR3PointPosition`
after the body-offset step), and spliced into the VR 3pt slots as
`[left_wrist, right_wrist, torso]`. No Pink IK — the encoder eats the cube
targets directly.

```bash
python -m sonic_python_inference.scripts.stage_vr3pt_cube_eval --num-envs 1
```

### Locomotion modes & speed bands

| id | name        | speed band    | notes |
|----|-------------|---------------|-------|
| 0  | `IDLE`      | 0.0 m/s       | stand in place; `movement_direction` ignored |
| 1  | `SLOW_WALK` | 0.1 – 0.8 m/s | natural walking cadence |
| 2  | `WALK`      | 0.8 – 2.5 m/s | fast walk / jog |
| 3  | `RUN`       | 2.5 – 7.5 m/s | full-speed run |

Source: `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/localmotion_kplanner.hpp:78-82`.
Exported as `PLANNER_MODE_IDLE / _SLOW_WALK / _WALK / _RUN` from
`sonic_python_inference/sonic_inference.py`. Higher-id idle/special poses
(SQUAT, KNEEL, CRAWLING, …) exist in the planner but are out of scope here.

---

## Pipeline 2 — SMPL tracking (`stage_smpl_eval.py`)

Drives the robot through the SMPL encoder: each env replays a retargeted SMPL
motion, the SMPL encoder sees 10 future frames @ 20 ms (`frame_skip=1` at
target_fps=50):

```
per-frame  [ smpl_joints_local(72) | root_ori_6d(6) | wrist_dof(6) ] = 84
flatten 10 frames                                                   → 840
```

- `smpl_joints_local`: each future frame's 24 SMPL joints rotated into
  **that frame's own** SMPL root quat (per-frame canonicalization, not the
  first frame's). See `observations.py:1716`.
- `root_ori_6d`: 6D of `quat_mul(quat_inv(robot_anchor_quat_w), smpl_root_quat_w)`
  per frame. The SMPL root quat goes through Y→Z-up (`+90° about X`) and
  `remove_smpl_base_rot([0.5,-0.5,-0.5,-0.5])`. See `commands.py:1343`.
- `wrist_dof`: retargeted robot DOF at IL indices `[23..28]` per frame.

### Run

```bash
python -m sonic_python_inference.scripts.stage_smpl_eval \
    --motion walk_forward_amateur_001__A001 --num-envs 4
```

Needs both `sample_data/smpl_filtered/{motion}.pkl` and
`sample_data/robot_filtered/.../{motion}.pkl`. Frame 0 of the robot pkl seeds
the initial pose (root pos/quat + 29-DOF). Runs for one pass of the motion
(`max_step - 1` policy ticks), then prints a summary and exits. Ctrl+C or
close the window to stop early.

### CLI flags

| flag | default | meaning |
|------|---------|---------|
| `--motion` | `walk_forward_amateur_001__A001` | basename shared between SMPL and robot pkls |
| `--num-envs` | `4` | parallel envs (all replay the same motion) |
| `--headless` | off | no viewer |
| `--smpl-dir` | `sample_data/smpl_filtered` | |
| `--robot-dir` | `sample_data/robot_filtered` | |
| `--smpl-encoder-onnx` | `sonic_python_inference/assets/smpl_encoder_dyn.onnx` | |
| `--decoder-onnx` | `sonic_python_inference/assets/decoder_dyn.onnx` | |

### Passing criteria

- no `fallen=True` (root z > 0.4 m) within the first ~5 s on walk clips
- `mean_joint_mae` stays bounded (< ~0.2 rad) over the clip
- visually: robot tracks the reference pose without drifting / spinning

---

## Pipeline 3 — G1 teacher tracking (`stage_g1_eval.py`)

Drives the robot through the **G1 encoder** — the training-time teacher. Only
needs `sample_data/robot_filtered/.../{motion}.pkl` (no SMPL pkl). Obs sees
10 future frames @ 100 ms (`frame_skip=5` at target_fps=50):

```
sources (frame-major flat [N, F*29]):
    jp_flat = joint_pos_future.reshape(N, -1)   # [pos_f0, pos_f1, …, pos_f9]
    jv_flat = joint_vel_future.reshape(N, -1)   # [vel_f0, vel_f1, …, vel_f9]  (finite-diff)
training layout (commands.py:897-903 + observations.py:584-587):
    cmd_flat    = cat([jp_flat, jv_flat], dim=1)          # [N, 2*F*29] = [N, 580]
    cmd_nonflat = cmd_flat.reshape(N, F, -1)              # [N, F, 58]
    anchor_6d   = 6D( quat_mul(quat_inv(robot_anchor), ref_root_quat_future) )   # [N, F, 6]
    slot        = cat([cmd_nonflat, anchor_6d], dim=-1)   # [N, F, 64]
    flat        = slot.reshape(N, -1)                     # [N, 640]
```

**Do not** "fix" this to a semantic `[pos_fi, vel_fi, ori_fi]` per-frame
layout. The training reshape crosses the pos/vel boundary — slot 0 holds
`[pos_f0(29), pos_f1(29)]`, slot 5 holds `[vel_f0(29), vel_f1(29)]` — and the
MLP was trained on that exact memory layout. `sonic_g1_inference.py`
reproduces it bit-for-bit.

The reference root quat comes straight from the robot pkl's `root_rot`
(xyzw→wxyz), **without** the SMPL Y→Z / base-rot preprocessing.

### Run

```bash
python -m sonic_python_inference.scripts.stage_g1_eval \
    --motion walk_forward_amateur_001__A001 --num-envs 4
```

Runs for one pass of the motion (`max_step - 1` policy ticks), then prints a
summary and exits. Ctrl+C or close the window to stop early.

### CLI flags

| flag | default | meaning |
|------|---------|---------|
| `--motion` | `walk_forward_amateur_001__A001` | basename under `--robot-dir` |
| `--num-envs` | `4` | parallel envs |
| `--headless` | off | no viewer |
| `--robot-dir` | `sample_data/robot_filtered` | |
| `--g1-encoder-onnx` | `sonic_python_inference/assets/g1_encoder_dyn.onnx` | |
| `--decoder-onnx` | `sonic_python_inference/assets/decoder_dyn.onnx` | |

### Passing criteria

- `action_scale_il min/max/mean` matches the SMPL eval (same regex resolver)
- no `fallen=True` within first ~5 s on walk clip
- `mean_joint_mae` **≤** SMPL eval on the same motion (G1 is the teacher)

---

## Pipeline 4 — Hybrid (kplanner + Pink IK) → G1 teacher (`stage_hybrid_eval.py`)

Replaces the robot pkl dependency of Pipeline 3 with two live upstream
signals, both evaluated each policy tick:

```
planner (IDLE/WALK mode) [N,108,36] MJ ──► lower 12 @ [0,5,…,45] → MJ→IL → [N,10,12]
Pink IK (wrist pelvis-frame targets) ───► upper 17 IL [N,17]    → pad  → [N,10,17]
                                                                        │
                                                     splice → joint_pos_future [N,10,29] IL
                                                                        │
ref_root_quat_future ← planner cache root quat @ [0,5,…,45]             ▼
                                                  SonicG1Inference (g1_encoder → decoder)
```

Lower 12 (legs) are sampled from the planner cache in MJ order at the
10-frame future window (step=5 at 50 Hz = 100 ms spacing, matching the G1
encoder's `dt_future_ref_frames = 0.1`) and gathered into IL order via
`G1_MUJOCO_TO_ISAACLAB_DOF` (the argsort-inverse of IL→MJ — see
`stage0_planner_only.py:89-98` for semantics). Upper 17 (3 waist + 14 arms)
come from one Pink IK solve per tick, broadcast across all 10 future frames
with `joint_vel = 0`. Wrist targets are **pelvis_contour_link (torso) frame**
— v1 freezes them at the hand-chosen rest poses in `g1_pink_ik_cfg.py`.

`ref_root_quat_future` stays **world-frame**, taken straight from the
planner cache's quat slice. The G1 encoder internally does
`quat_inv(robot_anchor) * ref_root_quat_future → 6D` per frame.

Self-contained — no magicsim import, no robot pkl. Pink IK uses IsaacLab's
`PinkIKController` directly, backed by `pin` + `pin-pink`, and a mesh-free
URDF at `assets/g1_pink_ik.urdf` (generated by stripping all `<visual>` and
`<collision>` blocks from `gear_sonic/data/assets/robot_description/urdf/g1/main.urdf`).
A no-op `set_target_from_configuration` is monkey-patched onto the
`DampingTask` instance because pink 4.1.0 doesn't implement it and IsaacLab's
init sweep expects it on every `variable_input_task`.

### Run

```bash
# Full hybrid (free-floating, 4 envs)
python -m sonic_python_inference.scripts.stage_hybrid_eval --num-envs 4 --headless

# Debug: fixed-base, no encoder/decoder — drive PD directly with the hybrid
# target. Isolates assembly/joint-order bugs from encoder OOD.
python -m sonic_python_inference.scripts.stage_hybrid_planner_only --num-envs 1

# Kneel (planner mode 5/6): two VisualCuboid targets (red=right, blue=left)
# spawn at the standing rest world pose. Each tick the script rewrites their
# world Z to `pelvis_z + rest_z_pelvis_scaled` so the cubes drop with the
# pelvis as the robot kneels (user keeps dragging X/Y).
#
#   --kneel-mode one_leg  # mode 6, kneelOneLeg (flip --random-seed for side)
#   --kneel-mode two_leg  # mode 5, kneelTwoLeg
#
python -m sonic_python_inference.scripts.stage_hybrid_kneel --kneel-mode two_leg --kneel-height 0.2

# Interactive IK demo: CPU sim, planner IDLE@0.0 m/s, full SONIC closed-loop.
# Two cubes (red=right, blue=left) spawn at the pelvis-frame rest poses via
# `isaacsim.core.api.objects.VisualCuboid` (no collider, no rigid body — pure
# Xform with a cube geom). Drag them in the viewer to move the hands; poses
# are read per-tick with `get_world_pose()` and transformed world→pelvis
# before Pink IK.
python -m sonic_python_inference.scripts.stage_hybrid_ik
```

### CLI flags (shared across the three scripts)

| flag | default | meaning |
|------|---------|---------|
| `--num-envs` | `4` (`1` for `ik`/`planner_only`) | parallel envs |
| `--headless` | off | no viewer |
| `--episode-sec` | `10.0` (`600.0` for `ik`) | run length |
| `--target-vel` | `0.3` | m/s (SLOW_WALK); `ik` fixes to 0 |
| `--planner-onnx` | `gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx` | batch=1, pooled |
| `--urdf-path` | `sonic_python_inference/assets/g1_pink_ik.urdf` | mesh-free kinematics URDF |
| `--g1-encoder-onnx` / `--decoder-onnx` | (see Pipeline 3) | |

### Passing criteria

This is a **sanity experiment**, not a production path. The G1 encoder was
trained on smooth retargeted robot pkls, so held-constant upper + planner
lower is OOD. Treat `stage_hybrid_eval.py` as diagnostic.

- `stage_hybrid_planner_only` (fixed base): legs visibly walk at 0.3 m/s,
  upper stays near rest pose, `tracking_mae < 0.05 rad`.
- `stage_hybrid_ik` (interactive, CPU physics, planner IDLE @ 0 m/s): when a
  cube is left at its init pose the hand stays at the rest pose; dragging
  the cube by ≤0.3 m in any direction tracks without the wrist posting to a
  joint limit. Cubes are `VisualCuboid` — no collider, no rigid dynamics —
  so moving one doesn't kick the robot.
- `stage_hybrid_eval` (closed-loop): if it falls within 5 s, first run
  `stage_g1_eval.py` on the same initial pose — if that tracks, the failure
  is specifically the hybrid upper-body OOD, not the decoder/proprio path.

---

## Rates (fixed — cannot be tuned without retraining)

| Layer                  | Hz  | How                             |
|------------------------|-----|---------------------------------|
| Isaac Lab physics      | 200 | `sim.dt = 0.005`                |
| Policy (encoder+decoder) | 50 | `decimation = 4`                |
| Planner ONNX call (VR 3pt only) | 10  | every 5th policy step    |
| Planner native output (VR 3pt)  | 30  | linear + slerp → resample to 50 |
| G1 future window       | 10 Hz | 10 frames × dt=0.1 s         |
| SMPL future window     | 50 Hz | 10 frames × dt=0.02 s        |

---

## Proprioception (shared by all three pipelines, 930D)

Proprio buffer layout matches training's `PolicyCfg` field declaration order
(`observations.py:107-128`), not YAML dict order:

```
[ his_ang_vel(30) | his_jp_rel(290) | his_jv(290) | his_last_action(290) | his_gravity(30) ]
```

Ring-buffer length = 10. `his_jp_rel = joint_pos - default_angles`.
`his_last_action` is rolled AFTER the decoder (so next tick sees it as
`action_{t-1}`), matching training's `ObservationManager` semantics.

Decoder input = `cat([token_flat(64), proprio(930)], -1)` → 994.

---

## Programmatic use (VR 3pt)

```python
from sonic_python_inference.sonic_inference import SonicVR3PTInference
from gear_sonic.envs.manager_env.robots.g1 import G1_ISAACLAB_TO_MUJOCO_DOF

infer = SonicVR3PTInference(
    num_envs=4,
    encoder_onnx="sonic_python_inference/assets/encoder_dyn.onnx",
    decoder_onnx="sonic_python_inference/assets/decoder_dyn.onnx",
    planner_onnx="gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx",
    default_angles=default_angles_il,
    action_scale=action_scale_il,
    isaaclab_to_mujoco_dof=G1_ISAACLAB_TO_MUJOCO_DOF,
    device="cuda",
)
infer.reset(joint_pos=q_il, root_pos=root_pos, root_quat_wxyz=root_quat)

# Per 50 Hz policy tick:
motor_targets_il = infer.step(
    vr_3pt_position=..., vr_3pt_orientation=...,
    mode=..., movement_direction=..., facing_direction=..., target_vel=..., height=...,
    joint_pos=..., joint_vel=...,
    base_ang_vel=..., gravity_in_base=...,
    root_pos=..., root_quat_wxyz=...,
)
```

SMPL and G1 variants follow the same pattern with `SonicSMPLInference` /
`SonicG1Inference` and future-frame tensors from
`SmplMotionLib.sample_future` / `sample_future_robot` respectively; see the
eval scripts for worked examples.

### DOF order

| Tensor | Order |
|--------|-------|
| All `step()` / `reset()` args and returns | IsaacLab |
| `default_angles`, `action_scale` | IsaacLab |
| Planner `context_mujoco_qpos` + `[N, 64, 36]` output (VR 3pt only) | MuJoCo |
| Encoder lower-body slice (first 12 of MJ joints, VR 3pt only) | MuJoCo |

Conversion uses `G1_ISAACLAB_TO_MUJOCO_DOF` inside `SonicVR3PTInference`. SMPL
and G1 paths stay in IsaacLab order end-to-end (the motion lib permutes the
pkl's MuJoCo DOFs at load time).

---

## Appendix: VR 3pt dataflow (one 50 Hz tick)

```
Isaac Lab robot.data                        (frozen-at-reset VR 3pt targets)
 joint_pos/vel [N,29] IL                     vr_3pt_position [N,9]
 root_ang_vel_b, gravity_b [N,3]             vr_3pt_orientation [N,12]
 root_quat_w [N,4]                           mode / move_dir / face_dir /
            │                                target_vel / height
            ▼
  ring-buffer roll                                    │
                                                      │
  planner every 5 ticks ────────────────────────────▶ │
    context [N,4,36] MJ → PlannerSessionPool(×N)     │
                         ↓                            │
                       [N, 64, 36] @ 30 Hz MJ         │
                         ↓ resample 30→50             │
                    planner_cache [N,108,36]          │
                         ↓ sample 10 frames step=5    │
            lower_pos(120)  lower_vel(120)  anchor_6d(6)
                         ↓                            │
      teleop_obs [N,267] ──► encoder_dyn.onnx ──► token [N,64]
                                                      │
      proprio [N,930] ──┐                             │
                        └──► decoder_input [N,994] ──► decoder_dyn.onnx
                                                      │
                                                      ▼
                                            action [N,29] IL
                                                      │
                                                      ▼
                target_il = default_angles + action_scale * action
                                                      │
                                                      ▼
                        robot.set_joint_position_target(target_il)
                           × 4 substeps @ 200 Hz
```

The SMPL and G1 pipelines replace the `planner → teleop_obs` branch with
`motion_lib.sample_future[_robot] → smpl_obs | g1_obs`. Everything downstream
of the encoder (proprio, decoder, action scaling, sim step) is identical.

---

## Out of scope

- **VR 3pt tracking stage** (per-env varying hand targets + wrist error metric).
- **N=1 numerical parity vs the C++ MuJoCo deploy.**
- **Dex3 fingers** as an independent controller.
