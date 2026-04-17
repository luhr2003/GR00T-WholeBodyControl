# SONIC VR 3PT Python Inference (Isaac Lab)

Pure-Python inference pipeline for SONIC VR-3PT teleop in Isaac Lab. Runs N envs
in parallel from a single checkpoint, closed-loop through Isaac Lab physics —
no DDS, no C++, no real-robot deploy.

**Scope (this phase):** G1 29-DOF body only, encoder `mode=teleop` (id=1).
Dex3 fingers and the SMPL / G1 motion-dataset encoders are out of scope (the
C++ deploy bypasses them for VR 3pt, so we do too).

---

## Layout

```
sonic_python_inference/
├── pyproject.toml                    # uv deps (Isaac Lab 2.3.2, torch cu128, onnxruntime-gpu, trl 0.28)
├── sonic_inference.py                # SonicVR3PTInference — encoder+decoder+planner wiring
├── sonic_planner_pool.py             # Concurrent ORT session pool (batch=1 planner × N streams)
├── assets/
│   ├── encoder_dyn.onnx              # GENERATED (Step 1), dynamic batch
│   └── decoder_dyn.onnx              # GENERATED (Step 1), dynamic batch
└── scripts/
    ├── export_dynamic_batch_onnx.py  # Step 1: rebuild encoder+decoder from sonic_release/last.pt
    └── stage1_loco_only.py           # Step 3: loco-only closed-loop smoke test in Isaac Lab
```

The **planner** ONNX stays batch=1 and is loaded from its original location
under `gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx`. It can't be
re-exported with a dynamic batch dim (the graph has 486 `Reshape` nodes whose
shapes come from `Concat` over ~612 `Constant(1)` tensors — batch=1 is a
structural invariant, not a dim label). We work around it with
`PlannerSessionPool`: N independent ORT sessions, each on its own CUDA stream,
dispatched in parallel via `ThreadPoolExecutor`.

---

## Setup

Run everything from the repo root `GR00T-WholeBodyControl/`.

### 1. Python env (uv, 3.11)

```bash
uv venv .venv_isaac --python 3.11
source .venv_isaac/bin/activate
```

### 2. Install deps

`pyproject.toml` declares the pinned versions. `uv pip install -e .` does NOT
honor `[tool.uv.sources]`, so you must pass the indices explicitly on the
first install:

```bash
uv pip install -e ./sonic_python_inference \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.nvidia.com \
    --index-strategy unsafe-best-match \
    --prerelease=allow
```

You also need `gear_sonic` itself on the path (for the G1 articulation cfg and
DOF mappings used by the Stage 1 runner):

```bash
uv pip install -e ./gear_sonic[training]
```

### 3. Isaac Sim EULA

The first-ever `isaacsim` / `isaaclab` launch prompts for EULA. Accept it once
interactively, then subsequent runs are non-interactive.

### 4. Checkpoint

Required: `sonic_release/last.pt` (+ `config.yaml`). If you haven't pulled it:

```bash
python download_from_hf.py --training --no-smpl
```

---

## One-time: export ONNX with dynamic batch

```bash
python -m sonic_python_inference.scripts.export_dynamic_batch_onnx \
    --checkpoint sonic_release/last.pt \
    --out-dir    sonic_python_inference/assets
```

Outputs:

- `sonic_python_inference/assets/encoder_dyn.onnx` — teleop encoder + FSQ, `[N,267] → [N,64]`
- `sonic_python_inference/assets/decoder_dyn.onnx` — g1_dyn decoder, `[N,994] → [N,29]`

Script validates `dim[0] == "batch"` and runs each model at N=1, 4, 64 to make
sure the dynamic axis actually propagates.

---

## Stage 1 — loco-only closed-loop smoke test

Verifies the movement / facing / speed / mode command path:
**planner → encoder → decoder → Isaac Lab → locomotion**. VR 3pt is frozen at
the root-local value read from the default pose at reset, so it cannot drive
the upper body.

### Run

```bash
python -m sonic_python_inference.scripts.stage1_loco_only \
    --num-envs 4 --episode-sec 10 --headless
```

Drop `--headless` to see the Isaac Sim viewer. Default N=4; don't change it
without editing the presets below.

### Per-env command presets

Each of the 4 envs runs a different fixed schedule:

| env | mode | movement  | facing                       | target_vel | expected outcome       |
|-----|------|-----------|------------------------------|------------|------------------------|
| 0   | WALK | `[0,0,0]` | `[1, 0, 0]`                  | 0.0        | stand, no fall         |
| 1   | WALK | `[1,0,0]` | `[1, 0, 0]`                  | 1.0        | walk forward >3 m/10s  |
| 2   | WALK | `[0,0,0]` | `[cos π/4, sin π/4, 0]`      | 0.0        | turn in place to 45°   |
| 3   | RUN  | `[1,0,0]` | `[1, 0, 0]`                  | 3.0        | run forward >8 m/10s   |

### What it prints

Every 1 s of sim time:
```
[t= 1.00s] travelled=[0.01, 0.42, 0.03, 0.81], z=[0.79, 0.78, 0.79, 0.77], any_fallen=False
```

At the end, a pass/fail summary listing `travelled` and `fallen` per env.

### CLI flags

| flag              | default                                                            | meaning                                 |
|-------------------|--------------------------------------------------------------------|-----------------------------------------|
| `--num-envs`      | `4`                                                                | parallel envs                           |
| `--episode-sec`   | `10`                                                               | sim seconds to run                      |
| `--headless`      | off                                                                | run without viewer                      |
| `--encoder-onnx`  | `sonic_python_inference/assets/encoder_dyn.onnx`                   | encoder ONNX path                       |
| `--decoder-onnx`  | `sonic_python_inference/assets/decoder_dyn.onnx`                   | decoder ONNX path                       |
| `--planner-onnx`  | `gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx`       | planner ONNX path (batch=1, pooled)     |

### Passing criteria

- env 0: `travelled < 0.5 m`, `fallen == False`
- env 1: `travelled > 3.0 m`, `fallen == False`
- env 2: robot stays in place, yaw converged ~45° (visually in viewer)
- env 3: `travelled > 8.0 m`, `fallen == False`

If any env falls or locomotion misbehaves, that's a wiring bug — don't move
on to Stage 2 until Stage 1 is clean.

---

## Rates (fixed — cannot be tuned without retraining)

| Layer                  | Hz  | How                             |
|------------------------|-----|---------------------------------|
| Isaac Lab physics      | 200 | `sim.dt = 0.005`                |
| Policy (encoder+decoder) | 50 | `decimation = 4`                |
| Planner ONNX call      | 10  | every 5th policy step           |
| Planner native output  | 30  | linear + slerp → resample to 50 |

---

## Programmatic use

```python
from sonic_python_inference.sonic_inference import SonicVR3PTInference

from gear_sonic.envs.manager_env.robots.g1 import G1_ISAACLAB_TO_MUJOCO_DOF

infer = SonicVR3PTInference(
    num_envs=4,
    encoder_onnx="sonic_python_inference/assets/encoder_dyn.onnx",
    decoder_onnx="sonic_python_inference/assets/decoder_dyn.onnx",
    planner_onnx="gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx",
    default_angles=default_angles_il,           # [29] float32, IsaacLab order
    action_scale=action_scale_il,               # [29] float32, IsaacLab order
    isaaclab_to_mujoco_dof=G1_ISAACLAB_TO_MUJOCO_DOF,
    device="cuda",
)

infer.reset(joint_pos=q_il, root_pos=root_pos, root_quat_wxyz=root_quat)

# Per 50 Hz policy tick:
motor_targets_il = infer.step(
    vr_3pt_position=...,        # [N, 9]   root-local xyz for torso, L wrist, R wrist
    vr_3pt_orientation=...,     # [N, 12]  root-local wxyz quats (3 × 4)
    mode=...,                   # [N]      0 = walk, 1 = run, ...
    movement_direction=...,     # [N, 3]
    facing_direction=...,       # [N, 3]
    target_vel=...,             # [N]
    height=...,                 # [N]
    joint_pos=...,              # [N, 29]  current pos in IsaacLab DOF order
    joint_vel=...,              # [N, 29]  IsaacLab order
    base_ang_vel=...,           # [N, 3]   base frame
    gravity_in_base=...,        # [N, 3]
    root_pos=...,               # [N, 3]
    root_quat_wxyz=...,         # [N, 4]
)
# motor_targets_il is directly feedable to robot.set_joint_position_target(...).
```

### DOF order reference

| Tensor                                              | Order       |
|-----------------------------------------------------|-------------|
| `step()` / `reset()` `joint_pos` / `joint_vel` args | IsaacLab    |
| `step()` return value (motor targets)               | IsaacLab    |
| `default_angles`, `action_scale` ctor args          | IsaacLab    |
| Decoder action output (internal)                    | IsaacLab    |
| Decoder proprio history (internal)                  | IsaacLab    |
| Planner `context_mujoco_qpos` / output trajectory   | MuJoCo      |
| Encoder `motion_joint_positions_lowerbody` (first 12 of MJ joints) | MuJoCo |

The class handles the IL↔MJ conversion internally (driven by the
`isaaclab_to_mujoco_dof` ctor arg) so the API boundary stays IsaacLab-order.

See `scripts/stage1_loco_only.py` for a worked example end-to-end.

---

## Out of scope (later stages)

- **Stage 2** — VR 3pt tracking: vary hand targets per env, measure wrist
  tracking error. Script not yet written.
- **Stage 3** — N=1 numerical parity vs the C++ MuJoCo deploy.
- **Stage 4** — Dex3 fingers as an independent controller.
