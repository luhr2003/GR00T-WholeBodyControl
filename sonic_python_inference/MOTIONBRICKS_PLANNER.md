# Self-Exported Motionbricks Planner — Design & Blocker Log

> **Status: BLOCKED on upstream release (2026-04-29).** The `motionbricks/motion_backbone/models/` subpackage is missing from NVIDIA's preview release, so the Pose/Root checkpoints cannot be loaded into PyTorch. Issue filed against [NVlabs/GR00T-WholeBodyControl] (or motionbricks repo). Waiting for upstream to ship `pose_model.py` / `root_model.py` / `sampling.py` (or their full GEAR-SONIC release, roadmap target ~2026-05-29).

This document captures the design for replacing `gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx` (structurally batch=1) with a self-exported, dynamic-batch motionbricks-based planner, plus the full investigation that led here. **Resume here once the upstream blocker clears.**

---

## 1. Why we want to do this

`PlannerSessionPool` (`sonic_python_inference/sonic_planner_pool.py`) keeps **N independent ORT sessions** on N CUDA streams just to fake batching, because the shipped `planner_sonic.onnx` has 486 `Reshape` nodes consuming `Concat` over ~612 `Constant(1)` tensors — batch=1 is a structural invariant we cannot fix at the ONNX level.

Motionbricks (paper SIGGRAPH 2026; merged into the repo at `motionbricks/`) is the upstream **PyTorch source** of the same planner family. Paper §7.4 confirms the deploy path matches: *"models exported via ONNX and loaded using TensorRT … replanning at 10 Hz"*. Re-exporting from `motionbricks/out/motionbricks_{vqvae,pose,root}/version_1/checkpoints/*.ckpt` as **three separate ONNX graphs with dynamic batch** kills the batch=1 trap.

Required modes (fixed-base test rig, robot suspended for visual debugging): `idle (0)`, `slow_walk (1)`, `run (3)`, `kneel_two_leg (5)`, `kneel_one_leg (6)`. Interface stays byte-compatible with `docs/source/references/planner_onnx.md`.

---

## 2. The blocker — what's missing upstream

### Evidence
```
$ git log --all --oneline -- motionbricks/motionbricks/motionbricks/motion_backbone/
b9b634b Add MotionBricks subproject (preview release)   # single commit, never contained models/

$ ls motionbricks/motionbricks/motionbricks/motion_backbone/
demo/  inference/  neural_modules/  __init__.py
                                    ^^^^^^^^^^ no models/ subpackage

$ python -c "from motionbricks.motion_backbone.models.pose_model import MotionModel"
ModuleNotFoundError: No module named 'motionbricks.motion_backbone.models'
```

### What's actually imported but absent
`motionbricks/motionbricks/motionbricks/motion_backbone/inference/motion_inference.py:4-6`:

```python
from motionbricks.motion_backbone.models.sampling   import gumbel_sample
from motionbricks.motion_backbone.models.pose_model import MotionModel as pose_model_cls
from motionbricks.motion_backbone.models.root_model import MotionModel as root_model_cls
```

None of the three files exist anywhere in the repo. `setup.py` uses `find_packages()` so `pip install -e .` cannot pull them from elsewhere either.

### What this blocks
- **Cannot load** `motionbricks_pose/.ckpt` or `motionbricks_root/.ckpt` — state_dict has no class to bind to.
- **Cannot run** `motion_inference.predict()` — the entire 3-stage inference pipeline is dead.
- **Cannot probe** determinism, **cannot export** root/pose to ONNX.

### What still works
- `motionbricks/vqvae/neural_modules/vqvae.py` — present, importable. ✓
- `motionbricks/motion_backbone/neural_modules/{pose_backbone,root_backbone,mlp,position_embedding}.py` — present. ✓ These are the actual NN backbones; the missing files are the PyTorch-Lightning wrappers (`MotionModel`) that bind `backbone_net` + `supporting_nets.pose_net` (the vqvae) and provide `predict()` plumbing.

### Reconstruction option (NOT recommended)
With Hydra `instantiate()` + the present backbone classes + per-ckpt `hparams.yaml`, it would be possible to reverse-engineer enough of `MotionModel` (~200-400 LOC) to load state_dict and call `backbone_net.forward(...)` directly. Rejected because: silent mis-instantiation risk; fragile reconstruction breaks the moment NVIDIA ships an official `MotionModel` with a different API.

---

## 3. Architecture (when unblocked)

New package `sonic_python_inference/motionbricks_planner/`. Public class:

```python
class MotionbricksPlanner:
    def __init__(self, root_onnx, pose_onnx, vqvae_onnx, clip_ckpt, num_envs, device_id=0): ...
    def run_batched(self, per_env_feeds: list[dict[str, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
        """Drop-in replacement for `PlannerSessionPool.run_batched`.
        Returns (traj [N, 64, 36] float32 MJ-ordered, num_pred_frames [N] int64).
        """
```

Inputs are the **exact 11 names** in `sonic_planner_pool.py:22-34` `PLANNER_INPUT_SPECS`. Internally runs **3 ORT calls per replan** (root/pose/decoder, all dynamic-batch), versus N×1 calls today. Stage-0 (smart-locomotion + spring + clip selection) ports to NumPy.

`sonic_inference.SonicVR3PTInference._run_planner` only changes its constructor — `_context_from_cache`, `resample_traj_30_to_50hz`, playback machinery all stay identical because motionbricks output is also 30 Hz MJ-ordered `[1, 64, 36]`.

### Joint-order discipline (load-bearing)

Verified by investigation:
- Motionbricks `out/G1-clip.ckpt` `mujoco_qpos [15, 150, 36]` = `[root_pos(3), root_quat_wxyz(4), joints_mj(29)]` — same DFS order as `motionbricks/assets/skeletons/g1/g1.xml`.
- SONIC `gear_sonic.envs.manager_env.robots.g1.G1_MUJOCO_TO_ISAACLAB_DOF` consumes the same 29-joint MJ order.
- `traj[..., 7:36][:, :, MJ_TO_IL]` works **without** name-by-name remap.

Keep the boundary assert from `sonic_python_inference/scripts/stage0_planner_only.py:99`:

```python
assert np.array_equal(MJ_TO_IL, np.argsort(IL_TO_MJ))
```

### Height contract

Per `docs/source/references/planner_onnx.md:179-192`, V2 implements height as *"find the keyframe in the reference clip whose pelvis_z is closest to the requested value."* Motionbricks's released `clip_holder_G1` does **not** support height-indexed lookup (each clip is a 5–90 frame contiguous snippet, sampled by random offset).

| mode | height behavior | implementation |
|------|----------------|----------------|
| 0,1,2,3 (locomotion) | Dead input — `assert height < 0` at boundary, log warning if non-default | NumPy stage-0 ignores `height` |
| 4,5,6 (squat/kneel) | Live — closest-pelvis-z keyframe lookup within clip | New `_select_keyframe_by_height(clip_idx, height)` in stage-0 (Stage D) |

Matches V2's contract exactly so all existing callers (e.g. `stage_hybrid_kneel.py`) work unchanged.

### Verified ckpt schemas

`motionbricks/out/G1-clip.ckpt` (7.5 MB, single state_dict, 7 buffers indexed by clip_id 0–14):

| Key | Shape | Dtype |
|-----|-------|-------|
| `global_root_positions` | `[15, 150, 3]` | float32 |
| `global_joint_positions` | `[15, 150, 34, 3]` | float32 (34 = pelvis + 33 body inc. 4 dummies) |
| `global_joint_rotations` | `[15, 150, 34, 3, 3]` | float32 (rotation matrices) |
| `global_headings` | `[15, 150]` | float32 |
| `motion_feature` | `[15, 150, 414]` | float32 (latent input to root model) |
| `mujoco_qpos` | `[15, 150, 36]` | float32 (`[pos(3), quat_wxyz(4), joints_mj(29)]`) |
| `num_frames_per_clip` | `[15]` | int32 (e.g. `[30, 30, 30, 30, 10, 5, 20, 8, 20, 20, 90, 76, 90, 5, 5]`) |

Existing 15 clips: `idle, slow_walk, walk, hand_crawling, walk_boxing, elbow_crawling, stealth_walk, injured_walk, walk_stealth, walk_happy_dance, walk_zombie, walk_gun, walk_scared, walk_left, walk_right`.

**No kneel/squat/sit clips in the public release** — even though paper Fig 18 confirms BONES-SEED training data contains "Sitting" (~5%), "On hands and knees" (~0.7%), "Crouching", "Kneeling". These are absent from the demo's `clip_holder_G1.ckpt`. We extract them ourselves in Stage D.

`gear_sonic` SONIC PKL schema (training_data.md, `gear_sonic/data_process/convert_soma_csv_to_motion_lib.py`):
```python
{name: {
    "root_trans_offset": [T, 3] float32,
    "pose_aa":           [T, 30, 3] float32,  # axis-angle (30 bodies)
    "dof":               [T, 29] float32,     # joint angles, MJ order
    "root_rot":          [T, 4] float32,      # scipy xyzw
    "smpl_joints":       [T, 24, 3] float32,  # placeholder (zeros for G1-only data)
    "fps":               int,
}}
```

---

## 4. Stages (when unblocked)

### Stage A — VQVAE decoder ONNX export + numerical parity

**Goal**: Export `vqvae.forward_decoder` only. Validate against PyTorch reference outputs.

**Files to ADD**:
- `sonic_python_inference/motionbricks_planner/__init__.py` — package marker.
- `sonic_python_inference/motionbricks_planner/export_vqvae.py` (~120 LOC) — wraps `vqvae.forward_decoder(pose_tokens, target_cond, has_target_cond, external_cond, token_mask, use_overall_indices=False)` in `nn.Module` returning `recon_state`. `dynamic_axes={input: {0: "batch"} for all I/O}`. Output: `sonic_python_inference/assets/motionbricks/vqvae_decoder.onnx`.
- `sonic_python_inference/motionbricks_planner/reference_dump.py` (~80 LOC) — runs `motion_inference.predict()` once with frozen `(mode=1, dir=[1,0,0], ctx=stand)` + `pose_token_sampling_use_argmax=True`, dumps `(pose_tokens, target_cond, has_target_cond, external_cond, token_mask, expected_pred_local_poses)` for B∈{1,4} to `assets/motionbricks/vqvae_ref.npz`.

**Test**: `sonic_python_inference/scripts/stage_mb_a_vqvae_parity.py` (~150 LOC). Pure NumPy/ORT, no Isaac Lab. Mirrors the validation block in `sonic_python_inference/scripts/export_dynamic_batch_onnx.py`.

**Pass**:
- `max_abs_err < 1e-4` on `pred_local_poses` for B=1 and B=4.
- ORT loads `CUDAExecutionProvider` cleanly.
- B=4 latency on RTX 4080 < 6 ms.

**Fail mode**: Shape error → enumerate `dynamic_axes` per input. Numerical drift → bisect via encoder determinism check first.

---

### Stage B — Root + pose ONNX export + NumPy stage-0 + planner_only Isaac Lab test

**Goal**: End-to-end planner running on Isaac Lab fixed-base G1, slow_walk for 5 s. Robot suspended.

**Files to ADD**:
- `sonic_python_inference/motionbricks_planner/export_root.py` (~150 LOC) — wraps `root_model.backbone_net.forward(...)` returning `(pred_num_tokens [B,1], pred_global_root_values [B,64,5], pred_local_root_values [B,64,4])`. Dynamic axis on B for all I/O.
- `sonic_python_inference/motionbricks_planner/export_pose.py` (~150 LOC) — wraps `pose_model.backbone_net.forward(...)` with **argmax baked into graph** (returns `pose_tokens [B,16,16]` int64 directly, not logits — avoids exporting `gumbel_sample`).
- `sonic_python_inference/motionbricks_planner/stage0_numpy.py` (~250 LOC) — pure NumPy port of:
  - `full_agent._generate_spring_model_position_and_heading` (critically damped spring, paper Eq 6).
  - `convert_mujoco_qpos_to_motion_transforms` (`motionbricks/.../helper/mujoco_helper.py:335`) — FK over G1 skeleton XML to produce `(global_joint_positions [B,4,34,3], global_joint_rotations [B,4,34,3,3])`.
  - Clip selection via mode-onehot over `mujoco_qpos [15, 150, 36]`.
  - Output: 8-frame `(global_root_values [B,8,5], local_root_values [B,8,4], local_poses [B,8,~297], has_*)`.
- `sonic_python_inference/motionbricks_planner/planner.py` (~250 LOC) — `MotionbricksPlanner` class. Translates SONIC's 11-input dict → stage0 → 3 ORT calls → NumPy port of `convert_motion_features_to_mujoco_qpos` (`mujoco_helper.py:265`) → 30 Hz `[N, 64, 36]` traj.

**Test**: `sonic_python_inference/scripts/stage_mb_b_planner_only.py` (~270 LOC). Clone of `stage0_planner_only.py`. Only changes:
1. `from sonic_python_inference.motionbricks_planner.planner import MotionbricksPlanner`
2. `planner_pool = MotionbricksPlanner(root_onnx=..., pose_onnx=..., vqvae_onnx=..., clip_ckpt=..., num_envs=N)`
3. Reuse `_build_planner_feeds`, `_context_from_cache`, `resample_traj_30_to_50hz` verbatim.

Fixed-base spawn z=1.2, mode=SLOW_WALK, target_vel=0.3, replan every 5 policy steps. Run 5 s. Drive PD directly with `joints_il = joints_mj[mj_to_il]`.

**Pass**:
- 5 s sim completes without exception.
- `planner_cache[..., 7:13]` (lower 6 leg joints, MJ) oscillation amplitude ≥ 0.2 rad over s 1–5.
- Mean forward velocity from traj root-pos delta in `[0.4, 0.8] m/s` over s 1–5 (motionbricks `slow_walk.avg_root_vel = 0.6 m/s`).
- Replan wall-clock at N=4 < 25 ms (vs ~40 ms baseline).
- No joint-cmd jump > 0.5 rad across replan boundary.

**Fail mode**:
- Feet penetrate floor → joint-order regression. Probe: visualize one motionbricks `mujoco_qpos` frame after `MJ_TO_IL` gather in MuJoCo viewer, confirm A-pose.
- Gait stalls → spring port mismatch. Diff against torch reference offline.
- Replan latency > 60 ms → ORT not on CUDA. Check provider list.

---

### Stage C — Locomotion mode coverage (idle / slow_walk / run)

**Goal**: Cycle through SONIC modes 0/1/3 within one run, mode transitions, per-mode tracking.

**Files to ADD**:
- `sonic_python_inference/motionbricks_planner/mode_map.py` (~50 LOC):
  ```python
  SONIC_MODE_TO_MB_CLIP = {0: 0, 1: 1, 2: 2, 3: 2}  # idle, slow_walk, walk, run→walk
  ```
  Note: motionbricks `walk.avg_root_vel = 2.0 m/s`, capped below SONIC's RUN range (2.5–7.5). Log warning on `target_vel > 2.0`. Document as known regression vs V2 — fixable in Stage D extension if BONES-SEED has a `run` category.

**Test**: `sonic_python_inference/scripts/stage_mb_c_modes.py` (~280 LOC). Same fixed-base rig as Stage B. State machine (`t` in 50 Hz policy ticks):
- t ∈ [0, 150)   — IDLE, target_vel=0
- t ∈ [150, 400) — SLOW_WALK, target_vel=0.6
- t ∈ [400, 650) — RUN, target_vel=2.0
- t ∈ [650, 750] — IDLE again

Mode change always triggers replan (per `planner_onnx.md:362-367`). Per-mode log every 25 ticks: commanded `target_vel`, measured forward velocity, max joint-cmd jump per replan, replan latency p50/p99.

**Pass**:
- IDLE: max joint-vel < 0.1 rad/s after t=10.
- SLOW_WALK: forward velocity in [0.4, 0.8] m/s over last 4 s of mode.
- RUN: forward velocity ≥ 1.5 m/s over last 4 s of mode (capped by walk-clip ceiling).
- Mode transition: max joint-cmd jump across replan boundary < 0.3 rad.
- Replan latency p99 at N=4 < 35 ms.

---

### Stage D — Kneel modes via BONES-SEED extraction

**Goal**: Add `kneel_two_leg (5)` and `kneel_one_leg (6)` to motionbricks clip catalog by ingesting filtered BONES-SEED clips. Implement V2-compatible height-indexed keyframe lookup.

**Pre-stage probe**: `sonic_python_inference/scripts/stage_mb_d_probe_dataset.py` (~80 LOC, read-only).
- Download BONES-SEED via `huggingface-cli download bones-studio/seed --repo-type dataset --local-dir bones-seed/` (one-time, ~30 GB).
- Walk filenames; grep for `kneel`, `kneel_two`, `kneel_one`, `squat`, `crouch`, `sit_down`, `kneeling`. Print counts per keyword.
- **Pass**: ≥ 3 candidates per kneel-class keyword.
- **Fail**: zero matches → BONES-SEED filename convention differs; manually inspect HF dataset card, adjust keyword list.

**Files to ADD**:
- `sonic_python_inference/motionbricks_planner/ingest_bones_seed.py` (~250 LOC):
  - Reads filtered BONES-SEED PKLs from `data/motion_lib_bones_seed/robot_filtered/` (after running existing `gear_sonic/data_process/convert_soma_csv_to_motion_lib.py` + `filter_and_copy_bones_data.py`).
  - PKL→ckpt converter:
    - `pose_aa [T, 30, 3]` axis-angle → `global_joint_rotations [T, 34, 3, 3]` (3×3 matrices, FK over G1 skeleton XML to derive the 4 dummy joints `left_toe_base`, `right_toe_base`, `left_hand_roll_skel`, `right_hand_roll_skel` per `motionbricks/.../motionlib/core/skeletons/g1.py:158`).
    - `root_rot [T, 4]` scipy xyzw → wxyz for `mujoco_qpos[..., 3:7]`.
    - `dof [T, 29]` → `mujoco_qpos[..., 7:36]` directly.
    - Computes `global_joint_positions` via FK; `global_headings` from `atan2`.
    - Computes `motion_feature [T, 414]` via `extract_feature_from_motion_rep(..., feature_mode='joint_positions_and_rotations')` (`motionbricks/.../helper/data_training_util.py:162`).
    - Picks 3-5 representative kneel clips, slices to ≤90 frames each.
- `sonic_python_inference/motionbricks_planner/build_kneel_ckpt.py` (~120 LOC):
  - Loads existing `motionbricks/out/G1-clip.ckpt`.
  - Appends new clips to all 7 buffers, expanding clip count from 15 → 18: `kneel_two_leg @ 15`, `kneel_one_leg @ 16`, optional `squat @ 17`.
  - Pads new clips' frames to `max(150, new_clip_len)`.
  - Saves to `sonic_python_inference/assets/motionbricks/G1-clip-extended.ckpt`.
- `sonic_python_inference/motionbricks_planner/stage0_numpy.py` PATCH (~80 LOC delta):
  - `_select_keyframe_by_height(clip_idx, height) -> frame_offset` — for kneel/squat clips, scans `mujoco_qpos[clip_idx, :num_frames, 2]` for closest-pelvis-z match (V2 contract).
  - Routes `mode ∈ {4,5,6}` through height-indexed selection; `mode ∈ {0,1,2,3}` keeps random-offset behavior.
- `sonic_python_inference/motionbricks_planner/mode_map.py` PATCH:
  ```python
  SONIC_MODE_TO_MB_CLIP = {0:0, 1:1, 2:2, 3:2, 4:17, 5:15, 6:16}
  ```

**Test**: `sonic_python_inference/scripts/stage_mb_d_kneel.py` (~350 LOC). Clone of `stage_hybrid_kneel.py` two-phase state machine, but **strip Pink IK** and **strip the encoder/decoder** (planner-only, fixed-base, drive PD directly). Target `MotionbricksPlanner` not `PlannerSessionPool`.

Phases (per kneel mode, run twice for KNEEL_TWO_LEG and KNEEL_ONE_LEG):
- t ∈ [0, 150)   — IDLE, settle.
- t ∈ [150, 400) — kneel_mode, target_vel=0, height=0.25 m.
- t ∈ [400, 550) — IDLE, height=-1, watch return.

Reuse `stage_hybrid_kneel.py:240-256 _lowered_rest_pose` pattern for visual reference.

**Pass**:
- Pelvis z descends from ~0.79 m to within ±0.05 m of `--kneel-height` by t=350.
- Knee flexion ≥ 1.5 rad for both legs (kneel_two) or one leg (kneel_one) at apex.
- After mode→IDLE: pelvis returns to ≥0.75 m within 2 s, no oscillation > 0.05 m.
- Replan latency p99 at N=1 < 40 ms.

**Fail mode**:
- Decoder produces jittery/exploding crouch trajectory → confirms in-distribution-but-rare hypothesis. Increase clip pool from 3 to 10 BONES-SEED kneel motions, retry. If still bad, fall back to **hybrid backend**: keep V2 ONNX loaded, route mode∈{4,5,6} through it inside `MotionbricksPlanner.run_batched`. Document inline with `# DEFERRED:` comment.
- Pelvis fails to descend → height-indexed selection wrong; print `[clip_idx, frame_idx, pelvis_z]` per replan to verify lookup picks correct keyframe.

---

### Stage E — Drop-in to existing rigs (post-merge)

Once Stages A–D pass:
- `sonic_python_inference/sonic_inference.py:270` — single constructor swap from `PlannerSessionPool` to `MotionbricksPlanner`. `_run_planner` body unchanged.
- All existing `stage_hybrid_*` scripts work unchanged (they consume `SonicVR3PTInference`).
- `PlannerSessionPool` stays in tree as fallback, gated by `SONIC_USE_V2_PLANNER=1` env var. Plan removal in a later milestone.

---

## 5. Pre-implementation probes (run when unblocked)

These need answering BEFORE Stage A code is written:

1. **Argmax determinism** (~30 LOC, 1 min): Run `motion_inference.predict()` twice with `pose_token_sampling_use_argmax=True` and identical inputs. Max abs diff in `pred_local_poses` must be < 1e-6. If not, pin Conv1d kernels with `torch.use_deterministic_algorithms(True)` before export.
2. **VQVAE export feasibility** (~50 LOC, 5 min): Try `torch.onnx.export(...)` with default tracer on `forward_decoder`; if it fails, switch to `torch.onnx.dynamo_export`. Identify any control-flow op needing rewriting BEFORE writing Stage A's full export script.
3. **BONES-SEED filename keyword sanity** (one-shot grep, 1 min after download): see Stage D probe.

---

## 6. Critical files (read-only references during implementation)

### Existing test pattern to mirror
- `sonic_python_inference/scripts/stage0_planner_only.py` — Stage B/C base.
- `sonic_python_inference/scripts/stage_hybrid_kneel.py` — Stage D base (strip Pink IK + encoder/decoder).

### Existing infra to reuse verbatim
- `sonic_python_inference/sonic_inference.py:_context_from_cache` (line 346)
- `sonic_python_inference/sonic_inference.py:resample_traj_30_to_50hz` (line 180)
- `sonic_python_inference/sonic_inference.py:slerp_torch` (line 167)

### Existing data pipeline (already in repo)
- `gear_sonic/data_process/convert_soma_csv_to_motion_lib.py` — BONES-SEED CSV → motion_lib PKL.
- `gear_sonic/data_process/filter_and_copy_bones_data.py` — keyword-based PKL filtering. Default blacklist does NOT block "kneel" — kneel clips survive.

### Motionbricks model code (preview release, partial)
- `motionbricks/motionbricks/motionbricks/motion_backbone/inference/motion_inference.py` — `predict()` 3-stage call sequence (BLOCKED on missing imports).
- `motionbricks/motionbricks/motionbricks/motion_backbone/demo/full_agent.py` — smart-locomotion + spring + clip selection.
- `motionbricks/motionbricks/motionbricks/motion_backbone/demo/clips.py` — `clip_holder_G1` with 15-clip catalog.
- `motionbricks/motionbricks/motionbricks/helper/mujoco_helper.py:265-427` — `convert_*` qpos↔motion-rep converters.
- `motionbricks/motionbricks/motionbricks/motionlib/core/skeletons/g1.py:158-200` — G1Skeleton34 definition.
- `motionbricks/motionbricks/motionbricks/motion_backbone/neural_modules/{pose,root}_backbone.py` — backbone networks (present, importable).

### SONIC contract authority
- `docs/source/references/planner_onnx.md` — full 11-input/2-output spec, mode list (0–26), height contract, replan logic.
- `docs/source/user_guide/training_data.md` — BONES-SEED schema and download.
- `gear_sonic.envs.manager_env.robots.g1.{G1_ISAACLAB_TO_MUJOCO_DOF, G1_MUJOCO_TO_ISAACLAB_DOF}` — joint-order constants.

---

## 7. Verification plan (when unblocked)

Per stage, run from repo root with `.venv_isaac` active:
```bash
# Stage A (offline, no Isaac Lab)
python -m sonic_python_inference.scripts.stage_mb_a_vqvae_parity

# Stage B (Isaac Lab)
python -m sonic_python_inference.scripts.stage_mb_b_planner_only --num-envs 4

# Stage C
python -m sonic_python_inference.scripts.stage_mb_c_modes --num-envs 4

# Stage D probe (one-time, after BONES-SEED download)
python -m sonic_python_inference.scripts.stage_mb_d_probe_dataset

# Stage D test
python -m sonic_python_inference.scripts.stage_mb_d_kneel --kneel-mode two_leg
python -m sonic_python_inference.scripts.stage_mb_d_kneel --kneel-mode one_leg
```

Each script logs pass/fail to stdout. Stage A is fully offline; Stages B/C/D need Isaac Sim EULA accepted once.

Cross-stage regression test (after Stage E merge, must still pass with `MotionbricksPlanner` underneath):
```bash
python -m sonic_python_inference.scripts.stage_hybrid_kneel --kneel-mode two_leg
python -m sonic_python_inference.scripts.stage_vr3pt_eval --num-envs 4
```

---

## 8. Risks & mitigation (when unblocked)

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| ONNX export of pose-net argmax breaks (TopK path) | Medium | Use TopK→Gather op pair explicitly; fall back to logits-out + numpy argmax post-call |
| BONES-SEED kneel clips visually OOD vs motionbricks training | Low–Med | Start with 10+ candidate clips, hand-select 3 best by visual review in MuJoCo viewer |
| Stage-0 spring port numerical drift vs torch | Medium | Stage B test compares forward velocity; if drift, port spring constants exactly per paper Eq 6 |
| `MJ_TO_IL` mismatch between motionbricks XML and SONIC URDF (silent permutation) | Low | Probe before Stage B: viz one A-pose frame in MuJoCo; named asserts in `stage0_planner_only.py:99` style |
| RUN mode capped at motionbricks `walk.avg_root_vel = 2.0 m/s` | Confirmed regression | Document in `mode_map.py`; revisit when BONES-SEED ingest pipeline (Stage D) extends to running clips |
| Upstream `MotionModel` API changes between preview and full release | High (this is the blocker) | Wait for full release rather than reverse-engineering. Re-validate `motion_inference.predict()` signature before resuming Stage A. |

---

## 9. Decision log

- **2026-04-29**: Discovered V2 `planner_sonic.onnx` is structurally batch=1 (cannot be re-exported with dynamic batch).
- **2026-04-29**: Read motionbricks paper + code; confirmed it's the upstream PyTorch source of the same planner family. Designed 4-stage Option B plan.
- **2026-04-29**: Discovered `motion_backbone/models/` subpackage is missing from preview release. **BLOCKED**. Issue filed upstream by user.
- **2026-04-29**: Considered reverse-engineering `MotionModel` from existing backbones — rejected (silent mis-instantiation risk; fragile against future official release).
- **TBD**: When upstream unblocks, rerun pre-implementation probes (Section 5) and resume from Stage A.

---

## 10. Interim policy

Until upstream unblocks:
- Keep using `gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx` via `PlannerSessionPool`. Status quo.
- Do not modify `sonic_python_inference/sonic_inference.py` or related rigs.
- All hybrid eval / kneel / VR-3pt tests continue to work via `PlannerSessionPool` as today.
- Resume here when NVIDIA ships either (a) the missing `motion_backbone/models/*.py` files, or (b) the full GEAR-SONIC integration release (paper roadmap target ~2026-05-29).
