"""Concurrent ORT session pool for the SONIC planner.

The shipped planner ONNX (planner_sonic.onnx) has batch=1 baked in structurally —
486 Reshape nodes compute their target shape via Concat over 612 Constant(1) tensors,
so `graph.input[...].dim[0].dim_param = "batch"` + shape-inference cannot propagate
a batch dim through to the output. We verified this inspection in the plan.

Workaround: keep one ORT session per env, each with its own CUDA stream so the
individual batch=1 calls can overlap on the GPU. ThreadPoolExecutor releases the
GIL across `sess.run(...)` calls, giving ~4× speedup at N=4 versus a serial loop.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import onnxruntime as ort


PLANNER_INPUT_SPECS: dict[str, tuple[tuple[int, ...], np.dtype]] = {
    "context_mujoco_qpos": ((4, 36), np.float32),
    "target_vel": ((), np.float32),
    "mode": ((), np.int64),
    "movement_direction": ((3,), np.float32),
    "facing_direction": ((3,), np.float32),
    "random_seed": ((), np.int64),
    "has_specific_target": ((1,), np.int64),
    "specific_target_positions": ((4, 3), np.float32),
    "specific_target_headings": ((4,), np.float32),
    "allowed_pred_num_tokens": ((11,), np.int64),
    "height": ((), np.float32),
}

PLANNER_OUTPUT_FRAMES = 64
PLANNER_FRAME_DIM = 36


class PlannerSessionPool:
    """Pool of batch=1 planner ORT sessions dispatched in parallel."""

    def __init__(
        self,
        model_path: str | Path,
        pool_size: int,
        device_id: int = 0,
        serial: bool = False,
    ):
        self.pool_size = pool_size
        self.serial = serial
        providers = [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": device_id,
                    "cudnn_conv_algo_search": "DEFAULT",
                },
            ),
            "CPUExecutionProvider",
        ]
        self.sessions = [
            ort.InferenceSession(str(model_path), providers=providers) for _ in range(pool_size)
        ]
        self._executor = None if serial else ThreadPoolExecutor(max_workers=pool_size)

    def _run_one(self, sess: ort.InferenceSession, feeds: dict[str, np.ndarray]):
        return sess.run(None, feeds)

    def run_batched(self, per_env_feeds: list[dict[str, np.ndarray]]):
        """Run the planner for each env.

        Args:
            per_env_feeds: list of length N; each dict maps planner input name to a
                batch=1 numpy array (leading dim 1).

        Returns:
            (traj [N, 64, 36] float32, num_pred_frames [N] int64)
        """
        assert len(per_env_feeds) == self.pool_size, (
            f"per_env_feeds length {len(per_env_feeds)} != pool_size {self.pool_size}"
        )
        if self.serial:
            outs = [self._run_one(s, f) for s, f in zip(self.sessions, per_env_feeds)]
        else:
            futs = [
                self._executor.submit(self._run_one, s, f)
                for s, f in zip(self.sessions, per_env_feeds)
            ]
            outs = [f.result() for f in futs]
        trajs = np.stack([o[0][0] for o in outs], axis=0)  # [N, 64, 36]
        num_frames = np.stack([o[1][0] for o in outs], axis=0)  # [N]
        return trajs, num_frames

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=False)
