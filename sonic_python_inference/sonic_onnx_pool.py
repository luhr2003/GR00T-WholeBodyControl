"""Concurrent ORT session pools for the SONIC deploy encoder + decoder.

Both deploy ONNX files ship with batch=1 baked in structurally
(`model_encoder.onnx`: in `obs_dict[1,1762]` → out `encoded_tokens[1,64]`;
 `model_decoder.onnx`: in `obs_dict[1,994]` → out `action[1,29]`),
so we run N concurrent sessions, mirroring `PlannerSessionPool`. Each session
releases the GIL during `sess.run(...)` so the ThreadPoolExecutor gives near-
linear speed-up until the GPU saturates.

All three pools (planner / encoder / decoder) share the same contract: feed a
list of batch=1 dicts, get a stacked [N, ...] batch back.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import onnxruntime as ort


ENCODER_OBS_DIM = 1762
ENCODER_TOKEN_DIM = 64
DECODER_OBS_DIM = 994
DECODER_ACTION_DIM = 29


class _OrtSessionPool:
    """Shared machinery for batch=1 ORT session pools."""

    INPUT_NAME: str = ""

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
                {"device_id": device_id, "cudnn_conv_algo_search": "DEFAULT"},
            ),
            "CPUExecutionProvider",
        ]
        self.sessions = [
            ort.InferenceSession(str(model_path), providers=providers)
            for _ in range(pool_size)
        ]
        self._executor = None if serial else ThreadPoolExecutor(max_workers=pool_size)

    def _run_one(self, sess: ort.InferenceSession, feed: np.ndarray) -> np.ndarray:
        return sess.run(None, {self.INPUT_NAME: feed})[0]

    def run_batched(self, batch: np.ndarray) -> np.ndarray:
        """batch [N, D] → output [N, D_out]. Each env runs on its own session."""
        assert batch.shape[0] == self.pool_size, (
            f"batch N={batch.shape[0]} != pool_size={self.pool_size}"
        )
        # Keep the [1, D] leading dim that the batch=1 graph requires
        feeds = [batch[i : i + 1].astype(np.float32, copy=False) for i in range(self.pool_size)]
        if self.serial:
            outs = [self._run_one(s, f) for s, f in zip(self.sessions, feeds)]
        else:
            futs = [
                self._executor.submit(self._run_one, s, f)
                for s, f in zip(self.sessions, feeds)
            ]
            outs = [f.result() for f in futs]
        return np.concatenate(outs, axis=0)  # [N, D_out]

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=False)


class SonicEncoderPool(_OrtSessionPool):
    """N parallel sessions of the deploy encoder (`model_encoder.onnx`)."""

    INPUT_NAME = "obs_dict"


class SonicDecoderPool(_OrtSessionPool):
    """N parallel sessions of the deploy decoder (`model_decoder.onnx`)."""

    INPUT_NAME = "obs_dict"
