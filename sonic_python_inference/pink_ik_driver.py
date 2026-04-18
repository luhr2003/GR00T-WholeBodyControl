"""Headless Pink IK driver for N parallel envs.

Wraps Isaac Lab's `PinkIKController` without touching the manager-env
framework. One `PinkIKController` instance per env (Pinocchio configuration
is stateful), stepped in a sequential Python loop. N ≤ 4 is the target scale
for v1; optimize later only if needed.

Frame convention:
    Wrist-pose targets (`left_target_pelvis`, `right_target_pelvis`) are in
    the **pelvis_contour_link (torso) frame** — matching the `LocalFrameTask`
    `base_link_frame_name` set in g1_pink_ik_cfg.py. Do NOT pass world-frame
    targets.

Units / order:
    Targets: [N, 7] = (x, y, z, qw, qx, qy, qz)  (wxyz quat).
    Joints: `curr_joint_pos_il` is [N, 29] in **IsaacLab** order.
    Output: [N, 17] in the order of `PINK_CONTROLLED_JOINTS_IL`.
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin
import torch

from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.controllers.pink_ik import PinkIKController
from isaaclab.controllers.pink_ik.local_frame_task import LocalFrameTask

from .g1_pink_ik_cfg import (
    LEFT_HAND_LINK,
    RIGHT_HAND_LINK,
    PINK_CONTROLLED_JOINTS_IL,
    build_pink_ik_cfg,
)


def _wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    """wxyz quaternion → 3×3 rotation matrix."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _pose7_to_se3(pose7: np.ndarray) -> pin.SE3:
    """(x, y, z, qw, qx, qy, qz) → pin.SE3."""
    t = np.asarray(pose7[:3], dtype=np.float64)
    R = _wxyz_to_rotmat(pose7[3:7])
    return pin.SE3(R, t)


class PinkIKDriver:
    """N parallel Pink IK solvers sharing a task recipe, run sequentially."""

    def __init__(
        self,
        num_envs: int,
        robot_cfg: ArticulationCfg,
        urdf_path: str,
        all_joint_names_il: list[str],
        device: str = "cuda",
        dt: float = 0.02,
        mesh_path: str | None = None,
    ):
        self.N = num_envs
        self.device = device
        self.dt = dt
        self.all_joint_names_il = list(all_joint_names_il)

        # Resolve IL indices of the 17 controlled joints (used by caller to
        # scatter `solve()` output into a 29-DoF buffer).
        self.controlled_joint_indices: list[int] = [
            self.all_joint_names_il.index(n) for n in PINK_CONTROLLED_JOINTS_IL
        ]

        # One controller per env — the pinocchio configuration is stateful
        # (curr q is stored on `self.pink_configuration` and updated each call).
        self.controllers: list[PinkIKController] = []
        for _ in range(num_envs):
            cfg = build_pink_ik_cfg(urdf_path, self.all_joint_names_il, mesh_path=mesh_path)
            ctl = PinkIKController(
                cfg=cfg,
                robot_cfg=robot_cfg,
                device=device,
                controlled_joint_indices=self.controlled_joint_indices,
            )
            self.controllers.append(ctl)

    def _find_frame_task(self, ctl: PinkIKController, frame_name: str) -> LocalFrameTask:
        for task in ctl.cfg.variable_input_tasks:
            if isinstance(task, LocalFrameTask) and task.frame == frame_name:
                return task
        raise RuntimeError(f"LocalFrameTask for frame '{frame_name}' not found.")

    def solve(
        self,
        curr_joint_pos_il: torch.Tensor,   # [N, 29] IL order
        left_target_pelvis: torch.Tensor,  # [N, 7]  (x,y,z, qw,qx,qy,qz) in pelvis frame
        right_target_pelvis: torch.Tensor, # [N, 7]  same
    ) -> torch.Tensor:
        """Returns [N, 17] IL-ordered target joint positions (pink-controlled)."""
        assert curr_joint_pos_il.shape == (self.N, 29)
        assert left_target_pelvis.shape == (self.N, 7)
        assert right_target_pelvis.shape == (self.N, 7)

        q_np = curr_joint_pos_il.detach().cpu().numpy().astype(np.float64)
        lt_np = left_target_pelvis.detach().cpu().numpy().astype(np.float64)
        rt_np = right_target_pelvis.detach().cpu().numpy().astype(np.float64)

        out = torch.zeros(
            self.N, len(PINK_CONTROLLED_JOINTS_IL),
            dtype=torch.float32, device=self.device,
        )

        for i, ctl in enumerate(self.controllers):
            self._find_frame_task(ctl, LEFT_HAND_LINK).set_target(_pose7_to_se3(lt_np[i]))
            self._find_frame_task(ctl, RIGHT_HAND_LINK).set_target(_pose7_to_se3(rt_np[i]))
            target = ctl.compute(q_np[i], self.dt)  # [17] tensor
            out[i] = target.to(self.device, dtype=torch.float32)
        return out
