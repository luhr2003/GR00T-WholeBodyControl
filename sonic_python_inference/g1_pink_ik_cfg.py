"""Self-contained Pink IK config for the hybrid G1 eval path.

Mirrors the task topology used in MagicSim (two hand FrameTasks in
pelvis_contour_link frame, a null-space posture task on the shoulders, a
damping task) but lives entirely inside sonic_python_inference — no magicsim
import, no cross-repo dep. Values copied verbatim from the MagicSim G1 cfg.

Joint-order convention: **IsaacLab (IL)** everywhere. `all_joint_names_il`
is filled at runtime from `robot.data.joint_names`; `PINK_CONTROLLED_JOINTS_IL`
is the 17 joints Pink IK owns (3 waist + 14 arms) in the order we want
`PinkIKController.compute()` to return positions.
"""

from __future__ import annotations

from pink.tasks import DampingTask

from isaaclab.controllers.pink_ik import (
    NullSpacePostureTask,
    PinkIKControllerCfg,
)
from isaaclab.controllers.pink_ik.local_frame_task import LocalFrameTask


PELVIS_BASE_LINK = "pelvis_contour_link"
LEFT_HAND_LINK = "left_hand_palm_link"
RIGHT_HAND_LINK = "right_hand_palm_link"

# Pink-controlled joints, IL names. Output of `PinkIKController.compute()` is
# a [17] tensor in this order.
PINK_CONTROLLED_JOINTS_IL: list[str] = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# v1 frozen wrist targets, pelvis_contour_link frame. (xyz, qw, qx, qy, qz).
LEFT_WRIST_REST_POSE_PELVIS: tuple[float, ...] = (
    0.24127, 0.15165, 0.14523, 1.0, 0.0, 0.0, 0.0,
)
RIGHT_WRIST_REST_POSE_PELVIS: tuple[float, ...] = (
    0.24127, -0.15164, 0.14523, 1.0, 0.0, 0.0, 0.0,
)


def build_pink_ik_cfg(
    urdf_path: str,
    all_joint_names_il: list[str],
    mesh_path: str | None = None,
) -> PinkIKControllerCfg:
    """Construct a PinkIKControllerCfg matching the MagicSim G1 task recipe.

    Task topology (copied from MagicSim G1_PINK_IK_CONTROLLER_CFG):
    - 2× LocalFrameTask on right/left_hand_palm_link in pelvis_contour_link
      frame (position_cost=8.0, orientation_cost=2.0, lm_damping=10, gain=0.5)
    - NullSpacePostureTask over 6 shoulder joints (cost=0.5, lm_damping=1,
      gain=0.3)
    - DampingTask (cost=0.8)
    """
    damping = DampingTask(cost=0.8)
    # IsaacLab's PinkIKController.__init__ calls `set_target_from_configuration`
    # on every variable_input_task that isn't a NullSpacePostureTask, but
    # pink 4.1.0's DampingTask doesn't implement it (it has no target). Inject
    # a no-op so the init sweep passes without neutering the task.
    if not hasattr(damping, "set_target_from_configuration"):
        damping.set_target_from_configuration = lambda _cfg: None  # type: ignore[attr-defined]

    tasks = [
        LocalFrameTask(
            RIGHT_HAND_LINK,
            base_link_frame_name=PELVIS_BASE_LINK,
            position_cost=8.0,
            orientation_cost=2.0,
            lm_damping=10.0,
            gain=0.5,
        ),
        LocalFrameTask(
            LEFT_HAND_LINK,
            base_link_frame_name=PELVIS_BASE_LINK,
            position_cost=8.0,
            orientation_cost=2.0,
            lm_damping=10.0,
            gain=0.5,
        ),
        NullSpacePostureTask(
            cost=0.5,
            lm_damping=1.0,
            controlled_frames=[LEFT_HAND_LINK, RIGHT_HAND_LINK],
            controlled_joints=[
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
            ],
            gain=0.3,
        ),
        damping,
    ]
    return PinkIKControllerCfg(
        urdf_path=urdf_path,
        mesh_path=mesh_path,
        base_link_name="pelvis",
        num_hand_joints=0,
        show_ik_warnings=True,
        fail_on_joint_limit_violation=False,
        variable_input_tasks=tasks,
        fixed_input_tasks=[],
        joint_names=list(PINK_CONTROLLED_JOINTS_IL),
        all_joint_names=list(all_joint_names_il),
    )
