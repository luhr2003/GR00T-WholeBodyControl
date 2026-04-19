"""Joint-order helpers for SONIC inference scripts.

The released SONIC policy was trained with the 29 body joints in
``gear_sonic.envs.env_utils.joint_utils.G1_ISAACLab_ORDER``. Some USD assets
can expose the same joint-name set in a different runtime order, so policy
inputs and outputs must be gathered/scattered by name instead of trusting the
articulation order.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from gear_sonic.envs.env_utils.joint_utils import G1_ISAACLab_ORDER


DEFAULT_HAND_RE = re.compile(r".*_hand_.*_joint")


def g1_body_indices_in_training_order(
    full_joint_names: Sequence[str],
    hand_re: re.Pattern[str] = DEFAULT_HAND_RE,
) -> tuple[list[int], list[str]]:
    """Return runtime indices for the canonical SONIC 29-DOF body order."""
    full_joint_names = list(full_joint_names)
    policy_joint_names = list(G1_ISAACLab_ORDER)

    missing = [name for name in policy_joint_names if name not in full_joint_names]
    if missing:
        raise RuntimeError(
            "Runtime articulation is missing SONIC policy joints: "
            + ", ".join(missing)
        )

    body_runtime_names = [name for name in full_joint_names if not hand_re.fullmatch(name)]
    extra_body = [name for name in body_runtime_names if name not in policy_joint_names]
    if extra_body:
        raise RuntimeError(
            "Runtime articulation has non-hand joints outside SONIC's 29-DOF "
            "policy order: "
            + ", ".join(extra_body)
        )

    return [full_joint_names.index(name) for name in policy_joint_names], policy_joint_names
