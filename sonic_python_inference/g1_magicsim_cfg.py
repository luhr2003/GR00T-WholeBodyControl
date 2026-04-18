"""G1 ArticulationCfg variant that spawns from MagicSim's USD instead of SONIC's URDF.

Reuses `G1_CYLINDER_MODEL_12_DEX_CFG`'s actuators, init state, PD gains, and
all joint metadata — only the `spawn` block is swapped from `UrdfFileCfg`
(`gear_sonic/data/assets/robot_description/urdf/g1/main.urdf`) to
`UsdFileCfg` pointing at `assets/g1_magicsim.usd` (copy of
`MagicSim/Assets/Robots/g1_new.usd`).

The URDF→USD conversion IsaacLab does on first URDF spawn can be slow and has
mesh-matching quirks. Skipping it by consuming MagicSim's pre-authored USD
keeps the kinematics and actuator setup identical while letting us iterate
faster. Joint-name set must match what the actuator regexes in
`G1_CYLINDER_MODEL_12_DEX_CFG.actuators` target — `g1_new.usd` was authored
with the same 29-DOF body + 14 dex-finger joints as SONIC's URDF, so the
regexes resolve the same joint list.
"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from gear_sonic.envs.manager_env.robots.g1 import G1_CYLINDER_MODEL_12_DEX_CFG


DEFAULT_MAGICSIM_USD_PATH = str(
    Path(__file__).parent / "assets" / "g1_magicsim.usd"
)


def make_g1_magicsim_cfg(usd_path: str | Path = DEFAULT_MAGICSIM_USD_PATH) -> ArticulationCfg:
    """Return G1_CYLINDER_MODEL_12_DEX_CFG with spawn replaced by a USD file.

    `rigid_props` and `articulation_props` are copied from the URDF spawn so
    solver iteration counts, self-collisions, linear/angular damping, etc.
    stay bit-identical. `activate_contact_sensors` is also preserved.
    Everything outside `spawn` — actuators (kp/kd/armature), init_state,
    `soft_joint_pos_limit_factor` — is untouched via `.replace()`.
    """
    base_spawn = G1_CYLINDER_MODEL_12_DEX_CFG.spawn
    new_spawn = sim_utils.UsdFileCfg(
        usd_path=str(usd_path),
        activate_contact_sensors=base_spawn.activate_contact_sensors,
        rigid_props=base_spawn.rigid_props,
        articulation_props=base_spawn.articulation_props,
    )
    return G1_CYLINDER_MODEL_12_DEX_CFG.replace(spawn=new_spawn)


G1_MAGICSIM_USD_CFG = make_g1_magicsim_cfg()
