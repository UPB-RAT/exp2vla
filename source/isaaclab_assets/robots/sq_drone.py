# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the quadcopters"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from pathlib import Path

UPB_SQUEEZABLE_DRONE_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(Path(__file__).parents[3].joinpath("isaaclab_tasks/isaaclab_tasks/direct/quadcopter/models/squeezable_uav.usd")),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.5, 0.0, 0.5),
        joint_pos={
            'LegJoint1':0.0,
            'LegJoint2':0.0,
            'LegJoint3':0.0,
            'LegJoint4':0.0,
            'PropellerJoint1':0.0,
            'PropellerJoint2':0.0,
            'PropellerJoint3':0.0,
            'PropellerJoint4':0.0,
        },
        joint_vel={
            'LegJoint1':0.0,
            'LegJoint2':0.0,
            'LegJoint3':0.0,
            'LegJoint4':0.0,
            'PropellerJoint1':-130.0,
            'PropellerJoint2':130.0,
            'PropellerJoint3':-130.0,
            'PropellerJoint4':130.0,
        },
    ),
    actuators={
        # Only applies to leg joints
        "leg_drive": ImplicitActuatorCfg(
            joint_names_expr=["LegJoint[1-4]"],
            stiffness=1e7,
            damping=1e4,
        ),
        # Applies to propellers: makes them passive
        "propeller_drive": ImplicitActuatorCfg(
            joint_names_expr=["PropellerJoint[1-4]"],
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
"""Configuration for the custom quadcopter."""