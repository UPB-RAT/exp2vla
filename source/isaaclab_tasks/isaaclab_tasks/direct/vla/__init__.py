# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Quacopter environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##
gym.register(
    id="execise_00",
    entry_point=f"{__name__}.execise_00:QuadcopterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.execise_00:QuadcopterEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_vexp2vla_ppo_cfg.yaml",
    },
)

gym.register(
    id="execise_01",
    entry_point=f"{__name__}.execise_01:QuadcopterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.execise_01:QuadcopterEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_vexp2vla_ppo_cfg.yaml",
    },
)

gym.register(
    id="execise_02",
    entry_point=f"{__name__}.execise_02:QuadcopterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.execise_02:QuadcopterEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_vexp2vla_ppo_cfg.yaml",
    },
)

gym.register(
    id="execise_03",
    entry_point=f"{__name__}.execise_03:QuadcopterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.execise_03:QuadcopterEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_vexp2vla_ppo_cfg.yaml",
    },
)

gym.register(
    id="execise_04",
    entry_point=f"{__name__}.execise_04:QuadcopterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.execise_04:QuadcopterEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_vexp2vla_ppo_cfg.yaml",
    },
)

gym.register(
    id="VLA-object-tracking",
    entry_point=f"{__name__}.exp2vla_object_tracking_env:QuadcopterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.exp2vla_object_tracking_env:QuadcopterEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_vexp2vla_ppo_cfg.yaml",
    },
)
