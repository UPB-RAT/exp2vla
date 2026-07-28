# Copyright (c) 2022-2025, The Isaac Lab Project Developers
#[](https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from PIL import Image
from datetime import datetime

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, subtract_frame_transforms

from isaaclab_assets import UPB_SQUEEZABLE_DRONE_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip
import random
# Propeller joint names (order matters for velocity targets)
PROPELLER_JOINT_NAMES = (
    "PropellerJoint1",
    "PropellerJoint2",
    "PropellerJoint3",
    "PropellerJoint4",
)
PROPELLER_VELOCITY_TARGETS = (-130.0, 130.0, -130.0, 130.0)

# Controller limits
MAX_SPEED = 2.0
MAX_YAWRATE_DEG = 15.0
MAX_INCLINATION_DEG = 30.0

# Death bounds (z)
MIN_HEIGHT = 0.25
MAX_HEIGHT = 2.0


# Define shapes and colors

SHAPES = ("cuboid", "cylinder", "cone")
COLORS = ("red", "blue", "green")

# All possible task instructions: every (color, shape) pair
TASKS = [
    {"instruction": f"Fly to the {color} {shape}", "color": color, "shape": shape}
    for color in COLORS
    for shape in SHAPES
]

# =============================================================================
# UI
# =============================================================================
class QuadcopterEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: "QuadcopterEnv", window_name: str = "IsaacLab"):
        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("targets", self.env)


# =============================================================================
# Environment configuration
# =============================================================================
@configclass
class QuadcopterEnvCfg(DirectRLEnvCfg):
    # --- env ---
    episode_length_s = 10.0
    decimation = 2
    action_space = 3
    observation_space = 3
    state_space = 0
    debug_vis = True
    ui_window_class_type = QuadcopterEnvWindow

    # --- simulation ---
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 50.0,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    background: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/background",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=(
                "/home/summer_school/summer_ws/IsaacLab/source/isaaclab_tasks/"
                "isaaclab_tasks/direct/quadcopter/rat_lab/multicorridor/empty_lab.usd"
            ),
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=True,
            ),
        ),
    )

    # --- scene ---
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=25,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    # --- robot ---
    robot: ArticulationCfg = UPB_SQUEEZABLE_DRONE_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    # --- camera (Intel RealSense D455 RGB) ---
    rgbd_camera: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/base_link/front_camera",
        offset=CameraCfg.OffsetCfg(
            pos=(0.2, 0.0, 0.015),
            rot=(0.5, -0.5, 0.5, -0.5),  # ROS convention
            convention="ros",
        ),
        data_types=["rgb"],  # add "distance_to_image_plane" if you need depth
        spawn=sim_utils.PinholeCameraCfg(
            # Matches Isaac Sim RealSense D455 RGB asset
            focal_length=1.93,           # cm  (real lens ~1.93 mm; Isaac uses cm)
            focus_distance=0.6,          # m   (typical focus plane)
            horizontal_aperture=3.896,   # cm  → ~86–90° HFOV
            clipping_range=(0.4, 10.0),   # m   (D455 recommended range)
            # optional: f_stop=2.0,      # real D455 is ~f/2.0
        ),
        update_period=1.0 / 30.0,        # 30 Hz (typical full-res RGB)
        width=640,                      # native RGB max is 1280×800
        height=480,                      # or 720 / 480 if you prefer lower res
    )

    # --- viewer ---
    viewer: ViewerCfg = ViewerCfg(
        eye=(-6.5, 0.0, 3.5),
        lookat=(2.0, 0.0, 1.36),
        origin_type="env",
    )

# =============================================================================
# Environment
# =============================================================================
class QuadcopterEnv(DirectRLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(
        self,
        cfg: QuadcopterEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)

        # Controller limits
        self.max_speed = MAX_SPEED
        self.max_yawrate = torch.deg2rad(
            torch.tensor(MAX_YAWRATE_DEG, device=self.device)
        )
        self.max_inclination_angle = torch.deg2rad(
            torch.tensor(MAX_INCLINATION_DEG, device=self.device)
        )

        # Action / force buffers
        action_dim = gym.spaces.flatdim(self.single_action_space)
        self._raw_actions = torch.zeros(self.num_envs, action_dim, device=self.device)
        self._vel_commands = torch.zeros(self.num_envs, 6, device=self.device)
        self._rotor_target_vel = torch.zeros(
            self.num_envs, self._robot.num_joints, device=self.device
        )
        # Goal
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Propellers
        self.propeller_indices = self._robot.find_joints(list(PROPELLER_JOINT_NAMES))[0]

        # Episode reward accumulators
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in ("lin_vel", "ang_vel", "distance_to_goal")
        }

        # Robot body / mass
        self._body_id = self._robot.find_bodies("base_link")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(
            self.sim.cfg.gravity, device=self.device
        ).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        
        # Debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

        # Adding objects with different shapes and colors
        self._object_pos_w = {
            shape: torch.zeros(self.num_envs, 3, device=self.device)
            for shape in SHAPES
        }
        # Color currently assigned to each shape  (string, updated every reset)
        self._object_color = {shape: "red" for shape in SHAPES}

        self._task_names = [""] * self.num_envs
        self._task_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)


        

    # -------------------------------------------------------------------------
    # Scene setup
    # -------------------------------------------------------------------------
    def _setup_scene(self) -> None:
        sim_utils.spawn_from_usd(
            "/World/envs/env_0/background",
            self.cfg.background.spawn,
        )

        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self._rgbd_camera = Camera(self.cfg.rgbd_camera)
        self.scene.sensors["rgbd_camera"] = self._rgbd_camera

        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(
                global_prim_paths=[self.cfg.terrain.prim_path]
            )

        light_cfg = sim_utils.DomeLightCfg(
            intensity=2000.0,
            color=(0.75, 0.75, 0.75),
        )
        light_cfg.func("/World/Light", light_cfg)

    # -------------------------------------------------------------------------
    # RL interface
    # -------------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:

        self._raw_actions = torch.clamp(actions, -1.0, 1.0)

        max_vx = 3.0
        max_vz = 0.75
        max_yawrate = torch.deg2rad(torch.tensor(90.0, device=self.device))
        max_inclination = torch.deg2rad(torch.tensor(22.0, device=self.device))

        vx_throttle = (self._raw_actions[:, 0] + 1.0) / 2.0
        angle = max_inclination * self._raw_actions[:, 1]
        v_x = vx_throttle * torch.cos(angle) * max_vx
        v_z = vx_throttle * torch.sin(angle) * max_vz / torch.sin(max_inclination)
        yaw_rate = self._raw_actions[:, 2] * max_yawrate

        lin_vel_b = torch.stack([v_x, torch.zeros_like(v_x), v_z], dim=1)
        ang_vel_b = torch.stack(
            [torch.zeros_like(yaw_rate), torch.zeros_like(yaw_rate), yaw_rate], dim=1
        )
        self._vel_commands = torch.cat(
            [
                quat_apply(self._robot.data.root_quat_w, lin_vel_b),
                quat_apply(self._robot.data.root_quat_w, ang_vel_b),
            ],
            dim=1,
        )

        # Spin propellers (visual / dynamics)
        for idx, vel in zip(self.propeller_indices, PROPELLER_VELOCITY_TARGETS):
            self._rotor_target_vel[:, idx] = vel
        

    def _apply_action(self) -> None:
        self._robot.write_root_velocity_to_sim(self._vel_commands)
        self._robot.set_joint_velocity_target(self._rotor_target_vel)

    def _get_observations(self) -> dict:

        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._desired_pos_w,
        )
        obs = torch.cat(
            (
                desired_pos_b,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        distance = torch.linalg.norm(
            self._desired_pos_w - self._robot.data.root_pos_w, dim=1
        )
        distance_mapped = 1.0 - torch.tanh(distance / 1.2)

        rewards = {
            "lin_vel": -0.05 * lin_vel * self.step_dt,
            "ang_vel": -1.0 * ang_vel * self.step_dt,
            "distance_to_goal": (
                distance_mapped
                * 15.0
                * self.step_dt
            ),
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        z = self._robot.data.root_pos_w[:, 2]
        died = torch.logical_or(z < MIN_HEIGHT, z > MAX_HEIGHT)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # ----- logging (uses previous episode's goal / rewards) -----
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()

        extras = {}
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)

        extras = {
            "Episode_Termination/died": torch.count_nonzero(
                self.reset_terminated[env_ids]
            ).item(),
            "Episode_Termination/time_out": torch.count_nonzero(
                self.reset_time_outs[env_ids]
            ).item(),
            "Metrics/final_distance_to_goal": final_distance_to_goal.item(),
        }
        self.extras["log"].update(extras)

        # ----- robot reset -----
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        self._raw_actions[env_ids] = 0.0
        self._vel_commands[env_ids] = 0.0

        n = len(env_ids)

        task_idx = int(torch.randint(0, len(TASKS), (1,), device=self.device).item())
        task = TASKS[task_idx]
        # self._task_name = task["instruction"]
        target_color = task["color"]
        target_shape = task["shape"]

        for env_id in env_ids.tolist():
            self._task_names[env_id] = task["instruction"]

        self._task_ids[env_ids] = task_idx


        # ----- assign colors and shapes shuffle the rest to form a new goal -----
        self._object_color[target_shape] = target_color
        remaining_colors = [c for c in COLORS if c != target_color]
        remaining_shapes = [s for s in SHAPES if s != target_shape]
        perm = torch.randperm(len(remaining_colors), device=self.device)
        for i, shape in enumerate(remaining_shapes):
            self._object_color[shape] = remaining_colors[int(perm[i].item())]

        # # ----- sample object positions -----
        # # This approach can produce occlusion
        xs = torch.empty(n, 3, device=self.device).uniform_(2.5, 3.0)
        ys = torch.empty(n, 3, device=self.device).uniform_(-1.5, 1.5)
        for i, shape in enumerate(SHAPES):
            self._object_pos_w[shape][env_ids, 0] = xs[:, i]
            self._object_pos_w[shape][env_ids, 1] = ys[:, i]
            self._object_pos_w[shape][env_ids, 2] = 1.5
            self._object_pos_w[shape][env_ids, :2] += self._terrain.env_origins[env_ids, :2]

        # ----- goal = target object, with x offset -----
        self._desired_pos_w[env_ids] = self._object_pos_w[target_shape][env_ids].clone()
        self._desired_pos_w[env_ids, 0] -= 1.0

        # ----- print task information -----
        for env_id in env_ids.tolist():
            desired_pos = self._desired_pos_w[env_id].cpu().numpy()

            print("\n========== Task Info ==========")
            print(f"Instruction: {task['instruction']}")
            print(f"Desired position: x={desired_pos[0]:.3f}, "
                f"y={desired_pos[1]:.3f}, "
                f"z={desired_pos[2]:.3f}")

            print(f"Object at desired position:")
            print(f"  Shape: {target_shape}")
            print(f"  Color: {target_color}")
            print("===============================")

        # ----- randomize robot start pose -----
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, 0] = torch.empty(n, device=self.device).uniform_(-3.5, -3.0)
        default_root_state[:, 1] = torch.empty(n, device=self.device).uniform_(-0.5, 0.5)
        default_root_state[:, 2] = torch.empty(n, device=self.device).uniform_(0.75, 1.5)
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        # ----- write robot state -----
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    # def _set_debug_vis_impl(self, debug_vis: bool):
    #     # create markers if necessary for the first time
    #     if debug_vis:
    #         if not hasattr(self, "goal_pos_visualizer"):
    #             marker_cfg = CUBOID_MARKER_CFG.copy()
    #             marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
    #             # -- goal pose
    #             marker_cfg.prim_path = "/Visuals/Command/goal_position"
    #             self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
    #         # set their visibility to true
    #         self.goal_pos_visualizer.set_visibility(True)
    #     else:
    #         if hasattr(self, "goal_pos_visualizer"):
    #             self.goal_pos_visualizer.set_visibility(False)

    # def _debug_vis_callback(self, event):
    #     # update the markers
    #     self.goal_pos_visualizer.visualize(self._desired_pos_w)

    # Update visualization callback
    def _set_debug_vis_impl(self, debug_vis: bool):
        if not debug_vis:
            for v in getattr(self, "shape_color_visualizers", {}).values():
                v.set_visibility(False)
            return

        if not hasattr(self, "shape_color_visualizers"):
            self.shape_color_visualizers = {}

            color_rgb = {
                "red": (1.0, 0.0, 0.0),
                "blue": (0.0, 0.0, 1.0),
                "green": (0.0, 1.0, 0.0),
            }
            geom_builders = {
                "cuboid": lambda mat: sim_utils.CuboidCfg(
                    size=(0.15, 0.15, 0.15), visual_material=mat
                ),
                "cylinder": lambda mat: sim_utils.CylinderCfg(
                    radius=0.06, height=0.20, visual_material=mat
                ),
                "cone": lambda mat: sim_utils.ConeCfg(
                    radius=0.10, height=0.20, visual_material=mat
                ),
            }

            for shape in SHAPES:
                for color in COLORS:
                    mat = sim_utils.PreviewSurfaceCfg(diffuse_color=color_rgb[color])
                    geom = geom_builders[shape](mat)
                    name = f"{color}_{shape}"
                    cfg = VisualizationMarkersCfg(
                        prim_path=f"/Visuals/Targets/{name}",
                        markers={name: geom},
                    )
                    self.shape_color_visualizers[(shape, color)] = VisualizationMarkers(cfg)

        # visibility is controlled every frame in the callback
        for v in self.shape_color_visualizers.values():
            v.set_visibility(False)


    def _debug_vis_callback(self, event):
        if not hasattr(self, "shape_color_visualizers"):
            return

        for shape in SHAPES:
            current_color = self._object_color[shape]  # set at reset
            for color in COLORS:
                key = (shape, color)
                vis = self.shape_color_visualizers[key]
                if color == current_color:
                    vis.set_visibility(True)
                    vis.visualize(translations=self._object_pos_w[shape])
                else:
                    vis.set_visibility(False)




   