# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import json
import shutil
from pathlib import Path

import gymnasium as gym
import torch
import numpy as np
from PIL import Image

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, AssetBase
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.sensors import CameraCfg, Camera
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg

##
# Pre-defined configs
##
from isaaclab_assets import UPB_SQUEEZABLE_DRONE_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip


# =============================================================================
# Runtime options
# =============================================================================

enable_camera = True

ROOT_DATASET_PATH = "~/summer_ws/Dataset/Eval/Exp2VLA-Pi05-MOv1"

DataCollector = False
max_dataset_episodes = 10

GOAL_OFFSET = 0.5  # 0.5 Needed when collecting Dataset


# =============================================================================
# UI
# =============================================================================

class QuadcopterEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: QuadcopterEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
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
    # env
    episode_length_s = 10
    decimation = 2
    action_space = 3
    observation_space = 12
    state_space = 0
    debug_vis = True

    ui_window_class_type = QuadcopterEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
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
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/summer_school/summer_ws/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/quadcopter/rat_lab/multicorridor/empty_lab.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=True,
            ),
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    # robot
    robot: ArticulationCfg = UPB_SQUEEZABLE_DRONE_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    rgbd_camera: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/base_link/front_camera",
        offset=CameraCfg.OffsetCfg(
            pos=(0.2, 0.0, 0.015),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
        ),
        update_period=0.02,
        width=640,
        height=480,
    )

    thrust_to_weight = 1.9
    moment_scale = 0.01

    # reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0

    ########## camera viewer setting ######################################
    # viewer: ViewerCfg = ViewerCfg(
    #     env_index=0,
    #     origin_type="env",
    #     resolution=(1280, 720),
    # )
    viewer: ViewerCfg = ViewerCfg(
        eye=(-6.5, 0.0, 3.5),
        lookat=(2.0, 0.0, 1.36),
        origin_type="env",
    )


# =============================================================================
# Dataset recorder
# =============================================================================

class IsaacDatasetRecorder:
    """
    Lightweight dataset recorder.

    Produces:
      - meta/meta.json
      - data/chunk-000/episode_XXXXXX.jsonl
      - videos/chunk-000/observation.images.camera1/episode_XXXXXX_frames/frame_XXXXX.jpg
    """

    def __init__(
        self,
        root_path,
        camera_key,
        video_width,
        video_height,
        allow_existing=False,
    ):
        self.root = Path(os.path.expanduser(root_path))

        if self.root.exists() and not allow_existing:
            raise FileExistsError(
                f"Dataset path already exists: {self.root}\n"
                f"Delete existing folder or change ROOT_DATASET_PATH"
            )

        self.camera_key = camera_key
        self.video_width = video_width
        self.video_height = video_height

        self.data_path = self.root / "data" / "chunk-000"
        self.videos_base = self.root / "videos" / "chunk-000" / camera_key
        self.meta_path = self.root / "meta"

        self._mkdir(self.data_path)
        self._mkdir(self.videos_base)
        self._mkdir(self.meta_path)

        self.episode_index = 0
        self.frame_index = 0
        self.temp_image_folder = None

        self.task_name = "default_task"
        self.episode_actions = []
        self.episode_observations = []
        self.episode_timestamps = []
        self.episode_extras = []

    def _mkdir(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)

    def start_episode(self, task_name: str, fps: int):
        """Start a new dataset episode."""

        self.task_name = task_name
        self.frame_index = 0

        self.episode_actions = []
        self.episode_observations = []
        self.episode_timestamps = []
        self.episode_extras = []

        self.episode_index = self._get_next_episode_index()
        self._create_or_update_meta(fps)

        temp_dir = self.root / "temp_episode_frames"

        if temp_dir.exists():
            shutil.rmtree(temp_dir)

        temp_dir.mkdir(parents=True, exist_ok=True)
        self.temp_image_folder = temp_dir

    def record_step(
        self,
        action: list | np.ndarray,
        observation: list | np.ndarray,
        timestamp: float,
        image_rgb_np: np.ndarray,
        extras: dict | None = None,
    ):
        """Record one dataset step."""

        if self.temp_image_folder is None:
            raise RuntimeError("Cannot record step before start_episode().")

        self.episode_actions.append(np.array(action, dtype=float).tolist())
        self.episode_observations.append(np.array(observation, dtype=float).tolist())
        self.episode_timestamps.append(float(timestamp))
        self.episode_extras.append(extras or {})

        frame_path = self.temp_image_folder / f"frame_{self.frame_index:05d}.jpg"
        Image.fromarray(image_rgb_np).save(frame_path)

        self.frame_index += 1

    def finalize_episode(self):
        """Finalize the current dataset episode."""

        if self.temp_image_folder is None:
            return

        if self.frame_index == 0:
            self.discard_episode()
            return

        out_file = self.data_path / f"episode_{self.episode_index:06d}.jsonl"

        lines = []
        for i in range(self.frame_index):
            frame_data = {
                "timestamp": self.episode_timestamps[i],
                "action": self.episode_actions[i],
                "observation.state": self.episode_observations[i],
                "task": self.task_name,
                **self.episode_extras[i],
            }
            lines.append(json.dumps(frame_data))

        out_file.write_text("\n".join(lines))

        final_frames_folder = (
            self.videos_base / f"episode_{self.episode_index:06d}_frames"
        )

        if final_frames_folder.exists():
            shutil.rmtree(final_frames_folder)

        shutil.move(str(self.temp_image_folder), str(final_frames_folder))
        self.temp_image_folder = None

    def discard_episode(self):
        """Discard the current temporary episode."""

        if self.temp_image_folder and self.temp_image_folder.exists():
            shutil.rmtree(self.temp_image_folder, ignore_errors=True)

        self.temp_image_folder = None

    def _get_next_episode_index(self) -> int:
        files = list(self.data_path.glob("episode_*.jsonl"))

        if not files:
            return 0

        indices = [int(f.stem.split("_")[1]) for f in files]
        return max(indices) + 1

    def _create_or_update_meta(self, fps: int):
        meta_file = self.meta_path / "meta.json"

        if meta_file.exists():
            metadata = json.loads(meta_file.read_text())
        else:
            metadata = {
                "dataset_name": self.root.name,
                "fps": fps,
                "video_width": self.video_width,
                "video_height": self.video_height,
                "tasks": [],
            }

        tasks = metadata.get("tasks", [])

        if not any(t.get("task_name") == self.task_name for t in tasks):
            next_index = max(
                (t.get("task_index", -1) for t in tasks),
                default=-1,
            ) + 1

            tasks.append(
                {
                    "task_name": self.task_name,
                    "task_index": next_index,
                }
            )

        metadata["tasks"] = tasks
        metadata["fps"] = fps

        meta_file.write_text(json.dumps(metadata, indent=4))


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

        # Controller parameters
        self.max_speed = 2.0
        self.max_yawrate = torch.deg2rad(torch.tensor(15.0, device=self.device))
        self.max_inclination_angle = torch.deg2rad(
            torch.tensor(30, device=self.device)
        )

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(
            self.num_envs,
            gym.spaces.flatdim(self.single_action_space),
            device=self.device,
        )
        self._raw_actions = torch.zeros(
            self.num_envs,
            gym.spaces.flatdim(self.single_action_space),
            device=self.device,
        )

        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        self.propeller_indices = self._robot.find_joints(
            [
                "PropellerJoint1",
                "PropellerJoint2",
                "PropellerJoint3",
                "PropellerJoint4",
            ]
        )[0]

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
            ]
        }

        # Get specific body indices
        self._body_id = self._robot.find_bodies("base_link")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(
            self.sim.cfg.gravity,
            device=self.device,
        ).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # Cube properties
        self._cube_pos_w = {
            "red": torch.zeros(self.num_envs, 3, device=self.device),
            "blue": torch.zeros(self.num_envs, 3, device=self.device),
            "green": torch.zeros(self.num_envs, 3, device=self.device),
        }

        # Which cube is the goal: 0=red, 1=blue, 2=green
        self._active_cube_id = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )

        self._cube_colors = ["red", "blue", "green"]

        self._target_shape_name = {
            "red": "cube",
            "blue": "cylinder",
            "green": "cone",
        }

        # Debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

        # Dataset recording
        if DataCollector:
            self._setup_dataset_recording()

    # -------------------------------------------------------------------------
    # Scene setup
    # -------------------------------------------------------------------------

    def _setup_scene(self):
        sim_utils.spawn_from_usd(
            "/World/envs/env_0/background",
            self.cfg.background.spawn,
        )

        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        if enable_camera:
            self._rgbd_camera = Camera(self.cfg.rgbd_camera)
            self.scene.sensors["rgbd_camera"] = self._rgbd_camera

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        # explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(
                global_prim_paths=[self.cfg.terrain.prim_path]
            )

        # add lights
        light_cfg = sim_utils.DomeLightCfg(
            intensity=2000.0,
            color=(0.75, 0.75, 0.75),
        )
        light_cfg.func("/World/Light", light_cfg)

    # -------------------------------------------------------------------------
    # RL methods
    # -------------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        with torch.no_grad():
            self._raw_actions = actions.clone()

            clamped_action = torch.clamp(actions, -1.0, 1.0)
            clamped_action[:, 0] += 1.0

            angle = self.max_inclination_angle * clamped_action[:, 1]
            cos_angle = torch.cos(angle)
            sin_angle = torch.sin(angle)

            self._actions = torch.zeros(
                self.num_envs,
                gym.spaces.flatdim(self.single_action_space),
                device=self.device,
            )

            self._actions[:, 0] = (
                clamped_action[:, 0] * cos_angle * (self.max_speed / 2.0)
            )
            self._actions[:, 1] = (
                clamped_action[:, 0] * sin_angle * (self.max_speed / 4.0)
            )
            self._actions[:, 2] = clamped_action[:, 2] * self.max_yawrate

            # Extract and build linear velocity in body frame [vx, vy, vz]
            lin_vel_b = torch.stack(
                (
                    self._actions[:, 0],
                    torch.zeros_like(self._actions[:, 0]),
                    self._actions[:, 1],
                ),
                dim=1,
            )

            # Rotate to world frame
            lin_vel_w = self.quat_rotate_b2w(lin_vel_b)

            yaw_rate = self._actions[:, 2:3]
            zeros = torch.zeros_like(yaw_rate)
            ang_vel_w = torch.cat([zeros, zeros, yaw_rate], dim=1)

            self._vel_commands = torch.cat((lin_vel_w, ang_vel_w), dim=1)

            target_vel = torch.zeros(
                self.num_envs,
                self._robot.num_joints,
                device=self.device,
            )
            target_vel[:, self.propeller_indices[0]] = -130.0
            target_vel[:, self.propeller_indices[1]] = 130.0
            target_vel[:, self.propeller_indices[2]] = -130.0
            target_vel[:, self.propeller_indices[3]] = 130.0

            self._robot.set_joint_velocity_target(target_vel)

    def _apply_action(self):
        self._robot.write_root_velocity_to_sim(self._vel_commands)

    def _get_observations(self) -> dict:
        if (
            DataCollector
            and hasattr(self, "_recording_enabled")
            and self._recording_enabled
        ):
            self._data_recorder(self._raw_actions)

        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._desired_pos_w,
        )

        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                desired_pos_b,
            ],
            dim=-1,
        )

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(
            torch.square(self._robot.data.root_lin_vel_b),
            dim=1,
        )
        ang_vel = torch.sum(
            torch.square(self._robot.data.root_ang_vel_b),
            dim=1,
        )

        distance_to_goal = torch.linalg.norm(
            self._desired_pos_w - self._robot.data.root_pos_w,
            dim=1,
        )
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 1.2)

        rewards = {
            "lin_vel": -0.05 * lin_vel * self.step_dt,
            "ang_vel": -1 * ang_vel * self.step_dt,
            "distance_to_goal": (
                distance_to_goal_mapped
                * self.cfg.distance_to_goal_reward_scale
                * self.step_dt
            ),
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        died = torch.logical_or(
            self._robot.data.root_pos_w[:, 2] < 0.25,
            self._robot.data.root_pos_w[:, 2] > 2.0,
        )

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        # Dataset handling.
        # Guard with hasattr because Isaac Lab may call reset during initialization.
        if DataCollector and hasattr(self, "_recording_enabled"):
            self._handle_dataset_episode_reset(env_ids)

        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids],
            dim=1,
        ).mean()

        extras = dict()

        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = (
                episodic_sum_avg / self.max_episode_length_s
            )
            self._episode_sums[key][env_ids] = 0.0

        self.extras["log"] = dict()
        self.extras["log"].update(extras)

        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(
            self.reset_terminated[env_ids]
        ).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(
            self.reset_time_outs[env_ids]
        ).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        self._actions[env_ids] = 0.0

        # Debug prints
        eid = env_ids[0].item()

        print("*******************************************************")
        print("***********NEW EXC START*******************************")
        print("*******************************************************")
        print("Object layout:")

        for c in self._cube_colors:
            p = self._cube_pos_w[c][eid]
            print(f"  {c}: x={p[0]:.2f}, y={p[1]:.2f}, z={p[2]:.2f}")

        print(
            f"agent pos before reset: "
            f"{torch.round(self._robot.data.root_pos_w * 1000) / 1000}"
        )

        # Fixed cube positions: x=2, y in [-1, 0, 1], z=1.375
        cube_y_positions = [-1.0, 0.0, 1.0]

        for i, color in enumerate(self._cube_colors):
            self._cube_pos_w[color][env_ids, 0] = 2.0
            self._cube_pos_w[color][env_ids, 1] = cube_y_positions[i]
            self._cube_pos_w[color][env_ids, 2] = 1.375

        # Shift by environment origin
        for color in self._cube_colors:
            self._cube_pos_w[color][env_ids, :2] += (
                self._terrain.env_origins[env_ids, :2]
            )

        # Fixed drone start: x=-3, y=0, z=1.5
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]

        # Preserve original IsaacLab behavior.
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, 0] = -3.0
        default_root_state[:, 1] = 0.0
        default_root_state[:, 2] = 1.5
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        # Pick which cube is the target randomly each episode
        self._active_cube_id[env_ids] = torch.randint(
            0,
            3,
            (len(env_ids),),
            device=self.device,
        )

        for i, color in enumerate(self._cube_colors):
            mask = self._active_cube_id[env_ids] == i

            if mask.any():
                self._desired_pos_w[env_ids[mask]] = (
                    self._cube_pos_w[color][env_ids[mask]]
                )
                self._desired_pos_w[env_ids[mask], 0] -= GOAL_OFFSET

        self._update_task_name()

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    # -------------------------------------------------------------------------
    # Debug visualization
    # -------------------------------------------------------------------------

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "cube_visualizers"):
                self.cube_visualizers = {}

                color_map = {
                    "red": (1.0, 0.0, 0.0),
                    "blue": (0.0, 0.0, 1.0),
                    "green": (0.0, 1.0, 0.0),
                }

                for color in self._cube_colors:
                    mat = sim_utils.PreviewSurfaceCfg(
                        diffuse_color=color_map[color]
                    )

                    if color == "red":
                        marker_geom = sim_utils.CuboidCfg(
                            size=(0.15, 0.15, 0.15),
                            visual_material=mat,
                        )
                        marker_name = "red_cuboid"

                    elif color == "blue":
                        marker_geom = sim_utils.CylinderCfg(
                            radius=0.05,
                            height=0.2,
                            visual_material=mat,
                        )
                        marker_name = "blue_cylinder"

                    else:
                        marker_geom = sim_utils.ConeCfg(
                            radius=0.15,
                            height=0.15,
                            visual_material=mat,
                        )
                        marker_name = "green_cone"

                    cfg = VisualizationMarkersCfg(
                        markers={marker_name: marker_geom}
                    )
                    cfg.prim_path = f"/Visuals/Targets/{color}"

                    self.cube_visualizers[color] = VisualizationMarkers(cfg)

            for v in self.cube_visualizers.values():
                v.set_visibility(True)

        else:
            for v in getattr(self, "cube_visualizers", {}).values():
                v.set_visibility(False)

    def _debug_vis_callback(self, event):
        for color in self._cube_colors:
            pos = self._cube_pos_w[color].clone()
            self.cube_visualizers[color].visualize(pos)

    # -------------------------------------------------------------------------
    # Math helpers
    # -------------------------------------------------------------------------

    def quat_rotate_b2w(self, v):
        """
        Rotate vector(s) v by robot root quaternion.

        Quaternion convention:
            q = [w, x, y, z]
        """

        q_w = self._robot.data.root_quat_w[..., 0:1]
        q_vec = self._robot.data.root_quat_w[..., 1:4]

        t = 2.0 * torch.cross(q_vec, v, dim=-1)
        return v + q_w * t + torch.cross(q_vec, t, dim=-1)

    # -------------------------------------------------------------------------
    # Dataset recording setup
    # -------------------------------------------------------------------------

    def _setup_dataset_recording(self):
        """Initialize dataset recording without changing Isaac Lab simulation setup."""

        if not enable_camera:
            raise RuntimeError("DataCollector=True requires enable_camera=True.")

        self._recorder = IsaacDatasetRecorder(
            root_path=ROOT_DATASET_PATH,
            camera_key="observation.images.camera1",
            video_width=self.cfg.rgbd_camera.width,
            video_height=self.cfg.rgbd_camera.height,
        )

        self._step_counter = 0
        self._episodes_saved = 0
        self._max_episodes_to_save = int(max_dataset_episodes)
        self._recording_enabled = True
        self._skip_next_record = False

        self._step_dt = self._compute_dataset_step_dt()

        if self._step_dt > 0:
            self._fps = int(round(1.0 / self._step_dt))
        else:
            self._fps = 50

        self._update_task_name()

        self._recorder.start_episode(
            task_name=self._task_name,
            fps=self._fps,
        )

    def _compute_dataset_step_dt(self):
        try:
            sim_dt = float(self.cfg.sim.dt)
            decimation = int(getattr(self.cfg, "decimation", 1))
            return sim_dt * decimation
        except Exception:
            return 0.02

    def _handle_dataset_episode_reset(self, env_ids):
        """Finalize current dataset episode and start a new one when env resets."""

        if env_ids is not None and len(env_ids) != self.num_envs:
            return

        if self._recording_enabled and self._step_counter > 0:
            try:
                self._recorder.finalize_episode()
                self._episodes_saved += 1

                print(
                    f"Dataset episode saved: "
                    f"{self._episodes_saved}/{self._max_episodes_to_save}"
                )

            except Exception as e:
                print(f"Warning: failed to finalize episode: {e}")

            if self._episodes_saved >= self._max_episodes_to_save:
                self._recording_enabled = False

                print(
                    f"Reached {self._max_episodes_to_save} episodes. "
                    "Stopping dataset recording."
                )

        if self._recording_enabled:
            self._step_counter = 0
            self._skip_next_record = True

            self._update_task_name()

            self._recorder.start_episode(
                fps=self._fps,
                task_name=self._task_name,
            )

    # -------------------------------------------------------------------------
    # Dataset recording per-step logic
    # -------------------------------------------------------------------------

    def _data_recorder(self, raw_actions):
        """Record one dataset step for environment 0."""

        if self._skip_next_record:
            self._skip_next_record = False
            self._step_counter += 1
            return

        env_id = 0
        timestamp = self._step_counter * self._step_dt

        self._update_task_name()
        self._recorder.task_name = self._task_name

        action = self._get_dataset_action(raw_actions, env_id)
        observation = self._get_dataset_observation(env_id)
        image_rgb_np = self._get_dataset_image(env_id)
        extras = self._get_dataset_extras(env_id)

        if self._step_counter > 1:
            self._recorder.record_step(
                action=action,
                observation=observation,
                timestamp=timestamp,
                image_rgb_np=image_rgb_np,
                extras=extras,
            )

        self._step_counter += 1

    def _get_dataset_action(self, raw_actions, env_id: int):
        vx = raw_actions[env_id, 0].item()
        vz = raw_actions[env_id, 1].item()
        yaw = raw_actions[env_id, 2].item()

        return [
            vx,     # vx
            0.0,    # vy
            vz,     # vz
            0.0,    # pitch
            0.0,    # roll
            yaw,    # yaw
        ]

    def _get_dataset_observation(self, env_id: int):
        lin_vel_b = (
            self._robot.data.root_lin_vel_b[env_id]
            .detach()
            .cpu()
            .numpy()
        )
        ang_vel_b = (
            self._robot.data.root_ang_vel_b[env_id]
            .detach()
            .cpu()
            .numpy()
        )

        return [
            float(lin_vel_b[0]),
            float(lin_vel_b[1]),
            float(lin_vel_b[2]),
            float(ang_vel_b[0]),
            float(ang_vel_b[1]),
            float(ang_vel_b[2]),
        ]

    def _get_dataset_image(self, env_id: int):
        if not enable_camera:
            raise RuntimeError("Dataset recording requires enable_camera=True.")

        img = self.scene.sensors["rgbd_camera"].data.output["rgb"][env_id]

        return img.detach().cpu().numpy().astype(np.uint8)

    def _get_dataset_extras(self, env_id: int):
        cube_positions = {
            color: self._tensor_to_float_list(self._cube_pos_w[color][env_id])
            for color in self._cube_colors
        }

        active_cube_color = self._cube_colors[
            int(self._active_cube_id[env_id].item())
        ]

        return {
            "active_cube_color": active_cube_color,
            "cube_positions_w": cube_positions,
            "goal_pos_w": self._tensor_to_float_list(
                self._desired_pos_w[env_id]
            ),
            "robot_pos_w": self._tensor_to_float_list(
                self._robot.data.root_pos_w[env_id]
            ),
            "robot_quat_w": self._tensor_to_float_list(
                self._robot.data.root_quat_w[env_id]
            ),
        }

    def _tensor_to_float_list(self, tensor):
        return tensor.detach().cpu().numpy().astype(float).tolist()

    def _update_task_name(self):
        env_id = 0

        cube_id = self._active_cube_id[env_id].item()
        color = self._cube_colors[cube_id]
        shape = self._target_shape_name.get(color, "object")

        self._task_name = f"Fly to the {color} {shape}"