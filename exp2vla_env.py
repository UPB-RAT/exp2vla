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
# =============================================================================
# Runtime options
# =============================================================================
ENABLE_CAMERA = True
DATA_COLLECTOR = True
IS_ADDING_OBJECTS = True
ROOT_DATASET_PATH = "/home/summer_school/summer_ws/Dataset/exp2vla-dataset-v0"
MAX_DATASET_EPISODES = 1000



SHAPES = ("cuboid", "cylinder", "cone")
COLORS = ("red", "blue", "green")

# Position sampling ranges (relative to env origin)
OBJECT_X_RANGE = (1.5, 2.5)
OBJECT_Y_RANGE = (-1.5, 1.5)
OBJECT_Z = 1.375

GOAL_OFFSET = 1.0

# All possible task instructions: every (color, shape) pair
TASKS = [
    {"instruction": f"Fly to the {color} {shape}", "color": color, "shape": shape}
    for color in COLORS
    for shape in SHAPES
]

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
# Dataset recorder
# =============================================================================
class IsaacDatasetRecorder:
    """
    Lightweight dataset recorder.

    Layout produced:
      - meta/meta.json
      - data/chunk-000/episode_XXXXXX.jsonl
      - videos/chunk-000/<camera_key>/episode_XXXXXX_frames/frame_XXXXX.jpg
    """

    def __init__(
        self,
        root_path: str | Path,
        camera_key: str,
        video_width: int,
        video_height: int,
        allow_existing: bool = False,
    ):
        self.root = Path(os.path.expanduser(str(root_path)))
        if self.root.exists() and not allow_existing:
            raise FileExistsError(
                f"Dataset path already exists: {self.root}\n"
                "Delete the existing folder or change ROOT_DATASET_PATH."
            )

        self.camera_key = camera_key
        self.video_width = video_width
        self.video_height = video_height

        self.data_path = self.root / "data" / "chunk-000"
        self.videos_base = self.root / "videos" / "chunk-000" / camera_key
        self.meta_path = self.root / "meta"

        for path in (self.data_path, self.videos_base, self.meta_path):
            path.mkdir(parents=True, exist_ok=True)

        self.episode_index = 0
        self.frame_index = 0
        self.temp_image_folder: Path | None = None
        self.task_name = "default_task"

        self.episode_actions: list = []
        self.episode_observations: list = []
        self.episode_timestamps: list = []
        self.episode_extras: list = []

    def start_episode(self, task_name: str, fps: int) -> None:
        """Begin a new episode (clears buffers and prepares temp frame folder)."""
        self.task_name = task_name
        self.frame_index = 0
        self.episode_actions = []
        self.episode_observations = []
        self.episode_timestamps = []
        self.episode_extras = []
        self.episode_index = self._next_episode_index()
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
    ) -> None:
        """Record a single step (state + RGB frame)."""
        if self.temp_image_folder is None:
            raise RuntimeError("Cannot record step before start_episode().")

        self.episode_actions.append(np.asarray(action, dtype=float).tolist())
        self.episode_observations.append(np.asarray(observation, dtype=float).tolist())
        self.episode_timestamps.append(float(timestamp))
        self.episode_extras.append(extras or {})

        frame_path = self.temp_image_folder / f"frame_{self.frame_index:05d}.jpg"
        Image.fromarray(image_rgb_np).save(frame_path)
        self.frame_index += 1

    def finalize_episode(self) -> None:
        """Write jsonl + move frames into the final episode folder."""
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

    def discard_episode(self) -> None:
        """Remove temporary frames for the current (incomplete) episode."""
        if self.temp_image_folder is not None and self.temp_image_folder.exists():
            shutil.rmtree(self.temp_image_folder, ignore_errors=True)
        self.temp_image_folder = None

    def _next_episode_index(self) -> int:
        files = list(self.data_path.glob("episode_*.jsonl"))
        if not files:
            return 0
        indices = [int(f.stem.split("_")[1]) for f in files]
        return max(indices) + 1

    def _create_or_update_meta(self, fps: int) -> None:
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
            next_index = max((t.get("task_index", -1) for t in tasks), default=-1) + 1
            tasks.append({"task_name": self.task_name, "task_index": next_index})

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

        if IS_ADDING_OBJECTS:
            # One slot per shape
            self._object_pos_w = {
                shape: torch.zeros(self.num_envs, 3, device=self.device)
                for shape in SHAPES
            }
            # Color currently assigned to each shape  (string, updated every reset)
            self._object_color = {shape: "red" for shape in SHAPES}

            self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
            self._task_name = TASKS[0]["instruction"]

        if DATA_COLLECTOR:
            self._setup_dataset_recording()

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

        if ENABLE_CAMERA:
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
        if (
            DATA_COLLECTOR
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

        # ----- randomize robot start pose -----
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, 0] = torch.empty(n, device=self.device).uniform_(-3.5, -3.0)
        default_root_state[:, 1] = torch.empty(n, device=self.device).uniform_(-0.5, 0.5)
        default_root_state[:, 2] = torch.empty(n, device=self.device).uniform_(0.75, 1.5)
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        # ----- sample task first -----
        task_idx = int(torch.randint(0, len(TASKS), (1,), device=self.device).item())
        task = TASKS[task_idx]
        self._task_name = task["instruction"]
        target_color = task["color"]
        target_shape = task["shape"]

        # ----- assign colors: force task (color, shape), shuffle the rest -----
        self._object_color[target_shape] = target_color
        remaining_colors = [c for c in COLORS if c != target_color]
        remaining_shapes = [s for s in SHAPES if s != target_shape]
        perm = torch.randperm(len(remaining_colors), device=self.device)
        for i, shape in enumerate(remaining_shapes):
            self._object_color[shape] = remaining_colors[int(perm[i].item())]

        # # ----- sample object positions -----
        # # This approach can produce occlusion
        # xs = torch.empty(n, 3, device=self.device).uniform_(*OBJECT_X_RANGE)
        # ys = torch.empty(n, 3, device=self.device).uniform_(*OBJECT_Y_RANGE)
        # for i, shape in enumerate(SHAPES):
        #     self._object_pos_w[shape][env_ids, 0] = xs[:, i]
        #     self._object_pos_w[shape][env_ids, 1] = ys[:, i]
        #     self._object_pos_w[shape][env_ids, 2] = OBJECT_Z
        #     self._object_pos_w[shape][env_ids, :2] += self._terrain.env_origins[env_ids, :2]

        # Solution is to set min separation if possible
        # ----- sample object positions (with separation) -----
        xs, ys = self._sample_object_xy(n, min_dist=0.8)

        for i, shape in enumerate(SHAPES):
            self._object_pos_w[shape][env_ids, 0] = xs[:, i]
            self._object_pos_w[shape][env_ids, 1] = ys[:, i]
            self._object_pos_w[shape][env_ids, 2] = OBJECT_Z
            self._object_pos_w[shape][env_ids, :2] += self._terrain.env_origins[env_ids, :2]

        # ----- goal = target object, with x offset -----
        self._desired_pos_w[env_ids] = self._object_pos_w[target_shape][env_ids].clone()
        self._desired_pos_w[env_ids, 0] -= GOAL_OFFSET

        print(f"Task: {self._task_name}")
        print("Layout:", {s: self._object_color[s] for s in SHAPES})

        # ----- dataset: finalize old episode, start new one with *this* task -----
        if DATA_COLLECTOR and hasattr(self, "_recording_enabled"):
            self._handle_dataset_episode_reset(env_ids)

        # ----- write robot state -----
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    # -------------------------------------------------------------------------
    # Add objects for data collection
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Dataset recording
    # -------------------------------------------------------------------------
    def _setup_dataset_recording(self) -> None:
        if not ENABLE_CAMERA:
            raise RuntimeError("DATA_COLLECTOR=True requires ENABLE_CAMERA=True.")

        # unique folder per run: .../exp2vla-dataset-v0_20260726_152530
        base = Path(os.path.expanduser(ROOT_DATASET_PATH))
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_path = base.parent / f"{base.name}_{stamp}"

        self._recorder = IsaacDatasetRecorder(
            root_path=str(run_path),
            camera_key="observation.images.camera1",
            video_width=self.cfg.rgbd_camera.width,
            video_height=self.cfg.rgbd_camera.height,
            allow_existing=False,
        )
        print(f"[Dataset] Writing to: {run_path}")

        self._step_counter = 0
        self._episodes_saved = 0
        self._max_episodes_to_save = int(MAX_DATASET_EPISODES)
        self._recording_enabled = True
        self._skip_next_record = False
        self._step_dt = self._compute_dataset_step_dt()
        self._fps = int(round(1.0 / self._step_dt)) if self._step_dt > 0 else 50

        if not hasattr(self, "_task_name"):
            self._task_name = TASKS[0]["instruction"]

        self._recorder.start_episode(
            task_name=self._task_name,
            fps=self._fps,
        )


    def _compute_dataset_step_dt(self) -> float:
        try:
            return float(self.cfg.sim.dt) * int(getattr(self.cfg, "decimation", 1))
        except Exception:
            return 0.02

    def _handle_dataset_episode_reset(self, env_ids) -> None:
        """Finalize current episode and start a new one on full reset."""
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
            self._recorder.start_episode(
                fps=self._fps,
                task_name=self._task_name,
            )


    def _data_recorder(self, raw_actions: torch.Tensor) -> None:
        """Record one dataset step for environment 0."""
        if self._skip_next_record:
            self._skip_next_record = False
            self._step_counter += 1
            return

        env_id = 0
        timestamp = self._step_counter * self._step_dt

        # keep recorder task string in sync
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


    def _get_dataset_action(self, raw_actions: torch.Tensor, env_id: int) -> list:
        vx = raw_actions[env_id, 0].item()
        vz = raw_actions[env_id, 1].item()
        yaw_rate = raw_actions[env_id, 2].item()
        return [vx, 0.0, vz, 0.0, 0.0, yaw_rate]  # [vx, vy, vz, pitch rate, roll rate, yaw rate]


    def _get_dataset_observation(self, env_id: int) -> list:
        # Bug happens here. We can simplify this to have only relative pos from the agent to goal in body coordinate

        # lin = self._robot.data.root_lin_vel_b[env_id].detach().cpu().numpy()
        # ang = self._robot.data.root_ang_vel_b[env_id].detach().cpu().numpy()
        # return [
        #     float(lin[0]), float(lin[1]), float(lin[2]),
        #     float(ang[0]), float(ang[1]), float(ang[2]),
        # ]

        relative_pos_body_frame, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._desired_pos_w,
        )
        relative_pos_b = relative_pos_body_frame[env_id].detach().cpu().numpy()
        return [
            float(relative_pos_b[0]), float(relative_pos_b[1]), float(relative_pos_b[2])
        ]



    def _get_dataset_image(self, env_id: int) -> np.ndarray:
        if not ENABLE_CAMERA:
            raise RuntimeError("Dataset recording requires ENABLE_CAMERA=True.")
        img = self.scene.sensors["rgbd_camera"].data.output["rgb"][env_id]
        return img.detach().cpu().numpy().astype(np.uint8)


    def _get_dataset_extras(self, env_id: int) -> dict:
        """Layout + goal + robot state for the new shape/color system."""
        object_layout = {}
        for shape in SHAPES:
            object_layout[shape] = {
                "color": self._object_color[shape],
                "pos_w": self._tensor_to_float_list(self._object_pos_w[shape][env_id]),
            }

        return {
            "task": self._task_name,
            "object_layout": object_layout,
            "goal_pos_w": self._tensor_to_float_list(self._desired_pos_w[env_id]),
            "robot_pos_w": self._tensor_to_float_list(
                self._robot.data.root_pos_w[env_id]
            ),
            "robot_quat_w": self._tensor_to_float_list(
                self._robot.data.root_quat_w[env_id]
            ),
        }

    # This helper function is to make sure no occluded scenarios happening
    def _sample_object_xy(
        self,
        n: int,
        min_dist: float = 0.8,
        max_tries: int = 100,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample (n, 3) x and y positions inside OBJECT_X_RANGE / OBJECT_Y_RANGE
        with pairwise distance >= min_dist (in the env-local frame).
        """
        xs = torch.zeros(n, 3, device=self.device)
        ys = torch.zeros(n, 3, device=self.device)

        for env_i in range(n):
            ok = False
            for _ in range(max_tries):
                cand_x = torch.empty(3, device=self.device).uniform_(*OBJECT_X_RANGE)
                cand_y = torch.empty(3, device=self.device).uniform_(*OBJECT_Y_RANGE)
                # pairwise distances
                dx = cand_x.unsqueeze(0) - cand_x.unsqueeze(1)  # (3,3)
                dy = cand_y.unsqueeze(0) - cand_y.unsqueeze(1)
                dist = torch.sqrt(dx * dx + dy * dy)
                # ignore diagonal
                dist.fill_diagonal_(1e6)
                if dist.min() >= min_dist:
                    xs[env_i] = cand_x
                    ys[env_i] = cand_y
                    ok = True
                    break
            if not ok:
                # fallback: spread along y
                xs[env_i] = sum(OBJECT_X_RANGE) / 2.0
                ys[env_i] = torch.tensor(
                    [-min_dist, 0.0, min_dist], device=self.device
                ).clamp(OBJECT_Y_RANGE[0], OBJECT_Y_RANGE[1])

        return xs, ys

    @staticmethod
    def _tensor_to_float_list(tensor: torch.Tensor) -> list:
        return tensor.detach().cpu().numpy().astype(float).tolist()

    def _update_task_name(self, env_id: int = 0) -> str:
        """Return the current task instruction for the given env."""
        cube_id = int(self._active_cube_id[env_id].item())
        self._task_name = TASKS[cube_id]["instruction"]
        return self._task_name