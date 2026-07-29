<div align="center">

# Exp2VLA: Enabling Vision–Language–Action for Drone Navigation from Expert Demonstrations

**Van Huyen Dang** · **Kabilesh Rajendran** · **Erdi Sayar** · **Erdal Kayacan**  
University of Paderborn

[![Paper](https://img.shields.io/badge/Paper-arXiv%202607.03146-b31b1b.svg)](https://arxiv.org/abs/2607.03146)
<!-- [![Code](https://img.shields.io/badge/Code-GitHub-181717.svg?logo=github)](https://github.com/UPB-RAT/exp2vla) -->
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-HuggingFace-yellow.svg)](https://huggingface.co/datasets/UPB-RAT/vla-drone-v0.1)
[![Model](https://img.shields.io/badge/🤗%20Model-HuggingFace-yellow.svg)](https://huggingface.co/UPB-RAT/my_pi05_drone_policy_test)
[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)

![Overview](docs/source/_static/Abstract_figure.drawio.png)

</div>

---

## Abstract

Vision-language-action (VLA) models open a new path toward intuitive robot control by directly linking perception, language, and action in a single end-to-end framework. Yet for UAVs, practical adoption remains difficult because existing solutions are either computationally heavy or insufficiently capable in complex environments. Exp2VLA proposes a practical expert-distillation pipeline for language-conditioned drone navigation, distilling expert behavior from reinforcement learning, teleoperation, or other controllers into training data that fine-tunes compact VLA models.

---

# Features

- Language-conditioned drone navigation in Isaac Lab / Isaac Sim.
- Multi-object scenes: cuboid, cylinder, and cone with randomized colors and positions.
- Task instructions such as `"Fly to the red cuboid"`, `"Fly to the blue cylinder"`, `"Fly to the green cone"`.
- Expert demonstration collection (RL / teleop) → LeRobot-compatible dataset.
- Fine-tuning of compact VLA policies (e.g. π₀.₅-style) on expert data.
- Public dataset and model on Hugging Face under [UPB-RAT](https://huggingface.co/UPB-RAT).

---

## Repository Structure

```text
exp2vla/
├── README.md
├── docs/
│   └──  # figures
├── scripts/reinforcement_learning/rl_games/
│   └──  play.py
    └──  train.py
    └──  play_vla.py
├── source/isaaclab_tasks/.../   # Isaac Lab env, recorder, conversion
          └──  isaaclab_assets/robots/sq_drone.py
          └──  isaaclab_tasks/isaaclab_tasks/direct/vla/....
└── ...
```

---

## Installation

### Requirements

- Linux (x86_64).
- Python 3.11.
- Isaac Sim 5.0.0 / Isaac Lab: Follow the [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/)
- CUDA-capable GPU recommended.
- Lerobot: Install LeRobot’s dataset tools as described in the [LeRobot repository](https://github.com/huggingface/lerobot).

---
## Instructions

### 1. Train an Expert Policy (Without Vision)

We use **RL Games** to train PPO agents, which serve as the expert policies for later stages.

The task `execise_00.py` is configured to train a PPO policy **without visual observations**.

Launch training with:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task execise_00 \
    --num_envs 1024 \
    --headless
```

After training completes, the checkpoints will be saved under:

```text
logs/rl_games/...
```

Before moving to the next step, verify that the trained policy works correctly by running:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py \
    --task execise_00 \
    --num_envs 1 \
    --checkpoint <path_to_checkpoint>
```


### 2. Add a Front-Facing Camera

After completing Step 1, add a front-facing RGB camera to the robot. This camera is required for collecting RGB observations for Vision-Language-Action (VLA) training.

Configure the camera as shown in `execise_01.py`:

```python
# --- Front RGB camera (Intel RealSense D455 RGB) ---
rgbd_camera: CameraCfg = CameraCfg(
    prim_path="/World/envs/env_.*/Robot/base_link/front_camera",
    offset=CameraCfg.OffsetCfg(
        pos=(0.2, 0.0, 0.015),
        rot=(0.5, -0.5, 0.5, -0.5),  # ROS convention
        convention="ros",
    ),
    data_types=["rgb"],  # Add "distance_to_image_plane" if depth images are needed.
    spawn=sim_utils.PinholeCameraCfg(
        # Parameters matching the Isaac Sim Intel RealSense D455 RGB camera
        focal_length=1.93,           # cm (real lens ≈ 1.93 mm; Isaac uses cm)
        focus_distance=0.6,          # m
        horizontal_aperture=3.896,   # cm (~86–90° horizontal FOV)
        clipping_range=(0.4, 10.0),  # m
        # Optional: f_stop=2.0
    ),
    update_period=1.0 / 30.0,        # 30 Hz
    width=640,
    height=480,
)
```

| Parameter | Value |
|-----------|-------|
| Sensor | Intel RealSense D455 RGB |
| Resolution | 640 × 480 |
| Frame rate | 30 Hz |
| Data type | `rgb` |
| Optional | Add `"distance_to_image_plane"` to `data_types` if depth images are required. |

After adding the camera, launch the environment and verify that RGB images are being published correctly before proceeding to data collection and VLA training.

### 3. Add Task Objects and Language Instructions

Next, we add the task objects and define the corresponding language instructions. This is implemented in `execise_02.py` (object spawning) and `execise_03.py` (task instruction generation).

In this example, we use **three object shapes** and **three colors**. Every color is paired with every shape, resulting in a total of **nine possible task instructions**.

```python
# Define available shapes and colors
SHAPES = ("cuboid", "cylinder", "cone")
COLORS = ("red", "blue", "green")

# Generate all possible task instructions
TASKS = [
    {
        "instruction": f"Fly to the {color} {shape}",
        "color": color,
        "shape": shape,
    }
    for color in COLORS
    for shape in SHAPES
]
```

At every environment reset:

- The objects are assigned random color-shape combinations.
- Their positions are randomized within the workspace.
- One of the valid `(color, shape)` pairs is randomly selected as the task.
- The corresponding language instruction (e.g., **"Fly to the red cone"**) is provided to the agent.

Randomizing both the scene and the language instruction at every reset encourages the policy to generalize across different object configurations instead of memorizing a fixed environment.

### 4. Add data collection function

#### 4.1. Dataset format (LeRobot v3.0)

**Fields:**

| Field                        | Type / shape      | Description                                        |
|-----------------------------|-------------------|----------------------------------------------------|
| `observation.images.camera1` | video 480×640×3   | Front RGB (RealSense D455–style)                   |
| `observation.state`          | float32 (3,)      | Relative goal in body frame `[rel_px, rel_py, rel_pz]` |
| `action`                     | float32 (6,)      | `[vx, vy, vz, pitch_rate, roll_rate, yaw_rate]`    |
| `task`                       | string            | Language goal, e.g. `"Fly to the red cuboid"`      |

**On-disk layout:**

```text
vla-drone-v0.1/
├── meta/
│   ├── info.json
│   ├── stats.json
│   ├── tasks.parquet
│   └── episodes/chunk-000/file-000.parquet
├── data/chunk-000/file-000.parquet
└── videos/observation.images.camera1/chunk-000/file-000.mp4
```

---

#### 4.2. Collect expert demonstrations (Isaac Lab)

Run your `play.py` / teleop / expert policy script
```bash
  ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py \
    --task execise_04 \
    --num_envs 1 \
    --enable_cameras \
    --checkpoint <path_to_checkpoint> 
```

so the env writes:

- `data/chunk-000/episode_XXXXXX.jsonl`
- `videos/.../episode_XXXXXX_frames/frame_XXXXX.jpg`
- `meta/meta.json`

Each run can use a timestamped folder (e.g. `exp2vla-dataset-v0_YYYYMMDD_HHMMSS`) to avoid overwriting previous collections.

---

## 5. Convert Isaac recordings → LeRobot

Use:

```bash
python source/isaaclab_tasks/isaaclab_tasks/direct/vla/process_dataset.py
```

In `process_dataset.py`, ensure:

- `dataset_path` points to the Isaac recording folder.
- `converted_dataset_path` is a different output folder.
- `IMAGE_HEIGHT` / `IMAGE_WIDTH` match the camera (e.g. 480×640).
- `observation.state` shape is `(3,)` if you record relative goal position.

After conversion, validate parquet locally.


## 6. Upload to Hugging Face

## 7. Fine-tune VLAs


---
<!-- 
## Simulation Environment (Summary)

- **Simulator:** Isaac Sim 5.0 + Isaac Lab.
- **Robot:** Quadcopter (UPB squeezable drone asset).
- **Camera:** RGB, RealSense D455–style intrinsics, 640×480 (configurable).
- **Scene:** Three objects (cuboid / cylinder / cone); colors and XY positions randomized each episode with minimum separation.
- **Task:** Natural-language instruction selects the target object; goal is offset along x from that object.
- **Actions:** Body-frame velocity-style commands mapped to sim root velocity.
- **Observations (policy):** velocities, gravity, relative goal (and RGB for VLA).

--- -->

## Citation

If you use this code, dataset, or models, please cite:

```bibtex
@article{dang2026exp2vla,
  title   = {Exp2VLA: Enabling Vision--Language--Action for Drone Navigation from Expert Demonstrations},
  author  = {Dang, Van Huyen and Rajendran, Kabilesh and Sayar, Erdi and Kayacan, Erdal},
  journal = {arXiv preprint arXiv:2607.03146},
  year    = {2026}
}
```

---

## Links

| Resource | URL |
|---------|-----|
| Paper   | https://arxiv.org/abs/2607.03146 |
| Code    | https://github.com/UPB-RAT/exp2vla |
| Dataset | https://huggingface.co/datasets/UPB-RAT/vla-drone-v0.1 |
| Model   | https://huggingface.co/UPB-RAT/my_pi05_drone_policy_test |
| Org     | https://huggingface.co/UPB-RAT |

---

<!-- ## License

Please refer to the license files in this repository and on the Hugging Face dataset/model cards for terms of use.

---

## Acknowledgements

This work was developed at the University of Paderborn (UPB-RAT). We thank the Isaac Lab and LeRobot communities for open tools that made data collection and VLA fine-tuning practical.

--- -->
