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

## Features

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
│   └── source/_static/          # figures
├── source/isaaclab_tasks/.../   # Isaac Lab env, recorder, conversion
└── ...
```

Exact paths may vary depending on how this repo is packaged with Isaac Lab.

---

## Installation

### Requirements

- Linux (x86_64).
- Python 3.11.
- Isaac Sim 5.0.0 / Isaac Lab (for simulation and data collection).
- CUDA-capable GPU recommended.

### Clone and setup

```bash
git clone https://github.com/UPB-RAT/exp2vla.git
cd exp2vla
```

Follow the [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/) for the simulator environment, then install extra Python deps:

```bash
pip install lerobot huggingface_hub pillow numpy
```

For dataset visualization, install LeRobot’s dataset tools as described in the [LeRobot repository](https://github.com/huggingface/lerobot).

---

## Instructions

### 1. Download the dataset

**Hugging Face CLI:**

```bash
huggingface-cli download UPB-RAT/vla-drone-v0.1 \
  --repo-type dataset \
  --local-dir ./vla-drone-v0.1
```

**Python (Hugging Face Hub):**

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="UPB-RAT/vla-drone-v0.1",
    repo_type="dataset",
    local_dir="./vla-drone-v0.1",
)
```

**LeRobot API:**

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("UPB-RAT/vla-drone-v0.1")
print(dataset)
print(f"Episodes: {dataset.num_episodes}, frames: {len(dataset)}")
sample = dataset
print(sample.keys())
```

---

### 2. Visualize episodes

**Browser (LeRobot Space):**

Open:

```text
https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2FUPB-RAT%2Fvla-drone-v0.1%2Fepisode_0%3Ft%3D0
```

**Local CLI:**

```bash
lerobot-dataset-viz \
  --repo-id UPB-RAT/vla-drone-v0.1 \
  --episode-index 0
```

With a local copy:

```bash
lerobot-dataset-viz \
  --repo-id UPB-RAT/vla-drone-v0.1 \
  --root ./vla-drone-v0.1 \
  --episode-index 0
```

---

### 3. Dataset format (LeRobot v3.0)

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

### 4. Collect expert demonstrations (Isaac Lab)

Enable recording in the env config (example):

```python
ENABLE_CAMERA = True
DATA_COLLECTOR = True
ROOT_DATASET_PATH = "~/summer_ws/Dataset/exp2vla-dataset-v0"
MAX_DATASET_EPISODES = 10
```

Run your `play.py` / teleop / expert policy script so the env writes:

- `data/chunk-000/episode_XXXXXX.jsonl`
- `videos/.../episode_XXXXXX_frames/frame_XXXXX.jpg`
- `meta/meta.json`

Each run can use a timestamped folder (e.g. `exp2vla-dataset-v0_YYYYMMDD_HHMMSS`) to avoid overwriting previous collections.

---

### 5. Convert Isaac recordings → LeRobot

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

**Check footer magic:**

```bash
tail -c 4 /path/to/vla-drone-v0.1/data/chunk-000/file-000.parquet | xxd
# expect: 50 41 52 31  ("PAR1")
```

**Inspect via PyArrow:**

```python
import pyarrow.parquet as pq

t = pq.read_table("/path/to/vla-drone-v0.1/data/chunk-000/file-000.parquet")
print(t.num_rows, t.column_names)
```

---

### 6. Upload to Hugging Face

Set your token:

```bash
export HF_TOKEN=hf_xxxxxxxx   # write access to UPB-RAT
```

**Python API:**

```python
from huggingface_hub import HfApi
import os

api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(
    repo_id="UPB-RAT/vla-drone-v0.1",
    repo_type="dataset",
    private=False,
    exist_ok=True,
)
api.upload_folder(
    folder_path="/path/to/vla-drone-v0.1",
    repo_id="UPB-RAT/vla-drone-v0.1",
    repo_type="dataset",
)
```

**CLI:**

```bash
huggingface-cli upload UPB-RAT/vla-drone-v0.1 \
  /path/to/vla-drone-v0.1 \
  . --repo-type dataset
```

---

### 7. Policy checkpoint

Example policy weights:

- [`UPB-RAT/my_pi05_drone_policy_test`](https://huggingface.co/UPB-RAT/my_pi05_drone_policy_test)

**Download:**

```bash
huggingface-cli download UPB-RAT/my_pi05_drone_policy_test \
  --local-dir ./my_pi05_drone_policy_test
```

Use your training / evaluation scripts from this repository (or LeRobot policy APIs) to fine-tune and evaluate π₀.₅-style VLA policies.

---

## Simulation Environment (Summary)

- **Simulator:** Isaac Sim 5.0 + Isaac Lab.
- **Robot:** Quadcopter (UPB squeezable drone asset).
- **Camera:** RGB, RealSense D455–style intrinsics, 640×480 (configurable).
- **Scene:** Three objects (cuboid / cylinder / cone); colors and XY positions randomized each episode with minimum separation.
- **Task:** Natural-language instruction selects the target object; goal is offset along x from that object.
- **Actions:** Body-frame velocity-style commands mapped to sim root velocity.
- **Observations (policy):** velocities, gravity, relative goal (and RGB for VLA).

---

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

## License

Please refer to the license files in this repository and on the Hugging Face dataset/model cards for terms of use.

---

## Acknowledgements

This work was developed at the University of Paderborn (UPB-RAT). We thank the Isaac Lab and LeRobot communities for open tools that made data collection and VLA fine-tuning practical.

---
