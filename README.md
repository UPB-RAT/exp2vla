<div align="center">

# Exp2VLA: Enabling Vision–Language–Action for Drone Navigation from Expert Demonstrations

**Van Huyen Dang** · **Kabilesh Rajendran** · **Erdi Sayar** · **Erdal Kayacan**  
University of Paderborn

[![Paper](https://img.shields.io/badge/Paper-arXiv%202607.03146-b31b1b.svg)](https://arxiv.org/abs/2607.03146)
[![Code](https://img.shields.io/badge/Code-GitHub-181717.svg?logo=github)](https://github.com/UPB-RAT/exp2vla)
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

- **Language-conditioned drone navigation** in Isaac Lab / Isaac Sim
- **Multi-object scenes**: cuboid, cylinder, and cone with randomized colors and positions
- **Task instructions** such as `"Fly to the red cuboid"`, `"Fly to the blue cylinder"`, `"Fly to the green cone"`
- **Expert demonstration collection** (RL / teleop) → LeRobot-compatible dataset
- **Fine-tuning** of compact VLA policies (e.g. π₀.₅-style) on expert data
- **Public dataset and model** on Hugging Face under [UPB-RAT](https://huggingface.co/UPB-RAT)

---

## Repository structure

```text
exp2vla/
├── README.md
├── docs/
│   └── source/_static/          # figures
├── source/isaaclab_tasks/.../   # Isaac Lab env, recorder, conversion
└── ...