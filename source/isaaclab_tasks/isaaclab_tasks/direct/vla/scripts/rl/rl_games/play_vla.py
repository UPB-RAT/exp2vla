# Copyright (c) 2022-2025, The Isaac Lab Project Developers[](https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Updated for VLA model"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import sys
from isaaclab.app import AppLauncher

import torch
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.utils import build_inference_frame

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import math
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from rl_games.common import env_configurations, vecenv

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
# from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


# =========================================================
# ================= USER CONFIGURATION =====================
# =========================================================
TASKS = ["Fly to the green cone", "Fly to the blue cylinder", "Fly to the red cube"]
MODEL_TYPE = "pi05"   # "pi05" or "smolvla"
MODEL_ID = "/home/summer_school/summer_ws/vla_models/Exp2VLA-Pi05-MOv1"
HF_TOKEN = None  # For private repos
CHUNK_EXEC_LEN = 50
ACTION_IDXS = [0, 2, 5]
SEED = 42
NUM_EVAL_STEPS = 500  # Number of completed episodes (done=True) before stopping
# =========================================================


def load_vla_policy(model_type: str, model_id: str, device: torch.device, hf_token: str | None = None):
    """Load VLA model on GPU, force float16 via direct parameter cast."""
    model_type = model_type.lower().strip()

    if model_type == "pi05":
        print(f"Loading PI05 model: {model_id}")
        model = PI05Policy.from_pretrained(model_id, token=hf_token)
        model.to(device)

        # Force float16 at parameter level — most reliable approach
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.is_floating_point() and param.dtype == torch.float32:
                    param.data = param.data.to(torch.float16)
            for name, buf in model.named_buffers():
                if buf.is_floating_point() and buf.dtype == torch.float32:
                    buf.data = buf.data.to(torch.float16)

    else:
        raise ValueError(
            f"Unsupported MODEL_TYPE='{model_type}'. Use 'pi05' or 'smolvla'."
        )

    model.eval()
    print(f"[DTYPE] Model dtype after cast: {next(model.parameters()).dtype}")
    print(f"[MEM] After VLA cast: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated")
    return model


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"[INFO] Global seed set to {SEED}")
print(f"[INFO] MODEL_TYPE : {MODEL_TYPE}")
print(f"[INFO] MODEL_ID   : {MODEL_ID}")
print(f"[INFO] NUM_EVAL_STEPS : {NUM_EVAL_STEPS}")


def get_task(eval_count):
    """Cycle tasks every 10 episodes: 3x green, 3x blue, 4x red."""
    pos_in_group = eval_count % 10
    if pos_in_group < 6:
        return TASKS[0]  # green cone: episodes 0-2
    elif pos_in_group < 8:
        return TASKS[1]  # blue cylinder: episodes 3-5
    else:
        return TASKS[2]  # red cube: episodes 6-9


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Play with RL-Games agent."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    env_cfg.seed = agent_cfg["params"]["seed"]

    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    log_dir = "/home/huyen-admin/spear_upb_ws/IsaacLab/rl4vla_drone"
    env_cfg.log_dir = log_dir

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    # === Environment creation (this was failing before) ===
    print("[INFO] Creating environment...")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    print("[INFO] Environment created successfully!")

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    dt = env.unwrapped.step_dt

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]

    # NOW Isaac Sim has actually allocated GPU memory
    print(f"[MEM] After env.reset(): {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated")
    print(f"[MEM] After env.reset(): {torch.cuda.memory_reserved()/1024**3:.2f} GB reserved")

    timestep = 0

    # Isaac Sim runs on GPU
    sim_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Free GPU memory before loading the large VLA model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[MEM] After cache clear: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated")
    print(f"[MEM] After cache clear: {torch.cuda.memory_reserved()/1024**3:.2f} GB reserved")

    # ---------------- Load VLA model on GPU ----------------
    model = load_vla_policy(
        model_type=MODEL_TYPE,
        model_id=MODEL_ID,
        device=sim_device,
        hf_token=HF_TOKEN,
    )

    vla_device = next(model.parameters()).device  # cuda
    model_dtype = next(model.parameters()).dtype   # float16
    print(f"[INFO] VLA model loaded on: {vla_device} | dtype: {model_dtype}")
    print(f"[MEM] After VLA load: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated")
    print(f"[MEM] After VLA load: {torch.cuda.memory_reserved()/1024**3:.2f} GB reserved")

    # Preprocessors run on GPU
    preprocess, postprocess = make_pre_post_processors(
        model.config,
        MODEL_ID,
        preprocessor_overrides={"device_processor": {"device": str(sim_device)}},
    )

    # ---------------- Timing stats ----------------
    total_steps = 0
    loop_time_acc = 0.0
    model_time_acc = 0.0
    preprocess_time_acc = 0.0
    env_step_time_acc = 0.0
    report_every = 1000



    eval_count   = 0
    episode_step = 0
    done = torch.zeros(1, dtype=torch.bool)  # init for first iteration check

    print(f"[INFO] Starting evaluation for {NUM_EVAL_STEPS} episodes (ends when done=True)...")

    while simulation_app.is_running():
        loop_t0 = time.perf_counter()

        # Update task only at episode boundaries
        if done.any() or total_steps == 0:
            TASK = get_task(eval_count)
            print(f"[INFO] TASK       : {TASK}")
            print(f"[INFO] MODEL_TYPE : {MODEL_TYPE}")
            print(f"[INFO] MODEL_ID   : {MODEL_ID}")

        with torch.inference_mode():
            # Get RGB from GPU sensor, immediately pull to CPU
            rgb = env.unwrapped.scene.sensors["rgbd_camera"].data.output["rgb"][0].cpu()
            state_np = obs["obs"] if isinstance(obs, dict) else obs

            # State tensor on GPU for env computations
            state_tensor = torch.as_tensor(state_np, dtype=torch.float32, device=sim_device)
            s3 = state_tensor[0, :3].detach().cpu().numpy()

            # Build raw observation — all CPU/numpy
            raw_obs = {
                "rel_px":      float(s3[0]),
                "rel_py":      float(s3[1]),
                "rel_pz":      float(s3[2]),
                "camera1": rgb.numpy().astype(np.float32),
            }

            hw_obs_features = {
                "rel_px": float, "rel_py": float, "rel_pz": float,
                "camera1": (480, 640, 3),
            }
            ds_obs_features = hw_to_dataset_features(hw_obs_features, "observation")

            # ---- Preprocess on GPU ----
            t_pre0 = time.perf_counter()
            obs_frame = build_inference_frame(
                observation=raw_obs,
                ds_features=ds_obs_features,
                device=vla_device,  # cuda
                task=TASK,
            )
            vla_obs = preprocess(obs_frame)
            if vla_device.type == "cuda":
                torch.cuda.synchronize()
            preprocess_time_acc += time.perf_counter() - t_pre0

            # ---- Run model on GPU with autocast to handle float16/float32 mixing ----
            t_model0 = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                vla_action = model.select_action(vla_obs)
            if vla_device.type == "cuda":
                torch.cuda.synchronize()
            model_time_acc += time.perf_counter() - t_model0

            # ---- Postprocess on GPU ----
            vla_action = postprocess(vla_action)

            # extracted action already on GPU for env.step
            extracted_action = vla_action[:, ACTION_IDXS].squeeze(1)

            # ---- Step environment on GPU ----
            t_env0 = time.perf_counter()
            obs, _, done, _ = env.step(extracted_action)

            if isinstance(obs, dict):
                obs = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in obs.items()}
            else:
                obs = obs.detach()

            env_step_time_acc += time.perf_counter() - t_env0

            # Each done=True counts as 1 completed evaluation
            if done.any():
                eval_count += 1
                print(f"[EVAL] Episode done — {eval_count} / {NUM_EVAL_STEPS} evaluations completed.")
                if eval_count >= NUM_EVAL_STEPS:
                    break
                episode_step = 0

        # Update loop timers
        loop_dt = time.perf_counter() - loop_t0
        loop_time_acc += loop_dt
        total_steps  += 1
        episode_step += 1

        # Reporting FPS
        if total_steps % report_every == 0:
            real_fps        = total_steps / loop_time_acc
            preprocess_fps  = total_steps / preprocess_time_acc if preprocess_time_acc > 0 else 0.0
            model_fps       = total_steps / model_time_acc if model_time_acc > 0 else 0.0
            env_fps         = total_steps / env_step_time_acc if env_step_time_acc > 0 else 0.0
            print(
                f"[STEP {total_steps}] "
                f"real FPS={real_fps:.2f} | "
                f"preprocess FPS={preprocess_fps:.2f} | "
                f"model FPS={model_fps:.2f} | "
                f"env.step FPS={env_fps:.2f}"
            )

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        # Real-time sleep
        if args_cli.real_time:
            sleep_time = dt - loop_dt
            if sleep_time > 0:
                time.sleep(sleep_time)

    print(f"[EVAL] Done — completed {eval_count} / {NUM_EVAL_STEPS} evaluations over {total_steps} steps.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()