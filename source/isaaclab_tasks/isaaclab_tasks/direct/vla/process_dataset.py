import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from huggingface_hub import HfApi
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# =========================
# ===== Configuration =====
# =========================
VERSION = "0.1"

# Folder produced by IsaacDatasetRecorder (timestamped run)
dataset_path = Path(
    os.path.expanduser(
        "/home/summer_school/summer_ws/Dataset/exp2vla-dataset-v0_20260726_165847"  # <- set real folder
    )
).resolve()

# LeRobot output (different folder!)
converted_dataset_path = Path(
    os.path.expanduser(f"/home/summer_school/summer_ws/Dataset/vla-drone-v{VERSION}")
).resolve()

HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_REPO = "UPB-RAT/vla-drone-v0.1"  # <- change
PRIVATE_REPO = False
UPLOAD_TO_HF = True

# Must match camera resolution used during collection
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640



def convert_jsonl_to_parquet(upload_to_hf: bool = False) -> None:
    data_path = dataset_path / "data" / "chunk-000"
    meta_path = dataset_path / "meta" / "meta.json"
    videos_path = dataset_path / "videos" / "chunk-000" / "observation.images.camera1"

    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta path not found: {meta_path}")

    # Clean previous conversion output
    if converted_dataset_path.exists():
        shutil.rmtree(converted_dataset_path)

    with open(meta_path, "r") as f:
        metadata = json.load(f)
    fps = int(metadata["fps"])

    jsonl_files = sorted(data_path.glob("episode_*.jsonl"))
    if not jsonl_files:
        print("No JSONL files found.")
        return

    # Features must match recorded arrays
    features = {
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["vx", "vy", "vz", "pitch_rate", "roll_rate", "yaw_rate"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (3,),
            "names": [
                "rel_px",
                "rel_py",
                "rel_pz",
            ],
        },
        "observation.images.camera1": {
            "dtype": "video",
            "shape": (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
            "names": ["height", "width", "channel"],
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=DATASET_REPO,
        root=str(converted_dataset_path),
        features=features,
        fps=fps,
        image_writer_processes=3,
        image_writer_threads=4,
    )

    print(f"Found {len(jsonl_files)} episodes")
    print(f"Source: {dataset_path}")
    print(f"Output: {converted_dataset_path}")

    for jsonl_file in jsonl_files:
        if jsonl_file.stat().st_size == 0:
            print(f"Warning: empty {jsonl_file.name}, skip")
            continue

        episode_index = int(jsonl_file.stem.split("_")[1])
        frames_folder = videos_path / f"episode_{episode_index:06d}_frames"

        if not frames_folder.exists():
            print(f"Warning: missing frames for episode {episode_index}, skip")
            continue

        with open(jsonl_file, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        if not lines:
            print(f"Warning: no lines in {jsonl_file.name}, skip")
            continue

        first = json.loads(lines[0])
        task_name = first.get("task", "unknown")

        frame_index = 0
        n_added = 0
        for line in lines:
            frame_data = json.loads(line)
            image_path = frames_folder / f"frame_{frame_index:05d}.jpg"

            if not image_path.exists():
                print(f"Warning: missing {image_path.name}, skip frame")
                frame_index += 1
                continue

            image_array = np.asarray(Image.open(image_path), dtype=np.uint8)

            # Optional sanity check
            if image_array.shape[:2] != (IMAGE_HEIGHT, IMAGE_WIDTH):
                print(
                    f"Warning: image shape {image_array.shape} "
                    f"!= expected ({IMAGE_HEIGHT}, {IMAGE_WIDTH}, 3)"
                )

            action = np.asarray(frame_data["action"], dtype=np.float32)
            obs_state = np.asarray(frame_data["observation.state"], dtype=np.float32)

            if action.shape != (6,):
                raise ValueError(f"action shape {action.shape}, expected (6,)")
            if obs_state.shape != (3,):
                raise ValueError(
                    f"observation.state shape {obs_state.shape}, expected (6,)"
                )

            dataset.add_frame(
                {
                    "action": action,
                    "observation.state": obs_state,
                    "observation.images.camera1": image_array,
                    "task": task_name,
                }
            )
            frame_index += 1
            n_added += 1

        if n_added == 0:
            print(f"Warning: episode {episode_index} had 0 frames, not saving")
            continue

        dataset.save_episode()
        print(f"Saved episode {episode_index:06d} ({n_added} frames) | task: {task_name}")

    print("\nConversion complete!")

    # Remove temporary images if LeRobot created any
    images_dir = converted_dataset_path / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)

    if not upload_to_hf:
        print("UPLOAD_TO_HF=False, skipping upload.")
        return

    if not HF_TOKEN:
        print("HF_TOKEN not set. Skipping upload.")
        return

    api = HfApi(token=HF_TOKEN)

    api.create_repo(
        repo_id=DATASET_REPO,      # "UPB-RAT/pi05-drone-v0.1"
        repo_type="dataset",
        private=False,             # public
        exist_ok=True,
    )

    api.upload_folder(
        folder_path=str(converted_dataset_path),
        repo_id=DATASET_REPO,
        repo_type="dataset",
    )

    print(f"Uploaded to: https://huggingface.co/datasets/{DATASET_REPO}")


if __name__ == "__main__":
    convert_jsonl_to_parquet(upload_to_hf=UPLOAD_TO_HF)