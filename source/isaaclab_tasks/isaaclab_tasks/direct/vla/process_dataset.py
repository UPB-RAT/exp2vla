import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# =========================
# ===== Configuration =====
# =========================
VERSION = "0.2"

# Folder produced by IsaacDatasetRecorder
dataset_path = Path(
    os.path.expanduser(
        "/home/summer_school/summer_ws/Dataset/exp2vla-dataset-test_20260728_122355"
    )
).resolve()

# Local output folder
converted_dataset_path = Path(
    os.path.expanduser(f"/home/summer_school/summer_ws/Dataset/vla-drone-v{VERSION}")
).resolve()

# Camera resolution
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640


def convert_jsonl_to_parquet() -> None:
    data_path = dataset_path / "data" / "chunk-000"
    meta_path = dataset_path / "meta" / "meta.json"
    videos_path = dataset_path / "videos" / "chunk-000" / "observation.images.camera1"

    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta path not found: {meta_path}")

    # Remove previous conversion
    if converted_dataset_path.exists():
        shutil.rmtree(converted_dataset_path)

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    fps = int(metadata["fps"])

    jsonl_files = sorted(data_path.glob("episode_*.jsonl"))
    if not jsonl_files:
        print("No JSONL files found.")
        return

    features = {
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": [
                "vx",
                "vy",
                "vz",
                "pitch_rate",
                "roll_rate",
                "yaw_rate",
            ],
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
        repo_id="local/vla-drone",
        root=str(converted_dataset_path),
        features=features,
        fps=fps,
        image_writer_processes=3,
        image_writer_threads=4,
    )

    print(f"Found {len(jsonl_files)} episodes")
    print(f"Source: {dataset_path}")
    print(f"Output: {converted_dataset_path}")

    total_frames = 0

    for jsonl_file in jsonl_files:
        if jsonl_file.stat().st_size == 0:
            print(f"Warning: empty {jsonl_file.name}, skipping.")
            continue

        episode_index = int(jsonl_file.stem.split("_")[1])
        frames_folder = videos_path / f"episode_{episode_index:06d}_frames"

        if not frames_folder.exists():
            print(f"Warning: missing frames for episode {episode_index}")
            continue

        with open(jsonl_file, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        if not lines:
            continue

        first = json.loads(lines[0])
        task_name = first.get("task", "unknown")

        frame_index = 0
        added = 0

        for line in lines:
            frame_data = json.loads(line)

            image_path = frames_folder / f"frame_{frame_index:05d}.jpg"

            if not image_path.exists():
                print(f"Missing {image_path.name}, skipping.")
                frame_index += 1
                continue

            image = np.asarray(Image.open(image_path), dtype=np.uint8)

            if image.shape[:2] != (IMAGE_HEIGHT, IMAGE_WIDTH):
                print(
                    f"Warning: image shape {image.shape} "
                    f"!= ({IMAGE_HEIGHT}, {IMAGE_WIDTH}, 3)"
                )

            action = np.asarray(frame_data["action"], dtype=np.float32)
            obs_state = np.asarray(frame_data["observation.state"], dtype=np.float32)

            if action.shape != (6,):
                raise ValueError(f"Invalid action shape: {action.shape}")

            if obs_state.shape != (3,):
                raise ValueError(f"Invalid observation.state shape: {obs_state.shape}")

            dataset.add_frame(
                {
                    "action": action,
                    "observation.state": obs_state,
                    "observation.images.camera1": image,
                    "task": task_name,
                }
            )

            frame_index += 1
            added += 1
            total_frames += 1

        if added > 0:
            dataset.save_episode()
            print(
                f"Saved episode {episode_index:06d} "
                f"({added} frames) | task: {task_name}"
            )

    print("\nConversion complete!")
    print(f"Episodes: {len(jsonl_files)}")
    print(f"Frames:   {total_frames}")
    print(f"Saved to: {converted_dataset_path}")

    # Remove temporary images created during conversion
    images_dir = converted_dataset_path / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)


if __name__ == "__main__":
    convert_jsonl_to_parquet()