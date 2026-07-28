import os
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq


VERSION = "0.2"
HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_REPO = "UPB-RAT/vla-drone-v0.2"

converted_dataset_path = Path(
    os.path.expanduser(f"/home/summer_school/summer_ws/Dataset/vla-drone-v{VERSION}")
).resolve()

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not set.")


def validate_local_parquet_files(root: Path) -> list[Path]:
    parquet_files = sorted(root.rglob("*.parquet"))
    if not parquet_files:
        raise RuntimeError(f"No parquet files found under {root}")

    bad_files = []
    for pf_path in parquet_files:
        try:
            meta = pq.ParquetFile(pf_path).metadata
            print(f"OK: {pf_path.relative_to(root)} | rows={meta.num_rows} row_groups={meta.num_row_groups}")
        except Exception as e:
            print(f"CORRUPT: {pf_path.relative_to(root)} | {e}")
            bad_files.append(pf_path)

    if bad_files:
        raise RuntimeError(f"Found {len(bad_files)} corrupted parquet file(s) locally. Fix before uploading.")

    return parquet_files


def verify_remote_parquet_files(repo_id: str, parquet_files: list[Path], root: Path) -> None:
    for pf_path in parquet_files:
        path_in_repo = str(pf_path.relative_to(root)).replace(os.sep, "/")
        try:
            downloaded = hf_hub_download(repo_id=repo_id, filename=path_in_repo, repo_type="dataset")
            meta = pq.ParquetFile(downloaded).metadata
            print(f"REMOTE OK: {path_in_repo} | rows={meta.num_rows} row_groups={meta.num_row_groups}")
        except Exception as e:
            print(f"REMOTE CORRUPT: {path_in_repo} | {e}")


print("Validating local parquet files before upload...")
parquet_files = validate_local_parquet_files(converted_dataset_path)

api = HfApi(token=HF_TOKEN)

api.create_repo(
    repo_id=DATASET_REPO,
    repo_type="dataset",
    private=False,
    exist_ok=True,
)

api.upload_folder(
    folder_path=str(converted_dataset_path),
    repo_id=DATASET_REPO,
    repo_type="dataset",
)

print(f"Uploaded to: https://huggingface.co/datasets/{DATASET_REPO}")

print("\nVerifying uploaded parquet files on the Hub...")
verify_remote_parquet_files(DATASET_REPO, parquet_files, converted_dataset_path)