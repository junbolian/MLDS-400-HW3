# src/download_data.py
import os
import pathlib
import requests

DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

KAGGLE_FILES = ["train.csv", "test.csv", "gender_submission.csv"]
FALLBACK_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
FALLBACK_DEST = DATA_DIR / "titanic.csv"

def main():
    # 如果已有 Kaggle 三件套中的任意一个，就认为用户已手动下载，不再自动拉取
    existing = [f for f in KAGGLE_FILES if (DATA_DIR / f).exists()]
    if existing:
        print(f"[download_data] Detected local Kaggle files: {existing}. Skip download.")
        return

    if FALLBACK_DEST.exists():
        print(f"[download_data] Found existing file: {FALLBACK_DEST}")
        return

    print(f"[download_data] Downloading fallback Titanic CSV to {FALLBACK_DEST} ...")
    resp = requests.get(FALLBACK_URL, timeout=30)
    resp.raise_for_status()
    FALLBACK_DEST.write_bytes(resp.content)
    print(f"[download_data] Done. Size = {FALLBACK_DEST.stat().st_size} bytes.")

if __name__ == "__main__":
    main()
