"""
AgriSmart AI - PlantVillage downloader and 3-way splitter
=========================================================
Downloads PlantVillage from Kaggle, then creates a reproducible
train / val / test split (default 70 / 15 / 15) with a fixed seed.

Run:
  python train/download_dataset.py                # every image (~54,000)
  python train/download_dataset.py --per-class 400  # smaller, faster run
  python train/download_dataset.py --per-class 200  # quick smoke test

The downloader needs a free Kaggle token:
kaggle.com -> Settings -> API -> Create New Token -> save as ~/.kaggle/kaggle.json
"""

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

KAGGLE_DATASET = "abdallahalidev/plantvillage-dataset"
DATASET_DIR = Path("train/plantvillage_dataset")
DOWNLOAD_DIR = Path("train/downloads")
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".JPG", ".PNG"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--per-class", type=int, default=0,
                   help="images per class; 0 means use all (~1400 per class)")
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def check_kaggle():
    token = Path.home() / ".kaggle" / "kaggle.json"
    if token.exists():
        return True
    print("""
Kaggle API token not found.
  1. Open https://www.kaggle.com/settings
  2. API section -> Create New Token
  3. Save the downloaded kaggle.json to ~/.kaggle/kaggle.json
  4. Run this script again
""")
    return False


def download():
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {KAGGLE_DATASET} (~1.4 GB, 5-15 min)...")
    if os.system(f"kaggle datasets download -d {KAGGLE_DATASET} -p {DOWNLOAD_DIR} --unzip") != 0:
        sys.exit("Download failed - check your internet connection and Kaggle token.")


def find_color_dir():
    """Locate the RGB ('color') image folder with >= 35 class subfolders."""
    best = None
    for root, dirs, _ in os.walk(DOWNLOAD_DIR):
        if "color" in os.path.basename(root).lower():
            subdirs = [d for d in dirs if os.path.isdir(os.path.join(root, d))]
            if len(subdirs) >= 35:
                return Path(root)
            best = best or Path(root)
    if best:
        return best
    for root, dirs, _ in os.walk(DOWNLOAD_DIR):
        if len([d for d in dirs if os.path.isdir(os.path.join(root, d))]) >= 35:
            return Path(root)
    return None


def prepare(per_class, val_frac, test_frac, seed):
    source = find_color_dir()
    if source is None:
        sys.exit("Could not find the dataset class folders. Delete train/downloads and retry.")
    print(f"Source: {source}")

    random.seed(seed)
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    for split in ("train", "val", "test"):
        (DATASET_DIR / split).mkdir(parents=True)

    summary = {"seed": seed, "per_class": per_class, "splits": {}, "classes": {}}
    totals = {"train": 0, "val": 0, "test": 0}

    for class_dir in sorted(p for p in source.iterdir() if p.is_dir()):
        images = sorted(f for f in class_dir.iterdir() if f.suffix in IMAGE_EXT)
        if not images:
            continue
        random.shuffle(images)
        if per_class > 0:
            images = images[:per_class]

        n = len(images)
        n_test = max(1, int(n * test_frac))
        n_val = max(1, int(n * val_frac))
        parts = {
            "test": images[:n_test],
            "val": images[n_test:n_test + n_val],
            "train": images[n_test + n_val:],
        }

        for split, files in parts.items():
            dest = DATASET_DIR / split / class_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, dest / f.name)
            totals[split] += len(files)
            summary["classes"].setdefault(class_dir.name, {})[split] = len(files)

    summary["splits"] = totals
    (DATASET_DIR / "dataset_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nClasses: {len(summary['classes'])}")
    print(f"Train: {totals['train']} | Val: {totals['val']} | Test: {totals['test']}")
    print(f"Saved to {DATASET_DIR} (split seed {seed})")
    print("\nNext: python train/train.py")


def main():
    args = parse_args()
    if DATASET_DIR.is_dir() and len(list((DATASET_DIR / "train").glob("*"))) >= 30:
        print(f"Dataset already prepared at {DATASET_DIR}")
        print("Delete that folder to rebuild it.")
        return
    if not check_kaggle():
        sys.exit(1)
    download()
    prepare(args.per_class, args.val, args.test, args.seed)


if __name__ == "__main__":
    main()
