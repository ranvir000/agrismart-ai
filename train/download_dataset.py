"""
AgriSmart AI — Dataset Downloader
Downloads PlantVillage dataset from Kaggle and prepares it for training.

Run: python train/download_dataset.py
Requires: kaggle.json in C:/Users/<you>/.kaggle/kaggle.json
Get it from: https://www.kaggle.com/settings -> API -> Create New Token
"""

import os
import sys
import zipfile
import shutil
import random
from pathlib import Path

DATASET_DIR = Path("train/plantvillage_dataset")
KAGGLE_DATASET = "abdallahalidev/plantvillage-dataset"
IMAGES_PER_CLASS = 200   # Use 200 images per class (38 classes = 7,600 total)
                          # Full dataset has ~1,400 images per class
                          # Subset keeps training fast on CPU (~45 min)

CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy", "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_", "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy", "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot", "Peach___healthy", "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy", "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy", "Tomato___Bacterial_spot",
    "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]


def check_kaggle():
    """Check if Kaggle API is set up."""
    kaggle_path = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_path.exists():
        print("\n" + "="*60)
        print("KAGGLE API KEY NOT FOUND")
        print("="*60)
        print("\nSteps to get your Kaggle API key:")
        print("1. Go to: https://www.kaggle.com/settings")
        print("2. Scroll to 'API' section")
        print("3. Click 'Create New Token'")
        print("4. Download kaggle.json")
        print(f"5. Move it to: {kaggle_path}")
        print("\nThen run this script again.")
        return False
    print("✅ Kaggle API key found!")
    return True


def download_dataset():
    """Download PlantVillage dataset from Kaggle."""
    print(f"\n📥 Downloading PlantVillage dataset from Kaggle...")
    print(f"   Dataset: {KAGGLE_DATASET}")
    print(f"   This is ~1.4 GB and may take 5-15 minutes depending on internet speed.\n")

    os.makedirs("train/downloads", exist_ok=True)

    result = os.system(f'kaggle datasets download -d {KAGGLE_DATASET} -p train/downloads --unzip')
    if result != 0:
        print("❌ Download failed. Check your internet connection and Kaggle API key.")
        sys.exit(1)
    print("✅ Download complete!")


def find_segmented_dir():
    """Find the segmented (color) images directory."""
    for root, dirs, files in os.walk("train/downloads"):
        for d in dirs:
            if "segmented" in d.lower() or "color" in d.lower():
                full = os.path.join(root, d)
                # Check if it has subdirs
                subdirs = [x for x in os.listdir(full) if os.path.isdir(os.path.join(full, x))]
                if len(subdirs) >= 30:
                    return full
    # Fallback: find any dir with 38 class subdirectories
    for root, dirs, files in os.walk("train/downloads"):
        subdirs = [x for x in dirs if os.path.isdir(os.path.join(root, x))]
        if len(subdirs) >= 35:
            return root
    return None


def prepare_dataset():
    """Create train/val split with subsampling."""
    print(f"\n📁 Preparing dataset (using {IMAGES_PER_CLASS} images per class)...")

    source_dir = find_segmented_dir()
    if not source_dir:
        print("❌ Could not find dataset directory structure.")
        print("   Expected: train/downloads/<name>/color/ or train/downloads/<name>/segmented/")
        sys.exit(1)

    print(f"   Source: {source_dir}")

    # Create train/val directories
    for split in ["train", "val"]:
        (DATASET_DIR / split).mkdir(parents=True, exist_ok=True)

    total_train = 0
    total_val = 0
    found_classes = 0

    for class_name in os.listdir(source_dir):
        class_src = Path(source_dir) / class_name
        if not class_src.is_dir():
            continue

        images = list(class_src.glob("*.jpg")) + list(class_src.glob("*.JPG")) + \
                 list(class_src.glob("*.png")) + list(class_src.glob("*.PNG"))

        if len(images) == 0:
            continue

        random.shuffle(images)
        selected = images[:IMAGES_PER_CLASS]

        val_count = max(1, int(len(selected) * 0.2))  # 20% validation
        train_imgs = selected[val_count:]
        val_imgs = selected[:val_count]

        # Copy to split dirs
        for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs)]:
            dest = DATASET_DIR / split_name / class_name
            dest.mkdir(parents=True, exist_ok=True)
            for img in split_imgs:
                shutil.copy2(img, dest / img.name)

        total_train += len(train_imgs)
        total_val += len(val_imgs)
        found_classes += 1
        print(f"   ✅ {class_name[:50]}: {len(train_imgs)} train, {len(val_imgs)} val")

    print(f"\n✅ Dataset ready!")
    print(f"   Classes: {found_classes}")
    print(f"   Training images: {total_train}")
    print(f"   Validation images: {total_val}")
    print(f"   Saved to: {DATASET_DIR}")


if __name__ == "__main__":
    print("="*60)
    print("  AgriSmart AI — PlantVillage Dataset Downloader")
    print("="*60)

    if DATASET_DIR.exists() and len(list(DATASET_DIR.glob("train/*"))) >= 30:
        print(f"\n✅ Dataset already prepared at: {DATASET_DIR}")
        print("   Delete 'train/plantvillage_dataset' to re-download.")
        sys.exit(0)

    if not check_kaggle():
        sys.exit(1)

    download_dataset()
    prepare_dataset()
    print("\n🎉 Ready to train! Run: python train/train.py")
