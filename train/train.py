"""
AgriSmart AI — MobileNetV2 Transfer Learning Training Script
============================================================
Model:   MobileNetV2 (pretrained on ImageNet)
Dataset: PlantVillage (38 classes, ~7,600 images subset)
Method:  Transfer Learning — frozen backbone + trained classifier head

Run:   python train/train.py
Output:
  - train/outputs/plant_disease_model.pth   (trained weights)
  - train/outputs/training_curves.png
  - train/outputs/confusion_matrix.png
  - train/outputs/classification_report.txt
  - train/outputs/training_log.csv
"""

import os, sys, time, csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import MobileNet_V2_Weights
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report,
                              precision_score, recall_score, f1_score)
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────
DATASET_DIR   = Path("train/plantvillage_dataset")
OUTPUT_DIR    = Path("train/outputs")
WEIGHTS_PATH  = OUTPUT_DIR / "plant_disease_model.pth"
NUM_EPOCHS    = 10          # 10 epochs gives good accuracy on subset
BATCH_SIZE    = 16          # Smaller batch = less RAM usage
LEARNING_RATE = 0.001
NUM_WORKERS   = 0           # 0 for Windows (avoids multiprocessing issues)
IMG_SIZE      = 224
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Data Transforms ──────────────────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_data():
    """Load train and validation datasets."""
    if not (DATASET_DIR / "train").exists():
        print("❌ Dataset not found!")
        print("   Run first: python train/download_dataset.py")
        sys.exit(1)

    train_dataset = datasets.ImageFolder(DATASET_DIR / "train", transform=train_transforms)
    val_dataset   = datasets.ImageFolder(DATASET_DIR / "val",   transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=NUM_WORKERS, pin_memory=False)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)

    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"✅ Loaded {len(train_dataset)} training images, {len(val_dataset)} validation images")
    print(f"   Classes: {num_classes}")

    return train_loader, val_loader, class_names, num_classes


def build_model(num_classes):
    """Build MobileNetV2 with custom classifier head."""
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

    # Freeze all backbone layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier head only (this is what we train)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes)
    )

    # Unfreeze last few backbone layers for better fine-tuning
    for layer in list(model.features.children())[-3:]:
        for param in layer.parameters():
            param.requires_grad = True

    total_params    = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Model: MobileNetV2")
    print(f"   Total parameters:     {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")

    return model.to(DEVICE)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds   = outputs.max(1)
        correct    += preds.eq(labels).sum().item()
        total      += labels.size(0)

        if (batch_idx + 1) % 20 == 0:
            print(f"   Batch {batch_idx+1}/{len(loader)} — "
                  f"Loss: {total_loss/total:.4f}, Acc: {100*correct/total:.1f}%", end="\r")

    return total_loss / total, 100.0 * correct / total


def validate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, preds   = outputs.max(1)
            correct    += preds.eq(labels).sum().item()
            total      += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / total, 100.0 * correct / total, all_preds, all_labels


def plot_training_curves(history):
    """Plot and save training/validation loss and accuracy curves."""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('AgriSmart AI — MobileNetV2 Training History', fontsize=14, fontweight='bold')

    # Loss curve
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'],   'r-o', label='Validation Loss', linewidth=2)
    ax1.set_title('Loss Curve')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(epochs)

    # Accuracy curve
    ax2.plot(epochs, history['train_acc'], 'b-o', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'],   'r-o', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Accuracy Curve')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(epochs)
    ax2.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {OUTPUT_DIR}/training_curves.png")


def plot_confusion_matrix(preds, labels, class_names):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Abbreviate class names for display
    short_names = []
    for name in class_names:
        parts = name.split("___")
        crop = parts[0][:8] if len(parts) > 0 else name
        dis  = parts[1][:10] if len(parts) > 1 else ""
        short_names.append(f"{crop}\n{dis}" if dis else crop)

    fig, ax = plt.subplots(figsize=(20, 16))
    sns.heatmap(cm_norm, annot=False, fmt='.0%', cmap='YlOrRd',
                xticklabels=short_names, yticklabels=short_names,
                ax=ax, linewidths=0.5, linecolor='gray')

    ax.set_title('AgriSmart AI — Confusion Matrix (Normalized)\nMobileNetV2 on PlantVillage Dataset',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.tick_params(axis='x', rotation=90, labelsize=7)
    ax.tick_params(axis='y', rotation=0, labelsize=7)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'confusion_matrix.png', dpi=120, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {OUTPUT_DIR}/confusion_matrix.png")


def save_classification_report(preds, labels, class_names):
    """Save detailed classification report."""
    report = classification_report(labels, preds, target_names=class_names, digits=4)
    precision = precision_score(labels, preds, average='weighted')
    recall    = recall_score(labels, preds, average='weighted')
    f1        = f1_score(labels, preds, average='weighted')

    with open(OUTPUT_DIR / 'classification_report.txt', 'w') as f:
        f.write("AgriSmart AI — Plant Disease Detection Model\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model:     MobileNetV2 (Transfer Learning)\n")
        f.write(f"Dataset:   PlantVillage (38 classes)\n")
        f.write(f"Device:    {DEVICE}\n")
        f.write(f"Epochs:    {NUM_EPOCHS}\n")
        f.write(f"Batch Size:{BATCH_SIZE}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Weighted Precision: {precision:.4f}\n")
        f.write(f"Weighted Recall:    {recall:.4f}\n")
        f.write(f"Weighted F1-Score:  {f1:.4f}\n\n")
        f.write("Per-Class Report:\n")
        f.write("-" * 60 + "\n")
        f.write(report)

    print(f"✅ Saved: {OUTPUT_DIR}/classification_report.txt")
    print(f"\n📊 Final Metrics:")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")


def main():
    print("=" * 60)
    print("  AgriSmart AI — Plant Disease Model Training")
    print("=" * 60)
    print(f"\n⚙️  Device: {DEVICE}")
    print(f"   Epochs: {NUM_EPOCHS}, Batch: {BATCH_SIZE}, LR: {LEARNING_RATE}")

    # Load data
    train_loader, val_loader, class_names, num_classes = load_data()

    # Build model
    model     = build_model(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    # Training loop
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc  = 0.0
    best_epoch    = 0
    log_rows      = []

    print(f"\n🚀 Starting training...\n")
    total_start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        print(f"Epoch [{epoch}/{NUM_EPOCHS}]")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss,   val_acc, val_preds, val_labels = validate(model, val_loader, criterion)
        scheduler.step()

        elapsed = time.time() - epoch_start
        print(f"\r   Train — Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"   Val   — Loss: {val_loss:.4f},   Acc: {val_acc:.2f}%  [{elapsed:.0f}s]")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        log_rows.append({
            'epoch': epoch, 'train_loss': round(train_loss, 4),
            'val_loss': round(val_loss, 4), 'train_acc': round(train_acc, 2),
            'val_acc': round(val_acc, 2), 'time_sec': round(elapsed, 1)
        })

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'class_names': class_names,
                'val_acc':     val_acc,
                'optimizer_state': optimizer.state_dict(),
            }, WEIGHTS_PATH)
            print(f"   💾 Saved best model (val_acc={val_acc:.2f}%)")
        print()

    total_time = (time.time() - total_start) / 60
    print(f"\n✅ Training complete in {total_time:.1f} minutes")
    print(f"   Best epoch: {best_epoch}, Best val accuracy: {best_val_acc:.2f}%")

    # Save training log CSV
    with open(OUTPUT_DIR / 'training_log.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)

    # Generate plots and report
    print("\n📊 Generating visualizations...")
    plot_training_curves(history)

    # Load best model for final evaluation
    checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    _, final_acc, final_preds, final_labels = validate(model, val_loader, criterion)
    print(f"\n📊 Final validation accuracy: {final_acc:.2f}%")

    plot_confusion_matrix(final_preds, final_labels, class_names)
    save_classification_report(final_preds, final_labels, class_names)

    print(f"\n🎉 All outputs saved to: {OUTPUT_DIR}/")
    print(f"   ├── plant_disease_model.pth  ← Load this in Flask app")
    print(f"   ├── training_curves.png")
    print(f"   ├── confusion_matrix.png")
    print(f"   ├── classification_report.txt")
    print(f"   └── training_log.csv")
    print(f"\n✅ Next: Copy model to models/ folder and update model_loader.py")


if __name__ == "__main__":
    main()
