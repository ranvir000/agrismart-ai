"""
AgriSmart AI - MobileNetV2 transfer-learning trainer (v2)
=========================================================
Two-phase transfer learning on PlantVillage (38 classes):

  Phase 1  backbone frozen, only the new classifier head is trained
  Phase 2  last N backbone blocks unfrozen, cosine-decayed fine-tuning,
           early stopping on validation accuracy, best checkpoint kept

Outputs in train/outputs/:
  plant_disease_model.pth      best checkpoint (what export_onnx.py needs)
  class_names.json             class order the model was trained with
  metrics.json                 test-set numbers for the report
  classification_report.txt    per-class precision / recall / F1
  confusion_matrix.png         normalized confusion matrix
  training_curves.png          loss + accuracy per epoch
  training_log.csv             every epoch's numbers
  threshold_sweep.csv          confidence threshold vs coverage / accuracy
  model_card.md                one-page summary you can paste in the report

Usage:
  python train/train.py
  python train/train.py --epochs 30 --batch 48 --unfreeze-blocks 7
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import MobileNet_V2_Weights

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# Must match models/model_loader.py at inference time.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args():
    p = argparse.ArgumentParser(description="Train the AgriSmart crop-disease classifier")
    p.add_argument("--data", default="train/plantvillage_dataset")
    p.add_argument("--out", default="train/outputs")
    p.add_argument("--epochs", type=int, default=25, help="total epochs (phase 1 + phase 2)")
    p.add_argument("--phase1-epochs", type=int, default=3, help="epochs with a frozen backbone")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-finetune", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--unfreeze-blocks", type=int, default=5, help="backbone blocks unfrozen in phase 2")
    p.add_argument("--patience", type=int, default=5, help="early-stopping patience on val accuracy")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(args):
    """
    Eval transforms deliberately use a plain resize (no CenterCrop) so the
    validation numbers measure exactly what the Flask app does: PIL resize to
    224x224, then ImageNet normalisation.
    """
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.65, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    data_dir = Path(args.data)
    if not (data_dir / "train").is_dir():
        sys.exit(f"Dataset not found at {data_dir}. Run: python train/download_dataset.py")
    if not (data_dir / "val").is_dir():
        sys.exit(f"Missing {data_dir}/val - re-run python train/download_dataset.py")

    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=eval_tf)
    test_ds = None
    if (data_dir / "test").is_dir() and len(list((data_dir / "test").glob("*"))) > 0:
        test_ds = datasets.ImageFolder(data_dir / "test", transform=eval_tf)
    else:
        print("[WARN] No test split found - using the validation split for the final report.")
        print("       Re-run download_dataset.py to get an untouched test set.")

    if train_ds.classes != val_ds.classes:
        sys.exit("Train and val class folders differ - rebuild the dataset.")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)
    test_loader = (DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)
                   if test_ds else val_loader)

    print(f"Train: {len(train_ds)} images | Val: {len(val_ds)} | Test: {len(test_ds) if test_ds else 'n/a'}")
    print(f"Classes: {len(train_ds.classes)}")
    return train_loader, val_loader, test_loader, train_ds.classes


def build_model(num_classes):
    """Same architecture export_onnx.py rebuilds, so keep both in sync."""
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes),
    )
    return model


def freeze_backbone(model):
    for param in model.features.parameters():
        param.requires_grad = False


def unfreeze_last_blocks(model, n_blocks):
    freeze_backbone(model)
    blocks = list(model.features.children())[-n_blocks:]
    for block in blocks:
        for param in block.parameters():
            param.requires_grad = True


def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]


def run_epoch(model, loader, criterion, optimizer, scaler, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=(scaler is not None)):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs.detach().float(), dim=1)
            correct += (probs.argmax(1) == labels).sum().item()
            total += labels.size(0)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    return total_loss / total, 100.0 * correct / total, probs, labels


def top_k_accuracy(probs, labels, k):
    top = np.argsort(probs, axis=1)[:, -k:]
    hits = sum(1 for i, row in enumerate(top) if labels[i] in row)
    return 100.0 * hits / len(labels)


def expected_calibration_error(probs, labels, bins=15):
    """Mean |accuracy - confidence| across confidence bins (lower is better)."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == labels).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if not mask.any():
            continue
        ece += mask.mean() * abs(acc[mask].mean() - conf[mask].mean())
    return float(ece)


def threshold_sweep(probs, labels, thresholds=None):
    thresholds = thresholds or [round(t, 2) for t in np.arange(0.30, 0.96, 0.05)]
    rows = []
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    for t in thresholds:
        accepted = conf >= t
        rows.append({
            "threshold": float(t),
            "coverage": float(accepted.mean()),
            "accuracy_on_accepted": float((pred[accepted] == labels[accepted]).mean()) if accepted.any() else 0.0,
        })
    return rows


def plot_training_curves(log, out_dir):
    epochs = [r["epoch"] for r in log]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("AgriSmart AI - MobileNetV2 training history", fontsize=14, fontweight="bold")
    ax1.plot(epochs, [r["train_loss"] for r in log], "b-o", label="Training loss", linewidth=2)
    ax1.plot(epochs, [r["val_loss"] for r in log], "r-o", label="Validation loss", linewidth=2)
    ax1.set_title("Loss per epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-entropy loss")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(epochs, [r["train_acc"] for r in log], "b-o", label="Training accuracy", linewidth=2)
    ax2.plot(epochs, [r["val_acc"] for r in log], "r-o", label="Validation accuracy", linewidth=2)
    ax2.set_title("Accuracy per epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(0, 105)
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(probs, labels, class_names, out_dir):
    pred = probs.argmax(axis=1)
    cm = confusion_matrix(labels, pred)
    cm_norm = cm.astype("float") / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1)
    short = []
    for name in class_names:
        parts = name.split("___")
        crop = parts[0][:10]
        dis = parts[1][:12] if len(parts) > 1 else ""
        short.append(f"{crop}\n{dis}" if dis else crop)
    fig, ax = plt.subplots(figsize=(20, 16))
    sns.heatmap(cm_norm, annot=False, cmap="YlOrRd", xticklabels=short, yticklabels=short,
                ax=ax, linewidths=0.5, linecolor="gray")
    ax.set_title("AgriSmart AI - normalized confusion matrix (test split)", fontsize=13,
                 fontweight="bold", pad=15)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=120, bbox_inches="tight")
    plt.close()


def write_model_card(out_dir, args, class_names, metrics):
    test = metrics["test"]
    card = f"""# Model Card - AgriSmart AI crop-disease classifier

## Model
- Architecture: MobileNetV2 (ImageNet-pretrained), custom head
  (Dropout 0.3 -> Linear 1280->512 -> ReLU -> Dropout 0.2 -> Linear 512->{metrics['num_classes']})
- Method: two-phase transfer learning ({args.phase1_epochs} frozen epochs, then
  {args.unfreeze_blocks} unfrozen backbone blocks with cosine learning-rate decay)
- Input: {args.img_size}x{args.img_size} RGB, ImageNet mean/std normalisation
- Optimiser: AdamW, weight decay {args.weight_decay}, label smoothing {args.label_smoothing}
- Deployment: exported to ONNX and dynamically quantized to int8 (runs on CPU)

## Data
- PlantVillage, {metrics['num_classes']} classes across 14 crops
- Augmentation: random resized crop, flips, rotation, colour jitter

## Test-set results
- Top-1 accuracy: {test['accuracy']:.2f}%
- Top-3 accuracy: {test['top3_accuracy']:.2f}%
- Macro F1: {test['macro_f1']:.4f} | Weighted F1: {test['weighted_f1']:.4f}
- Expected calibration error: {test['ece']:.4f}
- Best validation accuracy during training: {metrics['best_val_accuracy']:.2f}%

## Limitations
- Only the {metrics['num_classes']} PlantVillage classes are supported; photos of other
  crops or non-plant images are designed to be rejected via a confidence threshold
  (see threshold_sweep.csv) rather than guessed.
- PlantVillage images were captured on plain backgrounds; real field photos with soil,
  clutter and variable lighting are harder, which is why the app also offers a
  cloud vision model as a second engine.

## Classes
{', '.join(class_names)}
"""
    (out_dir / "model_card.md").write_text(card)


def main():
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  AgriSmart AI - training")
    print("=" * 60)
    print(f"Device: {device} | epochs: {args.epochs} | batch: {args.batch}")

    train_loader, val_loader, test_loader, class_names = build_loaders(args)
    if class_names != sorted(class_names):
        sys.exit("Class folders are not alphabetical - the ONNX index mapping in "
                 "models/model_loader.py would break. Rename the folders.")

    model = build_model(len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler(device.type) if device.type == "cuda" else None

    log, best_val, best_epoch, epochs_without_improvement, best_state = [], 0.0, 0, 0, None
    started = time.time()

    # ---- Phase 1: frozen backbone -------------------------------------------
    freeze_backbone(model)
    head_optimizer = torch.optim.AdamW(trainable_parameters(model), lr=args.lr_head,
                                       weight_decay=args.weight_decay)
    print(f"\nPhase 1 - frozen backbone for {args.phase1_epochs} epoch(s)")

    # ---- Phase 2 optimiser is created lazily after unfreezing ----------------
    fine_tune_epochs = max(1, args.epochs - args.phase1_epochs)
    model_unfrozen = False

    for epoch in range(1, args.epochs + 1):
        if epoch == args.phase1_epochs + 1 and not model_unfrozen:
            unfreeze_last_blocks(model, args.unfreeze_blocks)
            optimizer = torch.optim.AdamW(trainable_parameters(model), lr=args.lr_finetune,
                                          weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=fine_tune_epochs)
            model_unfrozen = True
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"\nPhase 2 - unfroze last {args.unfreeze_blocks} blocks "
                  f"({trainable:,} trainable parameters)")

        active_optimizer = optimizer if model_unfrozen else head_optimizer

        train_loss, train_acc, _, _ = run_epoch(model, train_loader, criterion,
                                                active_optimizer, scaler, device, train=True)
        val_loss, val_acc, _, _ = run_epoch(model, val_loader, criterion, None, None, device, train=False)

        if model_unfrozen:
            scheduler.step()

        log.append({"epoch": epoch, "train_loss": round(train_loss, 4), "train_acc": round(train_acc, 2),
                    "val_loss": round(val_loss, 4), "val_acc": round(val_acc, 2)})
        print(f"Epoch {epoch:2d}/{args.epochs} | train loss {train_loss:.4f} acc {train_acc:5.2f}% "
              f"| val loss {val_loss:.4f} acc {val_acc:5.2f}%")

        if val_acc > best_val:
            best_val, best_epoch = val_acc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping: no val improvement for {args.patience} epochs.")
                break

    if best_state is None:
        sys.exit("Training produced no checkpoint.")

    model.load_state_dict(best_state)
    torch.save(best_state, out_dir / "plant_disease_model.pth")
    (out_dir / "class_names.json").write_text(json.dumps(class_names, indent=2))

    with open(out_dir / "training_log.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        writer.writeheader()
        writer.writerows(log)
    plot_training_curves(log, out_dir)

    # ---- Final evaluation on the untouched test split ------------------------
    print("\nEvaluating the best checkpoint on the test split...")
    _, test_acc, test_probs, test_labels = run_epoch(model, test_loader, criterion, None, None, device, train=False)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, test_probs.argmax(axis=1), average="macro", zero_division=0)
    w_precision, w_recall, w_f1, _ = precision_recall_fscore_support(
        test_labels, test_probs.argmax(axis=1), average="weighted", zero_division=0)

    sweep = threshold_sweep(test_probs, test_labels)
    metrics = {
        "num_classes": len(class_names),
        "best_val_accuracy": round(best_val, 2),
        "best_epoch": best_epoch,
        "epochs_run": len(log),
        "train_seconds": round(time.time() - started, 1),
        "test": {
            "accuracy": round(float(test_acc), 2),
            "top3_accuracy": round(top_k_accuracy(test_probs, test_labels, 3), 2),
            "precision_macro": round(float(precision), 4),
            "recall_macro": round(float(recall), 4),
            "macro_f1": round(float(f1), 4),
            "precision_weighted": round(float(w_precision), 4),
            "recall_weighted": round(float(w_recall), 4),
            "weighted_f1": round(float(w_f1), 4),
            "ece": round(expected_calibration_error(test_probs, test_labels), 4),
        },
        "thresholds": sweep,
        "class_names": class_names,
        "config": vars(args),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    with open(out_dir / "threshold_sweep.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["threshold", "coverage", "accuracy_on_accepted"])
        writer.writeheader()
        for row in sweep:
            writer.writerow({k: round(v, 4) for k, v in row.items()})

    report = precision_recall_fscore_support(test_labels, test_probs.argmax(axis=1), labels=range(len(class_names)))
    with open(out_dir / "classification_report.txt", "w") as fh:
        fh.write("AgriSmart AI - plant disease classifier\n")
        fh.write("=" * 62 + "\n")
        fh.write(f"Model: MobileNetV2, two-phase transfer learning\n")
        fh.write(f"Classes: {len(class_names)} | Test images: {len(test_labels)}\n")
        fh.write(f"Top-1 accuracy: {test_acc:.2f}% | Top-3: {top_k_accuracy(test_probs, test_labels, 3):.2f}%\n")
        fh.write(f"Macro F1: {f1:.4f} | Weighted F1: {w_f1:.4f}\n")
        fh.write(f"ECE: {metrics['test']['ece']:.4f}\n\n")
        fh.write(f"{'class':52s} {'prec':>6s} {'rec':>6s} {'f1':>6s} {'n':>6s}\n")
        fh.write("-" * 80 + "\n")
        for i, name in enumerate(class_names):
            support = int((test_labels == i).sum())
            fh.write(f"{name[:52]:52s} {report[0][i]:6.3f} {report[1][i]:6.3f} {report[2][i]:6.3f} {support:6d}\n")

    plot_confusion_matrix(test_probs, test_labels, class_names, out_dir)
    write_model_card(out_dir, args, class_names, metrics)

    print("\nTest-set results")
    print(f"  Top-1 accuracy : {metrics['test']['accuracy']:.2f}%")
    print(f"  Top-3 accuracy : {metrics['test']['top3_accuracy']:.2f}%")
    print(f"  Macro F1       : {metrics['test']['macro_f1']:.4f}")
    print(f"  ECE            : {metrics['test']['ece']:.4f}")
    print("\nSuggested confidence threshold (accuracy vs coverage):")
    for row in sweep:
        if row["threshold"] in (0.40, 0.45, 0.50, 0.60):
            print(f"  >= {row['threshold']:.2f}: accepts {row['coverage']*100:5.1f}% of photos, "
                  f"{row['accuracy_on_accepted']*100:5.2f}% correct on those")
    print(f"\nAll outputs written to {out_dir}")
    print("Next: python train/export_onnx.py")


if __name__ == "__main__":
    main()
