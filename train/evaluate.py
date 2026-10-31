"""
AgriSmart AI - evaluate the model the web app actually serves
============================================================
Loads models/weights/mobilenet_v2_plantvillage_quantized.onnx (or --onnx PATH),
runs it exactly like models/model_loader.py does, and reports:

  - top-1 / top-3 accuracy on a labelled split
  - macro + weighted precision / recall / F1
  - expected calibration error (is the stated confidence trustworthy?)
  - confidence-threshold sweep (coverage vs accuracy) -> justifies MIN_CONFIDENCE
  - optional: --ood-dir of non-plant photos and how many are correctly rejected

Usage:
  python train/evaluate.py
  python train/evaluate.py --split test --ood-dir train/ood_examples
  python train/evaluate.py --onnx train/outputs/plant_disease_model_quantized.onnx
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMG_SIZE = 224


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", default="models/weights/mobilenet_v2_plantvillage_quantized.onnx")
    p.add_argument("--data", default="train/plantvillage_dataset")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--ood-dir", default=None, help="folder of non-plant photos to test rejection")
    p.add_argument("--batch", type=int, default=16)
    return p.parse_args()


def preprocess(path):
    """Identical to models/model_loader.analyze_image."""
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(arr, (2, 0, 1))


def softmax(logits):
    exp = np.exp(logits - logits.max())
    return exp / exp.sum()


def collect(split_dir):
    classes = sorted([d.name for d in Path(split_dir).iterdir() if d.is_dir()])
    paths, labels = [], []
    for idx, name in enumerate(classes):
        for f in sorted((Path(split_dir) / name).iterdir()):
            if f.suffix.lower() in IMAGE_EXT:
                paths.append(f)
                labels.append(idx)
    return classes, paths, np.array(labels)


def infer_all(session, paths, batch):
    input_name = session.get_inputs()[0].name
    out = []
    for start in range(0, len(paths), batch):
        chunk = paths[start:start + batch]
        batch_arr = np.stack([preprocess(p) for p in chunk])
        logits = session.run(None, {input_name: batch_arr})[0]
        out.append(np.stack([softmax(row) for row in logits]))
    return np.concatenate(out)


def expected_calibration_error(probs, labels, bins=15):
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    acc = (pred == labels).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(acc[mask].mean() - conf[mask].mean())
    return float(ece)


def main():
    args = parse_args()
    split_dir = Path(args.data) / args.split
    if not split_dir.is_dir():
        raise SystemExit(f"{split_dir} not found - run python train/download_dataset.py first")
    if not Path(args.onnx).exists():
        raise SystemExit(f"{args.onnx} not found - run python train/export_onnx.py first")

    try:
        from models.model_loader import ONNX_INDEX_TO_DISEASE_KEY as DISEASE_KEYS
    except Exception:
        DISEASE_KEYS = None

    classes, paths, labels = collect(split_dir)
    print(f"Evaluating {args.onnx}")
    print(f"{len(paths)} images x {len(classes)} classes from {split_dir}")

    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    probs = infer_all(session, paths, args.batch)
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)

    top1 = 100.0 * float((pred == labels).mean())
    top3 = 100.0 * float(np.mean([labels[i] in np.argsort(probs[i])[-3:] for i in range(len(labels))]))
    prec, rec, f1, _ = precision_recall_fscore_support(labels, pred, average="macro", zero_division=0)
    w_prec, w_rec, w_f1, _ = precision_recall_fscore_support(labels, pred, average="weighted", zero_division=0)
    ece = expected_calibration_error(probs, labels)

    lines = []
    lines.append("AgriSmart AI - deployed ONNX model evaluation")
    lines.append("=" * 58)
    lines.append(f"Model file : {args.onnx}")
    lines.append(f"Split      : {args.split} ({len(paths)} images, {len(classes)} classes)")
    lines.append(f"Top-1 acc  : {top1:.2f}%")
    lines.append(f"Top-3 acc  : {top3:.2f}%")
    lines.append(f"Macro P/R/F1     : {prec:.4f} / {rec:.4f} / {f1:.4f}")
    lines.append(f"Weighted P/R/F1  : {w_prec:.4f} / {w_rec:.4f} / {w_f1:.4f}")
    lines.append(f"Calibration (ECE): {ece:.4f}")
    lines.append("")
    lines.append("Threshold sweep (what the app shows vs what it refuses):")
    lines.append(f"{'threshold':>10s} {'coverage':>10s} {'accuracy':>10s}")
    sweep = []
    for t in np.arange(0.30, 0.96, 0.05):
        accepted = conf >= t
        acc = float((pred[accepted] == labels[accepted]).mean() * 100) if accepted.any() else 0.0
        sweep.append({"threshold": round(float(t), 2),
                      "coverage": round(float(accepted.mean()) * 100, 2),
                      "accuracy": round(acc, 2)})
        lines.append(f"{t:10.2f} {accepted.mean()*100:9.2f}% {acc:9.2f}%")

    if args.ood_dir:
        ood_paths = [p for p in Path(args.ood_dir).rglob("*") if p.suffix.lower() in IMAGE_EXT]
        if ood_paths:
            ood_probs = infer_all(session, ood_paths, args.batch)
            ood_conf = ood_probs.max(axis=1)
            rows = []
            for t in (0.40, 0.45, 0.50, 0.60):
                rejected = float((ood_conf < t).mean() * 100)
                rows.append((t, rejected))
            lines.append("")
            lines.append(f"Out-of-scope test ({len(ood_paths)} non-plant photos):")
            for t, rejected in rows:
                lines.append(f"  rejected at threshold {t:.2f}: {rejected:5.1f}%")

    lines.append("")
    lines.append("Per-class F1:")
    per_class = precision_recall_fscore_support(labels, pred, labels=range(len(classes)), zero_division=0)
    for i, name in enumerate(classes):
        label = DISEASE_KEYS[i] if DISEASE_KEYS and i < len(DISEASE_KEYS) else name
        lines.append(f"  {name[:52]:52s} f1={per_class[2][i]:.3f} n={int((labels == i).sum())}")

    report = "\n".join(lines)
    print("\n" + report)
    Path("train/outputs").mkdir(parents=True, exist_ok=True)
    Path("train/outputs/evaluate_report.txt").write_text(report)
    Path("train/outputs/evaluate_report.json").write_text(json.dumps({
        "onnx": str(args.onnx), "split": args.split, "images": len(paths),
        "top1_accuracy": round(top1, 2), "top3_accuracy": round(top3, 2),
        "macro_f1": round(float(f1), 4), "weighted_f1": round(float(w_f1), 4),
        "ece": round(ece, 4), "thresholds": sweep,
        "ood_dir": args.ood_dir,
    }, indent=2))
    print("\nSaved train/outputs/evaluate_report.txt and .json")


if __name__ == "__main__":
    main()
