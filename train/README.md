# AgriSmart AI — ML Training Pipeline

This project is a **hybrid crop-disease classifier**, not a thin wrapper around a
cloud API. It runs a self-trained CNN on-device as the primary/fallback engine
and optionally consults the Gemini vision API for extra accuracy:

```
photo -> on-device MobileNetV2 (own training, PlantVillage, 38 classes, ONNX)
      -> Gemini vision (cloud, when GEMINI_API_KEY is set)
      -> saved to PostgreSQL with alternatives + which engine answered
```

## Pipeline overview

| Step | Command | Output |
|------|---------|--------|
| 1. Download & prepare dataset | `python train/download_dataset.py` | `train/plantvillage_dataset/` (38 classes, 200 imgs/class train/val split) |
| 2. Train (transfer learning) | `python train/train.py` | `plant_disease_model.pth`, `training_curves.png`, `confusion_matrix.png`, `classification_report.txt`, `training_log.csv` |
| 3. Export to ONNX (+quantize) | `python train/export_onnx.py` | `plant_disease_model_quantized.onnx` |
| 4. Deploy | copy quantized file to `models/weights/mobilenet_v2_plantvillage_quantized.onnx` and push | the app serves your trained model |

## Setup (local machine or Google Colab)

```bash
pip install torch torchvision onnx onnxruntime matplotlib seaborn scikit-learn kaggle
```

The dataset downloader needs a free Kaggle API token
(kaggle.com -> Settings -> API -> Create New Token -> save as `~/.kaggle/kaggle.json`).

## The ML work in detail

- **Architecture**: MobileNetV2 (ImageNet-pretrained) — the classifier head is
  replaced (Dropout -> Linear 1280->512 -> ReLU -> Dropout -> Linear 512->38) and the
  last 3 backbone blocks are unfrozen for fine-tuning. ~2.6M trainable parameters.
- **Method**: transfer learning — frozen backbone first, partial fine-tuning of
  the deepest layers; Adam optimizer, cross-entropy loss, 10 epochs, batch 16.
- **Data**: PlantVillage, 38 classes (14 crops + healthy), 200 images per class,
  stratified 80/20 train/val split.
- **Augmentation** (training only): random resized crop, horizontal/vertical
  flips, rotation ±15°, color jitter.
- **Preprocessing at inference** (must match training): 224×224, ImageNet
  mean/std normalization, NCHW float32 — implemented in `models/model_loader.py`.
- **Evaluation**: per-epoch loss/accuracy curves, confusion matrix,
  per-class precision/recall/F1 (`classification_report.txt`).
- **Deployment**: exported to ONNX and dynamically quantized (8-bit) so
  inference runs on a free-tier CPU at ~50ms/image.
- **Operational ML**: training log CSV, model card, reproducible scripts,
  hybrid edge+cloud inference with graceful fallback.

## Demo talking points (for the evaluation)

1. Show `training_curves.png` and `confusion_matrix.png` — that is the model
   you trained, learning in real epochs.
2. Show the class-index mapping in `models/model_loader.py` — ONNX output
   index -> disease, exactly matching the training class order.
3. Scan a leaf in the live app; the result page states which engine answered
   (on-device vs Gemini) and the top-3 probabilities.
4. Show the PostgreSQL history — every scan persists with its prediction,
   confidence and alternatives.
