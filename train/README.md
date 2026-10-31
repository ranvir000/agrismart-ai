# AgriSmart AI - ML pipeline (v2)

This is a **self-trained image classifier** with a cloud model used as a second
opinion, not a wrapper around an API:

```
photo -> on-device MobileNetV2 (trained here, PlantVillage, 38 classes, ONNX int8)
      -> Gemini vision API (optional second engine, needs GEMINI_API_KEY)
      -> confidence below threshold => "outside my scope" page, no made-up disease
      -> every accepted scan saved to PostgreSQL with confidence + alternatives
```

## Pipeline

| Step | Command | Output |
|------|---------|--------|
| 1. Dataset | `python train/download_dataset.py` | `train/plantvillage_dataset/{train,val,test}` + `dataset_summary.json` |
| 2. Train | `python train/train.py` | `plant_disease_model.pth`, `class_names.json`, `metrics.json`, `classification_report.txt`, `confusion_matrix.png`, `training_curves.png`, `training_log.csv`, `threshold_sweep.csv`, `model_card.md` |
| 3. Export | `python train/export_onnx.py` | `plant_disease_model_quantized.onnx` |
| 4. Evaluate the deployed file | `python train/evaluate.py --ood-dir train/ood_examples` | `evaluate_report.txt` / `.json` |
| 5. Deploy | copy the quantized file to `models/weights/mobilenet_v2_plantvillage_quantized.onnx` and push | Render serves your model |

Install once: `pip install -r requirements-train.txt`

## Training method (what the report should describe)

**Phase 1 - frozen backbone.** MobileNetV2 pretrained on ImageNet, all 15
`features` blocks frozen, only the new head is trained
(`Dropout 0.3 -> Linear 1280->512 -> ReLU -> Dropout 0.2 -> Linear 512->38`),
AdamW, lr 1e-3, 3 epochs.

**Phase 2 - fine-tuning.** The last 5 backbone blocks are unfrozen and trained
with AdamW at lr 1e-4, cosine annealing over the remaining epochs, early
stopping (patience 5) on validation accuracy, and the best checkpoint is kept.

Other choices: label smoothing 0.1, weight decay 1e-4, mixed precision (AMP) on
GPU, augmentation = random resized crop (0.65-1.0), horizontal + vertical flip,
rotation 20 deg, colour jitter. Evaluation resize matches the Flask app exactly
(plain 224x224 resize, no centre crop), so the reported numbers describe what
ships.

## Evaluation (why these extra numbers matter)

`train/evaluate.py` runs the **quantized ONNX file the web app actually uses**
and reports:

- **Top-1 and top-3 accuracy** on the untouched test split
- **Macro and weighted precision / recall / F1** per class
- **Expected calibration error (ECE)** - are the stated confidences honest?
- **Threshold sweep** - coverage vs accuracy for every threshold, which is how
  `MIN_CONFIDENCE` (default 0.45) was chosen
- **Out-of-scope rejection** - percentage of non-plant photos correctly refused

Put an `ood_examples/` folder of 10-30 non-plant photos (random objects,
buildings, animals) next to the script and run with `--ood-dir` to get the
rejection number for the report.

## Running it on Google Colab (free GPU, ~40 min)

1. Upload/clone this repo in a Colab notebook and `pip install -r requirements-train.txt`.
2. Upload your `kaggle.json` to `/root/.kaggle/`.
3. `!python train/download_dataset.py` (all images) or add
   `--per-class 400` for a faster run.
4. `!python train/train.py --epochs 25 --batch 48`
5. `!python train/export_onnx.py` and `!python train/evaluate.py --ood-dir train/ood_examples`
6. Download `train/outputs/plant_disease_model_quantized.onnx`, replace
   `models/weights/mobilenet_v2_plantvillage_quantized.onnx`, commit, push.

## Report checklist

| Item | Where it comes from |
|------|--------------------|
| Train/val/test split sizes | `dataset_summary.json` |
| Learning curves (loss + accuracy) | `training_curves.png` |
| Per-class precision / recall / F1 | `classification_report.txt` |
| Confusion matrix | `confusion_matrix.png` |
| Top-1 / top-3 accuracy, macro F1 | `metrics.json`, `evaluate_report.json` |
| Calibration (ECE) | `metrics.json` |
| Threshold choice justification | `threshold_sweep.csv` |
| Out-of-scope rejection rate | `evaluate_report.txt` |
| Model card (architecture, limits) | `model_card.md` |
| Edge vs cloud engine comparison | app results page + `analysis_source` column in the `scans` table |

## Demo script (evaluation day)

1. Scan 3-4 crop leaves -> correct disease, confidence, top-3 alternatives.
2. Scan a random non-plant photo -> app answers **"outside my scope"** with the
   threshold explanation. This is a feature, not a failure.
3. Show `training_curves.png` and `confusion_matrix.png` - your own epochs.
4. Show `evaluate_report.txt` - the numbers for the file running in production.
5. Show the `analysis_source` column - on-device vs Gemini, side by side.
