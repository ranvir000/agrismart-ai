# Model Card - AgriSmart AI crop-disease classifier

## Model
- Architecture: MobileNetV2 (ImageNet-pretrained), custom head
  (Dropout 0.3 -> Linear 1280->512 -> ReLU -> Dropout 0.2 -> Linear 512->38)
- Method: two-phase transfer learning (3 frozen epochs, then
  5 unfrozen backbone blocks with cosine learning-rate decay)
- Input: 224x224 RGB, ImageNet mean/std normalisation
- Optimiser: AdamW, weight decay 0.0001, label smoothing 0.1
- Deployment: exported to ONNX and dynamically quantized to int8 (runs on CPU)

## Data
- PlantVillage, 38 classes across 14 crops
- Augmentation: random resized crop, flips, rotation, colour jitter

## Evaluation results (held-out PlantVillage test split, 2,207 images)

The repository contains results for two different artifacts. Cite the deployed
quantized ONNX figures when describing the running website.

| Artifact | Top-1 | Top-3 | Macro F1 | Weighted F1 | ECE |
|---|---:|---:|---:|---:|---:|
| Full-precision PyTorch checkpoint (`metrics.json`) | 98.73% | 99.95% | 0.9876 | 0.9873 | 0.1162 |
| Deployed dynamically quantized ONNX (`evaluate_report.json`) | 97.73% | 99.91% | 0.9777 | 0.9772 | 0.1426 |

The 1.00 percentage-point top-1 difference is the observed trade-off for the
quantized deployment artifact. No independent CPU latency benchmark is included,
so do not claim a measured sub-second or under-three-second runtime.

## Limitations
- The classifier is closed-set: its softmax always ranks the 38 PlantVillage labels.
  A confidence threshold is not a reliable detector of unseen crops or non-plants.
- The checked-in ONNX evaluation has `ood_dir: null`; no out-of-distribution or
  field-photo rejection benchmark is included. Do not quote a field accuracy such
  as 26% as this project's measured result.
- When configured, Gemini is called as a separate vision cross-check and its explicit
  out-of-scope verdict can veto a CNN label. Without the API key, only the local CNN
  runs; unsupported inputs can still receive a confident closed-set guess.
- Severity is a static class-level default, and disease/treatment information is
  reference content, not an image-specific damage estimate or expert prescription.

## Deployment threshold
- Operating rejection threshold: **0.60** (`MIN_CONFIDENCE`). The in-scope threshold
  sweep for the deployed artifact is in `evaluate_report.json`; the separate
  `threshold_sweep.csv` was generated from the full-precision checkpoint.
- At 0.60 on the 2,207-image in-scope test split: 89.62% coverage and 99.39%
  accuracy among accepted images. This does not measure rejection of unknown crops.
- At 0.85, the deployed artifact had 63.25% coverage and 100% accepted accuracy on
  this test split. This is an observed benchmark point, not a field guarantee.
- When Gemini is configured, the application attempts a cross-check on every scan;
  this has network latency and sends the uploaded image to Google's Gemini API.

## Classes
Apple___Apple_scab, Apple___Black_rot, Apple___Cedar_apple_rust, Apple___healthy, Blueberry___healthy, Cherry_(including_sour)___Powdery_mildew, Cherry_(including_sour)___healthy, Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot, Corn_(maize)___Common_rust_, Corn_(maize)___Northern_Leaf_Blight, Corn_(maize)___healthy, Grape___Black_rot, Grape___Esca_(Black_Measles), Grape___Leaf_blight_(Isariopsis_Leaf_Spot), Grape___healthy, Orange___Haunglongbing_(Citrus_greening), Peach___Bacterial_spot, Peach___healthy, Pepper,_bell___Bacterial_spot, Pepper,_bell___healthy, Potato___Early_blight, Potato___Late_blight, Potato___healthy, Raspberry___healthy, Soybean___healthy, Squash___Powdery_mildew, Strawberry___Leaf_scorch, Strawberry___healthy, Tomato___Bacterial_spot, Tomato___Early_blight, Tomato___Late_blight, Tomato___Leaf_Mold, Tomato___Septoria_leaf_spot, Tomato___Spider_mites Two-spotted_spider_mite, Tomato___Target_Spot, Tomato___Tomato_Yellow_Leaf_Curl_Virus, Tomato___Tomato_mosaic_virus, Tomato___healthy
