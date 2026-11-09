# AgriSmart AI — Viva Readiness Guide

Use this to understand and explain the project, not as a script to recite. Be transparent about AI assistance if asked, and do not claim work, tests, or field results that you did not personally verify.

## 30-second project explanation

AgriSmart AI is a Flask web application for classifying leaf photos from 14 supported PlantVillage crops. Its image classifier is MobileNetV2: the training code uses PyTorch transfer learning, while the website serves a dynamically quantized ONNX model on CPU. The result is a score over 38 crop-condition labels, including healthy classes, plus a disease-library reference page. When a Gemini API key is configured, the same photo is sent for a separate vision cross-check; live weather also depends on an OpenWeatherMap key. This is an academic prototype, not a field-validated diagnostic or treatment system.

## Suggested 5-minute demonstration

1. Open the homepage and point out the supported crop list. Explain that 38 means disease **and healthy** labels, not 38 diseases.
2. Sign in with a demo farmer account prepared before the viva. Do not create or reveal an administrator account during the presentation.
3. Upload one clear image of a supported crop, such as a tomato leaf. Use a known example if available; avoid claiming the prediction is correct just because the confidence score is high.
4. On the result page, explain the model label and score, alternatives, and the class-level severity value. If a Gemini panel appears, compare the two outputs without saying they are independent proof.
5. Show the disease library and one detail page. Describe the treatment content as general reference guidance that must be checked with a local expert and product labels.
6. Show scan history or a PDF export if time allows. Weather is live only when the service key is configured; otherwise the UI marks it as sample data.
7. End with the most important limitation: the evaluation is on a held-out PlantVillage split, not a field-photo or unknown-crop test set.

Before the viva, sign in once, test one supported photo, confirm the result and PDF links, and check whether the weather card says live or sample. If Gemini is enabled, remember the photo is sent to Google's Gemini API for that cross-check.

## How the system works

**Training:** MobileNetV2 starts with ImageNet weights. The first training phase freezes the backbone and trains a new classifier head. The second phase unfreezes the last five backbone blocks and fine-tunes them. The head is Dropout(0.3) → Linear(1280→512) → ReLU → Dropout(0.2) → Linear(512→38). Training and evaluation scripts use PyTorch.

**Input and inference:** the app validates and decodes the upload, corrects image orientation, converts it to RGB, resizes and normalizes it for the network, then runs the quantized ONNX artifact using CPU inference. The classifier is closed-set: it ranks the 38 trained labels and does not have a learned “unknown crop” class.

**Decision and guidance:** a prediction below the configured 0.60 minimum is refused. A configured Gemini cross-check can also explicitly say the plant is outside the supported scope, which now prevents a CNN label from being saved; a crop disagreement also produces a review page. Without Gemini, confidence thresholding alone cannot guarantee rejection of an unfamiliar crop. The displayed severity is a fixed class-level default, not an image measurement. Treatments are static library references, not personalized prescriptions.

**Web application:** Flask routes handle registration and login, scans, history, profile, analytics, weather, disease reference pages, and admin functions. SQLAlchemy supports PostgreSQL in the Render deployment and SQLite locally. Scan images are persisted in the database; newly uploaded normalized files are temporary and removed after inference.

## Results to quote accurately

The deployed quantized ONNX model was evaluated on **2,207 held-out PlantVillage test images** across 38 labels:

| Metric | Deployed quantized ONNX |
|---|---:|
| Top-1 accuracy | 97.73% |
| Top-3 accuracy | 99.91% |
| Macro F1 | 0.9777 |
| Weighted F1 | 0.9772 |
| Expected Calibration Error | 0.1426 |

At a 0.60 threshold on this in-scope test split, the deployed model accepted 89.62% of images, with 99.39% accuracy among accepted images. At 0.85, it accepted 63.25% at 100% accuracy on that same test split. These figures do **not** demonstrate detection of unseen crops, and 100% on one split is not a guarantee.

The full-precision PyTorch checkpoint has separate metrics: 98.73% top-1, 99.95% top-3, macro F1 0.9876, weighted F1 0.9873, ECE 0.1162. When asked about the website, lead with the **quantized ONNX** metrics; do not mix the two artifacts. Sources: `train/outputs/evaluate_report.json` (deployed ONNX) and `train/outputs/metrics.json` (training checkpoint).

## Likely questions

**Why MobileNetV2?** It is a compact CNN using depthwise-separable convolutions, making it a reasonable choice for CPU inference. It also starts from general visual features learned on ImageNet, so the project fine-tunes rather than learning all visual features from random initialization.

**What is transfer learning?** Reusing a model pretrained on a broad image dataset, replacing its final classifier with one for this project's labels, and fine-tuning it on the crop dataset.

**What does quantization do?** It changes model weights to a lower-precision representation (dynamic int8 here) for deployment. In the checked-in evaluations, top-1 accuracy changes from 98.73% for the full-precision checkpoint to 97.73% for the quantized deployed artifact.

**What does 97.73% mean?** It is the fraction of correct top-1 predictions on the specified 2,207-image held-out PlantVillage test split. It is not a claim about every farm, camera, crop, or lighting condition.

**What does ECE 0.1426 tell us?** Expected Calibration Error summarizes the weighted gap between model score and accuracy across confidence bins. It indicates that displayed model scores should not be interpreted as perfectly calibrated probabilities.

**How does the app handle rice or another unsupported crop?** Scores below 0.60 are refused. When the Gemini API is configured, its explicit out-of-scope result vetoes the classifier's closed-set guess. Without Gemini, the classifier can still produce a confident known-class guess for an unknown crop. The repository has no OOD evaluation (`ood_dir` is null), so rejection performance is unproven.

**Is this a treatment recommendation?** It is a predicted class plus general information stored in the disease library. Severity is a class-level default; the model does not measure the damage in the uploaded photo, and the app does not generate a personalized pesticide prescription.

**What if an external API is unavailable?** If the hosted web app is reachable, its server-side ONNX classifier can run without Gemini; the cross-check and live weather are unavailable. If the user's internet connection or the app server is down, this web app cannot be reached for a scan. When no weather key is set, the site uses simulated sample data and labels it accordingly.

## Do not claim

- Do not say “any crop,” “38 diseases,” “expert diagnosis,” or “individualized treatment.” Say “38 crop-condition labels, including healthy classes, across 14 supported crops” and “general reference guidance.”
- Do not quote 26% field accuracy, a guaranteed OOD refusal rate, or any field-photo result. No such evaluation is included in the repository.
- Do not promise under one second or under three seconds. The repository has no formal end-to-end latency benchmark; Gemini can add network delay.
- Do not say weather is always live. It depends on a configured API key; otherwise the app shows sample data.
- Do not describe the production web inference path as PyTorch. PyTorch is used in training; the deployed app loads ONNX Runtime.
- Do not describe the severity badge as an image-derived measurement or reference guidance as expert-verified prescription.
