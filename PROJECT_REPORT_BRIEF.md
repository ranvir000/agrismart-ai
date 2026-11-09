# Project Report Brief — AgriSmart AI

> **How to use this file:** paste the whole thing into Claude (or any assistant) and say:
> *"Write my college project report using this brief. Follow the structure in section 9, keep my
> numbers exactly as given, and keep the language simple enough for a non-ML reader."*
> Everything below is verified from the actual repository, not estimated.

---

## 1. Project identity

| Field | Value |
|---|---|
| Project name | AgriSmart AI — Crop Disease Detection & Advisory System |
| Type | Final-year / semester academic project (machine learning + web application) |
| Core claim | A MobileNetV2 classifier trained with transfer learning predicts among 38 crop-condition labels (including healthy classes) across 14 crops, with ONNX CPU inference in a Flask web app |
| Student | _(fill in name, roll number, branch, college, guide)_ |
| Year | _(fill in)_ |

One-line pitch: *"A farmer uploads a photo from one of 14 supported crops; a MobileNetV2 model
fine-tuned from ImageNet weights returns a PlantVillage class score, and the web app shows related
reference guidance. The deployed quantized ONNX model runs on CPU; latency and field accuracy have
not been benchmarked by this project."*

---

## 2. Problem statement

Crop diseases reduce yield by 20–40% globally, and diagnosis usually requires an agricultural
extension officer who is not available in most villages. Existing "AI plant doctor" apps are mostly
thin wrappers around a paid cloud vision API: they need constant internet, they cost money per
image, and they cannot explain or defend their answer. This project builds the *model itself*,
trains it, evaluates it honestly, and deploys it with a clear rule for when it should refuse to
answer.

Objectives:
1. Train a lightweight image classifier on the public PlantVillage dataset (38 crop-condition labels, 14 crops).
2. Export and dynamically quantize it to ONNX for CPU inference; a formal latency benchmark is not included.
3. Measure it properly: top-1, top-3, macro F1, per-class F1, calibration error, and a confidence-threshold sweep.
4. Deploy it in a full web application (accounts, scan history, PDF report, disease library, analytics).
5. Reject low-confidence in-scope predictions and, when Gemini is configured, respect its explicit out-of-scope result; this is best-effort, not a validated OOD guarantee.
6. Compare the custom model with a large pretrained vision model (Gemini) when an API key is configured and show the separate outputs rather than blending them.

---

## 3. Technology stack

- **Language:** Python 3.11
- **Deep learning:** PyTorch + torchvision (training), ONNX Runtime (inference, CPU only)
- **Web:** Flask, Flask-Login (sessions/auth), Flask-SQLAlchemy, Jinja2 templates, custom CSS
- **Database:** PostgreSQL in production (Render), SQLite locally — same SQLAlchemy models for both
- **Extras:** OpenWeatherMap API (optional live weather; sample data is used without a key), Google Gemini API (optional vision cross-check), ReportLab/HTML-to-PDF for the downloadable scan report
- **Deployment:** Render web service + GitHub Actions keep-alive workflow (pings the free instance so it does not cold-start during a demo)

---

## 4. Dataset

- **PlantVillage** — public leaf-image dataset, photographed mostly on plain backgrounds under controlled lighting.
- **38 crop-condition labels, including healthy classes**, covering **14 crops**: apple (4 labels), blueberry, cherry (2), corn/maize (4), grape (4), orange, peach (2), bell pepper (2), potato (3), raspberry, soybean, squash, strawberry (2), tomato (10).
- **Split:** 70% train / 15% validation / 15% test, stratified, fixed seed 42.
- **Test set used for the final numbers: 2,207 images.**
- **Augmentation (training only):** random resized crop, horizontal/vertical flips, small rotation, colour jitter. Input resolution 224×224, ImageNet mean/std normalisation.

---

## 5. Model and training

**Architecture:** MobileNetV2 pretrained on ImageNet, used as a feature extractor, with a custom head:

```
Input 224×224×3
 → MobileNetV2 backbone (ImageNet weights)
 → Dropout(0.3) → Linear 1280→512 → ReLU → Dropout(0.2) → Linear 512→38
```

**Why MobileNetV2:** depthwise-separable convolutions give a compact architecture appropriate for CPU inference, while retaining useful features for fine-grained leaf textures. The repository does not include a measured CPU latency benchmark or a mobile app.

**Two-phase transfer learning** (this is the part to emphasise in the report — it is the standard modern recipe):
1. **Phase 1 — head only (3 epochs):** backbone frozen, LR 1e-3, so the random head does not destroy pretrained features.
2. **Phase 2 — fine-tune (up to 22 more epochs):** last 5 backbone blocks unfrozen, LR 1e-4 with cosine decay, early stopping (patience 5).

**Regularisation:** AdamW (weight decay 1e-4), label smoothing 0.1, and dropout as above. These methods can reduce overfitting; the measured ECE still shows that model scores are not perfectly calibrated.

**Actual training run:** 21 epochs, best epoch 16, ~2,320 seconds (~39 min) on a single Colab-class GPU.

**Compression for deployment:** export to ONNX (`dynamo=False`, opset as in `train/export_onnx.py`), then **dynamic int8 quantization** → `models/weights/mobilenet_v2_plantvillage_quantized.onnx`. The observed top-1 accuracy is 1.00 percentage point lower than the full-precision checkpoint. Exact file-size and CPU-latency comparisons are not included in the repository.

---

## 6. Results (use these numbers exactly)

### 6.1 Held-out test set — full-precision PyTorch model

| Metric | Value |
|---|---|
| Top-1 accuracy | **98.73%** |
| Top-3 accuracy | **99.95%** |
| Macro precision / recall / F1 | 0.9877 / 0.9877 / **0.9876** |
| Weighted F1 | 0.9873 |
| Expected calibration error (ECE) | **0.1162** (15 bins) |

### 6.2 Same test set — deployed quantized ONNX model

| Metric | Value |
|---|---|
| Top-1 accuracy | **97.73%** |
| Top-3 accuracy | **99.91%** |
| Macro F1 | 0.9777 |
| Weighted F1 | 0.9772 |
| ECE | 0.1426 |
| Model size | int8 quantized, runs on CPU |

**Talking point:** quantization cost exactly **1.00 percentage point** of top-1 accuracy in exchange for CPU-only, ~free inference. Top-3 barely moved (99.95 → 99.91).

### 6.3 Hardest classes (per-class F1, deployed model)

| Class | F1 | Why it is hard |
|---|---|---|
| Tomato Bacterial Spot | 0.862 | Visually overlaps Tomato Septoria and Target Spot |
| Tomato Septoria Leaf Spot | 0.891 | Small lesions, similar to Early Blight |
| Tomato Late Blight | 0.897 | Early lesions look like several other blights |
| Tomato Target Spot | 0.919 | Shares symptoms with Septoria |
| Corn Northern Leaf Blight | 0.944 | Long lesions can look like Gray Leaf Spot |

All 38 per-class F1 scores are in `train/outputs/evaluate_report.txt`; 21 of 38 classes score ≥ 0.98.

### 6.4 Confidence-threshold sweep (the most "engineering" result in the project)

This is the experiment that justifies the app's behaviour — it trades **coverage** (how often it answers) for **accuracy** (how often that answer is right):

| Threshold | Coverage | Accuracy when it answers |
|---|---|---|
| 0.30 | 99.18% | 98.04% |
| 0.50 | 94.38% | 98.99% |
| **0.60** | **89.62%** | **99.39%** |
| 0.75 | 79.52% | 99.83% |
| **0.85** | **63.25%** | **100.00%** |
| 0.95 | 18.99% | 100.00% |

**Chosen operating points:** refuse predictions below **0.60** and show the highest-ranked candidates as untrusted suggestions. At **0.85**, the model was 100% accurate on 63.25% of this held-out in-scope split. This is not a guarantee for new crops or field photos, and the 0.85 value is not an independent unknown-plant detector.

### 6.5 The honest limitation (state it before the examiner does)

PlantVillage images are clean, centred, plain-background leaves. The repository contains **no independent field-photo or out-of-distribution evaluation** (`ood_dir` is null in the deployed-model report), so this project cannot state a measured field accuracy or claim robust rejection of unseen crops. The confidence score and threshold results apply only to the held-out PlantVillage split. This is the most important limitation, and it is why the system:
- shows top-3 alternatives rather than one name,
- refuses low-confidence answers,
- compares itself against a large pretrained vision model instead of hiding the disagreement,
- and must not claim field accuracy without collecting and evaluating a field-photo test set.

---

## 7. System architecture (hybrid, two engines, one visible answer)

```
Farmer uploads photo (any format)
        │
        ▼
normalize_upload()  → decoded, EXIF-rotated, re-encoded as RGB JPEG   [models/model_loader.py]
        │
        ├──► Engine 1: MobileNetV2 → ONNX int8 → 38 class scores   (local CPU inference)
        │        top-1 + top-3 + class-level severity default
        │
        └──► Engine 2: Gemini vision model (only if GEMINI_API_KEY set)
                 returns a disease name OR an explicit "this plant is not one of your 14 crops"
        │
        ▼
Decision logic                                                         [routes/scan.py]
  • The trained CNN is the headline result when it returns a usable in-scope label.
  • Below 0.60 confidence, a Gemini out-of-scope verdict, or a cross-engine crop conflict
    → do NOT name a disease; show an "outside my scope" page with untrusted candidates.
  • Without Gemini, the closed-set CNN cannot guarantee that an unsupported crop is rejected.
  • The other engine's answer is attached as a visible alternative, never merged or averaged.
        │
        ▼
Stored in Scan table (image bytes, predicted label, score, alternatives and source)
        │
        ▼
Results page: model answer + score + top-3 + class-level severity + general reference guidance
              + "Model Comparison" panel (both engines side by side, agree/disagree banner)
              + downloadable PDF
```

**Two model outputs are stored per scan** (`analysis_source` and `engine_outputs`), so a scan can be
re-examined later without re-running inference — and so the comparison can be shown in the demo.

---

## 8. Application features (non-ML work also counts as engineering)

1. **Authentication & roles** — register/login, password hashing (Werkzeug), two roles: farmer and admin (admin cannot scan; admin gets a management panel). Sessions expire after 12 hours; "remember me" lasts 1 day.
2. **Dashboard** — total scans, healthy count, disease count, live temperature, recent scans, weather-based advisory.
3. **Scan flow** — drag-and-drop upload, any image format normalised to JPEG, 16 MB limit, live preview.
4. **Results page** — headline model answer, confidence bar, top-3 alternatives, severity, symptom description, treatment/fertilizer recommendations, model comparison panel, PDF download.
5. **Out-of-scope page** — honest refusal with the top-3 "did you mean" candidates and an explanation of *why* it refused.
6. **History** — paginated scan history with thumbnails, per-scan detail.
7. **Disease library** — browsable reference for all 38 classes with symptoms and remedies.
8. **Analytics** — charts over scan history (disease frequency, severity distribution, trend over time).
9. **Profile** — farm details, city (feeds the weather widget), password change.
10. **Schema self-migration** — `ensure_schema()` adds columns introduced after a database was created, so the deployed PostgreSQL instance keeps working across releases.

---

## 9. Suggested report structure (adapt to your college's format)

1. **Abstract** (~200 words) — problem, method, deployed artifact result (97.73% top-1 on the held-out PlantVillage test split), honest limitations.
2. **Introduction** — crop-disease impact, why an on-device model beats an API wrapper, objectives.
3. **Literature review** — cite PlantVillage and MobileNetV2 sources you have actually read; explain domain shift as a known risk without claiming a field accuracy measured by this project.
4. **Dataset and preprocessing** — PlantVillage, 38/14, splits, augmentation, normalisation.
5. **Methodology** — architecture diagram, two-phase training, hyperparameter table, why each choice.
6. **Implementation** — Flask blueprints, database schema (User, Scan), the two-engine decision logic, deployment on Render, ONNX quantization.
7. **Results** — separate full-precision checkpoint metrics from deployed quantized ONNX metrics; use `evaluate_report.json` for the deployed model's threshold sweep (`threshold_sweep.csv` is from the PyTorch checkpoint).
8. **Discussion** — lab-vs-field gap, why 100% confidence at 0.85 is not the same as field accuracy, quantization trade-off, why refusing is better than guessing.
9. **Limitations & future work** — see section 10.
10. **Conclusion**.
11. **References** — PlantVillage, MobileNetV2, PyTorch, ONNX Runtime, Flask, Gemini API docs.
12. **Appendix** — file map, how to run, screenshots.

Figures/artifacts that already exist in the repo and may be embedded:
- `train/outputs/training_curves.png` — accuracy/loss per epoch
- `train/outputs/confusion_matrix.png` — 38×38 confusion matrix
- `train/outputs/evaluate_report.json` — deployed ONNX metrics and threshold sweep; `threshold_sweep.csv` is the full-precision checkpoint sweep
- Screenshots to capture yourself: dashboard, scan upload, results page (with Model Comparison), out-of-scope page, PDF report.

---

## 10. Limitations and future work (be specific — examiners reward this)

**Limitations**
1. No field-photo or out-of-distribution evaluation is included; performance on real farms is unknown and cannot be inferred from PlantVillage scores.
2. Confidence calibration is imperfect on the in-scope test data (ECE 0.1426 for deployed ONNX); the score is not a real-world probability.
3. 38 classes only; no nutrient deficiency, pest damage, or non-leaf symptoms.
4. Fine-grained confusion within tomato and corn classes (see 6.3).
5. The Gemini cross-check needs an API key and internet; without it the closed-set CNN runs alone and cannot reliably detect unsupported inputs.
6. No offline mobile app yet — the model is phone-sized but is not packaged for Android/iOS.

**Future work**
1. Fine-tune on field images (PlantDoc, a real field dataset) and re-run the calibration/threshold sweep — the obvious next experiment.
2. Temperature scaling or focal loss to reduce ECE.
3. On-device inference with TFLite/ONNX Mobile so the app works with zero connectivity.
4. Grad-CAM heatmaps so a farmer sees *which part of the leaf* drove the decision.
5. Add nutrient-deficiency and pest classes; multi-leaf / whole-plant photographs.

---

## 11. Plain-English explanation (say it like this in the viva)

**What the project is:** "I used the PlantVillage dataset of about 54,000 controlled leaf photos, covering 38 crop-condition labels across 14 crops. I fine-tuned an ImageNet-pretrained MobileNetV2, exported and quantized it to ONNX for CPU inference, and built a Flask app around it. The app returns a class score and links to general reference information; it is not a field-validated diagnosis or individual treatment prescription."

**How the model learns (the analogy that works):** "Think of the network as someone learning to recognise diseases from flashcards. At first they only know general shapes — edges, colours, textures — because they were already trained on a million everyday photos. I freeze that general knowledge, teach them the 38 disease names on top, then let them adjust the finer detail-detecting layers slightly. That's transfer learning: teach the specialist last, not first."

**Why a confidence number matters:** "The classifier returns scores over its 38 known labels. On the 2,207-image held-out PlantVillage test split, the deployed model was correct for 100% of the accepted examples above 0.85, but that covered only 63.25% of the split. Below 0.60 the app refuses a label. These are in-scope benchmark results, not a guarantee for field photos or unknown crops."

**Why two engines:** "When the Gemini API is configured, the app sends the same photo to a separate vision model and shows its answer next to the local CNN output. I do not average the scores; disagreement is visible, and an explicit out-of-scope result prevents the app from saving a disease label. The cross-check is not a substitute for field validation."

**The number to lead with:** 97.73% top-1 and 99.91% top-3 for the deployed quantized ONNX model on the 2,207-image held-out PlantVillage test split. The full-precision checkpoint scored 98.73% top-1; do not conflate the two artifacts.

**The limitation to volunteer:** "Those numbers are measured on PlantVillage's controlled test split. I did not evaluate a field-photo or out-of-distribution set, so I cannot claim a field accuracy or guarantee that an unseen crop will be rejected. That is a clear next step for the project."

---

## 12. Likely viva questions with good answers

- *Why MobileNetV2 and not ResNet-50?* — Depthwise-separable convolutions: ~3.4M parameters vs 25M, CPU-friendly, and 98.7% is already near the ceiling for this dataset; a bigger model would not raise accuracy, only hosting cost.
- *Why 224×224?* — Native MobileNetV2 input, matching pretrained normalisation; enough detail for lesion texture.
- *Why label smoothing?* — Prevents the network becoming overconfident, which improves both calibration and generalisation; directly relevant because the app's decision rule depends on confidence.
- *Why two-phase training?* — Fine-tuning everything from epoch 1 destroys pretrained features with large gradients; freezing first lets the new head settle.
- *What does int8 quantization do?* — Stores weights in 8-bit integers instead of 32-bit floats: ~4× smaller, faster CPU inference; here it costs 1% top-1.
- *Isn't 98% too good?* — It is a high score on a controlled held-out PlantVillage split. The running website uses the quantized artifact, whose measured top-1 is 97.73%. It is not evidence of similar performance on field photos.
- *What is ECE?* — Expected Calibration Error: group predictions into confidence bins and compute the weighted gap between average score and accuracy. The deployed model's ECE is 0.1426 on the in-scope test split; this does not establish field calibration.
- *How do you stop it naming a disease for a rice leaf?* — A score below 0.60 is rejected, and when Gemini is configured its explicit out-of-scope verdict vetoes the CNN label. But the CNN is closed-set; without Gemini, it can still assign a known class to an unsupported leaf. We have not validated OOD rejection.
- *What happens if the internet is down?* — The CNN is fully local (ONNX, CPU), so the core diagnosis still works; only the cross-check and weather widget degrade.

---

## 13. Repository map (for the appendix)

```
app.py                     Flask factory, config, schema self-migration, admin seeding
routes/auth.py             register / login / logout / forgot-password
routes/dashboard.py        stats + weather + recent scans
routes/scan.py             upload → normalize → two-engine inference → decision → results
routes/history.py          paginated scan history
routes/analytics.py        charts
routes/diseases.py         disease library
routes/profile.py          farm details, password change
routes/admin.py            admin panel
models/model_loader.py     ONNX session, preprocessing, top-3, severity, weather
models/vision_analyzer.py  Gemini client (model fallback + retry on 503)
models/disease_info.py     knowledge base: symptoms, treatment, fertilizer for 38 classes
models/weights/*.onnx      quantized deployed model + class config
database/db.py             User, Scan, ContactMessage models
templates/                 Jinja2 UI (incl. _engine_compare.html — the side-by-side panel)
train/train.py             two-phase training
train/evaluate.py          test metrics, per-class F1, ECE, threshold sweep
train/export_onnx.py       ONNX export + int8 quantization
train/outputs/             metrics.json, evaluate_report.txt, model_card.md, curves, confusion matrix
```

---

## 14. Instructions to whoever writes the report

- Use **only** the numbers in section 6. Do not invent datasets, comparisons or accuracies.
- Write in **simple, precise English**; define every ML term the first time it is used.
- Prefer tables and one clear diagram over long prose.
- Keep the lab-vs-field limitation prominent — it is the project's strongest sign of engineering maturity, not a weakness.
- Target length: 25–35 pages plus figures, unless the college specifies otherwise.
- Mark every placeholder I must fill in (name, college, guide, date) with **bold square brackets**.
