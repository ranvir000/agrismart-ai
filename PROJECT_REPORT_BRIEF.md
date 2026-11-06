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
| Core claim | A custom-trained CNN (not an API wrapper) detects 38 plant-disease classes across 14 crops, runs offline on CPU, and is deployed as a real web app |
| Student | _(fill in name, roll number, branch, college, guide)_ |
| Year | _(fill in)_ |

One-line pitch: *"A farmer uploads a photo of a leaf; a MobileNetV2 model trained from scratch on
PlantVillage returns the disease, its confidence and a treatment plan in under two seconds — on a
free CPU-only server, with no GPU and no paid inference."*

---

## 2. Problem statement

Crop diseases reduce yield by 20–40% globally, and diagnosis usually requires an agricultural
extension officer who is not available in most villages. Existing "AI plant doctor" apps are mostly
thin wrappers around a paid cloud vision API: they need constant internet, they cost money per
image, and they cannot explain or defend their answer. This project builds the *model itself*,
trains it, evaluates it honestly, and deploys it with a clear rule for when it should refuse to
answer.

Objectives:
1. Train a lightweight image classifier on the public PlantVillage dataset (38 classes, 14 crops).
2. Compress it (ONNX + int8 quantization) so inference runs in well under a second on a free CPU host.
3. Measure it properly: top-1, top-3, macro F1, per-class F1, calibration error, and a confidence-threshold sweep.
4. Deploy it in a full web application (accounts, scan history, PDF report, disease library, analytics).
5. Handle failure honestly: an unsupported plant must return "outside my scope", never a confident wrong name.
6. Cross-check the custom model against a large pretrained vision model (Gemini) and *show both answers side by side*, so the disagreement is visible rather than hidden.

---

## 3. Technology stack

- **Language:** Python 3.11
- **Deep learning:** PyTorch + torchvision (training), ONNX Runtime (inference, CPU only)
- **Web:** Flask, Flask-Login (sessions/auth), Flask-SQLAlchemy, Jinja2 templates, custom CSS
- **Database:** PostgreSQL in production (Render), SQLite locally — same SQLAlchemy models for both
- **Extras:** OpenWeatherMap API (dashboard weather widget), Google Gemini API (cross-check only), ReportLab/HTML-to-PDF for the downloadable scan report
- **Deployment:** Render web service + GitHub Actions keep-alive workflow (pings the free instance so it does not cold-start during a demo)

---

## 4. Dataset

- **PlantVillage** — public leaf-image dataset, photographed mostly on plain backgrounds under controlled lighting.
- **38 classes** covering **14 crops**: apple (4 classes), blueberry, cherry (2), corn/maize (4), grape (4), orange, peach (2), bell pepper (2), potato (3), raspberry, soybean, squash, strawberry (2), tomato (10).
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

**Why MobileNetV2:** depthwise-separable convolutions give ~3.4M parameters and ~300M MACs — small enough to run on a free CPU instance (and later on a phone), while staying accurate on fine-grained leaf textures.

**Two-phase transfer learning** (this is the part to emphasise in the report — it is the standard modern recipe):
1. **Phase 1 — head only (3 epochs):** backbone frozen, LR 1e-3, so the random head does not destroy pretrained features.
2. **Phase 2 — fine-tune (up to 22 more epochs):** last 5 backbone blocks unfrozen, LR 1e-4 with cosine decay, early stopping (patience 5).

**Regularisation:** AdamW (weight decay 1e-4), label smoothing 0.1 (prevents overconfident logits — the reason calibration error is acceptable), dropout as above.

**Actual training run:** 21 epochs, best epoch 16, ~2,320 seconds (~39 min) on a single Colab-class GPU.

**Compression for deployment:** export to ONNX (`dynamo=False`, opset as in `train/export_onnx.py`), then **dynamic int8 quantization** → `models/weights/mobilenet_v2_plantvillage_quantized.onnx`. Roughly 4× smaller, ~2–3× faster on CPU, with a small accuracy drop (see below — this drop is itself a nice result to discuss).

---

## 6. Results (use these numbers exactly)

### 6.1 Held-out test set — full-precision PyTorch model

| Metric | Value |
|---|---|
| Top-1 accuracy | **98.73%** |
| Top-3 accuracy | **99.95%** |
| Macro precision / recall / F1 | 0.9877 / 0.9877 / **0.9876** |
| Weighted F1 | 0.9873 |
| Expected calibration error (ECE) | **0.1162** (10 bins) |

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

**Chosen operating points:** refuse below **0.60** (show "outside my scope" with the top-3 candidates instead of a fake answer), and treat the model's answer as authoritative only from **0.85**, where it is 100% correct on the test split.

### 6.5 The honest limitation (state it before the examiner does)

PlantVillage images are clean, centred, plain-background leaves. On **real field photos and random internet photos**, the same model drops to roughly **26% top-1 accuracy** even when it reports high confidence. The confidence score is therefore **not calibrated to real-world photos** — it is calibrated to PlantVillage-like photos. This is the single most important limitation, and it is why the system:
- shows top-3 alternatives rather than one name,
- refuses low-confidence answers,
- compares itself against a large pretrained vision model instead of hiding the disagreement,
- and never claims field accuracy.

---

## 7. System architecture (hybrid, two engines, one visible answer)

```
Farmer uploads photo (any format)
        │
        ▼
normalize_upload()  → decoded, EXIF-rotated, re-encoded as RGB JPEG   [models/model_loader.py]
        │
        ├──► Engine 1: MobileNetV2 → ONNX int8 → 38 class probabilities   (always runs, CPU, offline)
        │        top-1 + top-3 + severity
        │
        └──► Engine 2: Gemini vision model (only if GEMINI_API_KEY set)
                 returns a disease name OR an explicit "this plant is not one of your 14 crops"
        │
        ▼
Decision logic                                                         [routes/scan.py]
  • The trained CNN's answer is ALWAYS the headline result — even with low confidence.
  • Below 0.60 confidence (or when Gemini says the plant is out of scope and the CNN is not
    certain) → do NOT name a disease: show the "outside my scope" page with the top-3 candidates.
  • The other engine's answer is attached as a visible alternative, never merged or averaged.
        │
        ▼
Stored in Scan table (image bytes, probabilities, engine_outputs JSON)
        │
        ▼
Results page: model answer + confidence + top-3 + severity + treatment plan
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

1. **Abstract** (~200 words) — problem, method, headline result (98.73% top-1, 97.73% deployed), honest limitation.
2. **Introduction** — crop-disease impact, why an on-device model beats an API wrapper, objectives.
3. **Literature review** — PlantVillage paper (Mohanty et al., 2016 — 99.35% on lab images but 31% on field images; cite it, it directly explains this project's own field drop), MobileNetV2 (Sandler et al., 2018), transfer learning, label smoothing, post-training quantization, calibration/ECE.
4. **Dataset and preprocessing** — PlantVillage, 38/14, splits, augmentation, normalisation.
5. **Methodology** — architecture diagram, two-phase training, hyperparameter table, why each choice.
6. **Implementation** — Flask blueprints, database schema (User, Scan), the two-engine decision logic, deployment on Render, ONNX quantization.
7. **Results** — the tables in section 6 above, training curves, confusion matrix (`train/outputs/confusion_matrix.png`), per-class F1 table, threshold sweep graph.
8. **Discussion** — lab-vs-field gap, why 100% confidence at 0.85 is not the same as field accuracy, quantization trade-off, why refusing is better than guessing.
9. **Limitations & future work** — see section 10.
10. **Conclusion**.
11. **References** — PlantVillage, MobileNetV2, PyTorch, ONNX Runtime, Flask, Gemini API docs.
12. **Appendix** — file map, how to run, screenshots.

Figures that already exist in the repo and should be embedded:
- `train/outputs/training_curves.png` — accuracy/loss per epoch
- `train/outputs/confusion_matrix.png` — 38×38 confusion matrix
- `train/outputs/threshold_sweep.csv` — source for a coverage-vs-accuracy chart
- Screenshots to capture yourself: dashboard, scan upload, results page (with Model Comparison), out-of-scope page, PDF report.

---

## 10. Limitations and future work (be specific — examiners reward this)

**Limitations**
1. Training distribution mismatch: PlantVillage backgrounds vs real farm photos → ~26% real-world top-1 despite high confidence.
2. Confidence is not calibrated (ECE ≈ 0.12–0.14) — a "95%" on a field photo does not mean 95% correct.
3. 38 classes only; no nutrient deficiency, pest damage, or non-leaf symptoms.
4. Fine-grained confusion within tomato and corn classes (see 6.3).
5. The Gemini cross-check needs an API key and internet; without it the app runs on the CNN alone.
6. No offline mobile app yet — the model is phone-sized but is not packaged for Android/iOS.

**Future work**
1. Fine-tune on field images (PlantDoc, a real field dataset) and re-run the calibration/threshold sweep — the obvious next experiment.
2. Temperature scaling or focal loss to reduce ECE.
3. On-device inference with TFLite/ONNX Mobile so the app works with zero connectivity.
4. Grad-CAM heatmaps so a farmer sees *which part of the leaf* drove the decision.
5. Add nutrient-deficiency and pest classes; multi-leaf / whole-plant photographs.

---

## 11. Plain-English explanation (say it like this in the viva)

**What the project is:** "I took a public dataset of about 54,000 leaf photos covering 38 diseases across 14 crops. I trained a small neural network to look at a leaf photo and say which disease it is. Then I squeezed that network down so it can answer in under a second on a free computer with no GPU, and built a website around it where a farmer can upload a photo and get an answer, a confidence number, and a treatment plan."

**How the model learns (the analogy that works):** "Think of the network as someone learning to recognise diseases from flashcards. At first they only know general shapes — edges, colours, textures — because they were already trained on a million everyday photos. I freeze that general knowledge, teach them the 38 disease names on top, then let them adjust the finer detail-detecting layers slightly. That's transfer learning: teach the specialist last, not first."

**Why a confidence number matters:** "The model always produces 38 numbers that add up to 100%. The biggest number is its guess. I tested what happens at different cut-offs: if I only trust guesses above 85%, it was right 100% of the time on my test set — but it only answered 63% of the photos. Below 60% it refuses and shows the top three candidates instead. Refusing is a feature: a wrong disease name can mean the wrong pesticide."

**Why two engines:** "I also send the same photo to a big pretrained vision model. If both agree, that's strong evidence. If they disagree, my app shows both answers side by side instead of hiding it — that is the honest thing to do, and it is what my evaluation panel displays."

**The number to lead with:** 98.73% top-1 on the held-out test set, 99.95% top-3; 97.73% after compression for deployment.

**The limitation to volunteer:** "Those numbers are on clean, lab-style photos. On real farm photos the same model drops to about 26%. I measured that and documented it, because the interesting engineering question is what the app should do when it is unsure — and my answer is refuse, show alternatives, and cross-check with a second model."

---

## 12. Likely viva questions with good answers

- *Why MobileNetV2 and not ResNet-50?* — Depthwise-separable convolutions: ~3.4M parameters vs 25M, CPU-friendly, and 98.7% is already near the ceiling for this dataset; a bigger model would not raise accuracy, only hosting cost.
- *Why 224×224?* — Native MobileNetV2 input, matching pretrained normalisation; enough detail for lesion texture.
- *Why label smoothing?* — Prevents the network becoming overconfident, which improves both calibration and generalisation; directly relevant because the app's decision rule depends on confidence.
- *Why two-phase training?* — Fine-tuning everything from epoch 1 destroys pretrained features with large gradients; freezing first lets the new head settle.
- *What does int8 quantization do?* — Stores weights in 8-bit integers instead of 32-bit floats: ~4× smaller, faster CPU inference; here it costs 1% top-1.
- *Isn't 98% too good?* — On PlantVillage, no: it is a clean, near-saturated benchmark. Cite Mohanty et al. (2016) which got 99.35% and only 31% on field photos — the gap is the well-known result your project reproduces.
- *What is ECE?* — Expected Calibration Error: bin predictions by confidence and average |accuracy − confidence|. 0.12 means the confidence is off by ~12 points on average.
- *How do you stop it naming a disease for a rice leaf?* — Two guards: the confidence floor (below 0.60 → out-of-scope page, no disease name shown) and the vision model's crop-level scope check, which can veto a merely "fairly sure" CNN guess.
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
