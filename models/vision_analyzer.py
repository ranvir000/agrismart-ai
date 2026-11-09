"""
Optional vision-model analysis (Google Gemini).

The bundled PlantVillage classifier was trained on lab photos and does not read
real-world phone/internet pictures reliably, so when an API key is configured the app
asks a vision model instead - but only about the 14 crops the app actually covers. The
crop is identified first: a plant that is not one of them is reported as out of scope
instead of being pushed into the nearest supported disease.

Google retires model names over time and also answers with transient 503 "high demand"
errors, so several models are tried in order with one retry each.
The exact failure reason of the last call is available via last_error().
"""

import base64
import json
import mimetypes
import os
import time

import requests

from models.disease_info import DISEASES

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
# Prefer the current stable model; keep one current fallback for deployments with
# account-specific model availability. See the Google AI Studio model catalog.
CANDIDATE_MODELS = ["gemini-3.8-flash", "gemini-3.6-flash"]
GEMINI_TIMEOUT_SECONDS = 20

# The crops the 38 classes cover. A photo of any other plant is out of scope.
SUPPORTED_CROPS = sorted({d.get("crop") for d in DISEASES.values() if d.get("crop")})
UNSUPPORTED_CROP_LABELS = ["Other", "Not a plant"]
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
RETRY_DELAYS = (1.5,)

PROMPT = """You are a plant pathologist examining one photo.

STEP 1 - Identify the crop (the plant species), choosing exactly one of:
{crops}
Answer "Other" if the plant is not one of these crops, or "Not a plant" if there is no plant leaf in the photo.

STEP 2 - Only if STEP 1 named one of those crops, choose the single best matching class key from this list (its crop must be the same as STEP 1). Otherwise leave disease_key as an empty string:
{keys}

Rules:
- Be strict about the crop. Never force a crop from the list onto a plant that looks different.
- Answer from what is visible in the photo; phone, field and internet photos are expected.
- confidence is your own probability between 0 and 1 for the disease_key you chose.
- note: one short sentence naming the plant you see and the symptoms visible."""

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "crop": {"type": "STRING", "enum": SUPPORTED_CROPS + UNSUPPORTED_CROP_LABELS},
        "disease_key": {"type": "STRING"},
        "confidence": {"type": "NUMBER"},
        "note": {"type": "STRING"},
    },
    "required": ["crop", "disease_key", "confidence", "note"],
}

_last_error = None


class VisionError(RuntimeError):
    """A failed vision call; status carries the HTTP code when there was one."""

    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


def api_key():
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def is_available():
    return bool(api_key())


def last_error():
    """Why the most recent analyze_with_vision call failed (None if it did not)."""
    return _last_error


def _post(model, key, image_path):
    with open(image_path, "rb") as handle:
        image_b64 = base64.b64encode(handle.read()).decode("ascii")
    mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"

    payload = {
        "contents": [{
            "parts": [
                {"text": PROMPT.format(
                    crops="\n".join(SUPPORTED_CROPS + UNSUPPORTED_CROP_LABELS),
                    keys=", ".join(DISEASES.keys()),
                )},
                {"inline_data": {"mime_type": mime, "data": image_b64}},
            ]
        }],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": SCHEMA,
            "temperature": 0,
        },
    }
    # Send the key as a header (works for both old AIza and new AQ.* key formats).
    return requests.post(
        API_URL.format(model=model),
        json=payload,
        headers={"x-goog-api-key": key},
        timeout=GEMINI_TIMEOUT_SECONDS,
    )


def _call_gemini(model, key, image_path):
    """One request. Returns a prediction, or raises VisionError with the reason."""
    try:
        response = _post(model, key, image_path)
    except requests.RequestException as exc:
        raise VisionError(f"Network error from {model}: {exc}") from exc
    if response.status_code != 200:
        raise VisionError(
            f"HTTP {response.status_code} from {model}: {response.text[:150]}",
            response.status_code,
        )

    try:
        text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        data = json.loads(text)
    except (KeyError, IndexError, ValueError) as exc:
        raise VisionError(f"{model} returned an unreadable answer: {exc}")

    crop = (data.get("crop") or "").strip()
    disease_key = (data.get("disease_key") or "").strip()
    note = (data.get("note") or "").strip()
    try:
        confidence = max(0.0, min(1.0, float(data.get("confidence") or 0)))
    except (TypeError, ValueError):
        confidence = 0.0

    # The plant itself is outside the library: refuse to name a disease at all.
    if crop not in SUPPORTED_CROPS:
        return {
            "ok": False,
            "out_of_scope": True,
            "crop": crop or "Unknown",
            "crop_count": len(SUPPORTED_CROPS),
            "confidence": round(confidence, 4),
            "note": note,
            "reason": f"the model identified the plant as {crop or 'unrecognised'}",
        }

    if disease_key not in DISEASES:
        raise VisionError(f"{model} returned unknown class key {disease_key!r}")
    disease = DISEASES[disease_key]
    if disease.get("crop") != crop:
        raise VisionError(f"{model} named {disease_key} but said the crop was {crop}")

    return {
        "ok": True,
        "disease_key": disease_key,
        "confidence": round(confidence, 4),
        "severity": disease.get("severity_default", "Moderate"),
        "alternatives": [],
        "source": "vision",
        "crop": crop,
        "note": note,
    }


def analyze_with_vision(image_path):
    """Return a prediction dict, or None when vision analysis is unavailable or fails."""
    global _last_error
    _last_error = None

    key = api_key()
    if not key:
        _last_error = "GEMINI_API_KEY is not configured"
        return None

    models = CANDIDATE_MODELS
    if os.environ.get("GEMINI_MODEL"):
        models = [os.environ["GEMINI_MODEL"]] + CANDIDATE_MODELS

    tried = []
    for model in models:
        if model in tried:
            continue
        tried.append(model)
        for attempt in range(1 + len(RETRY_DELAYS)):
            try:
                return _call_gemini(model, key, image_path)
            except VisionError as exc:
                _last_error = str(exc)
                print(f"[WARN] Vision analysis with {model} failed: {exc}")
                if exc.status in RETRYABLE_STATUS:
                    if attempt < len(RETRY_DELAYS):
                        time.sleep(RETRY_DELAYS[attempt])
                        continue
                    break      # model still overloaded: try the next one
                if exc.status in (400, 404):
                    break      # retired / unavailable model name: try the next one
                return None    # inconsistent answer: do not guess
    return None
