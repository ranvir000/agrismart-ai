"""
Optional vision-model analysis (Google Gemini).

The bundled PlantVillage classifier was trained on lab photos and does not read
real-world phone/internet pictures reliably. When GEMINI_API_KEY is configured the
app asks a vision model instead and maps its answer onto the app's 38 disease keys,
so the rest of the app (treatments, history, PDF) keeps working unchanged.

Google retires old model names over time, so several models are tried in order.
The exact failure reason of the last call is available via last_error().
"""

import base64
import json
import mimetypes
import os

import requests

from models.disease_info import DISEASES

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
# Google has been retiring older model names; try the default first, then newer ones.
CANDIDATE_MODELS = ["gemini-2.5-flash", "gemini-3.6-flash", "gemini-3.7-flash"]

PROMPT = """You are a plant pathologist examining one photo.

Pick the single best matching class key from this list:
{keys}

Rules:
- Answer from what is visible in the photo; phone, field and internet photos are expected.
- confidence is your own probability between 0 and 1 for the key you chose.
- If the photo is not a plant or leaf, still return the closest key but with confidence below 0.25.
- note: one short sentence describing the crop and the symptoms you can see."""

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "disease_key": {"type": "STRING", "enum": list(DISEASES.keys())},
        "confidence": {"type": "NUMBER"},
        "note": {"type": "STRING"},
    },
    "required": ["disease_key", "confidence", "note"],
}

_last_error = None


def api_key():
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def is_available():
    return bool(api_key())


def last_error():
    """Why the most recent analyze_with_vision call failed (None if it did not)."""
    return _last_error


def _call_gemini(model, key, image_path):
    """One request. Returns a prediction dict, or raises RuntimeError with the reason."""
    with open(image_path, "rb") as handle:
        image_b64 = base64.b64encode(handle.read()).decode("ascii")
    mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"

    payload = {
        "contents": [{
            "parts": [
                {"text": PROMPT.format(keys=", ".join(DISEASES.keys()))},
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
    response = requests.post(
        API_URL.format(model=model),
        json=payload,
        headers={"x-goog-api-key": key},
        timeout=60,
    )
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code} from {model}: {response.text[:150]}")

    text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
    data = json.loads(text)

    disease_key = data.get("disease_key")
    if disease_key not in DISEASES:
        raise RuntimeError(f"{model} returned unknown class key {disease_key!r}")

    confidence = max(0.0, min(1.0, float(data.get("confidence") or 0)))
    disease = DISEASES[disease_key]
    return {
        "ok": True,
        "disease_key": disease_key,
        "confidence": round(confidence, 4),
        "severity": disease.get("severity_default", "Moderate"),
        "alternatives": [],
        "source": "vision",
        "note": (data.get("note") or "").strip(),
    }


def analyze_with_vision(image_path):
    """Return a prediction dict, or None when vision analysis is unavailable or fails."""
    global _last_error
    _last_error = None

    key = api_key()
    if not key:
        _last_error = "GEMINI_API_KEY is not configured"
        return None

    models = [os.environ.get("GEMINI_MODEL")] + CANDIDATE_MODELS if os.environ.get("GEMINI_MODEL") else CANDIDATE_MODELS
    tried = []
    for model in models:
        if model in tried:
            continue
        tried.append(model)
        try:
            return _call_gemini(model, key, image_path)
        except Exception as exc:
            reason = str(exc)
            _last_error = reason
            print(f"[WARN] Vision analysis with {model} failed: {reason}")
            # 400/404 usually mean the model name is retired/unavailable: try the next one.
            if "HTTP 404" in reason or "HTTP 400" in reason:
                continue
            return None
    return None
