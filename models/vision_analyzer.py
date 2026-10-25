"""
Optional vision-model analysis (Google Gemini).

The bundled PlantVillage classifier was trained on lab photos and does not read
real-world phone/internet pictures reliably. When GEMINI_API_KEY is configured the
app asks a vision model instead and maps its answer onto the app's 38 disease keys,
so the rest of the app (treatments, history, PDF) keeps working unchanged.
"""

import base64
import json
import mimetypes
import os

import requests

from models.disease_info import DISEASES

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_MODEL = "gemini-2.5-flash"

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


def api_key():
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def is_available():
    return bool(api_key())


def analyze_with_vision(image_path):
    """Return a prediction dict, or None when vision analysis is unavailable or fails."""
    key = api_key()
    if not key:
        return None

    model = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
    try:
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

        response = requests.post(
            API_URL.format(model=model), params={"key": key}, json=payload, timeout=60
        )
        if response.status_code != 200:
            print(f"[WARN] Vision analysis HTTP {response.status_code}: {response.text[:200]}")
            return None

        text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        data = json.loads(text)

        disease_key = data.get("disease_key")
        if disease_key not in DISEASES:
            return None

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
    except Exception as exc:  # network error, quota, malformed JSON
        print(f"[WARN] Vision analysis failed: {exc}")
        return None
