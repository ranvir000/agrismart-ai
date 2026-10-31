"""
Image analysis and disease prediction engine.
Uses PyTorch and MobileNetV2 for Transfer Learning with intelligent fallback.
"""

import os
from PIL import Image
import numpy as np
import requests
import datetime
from werkzeug.utils import secure_filename
import random
from models.disease_info import DISEASES, DISEASE_KEYS, get_disease

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static', 'uploads')
# Every common camera / phone / download format. Uploads are decoded with Pillow
# and re-saved as a clean RGB JPEG, so a format only has to be *decodable* to be
# accepted. AVIF/HEIC work when their optional decoders are installed.
ALLOWED_EXTENSIONS = {
    'png', 'jpg', 'jpeg', 'jpe', 'jfif', 'gif', 'webp', 'bmp', 'dib', 'tif', 'tiff',
    'ico', 'ppm', 'pgm', 'pbm', 'pnm', 'tga', 'jp2', 'j2k', 'jpf', 'avif', 'heic', 'heif',
}
SUPPORTED_FORMATS_LABEL = 'JPG, PNG, WEBP, GIF, BMP, TIFF, ICO, TGA, PPM, JP2, AVIF, HEIC'
MAX_STORED_EDGE = 1600
STORED_JPEG_QUALITY = 88

# Optional decoders: used automatically when present, never required.
try:  # pip install pillow-heif
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pillow_heif = None
try:  # pip install pillow-avif-plugin
    import pillow_avif  # noqa: F401
except Exception:
    pillow_avif = None


class ImageFormatError(ValueError):
    """Raised when an uploaded file cannot be decoded as an image."""

# Global cache for ONNX inference session
_onnx_session = None
_onnx_available = None

# Mapping from ONNX model class indices (0..37) to our DISEASES keys
ONNX_INDEX_TO_DISEASE_KEY = [
    'apple_scab',
    'apple_black_rot',
    'apple_cedar_rust',
    'apple_healthy',
    'blueberry_healthy',
    'cherry_powdery_mildew',
    'cherry_healthy',
    'corn_gray_leaf_spot',
    'corn_common_rust',
    'corn_northern_leaf_blight',
    'corn_healthy',
    'grape_black_rot',
    'grape_black_measles',
    'grape_leaf_blight',
    'grape_healthy',
    'orange_citrus_greening',
    'peach_bacterial_spot',
    'peach_healthy',
    'pepper_bacterial_spot',
    'pepper_healthy',
    'potato_early_blight',
    'potato_late_blight',
    'potato_healthy',
    'raspberry_healthy',
    'soybean_healthy',
    'squash_powdery_mildew',
    'strawberry_leaf_scorch',
    'strawberry_healthy',
    'tomato_bacterial_spot',
    'tomato_early_blight',
    'tomato_late_blight',
    'tomato_leaf_mold',
    'tomato_septoria_leaf_spot',
    'tomato_spider_mites',
    'tomato_target_spot',
    'tomato_yellow_leaf_curl',
    'tomato_mosaic_virus',
    'tomato_healthy'
]


def get_onnx_session():
    """Lazily load the quantized ONNX MobileNetV2 PlantVillage model."""
    global _onnx_session, _onnx_available
    if _onnx_available is False:
        return None
    if _onnx_session is not None:
        return _onnx_session

    try:
        import onnxruntime as ort
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'weights',
            'mobilenet_v2_plantvillage_quantized.onnx'
        )
        if not os.path.exists(model_path):
            print(f"[WARN] ONNX model not found at {model_path}")
            _onnx_available = False
            return None

        # Lightweight CPU session with single thread for minimal resource usage
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 2
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])
        _onnx_session = session
        _onnx_available = True
        return _onnx_session
    except Exception as e:
        print(f"[WARN] Failed to load ONNX model: {e}")
        _onnx_available = False
        return None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def normalize_upload(file_storage):
    """Decode any supported image and store it as a clean RGB JPEG.

    Flattens transparency, ignores animation, applies camera rotation metadata,
    caps the long edge and re-encodes - so the saved photo, the preview and the
    model all receive a well-formed image whatever the user uploaded.

    Returns (unique_filename, filepath, jpeg_bytes); raises ImageFormatError with a
    user-facing message when the bytes are not a readable image.
    """
    import uuid
    from io import BytesIO
    from PIL import Image, ImageOps

    raw = file_storage.read()
    if not raw:
        raise ImageFormatError('That file was empty. Please choose an image and try again.')

    try:
        with Image.open(BytesIO(raw)) as opened:
            opened.load()
            img = ImageOps.exif_transpose(opened)
            if img.mode in ('RGBA', 'LA', 'PA') or (img.mode == 'P' and 'transparency' in img.info):
                rgba = img.convert('RGBA')
                flat = Image.new('RGB', rgba.size, (255, 255, 255))
                flat.paste(rgba, mask=rgba.split()[-1])
                img = flat
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            img.thumbnail((MAX_STORED_EDGE, MAX_STORED_EDGE))
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=STORED_JPEG_QUALITY, optimize=True)
    except Exception:
        raise ImageFormatError(
            f'We could not read that file as an image. Supported types: {SUPPORTED_FORMATS_LABEL}.'
        )

    name = secure_filename(file_storage.filename or '') or 'photo'
    stem = os.path.splitext(name)[0][:60] or 'photo'
    unique_filename = f'{uuid.uuid4().hex}_{stem}.jpg'
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
    with open(filepath, 'wb') as out:
        out.write(buffer.getvalue())
    return unique_filename, filepath, buffer.getvalue()


def analyze_image(image_path):
    """
    Run the on-device MobileNetV2 (PlantVillage, 38 classes) model.

    Returns a dict describing the top-3 classes with the model's real softmax
    probabilities. It never invents an answer: if the model cannot produce a
    prediction, {"ok": False} is returned and the caller tells the user instead
    of showing a made-up disease.
    """
    session = get_onnx_session()
    if session is None:
        return {"ok": False, "error": "on-device model unavailable"}

    try:
        img = Image.open(image_path).convert('RGB').resize((224, 224))

        # These are the torchvision ImageNet statistics the model was trained with
        # (see train/train.py). Feeding a different normalisation silently wrecks accuracy.
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        arr = np.expand_dims(np.transpose(arr, (2, 0, 1)), axis=0)

        input_name = session.get_inputs()[0].name
        logits = session.run(None, {input_name: arr})[0][0]

        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        ranked = np.argsort(probs)[::-1][:3]
        top = []
        for idx in ranked:
            key = ONNX_INDEX_TO_DISEASE_KEY[int(idx)]
            top.append({
                "key": key,
                "name": get_disease(key)["name"],
                "confidence": round(float(probs[int(idx)]), 4),
            })

        best = top[0]
        disease_info = get_disease(best["key"])
        return {
            "ok": True,
            "disease_key": best["key"],
            "confidence": best["confidence"],
            "severity": disease_info.get("severity_default", "Moderate"),
            "alternatives": top[1:],
            "source": "on-device",
            "note": "",
        }
    except Exception as exc:
        print(f"[ERROR] On-device inference failed: {exc}")
        return {"ok": False, "error": str(exc)}


def get_weather(city="New Delhi"):
    """
    Get weather data for a city. Uses OpenWeatherMap API if WEATHER_API_KEY is set,
    otherwise returns realistic mock weather data.
    """
    api_key = os.environ.get('WEATHER_API_KEY', 'demo')

    if api_key and api_key != 'demo':
        try:
            url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric&cnt=5"
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                forecasts = []
                for item in data['list'][:5]:
                    forecasts.append({
                        'date': datetime.datetime.fromtimestamp(item['dt']).strftime('%a %d %b'),
                        'temp_max': round(item['main']['temp_max']),
                        'temp_min': round(item['main']['temp_min']),
                        'humidity': item['main']['humidity'],
                        'description': item['weather'][0]['description'].title(),
                        'icon': item['weather'][0]['icon'],
                        'wind': round(item['wind']['speed'] * 3.6),
                    })
                return {
                    'city': data['city']['name'],
                    'country': data['city']['country'],
                    'forecasts': forecasts,
                    'current_temp': round(data['list'][0]['main']['temp']),
                    'current_humidity': data['list'][0]['main']['humidity'],
                    'source': 'live'
                }
        except Exception as e:
            print(f"Weather API error: {e}")

    # Realistic mock weather data for any location when API key is demo
    now = datetime.datetime.now()
    month = now.month
    base_temps = {
        1: (8, 20), 2: (10, 24), 3: (15, 30), 4: (20, 37),
        5: (25, 42), 6: (28, 40), 7: (26, 35), 8: (25, 34),
        9: (24, 33), 10: (18, 32), 11: (13, 27), 12: (9, 22)
    }
    t_min, t_max = base_temps.get(month, (20, 32))
    conditions = [
        ("Clear Sky", "01d"), ("Few Clouds", "02d"),
        ("Scattered Clouds", "03d"), ("Light Rain", "10d")
    ]
    forecasts = []
    for i in range(5):
        day = now + datetime.timedelta(days=i)
        variation = (i * 2) % 5 - 2
        cond_idx = (i + month) % len(conditions)
        humidity = 65 + (i * 3) % 20
        forecasts.append({
            'date': day.strftime('%a %d %b'),
            'temp_max': t_max + variation,
            'temp_min': t_min + variation,
            'humidity': humidity,
            'description': conditions[cond_idx][0],
            'icon': conditions[cond_idx][1],
            'wind': 15 + (i * 4) % 12,
        })

    return {
        'city': city,
        'country': 'IN',
        'forecasts': forecasts,
        'current_temp': t_max - 2,
        'current_humidity': forecasts[0]['humidity'],
        'source': 'demo'
    }


def get_weather_advisory(weather_data, recent_scans):
    """Generate crop advisory based on weather conditions."""
    advisories = []
    humidity = weather_data.get('current_humidity', 60)
    temp = weather_data.get('current_temp', 25)
    forecasts = weather_data.get('forecasts', [])

    rain_days = sum(1 for f in forecasts if 'rain' in f.get('description', '').lower())

    if humidity > 80:
        advisories.append({
            'icon': 'fa-tint',
            'color': '#3498db',
            'title': 'High Humidity Alert',
            'message': f'Humidity is {humidity}%. High risk of fungal diseases (Late Blight, Leaf Mold, Powdery Mildew). Apply preventive fungicide spray within 24 hours.'
        })
    if rain_days >= 3:
        advisories.append({
            'icon': 'fa-cloud-rain',
            'color': '#2980b9',
            'title': 'Extended Wet Period',
            'message': f'{rain_days} rainy days forecast. Avoid overhead irrigation. Ensure good field drainage. Fungicide protection is critical.'
        })
    if temp > 38:
        advisories.append({
            'icon': 'fa-thermometer-full',
            'color': '#e74c3c',
            'title': 'Heat Stress Warning',
            'message': f'Temperature {temp}°C. Risk of Spider Mites and tip burn. Increase irrigation frequency. Apply kaolin clay to reflect heat.'
        })
    if temp < 12:
        advisories.append({
            'icon': 'fa-snowflake',
            'color': '#3498db',
            'title': 'Cold Stress Risk',
            'message': f'Temperature dropping to {temp}°C. Protect sensitive crops with row covers. Risk of frost damage to blossoms.'
        })
    if humidity < 40:
        advisories.append({
            'icon': 'fa-sun',
            'color': '#f39c12',
            'title': 'Low Humidity / Dry Conditions',
            'message': f'Humidity at {humidity}%. Ideal conditions for Spider Mites and Powdery Mildew. Monitor plants daily. Maintain soil moisture.'
        })

    if recent_scans:
        disease_keys = [s.disease_key for s in recent_scans[:3]]
        if any('blight' in k for k in disease_keys):
            advisories.append({
                'icon': 'fa-exclamation-triangle',
                'color': '#e74c3c',
                'title': 'Active Blight — Critical Action Needed',
                'message': 'Blight detected in your recent scans. In current weather conditions, disease spreads rapidly. Apply fungicide immediately and remove infected plant parts.'
            })

    if not advisories:
        advisories.append({
            'icon': 'fa-check-circle',
            'color': '#27ae60',
            'title': 'Conditions Look Favorable',
            'message': f'Current conditions (Temp: {temp}°C, Humidity: {humidity}%) are within normal range. Continue regular monitoring. Good time for preventive spray.'
        })

    return advisories
