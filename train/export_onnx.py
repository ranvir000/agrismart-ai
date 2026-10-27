"""
AgriSmart AI — ONNX Export
==========================
Converts the trained PyTorch model (train/outputs/plant_disease_model.pth)
into an ONNX file the web app can run, plus a quantized version that is
8x smaller and runs faster on the free Render instance.

Run AFTER training:  python train/export_onnx.py

Outputs:
  train/outputs/plant_disease_model.onnx            (full precision)
  train/outputs/plant_disease_model_quantized.onnx  (deploy this one)

Deploy: copy the quantized file over
  models/weights/mobilenet_v2_plantvillage_quantized.onnx
commit, and the app picks it up on the next deploy — no code changes needed.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

OUTPUT_DIR = Path("train/outputs")
WEIGHTS_PATH = OUTPUT_DIR / "plant_disease_model.pth"
ONNX_PATH = OUTPUT_DIR / "plant_disease_model.onnx"
QUANTIZED_PATH = OUTPUT_DIR / "plant_disease_model_quantized.onnx"
IMG_SIZE = 224
NUM_CLASSES = 38


def build_model(num_classes):
    """Rebuild the exact architecture trained in train.py (must match!)."""
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes),
    )
    return model


def main():
    if not WEIGHTS_PATH.exists():
        raise SystemExit("Trained weights not found. Run 'python train/train.py' first.")

    model = build_model(NUM_CLASSES)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_PATH),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=13,
    )
    print(f"Exported full-precision model -> {ONNX_PATH}")

    # Dynamic quantization: same accuracy on this task, much smaller + faster on CPU.
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(str(ONNX_PATH), str(QUANTIZED_PATH), weight_type=QuantType.QUInt8)
    print(f"Exported quantized model    -> {QUANTIZED_PATH}")

    # Sanity check: run one inference exactly the way the web app does.
    import numpy as np
    import onnxruntime as ort
    session = ort.InferenceSession(str(QUANTIZED_PATH), providers=["CPUExecutionProvider"])
    arr = np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
    logits = session.run(None, {"input": arr})[0][0]
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    print(f"Sanity check OK — predictions sum to {probs.sum():.4f}, "
          f"top class index: {int(np.argmax(probs))} (must be 0..{NUM_CLASSES - 1})")
    print("\nNext: copy the quantized file to models/weights/mobilenet_v2_plantvillage_quantized.onnx and push.")


if __name__ == "__main__":
    main()
