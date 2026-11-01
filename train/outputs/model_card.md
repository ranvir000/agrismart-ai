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

## Test-set results
- Top-1 accuracy: 98.73%
- Top-3 accuracy: 99.95%
- Macro F1: 0.9876 | Weighted F1: 0.9873
- Expected calibration error: 0.1162
- Best validation accuracy during training: 98.73%

## Limitations
- Only the 38 PlantVillage classes are supported; photos of other
  crops or non-plant images are designed to be rejected via a confidence threshold
  (see threshold_sweep.csv) rather than guessed.
- PlantVillage images were captured on plain backgrounds; real field photos with soil,
  clutter and variable lighting are harder, which is why the app also offers a
  cloud vision model as a second engine.

## Classes
Apple___Apple_scab, Apple___Black_rot, Apple___Cedar_apple_rust, Apple___healthy, Blueberry___healthy, Cherry_(including_sour)___Powdery_mildew, Cherry_(including_sour)___healthy, Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot, Corn_(maize)___Common_rust_, Corn_(maize)___Northern_Leaf_Blight, Corn_(maize)___healthy, Grape___Black_rot, Grape___Esca_(Black_Measles), Grape___Leaf_blight_(Isariopsis_Leaf_Spot), Grape___healthy, Orange___Haunglongbing_(Citrus_greening), Peach___Bacterial_spot, Peach___healthy, Pepper,_bell___Bacterial_spot, Pepper,_bell___healthy, Potato___Early_blight, Potato___Late_blight, Potato___healthy, Raspberry___healthy, Soybean___healthy, Squash___Powdery_mildew, Strawberry___Leaf_scorch, Strawberry___healthy, Tomato___Bacterial_spot, Tomato___Early_blight, Tomato___Late_blight, Tomato___Leaf_Mold, Tomato___Septoria_leaf_spot, Tomato___Spider_mites Two-spotted_spider_mite, Tomato___Target_Spot, Tomato___Tomato_Yellow_Leaf_Curl_Virus, Tomato___Tomato_mosaic_virus, Tomato___healthy
