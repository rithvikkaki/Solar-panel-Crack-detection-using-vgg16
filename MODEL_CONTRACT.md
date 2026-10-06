# SolarSentinel AI - Model Contract

**Document Version:** 1.0.0  
**Authors:** ML Engineer (Agent 2) & Lead Orchestrator  
**Status:** Signed Contract  
**Consumer Agents:** Explainable AI Engineer (Agent 3), Backend Engineer (Agent 4), QA Engineer (Agent 7)  
**Date:** September 11, 2026  

---

## 1. Contract Overview

This document specifies the interface contract between the machine learning artifacts (`ml/`) and downstream components including Grad-CAM explainability, backend inference services, and evaluation harnesses. Any modification to the model architecture, input shapes, normalization methods, or output formats must update this contract.

---

## 2. Model Specifications

| Parameter | Specification | Scientific / Engineering Rationale |
|---|---|---|
| **Model Format** | Keras V3 native format (`.keras`) | Modern, portable, self-contained architecture + weights standard. Legacy H5 and SavedModel folders are deprecated. |
| **Model Path** | `ml/models/solar_sentinel_vgg16.keras` | Centralized repository path. Backend must load from this relative path or environment variable `MODEL_PATH`. |
| **Backbone** | VGG16 (ImageNet pre-trained) | Feature extractor preserving original project research foundation. |
| **Classification Head** | Head B: GAP -> BatchNorm -> Dense(512, ReLU) -> Dropout(0.3) -> Dense(256, ReLU) -> Dropout(0.2) -> Softmax(6) | 2-stage MLP head with Batch Normalization preserving localized spatial representations and preventing representation collapse. |
| **Input Dimensions** | `(244, 244, 3)` *(Baseline & Production)*<br>`(224, 224, 3)` *(Alternative)* | Preserves original notebook baseline (244x244x3). Configurable via `TrainingConfig.img_height` / `img_width`. |
| **Input Color Order** | RGB (at ingestion) -> BGR (after preprocessing) | Standard PIL/FastAPI RGB decoded image converted to BGR for VGG16 ImageNet weights. |
| **Input Data Type** | `float32` | 32-bit floating point precision tensor. |
| **Output Activation** | `Softmax` | Dynamic multi-class probability distribution summing strictly to 1.0 (replaces unactivated Dense(90)). |
| **Output Dimension** | Dynamic `(None, num_classes)` | Automatically derived from verified class count (default 6 for verified dataset). |

---

## 3. Preprocessing Specification

All downstream components (training, validation, inference, Grad-CAM) **MUST** utilize identical preprocessing implemented in [`ml/inference/preprocessing.py`](file:///c:/Users/victus/Downloads/Solar-panel-Crack-detection-using-vgg16-main/ml/inference/preprocessing.py):

1. **Resize:** Bilinear interpolation to `(244, 244)` (or configured dimension).
2. **Channel Reversal:** Convert RGB to BGR (`x[..., ::-1]`).
3. **Mean Subtraction:** Subtract ImageNet BGR channel means without 1/255 division:
   * **Blue (B):** $103.939$
   * **Green (G):** $116.779$
   * **Red (R):** $123.680$

```python
# Canonical NumPy implementation
x = img_array.astype(np.float32, copy=True)[..., ::-1]
x[..., 0] -= 103.939  # Blue
x[..., 1] -= 116.779  # Green
x[..., 2] -= 123.680  # Red
```

---

## 4. Class Mapping & Vocabulary

The canonical verified classes from the source dataset audit are indexed alphabetically by default `image_dataset_from_directory` behavior:

```json
{
  "classes": [
    "Bird-drop",
    "Clean",
    "Dusty",
    "Electrical-damage",
    "Physical-Damage",
    "Snow-Covered"
  ],
  "class_to_idx": {
    "Bird-drop": 0,
    "Clean": 1,
    "Dusty": 2,
    "Electrical-damage": 3,
    "Physical-Damage": 4,
    "Snow-Covered": 5
  }
}
```

*Note: If a new dataset is supplied with additional or modified classes, `ml/training/train.py` dynamically writes the detected class mapping to `ml/metadata/class_names.json`, and the inference engine loads this dynamically.*

---

## 5. Output Probability & Response Format

The inference engine (`SolarSentinelPredictor.predict()`) returns a typed dictionary structured as follows:

```json
{
  "predicted_condition": "Physical-Damage",
  "confidence": 0.9421,
  "class_index": 4,
  "class_probabilities": {
    "Bird-drop": 0.0081,
    "Clean": 0.0012,
    "Dusty": 0.0145,
    "Electrical-damage": 0.0341,
    "Physical-Damage": 0.9421,
    "Snow-Covered": 0.0000
  },
  "all_classes": [
    "Bird-drop",
    "Clean",
    "Dusty",
    "Electrical-damage",
    "Physical-Damage",
    "Snow-Covered"
  ],
  "num_classes": 6,
  "input_dimensions": [244, 244]
}
```

### Constraints:
* All probabilities must be floats between `0.0` and `1.0`.
* The sum of `class_probabilities.values()` must equal `1.0 ± 0.001`.
* `confidence` must equal `max(class_probabilities.values())`.
* `predicted_condition` must equal the key with the highest confidence score.

---

## 6. Scientific Honesty & Graceful Absence Protocol

In accordance with strict scientific honesty principles:

1. **Missing Weights Behavior:** If `solar_sentinel_vgg16.keras` does not exist on disk, `is_model_loaded` evaluates to `False`.
2. **Refusal to Fabricate:** Under no circumstances will mock predictions or hardcoded random probabilities be generated in lieu of model inference.
3. **Backend Reporting:** The backend `/health` endpoint reports `model_loaded: false`.
4. **Endpoint Behavior:** Calls to `/api/v1/inspect` when `model_loaded: false` return HTTP 503 (Service Unavailable) with descriptive guidance:
   ```json
   {
     "detail": "Model weights are not loaded. Please train the model using `python ml/training/train.py` with the dataset before running inference."
   }
   ```
5. **Zero Invented Metrics:** `evaluation_metrics.json` is generated **only** when `ml/training/evaluate.py` successfully completes a pass over validation/test data.
