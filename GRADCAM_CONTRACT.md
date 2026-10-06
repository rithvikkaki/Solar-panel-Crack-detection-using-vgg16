# SolarSentinel AI - Grad-CAM Explainability Contract

**Document Version:** 1.0.0  
**Authors:** Explainable AI Engineer (Agent 3) & Lead Orchestrator  
**Status:** Signed Contract  
**Consumer Agents:** Backend Engineer (Agent 4), Frontend Engineer (Agent 6), QA Engineer (Agent 7)  
**Date:** September 11, 2026  

---

## 1. Scientific & Ethical Mandate

> [!IMPORTANT]
> **Strict Interpretability Boundary:**  
> All system communications, API docstrings, UI labels, and reports **must** use the following exact scientific language or its direct equivalent:  
> *"Grad-CAM visualizes image regions that contributed strongly to the model's classification."*  
>
> The system **MUST NOT** claim:
> * Exact defect localization
> * Pixel-level segmentation boundaries
> * Physical fracture measurement
> * Certified diagnostic or electrical certainty  
>
> Grad-CAM is an explanatory visual attribution method for convolutional neural networks, not a physical defect measurement tool.

---

## 2. Input Specification

| Parameter | Specification | Constraints |
|---|---|---|
| **Image Source** | `Union[str, bytes, Image.Image, np.ndarray]` | Validated image filepath, in-memory byte buffer, PIL Image, or RGB uint8 array. |
| **Color Space** | RGB (at entry) | Single-channel grayscale images are automatically expanded to 3-channel RGB. |
| **Model** | Keras Functional Model | Pre-trained VGG16 backbone or compatible CNN with at least one 2D convolutional layer. |
| **Preprocessing** | [`ml/inference/preprocessing.py`](file:///c:/Users/victus/Downloads/Solar-panel-Crack-detection-using-vgg16-main/ml/inference/preprocessing.py) | **Mandatory:** Same canonical VGG16 zero-centered BGR preprocessing used for inference. |
| **Target Class** | `Optional[int]` | If `None`, defaults automatically to $\operatorname{argmax}(\hat{y})$ (the model's top predicted class). |
| **Overlay Alpha** | `float` (default `0.5`) | Blending parameter between `0.0` (pure heatmap) and `1.0` (pure original image). |

---

## 3. Output Specification

Grad-CAM produces a structured [`GradCAMResult`](file:///c:/Users/victus/Downloads/Solar-panel-Crack-detection-using-vgg16-main/ml/explainability/gradcam.py) instance:

| Artifact | Data Type | Dimensions | Value Range | Encoding |
|---|---|---|---|---|
| **Original Image** | `np.ndarray` (uint8) | $(H, W, 3)$ | $[0, 255]$ | Base64 PNG Data URI |
| **Raw Heatmap** | `np.ndarray` (float32) | $(H, W)$ | $[0.0, 1.0]$ | 2D numerical array |
| **Colorized Heatmap** | `np.ndarray` (uint8) | $(H, W, 3)$ | $[0, 255]$ (JET colormap) | Base64 PNG Data URI |
| **Overlay Image** | `np.ndarray` (uint8) | $(H, W, 3)$ | $[0, 255]$ | Base64 PNG Data URI |

### Invariance Rules:
1. **Dimension Parity:** All output images are resized to $(H, W)$, matching the **original uploaded image resolution** exactly.
2. **Immutability:** The input image buffer is copied; the original pixel values remain strictly unaltered.

---

## 4. Backend Integration Contract

### Primary Interface:
```python
from ml.explainability.gradcam import GradCAMExplainer, GradCAMResult, GradCAMError

explainer = GradCAMExplainer(
    model=loaded_keras_model,
    class_names=classes_list,
    target_layer_name=None,  # None triggers automatic discovery of deepest Conv2D layer
    input_size=(244, 244)
)

result: GradCAMResult = explainer.explain(
    image_source=raw_image_bytes,
    target_class_index=None,  # Auto-selects top prediction
    overlay_alpha=0.5
)
```

### Serialized Payload for REST Responses:
```json
{
  "target_class_index": 4,
  "target_class_name": "Physical-Damage",
  "target_layer_name": "block5_conv3",
  "original_dimensions": [600, 800],
  "disclaimer": "Grad-CAM visualizes image regions that contributed strongly to the model's classification. It reflects neural network feature activations and does not constitute precise physical defect boundary detection or certified diagnostic certainty.",
  "images": {
    "original": "data:image/png;base64,iVBORw0KGgo...",
    "heatmap": "data:image/png;base64,iVBORw0KGgo...",
    "overlay": "data:image/png;base64,iVBORw0KGgo..."
  }
}
```

---

## 5. Error Behavior & Exception Hierarchy

All explainability failures raise [`GradCAMError`](file:///c:/Users/victus/Downloads/Solar-panel-Crack-detection-using-vgg16-main/ml/explainability/gradcam.py):

| Failure Mode | Root Cause | Handling Strategy |
|---|---|---|
| **Model Unavailable** | Model is `None` | Raise `GradCAMError("Cannot initialize GradCAMExplainer with None model.")` |
| **No Conv Layer** | Architecture lacks Conv2D | Raise `GradCAMError("No convolutional layer (Conv2D) found in model architecture.")` |
| **Invalid Image** | Corrupt/unreadable file | Raise `GradCAMError("Unsupported image source type...")` |
| **Gradient Failure** | Disconnected graph | Raise `GradCAMError("Gradient computation failed during Grad-CAM: ...")` |

The FastAPI backend must intercept `GradCAMError` and return an informative HTTP 500 or HTTP 422 JSON response without leaking raw tracebacks.
