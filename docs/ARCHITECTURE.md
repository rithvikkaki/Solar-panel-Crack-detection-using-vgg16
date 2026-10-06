# SolarSentinel AI — System Architecture & Scientific Decisions

## 1. System Overview

**SolarSentinel AI** is an explainable AI (XAI) platform designed for photovoltaic (PV) fault intelligence. It combines transfer learning for image classification with Gradient-weighted Class Activation Mapping (Grad-CAM) to provide transparent, interpretable visual evidence alongside condition classification and rule-based maintenance recommendations.

```
┌─────────────────┐       ┌──────────────────────┐       ┌────────────────────────┐
│  React 19 + TS  │ <───> │  FastAPI Backend     │ <───> │  TensorFlow / Keras 3  │
│  Tailwind + Lucide │   │  (Port 8000)         │       │  VGG16 + Grad-CAM Engine│
└─────────────────┘       └──────────────────────┘       └────────────────────────┘
```

---

## 2. Component Architecture

### 2.1 Backend (`backend/`)
- **FastAPI Application (`backend/app/main.py`)**: Asynchronous REST API server configured with CORS middleware, lifespan events for single-load model caching, and custom structured error handlers.
- **Inference Service (`backend/app/services/inspection_service.py`)**:
  - Image preprocessing: Resizing to $(224, 224)$, RGB conversion, VGG16 standard mean-subtraction preprocessing.
  - Softmax prediction across 6 defect classes.
  - Confidence scoring and rule-based confidence policy (`HIGH` $\ge 0.85$, `MODERATE` $0.60-0.85$, `LOW` $< 0.60$).
  - Heuristic Health Index calculation ($0-100$) and advisory maintenance action mapping.
- **Grad-CAM Engine (`backend/app/services/gradcam_service.py`)**:
  - Targets the final convolutional layer of VGG16: `block5_conv3`.
  - Computes exact gradients of the target class score with respect to convolutional feature maps.
  - Generates pooled weight attribution, applies ReLU rectification, normalizes heatmaps, and overlays viridis/jet attributions as base64 data URIs.
- **API Endpoints**:
  - `POST /api/v1/inspect`: Real production inference with Grad-CAM generation.
  - `POST /api/v1/demo/inspect`: Fallback demo mode for UI evaluation.
  - `GET /api/v1/model/metrics`: Serves verified test set evaluation and hardware latency telemetry.
  - `GET /api/v1/demo/sample-images`: Serves verified sample images for all 6 defect categories.
  - `GET /api/v1/health`: Liveness & model readiness probe.

### 2.2 Frontend (`frontend/`)
- **React 19 & TypeScript**: Strict typing aligned with backend schemas.
- **Vite 8**: Modern bundling and ultra-fast dev server.
- **Tailwind CSS**: Dark-mode industrial interface designed for field inspection clarity.
- **Key Modules**:
  - `InspectionWorkspace`: Dual-pane ingestion and telemetry visualization.
  - `ModelPerformance`: Real test set metrics table and latency profile.
  - `GradCAMViewer`: Interactive side-by-side and blend comparison slider.
  - `ImageUpload`: File drag-and-drop and 6-class instant sample fixture loader.

---

## 3. Scientific Decision: Why VGG16 Was Retained

During Phase 6 benchmarking and exploration, candidate lightweight backbones were evaluated against the primary VGG16 transfer learning model.

### 3.1 Empirical Comparison
| Metric / Characteristic | VGG16 Production Model | MobileNetV2 (Benchmark) | EfficientNetB0 (Benchmark) |
| :--- | :--- | :--- | :--- |
| **Test Accuracy (131 held-out)** | **82.44%** | 83.21% | 84.73% |
| **Macro F1 Score** | **0.8378** | 0.8245 | 0.8418 |
| **Weighted F1 Score** | **0.8235** | 0.8309 | 0.8462 |
| **Physical-Damage Recall** | **80.00%** (8/10) | 80.00% | 70.00% |
| **Physical-Damage F1** | **0.8000** | 0.7619 | 0.7778 |
| **Grad-CAM Feature Coherence** | **High (Direct Conv Blocks)** | Diffuse (Depthwise convs) | Diffuse (Compound blocks) |
| **Forward Inference Latency** | **209.1 ms** (p50: 207.3 ms) | 62.4 ms | 118.7 ms |
| **Grad-CAM Latency** | **650.58 ms** (p50: 649.0 ms) | ~420 ms | ~480 ms |
| **Total Pipeline Latency** | **859.68 ms** (p50: 860.5 ms) | ~482 ms | ~598 ms |
| **Model Disk Size** | 115.6 MB archive (56.6 MB weights) | 14.2 MB | 21.8 MB |

### 3.2 Key Justification Factors
1. **Explainability Quality (Grad-CAM Coherence)**:
   - VGG16's standard $3 \times 3$ convolutional feature hierarchy in `block5_conv3` produces clean, localized activation maps corresponding directly to physical panel fissures, dust accumulation, and localized hotspots.
   - Depthwise separable convolutions (MobileNetV2) and compound-scaled blocks (EfficientNet) often produce noisy, diffuse gradients with Grad-CAM, requiring heavier Score-CAM or Grad-CAM++ post-processing to obtain equivalent spatial clarity.
2. **Defect Generalization on Critical Classes**:
   - VGG16 achieved **80.00% recall** and **0.8000 F1** on the high-risk `Physical-Damage` class, outperforming EfficientNetB0 (70.00% recall, 0.7778 F1).
3. **Acceptable Latency Budget**:
   - The end-to-end CPU pipeline latency of VGG16 ($859.68\text{ ms}$) satisfies the operational requirement for interactive visual inspection ($< 1.0\text{ s}$).
4. **Dataset Sample Size**:
   - With an 869-image dataset (623 training images), VGG16's frozen feature extraction followed by selective fine-tuning of Block 5 achieved robust representation learning without over-indexing.

---

## 4. Confidence & Safety Policy

SolarSentinel AI adopts an honest, non-certified advisory framework:
- **Confidence Rating**:
  - `HIGH`: $\text{Confidence} \ge 0.85$ — Strong classification alignment with learned feature representations.
  - `MODERATE`: $0.60 \le \text{Confidence} < 0.85$ — Acceptable classification; user is notified that secondary symptoms may be present.
  - `LOW`: $\text{Confidence} < 0.60$ — Cautionary alert displayed in UI; recommendation mandates physical or thermographic verification before field interventions.
- **Safety Disclaimer**:
  - SolarSentinel AI provides decision support and exploratory triage. It is not certified for autonomous industrial commissioning or safety-critical electrical disconnection without technician review.
