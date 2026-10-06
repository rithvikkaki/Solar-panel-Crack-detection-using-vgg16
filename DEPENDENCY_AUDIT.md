# SolarSentinel AI - Dependency Audit Report

## 1. Overview
This document records the exact dependency audit conducted during Phase 6 across all repository component layers.

---

## 2. Environment Specifications
- Python Version: 3.10.15 (64-bit Windows)
- TensorFlow: 2.21.0
- Keras: 3.12.4
- FastAPI: 0.115.0+
- Pydantic: 2.9.2
- Node.js (for frontend): Node 20 / npm 10+
- Vite: 8.3.0
- React: 19.2.8

---

## 3. Dependency File Inventory & Status

| File | Purpose | Audit Findings | Resolution |
|---|---|---|---|
| 
equirements.txt | Root / Docker build dependency manifest | Missing at root; caused Docker builds referencing root requirements.txt to fail | Created unified root requirements.txt uniting backend and ML requirements |
| ackend/requirements.txt | FastAPI backend dependencies | Clean; specified fastapi, uvicorn, pydantic-settings, pillow, opencv-python, numpy | Preserved for standalone backend venvs |
| ml/requirements.txt | Training, evaluation, and explainability dependencies | Clean; specifies tensorflow-cpu on Windows and tensorflow on Linux, with keras>=3.0.0 | Preserved for ML development |
| rontend/package.json | React frontend SPA dependencies | Clean; React 19, Vite 8, Tailwind CSS v4, Lucide icons | Fully resolved and builds cleanly |

---

## 4. Platform-Specific Constraints
- TensorFlow on Windows: TensorFlow >= 2.11 does not provide native GPU support on Windows without WSL2 or DirectML. Standard CPU instructions (AVX, AVX2, FMA) are utilized.
- Keras 3 Submodel API: Keras 3 replaces older Keras 2 graph internals. Grad-CAM handles nested submodels by executing the backbone and classification head layers sequentially through tf.GradientTape().
- FastAPI Lifespan Compatibility: TestClient uses ASGI lifespan hooks to ensure singleton ModelService weights are loaded once before request execution.