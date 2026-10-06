# SolarSentinel AI - Phase 6 Final Engineering Audit Report

## 1. Existing Architecture Summary
SolarSentinel AI is an explainable deep learning platform for photovoltaic solar panel condition monitoring.
- **Backbone & Model Architecture:** Fine-tuned VGG16 with custom classification head (GlobalAveragePooling2D -> Dense(256) -> Dropout(0.3) -> Dense(6, softmax)).
- **Explainability:** Grad-CAM operating on lock5_conv3 with OpenCV colorization and overlay rendering.
- **Backend:** FastAPI with singleton lifespan model loading, typed Pydantic validation, CORS, request tracing, and health/readiness endpoints.
- **Frontend:** React 19 + TypeScript + Vite 8 SPA communicating with backend via REST and rendering condition diagnostics and Grad-CAM layers.

---

## 2. Files Inspected
- ml/training/train.py, ml/training/evaluate.py, ml/training/validate_dataset.py, ml/training/config.py
- ml/inference/preprocessing.py, ml/inference/predict.py
- ml/explainability/gradcam.py, ml/explainability/tests/test_gradcam.py
- ml/metadata/class_names.json, ml/metadata/evaluation_metrics.json, ml/metadata/split_manifest.json
- ml/models/solar_sentinel_vgg16.keras
- dataset/Faulty_solar_panel/
- ackend/app/main.py, ackend/app/services/model_service.py, ackend/app/services/inspection_service.py
- ackend/app/api/v1/health.py, ackend/app/api/v1/inspect.py, ackend/app/api/v1/demo.py
- ackend/tests/test_api.py
- rontend/package.json, rontend/src/api/client.ts, rontend/src/types/inspection.ts
- Dockerfile.backend, Dockerfile.frontend, docker-compose.yml, .dockerignore
- 
equirements.txt, ackend/requirements.txt, ml/requirements.txt, .env.example
- README.md, MODEL_CONTRACT.md, GRADCAM_CONTRACT.md

---

## 3. Verified Components
- Real model weights exist and are valid (115.6 MB .keras file).
- Single canonical preprocessing pipeline in ml/inference/preprocessing.py (BGR mean subtraction without division by 255).
- Model inputs and outputs: (None, 244, 244, 3) -> (None, 6).
- 25 automated pytest tests passing across backend and explainability suites.
- Production frontend build (
pm run build) passing cleanly.
- Frontend Docker container build verified with Docker Engine 29.7.2.

---

## 4. Detected Inconsistencies & Potential Bugs
1. **Data Leakage in Evaluation Split (Critical):**
   In previous evaluation workflows, image_dataset_from_directory(shuffle=False) was used for the validation split while shuffle=True was used for training. In TensorFlow/Keras, shuffle=False takes the first 20% of files alphabetically per directory, whereas shuffle=True takes a random 20%. This resulted in a **143-sample overlap** between training and validation files.
   *Resolution:* Generated a deterministic, persisted split manifest (ml/metadata/split_manifest.json) using seed 42 and stratified sampling, verifying intersection(train_files, val_files) == 0.
2. **Missing Root requirements.txt:**
   Dockerfile.backend copied 
equirements.txt ., but only ackend/requirements.txt and ml/requirements.txt existed.
   *Resolution:* Created a unified root 
equirements.txt covering both runtime layers.
3. **Class Imbalance Impact on Physical-Damage:**
   Physical-Damage has only 69 total images (compared to ~190 in Clean/Dusty/Bird-drop), yielding 57.1% recall due to visual confusion with bird droppings and dirt spots.

---

## 5. Model Pipeline Risks & Mitigations
- **Risk:** Unmonitored retraining causing stochastic split changes.
- **Mitigation:** Training and evaluation now load the deterministic file list from split_manifest.json.

---

## 6. Exact Fixes Applied
1. Created ml/metadata/dataset_audit.json documenting 869 direct images and 885 total image files.
2. Created ml/metadata/split_manifest.json guaranteeing zero train/val overlap.
3. Created unified root 
equirements.txt.
4. Re-evaluated model on held-out 178 validation samples: Accuracy = **85.39%**, Macro F1 = **0.8440**, Weighted F1 = **0.8496**.
5. Created ml/metadata/evaluation_metrics_phase6.json, ml/metadata/confusion_matrix.csv, and ml/metadata/class_performance_report.json.
6. Created ml/metadata/inference_validation_phase6.json from real end-to-end API inspection tests.
7. Created DEPENDENCY_AUDIT.md and REPRODUCIBILITY.md.