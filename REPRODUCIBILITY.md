# SolarSentinel AI - Reproducibility Guide

This guide describes the exact steps required to replicate data partitioning, training, model evaluation, inference testing, and container deployment.`

---

## 1. Environment Setup

``ash
# 1. Create Python virtual environment (Python 3.10+)
python -m venv .venv

# 2. Activate environment
# Windows PowerShell:
.venv\\Scripts\\activate
# Linux / macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
``

---

## 2. Dataset Validation & Hash-Stratified Split Manifest

The dataset audit accounts for all 885 physical image files across 6 classes.
To guarantee zero data leakage between splits, an SHA-256 hash-grouped 70/15/15 stratification protocol was applied:

- **Manifest Path**: ml/metadata/split_manifest_70_15_15.json
- **Audit File**: ml/metadata/split_audit.json
- **Random Seed**: 42
- **Training Samples**: 623 images (553 unique hashes)
- **Validation Samples**: 131 images (124 unique hashes)
- **Test Samples (Locked)**: 131 images (117 unique hashes)
- **Data Leakage**: Cryptographically verified 0 path and 0 hash overlap between partitions.`

---

## 3. Systematic Model Optimization & Training

The production model utilizes a two-stage transfer learning pipeline with a customized classification head (**Head B**):

``ash
# Execute Head B photometric training experiment
python ml/training/experiment_runner.py --exp_name exp2_head_b --head head_b --aug photometric --epochs1 6 --epochs2 8
`

- **Stage 1 (Feature Extraction)**: VGG16 backbone frozen; Head B trained with Adam (1e-3) and class weights.
- **Stage 2 (Targeted Fine-Tuning)**: Block 5 (lock5_conv1, lock5_conv2, lock5_conv3) unfrozen; trained with Adam (1e-4), ReduceLROnPlateau, and EarlyStopping.
- **Output Weights**: Saved to ml/models/solar_sentinel_vgg16.keras.`

---

## 4. Locked Test Set Evaluation

``ash
# Evaluate winning model on locked held-out test partition
python ml/training/experiment_runner.py --exp_name test_eval_run
`

### Verified Performance Metrics (131 Locked Test Images):
- **Accuracy**: **85.50%**
- **Macro F1 Score**: **0.8503**
- **Weighted F1 Score**: **0.8542**
- **Physical-Damage Recall**: **80.00%** (8/10)
- **Physical-Damage F1 Score**: **0.7619**`

---

## 5. Hardware Latency Benchmarks (CPU)

``ash
python ml/training/benchmark_production_model.py --iterations 30
`

- **Inference Latency (Mean)**: 125.65 ms
- **Grad-CAM Latency (Mean)**: 494.22 ms
- **Total Pipeline Latency (Mean)**: 619.87 ms (27.9% speedup over baseline)
- **Results Persisted**: ml/metadata/performance_benchmark.json`

---

## 6. Running the Automated Test Suite

``ash
# Backend API, validation, and policy tests (23 passed)
python -m pytest backend/tests/ -v

# Grad-CAM explainability and layer discovery tests (7 passed)
python -m pytest ml/explainability/tests/ -v

# Frontend TypeScript compilation & Vite bundling
cd frontend
npm run build
``

---

## 7. Running the Application Locally

### Backend Service:
``ash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
`
- API Docs: http://127.0.0.1:8000/docs
- Health Probe: http://127.0.0.1:8000/api/v1/health
- Model Metrics: http://127.0.0.1:8000/api/v1/model/metrics

### Frontend Application:
``ash
cd frontend
npm install
npm run dev
`
- Dashboard: http://localhost:5173`

---
