# SolarSentinel AI — Reproducibility & Scientific Verification Guide

This document provides exact, deterministic instructions to verify datasets, audit data splits, reproduce model training, run performance benchmarks, and launch services.

---

## 1. Environment Requirements

- **Python**: 3.10+ (Verified on Python 3.10.11 / 3.12)
- **Node.js**: 18+ or 20+ (Verified on Node 20.x, npm 10.x)
- **TensorFlow**: 2.16+ / 2.21+ with Keras 3.x
- **OS**: Linux, macOS, or Windows 10/11 (PowerShell / Bash)

### Installation
```bash
# Backend virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r backend/requirements.txt
pip install pytest pytest-cov Pillow

# Frontend dependencies
cd frontend
npm ci
cd ..
```

---

## 2. Dataset Verification & Hash-Stratified Split Audit

The dataset consists of 869 images across 6 classes. An exact audit identified duplicate images across classes. A cryptographic SHA-256 hash-stratified split (70% train, 15% val, 15% test) was generated to ensure zero data leakage.

### Run Split Audit:
```bash
python ml/src/audit_dataset.py
```
**Verification Invariant**:
- Exact duplicates grouped by hash: `ml/metadata/split_audit.json`
- `path_overlap_count`: `0`
- `hash_leakage_groups_count`: `0`
- `data_leakage_detected`: `false`

Split counts:
- **Train**: 623 images
- **Val**: 131 images
- **Held-out Test**: 131 images

---

## 3. Model Training & Checkpointing

The primary model is a two-stage fine-tuned VGG16 network:
1. **Stage 1**: Frozen ImageNet feature extractor with classification head (`GlobalAveragePooling2D` -> `Dense(256, relu)` -> `Dropout(0.4)` -> `Dense(6, softmax)`).
2. **Stage 2**: Fine-tuning top convolutional block (`block5_conv1`, `block5_conv2`, `block5_conv3`) at low learning rate ($10^{-5}$).

```bash
python ml/src/train_vgg16.py
```
Artifact generated:
- `ml/models/solar_sentinel_vgg16.keras`

---

## 4. Test Set Evaluation & Latency Benchmarks

Evaluate the trained checkpoint against the untouched 131-sample test split:
```bash
python ml/training/evaluate.py
```
Output:
- `ml/metadata/test_set_evaluation.json`:
  - Test Accuracy: **82.44%**
  - Macro F1: **0.8378**
  - Weighted F1: **0.8235**
  - Physical-Damage Recall: **80.00%** (8/10)
  - Physical-Damage F1: **0.8000**

Run reproducible hardware latency benchmarks (30 timed iterations + 5 warmup passes on CPU):
```bash
python ml/training/benchmark_production_model.py --iterations 30 --warmup 5
```
Output:
- `ml/metadata/performance_benchmark.json`:
  - Forward Inference Latency: **209.1 ms** (p50: 207.3 ms)
  - Grad-CAM Attribution Latency: **650.58 ms** (p50: 649.0 ms)
  - Total Pipeline Latency: **859.68 ms** (p50: 860.5 ms)
  - *Notice: Latency measurements are hardware-dependent and reflect host CPU capabilities.*

---

## 5. Automated Testing Suite

Execute backend integration tests covering inference, Grad-CAM generation, schema validation, confidence policy, and demo mode:
```bash
python -m pytest backend/tests/ -v
```

Execute frontend build verification:
```bash
cd frontend
npm run build
cd ..
```

---
