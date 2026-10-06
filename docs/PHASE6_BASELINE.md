# SolarSentinel AI - Phase 6 Baseline Model Audit

## 1. Executive Summary
This document establishes the verified baseline for SolarSentinel AI prior to Phase 6 improvements. All metrics represent the current production VGG16 model (ml/models/solar_sentinel_vgg16.keras) evaluated on the deterministic held-out validation set.

## 2. Current Model Architecture
- Backbone: VGG16 (ImageNet pre-trained weights)
- Input Dimensions: (244, 244, 3) (Notebook baseline preserved)
- Top Head Architecture:
  - InputLayer((244, 244, 3))
  - RandomFlip, RandomRotation(0.04), RandomZoom(0.05) (Training only)
  - vgg16.preprocess_input (BGR zero-centering, ImageNet mean subtraction)
  - VGG16 backbone (layers 0-13 frozen in Stage 1; Block 5 unfrozen in Stage 2)
  - GlobalAveragePooling2D
  - Dropout(0.3)
  - Dense(6, activation='softmax')
- Output Classes: 6 (Bird-drop, Clean, Dusty, Electrical-damage, Physical-Damage, Snow-Covered)
- Total Parameters: 14,847,942 (56.6 MB weights / 115.6 MB .keras archive)

## 3. Dataset Size and Class Distribution
- Dataset Root: dataset/Faulty_solar_panel
- Total Valid Images Directly Located in Class Folders: 869 images
- Class Breakdown:
  - Clean: 193 (22.2%)
  - Bird-drop: 191 (22.0%)
  - Dusty: 190 (21.9%)
  - Snow-Covered: 123 (14.2%)
  - Electrical-damage: 103 (11.9%)
  - Physical-Damage: 69 (7.9%)
- Imbalance Ratio: 2.8:1 (Clean vs Physical-Damage)

## 4. Current Preprocessing and Validation Methodology
- Preprocessing Pipeline: PIL RGB -> Bilinear resize to (244, 244) -> BGR conversion -> ImageNet BGR mean subtraction [103.939, 116.779, 123.68]
- Validation Methodology: Stratified split_manifest.json (Seed 42), 707 train (80%), 178 val (20%), zero overlap.

## 5. Baseline Performance Metrics

| Metric | Score | Sample Count |
|---|---|---|
| Overall Accuracy | 85.39% | 178 |
| Weighted F1 Score | 0.8496 | 178 |
| Macro F1 Score | 0.8440 | 178 |
| Weighted Precision | 0.8666 | 178 |
| Weighted Recall | 0.8539 | 178 |

### Per-Class Baseline Performance

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| Bird-drop | 0.7959 | 0.9512 | 0.8667 | 41 |
| Clean | 0.7708 | 0.9487 | 0.8506 | 39 |
| Dusty | 0.8966 | 0.6842 | 0.7761 | 38 |
| Electrical-damage | 1.0000 | 0.8095 | 0.8947 | 21 |
| Physical-Damage | 0.8889 | 0.5714 | 0.6957 | 14 |
| Snow-Covered | 0.9615 | 1.0000 | 0.9804 | 25 |

## 6. Known Weaknesses and Investigation Target
- Physical-Damage Class Deficit:
  - Lowest support in dataset (69 images total, 14 in validation).
  - Recall is only 57.14% (lowest of all classes).
  - F1 score is 0.6957 (lowest of all classes).
  - Primary error mode: 5 out of 14 validation samples misclassified as Bird-drop.
- Target for Phase 6:
  - Investigate whether class weighting and data augmentation can improve Physical-Damage recall without degrading overall accuracy.
  - Benchmark modern lightweight architectures (MobileNetV2, EfficientNetB0) against VGG16 on a rigorous 70/15/15 train/val/test split.
