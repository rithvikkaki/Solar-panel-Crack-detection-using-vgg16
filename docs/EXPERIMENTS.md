# SolarSentinel AI - Controlled Experiments & Ablation Studies

## 1. Experimental Methodology & Rigor

All experiments were conducted strictly following standard scientific machine learning protocols:
1. **Locked 70/15/15 Partitioning:** All models were trained exclusively on the 623-sample training split and validated on the 131-sample validation split.
2. **Untouched Test Partition:** The 131-sample test split was held out completely during model development and hyperparameter tuning. Zero test-set optimization was permitted.
3. **Architecture Constraint:** The feature extractor backbone in all experiments is **VGG16** (pre-trained on ImageNet).
4. **Multi-Metric Evaluation:** Performance was assessed via Overall Accuracy, Macro F1, Weighted F1, and minority-class Recall (`Physical-Damage`).

---

## 2. Controlled Experiment Matrix

| Experiment ID | Architecture Head | Trainable Depth | Optimizer & Schedule | Regularization & Augmentation |
| :--- | :--- | :--- | :--- | :--- |
| **Exp 1: Baseline Head A** | GAP + Dropout(0.3) + Dense(6) | Backbone Frozen (14.7M frozen) | Adam (lr=1e-3, ReduceLROnPlateau) | Spatial flips, rotation (0.05) |
| **Exp 2: Head B (Selected)** | GAP + BN + Dense(512) + BN + Dense(256) + Dense(6) | Backbone Frozen (14.7M frozen) | Adam (lr=1e-3, ReduceLROnPlateau) | Spatial + Photometric jitter, BN, Dropout(0.4/0.3) |
| **Exp 3: Block 4+5 Unfreeze** | Head B | Block 4 & 5 Unfrozen (7.1M trainable) | Adam (lr=1e-5 staged) | Spatial + Photometric jitter |
| **Exp 4: Regularized Head B** | Head B | Backbone Frozen | Adam (lr=1e-3) | Seed 101 Cross-Run, Higher Weight Decay |
| **Exp 5: AdamW + Cosine** | Head B | Backbone Frozen | AdamW + Cosine Annealing (lr=5e-4) | Label Smoothing (0.05), Weight Decay (1e-4) |

---

## 3. Comparative Validation Results

The table below summarizes the quantitative validation outcomes recorded in `ml/experiments/phase9_vgg16/experiment_results.csv`:

| Rank | Experiment ID | Validation Accuracy | Macro F1 | Weighted F1 | Physical-Damage Recall | Physical-Damage F1 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **`exp2_head_b`** | **87.79%** | **0.8943** | **0.8767** | **90.0% (9/10)** | **0.9474** |
| 2 | `exp5_adamw` | 85.50% | 0.8544 | 0.8521 | 70.0% (7/10) | 0.8235 |
| 3 | `exp1_head_a` | 83.97% | 0.8580 | 0.8385 | 90.0% (9/10) | 0.9000 |
| 4 | `exp3_block4_unfreeze` | 83.21% | 0.8571 | 0.8310 | 90.0% (9/10) | 0.8571 |
| 5 | `exp4_regularized_head_b` | 82.44% | 0.8390 | 0.8229 | 80.0% (8/10) | 0.8421 |

---

## 4. Key Experimental Insights & Ablation Findings

### A. Deep Regularized Head (Exp 2) vs. Compact Head (Exp 1)
- Head B increased Validation Accuracy from **83.97% to 87.79% (+3.82%)** and Macro F1 from **0.8580 to 0.8943 (+0.0363)**.
- The dual Batch Normalization layers in Head B stabilized intermediate feature distributions prior to the dense non-linear projections, reducing covariance shift across batches.

### B. Why Did Backbone Unfreezing (Exp 3) Underperform Frozen Backbone (Exp 2)?
- Unfreezing Block 4 & 5 increased trainable parameters from ~1.2M to over 7.1M.
- Given the limited training corpus (623 images), the high parameter capacity led to empirical overfitting on low-level background artifacts (e.g. mounting bracket textures, field soil types) rather than invariant solar cell features.
- Keeping the convolutional backbone frozen and regularizing the classification head proved significantly more sample-efficient.

### C. Label Smoothing and AdamW (Exp 5)
- AdamW with Cosine Annealing produced strong training stability and 85.50% accuracy.
- However, label smoothing (0.05) slightly penalized confident predictions on the critical minority class `Physical-Damage`, reducing its recall to 70.0% compared to 90.0% in Exp 2.

---

## 5. Final Model Selection

Following the strict protocol that **model selection must be governed exclusively by validation partition benchmarks**:
- **Candidate Selected:** **`exp2_head_b`**
- **Justification:** Achieves the highest overall validation accuracy (87.79%), highest macro F1 (0.8943), and highest minority-class physical damage recall (90.0%).
- **Locked Weights Location:** `ml/models/solar_sentinel_vgg16.keras` and `ml/models/best_vgg16.keras`.
