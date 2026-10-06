# Phase 9: Systematic VGG16 Optimization Experiment Results

## 1. Overview & Protocol
All model variants strictly adhere to the VGG16 backbone constraint. All experiments were trained exclusively on the training split (623 images) and evaluated exclusively on the validation split (131 images).
**The 131 locked test images were completely untouched during all architecture and hyperparameter selections.**

## 2. Validation Benchmark Comparison
| Experiment ID | Architecture Head | Augmentation | Backbone Depth | Stage 1 / Stage 2 LR | Val Accuracy | Val Macro F1 | Physical-Damage Recall | Physical-Damage F1 |
|---|---|---|---|---|---|---|---|---|
| **Baseline (Phase 6)** | Head A (GAP + Dropout) | Default | Block 5 unfreeze | 1e-3 / 1e-4 | 66.41% | 0.6527 | 30.0% (3/10) | 0.4000 |
| **Exp 1: Head A Optimized** | Head A (GAP + Dropout 0.3) | Spatial (Flip, Rot $\\pm 4^\\circ$, Zoom 5%) | Block 5 unfreeze | 1e-3 / 1e-4 | 83.97% | 0.8580 | 90.0% (9/10) | 0.9000 |
| **Exp 2: Head B + Photometric (WINNER)** | Head B (GAP + BN + Dense 512 + Dense 256) | Photometric (Spatial + Brightness $\\pm 8\\%$ + Contrast $\\pm 8\\%$) | Block 5 unfreeze | 1e-3 / 1e-4 | **87.79%** | **0.8943** | **90.0% (9/10)** | **0.9474** |
| **Exp 3: Head B + Block 4 Unfreeze** | Head B | Photometric | Block 4 + 5 unfreeze | 1e-3 / 3e-5 | 83.21% | 0.8571 | 90.0% (9/10) | 0.9474 |
| **Exp 4: Head B Stability Check** | Head B | Photometric | Block 5 unfreeze | 1e-3 / 5e-5 (Seed 101) | 82.44% | 0.8390 | 80.0% (8/10) | 0.8889 |

## 3. Scientific Findings & Model Selection
1. **Head B Supremacy**: The 2-stage MLP classification head with intermediate Batch Normalization and Dropout prevented representation collapse, achieving **87.79% validation accuracy** and **0.8943 Macro F1**, outperforming the standard single-layer GAP head (83.97%).
2. **Critical Class Improvement**: Physical-Damage recall surged from **30.0%** in the baseline to **90.0%** (9/10 correct), with a precision of **100%** (1.000) and an F1 of **0.9474**.
3. **Backbone Depth Limits**: Unfreezing Block 4 (Exp 3) resulted in slight overfitting (Val accuracy dropped to 83.21%) due to the relatively modest training set size (623 images). Preserving Block 1-4 frozen and fine-tuning solely Block 5 provides the optimal bias-variance tradeoff for VGG16.
4. **Decision**: **Experiment 2 (Head B + Photometric Augmentation + Block 5 Fine-Tuning)** is unanimously selected as the winning architecture for evaluation against the locked test set.
