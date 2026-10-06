# Phase 9: Systematic Dataset & Integrity Audit

## 1. Scope & Objective
A full forensic audit was executed across all physical image files in the repository to guarantee absolute scientific integrity, assess resolution distributions, verify data cleanliness, and prove zero leakage across training, validation, and test partitions.

## 2. Dataset Overview & File Verification
- **Total Physical Image Files Audited**: 885 images
- **Corrupted / Unreadable Files**: 0 (all images opened and parsed via PIL)
- **Unique SHA-256 Hashes**: 794
- **Exact Duplicate Groups (Identical Content)**: 74 groups (91 duplicate files)
- **Color Channels**: All 885 images verified standard RGB (3 channels).

### Resolution & Aspect Ratio Profile
The dataset exhibits significant variance in resolution and aspect ratio, ranging from mobile camera captures (474x355) up to high-resolution drone/DSLR photography (2048x1536).
Top resolution groupings:
1. 720x960 (portrait): 58 images
2. 2048x1152 (16:9): 23 images
3. 1536x2048 (portrait): 22 images
4. 474x355 (4:3): 22 images
5. 2048x1536 (4:3): 22 images

## 3. Split Isolation & Leakage Verification
The dataset is split according to the hash-grouped stratified manifest (ml/metadata/split_manifest_70_15_15.json):

| Partition | Total Images | Unique Hashes | % of Dataset |
|---|---|---|---|
| **Train** | 623 | 553 | 70.4% |
| **Validation** | 131 | 124 | 14.8% |
| **Test (Locked)** | 131 | 117 | 14.8% |
| **Total** | **885** | **794** | **100%** |

### Leakage Audit Results:
\\text{Train Hashes} \\cap \\text{Val Hashes} = 0
\\text{Train Hashes} \\cap \\text{Test Hashes} = 0
\\text{Val Hashes} \\cap \\text{Test Hashes} = 0
- **Path Overlap**: 0 files
- **Content Hash Overlap**: 0 files
- **Data Leakage Detected**: **NONE** (Zero data leakage mathematically proven).

## 4. Class Imbalance Analysis
| Class Index | Class Name | Train Count | Val Count | Test Count | Total |
|---|---|---|---|---|---|
| 0 | Bird-drop | 147 | 31 | 31 | 209 |
| 1 | Clean | 135 | 29 | 29 | 193 |
| 2 | Dusty | 133 | 28 | 28 | 189 |
| 3 | Electrical-damage | 73 | 15 | 15 | 103 |
| 4 | Physical-Damage | 48 | 10 | 10 | 68 |
| 5 | Snow-Covered | 87 | 18 | 18 | 123 |
| **Total** | | **623** | **131** | **131** | **885** |

### Key Observations:
1. **Severe Imbalance in Physical-Damage**: The minority class Physical-Damage has only 48 training images (7.7% of train data), while Bird-drop has 147 images (23.6% of train data). Imbalance ratio: **3.06 : 1**.
2. **Support Constraint**: The validation and test sets each have exactly 10 samples of Physical-Damage. A single misclassified test sample causes a 10% swing in class recall (0.80 -> 0.90 or 0.70).
3. **Mitigation Strategy**: Training optimizations must incorporate inverse class weights or focal loss to prevent the gradient descent updates from neglecting the critical safety-related damage classes (Physical-Damage and Electrical-damage).
