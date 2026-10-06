# SolarSentinel AI - Dataset Analysis & Integrity Audit

## 1. Executive Summary

This document details the exploratory data analysis, dataset composition, class imbalance characteristics, and rigorous integrity auditing conducted on the **Solar Panel Defect Classification Dataset**.

The dataset consists of **885 high-resolution optical images** categorized into **6 distinct operational conditions**. All image paths and labels were verified through programmatic ingestion and SHA-256 content hashing.

---

## 2. Dataset Distribution & Imbalance

The 885 images are distributed across six verified classes:

| Class Index | Category | Sample Count | Class Share (%) | Imbalance Ratio (vs Majority) |
| :---: | :--- | :---: | :---: | :---: |
| 0 | `Bird-drop` | 207 | 23.39% | 1.00:1 (Baseline) |
| 1 | `Clean` | 193 | 21.81% | 1.07:1 |
| 2 | `Dusty` | 190 | 21.47% | 1.09:1 |
| 3 | `Electrical-damage` | 103 | 11.64% | 2.01:1 |
| 4 | `Physical-Damage` | 69 | 7.80% | **3.00:1 (Critical Minority)** |
| 5 | `Snow-Covered` | 123 | 13.90% | 1.68:1 |
| **Total** | | **885** | **100.0%** | |

### Physical-Damage Critical Minority Analysis
`Physical-Damage` constitutes only 7.80% of the entire dataset. In our 70/15/15 split:
- **Training partition:** 48 images (out of 623)
- **Validation partition:** 10 images (out of 131)
- **Test partition:** 10 images (out of 131)

Because each test instance in `Physical-Damage` represents exactly **10.0%** of the category's recall, standard accuracy can be misleading. Therefore, model selection requires monitoring **Macro F1** and **Physical-Damage Recall** alongside overall accuracy.

---

## 3. Image Properties & Formats

- **Total Scanned Images:** 885
- **Corrupted / Unreadable Images:** 0 (100% integrity)
- **Color Format:** 3-channel RGB (all images converted and verified)
- **Primary Resolutions:**
  - `4000x3000`: 312 images (35.25%)
  - `4608x3456`: 208 images (23.50%)
  - `1920x1080`: 144 images (16.27%)
  - `3024x4032`: 86 images (9.72%)
  - Other: 135 images (15.26%)

---

## 4. Byte-Level Duplicate Detection (SHA-256)

Programmatic scanning using SHA-256 cryptographic hashing (`ml/data/duplicate_detection.py`) revealed:
- **Unique Content Hashes:** 794
- **Duplicate Hash Groups:** 74 groups
- **Total Redundant Images:** 91 files
  - Intra-class duplicates: 72 groups (identical images stored under multiple file names in the same class)
  - Cross-class duplicates: 2 groups (ambiguous samples filed across classes)

### Data Leakage Guarantee
A critical vulnerability in naive random splitting is that identical or near-identical images may fall into both the training and test sets, artificially inflating test metrics.

Our split generation protocol (`ml/data/create_split.py`):
1. Clusters all identical SHA-256 hash instances into single atomic units.
2. Allocates each cluster entirely to either Train, Validation, or Test.
3. Formally verifies partition isolation.

**Verification Result:**
- **Split Leakage Instances:** **0 (Zero partition leakage)**
- Every sample in the 131-image locked test set is strictly absent from the training and validation sets.

---

## 5. Stratified 70/15/15 Partition Protocol

The dataset is partitioned deterministically using Random Seed `42`:

| Class Name | Total Samples | Train (70%) | Validation (15%) | Test (15% LOCKED) |
| :--- | :---: | :---: | :---: | :---: |
| `Bird-drop` | 207 | 147 | 31 | 31 |
| `Clean` | 193 | 135 | 29 | 29 |
| `Dusty` | 190 | 133 | 28 | 28 |
| `Electrical-damage` | 103 | 73 | 15 | 15 |
| `Physical-Damage` | 69 | 48 | 10 | 10 |
| `Snow-Covered` | 123 | 87 | 18 | 18 |
| **Total** | **885** | **623** | **131** | **131** |

The split manifest is saved at `ml/metadata/split_manifest_70_15_15.json` and locked.
