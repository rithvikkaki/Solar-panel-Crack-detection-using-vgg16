# SolarSentinel AI - Dataset Repository Specification

This directory holds dataset assets for the **SolarSentinel AI: Solar Panel Health & Fault Intelligence Platform**.

---

## Verified Dataset Specifications (from Project Audit)

Based on the original project audit (`PROJECT_ANALYSIS.md`), the baseline dataset contains **885 images** categorized across **6 classes**:

| Class Directory Name | Visual Condition | Original Split Distribution (approx.) |
|---|---|---|
| `Bird-drop` | Localized droppings and acidic residue | ~150 images |
| `Clean` | Unobstructed, undamaged solar panels | ~150 images |
| `Dusty` | Soil, sand, and particulate accumulation | ~150 images |
| `Electrical-damage` | Hotspot discoloration, burn marks, cell breakdown | ~140 images |
| `Physical-Damage` | Surface cracks, cell fractures, hail damage | ~150 images |
| `Snow-Covered` | Partial or complete snow occlusion | ~145 images |

---

## Directory Organization

```text
data/
├── raw/
│   ├── Bird-drop/
│   ├── Clean/
│   ├── Dusty/
│   ├── Electrical-damage/
│   ├── Physical-Damage/
│   └── Snow-Covered/
│
├── processed/
│   ├── train/
│   └── val/
│
└── README.md
```

---

## Getting Started

1. Place your uncompressed image folders into `data/raw/`.
2. Ensure folder names match the verified classes exactly (case-sensitive).
3. Supported image formats: `.jpg`, `.jpeg`, `.png`, `.webp`.
4. Run the dataset validation utility before initiating training:
   ```bash
   python ml/training/validate_dataset.py --data_dir data/raw
   ```
5. If validation passes, run the training pipeline:
   ```bash
   python ml/training/train.py --data_dir data/raw --input_size 244
   ```
