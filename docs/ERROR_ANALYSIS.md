# SolarSentinel AI - Misclassification & Error Analysis Report (TEST Set)

- **Total Samples:** 131
- **Total Errors:** 19 (Error Rate: 14.50%)

## Top Confusion Modalities

| True Category | Predicted Category | Error Count | Share of Total Errors | Primary Root Cause |
| :--- | :--- | :---: | :---: | :--- |
| **Dusty** | **Clean** | 4 | 21.05% | Fine uniform dust layer lacks high-frequency edges; high glare mimics clean glass reflectance. |
| **Bird-drop** | **Dusty** | 3 | 15.79% | Dried diffuse organic matter visually resembles localized dust crusts. |
| **Bird-drop** | **Electrical-damage** | 2 | 10.53% | Unknown / Feature Ambiguity |
| **Dusty** | **Physical-Damage** | 2 | 10.53% | Unknown / Feature Ambiguity |
| **Electrical-damage** | **Clean** | 2 | 10.53% | Subtle busbar discoloration or micro-cracks lack macroscopic contrast at standard input resolution. |
| **Bird-drop** | **Physical-Damage** | 1 | 5.26% | Unknown / Feature Ambiguity |
| **Clean** | **Dusty** | 1 | 5.26% | Reflected environmental shadows or non-uniform daylight gradient mistaken for dust accumulation. |
| **Dusty** | **Bird-drop** | 1 | 5.26% | Unknown / Feature Ambiguity |
| **Physical-Damage** | **Snow-Covered** | 1 | 5.26% | Unknown / Feature Ambiguity |
| **Physical-Damage** | **Bird-drop** | 1 | 5.26% | Localized shattered glass impact craters share irregular geometric contours with splattered bird droppings. |
| **Snow-Covered** | **Dusty** | 1 | 5.26% | Unknown / Feature Ambiguity |

## Scientific Assessment of Error Sources

1. **Photometric Glare vs. Uniform Dust (Clean <-> Dusty):**
   - Optical reflectance from specular direct sunlight can wash out high-frequency texture features, causing light dust layers to appear reflective like clean glass.
2. **Minority Class Sparsity (Physical-Damage):**
   - Physical damage accounts for only 7.7% of the training partition (48 images). This low sample count bounds the representation of rare fracture topologies.
3. **Morphological Mimicry (Bird-drop vs. Cracks):**
   - Droppings that dry into branched or jagged outlines share high-frequency spatial gradients with physical glass fractures.

## Concrete Mitigation Strategies

- **Resolution Scaling:** Tiled inference preserving local 224x224 patches from native high-res captures.
- **Multi-Spectral / Thermal Radiometry:** Complementing optical RGB with thermal infrared inspection to unambiguously distinguish electrical hot spots.
- **Minority Class Oversampling:** Weighted cross-entropy or focal loss to elevate physical damage recall.