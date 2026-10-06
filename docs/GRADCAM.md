# SolarSentinel AI - Explainable AI (Grad-CAM) Framework

## 1. Scientific Principles & Mandatory Definition

> [!IMPORTANT]
> **Scientific Language Mandate:**
> "Grad-CAM visualizes image regions that contributed strongly to the model's classification. It reflects neural network feature activations and does **not** constitute precise physical defect boundary detection, automated segmentation, or certified diagnostic certainty."

Gradient-weighted Class Activation Mapping (Grad-CAM) provides visual explanations for decisions made by convolutional neural networks, ensuring that solar panel defect predictions are grounded in relevant cell anomalies rather than background artifacts.

---

## 2. Mathematical Formulation

For a given defect class $c$, let $y^c$ represent the model's pre-softmax activation score (logit). Let $A^k$ represent the $k$-th feature activation map of the final convolutional layer (`block5_conv3` in VGG16), with spatial dimensions $i, j$.

### 1. Neuron Importance Weights ($\alpha_k^c$)
The importance weight $\alpha_k^c$ is computed by global average pooling the gradients of logit $y^c$ with respect to feature map activations $A_{i,j}^k$:
$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i, j}^k}$$
where $Z = H \times W$ is the spatial area of the convolutional feature map ($14 \times 14$ for $224 \times 224$ inputs).

### 2. Weighted Activation Combination & Rectification
A forward pass linear combination of feature maps is weighted by $\alpha_k^c$, followed by a Rectified Linear Unit (ReLU) to isolate features that have a positive influence on the target defect class:
$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_{k} \alpha_k^c A^k \right)$$

### 3. Bilinear Upsampling & Colormap Normalization
The coarse $14 \times 14$ activation map is normalized into $[0, 1]$ via min-max scaling and bilinearly upsampled to the native image resolution ($224 \times 224$). A `cv2.COLORMAP_JET` color transform is applied and blended with the original input at opacity $\alpha = 0.5$:
$$I_{\text{overlay}} = \alpha \cdot I_{\text{heatmap}} + (1 - \alpha) \cdot I_{\text{original}}$$

---

## 3. Class-Specific Activation Behavior

| Defect Class | Targeted Structural Features | Observed Activation Patterns |
| :--- | :--- | :--- |
| **`Physical-Damage`** | Glass fractures, cracked silicon wafers, shattered impact centers | **High localized focus** directly enveloping fracture fissures and spiderweb impact craters. |
| **`Bird-drop`** | Opaque organic deposits, calcified splatters | **Tight localized hotspots** matching splatter contours; distinguishes isolated droppings from pervasive surface coats. |
| **`Clean`** | Uniform reflective glass, regular busbar grid lines | **Diffuse, low-gradient response** distributed evenly across solar cells with no localized focal nodes. |
| **`Dusty`** | Fine particulate blankets, uniform optical attenuation | **Widespread, moderate-intensity activation** encompassing broad multi-cell quadrants. |
| **`Electrical-damage`** | Localized thermal discoloration, hotspot burns | **Concentrated hotspots** centered on scorched interconnects and discolored busbar junctions. |
| **`Snow-Covered`** | High-albedo occluding blankets | **Broad saturated activation** covering large snow piles and freeze boundaries. |

---

## 4. Visual Diagnostics: Correct Predictions vs. Ambiguities

High-resolution 3-panel visualization artifacts (Original Input, Colorized Heatmap, and Overlay) are stored under `ml/experiments/phase9_vgg16/gradcam/`:

1. **`gradcam_physical-damage.png`:**
   - Demonstrates decisive, focused activation along the diagonal fracture path across panel silicon cells.
2. **`gradcam_clean.png`:**
   - Demonstrates balanced, low-gradient activation across clean busbars, confirming the absence of anomalous features.
3. **`gradcam_dusty.png`:**
   - Demonstrates distributed attention across the dust-obscured cells.
4. **Diagnostic Misclassification (`gradcam_bird-drop.png`):**
   - High albedo from dried calcified white dropping caused strong localized reflectance resembling snow patches. Grad-CAM clearly reveals the network focused intensely on the bright white patch, explaining the prediction error.

---

## 5. Artifact Generation Tooling

To generate or refresh Grad-CAM artifacts for inspection samples, execute:
```bash
python ml/explainability/generate_sample_heatmaps.py
```
Outputs are written to `ml/experiments/phase9_vgg16/gradcam/`.
