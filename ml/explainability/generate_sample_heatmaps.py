"""
SolarSentinel AI - Grad-CAM Visual Artifact Generator
Selects representative sample images from validation/test sets
and renders high-resolution Grad-CAM overlays targeting VGG16 block5_conv3.
Saves outputs to ml/experiments/phase9_vgg16/gradcam/.
"""
import os
import sys
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.explainability.gradcam import GradCAMExplainer, load_image_as_rgb
from backend.app.services.model_service import ModelService

MANIFEST_PATH = os.path.join(PROJECT_ROOT, "ml", "metadata", "split_manifest_70_15_15.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "ml", "experiments", "phase9_vgg16", "gradcam")


def generate_gradcam_artifacts():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Initializing Model Service and Grad-CAM Explainer...")
    service = ModelService.get_instance()
    loaded = service.load_model()
    if not loaded:
        raise RuntimeError("Failed to load model weights in ModelService.")
    explainer = service.explainer

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    val_items = manifest.get("val") or manifest.get("val_files", [])
    classes = manifest.get("classes", [])

    # Pick 1 sample per class
    class_samples = {}
    for item in val_items:
        cls = item.get("class")
        if cls not in class_samples:
            class_samples[cls] = item

    print(f"Generating Grad-CAM visualizations for {len(class_samples)} classes...")

    for cls, item in class_samples.items():
        rel_p = item.get("path")
        abs_p = os.path.join(PROJECT_ROOT, rel_p) if not os.path.isabs(rel_p) else rel_p

        # Open PIL Image
        pil_img = Image.open(abs_p).convert("RGB")

        # Run inference
        pred_dict = service.predict(pil_img)
        pred_cls = pred_dict.get("predicted_condition") or pred_dict.get("predicted_class")
        conf = pred_dict["confidence"]

        # Run Grad-CAM
        result = explainer.explain(
            image_source=pil_img,
            overlay_alpha=0.5
        )

        # Render 3-panel comparison figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=200)

        axes[0].imshow(result.original_image)
        axes[0].set_title(f"Original Input\nTrue: {cls}", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(result.colorized_heatmap)
        axes[1].set_title(f"Grad-CAM Heatmap (Jet)\nTarget: {result.target_layer_name}", fontsize=11, fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(result.overlay)
        axes[2].set_title(f"Activation Overlay (alpha=0.5)\nPred: {pred_cls} ({conf*100:.1f}%)", fontsize=11, fontweight="bold")
        axes[2].axis("off")

        plt.suptitle(f"SolarSentinel Explainable AI - Class: {cls}", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()

        safe_cls = cls.replace(" ", "_").lower()
        out_fname = f"gradcam_{safe_cls}.png"
        out_fpath = os.path.join(OUTPUT_DIR, out_fname)
        plt.savefig(out_fpath, bbox_inches="tight", dpi=200)
        plt.close()
        print(f"  [+] Saved {out_fname} (Pred: {pred_cls}, Conf: {conf*100:.1f}%)")

    print(f"\nAll Grad-CAM visualizations saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    generate_gradcam_artifacts()
