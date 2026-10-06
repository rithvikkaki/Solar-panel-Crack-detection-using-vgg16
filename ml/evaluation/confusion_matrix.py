"""
SolarSentinel AI - Confusion Matrix Generator & Visualizer
Generates publication-quality, annotated confusion matrix figures
displaying both absolute counts and normalized percentages per class.
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CANONICAL_CLASSES = [
    "Bird-drop",
    "Clean",
    "Dusty",
    "Electrical-damage",
    "Physical-Damage",
    "Snow-Covered"
]


def plot_confusion_matrix(
    cm_matrix: list,
    classes: list = CANONICAL_CLASSES,
    output_path: str = "confusion_matrix.png",
    title: str = "SolarSentinel VGG16 Confusion Matrix"
) -> str:
    """
    Renders and saves a stylish, high-contrast confusion matrix.
    """
    cm = np.array(cm_matrix)
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    cax = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(cax)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9, fontweight="bold")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=9, fontweight="bold")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100
            txt_color = "white" if count > thresh else "black"
            ax.text(
                j, i, f"{count}\n({pct:.1f}%)",
                horizontalalignment="center",
                verticalalignment="center",
                color=txt_color,
                fontsize=8,
                fontweight="bold" if count > 0 else "normal"
            )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=15)
    ax.set_ylabel("True Category", fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted Category", fontsize=10, fontweight="bold")
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix plot saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Render confusion matrix from evaluation report")
    parser.add_argument("--report", type=str, default=os.path.join(PROJECT_ROOT, "ml", "metadata", "final_metrics.json"))
    parser.add_argument("--output", type=str, default=os.path.join(PROJECT_ROOT, "ml", "metadata", "confusion_matrix_test.png"))
    parser.add_argument("--title", type=str, default="SolarSentinel VGG16 Confusion Matrix (Locked Test Set)")
    args = parser.parse_args()

    if os.path.exists(args.report):
        with open(args.report, "r", encoding="utf-8") as f:
            data = json.load(f)
        matrix = data.get("confusion_matrix")
        classes = data.get("classes", CANONICAL_CLASSES)
        plot_confusion_matrix(matrix, classes=classes, output_path=args.output, title=args.title)
    else:
        print(f"Report not found at {args.report}")

