"""
SolarSentinel AI - Model Evaluation Pipeline
Evaluates trained VGG16 model on validation or test sets.
Computes:
- Overall Accuracy
- Macro & Weighted Precision, Recall, F1 Score
- Per-Class Metrics (Precision, Recall, F1, Support)
- Confusion Matrix (normalized & raw counts)
- Classification Report

Enforces Scientific Honesty:
Metrics are ONLY computed from actual dataset passes. Zero invented metrics.
Saves results strictly to ml/metadata/evaluation_metrics.json.
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from typing import Dict, List

import numpy as np


def evaluate_model(
    model_path: str,
    data_dir: str,
    class_names_path: str,
    output_metrics_path: str,
    img_size: int = 244,
    batch_size: int = 32,
    seed: int = 42
) -> Dict:
    """
    Evaluates model on validation partition and generates comprehensive metrics.
    """
    if not os.path.exists(model_path):
        print(f"[FATAL] Model file not found at: {model_path}")
        print("Cannot evaluate non-existent model. Please train model first.")
        return {}

    if not os.path.exists(data_dir):
        print(f"[FATAL] Dataset directory not found at: {data_dir}")
        print("Cannot evaluate without dataset images.")
        return {}

    import tensorflow as tf
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        confusion_matrix,
        classification_report
    )

    print(f"\n{'='*60}")
    print(f"SolarSentinel AI - Scientific Model Evaluation")
    print(f"Model Path: {model_path}")
    print(f"Dataset Path: {data_dir}")
    print(f"Evaluation Input Resolution: {img_size}x{img_size}")
    print(f"{'='*60}\n")

    # Load model
    print("[INFO] Loading serialized model...")
    model = tf.keras.models.load_model(model_path)

    # Load validation partition identically to training split
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        seed=seed,
        shuffle=False
    )

    class_names = val_ds.class_names
    num_classes = len(class_names)
    print(f"[INFO] Evaluating across {num_classes} classes: {class_names}")

    # Collect ground truth and model predictions
    y_true = []
    y_pred_probs = []

    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred_probs.extend(preds)

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Calculate exact metrics
    accuracy = float(accuracy_score(y_true, y_pred))

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    p_per_class, r_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0
    )

    cm_raw = confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).tolist()
    cm_norm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)), normalize="true").tolist()

    clf_report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )

    metrics_payload = {
        "evaluation_status": "COMPLETED",
        "input_dimensions": [img_size, img_size, 3],
        "total_evaluation_samples": int(len(y_true)),
        "num_classes": num_classes,
        "classes": class_names,
        "overall": {
            "accuracy": round(accuracy, 4),
            "precision_macro": round(float(precision_macro), 4),
            "recall_macro": round(float(recall_macro), 4),
            "f1_macro": round(float(f1_macro), 4),
            "precision_weighted": round(float(precision_weighted), 4),
            "recall_weighted": round(float(recall_weighted), 4),
            "f1_weighted": round(float(f1_weighted), 4)
        },
        "per_class": {
            class_names[i]: {
                "precision": round(float(p_per_class[i]), 4),
                "recall": round(float(r_per_class[i]), 4),
                "f1_score": round(float(f1_per_class[i]), 4),
                "support": int(support_per_class[i])
            }
            for i in range(num_classes)
        },
        "confusion_matrix": {
            "raw": cm_raw,
            "normalized": [[round(val, 4) for val in row] for row in cm_norm],
            "labels": class_names
        },
        "classification_report": clf_report
    }

    # Save to metadata
    os.makedirs(os.path.dirname(output_metrics_path), exist_ok=True)
    with open(output_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"\nEvaluation Results:")
    print(f"  - Accuracy: {metrics_payload['overall']['accuracy'] * 100:.2f}%")
    print(f"  - Macro F1: {metrics_payload['overall']['f1_macro']:.4f}")
    print(f"  - Weighted F1: {metrics_payload['overall']['f1_weighted']:.4f}")
    print(f"\n[INFO] Saved real evaluation metrics to: {output_metrics_path}")
    return metrics_payload


def main():
    parser = argparse.ArgumentParser(description="Evaluate SolarSentinel VGG16")
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join("ml", "models", "solar_sentinel_vgg16.keras"),
        help="Path to trained .keras model"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join("data", "raw"),
        help="Path to dataset"
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=244,
        choices=[244, 224],
        help="Input dimension (244 baseline or 224 modern)"
    )
    parser.add_argument(
        "--output_metrics",
        type=str,
        default=os.path.join("ml", "metadata", "evaluation_metrics.json"),
        help="Output metrics JSON path"
    )
    args = parser.parse_args()

    class_meta = os.path.join("ml", "metadata", "class_names.json")
    evaluate_model(
        model_path=args.model_path,
        data_dir=args.data_dir,
        class_names_path=class_meta,
        output_metrics_path=args.output_metrics,
        img_size=args.input_size
    )


if __name__ == "__main__":
    main()
