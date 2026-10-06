"""
SolarSentinel AI - Evaluation Engine
Evaluates VGG16 models on validation or locked test partitions.
Computes multi-class classification metrics: Overall Accuracy, Macro/Weighted F1,
Precision, Recall, and Per-Class breakdown with zero metric inflation.
"""
import os
import sys
import json
import argparse
from typing import Dict, Any, List
from PIL import Image
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.data.preprocessing import load_and_preprocess_image

DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "ml", "models", "solar_sentinel_vgg16.keras")
DEFAULT_MANIFEST = os.path.join(PROJECT_ROOT, "ml", "metadata", "split_manifest_70_15_15.json")
CANONICAL_CLASSES = [
    "Bird-drop",
    "Clean",
    "Dusty",
    "Electrical-damage",
    "Physical-Damage",
    "Snow-Covered"
]


def evaluate_model(
    model_path: str = DEFAULT_MODEL_PATH,
    manifest_path: str = DEFAULT_MANIFEST,
    split: str = "val",
    output_json: str = None
) -> Dict[str, Any]:
    """
    Evaluates a saved Keras model on the specified split.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)

    # Detect model expected input resolution
    in_shape = model.input_shape
    if isinstance(in_shape, list):
        in_shape = in_shape[0]
    target_h = in_shape[1] if in_shape and len(in_shape) > 1 and in_shape[1] is not None else 244
    target_w = in_shape[2] if in_shape and len(in_shape) > 2 and in_shape[2] is not None else 244
    target_size = (target_h, target_w)
    print(f"Model expected input shape: {in_shape} -> using target_size: {target_size}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    items = manifest.get(split) or manifest.get(f"{split}_files", [])
    if not items:
        raise ValueError(f"No items found for split '{split}' in manifest.")

    classes = manifest.get("classes", CANONICAL_CLASSES)
    cls_to_idx = {name: i for i, name in enumerate(classes)}

    y_true = []
    y_pred = []
    y_prob = []
    sample_records = []

    print(f"Evaluating {len(items)} samples on split '{split}'...")
    loaded_images = []
    item_metadata = []

    for idx, item in enumerate(items):
        p = item if isinstance(item, str) else item["path"]
        abs_p = os.path.join(PROJECT_ROOT, p) if not os.path.isabs(p) else p

        true_cls = item.get("class") if isinstance(item, dict) else os.path.basename(os.path.dirname(abs_p))
        true_idx = cls_to_idx.get(true_cls, item.get("class_index"))

        with Image.open(abs_p) as raw_img:
            img = raw_img.convert("RGB").resize((target_w, target_h), Image.Resampling.BILINEAR)
            loaded_images.append(np.array(img, dtype=np.float32))

        item_metadata.append((p, true_cls, true_idx))

    batch_tensor = np.array(loaded_images, dtype=np.float32)
    all_preds = model.predict(batch_tensor, batch_size=32, verbose=0)

    for (p, true_cls, true_idx), preds in zip(item_metadata, all_preds):
        pred_idx = int(np.argmax(preds))
        confidence = float(preds[pred_idx])

        y_true.append(true_idx)
        y_pred.append(pred_idx)
        y_prob.append(preds.tolist())

        sample_records.append({
            "path": p.replace("\\", "/"),
            "true_class": true_cls,
            "true_index": true_idx,
            "predicted_class": classes[pred_idx],
            "predicted_index": pred_idx,
            "confidence": round(confidence, 4),
            "correct": bool(true_idx == pred_idx),
            "probabilities": [round(float(v), 4) for v in preds]
        })

    # Metrics calculation
    acc = float(accuracy_score(y_true, y_pred))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    per_class_p, per_class_r, per_class_f1, per_class_supp = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(classes))), zero_division=0
    )

    per_class_metrics = {}
    for i, cls in enumerate(classes):
        per_class_metrics[cls] = {
            "precision": round(float(per_class_p[i]), 4),
            "recall": round(float(per_class_r[i]), 4),
            "f1_score": round(float(per_class_f1[i]), 4),
            "support": int(per_class_supp[i])
        }

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes)))).tolist()

    report = {
        "model_path": model_path.replace("\\", "/"),
        "split": split,
        "sample_count": len(items),
        "classes": classes,
        "overall": {
            "accuracy": round(acc, 4),
            "macro_precision": round(float(macro_p), 4),
            "macro_recall": round(float(macro_r), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_precision": round(float(weighted_p), 4),
            "weighted_recall": round(float(weighted_r), 4),
            "weighted_f1": round(float(weighted_f1), 4)
        },
        "per_class": per_class_metrics,
        "confusion_matrix": cm,
        "sample_records": sample_records
    }

    if output_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Evaluation report saved to: {output_json}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VGG16 model")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--manifest", type=str, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    rep = evaluate_model(
        model_path=args.model,
        manifest_path=args.manifest,
        split=args.split,
        output_json=args.output
    )

    print("\n" + "=" * 60)
    print(f"EVALUATION SUMMARY ({args.split.upper()} SET)")
    print("=" * 60)
    print(f"Accuracy:         {rep['overall']['accuracy']*100:.2f}%")
    print(f"Macro F1:         {rep['overall']['macro_f1']:.4f}")
    print(f"Weighted F1:      {rep['overall']['weighted_f1']:.4f}")
    print(f"Macro Recall:     {rep['overall']['macro_recall']:.4f}")
    print(f"Macro Precision:  {rep['overall']['macro_precision']:.4f}")
    print("\nPer-Class Breakdown:")
    for cls, m in rep["per_class"].items():
        print(f"  {cls:<18} Precision: {m['precision']:.3f} | Recall: {m['recall']:.3f} | F1: {m['f1_score']:.3f} (n={m['support']})")
    print("=" * 60)
