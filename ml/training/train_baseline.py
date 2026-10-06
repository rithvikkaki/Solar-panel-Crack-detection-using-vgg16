"""
SolarSentinel AI - VGG16 Frozen Backbone Baseline Trainer
Trains ImageNet pre-trained VGG16 with a compact GlobalAveragePooling2D head (Head A)
using the strictly isolated 70% train split and evaluating on the 15% validation split.
"""
import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.data.preprocessing import load_dataset_from_manifest
from ml.models.vgg16 import build_vgg16_classifier

MANIFEST_PATH = os.path.join(PROJECT_ROOT, "ml", "metadata", "split_manifest_70_15_15.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "ml", "experiments", "phase9_vgg16", "baseline")


def run_baseline_training(
    epochs: int = 15,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    output_dir: str = OUTPUT_DIR
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, "best_baseline_model.keras")

    print("=" * 60)
    print("VGG16 BASELINE TRAINING (HEAD A - FROZEN BACKBONE)")
    print("=" * 60)

    # 1. Load Data
    train_ds = load_dataset_from_manifest(
        manifest_path=MANIFEST_PATH,
        split="train",
        target_size=(224, 224),
        batch_size=batch_size,
        shuffle=True,
        augment=False
    )

    val_ds = load_dataset_from_manifest(
        manifest_path=MANIFEST_PATH,
        split="val",
        target_size=(224, 224),
        batch_size=batch_size,
        shuffle=False,
        augment=False
    )

    # 2. Build Model
    model = build_vgg16_classifier(
        input_shape=(224, 224, 3),
        num_classes=6,
        head_type="head_a",
        freeze_backbone=True
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        ModelCheckpoint(best_model_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    # Evaluate best model
    val_loss, val_acc = model.evaluate(val_ds)
    print(f"\nBaseline Validation Accuracy: {val_acc * 100:.2f}% | Loss: {val_loss:.4f}")

    metrics = {
        "model_name": "vgg16_baseline_head_a",
        "epochs_trained": len(history.history["loss"]),
        "val_accuracy": float(val_acc),
        "val_loss": float(val_loss),
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()}
    }

    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    run_baseline_training()
