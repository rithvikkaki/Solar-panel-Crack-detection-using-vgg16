"""
SolarSentinel AI - Two-Stage VGG16 Fine-Tuning Pipeline
Stage 1: Feature Extraction (Frozen VGG16 backbone + trainable head)
Stage 2: Differential Fine-Tuning (Unfreezes Block 5 with reduced learning rate 1e-5)
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
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "ml", "experiments", "phase9_vgg16", "fine_tuning")


def run_staged_fine_tuning(
    stage1_epochs: int = 10,
    stage2_epochs: int = 15,
    batch_size: int = 32,
    head_type: str = "head_b",
    output_dir: str = OUTPUT_DIR
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, "best_fine_tuned_model.keras")

    print("=" * 60)
    print("STAGE 1: TRAINING CLASSIFICATION HEAD")
    print("=" * 60)

    train_ds = load_dataset_from_manifest(
        manifest_path=MANIFEST_PATH,
        split="train",
        target_size=(224, 224),
        batch_size=batch_size,
        shuffle=True,
        augment=True
    )

    val_ds = load_dataset_from_manifest(
        manifest_path=MANIFEST_PATH,
        split="val",
        target_size=(224, 224),
        batch_size=batch_size,
        shuffle=False,
        augment=False
    )

    model = build_vgg16_classifier(
        input_shape=(224, 224, 3),
        num_classes=6,
        head_type=head_type,
        freeze_backbone=True
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage1_epochs,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        ]
    )

    print("\n" + "=" * 60)
    print("STAGE 2: UNFREEZING BLOCK 5 & FINE-TUNING")
    print("=" * 60)

    # Unfreeze block 5
    vgg_base = model.get_layer("vgg16")
    vgg_base.trainable = True
    trainable_flag = False
    for layer in vgg_base.layers:
        if layer.name == "block5_conv1":
            trainable_flag = True
        layer.trainable = trainable_flag

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        ModelCheckpoint(best_model_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)
    ]

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage2_epochs,
        callbacks=callbacks
    )

    val_loss, val_acc = model.evaluate(val_ds)
    print(f"\nFinal Fine-Tuned Validation Accuracy: {val_acc * 100:.2f}% | Loss: {val_loss:.4f}")

    metrics = {
        "model_name": f"vgg16_{head_type}_fine_tuned_block5",
        "val_accuracy": float(val_acc),
        "val_loss": float(val_loss),
        "stage2_epochs": len(history2.history["loss"])
    }

    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    run_staged_fine_tuning()
