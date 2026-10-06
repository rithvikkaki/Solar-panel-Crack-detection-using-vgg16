"""
SolarSentinel AI - Production Training Pipeline
Implements two-stage transfer learning with VGG16 backbone:
- Stage 1: Feature Extraction with frozen backbone
- Stage 2: Fine-Tuning top convolutional block (block5)

Fixes original defects:
- Replaces hardcoded Dense(90) with dynamic Dense(num_classes, activation='softmax')
- Adds proper data augmentation suited for PV surface images
- Implements reproducible random seeds and modern .keras model saving
- Preserves 244x244 baseline while supporting 224x224 via CLI
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from typing import List, Tuple
import numpy as np

from ml.training.config import (
    TrainingConfig,
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_CLASS_NAMES_PATH
)
from ml.training.validate_dataset import validate_dataset


def build_solar_vgg16_model(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    dropout_rate: float = 0.3
):
    """
    Constructs the SolarSentinel VGG16 model.
    Dynamically sizes output layer to num_classes with softmax activation.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import vgg16

    # 1. Input Layer
    inputs = layers.Input(shape=input_shape, name="solar_image_input")

    # 2. Data Augmentation (active only during training)
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal", name="aug_horizontal_flip"),
        layers.RandomRotation(0.04, fill_mode="nearest", name="aug_subtle_rotation"),
        layers.RandomZoom(0.05, fill_mode="nearest", name="aug_subtle_zoom"),
    ], name="solar_augmentation")

    x = data_augmentation(inputs)

    # 3. Canonical VGG16 Preprocessing (RGB -> BGR, ImageNet mean subtraction)
    x = vgg16.preprocess_input(x)

    # 4. VGG16 Backbone (pre-trained on ImageNet)
    base_model = vgg16.VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )
    base_model.trainable = False  # Frozen for Stage 1

    x = base_model(x, training=False)

    # 5. Classification Head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(dropout_rate, name="dropout_head")(x)
    
    # CRITICAL FIX: Dynamic num_classes with softmax activation (replaces Dense(90))
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name=f"classification_head_{num_classes}_classes"
    )(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="SolarSentinel_VGG16")
    return model, base_model


def run_training(config: TrainingConfig):
    """Executes the complete two-stage training workflow."""
    import tensorflow as tf

    # Set reproducible random seeds
    tf.keras.utils.set_random_seed(config.random_seed)
    np.random.seed(config.random_seed)

    print(f"\n{'='*60}")
    print(f"SolarSentinel AI - VGG16 Transfer Learning Training")
    print(f"Dataset Directory: {config.data_dir}")
    print(f"Input Resolution: {config.img_height}x{config.img_width}x{config.channels}")
    print(f"{'='*60}\n")

    # 1. Validate Dataset Exists
    from ml.training.validate_dataset import validate_dataset
    val_report = validate_dataset(config.data_dir)
    if not val_report["valid"]:
        print(f"[FATAL] Dataset validation failed: {val_report['summary']}")
        print("Please place verified dataset images in the data directory before training.")
        print(f"Reference: {os.path.join(config.data_dir, 'README.md')}")
        return False

    # 2. Load Datasets
    print("[INFO] Loading training and validation partitions...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        config.data_dir,
        validation_split=config.validation_split,
        subset="training",
        image_size=(config.img_height, config.img_width),
        batch_size=config.batch_size,
        seed=config.random_seed,
        shuffle=config.shuffle
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        config.data_dir,
        validation_split=config.validation_split,
        subset="validation",
        image_size=(config.img_height, config.img_width),
        batch_size=config.batch_size,
        seed=config.random_seed,
        shuffle=False
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"[INFO] Verified Classes ({num_classes}): {class_names}")

    # Persist detected class metadata
    os.makedirs(os.path.dirname(config.class_names_path), exist_ok=True)
    with open(config.class_names_path, "w", encoding="utf-8") as f:
        json.dump({
            "classes": class_names,
            "num_classes": num_classes,
            "class_to_idx": {c: i for i, c in enumerate(class_names)},
            "idx_to_class": {str(i): c for i, c in enumerate(class_names)}
        }, f, indent=2)

    # Prefetch for optimal I/O throughput
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    # 3. Build Model Architecture
    print(f"\n[INFO] Constructing VGG16 with {num_classes} dynamic output classes...")
    model, base_model = build_solar_vgg16_model(
        input_shape=config.input_shape,
        num_classes=num_classes,
        dropout_rate=config.dropout_rate
    )
    model.summary()

    # 4. Stage 1: Feature Extraction
    print(f"\n{'='*40}")
    print("STAGE 1: Training Classification Head (Backbone Frozen)")
    print(f"{'='*40}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.stage1_lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    stage1_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=config.early_stopping_min_delta,
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
    ]

    history_stage1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.stage1_epochs,
        callbacks=stage1_callbacks
    )

    # 5. Stage 2: Fine-Tuning Block 5
    print(f"\n{'='*40}")
    print("STAGE 2: Fine-Tuning Convolutional Block 5")
    print(f"{'='*40}")
    base_model.trainable = True
    for layer in base_model.layers[:config.fine_tune_from_layer]:
        layer.trainable = False

    trainable_count = sum(len(layer.trainable_weights) for layer in model.layers)
    print(f"[INFO] Layers up to {config.fine_tune_from_layer} frozen. Fine-tuning top layers.")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.stage2_lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    os.makedirs(os.path.dirname(config.model_path), exist_ok=True)
    stage2_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=config.early_stopping_min_delta,
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.reduce_lr_factor,
            patience=config.reduce_lr_patience,
            min_lr=config.reduce_lr_min,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=config.model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]

    history_stage2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.stage2_epochs,
        callbacks=stage2_callbacks
    )

    # 6. Save Final Model in .keras format
    print(f"\n[INFO] Saving final trained model to: {config.model_path}")
    model.save(config.model_path)
    print("[SUCCESS] Training pipeline completed successfully.")
    return True


def main():
    parser = argparse.ArgumentParser(description="SolarSentinel AI VGG16 Training")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="Path to raw dataset")
    parser.add_argument("--input_size", type=int, default=244, choices=[244, 224], help="Input dimension (244 baseline or 224 modern)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs1", type=int, default=15, help="Stage 1 epochs")
    parser.add_argument("--epochs2", type=int, default=15, help="Stage 2 epochs")
    parser.add_argument("--lr1", type=float, default=0.001, help="Stage 1 learning rate")
    parser.add_argument("--lr2", type=float, default=0.0001, help="Stage 2 fine-tuning learning rate")
    parser.add_argument("--fine_tune_from_layer", type=int, default=14, help="Layer index from which to unfreeze VGG16 backbone (default: 14, unfreezing Block 5)")
    parser.add_argument("--model_out", type=str, default=DEFAULT_MODEL_PATH, help="Output .keras model path")
    args = parser.parse_args()

    cfg = TrainingConfig(
        img_height=args.input_size,
        img_width=args.input_size,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        stage1_epochs=args.epochs1,
        stage1_lr=args.lr1,
        stage2_epochs=args.epochs2,
        stage2_lr=args.lr2,
        fine_tune_from_layer=args.fine_tune_from_layer,
        model_path=args.model_out
    )

    run_training(cfg)


if __name__ == "__main__":
    main()
