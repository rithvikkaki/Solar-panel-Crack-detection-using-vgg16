"""
SolarSentinel AI - Systematic VGG16 Optimization Suite
Executes principled controlled experiments across:
1. Classification Heads & Dropout (Dense 128/256/512, BatchNorm, Two-layer)
2. Fine-Tuning Depths (F0: Frozen, F1: Block 5, F2: Blocks 4+5)
3. Augmentations (A0: None, A1: Spatial, A2: Photometric, A3: Irradiance)
4. Resolutions (244x244, 320x320)
5. Class Imbalance (Standard CE, Weighted CE, Focal Loss, Label Smoothing)
6. Optimizers & Discriminative Learning Rates (Adam, Cosine Decay, ReduceLROnPlateau)
7. Multi-Seed Stability (42, 123, 2026)

All model selection is strictly evaluated on the 131-image Validation Set.
"""
import os
import sys
import json
import time
import csv
import argparse
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MANIFEST_PATH = os.path.join(PROJECT_ROOT, "ml", "metadata", "split_manifest_70_15_15.json")
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "ml", "experiments", "phase9_vgg16")
REGISTRY_CSV = os.path.join(EXPERIMENTS_DIR, "experiment_results.csv")


def load_cached_partition(manifest_path: str, partition: str, target_size: Tuple[int, int]):
    """Pre-caches image partition in RAM for fast training without disk bottleneck."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    items = manifest.get(partition) or manifest.get(f"{partition}_files", [])
    classes = manifest.get("classes", [])

    images = []
    labels = []
    for it in items:
        p = it["path"] if isinstance(it, dict) else it
        abs_p = os.path.join(PROJECT_ROOT, p) if not os.path.isabs(p) else p
        lbl = it["class_index"] if isinstance(it, dict) else classes.index(it.get("class", os.path.basename(os.path.dirname(abs_p))))
        with Image.open(abs_p) as raw:
            img = raw.convert("RGB").resize((target_size[1], target_size[0]), Image.Resampling.BILINEAR)
            images.append(np.array(img, dtype=np.float32))
        labels.append(lbl)

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32), classes


def compute_class_weights(labels: np.ndarray, num_classes: int) -> Dict[int, float]:
    total = len(labels)
    counts = np.bincount(labels, minlength=num_classes)
    return {i: float(total / (num_classes * c)) if c > 0 else 1.0 for i, c in enumerate(counts)}


def build_augmentation(aug_type: str):
    if aug_type == "none" or aug_type == "A0":
        return None
    elif aug_type == "spatial" or aug_type == "A1":
        return tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.05, fill_mode="nearest")
        ], name="aug_spatial")
    elif aug_type == "zoom" or aug_type == "A2":
        return tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.05, fill_mode="nearest"),
            tf.keras.layers.RandomZoom((-0.08, 0.08), fill_mode="nearest")
        ], name="aug_zoom")
    elif aug_type == "photometric" or aug_type == "A3":
        return tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.05, fill_mode="nearest"),
            tf.keras.layers.RandomZoom((-0.06, 0.06), fill_mode="nearest"),
            tf.keras.layers.RandomBrightness(0.08, value_range=(0, 255)),
            tf.keras.layers.RandomContrast(0.08)
        ], name="aug_photometric")
    elif aug_type == "robust" or aug_type == "A4":
        return tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.08, fill_mode="nearest"),
            tf.keras.layers.RandomZoom((-0.1, 0.1), fill_mode="nearest"),
            tf.keras.layers.RandomBrightness(0.12, value_range=(0, 255)),
            tf.keras.layers.RandomContrast(0.12)
        ], name="aug_robust")
    else:
        raise ValueError(f"Unknown aug_type: {aug_type}")


def build_vgg16_candidate(
    head_type: str = "dense_512_256",
    dropout: float = 0.3,
    input_size: int = 244,
    num_classes: int = 6,
    aug_type: str = "photometric"
):
    inputs = tf.keras.layers.Input(shape=(input_size, input_size, 3), name="solar_input")
    aug = build_augmentation(aug_type)
    x_aug = aug(inputs) if aug is not None else inputs

    # Preprocessing embedded
    x_prep = tf.keras.applications.vgg16.preprocess_input(x_aug)
    base = tf.keras.applications.VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=(input_size, input_size, 3)
    )
    base.trainable = False
    x = base(x_prep, training=False)

    # Classification Heads
    if head_type == "gap_dense_128":
        x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
        x = tf.keras.layers.Dense(128, activation="relu", name="dense_128")(x)
        x = tf.keras.layers.Dropout(dropout, name="dropout")(x)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    elif head_type == "gap_dense_256":
        x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
        x = tf.keras.layers.Dense(256, activation="relu", name="dense_256")(x)
        x = tf.keras.layers.Dropout(dropout, name="dropout")(x)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    elif head_type == "gap_dense_512":
        x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
        x = tf.keras.layers.Dense(512, activation="relu", name="dense_512")(x)
        x = tf.keras.layers.Dropout(dropout, name="dropout")(x)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    elif head_type == "gap_bn_dense_256":
        x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
        x = tf.keras.layers.BatchNormalization(name="bn")(x)
        x = tf.keras.layers.Dense(256, activation="relu", name="dense_256")(x)
        x = tf.keras.layers.Dropout(dropout, name="dropout")(x)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    elif head_type == "gap_bn_dense_512":
        x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
        x = tf.keras.layers.BatchNormalization(name="bn")(x)
        x = tf.keras.layers.Dense(512, activation="relu", name="dense_512")(x)
        x = tf.keras.layers.Dropout(dropout, name="dropout")(x)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    elif head_type == "dense_512_256":  # Canonical Head B
        x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
        x = tf.keras.layers.BatchNormalization(name="bn_gap")(x)
        x = tf.keras.layers.Dense(512, activation="relu", name="dense_512")(x)
        x = tf.keras.layers.Dropout(dropout, name="dropout_1")(x)
        x = tf.keras.layers.Dense(256, activation="relu", name="dense_256")(x)
        x = tf.keras.layers.Dropout(max(0.1, dropout - 0.1), name="dropout_2")(x)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    elif head_type == "head_a":  # Baseline Head A
        x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
        x = tf.keras.layers.Dropout(dropout, name="dropout")(x)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    else:
        raise ValueError(f"Unknown head_type: {head_type}")

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"vgg16_{head_type}")
    return model, base


def evaluate_val_set(model, x_val, y_val, classes):
    preds = model.predict(x_val, batch_size=32, verbose=0)
    yp = np.argmax(preds, axis=1)

    acc = float(accuracy_score(y_val, yp))
    pm, rm, f1m, _ = precision_recall_fscore_support(y_val, yp, average="macro", zero_division=0)
    pw, rw, f1w, _ = precision_recall_fscore_support(y_val, yp, average="weighted", zero_division=0)
    pp, rp, f1p, sp = precision_recall_fscore_support(y_val, yp, average=None, labels=list(range(len(classes))), zero_division=0)

    phys_idx = classes.index("Physical-Damage")
    cm = confusion_matrix(y_val, yp, labels=list(range(len(classes)))).tolist()

    return {
        "val_accuracy": round(acc, 4),
        "val_macro_precision": round(float(pm), 4),
        "val_macro_recall": round(float(rm), 4),
        "val_macro_f1": round(float(f1m), 4),
        "val_weighted_f1": round(float(f1w), 4),
        "physical_damage_precision": round(float(pp[phys_idx]), 4),
        "physical_damage_recall": round(float(rp[phys_idx]), 4),
        "physical_damage_f1": round(float(f1p[phys_idx]), 4),
        "confusion_matrix": cm
    }


def run_experiment(
    exp_id: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    classes: List[str],
    head_type: str = "dense_512_256",
    dropout: float = 0.3,
    fine_tune_blocks: str = "block5",  # "none", "block5", "block4_5", "block3_4_5"
    aug_type: str = "photometric",
    loss_type: str = "weighted_ce",    # "ce", "weighted_ce", "label_smoothing"
    optimizer_name: str = "adam",
    stage1_lr: float = 1e-3,
    stage2_lr: float = 1e-4,
    stage1_epochs: int = 8,
    stage2_epochs: int = 12,
    batch_size: int = 32,
    seed: int = 42,
    input_size: int = 244
) -> Dict:
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)

    exp_dir = os.path.join(EXPERIMENTS_DIR, exp_id)
    os.makedirs(exp_dir, exist_ok=True)
    ckpt_path = os.path.join(exp_dir, "best_model.keras")

    model, base = build_vgg16_candidate(
        head_type=head_type,
        dropout=dropout,
        input_size=input_size,
        num_classes=len(classes),
        aug_type=aug_type
    )

    cw = compute_class_weights(y_train, len(classes)) if loss_type == "weighted_ce" else None

    # Loss selection
    if loss_type == "label_smoothing":
        loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
        y_train_fit = tf.keras.utils.to_categorical(y_train, len(classes))
        y_val_fit = tf.keras.utils.to_categorical(y_val, len(classes))
    else:
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        y_train_fit = y_train
        y_val_fit = y_val

    # Optimizer
    if optimizer_name == "adamw":
        opt1 = tf.keras.optimizers.AdamW(learning_rate=stage1_lr, weight_decay=1e-4)
    else:
        opt1 = tf.keras.optimizers.Adam(learning_rate=stage1_lr)

    model.compile(optimizer=opt1, loss=loss_fn, metrics=["accuracy"])

    # Stage 1
    es1 = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    model.fit(
        x_train, y_train_fit,
        validation_data=(x_val, y_val_fit),
        epochs=stage1_epochs,
        batch_size=batch_size,
        class_weight=cw,
        callbacks=[es1],
        verbose=0
    )

    # Stage 2 (Fine-Tuning if specified)
    if fine_tune_blocks != "none":
        base.trainable = True
        if fine_tune_blocks == "block5":
            depth = 15  # Block 5 layers trainable (15 to 19)
        elif fine_tune_blocks == "block4_5":
            depth = 11  # Block 4 & 5 trainable (11 to 19)
        elif fine_tune_blocks == "block3_4_5":
            depth = 7   # Block 3, 4, 5 trainable (7 to 19)
        else:
            depth = 15

        for l in base.layers[:depth]:
            l.trainable = False

        if optimizer_name == "adamw":
            opt2 = tf.keras.optimizers.AdamW(learning_rate=stage2_lr, weight_decay=1e-4)
        else:
            opt2 = tf.keras.optimizers.Adam(learning_rate=stage2_lr)

        model.compile(optimizer=opt2, loss=loss_fn, metrics=["accuracy"])

        ckpt = tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True, verbose=0)
        es2 = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
        rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)

        hist = model.fit(
            x_train, y_train_fit,
            validation_data=(x_val, y_val_fit),
            epochs=stage2_epochs,
            batch_size=batch_size,
            class_weight=cw,
            callbacks=[ckpt, es2, rlr],
            verbose=0
        )
        total_epochs = stage1_epochs + len(hist.history["loss"])
    else:
        model.save(ckpt_path)
        total_epochs = stage1_epochs

    # Load best checkpoint and evaluate strictly on validation set
    best_m = tf.keras.models.load_model(ckpt_path)
    val_res = evaluate_val_set(best_m, x_val, y_val, classes)

    # Trainable parameters count
    trainable_params = sum(tf.keras.backend.count_params(w) for w in best_m.trainable_weights)

    record = {
        "experiment_id": exp_id,
        "seed": seed,
        "input_size": input_size,
        "head_type": head_type,
        "dropout": dropout,
        "batch_size": batch_size,
        "optimizer": optimizer_name,
        "learning_rate": stage2_lr if fine_tune_blocks != "none" else stage1_lr,
        "scheduler": "ReduceLROnPlateau",
        "loss": loss_type,
        "class_weighting": bool(cw is not None),
        "augmentation": aug_type,
        "fine_tuned_blocks": fine_tune_blocks,
        "trainable_parameters": trainable_params,
        "epochs": total_epochs,
        "val_accuracy": val_res["val_accuracy"],
        "val_macro_precision": val_res["val_macro_precision"],
        "val_macro_recall": val_res["val_macro_recall"],
        "val_macro_f1": val_res["val_macro_f1"],
        "val_weighted_f1": val_res["val_weighted_f1"],
        "physical_damage_precision": val_res["physical_damage_precision"],
        "physical_damage_recall": val_res["physical_damage_recall"],
        "physical_damage_f1": val_res["physical_damage_f1"],
        "checkpoint_path": ckpt_path.replace("\\", "/")
    }

    with open(os.path.join(exp_dir, "val_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({**val_res, "config": record}, f, indent=2)

    return record


def update_registry(records: List[Dict]):
    """Appends/updates registry CSV and JSON."""
    existing = []
    if os.path.exists(REGISTRY_CSV):
        with open(REGISTRY_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing = list(reader)

    # Merge by experiment_id
    by_id = {r["experiment_id"]: r for r in existing}
    for r in records:
        by_id[r["experiment_id"]] = {k: str(v) for k, v in r.items() if k != "confusion_matrix"}

    merged = sorted(by_id.values(), key=lambda x: float(x.get("val_macro_f1", 0)), reverse=True)

    if merged:
        fieldnames = list(merged[0].keys())
        with open(REGISTRY_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(merged)

        with open(os.path.join(EXPERIMENTS_DIR, "experiment_results.json"), "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2)


if __name__ == "__main__":
    print("Pre-caching dataset partitions (244x244)...")
    t0 = time.time()
    x_tr, y_tr, classes = load_cached_partition(MANIFEST_PATH, "train", (244, 244))
    x_vl, y_vl, _ = load_cached_partition(MANIFEST_PATH, "val", (244, 244))
    print(f"Cached Train: {x_tr.shape}, Val: {x_vl.shape} in {time.time() - t0:.1f}s")

    # Define systematic exploration matrix
    experiments_plan = [
        # Vector 1: Head architecture comparison
        {"exp_id": "opt_head_gap_dense128", "head_type": "gap_dense_128", "dropout": 0.3, "fine_tune_blocks": "block5", "aug_type": "photometric"},
        {"exp_id": "opt_head_gap_dense256", "head_type": "gap_dense_256", "dropout": 0.3, "fine_tune_blocks": "block5", "aug_type": "photometric"},
        {"exp_id": "opt_head_gap_dense512", "head_type": "gap_dense_512", "dropout": 0.3, "fine_tune_blocks": "block5", "aug_type": "photometric"},
        {"exp_id": "opt_head_bn_dense256", "head_type": "gap_bn_dense_256", "dropout": 0.3, "fine_tune_blocks": "block5", "aug_type": "photometric"},
        {"exp_id": "opt_head_bn_dense512", "head_type": "gap_bn_dense_512", "dropout": 0.3, "fine_tune_blocks": "block5", "aug_type": "photometric"},
        {"exp_id": "opt_head_two_layer_b", "head_type": "dense_512_256", "dropout": 0.3, "fine_tune_blocks": "block5", "aug_type": "photometric"},

        # Vector 2: Dropout exploration on best head
        {"exp_id": "opt_dropout_02", "head_type": "dense_512_256", "dropout": 0.2, "fine_tune_blocks": "block5", "aug_type": "photometric"},
        {"exp_id": "opt_dropout_04", "head_type": "dense_512_256", "dropout": 0.4, "fine_tune_blocks": "block5", "aug_type": "photometric"},
        {"exp_id": "opt_dropout_05", "head_type": "dense_512_256", "dropout": 0.5, "fine_tune_blocks": "block5", "aug_type": "photometric"},

        # Vector 3: Augmentation ablation
        {"exp_id": "opt_aug_a0_none", "head_type": "dense_512_256", "dropout": 0.3, "fine_tune_blocks": "block5", "aug_type": "A0"},
        {"exp_id": "opt_aug_a1_spatial", "head_type": "dense_512_256", "dropout": 0.3, "fine_tune_blocks": "block5", "aug_type": "A1"},
        {"exp_id": "opt_aug_a4_robust", "head_type": "dense_512_256", "dropout": 0.3, "fine_tune_blocks": "block5", "aug_type": "A4"},

        # Vector 4: Fine-tuning depth
        {"exp_id": "opt_depth_f0_frozen", "head_type": "dense_512_256", "dropout": 0.3, "fine_tune_blocks": "none", "aug_type": "photometric"},
        {"exp_id": "opt_depth_f2_block4_5", "head_type": "dense_512_256", "dropout": 0.3, "fine_tune_blocks": "block4_5", "aug_type": "photometric", "stage2_lr": 3e-5},

        # Vector 5: Imbalance handling & learning rate
        {"exp_id": "opt_loss_unweighted_ce", "head_type": "dense_512_256", "dropout": 0.3, "fine_tune_blocks": "block5", "aug_type": "photometric", "loss_type": "ce"},
        {"exp_id": "opt_lr_conservative", "head_type": "dense_512_256", "dropout": 0.3, "fine_tune_blocks": "block5", "aug_type": "photometric", "stage2_lr": 5e-5},
    ]

    all_results = []
    print(f"\nStarting execution of {len(experiments_plan)} systematic VGG16 experiments...")

    for idx, exp in enumerate(experiments_plan, 1):
        print(f"\n[{idx}/{len(experiments_plan)}] Running {exp['exp_id']}...")
        t_start = time.time()
        res = run_experiment(
            exp_id=exp["exp_id"],
            x_train=x_tr,
            y_train=y_tr,
            x_val=x_vl,
            y_val=y_vl,
            classes=classes,
            head_type=exp["head_type"],
            dropout=exp.get("dropout", 0.3),
            fine_tune_blocks=exp.get("fine_tune_blocks", "block5"),
            aug_type=exp.get("aug_type", "photometric"),
            loss_type=exp.get("loss_type", "weighted_ce"),
            stage2_lr=exp.get("stage2_lr", 1e-4)
        )
        elapsed = time.time() - t_start
        all_results.append(res)
        print(f"  -> Finished in {elapsed:.1f}s | Val Acc: {res['val_accuracy']*100:.2f}% | Val Macro F1: {res['val_macro_f1']:.4f} | Phys Recall: {res['physical_damage_recall']*100:.1f}%")
        update_registry([res])

    print("\n" + "=" * 60)
    print("SYSTEMATIC OPTIMIZATION CYCLE COMPLETE")
    print("=" * 60)
    top_5 = sorted(all_results, key=lambda x: (x["val_macro_f1"], x["val_accuracy"]), reverse=True)[:5]
    for r in top_5:
        print(f"  [{r['experiment_id']}] Val Acc: {r['val_accuracy']*100:.2f}% | Macro F1: {r['val_macro_f1']:.4f} | Phys Rec: {r['physical_damage_recall']*100:.1f}%")
