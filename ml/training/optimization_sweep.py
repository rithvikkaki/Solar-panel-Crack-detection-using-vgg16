"""
VGG16 Optimization Sweep for SolarSentinel-AI
=============================================
Systematically evaluates resolution, fine-tuning depth, head architecture,
augmentation, and loss function to maximize validation accuracy.

Usage:
    python ml/training/optimization_sweep.py --stage 1       # Resolution
    python ml/training/optimization_sweep.py --stage 2       # Depth x LR
    python ml/training/optimization_sweep.py --stage 3       # Head arch
    python ml/training/optimization_sweep.py --stage 4       # Augmentation
    python ml/training/optimization_sweep.py --stage 5       # Loss function
    python ml/training/optimization_sweep.py --stage final   # Multi-seed + test
    python ml/training/optimization_sweep.py --stage summary # Print results
"""

import os
import sys
import json
import time
import gc
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import numpy as np
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MANIFEST_PATH = PROJECT_ROOT / "ml" / "metadata" / "split_manifest_70_15_15.json"
EXPERIMENT_DIR = PROJECT_ROOT / "ml" / "experiments" / "optimization_sweep"
RESULTS_FILE = EXPERIMENT_DIR / "sweep_results.json"
PRODUCTION_MODEL = PROJECT_ROOT / "ml" / "models" / "best_vgg16.keras"

CLASSES = ["Bird-drop", "Clean", "Dusty", "Electrical-damage",
           "Physical-Damage", "Snow-Covered"]
NUM_CLASSES = len(CLASSES)

# Training-set class counts (from manifest)
TRAIN_COUNTS = {
    "Bird-drop": 147, "Clean": 135, "Dusty": 133,
    "Electrical-damage": 73, "Physical-Damage": 48, "Snow-Covered": 87
}
TOTAL_TRAIN = sum(TRAIN_COUNTS.values())  # 623

# VGG16 layer indices for fine-tuning depth
DEPTH_MAP = {
    "block5": 15,   # layers 15-18 (block5_conv1..block5_pool)
    "block4_5": 11,  # layers 11-18 (block4_conv1..block5_pool)
    "block3_5": 7,   # layers 7-18  (block3_conv1..block5_pool)
}


# ── Data Structures ───────────────────────────────────────────────────
@dataclass
class ExperimentConfig:
    name: str
    resolution: int = 244
    depth: str = "block5"        # block5, block4_5, block3_5
    fine_tune_lr: float = 1e-4
    head_type: str = "A"         # A, B, C
    augmentation: str = "current"  # weak, current, strong
    loss_type: str = "ce"        # ce, weighted_ce, focal
    dropout: float = 0.5
    batch_size: int = 32
    stage1_epochs: int = 20
    stage2_epochs: int = 40
    seed: int = 42


# ── Data Loading ──────────────────────────────────────────────────────
def load_split_data(resolution: int, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load images and labels from the canonical split manifest."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    items = manifest[f"{split}_files"]
    cls_to_idx = {c: i for i, c in enumerate(CLASSES)}

    images, labels = [], []
    for item in items:
        path = PROJECT_ROOT / item["path"]
        if not path.exists():
            # Try forward slashes
            path = PROJECT_ROOT / item["path"].replace("\\", "/")
        img = Image.open(path).convert("RGB").resize(
            (resolution, resolution), Image.BILINEAR
        )
        images.append(np.array(img, dtype=np.float32))
        labels.append(cls_to_idx[item["class"]])

    return np.array(images), np.array(labels, dtype=np.int32)


# ── Model Building ───────────────────────────────────────────────────
def build_augmentation(aug_type: str):
    """Build data augmentation layer (active only during training)."""
    import tensorflow as tf
    from tensorflow.keras import layers

    if aug_type == "weak":
        return tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.02, fill_mode="nearest"),
        ], name="augmentation")
    elif aug_type == "current":
        return tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.08, fill_mode="nearest"),
            layers.RandomZoom(0.1, fill_mode="nearest"),
            layers.RandomBrightness(0.1),
            layers.RandomContrast(0.1),
        ], name="augmentation")
    elif aug_type == "strong":
        return tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.10, fill_mode="nearest"),
            layers.RandomZoom(0.15, fill_mode="nearest"),
            layers.RandomBrightness(0.15),
            layers.RandomContrast(0.15),
            layers.RandomTranslation(0.05, 0.05, fill_mode="nearest"),
        ], name="augmentation")
    else:
        raise ValueError(f"Unknown augmentation: {aug_type}")


def build_model(config: ExperimentConfig):
    """Build VGG16 model with the specified configuration."""
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import vgg16

    res = config.resolution
    input_shape = (res, res, 3)
    inputs = layers.Input(shape=input_shape, name="solar_input")

    # Augmentation
    x = build_augmentation(config.augmentation)(inputs)

    # VGG16 caffe preprocessing (RGB→BGR, ImageNet mean subtraction)
    x = vgg16.preprocess_input(x)

    # VGG16 backbone (frozen initially)
    base = vgg16.VGG16(include_top=False, weights="imagenet",
                       input_shape=input_shape)
    base.trainable = False
    x = base(x, training=False)

    # Classification head
    if config.head_type == "A":
        # Current production head: GAP→BN→Dense512→Drop→Dense256→Drop
        x = layers.GlobalAveragePooling2D(name="gap")(x)
        x = layers.BatchNormalization(name="bn_gap")(x)
        x = layers.Dense(512, activation="relu", name="dense_512")(x)
        x = layers.Dropout(config.dropout, name="dropout_1")(x)
        x = layers.Dense(256, activation="relu", name="dense_256")(x)
        x = layers.Dropout(config.dropout, name="dropout_2")(x)
    elif config.head_type == "B":
        # GAP + GMP concatenation
        gap = layers.GlobalAveragePooling2D(name="gap")(x)
        gmp = layers.GlobalMaxPooling2D(name="gmp")(x)
        x = layers.Concatenate(name="concat_gap_gmp")([gap, gmp])
        x = layers.BatchNormalization(name="bn_pool")(x)
        x = layers.Dense(512, activation="relu", name="dense_512")(x)
        x = layers.Dropout(config.dropout, name="dropout_1")(x)
    elif config.head_type == "C":
        # Lightweight: GAP→BN→Dense256→Drop
        x = layers.GlobalAveragePooling2D(name="gap")(x)
        x = layers.BatchNormalization(name="bn_gap")(x)
        x = layers.Dense(256, activation="relu", name="dense_256")(x)
        x = layers.Dropout(config.dropout, name="dropout_1")(x)
    else:
        raise ValueError(f"Unknown head_type: {config.head_type}")

    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="classifier")(x)
    model = models.Model(inputs=inputs, outputs=outputs, name="Solar_VGG16")
    return model, base


# ── Loss Functions ────────────────────────────────────────────────────
class SparseFocalLoss:
    """Focal loss for sparse (integer) labels. gamma=2.0 standard."""

    def __init__(self, gamma=2.0):
        self.gamma = gamma
        self.__name__ = "sparse_focal_loss"

    def __call__(self, y_true, y_pred):
        import tensorflow as tf
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        y_true = tf.cast(tf.squeeze(y_true), tf.int32)
        y_one_hot = tf.one_hot(y_true, NUM_CLASSES)
        ce = -y_one_hot * tf.math.log(y_pred)
        weight = y_one_hot * tf.pow(1.0 - y_pred, self.gamma)
        return tf.reduce_sum(weight * ce, axis=-1)


def get_class_weights() -> Dict[int, float]:
    """Inverse-frequency class weights for weighted CE."""
    return {
        i: TOTAL_TRAIN / (NUM_CLASSES * TRAIN_COUNTS[cls])
        for i, cls in enumerate(CLASSES)
    }


# ── Training ─────────────────────────────────────────────────────────
def run_experiment(config: ExperimentConfig,
                   X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray) -> Dict:
    """Run a single two-stage training experiment and return results."""
    import tensorflow as tf
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                 confusion_matrix)

    tf.keras.utils.set_random_seed(config.seed)
    np.random.seed(config.seed)

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {config.name}")
    print(f"  Resolution={config.resolution}  Depth={config.depth}  "
          f"LR={config.fine_tune_lr}  Head={config.head_type}")
    print(f"  Aug={config.augmentation}  Loss={config.loss_type}  Seed={config.seed}")
    print(f"{'='*70}")

    model, base = build_model(config)

    # Loss
    if config.loss_type == "focal":
        loss_fn = SparseFocalLoss(gamma=2.0)
    else:
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    class_weight = get_class_weights() if config.loss_type == "weighted_ce" else None

    # Experiment output directory
    exp_dir = EXPERIMENT_DIR / config.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(exp_dir / "model.keras")

    # ── Stage 1: Frozen backbone ──
    print("\n--- Stage 1: Feature Extraction (Frozen Backbone) ---")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=loss_fn,
        metrics=["accuracy"]
    )

    s1_cb = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5,
            restore_best_weights=True, verbose=1
        )
    ]

    t0 = time.time()
    h1 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.stage1_epochs,
        batch_size=config.batch_size,
        callbacks=s1_cb,
        class_weight=class_weight,
        verbose=2
    )
    s1_time = time.time() - t0
    s1_best = max(h1.history["val_accuracy"])
    print(f"Stage 1 best val_acc: {s1_best:.4f} ({s1_time:.0f}s)")

    # ── Stage 2: Fine-tuning ──
    ft_layer = DEPTH_MAP[config.depth]
    print(f"\n--- Stage 2: Fine-Tuning (from layer {ft_layer}, depth={config.depth}) ---")
    base.trainable = True
    for layer in base.layers[:ft_layer]:
        layer.trainable = False

    trainable_n = sum(l.count_params() for l in model.layers if l.trainable)
    print(f"Trainable parameters: {trainable_n:,}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.fine_tune_lr),
        loss=loss_fn,
        metrics=["accuracy"]
    )

    s2_cb = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=7,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3,
            min_lr=1e-7, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path, monitor="val_accuracy",
            save_best_only=True, verbose=1
        )
    ]

    t0 = time.time()
    h2 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.stage2_epochs,
        batch_size=config.batch_size,
        callbacks=s2_cb,
        class_weight=class_weight,
        verbose=2
    )
    s2_time = time.time() - t0
    s2_best = max(h2.history["val_accuracy"])
    print(f"Stage 2 best val_acc: {s2_best:.4f} ({s2_time:.0f}s)")

    # ── Evaluate best checkpoint ──
    best_model = tf.keras.models.load_model(model_path, compile=False)
    preds = best_model.predict(X_val, batch_size=config.batch_size, verbose=0)
    y_pred = np.argmax(preds, axis=1)

    acc = float(accuracy_score(y_val, y_pred))
    m_p, m_r, m_f1, _ = precision_recall_fscore_support(
        y_val, y_pred, average="macro", zero_division=0)
    w_p, w_r, w_f1, _ = precision_recall_fscore_support(
        y_val, y_pred, average="weighted", zero_division=0)
    pc_p, pc_r, pc_f1, pc_s = precision_recall_fscore_support(
        y_val, y_pred, labels=list(range(NUM_CLASSES)), zero_division=0)
    cm = confusion_matrix(y_val, y_pred,
                          labels=list(range(NUM_CLASSES))).tolist()

    per_class = {}
    for i, cls in enumerate(CLASSES):
        per_class[cls] = {
            "precision": round(float(pc_p[i]), 4),
            "recall": round(float(pc_r[i]), 4),
            "f1": round(float(pc_f1[i]), 4),
            "support": int(pc_s[i])
        }

    phys_recall = per_class.get("Physical-Damage", {}).get("recall", 0.0)

    result = {
        "name": config.name,
        "config": asdict(config),
        "val_accuracy": round(acc, 4),
        "val_macro_f1": round(float(m_f1), 4),
        "val_weighted_f1": round(float(w_f1), 4),
        "val_macro_precision": round(float(m_p), 4),
        "val_macro_recall": round(float(m_r), 4),
        "physical_damage_recall": round(phys_recall, 4),
        "per_class": per_class,
        "confusion_matrix": cm,
        "stage1_best_val_acc": round(s1_best, 4),
        "stage2_best_val_acc": round(s2_best, 4),
        "training_time_s": round(s1_time + s2_time, 1),
        "model_path": model_path,
    }

    # Save individual experiment result
    with open(exp_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n>>> {config.name}: val_acc={acc:.4f}  macro_f1={float(m_f1):.4f}  "
          f"phys_dmg_recall={phys_recall:.4f}  time={s1_time+s2_time:.0f}s")

    # Cleanup
    del model, base, best_model
    tf.keras.backend.clear_session()
    gc.collect()

    return result


# ── Results Management ────────────────────────────────────────────────
def load_results() -> List[Dict]:
    """Load all sweep results from disk."""
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    return []


def save_result(result: Dict):
    """Append a result to the sweep results file."""
    results = load_results()
    # Replace if same name exists
    results = [r for r in results if r["name"] != result["name"]]
    results.append(result)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")


def get_best_from_stage(stage_prefix: str) -> Optional[Dict]:
    """Get the best result from a given stage by val_accuracy."""
    results = load_results()
    stage_results = [r for r in results if r["name"].startswith(stage_prefix)]
    if not stage_results:
        return None
    return max(stage_results, key=lambda r: (r["val_accuracy"], r["val_macro_f1"]))


def print_results_table(results: List[Dict]):
    """Print a formatted results table."""
    if not results:
        print("No results found.")
        return
    sorted_r = sorted(results, key=lambda r: r["val_accuracy"], reverse=True)
    print(f"\n{'Name':<30} {'Val Acc':>8} {'Macro F1':>9} {'PhysDmg':>8} {'Time(s)':>8}")
    print("-" * 70)
    for r in sorted_r:
        print(f"{r['name']:<30} {r['val_accuracy']:>7.4f}  {r['val_macro_f1']:>8.4f}  "
              f"{r['physical_damage_recall']:>7.4f}  {r['training_time_s']:>7.0f}")
    print()


# ── Stage Definitions ─────────────────────────────────────────────────
def run_stage_1():
    """Stage 1: Resolution sweep at 244, 320, 384."""
    print("\n" + "=" * 70)
    print("STAGE 1: RESOLUTION SWEEP")
    print("=" * 70)

    configs = [
        ExperimentConfig(name="s1_res_244", resolution=244, fine_tune_lr=1e-4),
        ExperimentConfig(name="s1_res_320", resolution=320, fine_tune_lr=1e-4),
        ExperimentConfig(name="s1_res_384", resolution=384, fine_tune_lr=1e-4),
    ]

    existing = {r["name"] for r in load_results()}
    for cfg in configs:
        if cfg.name in existing:
            print(f"[SKIP] {cfg.name} already completed")
            continue

        X_train, y_train = load_split_data(cfg.resolution, "train")
        X_val, y_val = load_split_data(cfg.resolution, "val")
        print(f"Loaded data: train={X_train.shape}, val={X_val.shape}")

        result = run_experiment(cfg, X_train, y_train, X_val, y_val)
        save_result(result)

        del X_train, y_train, X_val, y_val
        gc.collect()

    # Summary
    results = [r for r in load_results() if r["name"].startswith("s1_")]
    print("\n--- Stage 1 Results ---")
    print_results_table(results)

    best = get_best_from_stage("s1_")
    if best:
        print(f">>> Best resolution: {best['config']['resolution']} "
              f"(val_acc={best['val_accuracy']:.4f})")


def run_stage_2():
    """Stage 2: Fine-tuning depth × learning rate."""
    best_s1 = get_best_from_stage("s1_")
    if not best_s1:
        print("[ERROR] Stage 1 not complete. Run --stage 1 first.")
        return
    res = best_s1["config"]["resolution"]
    print(f"\n{'='*70}")
    print(f"STAGE 2: FINE-TUNING DEPTH × LR (resolution={res})")
    print(f"{'='*70}")

    # Load data once
    X_train, y_train = load_split_data(res, "train")
    X_val, y_val = load_split_data(res, "val")

    configs = []
    # Include 1e-4 for block5 (standard baseline) as well as conservative rates
    for depth in ["block5", "block4_5", "block3_5"]:
        lrs = [1e-4, 1e-5, 5e-6, 1e-6] if depth == "block5" else [1e-5, 5e-6, 1e-6]
        for lr in lrs:
            lr_str = f"{lr:.0e}".replace(".", "").replace("+", "").replace("-", "m").replace("m0", "m")
            name = f"s2_{depth}_lr{lr_str}"
            configs.append(ExperimentConfig(
                name=name, resolution=res, depth=depth, fine_tune_lr=lr
            ))

    existing = {r["name"] for r in load_results()}

    # Check if s1_res_384 can be mapped to s2_block5_lr1em4 directly
    if f"s2_block5_lr1em4" not in existing:
        s1_results = [r for r in load_results() if r["name"] == f"s1_res_{res}"]
        if s1_results and s1_results[0]["config"]["depth"] == "block5" and s1_results[0]["config"]["fine_tune_lr"] == 1e-4:
            s1_match = s1_results[0].copy()
            s1_match["name"] = f"s2_block5_lr1em4"
            s1_match["config"] = s1_match["config"].copy()
            s1_match["config"]["name"] = f"s2_block5_lr1em4"
            save_result(s1_match)
            existing.add(f"s2_block5_lr1em4")
            print(f"[REUSE] Mapped {s1_results[0]['name']} -> s2_block5_lr1em4 (val_acc={s1_match['val_accuracy']:.4f})")

    for cfg in configs:
        if cfg.name in existing:
            print(f"[SKIP] {cfg.name} already completed")
            continue
        result = run_experiment(cfg, X_train, y_train, X_val, y_val)
        save_result(result)

    del X_train, y_train, X_val, y_val
    gc.collect()

    results = [r for r in load_results() if r["name"].startswith("s2_")]
    print("\n--- Stage 2 Results ---")
    print_results_table(results)

    best = get_best_from_stage("s2_")
    if best:
        print(f">>> Best: depth={best['config']['depth']} lr={best['config']['fine_tune_lr']} "
              f"(val_acc={best['val_accuracy']:.4f})")


def run_stage_3():
    """Stage 3: Head architecture."""
    best_s2 = get_best_from_stage("s2_")
    if not best_s2:
        print("[ERROR] Stage 2 not complete. Run --stage 2 first.")
        return
    res = best_s2["config"]["resolution"]
    depth = best_s2["config"]["depth"]
    lr = best_s2["config"]["fine_tune_lr"]
    print(f"\n{'='*70}")
    print(f"STAGE 3: HEAD ARCHITECTURE (res={res}, depth={depth}, lr={lr})")
    print(f"{'='*70}")

    X_train, y_train = load_split_data(res, "train")
    X_val, y_val = load_split_data(res, "val")

    configs = [
        ExperimentConfig(name="s3_head_A", resolution=res, depth=depth,
                         fine_tune_lr=lr, head_type="A"),
        ExperimentConfig(name="s3_head_B", resolution=res, depth=depth,
                         fine_tune_lr=lr, head_type="B"),
        ExperimentConfig(name="s3_head_C", resolution=res, depth=depth,
                         fine_tune_lr=lr, head_type="C"),
    ]

    existing = {r["name"] for r in load_results()}
    for cfg in configs:
        if cfg.name in existing:
            print(f"[SKIP] {cfg.name} already completed")
            continue
        result = run_experiment(cfg, X_train, y_train, X_val, y_val)
        save_result(result)

    del X_train, y_train, X_val, y_val
    gc.collect()

    results = [r for r in load_results() if r["name"].startswith("s3_")]
    print("\n--- Stage 3 Results ---")
    print_results_table(results)

    best = get_best_from_stage("s3_")
    if best:
        print(f">>> Best head: {best['config']['head_type']} "
              f"(val_acc={best['val_accuracy']:.4f})")


def run_stage_4():
    """Stage 4: Augmentation ablation."""
    best_s3 = get_best_from_stage("s3_")
    if not best_s3:
        print("[ERROR] Stage 3 not complete. Run --stage 3 first.")
        return
    res = best_s3["config"]["resolution"]
    depth = best_s3["config"]["depth"]
    lr = best_s3["config"]["fine_tune_lr"]
    head = best_s3["config"]["head_type"]
    print(f"\n{'='*70}")
    print(f"STAGE 4: AUGMENTATION (res={res}, depth={depth}, lr={lr}, head={head})")
    print(f"{'='*70}")

    X_train, y_train = load_split_data(res, "train")
    X_val, y_val = load_split_data(res, "val")

    configs = [
        ExperimentConfig(name="s4_aug_weak", resolution=res, depth=depth,
                         fine_tune_lr=lr, head_type=head, augmentation="weak"),
        ExperimentConfig(name="s4_aug_current", resolution=res, depth=depth,
                         fine_tune_lr=lr, head_type=head, augmentation="current"),
        ExperimentConfig(name="s4_aug_strong", resolution=res, depth=depth,
                         fine_tune_lr=lr, head_type=head, augmentation="strong"),
    ]

    existing = {r["name"] for r in load_results()}
    for cfg in configs:
        if cfg.name in existing:
            print(f"[SKIP] {cfg.name} already completed")
            continue
        result = run_experiment(cfg, X_train, y_train, X_val, y_val)
        save_result(result)

    del X_train, y_train, X_val, y_val
    gc.collect()

    results = [r for r in load_results() if r["name"].startswith("s4_")]
    print("\n--- Stage 4 Results ---")
    print_results_table(results)

    best = get_best_from_stage("s4_")
    if best:
        print(f">>> Best augmentation: {best['config']['augmentation']} "
              f"(val_acc={best['val_accuracy']:.4f})")


def run_stage_5():
    """Stage 5: Loss function."""
    best_s4 = get_best_from_stage("s4_")
    if not best_s4:
        print("[ERROR] Stage 4 not complete. Run --stage 4 first.")
        return
    res = best_s4["config"]["resolution"]
    depth = best_s4["config"]["depth"]
    lr = best_s4["config"]["fine_tune_lr"]
    head = best_s4["config"]["head_type"]
    aug = best_s4["config"]["augmentation"]
    print(f"\n{'='*70}")
    print(f"STAGE 5: LOSS FUNCTION (res={res}, depth={depth}, lr={lr}, "
          f"head={head}, aug={aug})")
    print(f"{'='*70}")

    X_train, y_train = load_split_data(res, "train")
    X_val, y_val = load_split_data(res, "val")

    configs = [
        ExperimentConfig(name="s5_loss_ce", resolution=res, depth=depth,
                         fine_tune_lr=lr, head_type=head, augmentation=aug,
                         loss_type="ce"),
        ExperimentConfig(name="s5_loss_wce", resolution=res, depth=depth,
                         fine_tune_lr=lr, head_type=head, augmentation=aug,
                         loss_type="weighted_ce"),
        ExperimentConfig(name="s5_loss_focal", resolution=res, depth=depth,
                         fine_tune_lr=lr, head_type=head, augmentation=aug,
                         loss_type="focal"),
    ]

    existing = {r["name"] for r in load_results()}
    for cfg in configs:
        if cfg.name in existing:
            print(f"[SKIP] {cfg.name} already completed")
            continue
        result = run_experiment(cfg, X_train, y_train, X_val, y_val)
        save_result(result)

    del X_train, y_train, X_val, y_val
    gc.collect()

    results = [r for r in load_results() if r["name"].startswith("s5_")]
    print("\n--- Stage 5 Results ---")
    print_results_table(results)

    best = get_best_from_stage("s5_")
    if best:
        print(f">>> Best loss: {best['config']['loss_type']} "
              f"(val_acc={best['val_accuracy']:.4f})")


def run_final():
    """Final: Multi-seed top 3 + locked test evaluation."""
    import tensorflow as tf
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                 confusion_matrix)

    # Gather all results, sort by (val_accuracy, val_macro_f1)
    all_results = load_results()
    if len(all_results) < 3:
        print("[ERROR] Need at least 3 completed experiments. Run stages 1-5 first.")
        return

    # Exclude multi-seed runs from ranking
    base_results = [r for r in all_results if not r["name"].startswith("mseed_")]
    ranked = sorted(base_results,
                    key=lambda r: (r["val_accuracy"], r["val_macro_f1"]),
                    reverse=True)

    print("\n" + "=" * 70)
    print("FINAL STAGE: MULTI-SEED TOP 3 + LOCKED TEST")
    print("=" * 70)
    print("\nTop 10 configs by validation accuracy:")
    print_results_table(ranked[:10])

    top3 = ranked[:3]
    print("Selected for multi-seed verification:")
    for r in top3:
        print(f"  {r['name']}: val_acc={r['val_accuracy']:.4f}")

    # Multi-seed: seeds 42 (already done), 123, 2026
    extra_seeds = [123, 2026]
    existing = {r["name"] for r in all_results}

    for r in top3:
        cfg_base = r["config"]
        res = cfg_base["resolution"]

        X_train, y_train = load_split_data(res, "train")
        X_val, y_val = load_split_data(res, "val")

        for seed in extra_seeds:
            name = f"mseed_{r['name']}_s{seed}"
            if name in existing:
                print(f"[SKIP] {name} already completed")
                continue

            cfg = ExperimentConfig(
                name=name,
                resolution=res,
                depth=cfg_base["depth"],
                fine_tune_lr=cfg_base["fine_tune_lr"],
                head_type=cfg_base["head_type"],
                augmentation=cfg_base["augmentation"],
                loss_type=cfg_base["loss_type"],
                dropout=cfg_base["dropout"],
                seed=seed
            )
            result = run_experiment(cfg, X_train, y_train, X_val, y_val)
            save_result(result)

        del X_train, y_train, X_val, y_val
        gc.collect()

    # Compute mean val_accuracy across seeds for each config
    all_results = load_results()
    print("\n--- Multi-Seed Results ---")
    for r in top3:
        base_name = r["name"]
        seed_results = [r]  # seed 42
        for seed in extra_seeds:
            ms_name = f"mseed_{base_name}_s{seed}"
            ms = [x for x in all_results if x["name"] == ms_name]
            if ms:
                seed_results.append(ms[0])

        accs = [x["val_accuracy"] for x in seed_results]
        f1s = [x["val_macro_f1"] for x in seed_results]
        print(f"\n{base_name}:")
        print(f"  Seeds: {[x['config']['seed'] for x in seed_results]}")
        print(f"  Val accuracies: {accs}")
        print(f"  Mean val_acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"  Mean macro_f1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    # Select final candidate: highest mean val_accuracy across seeds
    best_mean_acc = -1
    best_config_name = None
    best_seed42_result = None

    for r in top3:
        base_name = r["name"]
        accs = [r["val_accuracy"]]
        for seed in extra_seeds:
            ms = [x for x in all_results
                  if x["name"] == f"mseed_{base_name}_s{seed}"]
            if ms:
                accs.append(ms[0]["val_accuracy"])
        mean_acc = np.mean(accs)
        if mean_acc > best_mean_acc:
            best_mean_acc = mean_acc
            best_config_name = base_name
            best_seed42_result = r

    print(f"\n>>> FINAL CANDIDATE: {best_config_name} "
          f"(mean val_acc={best_mean_acc:.4f})")

    # Evaluate on locked test set
    cfg = best_seed42_result["config"]
    model_path = best_seed42_result["model_path"]

    print(f"\n--- Evaluating on LOCKED TEST SET ---")
    print(f"Model: {model_path}")

    X_test, y_test = load_split_data(cfg["resolution"], "test")
    model = tf.keras.models.load_model(model_path, compile=False)
    preds = model.predict(X_test, batch_size=32, verbose=0)
    y_pred = np.argmax(preds, axis=1)

    test_acc = float(accuracy_score(y_test, y_pred))
    m_p, m_r, m_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0)
    w_p, w_r, w_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0)
    pc_p, pc_r, pc_f1, pc_s = precision_recall_fscore_support(
        y_test, y_pred, labels=list(range(NUM_CLASSES)), zero_division=0)
    cm = confusion_matrix(y_test, y_pred,
                          labels=list(range(NUM_CLASSES))).tolist()

    per_class = {}
    for i, cls in enumerate(CLASSES):
        per_class[cls] = {
            "precision": round(float(pc_p[i]), 4),
            "recall": round(float(pc_r[i]), 4),
            "f1": round(float(pc_f1[i]), 4),
            "support": int(pc_s[i])
        }

    print(f"\n{'='*70}")
    print(f"LOCKED TEST EVALUATION")
    print(f"{'='*70}")
    print(f"Test Accuracy:      {test_acc*100:.2f}% ({int(test_acc*131)}/131)")
    print(f"Test Macro F1:      {float(m_f1):.4f}")
    print(f"Test Weighted F1:   {float(w_f1):.4f}")
    print(f"\nPer-Class:")
    for cls, m in per_class.items():
        print(f"  {cls:<18} P={m['precision']:.3f} R={m['recall']:.3f} "
              f"F1={m['f1']:.3f} (n={m['support']})")

    print(f"\n{'='*70}")
    print(f"COMPARISON vs BASELINE")
    print(f"{'='*70}")
    print(f"                    Baseline    New")
    print(f"Val Accuracy:       87.79%      {best_seed42_result['val_accuracy']*100:.2f}%")
    print(f"Test Accuracy:      85.50%      {test_acc*100:.2f}%")
    print(f"Test Macro F1:      0.8503      {float(m_f1):.4f}")

    improved = (test_acc > 0.8550 and
                best_seed42_result["val_accuracy"] > 0.8779)

    if improved:
        print(f"\n>>> NEW MODEL IMPROVES BOTH METRICS!")
        print(f">>> Replacing production model: {PRODUCTION_MODEL}")
        import shutil
        backup = PRODUCTION_MODEL.with_suffix(".keras.backup")
        shutil.copy2(PRODUCTION_MODEL, backup)
        shutil.copy2(model_path, PRODUCTION_MODEL)
        print(f">>> Backup saved to: {backup}")
        print(f">>> Production model updated!")
    else:
        print(f"\n>>> New model does NOT improve both metrics.")
        print(f">>> Keeping current production model (85.50% test accuracy).")

    # Save final report
    final_report = {
        "candidate": best_config_name,
        "candidate_config": cfg,
        "mean_val_accuracy_across_seeds": round(best_mean_acc, 4),
        "val_accuracy_seed42": best_seed42_result["val_accuracy"],
        "test_accuracy": round(test_acc, 4),
        "test_macro_f1": round(float(m_f1), 4),
        "test_weighted_f1": round(float(w_f1), 4),
        "test_per_class": per_class,
        "test_confusion_matrix": cm,
        "improved_over_baseline": improved,
        "baseline_val_acc": 0.8779,
        "baseline_test_acc": 0.8550,
        "model_path": model_path,
    }
    with open(EXPERIMENT_DIR / "final_report.json", "w") as f:
        json.dump(final_report, f, indent=2)
    print(f"\nFinal report saved to: {EXPERIMENT_DIR / 'final_report.json'}")


def run_summary():
    """Print summary of all completed experiments."""
    results = load_results()
    print(f"\n{'='*70}")
    print(f"SWEEP SUMMARY — {len(results)} experiments completed")
    print(f"{'='*70}")
    print_results_table(results)

    # Per-stage summaries
    for stage, prefix in [("1-Resolution", "s1_"), ("2-Depth×LR", "s2_"),
                          ("3-Head", "s3_"), ("4-Augmentation", "s4_"),
                          ("5-Loss", "s5_"), ("Multi-seed", "mseed_")]:
        stage_r = [r for r in results if r["name"].startswith(prefix)]
        if stage_r:
            best = max(stage_r, key=lambda r: r["val_accuracy"])
            print(f"Stage {stage}: {len(stage_r)} experiments, "
                  f"best={best['name']} ({best['val_accuracy']:.4f})")


# ── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="VGG16 Optimization Sweep")
    parser.add_argument("--stage", type=str, required=True,
                        choices=["1", "2", "3", "4", "5", "final", "summary"],
                        help="Which stage to run")
    args = parser.parse_args()

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    dispatch = {
        "1": run_stage_1,
        "2": run_stage_2,
        "3": run_stage_3,
        "4": run_stage_4,
        "5": run_stage_5,
        "final": run_final,
        "summary": run_summary,
    }
    dispatch[args.stage]()


if __name__ == "__main__":
    main()
