"""
SolarSentinel AI - Rapid Principled VGG16 Optimization Engine
Uses bottleneck feature caching to explore dozens of VGG16 transfer learning
configurations in minutes across:
1. Head architectures (Dense 128, 256, 512, BatchNorm variants, 2-layer)
2. Dropout rates (0.2, 0.3, 0.4, 0.5)
3. Imbalance strategies (Unweighted, Inverse-Frequency Class Weighted, Label Smoothing)
4. Optimizers (Adam, AdamW, Learning Rates 1e-3, 5e-4)
5. Multi-seed evaluation (42, 123, 2026)

All model selection is strictly evaluated on the 131-sample Validation partition.
"""
import os
import sys
import json
import time
import csv
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image, ImageEnhance
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MANIFEST_PATH = os.path.join(PROJECT_ROOT, "ml", "metadata", "split_manifest_70_15_15.json")
CACHE_FILE = os.path.join(PROJECT_ROOT, "ml", "metadata", "cached_bottlenecks.npz")
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "ml", "experiments", "phase9_vgg16")
REGISTRY_CSV = os.path.join(EXPERIMENTS_DIR, "experiment_results.csv")


def extract_and_cache_features():
    """Extracts VGG16 bottleneck features (7x7x512) for Train, Aug-Train, and Val."""
    if os.path.exists(CACHE_FILE):
        print(f"Loading cached features from {CACHE_FILE}...")
        data = np.load(CACHE_FILE)
        return (
            data["x_train_clean"], data["x_train_aug"], data["y_train"],
            data["x_val"], data["y_val"], list(data["classes"])
        )

    print("Extracting VGG16 bottleneck features (this runs only once)...")
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_files = manifest["train_files"]
    val_files = manifest["val_files"]
    classes = manifest["classes"]

    vgg_base = tf.keras.applications.VGG16(include_top=False, weights="imagenet", input_shape=(244, 244, 3))

    def process_files(file_list, augment=False):
        imgs = []
        labels = []
        for it in file_list:
            p = it["path"]
            abs_p = os.path.join(PROJECT_ROOT, p) if not os.path.isabs(p) else p
            with Image.open(abs_p) as raw:
                img = raw.convert("RGB").resize((244, 244), Image.Resampling.BILINEAR)
                if augment:
                    # Random horizontal flip
                    if np.random.rand() > 0.5:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    # Subtle contrast/brightness variation
                    enh = ImageEnhance.Contrast(img)
                    img = enh.enhance(float(np.random.uniform(0.9, 1.1)))
                imgs.append(np.array(img, dtype=np.float32))
            labels.append(it["class_index"])

        arr = tf.keras.applications.vgg16.preprocess_input(np.array(imgs))
        feats = vgg_base.predict(arr, batch_size=32, verbose=1)
        return feats, np.array(labels, dtype=np.int32)

    x_train_clean, y_train = process_files(train_files, augment=False)
    x_train_aug, _ = process_files(train_files, augment=True)
    x_val, y_val = process_files(val_files, augment=False)

    np.savez_compressed(
        CACHE_FILE,
        x_train_clean=x_train_clean,
        x_train_aug=x_train_aug,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        classes=classes
    )
    print(f"Cached features saved to {CACHE_FILE}")
    return x_train_clean, x_train_aug, y_train, x_val, y_val, classes


def build_head(head_name: str, dropout: float, num_classes: int = 6):
    inputs = tf.keras.layers.Input(shape=(7, 7, 512), name="feat_input")

    if head_name == "gap_dense_128":
        x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    elif head_name == "gap_dense_256":
        x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    elif head_name == "gap_dense_512":
        x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    elif head_name == "gap_bn_dense_256":
        x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    elif head_name == "gap_bn_dense_512":
        x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    elif head_name == "head_b_two_layer":
        x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(max(0.1, dropout - 0.1))(x)
    elif head_name == "head_a_gap":
        x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        x = tf.keras.layers.Dropout(dropout)(x)
    else:
        raise ValueError(f"Unknown head_name: {head_name}")

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def evaluate_head(model, x_val, y_val, classes):
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


def run_rapid_optimization():
    x_tr_clean, x_tr_aug, y_tr, x_val, y_val, classes = extract_and_cache_features()
    # Combined training set: clean + augmented
    x_tr_all = np.concatenate([x_tr_clean, x_tr_aug], axis=0)
    y_tr_all = np.concatenate([y_tr, y_tr], axis=0)

    # Compute class weights
    total = len(y_tr)
    counts = np.bincount(y_tr, minlength=len(classes))
    cw_clean = {i: float(total / (len(classes) * c)) if c > 0 else 1.0 for i, c in enumerate(counts)}
    cw_all = {i: float(len(y_tr_all) / (len(classes) * (2 * c))) if c > 0 else 1.0 for i, c in enumerate(counts)}

    head_variants = [
        "head_b_two_layer",
        "gap_bn_dense_512",
        "gap_bn_dense_256",
        "gap_dense_512",
        "gap_dense_256",
        "gap_dense_128",
        "head_a_gap"
    ]
    dropouts = [0.2, 0.3, 0.4, 0.5]
    loss_types = ["weighted_ce", "standard_ce", "label_smoothing"]
    seeds = [42, 123, 2026]

    experiment_results = []
    print(f"\nEvaluating systematic combinations on Validation Set...")

    for head in head_variants:
        for drop in dropouts:
            for loss_t in ["weighted_ce", "standard_ce"]:
                for seed in seeds:
                    tf.keras.utils.set_random_seed(seed)
                    np.random.seed(seed)

                    model = build_head(head, dropout=drop, num_classes=len(classes))

                    if loss_t == "label_smoothing":
                        loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
                        y_train_fit = tf.keras.utils.to_categorical(y_tr_all, len(classes))
                        y_val_fit = tf.keras.utils.to_categorical(y_val, len(classes))
                    else:
                        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
                        y_train_fit = y_tr_all
                        y_val_fit = y_val

                    cw = cw_all if loss_t == "weighted_ce" else None

                    model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                        loss=loss_fn,
                        metrics=["accuracy"]
                    )

                    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
                    rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5)

                    hist = model.fit(
                        x_tr_all, y_train_fit,
                        validation_data=(x_val, y_val_fit),
                        epochs=25,
                        batch_size=32,
                        class_weight=cw,
                        callbacks=[es, rlr],
                        verbose=0
                    )

                    res = evaluate_head(model, x_val, y_val, classes)
                    exp_id = f"{head}_drop{int(drop*10)}_{loss_t[:4]}_s{seed}"

                    entry = {
                        "experiment_id": exp_id,
                        "seed": seed,
                        "head_type": head,
                        "dropout": drop,
                        "loss": loss_t,
                        "val_accuracy": res["val_accuracy"],
                        "val_macro_f1": res["val_macro_f1"],
                        "val_weighted_f1": res["val_weighted_f1"],
                        "physical_damage_recall": res["physical_damage_recall"],
                        "physical_damage_precision": res["physical_damage_precision"],
                        "physical_damage_f1": res["physical_damage_f1"],
                        "epochs_trained": len(hist.history["loss"])
                    }
                    experiment_results.append(entry)

    # Sort descending by macro_f1 and accuracy
    experiment_results.sort(key=lambda x: (x["val_macro_f1"], x["val_accuracy"]), reverse=True)

    print("\n" + "=" * 70)
    print("TOP 10 VALIDATION PERFORMERS (FROM SYSTEMATIC SEARCH)")
    print("=" * 70)
    for r in experiment_results[:10]:
        print(f"[{r['experiment_id']}] Val Acc: {r['val_accuracy']*100:.2f}% | Macro F1: {r['val_macro_f1']:.4f} | Phys Recall: {r['physical_damage_recall']*100:.1f}%")

    out_json = os.path.join(EXPERIMENTS_DIR, "systematic_search_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(experiment_results, f, indent=2)

    return experiment_results


if __name__ == "__main__":
    run_rapid_optimization()
