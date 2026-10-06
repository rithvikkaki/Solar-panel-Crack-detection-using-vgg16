"""
SolarSentinel AI - Systematic Scientific Investigation of High-Value VGG16 Improvements
Evaluates:
- A. Input Resolution (244x244 vs 320x320 vs 384x384)
- B. Spatial Information Pooling (GAP vs GMP vs GAP+GMP Concatenation)
- C. Fine-Tuning Depth (Block 5 vs Blocks 4-5 vs Blocks 3-5)
- D. Conservative Learning Rates (1e-4, 5e-5, 1e-5, 5e-6)
- E. Augmentation Ablation (None, Spatial Only, Photometric Only, Spatial+Photometric)
- F. Class Imbalance & Loss Formulations (Standard CE, Weighted CE, Label Smoothing, Focal Loss)
- G. High-Resolution Patch-Based Architecture (Zero Data Leakage: 5 patches per image, Mean vs Max vs Defect-Biased aggregation)
- H. Multi-Seed Stability (Seeds 42, 123, 2026 for top candidate models)
"""

import os
import sys
import json
import time
import gc
from typing import Dict, List, Tuple, Any
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import tensorflow as tf

MANIFEST_PATH = os.path.join(PROJECT_ROOT, "ml", "metadata", "split_manifest_70_15_15.json")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "ml", "experiments", "phase9_vgg16", "scientific_investigation_results.json")
CACHED_BOTTLENECKS_PATH = os.path.join(PROJECT_ROOT, "ml", "metadata", "cached_bottlenecks.npz")

with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    manifest = json.load(f)

classes = manifest["classes"]
train_files = manifest["train_files"]
val_files = manifest["val_files"]
test_files = manifest["test_files"]
num_classes = len(classes)
phys_idx = classes.index("Physical-Damage")


def compute_metrics(y_true, y_pred_probs, classes):
    yp = np.argmax(y_pred_probs, axis=1)
    acc = float(accuracy_score(y_true, yp))
    pm, rm, f1m, _ = precision_recall_fscore_support(y_true, yp, average="macro", zero_division=0)
    pw, rw, f1w, _ = precision_recall_fscore_support(y_true, yp, average="weighted", zero_division=0)
    pp, rp, f1p, sp = precision_recall_fscore_support(y_true, yp, average=None, labels=list(range(len(classes))), zero_division=0)
    
    per_cls = {}
    for i, c in enumerate(classes):
        per_cls[c] = {
            "precision": round(float(pp[i]), 4),
            "recall": round(float(rp[i]), 4),
            "f1": round(float(f1p[i]), 4),
            "support": int(sp[i])
        }
    
    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(float(f1m), 4),
        "weighted_f1": round(float(f1w), 4),
        "macro_precision": round(float(pm), 4),
        "macro_recall": round(float(rm), 4),
        "physical_damage_recall": round(float(rp[phys_idx]), 4),
        "physical_damage_precision": round(float(pp[phys_idx]), 4),
        "physical_damage_f1": round(float(f1p[phys_idx]), 4),
        "per_class": per_cls
    }


def get_class_weights(y_train, num_classes):
    counts = np.bincount(y_train, minlength=num_classes)
    total = len(y_train)
    weights = total / (num_classes * np.maximum(counts, 1).astype(np.float32))
    return dict(enumerate(weights))


def load_dataset_arrays(items, target_size=(244, 244)):
    images = []
    labels = []
    for it in items:
        p = it["path"]
        abs_p = os.path.join(PROJECT_ROOT, p) if not os.path.isabs(p) else p
        lbl = it["class_index"]
        with Image.open(abs_p) as raw_img:
            img = raw_img.convert("RGB").resize((target_size[1], target_size[0]), Image.Resampling.BILINEAR)
            images.append(np.array(img, dtype=np.float32))
            labels.append(lbl)
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)


# ==============================================================================
# MODULE B: Spatial Information Pooling (GAP vs GMP vs GAP+GMP)
# ==============================================================================
def run_spatial_pooling_experiment() -> Dict[str, Any]:
    print("\n" + "="*70)
    print("MODULE B: SPATIAL POOLING ABLATION (GAP vs GMP vs GAP+GMP)")
    print("="*70)
    
    data = np.load(CACHED_BOTTLENECKS_PATH)
    x_train = np.concatenate([data["x_train_clean"], data["x_train_aug"]], axis=0)
    y_train = np.concatenate([data["y_train"], data["y_train"]], axis=0)
    x_val = data["x_val"]
    y_val = data["y_val"]
    
    results = {}
    for pt in ["gap", "gmp", "gap_gmp_concat"]:
        tf.keras.backend.clear_session()
        tf.random.set_seed(42)
        np.random.seed(42)
        
        inp = tf.keras.layers.Input(shape=(7, 7, 512), name="bottleneck_input")
        if pt == "gap":
            pooled = tf.keras.layers.GlobalAveragePooling2D(name="gap")(inp)
        elif pt == "gmp":
            pooled = tf.keras.layers.GlobalMaxPooling2D(name="gmp")(inp)
        elif pt == "gap_gmp_concat":
            p1 = tf.keras.layers.GlobalAveragePooling2D(name="gap")(inp)
            p2 = tf.keras.layers.GlobalMaxPooling2D(name="gmp")(inp)
            pooled = tf.keras.layers.Concatenate(name="concat_pool")([p1, p2])
            
        x = tf.keras.layers.BatchNormalization(name="bn_pool")(pooled)
        x = tf.keras.layers.Dense(512, activation="relu", name="dense_512")(x)
        x = tf.keras.layers.Dropout(0.3, name="dropout_1")(x)
        x = tf.keras.layers.Dense(256, activation="relu", name="dense_256")(x)
        x = tf.keras.layers.Dropout(0.2, name="dropout_2")(x)
        out = tf.keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)
        
        model = tf.keras.Model(inputs=inp, outputs=out, name=f"head_{pt}")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        lr_sched = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)
        
        t0 = time.time()
        model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=20,
            batch_size=32,
            callbacks=[lr_sched],
            verbose=0
        )
        elapsed = round(time.time() - t0, 2)
        
        preds = model.predict(x_val, verbose=0)
        m = compute_metrics(y_val, preds, classes)
        m["training_time_sec"] = elapsed
        m["pooling_type"] = pt
        results[pt] = m
        print(f"  [{pt:<15}] Val Acc: {m['accuracy']*100:.2f}% | Macro F1: {m['macro_f1']:.4f} | Phys Rec: {m['physical_damage_recall']*100:.1f}% (Time: {elapsed}s)")
        
    return results


# ==============================================================================
# MODULE F: Loss Function & Class Imbalance Ablation
# ==============================================================================
def run_loss_formulation_experiment() -> Dict[str, Any]:
    print("\n" + "="*70)
    print("MODULE F: LOSS FORMULATION & CLASS IMBALANCE ABLATION")
    print("="*70)
    
    data = np.load(CACHED_BOTTLENECKS_PATH)
    x_train = np.concatenate([data["x_train_clean"], data["x_train_aug"]], axis=0)
    y_train = np.concatenate([data["y_train"], data["y_train"]], axis=0)
    x_val = data["x_val"]
    y_val = data["y_val"]
    
    cw = get_class_weights(y_train, num_classes)
    results = {}
    
    for l_name in ["standard_ce", "weighted_ce", "label_smoothing_05", "focal_loss_gamma2"]:
        tf.keras.backend.clear_session()
        tf.random.set_seed(42)
        np.random.seed(42)
        
        inp = tf.keras.layers.Input(shape=(7, 7, 512))
        pooled = tf.keras.layers.GlobalAveragePooling2D()(inp)
        x = tf.keras.layers.BatchNormalization()(pooled)
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
        
        model = tf.keras.Model(inputs=inp, outputs=out)
        
        if l_name == "standard_ce":
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
            fit_cw = None
        elif l_name == "weighted_ce":
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
            fit_cw = cw
        elif l_name == "label_smoothing_05":
            loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
            fit_cw = None
        elif l_name == "focal_loss_gamma2":
            class FocalLoss(tf.keras.losses.Loss):
                def __init__(self, gamma=2.0, alpha=0.25):
                    super().__init__()
                    self.gamma = gamma
                    self.alpha = alpha
                def call(self, y_true, y_pred):
                    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
                    y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes)
                    cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
                    weight = self.alpha * tf.math.pow(1.0 - y_pred, self.gamma)
                    return tf.reduce_mean(tf.reduce_sum(weight * cross_entropy, axis=-1))
            loss_fn = FocalLoss(gamma=2.0)
            fit_cw = None
            
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=loss_fn,
            metrics=["accuracy"]
        )
        
        lr_sched = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)
        
        if l_name == "label_smoothing_05":
            y_tr_in = tf.keras.utils.to_categorical(y_train, num_classes)
            y_va_in = tf.keras.utils.to_categorical(y_val, num_classes)
        else:
            y_tr_in = y_train
            y_va_in = y_val
            
        t0 = time.time()
        model.fit(
            x_train, y_tr_in,
            validation_data=(x_val, y_va_in),
            epochs=20,
            batch_size=32,
            class_weight=fit_cw,
            callbacks=[lr_sched],
            verbose=0
        )
        elapsed = round(time.time() - t0, 2)
        
        preds = model.predict(x_val, verbose=0)
        m = compute_metrics(y_val, preds, classes)
        m["training_time_sec"] = elapsed
        m["loss_formulation"] = l_name
        results[l_name] = m
        print(f"  [{l_name:<20}] Val Acc: {m['accuracy']*100:.2f}% | Macro F1: {m['macro_f1']:.4f} | Phys Rec: {m['physical_damage_recall']*100:.1f}% (Time: {elapsed}s)")
        
    return results


# ==============================================================================
# MODULE A: Input Resolution Comparison (244x244 vs 320x320 vs 384x384)
# ==============================================================================
def run_resolution_experiment() -> Dict[str, Any]:
    print("\n" + "="*70)
    print("MODULE A: INPUT RESOLUTION COMPARISON (244x244 vs 320x320 vs 384x384)")
    print("="*70)
    
    resolutions = [244, 320, 384]
    results = {}
    
    for res in resolutions:
        tf.keras.backend.clear_session()
        tf.random.set_seed(42)
        np.random.seed(42)
        
        print(f"\nEvaluating Resolution {res}x{res}...")
        t_load = time.time()
        x_tr, y_tr = load_dataset_arrays(train_files, target_size=(res, res))
        x_va, y_va = load_dataset_arrays(val_files, target_size=(res, res))
        load_time = time.time() - t_load
        
        # Build VGG16 backbone feature extractor
        inp_vgg = tf.keras.layers.Input(shape=(res, res, 3))
        x_prep = tf.keras.applications.vgg16.preprocess_input(inp_vgg)
        base = tf.keras.applications.VGG16(include_top=False, weights="imagenet", input_shape=(res, res, 3))
        base.trainable = False
        feat_out = base(x_prep, training=False)
        extractor = tf.keras.Model(inputs=inp_vgg, outputs=feat_out)
        
        # Measure feature extraction time and output shape
        t_feat = time.time()
        feat_tr = extractor.predict(x_tr, batch_size=32, verbose=0)
        feat_va = extractor.predict(x_va, batch_size=32, verbose=0)
        feat_time = round(time.time() - t_feat, 2)
        
        feat_shape = feat_tr.shape[1:]
        mem_mb = round(float(feat_tr.nbytes + feat_va.nbytes) / (1024 * 1024), 2)
        print(f"  Feature shape: {feat_shape} | Bottleneck Tensor RAM: {mem_mb} MB | Feature extract: {feat_time}s")
        
        # Train Head B on top of the feature representations
        head_inp = tf.keras.layers.Input(shape=feat_shape)
        p = tf.keras.layers.GlobalAveragePooling2D()(head_inp)
        bn = tf.keras.layers.BatchNormalization()(p)
        d1 = tf.keras.layers.Dense(512, activation="relu")(bn)
        dp1 = tf.keras.layers.Dropout(0.3)(d1)
        d2 = tf.keras.layers.Dense(256, activation="relu")(dp1)
        dp2 = tf.keras.layers.Dropout(0.2)(d2)
        out = tf.keras.layers.Dense(num_classes, activation="softmax")(dp2)
        
        head_model = tf.keras.Model(inputs=head_inp, outputs=out)
        head_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        lr_sched = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)
        
        t_tr = time.time()
        head_model.fit(
            feat_tr, y_tr,
            validation_data=(feat_va, y_va),
            epochs=20,
            batch_size=32,
            callbacks=[lr_sched],
            verbose=0
        )
        train_time = round(time.time() - t_tr, 2)
        
        preds = head_model.predict(feat_va, verbose=0)
        m = compute_metrics(y_va, preds, classes)
        m["resolution"] = f"{res}x{res}"
        m["feature_shape"] = list(feat_shape)
        m["tensor_memory_mb"] = mem_mb
        m["feature_extract_sec"] = feat_time
        m["head_train_sec"] = train_time
        
        results[f"res_{res}"] = m
        print(f"  [{res}x{res}] Val Acc: {m['accuracy']*100:.2f}% | Macro F1: {m['macro_f1']:.4f} | Phys Rec: {m['physical_damage_recall']*100:.1f}% | Mem: {mem_mb}MB | Time: {train_time}s")
        
        del x_tr, x_va, feat_tr, feat_va, extractor, head_model
        gc.collect()
        
    return results


# ==============================================================================
# MODULE E: Augmentation Ablation (None, Spatial, Photometric, Both)
# ==============================================================================
def run_augmentation_experiment() -> Dict[str, Any]:
    print("\n" + "="*70)
    print("MODULE E: AUGMENTATION ABLATION (None vs Spatial vs Photometric vs Both)")
    print("="*70)
    
    # Load 244x244 raw arrays
    x_tr, y_tr = load_dataset_arrays(train_files, target_size=(244, 244))
    x_va, y_va = load_dataset_arrays(val_files, target_size=(244, 244))
    
    aug_schemes = ["none", "spatial_only", "photometric_only", "spatial_and_photometric"]
    results = {}
    
    for aug_mode in aug_schemes:
        tf.keras.backend.clear_session()
        tf.random.set_seed(42)
        np.random.seed(42)
        
        layers = []
        if aug_mode in ["spatial_only", "spatial_and_photometric"]:
            layers.extend([
                tf.keras.layers.RandomFlip("horizontal_and_vertical"),
                tf.keras.layers.RandomRotation(0.08),
                tf.keras.layers.RandomZoom((-0.08, 0.08))
            ])
        if aug_mode in ["photometric_only", "spatial_and_photometric"]:
            layers.extend([
                tf.keras.layers.RandomBrightness(0.08, value_range=(0, 255)),
                tf.keras.layers.RandomContrast(0.08)
            ])
            
        aug_seq = tf.keras.Sequential(layers, name=f"aug_{aug_mode}") if layers else None
        
        inp = tf.keras.layers.Input(shape=(244, 244, 3))
        x = aug_seq(inp) if aug_seq else inp
        x = tf.keras.applications.vgg16.preprocess_input(x)
        base = tf.keras.applications.VGG16(include_top=False, weights="imagenet", input_shape=(244, 244, 3))
        base.trainable = False
        x = base(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
        
        model = tf.keras.Model(inputs=inp, outputs=out)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        lr_sched = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)
        
        t0 = time.time()
        model.fit(
            x_tr, y_tr,
            validation_data=(x_va, y_va),
            epochs=15,
            batch_size=32,
            callbacks=[lr_sched],
            verbose=0
        )
        elapsed = round(time.time() - t0, 2)
        
        preds = model.predict(x_va, verbose=0)
        m = compute_metrics(y_va, preds, classes)
        m["aug_mode"] = aug_mode
        m["training_time_sec"] = elapsed
        results[aug_mode] = m
        print(f"  [{aug_mode:<25}] Val Acc: {m['accuracy']*100:.2f}% | Macro F1: {m['macro_f1']:.4f} | Phys Rec: {m['physical_damage_recall']*100:.1f}% (Time: {elapsed}s)")
        
    del x_tr, x_va
    gc.collect()
    return results


# ==============================================================================
# MODULE C & D: Fine-Tuning Depth & Learning Rates
# ==============================================================================
def run_finetuning_depth_and_lr_experiment() -> Dict[str, Any]:
    print("\n" + "="*70)
    print("MODULE C & D: FINE-TUNING DEPTH & CONSERVATIVE LEARNING RATES")
    print("="*70)
    
    x_tr, y_tr = load_dataset_arrays(train_files, target_size=(244, 244))
    x_va, y_va = load_dataset_arrays(val_files, target_size=(244, 244))
    
    # Test Block 5, Block 4-5, Block 3-5 with learning rates [1e-4, 5e-5, 1e-5, 5e-6]
    depth_configs = [
        ("block5", ["block5_conv1", "block5_conv2", "block5_conv3"]),
        ("block4_5", ["block4_conv1", "block4_conv2", "block4_conv3", "block5_conv1", "block5_conv2", "block5_conv3"]),
        ("block3_4_5", ["block3_conv1", "block3_conv2", "block3_conv3", "block4_conv1", "block4_conv2", "block4_conv3", "block5_conv1", "block5_conv2", "block5_conv3"])
    ]
    
    lrs = [1e-4, 5e-5, 1e-5, 5e-6]
    results = {}
    
    for depth_name, trainable_layers in depth_configs:
        for lr in lrs:
            exp_key = f"{depth_name}_lr_{lr}"
            tf.keras.backend.clear_session()
            tf.random.set_seed(42)
            np.random.seed(42)
            
            inp = tf.keras.layers.Input(shape=(244, 244, 3))
            x = tf.keras.applications.vgg16.preprocess_input(inp)
            base = tf.keras.applications.VGG16(include_top=False, weights="imagenet", input_shape=(244, 244, 3))
            
            # Freeze all layers except designated
            for l in base.layers:
                l.trainable = (l.name in trainable_layers)
                
            x = base(x)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dense(512, activation="relu")(x)
            x = tf.keras.layers.Dropout(0.3)(x)
            x = tf.keras.layers.Dense(256, activation="relu")(x)
            x = tf.keras.layers.Dropout(0.2)(x)
            out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
            
            model = tf.keras.Model(inputs=inp, outputs=out)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )
            
            lr_sched = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7)
            
            t0 = time.time()
            model.fit(
                x_tr, y_tr,
                validation_data=(x_va, y_va),
                epochs=12,
                batch_size=32,
                callbacks=[lr_sched],
                verbose=0
            )
            elapsed = round(time.time() - t0, 2)
            
            preds = model.predict(x_va, verbose=0)
            m = compute_metrics(y_va, preds, classes)
            m["depth"] = depth_name
            m["learning_rate"] = lr
            m["training_time_sec"] = elapsed
            results[exp_key] = m
            print(f"  [{depth_name:<12} | lr={lr:.1e}] Val Acc: {m['accuracy']*100:.2f}% | Macro F1: {m['macro_f1']:.4f} | Phys Rec: {m['physical_damage_recall']*100:.1f}% (Time: {elapsed}s)")
            
    del x_tr, x_va
    gc.collect()
    return results


# ==============================================================================
# MODULE G: High-Resolution Patch Pipeline (Zero Data Leakage)
# ==============================================================================
def run_highres_patch_experiment() -> Dict[str, Any]:
    print("\n" + "="*70)
    print("MODULE G: HIGH-RESOLUTION PATCH EXPERIMENT (ZERO DATA LEAKAGE)")
    print("="*70)
    
    # Load base model (best_vgg16.keras) for patch inference
    base_model_path = os.path.join(PROJECT_ROOT, "ml", "models", "best_vgg16.keras")
    model = tf.keras.models.load_model(base_model_path, compile=False)
    
    def extract_image_patches(image_path: str, patch_size=(244, 244)) -> List[np.ndarray]:
        """Extracts 5 multi-scale representations: 1 full panel resized + 4 corner crops + 1 center crop."""
        with Image.open(image_path) as raw_img:
            img = raw_img.convert("RGB")
            w, h = img.size
            
            patches = []
            # 1. Global context
            patches.append(np.array(img.resize(patch_size, Image.Resampling.BILINEAR), dtype=np.float32))
            
            # High-res crops if image is large enough
            if w >= 350 and h >= 350:
                cw, ch = int(w * 0.5), int(h * 0.5)
                crops = [
                    (0, 0, cw, ch),                        # Top-left
                    (w - cw, 0, w, ch),                    # Top-right
                    (0, h - ch, cw, h),                    # Bottom-left
                    (w - cw, h - ch, w, h),                # Bottom-right
                    ((w - cw)//2, (h - ch)//2, (w + cw)//2, (h + ch)//2) # Center
                ]
                for box in crops:
                    cropped = img.crop(box).resize(patch_size, Image.Resampling.BILINEAR)
                    patches.append(np.array(cropped, dtype=np.float32))
            else:
                # Fallback: replicate global view
                for _ in range(5):
                    patches.append(patches[0].copy())
                    
            return patches[:6] # Return global + 5 crops
            
    print(f"Extracting multi-scale high-res patches for {len(val_files)} validation images...")
    
    val_y = []
    val_patch_preds = []
    
    t0 = time.time()
    for it in val_files:
        p = it["path"]
        abs_p = os.path.join(PROJECT_ROOT, p) if not os.path.isabs(p) else p
        val_y.append(it["class_index"])
        
        patches = extract_image_patches(abs_p, patch_size=(244, 244))
        patch_tensor = np.array(patches, dtype=np.float32)
        # Forward pass on all patches of this image
        preds = model.predict(patch_tensor, verbose=0)
        val_patch_preds.append(preds)
        
    eval_time = round(time.time() - t0, 2)
    val_y = np.array(val_y, dtype=np.int32)
    
    # Test Aggregation Schemes:
    results = {}
    
    # 1. Global Only (Baseline)
    global_probs = np.array([preds[0] for preds in val_patch_preds])
    m_glob = compute_metrics(val_y, global_probs, classes)
    results["global_view_baseline"] = m_glob
    print(f"  [Global View Baseline]        Val Acc: {m_glob['accuracy']*100:.2f}% | Macro F1: {m_glob['macro_f1']:.4f} | Phys Rec: {m_glob['physical_damage_recall']*100:.1f}%")
    
    # 2. Mean Probability Aggregation across all patches
    mean_probs = np.array([np.mean(preds, axis=0) for preds in val_patch_preds])
    m_mean = compute_metrics(val_y, mean_probs, classes)
    results["mean_patch_aggregation"] = m_mean
    print(f"  [Mean Patch Aggregation]      Val Acc: {m_mean['accuracy']*100:.2f}% | Macro F1: {m_mean['macro_f1']:.4f} | Phys Rec: {m_mean['physical_damage_recall']*100:.1f}%")
    
    # 3. Defect-Biased Max Aggregation (Weighting Physical-Damage or Electrical peak probabilities)
    defect_probs = []
    for preds in val_patch_preds:
        # Check if any high-res crop shows strong Physical-Damage or Electrical-damage
        crop_preds = preds[1:] # crops only
        max_crop_phys = np.max(crop_preds[:, phys_idx])
        if max_crop_phys > 0.45: # High localized crack confidence detected in a sub-tile
            # Boost physical damage posterior
            blended = 0.5 * preds[0] + 0.5 * np.max(crop_preds, axis=0)
            blended = blended / np.sum(blended)
        else:
            blended = 0.7 * preds[0] + 0.3 * np.mean(crop_preds, axis=0)
            blended = blended / np.sum(blended)
        defect_probs.append(blended)
        
    defect_probs = np.array(defect_probs)
    m_defect = compute_metrics(val_y, defect_probs, classes)
    results["defect_biased_aggregation"] = m_defect
    print(f"  [Defect-Biased Aggregation]   Val Acc: {m_defect['accuracy']*100:.2f}% | Macro F1: {m_defect['macro_f1']:.4f} | Phys Rec: {m_defect['physical_damage_recall']*100:.1f}%")
    
    return results


# ==============================================================================
# MODULE H: Multi-Seed Stability (Seeds 42, 123, 2026) for Top Configurations
# ==============================================================================
def run_multiseed_stability_experiment() -> Dict[str, Any]:
    print("\n" + "="*70)
    print("MODULE H: MULTI-SEED STABILITY EXPERIMENT (SEEDS 42, 123, 2026)")
    print("="*70)
    
    data = np.load(CACHED_BOTTLENECKS_PATH)
    x_train = np.concatenate([data["x_train_clean"], data["x_train_aug"]], axis=0)
    y_train = np.concatenate([data["y_train"], data["y_train"]], axis=0)
    x_val = data["x_val"]
    y_val = data["y_val"]
    
    seeds = [42, 123, 2026]
    configs = [
        ("Head_B_TwoLayer_StandardCE", "gap", 0.3, "standard_ce"),
        ("Head_B_TwoLayer_WeightedCE", "gap", 0.3, "weighted_ce"),
        ("GAP_Dense_256_Dropout05", "gap_dense_256", 0.5, "standard_ce"),
        ("GAP_BN_Dense_512_WeightedCE", "gap_bn_dense_512", 0.4, "weighted_ce")
    ]
    
    cw = get_class_weights(y_train, num_classes)
    results = {}
    
    for cfg_name, arch, drop, loss_type in configs:
        acc_list = []
        f1_list = []
        phys_rec_list = []
        
        for s in seeds:
            tf.keras.backend.clear_session()
            tf.random.set_seed(s)
            np.random.seed(s)
            
            inp = tf.keras.layers.Input(shape=(7, 7, 512))
            if arch == "gap":
                p = tf.keras.layers.GlobalAveragePooling2D()(inp)
                bn = tf.keras.layers.BatchNormalization()(p)
                d1 = tf.keras.layers.Dense(512, activation="relu")(bn)
                dp1 = tf.keras.layers.Dropout(drop)(d1)
                d2 = tf.keras.layers.Dense(256, activation="relu")(dp1)
                dp2 = tf.keras.layers.Dropout(drop * 0.67)(d2)
                out = tf.keras.layers.Dense(num_classes, activation="softmax")(dp2)
            elif arch == "gap_dense_256":
                p = tf.keras.layers.GlobalAveragePooling2D()(inp)
                d = tf.keras.layers.Dense(256, activation="relu")(p)
                dp = tf.keras.layers.Dropout(drop)(d)
                out = tf.keras.layers.Dense(num_classes, activation="softmax")(dp)
            elif arch == "gap_bn_dense_512":
                p = tf.keras.layers.GlobalAveragePooling2D()(inp)
                bn = tf.keras.layers.BatchNormalization()(p)
                d = tf.keras.layers.Dense(512, activation="relu")(bn)
                dp = tf.keras.layers.Dropout(drop)(d)
                out = tf.keras.layers.Dense(num_classes, activation="softmax")(dp)
                
            model = tf.keras.Model(inputs=inp, outputs=out)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )
            
            fit_cw = cw if loss_type == "weighted_ce" else None
            lr_sched = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)
            
            model.fit(
                x_train, y_train,
                validation_data=(x_val, y_val),
                epochs=20,
                batch_size=32,
                class_weight=fit_cw,
                callbacks=[lr_sched],
                verbose=0
            )
            
            preds = model.predict(x_val, verbose=0)
            m = compute_metrics(y_val, preds, classes)
            acc_list.append(m["accuracy"])
            f1_list.append(m["macro_f1"])
            phys_rec_list.append(m["physical_damage_recall"])
            
        mean_acc = round(float(np.mean(acc_list)), 4)
        std_acc = round(float(np.std(acc_list)), 4)
        mean_f1 = round(float(np.mean(f1_list)), 4)
        std_f1 = round(float(np.std(f1_list)), 4)
        mean_phys = round(float(np.mean(phys_rec_list)), 4)
        std_phys = round(float(np.std(phys_rec_list)), 4)
        
        results[cfg_name] = {
            "mean_val_accuracy": mean_acc,
            "std_val_accuracy": std_acc,
            "mean_macro_f1": mean_f1,
            "std_macro_f1": std_f1,
            "mean_physical_recall": mean_phys,
            "std_physical_recall": std_phys,
            "raw_accuracies": acc_list,
            "raw_macro_f1s": f1_list
        }
        print(f"  [{cfg_name:<30}] Acc: {mean_acc*100:5.2f}% +/- {std_acc*100:4.2f}% | Macro F1: {mean_f1:.4f} +/- {std_f1:.4f} | Phys Rec: {mean_phys*100:5.2f}% +/- {std_phys*100:4.2f}%")
        
    return results


# ==============================================================================
# MAIN ORCHESTRATOR
# ==============================================================================
def main():
    print("="*80)
    print("STARTING SYSTEMATIC INVESTIGATION OF HIGH-VALUE VGG16 IMPROVEMENTS")
    print("="*80)
    
    total_start = time.time()
    
    all_results = {
        "baseline_selected_model": {
            "model_id": "exp2_head_b",
            "val_accuracy": 0.8779,
            "val_macro_f1": 0.8943,
            "val_weighted_f1": 0.8767,
            "physical_damage_recall": 0.9000,
            "physical_damage_f1": 0.9474,
            "locked_test_accuracy": 0.8550,
            "locked_test_macro_f1": 0.8503
        }
    }
    
    # 1. Spatial pooling ablation
    all_results["spatial_pooling"] = run_spatial_pooling_experiment()
    
    # 2. Loss formulation ablation
    all_results["loss_formulation"] = run_loss_formulation_experiment()
    
    # 3. Input resolution comparison
    all_results["resolution_comparison"] = run_resolution_experiment()
    
    # 4. Augmentation ablation
    all_results["augmentation_ablation"] = run_augmentation_experiment()
    
    # 5. Fine-tuning depth & learning rates
    all_results["finetuning_depth_and_lr"] = run_finetuning_depth_and_lr_experiment()
    
    # 6. High-res patch pipeline
    all_results["highres_patch_pipeline"] = run_highres_patch_experiment()
    
    # 7. Multi-seed stability
    all_results["multiseed_stability"] = run_multiseed_stability_experiment()
    
    # Check if ANY new model beat 87.79% val acc AND 0.8943 macro F1
    best_new_acc = 0.0
    best_new_f1 = 0.0
    best_new_key = None
    
    for section, exps in all_results.items():
        if isinstance(exps, dict):
            for k, res in exps.items():
                if isinstance(res, dict) and "accuracy" in res and "macro_f1" in res:
                    acc = res["accuracy"]
                    f1 = res["macro_f1"]
                    if acc > best_new_acc or (acc == best_new_acc and f1 > best_new_f1):
                        best_new_acc = acc
                        best_new_f1 = f1
                        best_new_key = f"{section}::{k}"
                        
    all_results["summary"] = {
        "baseline_val_accuracy": 0.8779,
        "baseline_val_macro_f1": 0.8943,
        "best_new_val_accuracy": best_new_acc,
        "best_new_val_macro_f1": best_new_f1,
        "best_new_configuration": best_new_key,
        "strictly_exceeds_baseline": bool(best_new_acc > 0.8779 and best_new_f1 > 0.8943),
        "total_runtime_minutes": round((time.time() - total_start) / 60.0, 2)
    }
    
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
        
    print("\n" + "="*80)
    print("INVESTIGATION COMPLETE - SUMMARY")
    print("="*80)
    print(f"Baseline Validation: Acc: 87.79% | Macro F1: 0.8943")
    print(f"Best New Candidate:   {best_new_key}")
    print(f"Best New Validation: Acc: {best_new_acc*100:.2f}% | Macro F1: {best_new_f1:.4f}")
    print(f"Strictly Exceeds Baseline: {all_results['summary']['strictly_exceeds_baseline']}")
    print(f"Total Runtime: {all_results['summary']['total_runtime_minutes']} minutes")
    print(f"Results saved to: {RESULTS_PATH}")
    print("="*80)

if __name__ == "__main__":
    main()
