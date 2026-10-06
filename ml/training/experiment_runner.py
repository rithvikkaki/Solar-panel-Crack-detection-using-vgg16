# Complete Experiment Suite for SolarSentinel VGG16 Optimization
import os, sys, json, time, argparse
from typing import Dict, Tuple, List
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from ml.explainability.gradcam import GradCAMExplainer

def load_data(manifest_path, partition, target_size=(244, 244)):
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    items = manifest[partition + '_files']
    classes = manifest['classes']
    images = []
    labels = []
    for it in items:
        p = it['path']
        lbl = it['class_index']
        img = Image.open(p).convert('RGB').resize((target_size[1], target_size[0]), Image.Resampling.BILINEAR)
        arr = np.array(img, dtype=np.float32)
        images.append(arr)
        labels.append(lbl)
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32), classes

def compute_weights(labels, num_classes):
    total = len(labels)
    counts = np.bincount(labels, minlength=num_classes)
    return {i: float(total / (num_classes * c)) if c > 0 else 1.0 for i, c in enumerate(counts)}

def build_model(head='head_b', input_size=244, num_classes=6, aug='photometric'):
    inputs = tf.keras.layers.Input(shape=(input_size, input_size, 3), name='solar_input')
    if aug == 'spatial':
        a = tf.keras.Sequential([
            tf.keras.layers.RandomFlip('horizontal'),
            tf.keras.layers.RandomRotation(0.04, fill_mode='nearest'),
            tf.keras.layers.RandomZoom(0.05, fill_mode='nearest'),
        ])
        x_aug = a(inputs)
    elif aug == 'photometric':
        a = tf.keras.Sequential([
            tf.keras.layers.RandomFlip('horizontal'),
            tf.keras.layers.RandomRotation(0.04, fill_mode='nearest'),
            tf.keras.layers.RandomZoom(0.04, fill_mode='nearest'),
            tf.keras.layers.RandomBrightness(0.08, value_range=(0, 255)),
            tf.keras.layers.RandomContrast(0.08),
        ])
        x_aug = a(inputs)
    else:
        x_aug = inputs

    x = tf.keras.applications.vgg16.preprocess_input(x_aug)
    base = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(input_size, input_size, 3))
    base.trainable = False
    x = base(x, training=False)

    if head == 'head_a':
        x = tf.keras.layers.GlobalAveragePooling2D(name='gap')(x)
        x = tf.keras.layers.Dropout(0.3, name='dropout_1')(x)
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='classifier')(x)
    elif head == 'head_b':
        x = tf.keras.layers.GlobalAveragePooling2D(name='gap')(x)
        x = tf.keras.layers.BatchNormalization(name='bn_gap')(x)
        x = tf.keras.layers.Dense(512, activation='relu', name='dense_512')(x)
        x = tf.keras.layers.Dropout(0.3, name='dropout_1')(x)
        x = tf.keras.layers.Dense(256, activation='relu', name='dense_256')(x)
        x = tf.keras.layers.Dropout(0.2, name='dropout_2')(x)
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='classifier')(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='Solar_VGG16'), base

def eval_model(model, x, y, classes):
    preds = model.predict(x, batch_size=32, verbose=0)
    yp = np.argmax(preds, axis=1)
    acc = float(accuracy_score(y, yp))
    pm, rm, f1m, _ = precision_recall_fscore_support(y, yp, average='macro', zero_division=0)
    pw, rw, f1w, _ = precision_recall_fscore_support(y, yp, average='weighted', zero_division=0)
    pp, rp, f1p, sp = precision_recall_fscore_support(y, yp, average=None, labels=list(range(len(classes))), zero_division=0)
    
    cm = confusion_matrix(y, yp, labels=list(range(len(classes)))).tolist()
    phys_idx = classes.index('Physical-Damage')
    
    per_cls = {}
    for i, c in enumerate(classes):
        per_cls[c] = {
            'precision': round(float(pp[i]), 4),
            'recall': round(float(rp[i]), 4),
            'f1': round(float(f1p[i]), 4),
            'support': int(sp[i])
        }
    return {
        'accuracy': round(acc, 4),
        'macro_f1': round(float(f1m), 4),
        'weighted_f1': round(float(f1w), 4),
        'precision_macro': round(float(pm), 4),
        'recall_macro': round(float(rm), 4),
        'physical_damage_recall': round(float(rp[phys_idx]), 4),
        'physical_damage_f1': round(float(f1p[phys_idx]), 4),
        'per_class': per_cls,
        'confusion_matrix': cm
    }


def run_single_experiment(
    exp_name: str,
    manifest_path: str = 'ml/metadata/split_manifest_70_15_15.json',
    head: str = 'head_b',
    aug: str = 'photometric',
    use_class_weights: bool = True,
    fine_tune_depth: int = 15,
    stage1_epochs: int = 8,
    stage2_epochs: int = 12,
    stage1_lr: float = 1e-3,
    stage2_lr: float = 1e-4,
    random_seed: int = 42,
    input_size: int = 244,
    out_dir: str = 'ml/experiments/phase9_vgg16'
):
    os.makedirs(out_dir, exist_ok=True)
    exp_path = os.path.join(out_dir, exp_name)
    os.makedirs(exp_path, exist_ok=True)
    
    tf.keras.utils.set_random_seed(random_seed)
    np.random.seed(random_seed)
    
    print('=== Starting Experiment: ' + exp_name + ' ===')
    x_train, y_train, classes = load_data(manifest_path, 'train', (input_size, input_size))
    x_val, y_val, _ = load_data(manifest_path, 'val', (input_size, input_size))
    
    cw = compute_weights(y_train, len(classes)) if use_class_weights else None
    print('Class weights:', cw)
    
    model, base = build_model(head=head, input_size=input_size, num_classes=len(classes), aug=aug)
    
    # Stage 1: Frozen Backbone
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=stage1_lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    es1 = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=stage1_epochs,
        batch_size=32,
        class_weight=cw,
        callbacks=[es1],
        verbose=1
    )
    
    # Stage 2: Fine-Tuning
    base.trainable = True
    for l in base.layers[:fine_tune_depth]:
        l.trainable = False
        
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=stage2_lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    ckpt_path = os.path.join(exp_path, 'best_model.keras')
    ckpt = tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True)
    es2 = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
    
    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=stage2_epochs,
        batch_size=32,
        class_weight=cw,
        callbacks=[ckpt, es2, rlr],
        verbose=1
    )
    
    # Load best model for evaluation
    best_model = tf.keras.models.load_model(ckpt_path)
    val_metrics = eval_model(best_model, x_val, y_val, classes)
    
    metrics_file = os.path.join(exp_path, 'val_metrics.json')
    with open(metrics_file, 'w') as mf:
        json.dump(val_metrics, mf, indent=2)
        
    print('Finished ' + exp_name + ' | Val Acc: ' + str(val_metrics['accuracy']) + ' | Val Macro F1: ' + str(val_metrics['macro_f1']) + ' | Phys Recall: ' + str(val_metrics['physical_damage_recall']))
    return val_metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--head', type=str, default='head_b')
    parser.add_argument('--aug', type=str, default='photometric')
    parser.add_argument('--no_class_weights', action='store_true')
    parser.add_argument('--depth', type=int, default=15)
    parser.add_argument('--epochs1', type=int, default=8)
    parser.add_argument('--epochs2', type=int, default=12)
    parser.add_argument('--lr1', type=float, default=1e-3)
    parser.add_argument('--lr2', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--size', type=int, default=244)
    args = parser.parse_args()
    
    run_single_experiment(
        exp_name=args.exp_name,
        head=args.head,
        aug=args.aug,
        use_class_weights=not args.no_class_weights,
        fine_tune_depth=args.depth,
        stage1_epochs=args.epochs1,
        stage2_epochs=args.epochs2,
        stage1_lr=args.lr1,
        stage2_lr=args.lr2,
        random_seed=args.seed,
        input_size=args.size
    )
