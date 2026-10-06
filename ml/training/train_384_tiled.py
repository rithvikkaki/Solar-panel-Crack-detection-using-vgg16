# Train 384x384 VGG16 with High-Resolution Tiling
import os, sys, json, time
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd()))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from ml.training.experiment_runner import compute_weights, eval_model

manifest_path = 'ml/metadata/split_manifest_70_15_15.json'
with open(manifest_path) as f:
    manifest = json.load(f)

train_files = manifest['train_files']
val_files = manifest['val_files']
classes = manifest['classes']

def load_tiled_train(items, target_size=(384, 384)):
    images = []
    labels = []
    for it in items:
        p = it['path']
        lbl = it['class_index']
        img = Image.open(p).convert('RGB')
        w, h = img.size
        
        # 1. Full panel resized to 384x384
        im_full = img.resize(target_size, Image.Resampling.BILINEAR)
        images.append(np.array(im_full, dtype=np.float32))
        labels.append(lbl)
        
        # 2. Add center crop tile for high-res images to preserve micro-crack details
        if w >= 400 and h >= 400:
            c_w, c_h = int(w * 0.8), int(h * 0.8)
            left = (w - c_w) // 2
            top = (h - c_h) // 2
            tile = img.crop((left, top, left + c_w, top + c_h)).resize(target_size, Image.Resampling.BILINEAR)
            images.append(np.array(tile, dtype=np.float32))
            labels.append(lbl)
            
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)

def load_val(items, target_size=(384, 384)):
    images = []
    labels = []
    for it in items:
        p = it['path']
        lbl = it['class_index']
        img = Image.open(p).convert('RGB').resize(target_size, Image.Resampling.BILINEAR)
        images.append(np.array(img, dtype=np.float32))
        labels.append(lbl)
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)

print('Loading 384x384 datasets with high-res tiling...')
x_train, y_train = load_tiled_train(train_files, (384, 384))
x_val, y_val = load_val(val_files, (384, 384))
print('Train samples after tiling: ' + str(len(x_train)) + ' | Val samples: ' + str(len(x_val)))

cw = compute_weights(y_train, len(classes))
print('Class weights:', cw)

inputs = tf.keras.layers.Input(shape=(384, 384, 3), name='solar_384_input')
aug = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.04, fill_mode='nearest'),
    tf.keras.layers.RandomZoom(0.04, fill_mode='nearest'),
    tf.keras.layers.RandomBrightness(0.08, value_range=(0, 255)),
    tf.keras.layers.RandomContrast(0.08),
])
x = aug(inputs)
x = tf.keras.applications.vgg16.preprocess_input(x)

base = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(384, 384, 3))
base.trainable = False
x = base(x, training=False)

x = tf.keras.layers.GlobalAveragePooling2D(name='gap')(x)
x = tf.keras.layers.BatchNormalization(name='bn_gap')(x)
x = tf.keras.layers.Dense(512, activation='relu', name='dense_512')(x)
x = tf.keras.layers.Dropout(0.3, name='dropout_1')(x)
x = tf.keras.layers.Dense(256, activation='relu', name='dense_256')(x)
x = tf.keras.layers.Dropout(0.2, name='dropout_2')(x)
outputs = tf.keras.layers.Dense(len(classes), activation='softmax', name='classifier')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name='SolarSentinel_384_VGG16')

# Stage 1
print('Stage 1: Training Head on 384x384...')
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=5,
    batch_size=16,
    class_weight=cw,
    verbose=1
)

# Stage 2: Fine-Tuning Block 5
print('Stage 2: Fine-tuning Block 5 on 384x384...')
base.trainable = True
for l in base.layers[:15]:
    l.trainable = False

os.makedirs('ml/experiments/phase9_vgg16_384', exist_ok=True)
ckpt_path = 'ml/experiments/phase9_vgg16_384/best_model.keras'
ckpt = tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor='val_accuracy', save_best_only=True)
rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=6,
    batch_size=16,
    class_weight=cw,
    callbacks=[ckpt, rlr],
    verbose=1
)

best_m = tf.keras.models.load_model(ckpt_path)
val_metrics = eval_model(best_m, x_val, y_val, classes)

with open('ml/experiments/phase9_vgg16_384/val_metrics.json', 'w') as f:
    json.dump(val_metrics, f, indent=2)

print('=== 384x384 VGG16 Training Finished ===')
print('Validation Accuracy: ' + str(round(val_metrics['accuracy']*100, 2)) + '%')
print('Validation Macro F1: ' + str(val_metrics['macro_f1']))
print('Validation Phys Rec: ' + str(round(val_metrics['physical_damage_recall']*100, 1)) + '%')
