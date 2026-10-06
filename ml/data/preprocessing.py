"""
SolarSentinel AI - Data Preprocessing & Augmentation Pipeline
Provides standardized ImageNet preprocessing for VGG16 transfer learning,
custom augmentation pipelines, and deterministic tf.data / batch generators.
"""
import os
import sys
import json
import numpy as np
from PIL import Image
from typing import Tuple, List, Optional, Union, Dict, Any
import tensorflow as tf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_TARGET_SIZE = (224, 224)
# ImageNet mean values in BGR order (caffe mode standard for VGG16)
IMAGENET_MEAN_BGR = np.array([103.939, 116.779, 123.68], dtype=np.float32)


def preprocess_image_array(img_array: np.ndarray) -> np.ndarray:
    """
    Standard VGG16 caffe-style preprocessing:
    Converts RGB to BGR and subtracts ImageNet channel means.
    Input: float32 or uint8 RGB array with values in [0, 255].
    Output: zero-centered float32 array in BGR format.
    """
    arr = img_array.astype(np.float32)
    # RGB to BGR
    arr = arr[..., ::-1]
    # Mean subtraction
    arr[..., 0] -= IMAGENET_MEAN_BGR[0]
    arr[..., 1] -= IMAGENET_MEAN_BGR[1]
    arr[..., 2] -= IMAGENET_MEAN_BGR[2]
    return arr


def load_and_preprocess_image(
    image_input: Union[str, Image.Image, bytes, np.ndarray],
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE
) -> np.ndarray:
    """
    Loads an image from various input types and returns preprocessed batch-ready tensor (1, H, W, 3).
    """
    if isinstance(image_input, str):
        if not os.path.isabs(image_input):
            image_input = os.path.join(PROJECT_ROOT, image_input)
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, bytes):
        import io
        img = Image.open(io.BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    elif isinstance(image_input, np.ndarray):
        if image_input.dtype != np.uint8 and image_input.max() <= 1.0:
            image_input = (image_input * 255.0).astype(np.uint8)
        img = Image.fromarray(image_input).convert("RGB")
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    if img.size != (target_size[1], target_size[0]):
        img = img.resize((target_size[1], target_size[0]), Image.Resampling.BILINEAR)

    arr = np.array(img, dtype=np.float32)
    preprocessed = preprocess_image_array(arr)
    return np.expand_dims(preprocessed, axis=0)


def build_augmentation_layer() -> tf.keras.Sequential:
    """
    Builds a robust Keras augmentation sequential model for solar panel images.
    Applies spatial flips, subtle rotations, and random contrast/zoom.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.08),  # ~30 degrees
        tf.keras.layers.RandomZoom((-0.1, 0.1)),
        tf.keras.layers.RandomContrast(0.15)
    ], name="data_augmentation")


def load_dataset_from_manifest(
    manifest_path: str,
    split: str = "train",
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
    batch_size: int = 32,
    shuffle: bool = True,
    augment: bool = False,
    num_classes: int = 6
) -> tf.data.Dataset:
    """
    Constructs a high-performance tf.data.Dataset pipeline from a split manifest.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    items = manifest.get(split) or manifest.get(f"{split}_files", [])
    if not items:
        raise ValueError(f"No samples found for split '{split}' in manifest: {manifest_path}")

    filepaths = []
    labels = []

    for item in items:
        p = item if isinstance(item, str) else item["path"]
        if not os.path.isabs(p):
            p = os.path.join(PROJECT_ROOT, p)
        c_idx = item.get("class_index") if isinstance(item, dict) else None
        filepaths.append(p)
        labels.append(c_idx)

    # If class_indices not directly provided, deduce from directory structure
    if any(l is None for l in labels):
        class_names = manifest.get("classes", sorted(list({os.path.basename(os.path.dirname(p)) for p in filepaths})))
        cls_to_idx = {name: idx for idx, name in enumerate(class_names)}
        labels = [cls_to_idx[os.path.basename(os.path.dirname(p))] for p in filepaths]

    filepaths = np.array(filepaths)
    labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)

    def parse_function(filename, label):
        image_string = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.resize(image, target_size)
        # Convert RGB to BGR
        image = image[..., ::-1]
        # VGG16 mean subtraction
        image = tf.cast(image, tf.float32) - IMAGENET_MEAN_BGR
        return image, label

    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(filepaths), reshuffle_each_iteration=True)

    dataset = dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        aug_layer = build_augmentation_layer()
        dataset = dataset.map(lambda x, y: (aug_layer(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


if __name__ == "__main__":
    test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    tensor = load_and_preprocess_image(test_img)
    print("Preprocessing verification:")
    print("Output shape:", tensor.shape)
    print("Output dtype:", tensor.dtype)
    print("Mean per channel:", tensor.mean(axis=(0, 1, 2)))
