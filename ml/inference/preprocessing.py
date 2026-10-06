"""
SolarSentinel AI - Preprocessing Pipeline
Provides identical, deterministic preprocessing across training, evaluation, inference, and Grad-CAM.
Matches canonical VGG16 zero-centered BGR normalization.
"""

import io
from typing import Tuple, Union
import numpy as np
from PIL import Image

# Canonical ImageNet BGR channel means subtracted by VGG16
# Format: [Blue, Green, Red]
IMAGENET_BGR_MEANS = np.array([103.939, 116.779, 123.68], dtype=np.float32)

# Baseline dimension preserved from original notebook
DEFAULT_INPUT_SIZE: Tuple[int, int] = (244, 244)
MODERN_INPUT_SIZE: Tuple[int, int] = (224, 224)


def load_image_as_rgb(image_source: Union[str, bytes, Image.Image]) -> Image.Image:
    """
    Loads an image from filepath, byte buffer, or PIL image, guaranteeing RGB format.
    """
    if isinstance(image_source, Image.Image):
        return image_source.convert("RGB")
    elif isinstance(image_source, bytes):
        img = Image.open(io.BytesIO(image_source))
        return img.convert("RGB")
    elif isinstance(image_source, str):
        img = Image.open(image_source)
        return img.convert("RGB")
    else:
        raise ValueError(f"Unsupported image source type: {type(image_source)}")


def vgg16_preprocess_numpy(img_array: np.ndarray) -> np.ndarray:
    """
    Pure NumPy implementation of tf.keras.applications.vgg16.preprocess_input.
    
    1. Expects float32 input with RGB channels in range [0, 255].
    2. Reverses channel order from RGB to BGR.
    3. Subtracts ImageNet channel means: Blue: 103.939, Green: 116.779, Red: 123.68.
    Does NOT divide by 255.
    
    Compatible with 3D (H, W, C) or 4D (N, H, W, C) arrays.
    """
    x = img_array.astype(np.float32, copy=True)
    # Convert RGB to BGR along last axis
    x = x[..., ::-1]
    # Subtract BGR means
    x[..., 0] -= IMAGENET_BGR_MEANS[0]  # B
    x[..., 1] -= IMAGENET_BGR_MEANS[1]  # G
    x[..., 2] -= IMAGENET_BGR_MEANS[2]  # R
    return x


def preprocess_image(
    image_source: Union[str, bytes, Image.Image],
    target_size: Tuple[int, int] = DEFAULT_INPUT_SIZE,
    expand_batch: bool = True,
    apply_vgg16_caffe: bool = False
) -> np.ndarray:
    """
    Complete end-to-end preprocessing pipeline for inference.
    Production VGG16 models contain an internal preprocess_input layer,
    so raw RGB float32 arrays in range [0, 255] are passed by default to prevent
    double channel inversion and double mean subtraction.
    """
    pil_img = load_image_as_rgb(image_source)
    # Resize to target size (width, height in PIL)
    resized = pil_img.resize((target_size[1], target_size[0]), Image.Resampling.BILINEAR)
    # Convert to float array [0, 255]
    arr = np.array(resized, dtype=np.float32)
    # Apply external caffe normalization only if explicitly requested
    processed = vgg16_preprocess_numpy(arr) if apply_vgg16_caffe else arr
    
    if expand_batch:
        processed = np.expand_dims(processed, axis=0)
        
    return processed


def preprocess_for_vgg16(
    image_source: Union[str, bytes, Image.Image],
    target_size: Tuple[int, int] = DEFAULT_INPUT_SIZE
) -> np.ndarray:
    """Alias for preprocess_image returning batch format (1, H, W, 3)."""
    return preprocess_image(image_source, target_size=target_size, expand_batch=True)


def deprocess_image_for_display(
    img_array: np.ndarray,
    target_size: Tuple[int, int] = None
) -> np.ndarray:
    """
    Inverts VGG16 preprocessing for visual display (e.g. for Grad-CAM overlay).
    Returns uint8 array in RGB order, range [0, 255].
    """
    x = img_array.copy()
    if x.ndim == 4:
        x = x[0]
        
    # Re-add ImageNet BGR means
    x[..., 0] += IMAGENET_BGR_MEANS[0]
    x[..., 1] += IMAGENET_BGR_MEANS[1]
    x[..., 2] += IMAGENET_BGR_MEANS[2]
    
    # Reverse back from BGR to RGB
    x = x[..., ::-1]
    
    # Clip to valid [0, 255] range
    x = np.clip(x, 0, 255).astype(np.uint8)
    return x
