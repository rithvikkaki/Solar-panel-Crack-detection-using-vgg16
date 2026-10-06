"""
SolarSentinel AI - VGG16 Model Architecture Factory
Constructs ImageNet-pretrained VGG16 models tailored for solar panel defect detection.
Supports both Head A (compact GAP) and Head B (deep dense with BatchNorm),
as well as modular layer freezing/unfreezing for staged transfer learning.
"""
import os
import sys
from typing import Tuple, List, Optional
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    BatchNormalization
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def build_vgg16_classifier(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 6,
    head_type: str = "head_b",
    weights: str = "imagenet",
    freeze_backbone: bool = True,
    unfreeze_from_layer: Optional[str] = None,
    dropout_rate: float = 0.3
) -> Model:
    """
    Builds a VGG16 model with customizable top classification head and freezing strategy.

    Args:
        input_shape: Input image dimensions (H, W, C).
        num_classes: Number of target categories.
        head_type: 'head_a' (GAP + Dropout + Dense) or 'head_b' (GAP + BN + Dense(512) + Dense(256) + Dense).
        weights: Pre-trained weights, default 'imagenet'.
        freeze_backbone: If True, all convolutional layers are frozen initially.
        unfreeze_from_layer: Layer name from which layers become trainable (e.g., 'block5_conv1' or 'block4_conv1').
        dropout_rate: Dropout rate for regularization.

    Returns:
        Compiled or uncompiled tf.keras.Model.
    """
    base_model = VGG16(
        weights=weights,
        include_top=False,
        input_shape=input_shape
    )

    if freeze_backbone:
        base_model.trainable = False
        if unfreeze_from_layer is not None:
            trainable_flag = False
            for layer in base_model.layers:
                if layer.name == unfreeze_from_layer:
                    trainable_flag = True
                layer.trainable = trainable_flag
    else:
        base_model.trainable = True

    inputs = Input(shape=input_shape, name="input_tensor")
    x = base_model(inputs)

    if head_type.lower() == "head_a":
        # Head A: Compact Global Average Pooling
        x = GlobalAveragePooling2D(name="gap")(x)
        x = Dropout(dropout_rate, name="dropout_head_a")(x)
        outputs = Dense(num_classes, activation="softmax", name="predictions")(x)
    elif head_type.lower() == "head_b":
        # Head B: Multi-layer regularized dense head with Batch Normalization
        x = GlobalAveragePooling2D(name="gap")(x)
        x = BatchNormalization(name="bn_1")(x)
        x = Dense(512, activation="relu", name="dense_512")(x)
        x = Dropout(dropout_rate + 0.1, name="dropout_1")(x)
        x = BatchNormalization(name="bn_2")(x)
        x = Dense(256, activation="relu", name="dense_256")(x)
        x = Dropout(dropout_rate, name="dropout_2")(x)
        outputs = Dense(num_classes, activation="softmax", name="predictions")(x)
    else:
        raise ValueError(f"Unknown head_type '{head_type}'. Choose 'head_a' or 'head_b'.")

    model = Model(inputs=inputs, outputs=outputs, name=f"vgg16_{head_type.lower()}")
    return model


def print_model_summary(model: Model) -> None:
    """Prints architecture details, trainable vs non-trainable parameters."""
    total_params = model.count_params()
    trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    non_trainable_params = total_params - trainable_params

    print("=" * 60)
    print(f"Model: {model.name}")
    print(f"Total Parameters:         {total_params:,}")
    print(f"Trainable Parameters:     {trainable_params:,}")
    print(f"Non-Trainable Parameters: {non_trainable_params:,}")
    print("=" * 60)


if __name__ == "__main__":
    m_a = build_vgg16_classifier(head_type="head_a")
    print("--- Head A Summary ---")
    print_model_summary(m_a)

    m_b = build_vgg16_classifier(head_type="head_b", unfreeze_from_layer="block5_conv1")
    print("\n--- Head B (Block 5 Unfrozen) Summary ---")
    print_model_summary(m_b)
