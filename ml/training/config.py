"""
SolarSentinel AI - Training Configuration
Maintains baseline notebook parameters while providing clean, reproducible settings.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple

# Base paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "ml", "models")
DEFAULT_METADATA_DIR = os.path.join(PROJECT_ROOT, "ml", "metadata")
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "solar_sentinel_vgg16.keras")
DEFAULT_CLASS_NAMES_PATH = os.path.join(DEFAULT_METADATA_DIR, "class_names.json")
DEFAULT_METRICS_PATH = os.path.join(DEFAULT_METADATA_DIR, "evaluation_metrics.json")

# Verified canonical class names from original project audit
VERIFIED_CLASS_NAMES: List[str] = [
    "Bird-drop",
    "Clean",
    "Dusty",
    "Electrical-damage",
    "Physical-Damage",
    "Snow-Covered"
]


@dataclass
class TrainingConfig:
    """
    Configuration parameters for SolarSentinel VGG16 model.
    Preserves original notebook baseline (244x244) while supporting
    configurable/modernized comparison (224x224).
    """
    # Image dimensions
    # Original notebook baseline: (244, 244)
    # Canonical ImageNet size: (224, 224)
    img_height: int = 244
    img_width: int = 244
    channels: int = 3

    # Dataset & Partitioning
    data_dir: str = DEFAULT_DATA_DIR
    validation_split: float = 0.2
    batch_size: int = 32
    random_seed: int = 42
    shuffle: bool = True

    # Training Parameters - Stage 1 (Frozen Backbone Feature Extraction)
    stage1_epochs: int = 15
    stage1_lr: float = 0.001
    dropout_rate: float = 0.3

    # Training Parameters - Stage 2 (Fine-tuning Block 5)
    stage2_epochs: int = 15
    stage2_lr: float = 0.0001
    fine_tune_from_layer: int = 14  # Unfreezes block5 (layers 14 to end)

    # Callbacks
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.01
    reduce_lr_patience: int = 2
    reduce_lr_factor: float = 0.2
    reduce_lr_min: float = 1e-6

    # Output paths
    model_dir: str = DEFAULT_MODEL_DIR
    model_path: str = DEFAULT_MODEL_PATH
    metadata_dir: str = DEFAULT_METADATA_DIR
    class_names_path: str = DEFAULT_CLASS_NAMES_PATH
    metrics_path: str = DEFAULT_METRICS_PATH

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        return (self.img_height, self.img_width, self.channels)

    def to_dict(self) -> dict:
        return {
            "img_height": self.img_height,
            "img_width": self.img_width,
            "channels": self.channels,
            "validation_split": self.validation_split,
            "batch_size": self.batch_size,
            "random_seed": self.random_seed,
            "stage1_epochs": self.stage1_epochs,
            "stage1_lr": self.stage1_lr,
            "dropout_rate": self.dropout_rate,
            "stage2_epochs": self.stage2_epochs,
            "stage2_lr": self.stage2_lr,
            "fine_tune_from_layer": self.fine_tune_from_layer,
            "model_path": self.model_path,
        }
