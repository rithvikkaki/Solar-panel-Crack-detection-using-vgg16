"""
SolarSentinel AI - Inference Pipeline
Loads model, executes forward pass, and outputs verified probability distributions.
Enforces scientific honesty: refuses to fabricate predictions when model is absent.
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from ml.inference.preprocessing import DEFAULT_INPUT_SIZE, preprocess_image


class ModelNotLoadedError(Exception):
    """Raised when inference is requested but no model weights are available."""
    pass


class SolarSentinelPredictor:
    """
    Production inference engine for SolarSentinel AI.
    Loads Keras/TensorFlow model and verified class metadata.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        class_names_path: Optional[str] = None,
        input_size: Tuple[int, int] = DEFAULT_INPUT_SIZE
    ):
        self.input_size = input_size
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        
        self.model_path = model_path or os.path.join(
            self.project_root, "ml", "models", "solar_sentinel_vgg16.keras"
        )
        self.class_names_path = class_names_path or os.path.join(
            self.project_root, "ml", "metadata", "class_names.json"
        )
        
        self.classes: List[str] = self._load_class_metadata()
        self.model = None
        self._load_model_if_exists()

    def _load_class_metadata(self) -> List[str]:
        """Loads canonical verified class names."""
        if os.path.exists(self.class_names_path):
            try:
                with open(self.class_names_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("classes", [])
            except Exception as e:
                print(f"[WARN] Failed to load class metadata: {e}")
        # Default fallback to verified classes from audit
        return [
            "Bird-drop",
            "Clean",
            "Dusty",
            "Electrical-damage",
            "Physical-Damage",
            "Snow-Covered"
        ]

    def _load_model_if_exists(self) -> bool:
        """Attempts to load serialized Keras model."""
        if not os.path.exists(self.model_path):
            self.model = None
            return False
            
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"[INFO] SolarSentinel model loaded successfully from: {self.model_path}")
            return True
        except Exception as e:
            print(f"[WARN] Failed to load model from {self.model_path}: {e}")
            self.model = None
            return False

    @property
    def is_model_loaded(self) -> bool:
        return self.model is not None

    def predict(
        self,
        image_source: Union[str, bytes, Image.Image]
    ) -> Dict:
        """
        Executes inference on a single solar panel image.
        
        Returns:
            Dict containing:
                - predicted_condition: str
                - confidence: float (0.0 - 1.0)
                - class_probabilities: Dict[str, float]
                - class_index: int
                - num_classes: int
                - input_dimensions: List[int]
        """
        if not self.is_model_loaded:
            raise ModelNotLoadedError(
                f"Model weights not found at '{self.model_path}'. "
                "Per scientific honesty guidelines, predictions are disabled until a model is trained. "
                "Run `python ml/training/train.py` after supplying dataset."
            )

        # Preprocess image
        input_tensor = preprocess_image(
            image_source,
            target_size=self.input_size,
            expand_batch=True
        )

        # Run inference
        raw_output = self.model.predict(input_tensor, verbose=0)
        
        # Ensure softmax probabilities
        probs = raw_output[0]
        # If output was logits (sum != 1), compute softmax
        if not np.isclose(np.sum(probs), 1.0, atol=1e-2):
            exp_probs = np.exp(probs - np.max(probs))
            probs = exp_probs / np.sum(exp_probs)

        predicted_idx = int(np.argmax(probs))
        confidence = float(probs[predicted_idx])
        
        # Map probabilities to classes
        class_probs = {}
        for i, c in enumerate(self.classes):
            if i < len(probs):
                class_probs[c] = round(float(probs[i]), 4)
            else:
                class_probs[c] = 0.0

        predicted_condition = (
            self.classes[predicted_idx]
            if predicted_idx < len(self.classes)
            else f"Class_{predicted_idx}"
        )

        return {
            "predicted_condition": predicted_condition,
            "confidence": round(confidence, 4),
            "class_index": predicted_idx,
            "class_probabilities": class_probs,
            "all_classes": self.classes,
            "num_classes": len(self.classes),
            "input_dimensions": list(self.input_size)
        }
