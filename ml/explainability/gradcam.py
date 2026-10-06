"""
SolarSentinel AI - Explainable AI (Grad-CAM)
Implements Gradient-weighted Class Activation Mapping (Grad-CAM) for VGG16 models.

SCIENTIFIC LANGUAGE MANDATE:
"Grad-CAM visualizes image regions that contributed strongly to the model's classification."
It is NOT an object detector, segmentation tool, or physical defect boundary delineator.
"""

import base64
import io
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from ml.inference.preprocessing import (
    DEFAULT_INPUT_SIZE,
    load_image_as_rgb,
    preprocess_image
)

SCIENTIFIC_DISCLAIMER = (
    "Grad-CAM visualizes image regions that contributed strongly to the model's classification. "
    "It reflects neural network feature activations and does not constitute precise physical "
    "defect boundary detection or certified diagnostic certainty."
)

# Centralized architecture-aware target layer configuration
MODEL_GRADCAM_LAYERS: Dict[str, str] = {
    "vgg16": "block5_conv3",
    "mobilenetv2": "Conv_1",
    "efficientnetb0": "top_conv"
}


class GradCAMError(Exception):
    """Base exception for Grad-CAM generation errors."""
    pass



@dataclass
class GradCAMResult:
    """Encapsulates raw and rendered Grad-CAM outputs."""
    raw_heatmap: np.ndarray          # 2D float32 array in range [0, 1]
    colorized_heatmap: np.ndarray    # 3D uint8 RGB array (H, W, 3)
    overlay: np.ndarray              # 3D uint8 RGB array (H, W, 3)
    original_image: np.ndarray       # 3D uint8 RGB array (H, W, 3)
    target_class_index: int
    target_class_name: str
    target_layer_name: str
    original_dimensions: Tuple[int, int]  # (height, width)
    disclaimer: str = SCIENTIFIC_DISCLAIMER

    def to_base64(self, image_type: str = "overlay") -> str:
        """Encodes specified image ('original', 'heatmap', 'overlay') as base64 PNG data URI."""
        if image_type == "original":
            arr = self.original_image
        elif image_type == "heatmap":
            arr = self.colorized_heatmap
        elif image_type == "overlay":
            arr = self.overlay
        else:
            raise ValueError(f"Unknown image_type: {image_type}. Choose 'original', 'heatmap', or 'overlay'.")

        pil_img = Image.fromarray(arr)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    def to_dict(self) -> Dict:
        """Returns JSON-serializable dictionary with base64 encoded images."""
        return {
            "target_class_index": self.target_class_index,
            "target_class_name": self.target_class_name,
            "target_layer_name": self.target_layer_name,
            "original_dimensions": list(self.original_dimensions),
            "disclaimer": self.disclaimer,
            "images": {
                "original": self.to_base64("original"),
                "heatmap": self.to_base64("heatmap"),
                "overlay": self.to_base64("overlay")
            }
        }


def find_target_conv_layer(model, layer_name: Optional[str] = None):
    """
    Safely discovers the appropriate convolutional layer for Grad-CAM.
    If layer_name is provided, retrieves that specific layer.
    Otherwise, automatically locates the deepest Conv2D layer in the architecture,
    recursively traversing submodels if necessary.
    """
    if layer_name:
        try:
            # Check top level
            return model.get_layer(layer_name), None
        except ValueError:
            # Check within submodels
            for layer in model.layers:
                if hasattr(layer, "get_layer"):
                    try:
                        return layer.get_layer(layer_name), layer
                    except ValueError:
                        pass
            raise GradCAMError(f"Specified convolutional layer '{layer_name}' not found in model.")

    # Automatic discovery: search reversed for deepest Conv2D layer
    import tensorflow as tf

    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer, None
        if hasattr(layer, "layers"):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    return sub_layer, layer

    raise GradCAMError("No convolutional layer (Conv2D) found in the model architecture.")


class GradCAMExplainer:
    """
    Production Grad-CAM Explainer for SolarSentinel AI.
    Compatible with Keras Functional models and nested VGG16 backbones.
    """

    def __init__(
        self,
        model,
        class_names: Optional[List[str]] = None,
        target_layer_name: Optional[str] = None,
        input_size: Tuple[int, int] = DEFAULT_INPUT_SIZE
    ):
        if model is None:
            raise GradCAMError("Cannot initialize GradCAMExplainer with None model.")

        self.model = model
        self.class_names = class_names or []
        self.input_size = input_size
        self.target_layer, self.parent_submodel = find_target_conv_layer(model, target_layer_name)
        self.target_layer_name = self.target_layer.name

    def explain(
        self,
        image_source: Union[str, bytes, Image.Image, np.ndarray],
        target_class_index: Optional[int] = None,
        overlay_alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> GradCAMResult:
        """
        Executes Grad-CAM generation pipeline.
        
        Args:
            image_source: Input image (path, bytes, PIL Image, or uint8 ndarray).
            target_class_index: Optional index of class to explain (defaults to top model prediction).
            overlay_alpha: Weight of original image in overlay (0.0 to 1.0).
            colormap: OpenCV colormap enum for colorizing heatmap.
            
        Returns:
            GradCAMResult with original image, colorized heatmap, and overlay.
        """
        import tensorflow as tf

        # 1. Load and preserve original image
        orig_pil = load_image_as_rgb(image_source)
        orig_np = np.array(orig_pil, dtype=np.uint8, copy=True)
        orig_h, orig_w = orig_np.shape[:2]

        # 2. Shared Preprocessing
        preprocessed_tensor = preprocess_image(
            orig_pil,
            target_size=self.input_size,
            expand_batch=True
        )

        # 3. Model Prediction & Class Selection
        raw_preds = self.model.predict(preprocessed_tensor, verbose=0)
        probs = raw_preds[0]

        if target_class_index is None:
            target_class_index = int(np.argmax(probs))

        if target_class_index < 0 or target_class_index >= len(probs):
            raise GradCAMError(
                f"Invalid target class index {target_class_index}. "
                f"Model has {len(probs)} output units."
            )

        target_class_name = (
            self.class_names[target_class_index]
            if target_class_index < len(self.class_names)
            else f"Class_{target_class_index}"
        )

        # 4. Compute Gradients
        try:
            cam_2d = self._compute_gradcam_weights(preprocessed_tensor, target_class_index)
        except Exception as e:
            raise GradCAMError(f"Gradient computation failed during Grad-CAM: {e}") from e

        # 5. Normalize Heatmap
        cam_min, cam_max = np.min(cam_2d), np.max(cam_2d)
        if cam_max - cam_min > 1e-8:
            norm_heatmap = (cam_2d - cam_min) / (cam_max - cam_min)
        else:
            norm_heatmap = np.zeros_like(cam_2d)

        # 6. Resize to Original Image Dimensions
        resized_heatmap = cv2.resize(
            norm_heatmap,
            (orig_w, orig_h),
            interpolation=cv2.INTER_LINEAR
        )
        resized_heatmap = np.clip(resized_heatmap, 0.0, 1.0)

        # 7. Generate Colorized Heatmap
        heatmap_uint8 = np.uint8(255 * resized_heatmap)
        colorized_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
        colorized_rgb = cv2.cvtColor(colorized_bgr, cv2.COLOR_BGR2RGB)

        # 8. Generate Overlay
        alpha = float(np.clip(overlay_alpha, 0.0, 1.0))
        overlay_rgb = cv2.addWeighted(
            orig_np,
            alpha,
            colorized_rgb,
            1.0 - alpha,
            0.0
        )

        return GradCAMResult(
            raw_heatmap=resized_heatmap.astype(np.float32),
            colorized_heatmap=colorized_rgb,
            overlay=overlay_rgb,
            original_image=orig_np,
            target_class_index=target_class_index,
            target_class_name=target_class_name,
            target_layer_name=self.target_layer_name,
            original_dimensions=(orig_h, orig_w),
            disclaimer=SCIENTIFIC_DISCLAIMER
        )

    def _compute_gradcam_weights(self, preprocessed_tensor: np.ndarray, class_idx: int) -> np.ndarray:
        """Internal gradient extraction handling both direct and nested model architectures."""
        import tensorflow as tf

        input_tf = tf.convert_to_tensor(preprocessed_tensor, dtype=tf.float32)

        # If model has nested base model (VGG16 backbone)
        if self.parent_submodel is not None:
            sub = tf.keras.Model(
                inputs=self.parent_submodel.input,
                outputs=[self.target_layer.output, self.parent_submodel.output]
            )

            # Locate head layers following parent submodel
            head_layers = []
            found_sub = False
            for layer in self.model.layers:
                if layer == self.parent_submodel:
                    found_sub = True
                    continue
                if found_sub:
                    head_layers.append(layer)

            with tf.GradientTape() as tape:
                conv_out, base_out = sub(input_tf)
                tape.watch(conv_out)

                y = base_out
                for hl in head_layers:
                    y = hl(y)

                score = y[:, class_idx]

            grads = tape.gradient(score, conv_out)

        else:
            # Direct model architecture
            grad_model = tf.keras.Model(
                inputs=self.model.inputs,
                outputs=[self.target_layer.output, self.model.output]
            )

            with tf.GradientTape() as tape:
                conv_out, preds = grad_model(input_tf)
                tape.watch(conv_out)
                score = preds[:, class_idx]

            grads = tape.gradient(score, conv_out)

        if grads is None:
            raise GradCAMError("Gradient calculation returned None. Ensure target layer outputs are differentiable.")

        # Global Average Pooling of gradients across spatial dimensions (H, W)
        weights = tf.reduce_mean(grads, axis=(1, 2))  # Shape: (batch, channels)

        # Linear combination of feature maps weighted by channel importance
        cam = tf.reduce_sum(tf.multiply(weights[:, None, None, :], conv_out), axis=-1)

        # Rectified Linear Unit (ReLU) to isolate positive contributions
        cam = tf.nn.relu(cam)

        # Return 2D float array
        return cam[0].numpy()
