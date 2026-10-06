"""
SolarSentinel AI - Unit Tests for Grad-CAM Explainability
Tests:
1. Automatic Conv2D layer discovery.
2. Heatmap generation and normalization ([0.0, 1.0]).
3. Spatial dimension preservation matching arbitrary original image sizes.
4. Colorized heatmap and overlay generation.
5. Immutability of original input image.
6. Base64 encoding and JSON serialization.
7. Error handling on invalid models and out-of-range class indices.
"""

import numpy as np
import pytest
from PIL import Image

from ml.explainability.gradcam import (
    GradCAMExplainer,
    GradCAMResult,
    find_target_conv_layer,
    GradCAMError
)
from ml.training.train import build_solar_vgg16_model


@pytest.fixture(scope="module")
def vgg_model():
    """Builds a verified VGG16 model with 6 classes for testing."""
    model, _ = build_solar_vgg16_model(input_shape=(244, 244, 3), num_classes=6)
    return model


@pytest.fixture
def sample_image():
    """Generates an arbitrary-sized synthetic RGB image (height=180, width=320)."""
    np.random.seed(42)
    # Create an image with distinct color patterns
    arr = np.zeros((180, 320, 3), dtype=np.uint8)
    arr[:90, :160] = [200, 50, 50]    # Red quadrant
    arr[:90, 160:] = [50, 200, 50]    # Green quadrant
    arr[90:, :160] = [50, 50, 200]    # Blue quadrant
    arr[90:, 160:] = [220, 220, 50]   # Yellow quadrant
    return Image.fromarray(arr)


def test_layer_discovery(vgg_model):
    """Verifies that find_target_conv_layer locates the deepest Conv2D layer."""
    layer, parent = find_target_conv_layer(vgg_model)
    assert layer is not None
    assert "conv" in layer.name.lower() or isinstance(layer.name, str)
    assert layer.name == "block5_conv3"


def test_gradcam_execution_and_shapes(vgg_model, sample_image):
    """Verifies full Grad-CAM pipeline, shape preservation, and normalization."""
    classes = ["Bird-drop", "Clean", "Dusty", "Electrical-damage", "Physical-Damage", "Snow-Covered"]
    explainer = GradCAMExplainer(vgg_model, class_names=classes)

    orig_w, orig_h = sample_image.size  # (320, 180)

    result: GradCAMResult = explainer.explain(
        image_source=sample_image,
        target_class_index=4,  # Physical-Damage
        overlay_alpha=0.6
    )

    # 1. Check raw heatmap
    assert isinstance(result.raw_heatmap, np.ndarray)
    assert result.raw_heatmap.shape == (orig_h, orig_w)
    assert np.min(result.raw_heatmap) >= 0.0
    assert np.max(result.raw_heatmap) <= 1.0

    # 2. Check colorized heatmap
    assert result.colorized_heatmap.shape == (orig_h, orig_w, 3)
    assert result.colorized_heatmap.dtype == np.uint8

    # 3. Check overlay
    assert result.overlay.shape == (orig_h, orig_w, 3)
    assert result.overlay.dtype == np.uint8

    # 4. Check metadata
    assert result.target_class_index == 4
    assert result.target_class_name == "Physical-Damage"
    assert result.original_dimensions == (orig_h, orig_w)


def test_original_image_immutability(vgg_model, sample_image):
    """Verifies that the original input image is NOT altered in any way."""
    orig_copy = np.array(sample_image).copy()
    explainer = GradCAMExplainer(vgg_model)

    result = explainer.explain(sample_image, target_class_index=0)

    current_pixels = np.array(sample_image)
    assert np.array_equal(orig_copy, current_pixels), "Input image was mutated!"
    assert np.array_equal(orig_copy, result.original_image), "Result original copy corrupted!"


def test_base64_serialization(vgg_model, sample_image):
    """Verifies base64 PNG data URI formatting and dictionary export."""
    explainer = GradCAMExplainer(vgg_model, class_names=["C0", "C1", "C2", "C3", "C4", "C5"])
    result = explainer.explain(sample_image)

    data = result.to_dict()
    assert "images" in data
    assert data["images"]["original"].startswith("data:image/png;base64,")
    assert data["images"]["heatmap"].startswith("data:image/png;base64,")
    assert data["images"]["overlay"].startswith("data:image/png;base64,")
    assert "disclaimer" in data


def test_error_handling_invalid_class_index(vgg_model, sample_image):
    """Verifies that an out-of-bounds target class index raises a clean GradCAMError."""
    explainer = GradCAMExplainer(vgg_model)
    with pytest.raises(GradCAMError) as exc_info:
        explainer.explain(sample_image, target_class_index=99)
    assert "Invalid target class index 99" in str(exc_info.value)


def test_error_handling_none_model():
    """Verifies that initializing with None model raises GradCAMError."""
    with pytest.raises(GradCAMError):
        GradCAMExplainer(None)


def test_model_gradcam_layers_configuration():
    """Verifies that MODEL_GRADCAM_LAYERS contains all benchmarked architectures."""
    from ml.explainability.gradcam import MODEL_GRADCAM_LAYERS
    assert "vgg16" in MODEL_GRADCAM_LAYERS
    assert "mobilenetv2" in MODEL_GRADCAM_LAYERS
    assert "efficientnetb0" in MODEL_GRADCAM_LAYERS
    assert MODEL_GRADCAM_LAYERS["vgg16"] == "block5_conv3"
    assert MODEL_GRADCAM_LAYERS["mobilenetv2"] == "Conv_1"
    assert MODEL_GRADCAM_LAYERS["efficientnetb0"] == "top_conv"

