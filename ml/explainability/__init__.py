"""
SolarSentinel AI - Explainable AI (XAI) Package
Provides Grad-CAM visual explanations for deep learning model predictions.
"""

from ml.explainability.gradcam import (
    GradCAMExplainer,
    GradCAMResult,
    find_target_conv_layer,
    GradCAMError
)

__all__ = [
    "GradCAMExplainer",
    "GradCAMResult",
    "find_target_conv_layer",
    "GradCAMError"
]
