"""
SolarSentinel AI - Inspection Orchestration Service
Validates images, coordinates neural inference and Grad-CAM, and derives rule-based health assessments.
"""

import io
import uuid
from datetime import datetime, timezone
from typing import Optional, Tuple
import cv2
import numpy as np
from PIL import Image

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.schemas.inspection import (
    ExplainabilityPayload,
    GradCAMImages,
    HealthAssessment,
    InspectionResponse,
    ModelMetadata
)
from backend.app.services.model_service import model_service
from ml.explainability.gradcam import GradCAMResult, SCIENTIFIC_DISCLAIMER


class ImageValidationError(Exception):
    """Raised when an uploaded file fails validation checks."""
    pass


def validate_and_decode_image(
    file_bytes: bytes,
    filename: str,
    content_type: Optional[str] = None
) -> Tuple[Image.Image, Tuple[int, int]]:
    """
    Strictly validates image byte buffer:
    - Size within limits (< 10MB)
    - File extension in allowed set
    - Content-type MIME verification
    - PIL header verification & pixel decompression
    """
    # 1. Size check
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(file_bytes) == 0:
        raise ImageValidationError("Uploaded file is empty (0 bytes).")
    if len(file_bytes) > max_bytes:
        raise ImageValidationError(
            f"File size ({len(file_bytes) / 1024 / 1024:.2f} MB) exceeds maximum allowed "
            f"limit of {settings.MAX_UPLOAD_SIZE_MB} MB."
        )

    # 2. Extension check
    ext = "." + filename.split(".")[-1].lower() if "." in filename else ""
    if ext not in settings.ALLOWED_IMAGE_EXTENSIONS:
        raise ImageValidationError(
            f"Unsupported file extension '{ext}'. Allowed extensions: {settings.ALLOWED_IMAGE_EXTENSIONS}"
        )

    # 3. MIME type check if provided
    if content_type and content_type.lower() not in settings.ALLOWED_IMAGE_MIMETYPES and content_type != "application/octet-stream":
        raise ImageValidationError(
            f"Unsupported media type '{content_type}'. Allowed types: {settings.ALLOWED_IMAGE_MIMETYPES}"
        )

    # 4. Decompression & corruption check
    try:
        stream = io.BytesIO(file_bytes)
        img = Image.open(stream)
        img.verify()  # Check headers
        stream.seek(0)
        img = Image.open(stream)
        img.load()    # Decompress all pixel blocks
        img = img.convert("RGB")
    except Exception as e:
        raise ImageValidationError(f"Corrupted or non-decodable image file: {str(e)}") from e

    dimensions = (img.height, img.width)
    return img, dimensions


def get_confidence_level(confidence: float) -> str:
    """
    Transparent UI interpretation threshold:
    HIGH (>= 0.85), MODERATE (0.60 <= conf < 0.85), LOW (< 0.60).
    Described clearly as UI interpretation thresholds, not industrial certification.
    """
    if confidence >= 0.85:
        return "HIGH"
    elif confidence >= 0.60:
        return "MODERATE"
    else:
        return "LOW"


def compute_health_assessment(predicted_condition: str) -> HealthAssessment:
    """
    Rule-based mapping from predicted condition to maintenance health score.
    Strictly separates AI visual classification from deterministic maintenance rules.
    """
    rule = settings.HEALTH_RULES.get(
        predicted_condition,
        {"health_score": 50, "risk_level": "MEDIUM", "maintenance_priority": "EVALUATE"}
    )

    recommendations = {
        "Clean": "Solar panel surface appears nominal. No active cleaning required. Continue standard scheduled inspection.",
        "Dusty": "Particulate accumulation detected. Routine surface wash with demineralized water is recommended.",
        "Bird-drop": "Localized organic residue detected. Localized cleaning recommended to prevent hotspot cell degradation.",
        "Snow-Covered": "Surface obstruction detected. Follow certified operational clearing protocols.",
        "Electrical-damage": "Potential high-risk condition detected. Qualified technician electrical inspection and thermal imaging scan is recommended.",
        "Physical-Damage": "Potential high-risk condition detected. Qualified technician physical structural inspection is recommended."
    }

    return HealthAssessment(
        health_score=rule["health_score"],
        risk_level=rule["risk_level"],
        maintenance_priority=rule["maintenance_priority"],
        recommendation=recommendations.get(
            predicted_condition,
            "Conduct visual inspection and consult maintenance protocol."
        )
    )


class InspectionService:
    """Coordinates validation, ML inference, Grad-CAM, and reporting."""

    @staticmethod
    def process_inspection(file_bytes: bytes, filename: str, content_type: Optional[str] = None) -> InspectionResponse:
        """Executes full real-model inspection pipeline."""
        # 1. Validate and decode
        img, (h, w) = validate_and_decode_image(file_bytes, filename, content_type)

        # 2. Run forward pass
        pred_data = model_service.predict(img)

        # 3. Run Grad-CAM
        gradcam_res: GradCAMResult = model_service.explain(
            img,
            target_class_index=pred_data["class_index"]
        )

        # 4. Health evaluation
        health = compute_health_assessment(pred_data["predicted_condition"])
        conf = pred_data["confidence"]
        conf_level = get_confidence_level(conf)

        # 5. Build structured response
        return InspectionResponse(
            inspection_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            filename=filename,
            predicted_condition=pred_data["predicted_condition"],
            confidence=conf,
            confidence_level=conf_level,
            class_probabilities=pred_data["class_probabilities"],
            all_classes=model_service.class_names,
            health_assessment=health,
            explainability=ExplainabilityPayload(
                target_class_index=gradcam_res.target_class_index,
                target_class_name=gradcam_res.target_class_name,
                target_layer_name=gradcam_res.target_layer_name,
                original_dimensions=[h, w],
                disclaimer=gradcam_res.disclaimer,
                images=GradCAMImages(
                    original=gradcam_res.to_base64("original"),
                    heatmap=gradcam_res.to_base64("heatmap"),
                    overlay=gradcam_res.to_base64("overlay")
                )
            ),
            model_metadata=ModelMetadata(
                architecture="VGG16 Transfer Learning (Keras 3)",
                input_dimensions=[settings.MODEL_INPUT_HEIGHT, settings.MODEL_INPUT_WIDTH, 3],
                is_demo_preview=False
            )
        )


    @staticmethod
    def process_demo_preview(file_bytes: bytes, filename: str) -> InspectionResponse:
        """
        Generates an explicitly labeled demo preview with transparent disclaimer.
        Used only for testing UI components when model weights are not loaded.
        """
        img, (h, w) = validate_and_decode_image(file_bytes, filename)
        orig_np = np.array(img, dtype=np.uint8)

        # Generate illustrative Gaussian center heatmap for UI preview
        y, x = np.ogrid[:h, :w]
        cy, cx = h / 2, w / 2
        dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
        radius = min(h, w) * 0.35
        raw_heat = np.exp(-(dist_from_center**2) / (2 * (radius / 2)**2)).astype(np.float32)
        raw_heat = np.clip(raw_heat, 0.0, 1.0)

        heatmap_uint8 = np.uint8(255 * raw_heat)
        colorized_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        colorized_rgb = cv2.cvtColor(colorized_bgr, cv2.COLOR_BGR2RGB)
        overlay_rgb = cv2.addWeighted(orig_np, 0.5, colorized_rgb, 0.5, 0.0)

        demo_res = GradCAMResult(
            raw_heatmap=raw_heat,
            colorized_heatmap=colorized_rgb,
            overlay=overlay_rgb,
            original_image=orig_np,
            target_class_index=4,
            target_class_name="Physical-Damage",
            target_layer_name="block5_conv3 (demo preview)",
            original_dimensions=(h, w),
            disclaimer=(
                "DEMO VISUALIZATION: Illustrative interface preview. "
                "Results are NOT generated by the trained neural network."
            )
        )

        demo_probs = {
            "Bird-drop": 0.02,
            "Clean": 0.01,
            "Dusty": 0.04,
            "Electrical-damage": 0.08,
            "Physical-Damage": 0.85,
            "Snow-Covered": 0.00
        }

        return InspectionResponse(
            inspection_id=f"demo-{uuid.uuid4()}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            filename=filename,
            predicted_condition="Physical-Damage",
            confidence=0.85,
            confidence_level="HIGH",
            class_probabilities=demo_probs,
            all_classes=model_service.class_names,
            health_assessment=compute_health_assessment("Physical-Damage"),

            explainability=ExplainabilityPayload(
                target_class_index=4,
                target_class_name="Physical-Damage",
                target_layer_name="block5_conv3 (demo preview)",
                original_dimensions=[h, w],
                disclaimer=demo_res.disclaimer,
                images=GradCAMImages(
                    original=demo_res.to_base64("original"),
                    heatmap=demo_res.to_base64("heatmap"),
                    overlay=demo_res.to_base64("overlay")
                )
            ),
            model_metadata=ModelMetadata(
                architecture="Demo Visualization Preview",
                input_dimensions=[settings.MODEL_INPUT_HEIGHT, settings.MODEL_INPUT_WIDTH, 3],
                is_demo_preview=True
            )
        )
