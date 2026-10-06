"""
SolarSentinel AI - Health Check API Route
"""

from datetime import datetime, timezone
from fastapi import APIRouter, Depends

from backend.app.dependencies import get_model_service
from backend.app.core.config import settings
from backend.app.schemas.health import HealthResponse, ReadyResponse
from backend.app.services.model_service import ModelService

router = APIRouter(prefix="", tags=["System Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System and Model Health Check",
    description="Reports the operational readiness of the API and whether TensorFlow VGG16 model weights are loaded."
)
async def check_health(ms: ModelService = Depends(get_model_service)) -> HealthResponse:
    is_loaded = ms.is_loaded
    mode = "REAL_MODEL" if is_loaded else "MODEL_UNAVAILABLE"
    message = (
        "SolarSentinel AI is operational with active neural network weights."
        if is_loaded else
        "SolarSentinel backend is online, but model weights are not loaded. "
        "Inference is disabled per scientific honesty rules until weights are supplied."
    )

    return HealthResponse(
        service="SolarSentinel AI",
        status="online",
        model_loaded=is_loaded,
        model_mode=mode,
        version="1.0.0",
        verified_classes=ms.class_names,
        timestamp=datetime.now(timezone.utc).isoformat(),
        message=message
    )


@router.get(
    "/ready",
    response_model=ReadyResponse,
    summary="Readiness Probe",
    description="Reports whether the inference engine is fully initialized and ready for traffic."
)
async def check_readiness(ms: ModelService = Depends(get_model_service)) -> ReadyResponse:
    is_loaded = ms.is_loaded
    mode = "REAL_MODEL" if is_loaded else "MODEL_UNAVAILABLE"
    
    # Hide internal file paths in production mode
    model_repr = (
        "ml/models/solar_sentinel_vgg16.keras"
        if settings.APP_ENV == "production"
        else settings.MODEL_PATH
    )

    return ReadyResponse(
        ready=is_loaded,
        model_loaded=is_loaded,
        model_path=model_repr if is_loaded else "none",
        mode=mode,
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@router.get(
    "/model/metrics",
    summary="Get Scientific Model Evaluation and Benchmark Metrics",
    description="Loads real persisted benchmark evaluation metrics and model identity from disk."
)
async def get_model_metrics():
    """Returns genuine held-out evaluation metrics and performance benchmark data."""
    import os
    import json

    prod_model_path = os.path.join(settings.PROJECT_ROOT, "ml", "metadata", "production_model.json")
    prod_metrics_path = os.path.join(settings.PROJECT_ROOT, "ml", "metadata", "production_model_metrics.json")
    test_eval_path = os.path.join(settings.PROJECT_ROOT, "ml", "metadata", "test_set_evaluation.json")
    bench_path = os.path.join(settings.PROJECT_ROOT, "ml", "metadata", "performance_benchmark.json")

    prod_model = {}
    if os.path.exists(prod_model_path):
        try:
            with open(prod_model_path, "r", encoding="utf-8") as f:
                prod_model = json.load(f)
        except Exception:
            prod_model = {}

    prod_metrics = {}
    if os.path.exists(prod_metrics_path):
        try:
            with open(prod_metrics_path, "r", encoding="utf-8") as f:
                prod_metrics = json.load(f)
        except Exception:
            prod_metrics = {}

    test_eval = {}
    if os.path.exists(test_eval_path):
        try:
            with open(test_eval_path, "r", encoding="utf-8") as f:
                test_eval = json.load(f)
        except Exception:
            test_eval = {}

    benchmark = {}
    if os.path.exists(bench_path):
        try:
            with open(bench_path, "r", encoding="utf-8") as f:
                benchmark = json.load(f)
        except Exception:
            benchmark = {}

    # Standardized response structure preserving both legacy test_evaluation and rich production_model
    return {
        "status": "success",
        "production_model": prod_model or {
            "model_id": "vgg16_baseline_production",
            "model_name": "SolarSentinel VGG16 Transfer Learning",
            "architecture": "VGG16",
            "input_shape": [settings.MODEL_INPUT_HEIGHT, settings.MODEL_INPUT_WIDTH, 3],
            "gradcam_target_layer": "block5_conv3"
        },
        "architecture": prod_model.get("architecture", "VGG16"),
        "input_dimensions": prod_model.get("input_shape", [settings.MODEL_INPUT_HEIGHT, settings.MODEL_INPUT_WIDTH, 3]),
        "production_model_metrics": prod_metrics,
        "test_evaluation": test_eval,
        "performance_benchmark": benchmark,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }



