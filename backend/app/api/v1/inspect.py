"""
SolarSentinel AI - Primary Solar Panel Inspection Endpoint
"""

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from backend.app.dependencies import get_model_service
from backend.app.schemas.inspection import ErrorResponse, InspectionResponse
from backend.app.services.inspection_service import (
    ImageValidationError,
    InspectionService
)
from backend.app.services.model_service import ModelService
from ml.explainability.gradcam import GradCAMError
from ml.inference.predict import ModelNotLoadedError

router = APIRouter(prefix="", tags=["Inspection"])


@router.post(
    "/inspect",
    response_model=InspectionResponse,
    status_code=status.HTTP_200_OK,
    summary="Run AI Solar Panel Condition Inspection",
    description=(
        "Uploads a solar panel image, validates dimensions and format, executes VGG16 inference, "
        "generates Grad-CAM visual attribution maps, and returns rule-based maintenance assessments. "
        "Returns HTTP 503 if model weights are not loaded."
    ),
    responses={
        422: {"model": ErrorResponse, "description": "Image validation or decompression error"},
        503: {"model": ErrorResponse, "description": "Model weights not loaded (scientific honesty mandate)"}
    }
)
async def inspect_solar_panel(
    file: UploadFile = File(..., description="Solar panel image file (JPG, PNG, WEBP, max 10MB)"),
    ms: ModelService = Depends(get_model_service)
) -> InspectionResponse:
    # 1. Enforce scientific honesty: refuse to mock predictions if model is unavailable
    if not ms.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Model weights are not currently loaded. Per scientific honesty guidelines, "
                "real inference cannot be performed until the model is trained. "
                "Please run `python ml/training/train.py` with dataset images, or inspect the demo preview."
            )
        )

    # 2. Read file bytes
    try:
        file_bytes = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(e)}"
        )

    # 3. Process inspection
    try:
        return InspectionService.process_inspection(
            file_bytes=file_bytes,
            filename=file.filename or "uploaded_image.jpg",
            content_type=file.content_type
        )
    except ImageValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except ModelNotLoadedError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except GradCAMError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explainability failure: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected internal inspection failure: {str(e)}"
        )
