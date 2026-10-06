"""
SolarSentinel AI - Health and System Status Schemas
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """System health check payload."""
    service: str = Field("SolarSentinel AI", description="Service identifier")
    status: str = Field("online", description="Operational status of backend service")
    model_loaded: bool = Field(False, description="True if trained Keras weights are loaded into memory")
    model_mode: str = Field("MODEL_UNAVAILABLE", description="Current execution mode: REAL_MODEL or MODEL_UNAVAILABLE")
    version: str = Field("1.0.0", description="API version")
    verified_classes: List[str] = Field(default_factory=list, description="Verified dataset class names")
    timestamp: str = Field(..., description="UTC ISO timestamp of health check")
    message: str = Field(..., description="Human-readable operational status message")


class DemoStatusResponse(BaseModel):
    """Explicit demonstration capability report."""
    mode: str = Field(..., description="REAL_MODEL or DEMO_ONLY")
    model_available: bool = Field(..., description="Whether trained model is present")
    message: str = Field(..., description="Explanatory text regarding current capability")
    setup_instructions: Optional[List[str]] = Field(
        None,
        description="Step-by-step instructions to activate REAL_MODEL mode"
    )


class ReadyResponse(BaseModel):
    """Service readiness payload for orchestrators and load balancers."""
    ready: bool = Field(..., description="Whether service is ready to perform real inference")
    model_loaded: bool = Field(..., description="Whether model weights are loaded into memory")
    model_path: str = Field(..., description="Sanitized model identifier or path")
    mode: str = Field(..., description="Execution mode: REAL_MODEL or MODEL_UNAVAILABLE")
    timestamp: str = Field(..., description="UTC ISO timestamp")

