"""
SolarSentinel AI - FastAPI Dependency Injection
"""

from backend.app.services.model_service import ModelService, model_service


def get_model_service() -> ModelService:
    """Dependency providing singleton ModelService."""
    return model_service
