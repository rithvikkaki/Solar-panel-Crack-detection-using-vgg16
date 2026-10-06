"""
SolarSentinel AI - Backend Configuration
Uses Pydantic Settings for type-safe environment configuration.
"""

import os
from typing import List, Optional, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application runtime settings."""

    # Project metadata
    PROJECT_NAME: str = "SolarSentinel AI"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    APP_ENV: str = "development"

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # CORS Configuration
    # Local dev frontends + configurable production domains
    CORS_ORIGINS: Union[List[str], str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ]
    # Regex matching all Vercel deployments (production + preview)
    CORS_ORIGIN_REGEX: Optional[str] = r"^https:\/\/.*\.vercel\.app$"

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [x.strip() for x in v.split(",") if x.strip()]
        return v

    # File Upload Limits
    MAX_UPLOAD_SIZE_MB: int = 10
    ALLOWED_IMAGE_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".webp"]
    ALLOWED_IMAGE_MIMETYPES: List[str] = [
        "image/jpeg",
        "image/png",
        "image/webp"
    ]

    # Model & Metadata Paths (Resolved relative to project root)
    PROJECT_ROOT: str = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )

    MODEL_PATH: str = os.path.join(
        PROJECT_ROOT, "ml", "models", "best_vgg16.keras"
    )
    MODEL_HEAD_A_PATH: str = os.path.join(
        PROJECT_ROOT, "ml", "models", "solar_sentinel_vgg16_head_a.keras"
    )
    USE_ENSEMBLE: bool = False

    CLASS_NAMES_PATH: str = os.path.join(
        PROJECT_ROOT, "ml", "metadata", "class_names.json"
    )

    # Input resolution (Champion Optimized Model: 384x384)
    MODEL_INPUT_HEIGHT: int = 384
    MODEL_INPUT_WIDTH: int = 384

    # Health & Rule-based assessment score mappings
    HEALTH_RULES: dict = {
        "Clean": {"health_score": 98, "risk_level": "LOW", "maintenance_priority": "NONE"},
        "Dusty": {"health_score": 82, "risk_level": "LOW", "maintenance_priority": "ROUTINE"},
        "Bird-drop": {"health_score": 68, "risk_level": "MEDIUM", "maintenance_priority": "ELEVATED"},
        "Snow-Covered": {"health_score": 45, "risk_level": "MEDIUM", "maintenance_priority": "SCHEDULED"},
        "Electrical-damage": {"health_score": 25, "risk_level": "HIGH", "maintenance_priority": "URGENT"},
        "Physical-Damage": {"health_score": 15, "risk_level": "CRITICAL", "maintenance_priority": "IMMEDIATE"}
    }

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
