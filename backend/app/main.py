"""
SolarSentinel AI - Main FastAPI Application
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import time
import uuid

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


from backend.app.api.v1.demo import router as demo_router
from backend.app.api.v1.health import router as health_router
from backend.app.api.v1.inspect import router as inspect_router
from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.services.model_service import model_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for application startup and shutdown."""
    logger.info("Initializing SolarSentinel AI backend application...")
    # Attempt singleton model load on startup
    loaded = model_service.load_model()
    if loaded:
        logger.info("SolarSentinel AI operational in REAL_MODEL mode.")
    else:
        logger.info("SolarSentinel AI operational in MODEL_UNAVAILABLE mode.")
    yield
    logger.info("Shutting down SolarSentinel AI backend application.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "Production AI Platform for Photovoltaic (PV) Solar Panel Condition Inspection "
        "and Explainable Defect Attribution using Deep Learning (VGG16) and Grad-CAM."
    ),
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_origin_regex=settings.CORS_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.middleware("http")
async def add_request_id_and_timing(request: Request, call_next):
    """Adds X-Request-ID and X-Process-Time headers to every response and structured logs."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = round((time.time() - start_time) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = str(process_time)
    
    logger.info(
        f"[{request_id}] {request.method} {request.url.path} "
        f"-> {response.status_code} ({process_time}ms)"
    )
    return response



# Centralized Exception Handlers
@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    """Formats HTTP exceptions consistently."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error_code": f"HTTP_{exc.status_code}",
            "detail": exc.detail,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Formats request validation errors cleanly without stack traces."""
    errors = [
        {"loc": err.get("loc", []), "msg": err.get("msg", ""), "type": err.get("type", "")}
        for err in exc.errors()
    ]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error_code": "VALIDATION_ERROR",
            "detail": "Request payload failed schema validation.",
            "errors": errors,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catches all unhandled exceptions and prevents raw stack trace leakage."""
    logger.error(f"Unhandled exception during {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error_code": "INTERNAL_SERVER_ERROR",
            "detail": "An unexpected error occurred while processing the request.",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )


# Register API Routers
app.include_router(health_router, prefix=settings.API_V1_STR)
app.include_router(inspect_router, prefix=settings.API_V1_STR)
app.include_router(demo_router, prefix=settings.API_V1_STR)


@app.get("/", tags=["Root"])
async def root():
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs_url": "/docs",
        "api_v1": settings.API_V1_STR
    }
