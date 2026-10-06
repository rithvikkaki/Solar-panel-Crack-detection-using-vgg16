# SolarSentinel AI - FastAPI Backend Service

Production REST API powering the **SolarSentinel AI: Solar Panel Health & Fault Intelligence Platform**.

---

## 1. Architecture Overview

```text
backend/
├── app/
│   ├── main.py                         # FastAPI app instance, lifespan, CORS, global error handlers
│   ├── dependencies.py                 # Dependency injection (ModelService singleton)
│   ├── api/v1/
│   │   ├── health.py                   # GET /api/v1/health (operational readiness check)
│   │   ├── inspect.py                  # POST /api/v1/inspect (production inspection endpoint)
│   │   └── demo.py                     # GET /api/v1/demo/status, POST /api/v1/demo/inspect
│   ├── core/
│   │   ├── config.py                   # Pydantic BaseSettings, CORS, file limits
│   │   └── logging.py                  # Structured logging configuration
│   ├── schemas/
│   │   ├── health.py                   # Health and status Pydantic models
│   │   └── inspection.py               # Inspection request/response schemas
│   └── services/
│       ├── model_service.py            # Singleton model and Grad-CAM explainer holder
│       └── inspection_service.py       # Validation, inference orchestration, health rules
├── requirements.txt                    # Backend dependencies
└── tests/                              # Automated API and integration test suite
```

---

## 2. API Endpoints

| Method | Path | Summary | Description / Behavior |
|---|---|---|---|
| `GET` | `/` | Root Information | Service name, version, documentation links |
| `GET` | `/api/v1/health` | System Health Check | Returns service status, `model_loaded: bool`, and mode |
| `POST` | `/api/v1/inspect` | AI Inspection | Uploads image (`multipart/form-data`), runs inference & Grad-CAM. **Returns HTTP 503 if model is not loaded.** |
| `GET` | `/api/v1/demo/status` | Demo Status Report | Reports `REAL_MODEL` vs `DEMO_ONLY` mode with setup steps |
| `POST` | `/api/v1/demo/inspect` | Demo Visualization | Generates illustrative preview with explicit non-model disclaimer |
| `GET` | `/docs` | OpenAPI Swagger UI | Interactive API documentation |
| `GET` | `/redoc` | ReDoc UI | Alternative API documentation |

---

## 3. Running the Backend

Ensure the virtual environment is activated:
```bash
# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

Start the Uvicorn development server:
```bash
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

Interactive documentation will be available at:  
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
