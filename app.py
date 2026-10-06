"""
SolarSentinel AI - Universal Free Cloud Container & Production Server Entrypoint

Runs the existing production FastAPI application on the platform-assigned port
(e.g., Render $PORT, Hugging Face 7860, or local 8000).
"""

import os
import sys

# Set memory-saving environment variables before importing TensorFlow
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Ensure repository root is on sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import uvicorn
from backend.app.main import app

if __name__ == "__main__":
    # Render assigns $PORT dynamically; Hugging Face default is 7860; local default is 8000
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting SolarSentinel AI on {host}:{port}...")
    uvicorn.run("backend.app.main:app", host=host, port=port, reload=False)
