"""
SolarSentinel AI - Hugging Face Spaces & Production Server Entrypoint

Runs the existing production FastAPI application on the platform-assigned port
(default 7860 for Hugging Face Spaces, or 8000 for local development).
"""

import os
import sys

# Ensure repository root is on sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import uvicorn
from backend.app.main import app

if __name__ == "__main__":
    # Hugging Face Spaces routes traffic to port 7860 by default
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting SolarSentinel AI on {host}:{port}...")
    uvicorn.run("backend.app.main:app", host=host, port=port, reload=False)
