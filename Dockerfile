# SolarSentinel AI - Hugging Face Spaces Docker Deployment
# Platform: Hugging Face Spaces Free Tier (CPU: 2 vCPU, 16GB RAM)
# Exposed Port: 7860 (Hugging Face default)

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for OpenCV and image operations
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code, models, and metadata
COPY backend/ ./backend/
COPY ml/ ./ml/
COPY app.py .

# Hugging Face Spaces routes to port 7860
EXPOSE 7860

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "7860"]
