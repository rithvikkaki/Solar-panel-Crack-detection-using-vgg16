# SolarSentinel AI - Universal Free Cloud Container Deployment
# Compatible with Render, Hugging Face, or standard Docker environments

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for OpenCV, image operations, and curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Optimize TensorFlow memory usage for cloud environments
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV PYTHONUNBUFFERED=1

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code, models, and metadata
COPY backend/ ./backend/
COPY ml/ ./ml/
COPY app.py .

# Ensure Git LFS model weights are binary (download from GitHub media CDN if pointer was cloned)
RUN if [ $(wc -c < ml/models/best_vgg16.keras) -lt 1000000 ]; then \
        echo "Git LFS pointer detected. Downloading 120MB weights from GitHub media CDN..."; \
        curl -L -o ml/models/best_vgg16.keras https://media.githubusercontent.com/media/rithvikkaki/SolarSentinel-AI/main/ml/models/best_vgg16.keras; \
    fi

# Expose common cloud ports
EXPOSE 7860 10000 8000

# Start via app.py which dynamically honors platform $PORT
CMD ["python", "app.py"]
