# Backend Dockerfile for Flask API with GPU support
# Using CUDA 12.1.0 with cuDNN 8 runtime for Ubuntu 22.04
# Note: 12.4.0 tags are not available, using stable 12.1.0 version
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# CUDA_VISIBLE_DEVICES is set via docker-compose.yml to select GPU 1 (NVIDIA)
# Inside the container, it will appear as device 0 since only one GPU is exposed

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && curl --version

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip with extended timeout
RUN pip3 install --no-cache-dir --upgrade pip

# Set environment variables for pip network resilience
ENV PIP_DEFAULT_TIMEOUT=900
ENV PIP_FIND_LINKS=""

# Install dependencies with extended timeout and retry logic
# Using a shell script for better error handling and retries
# Prefer binary packages to avoid compilation and reduce download issues
RUN MAX_ATTEMPTS=5; \
    ATTEMPT=1; \
    while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do \
        echo "=== Installation attempt $ATTEMPT of $MAX_ATTEMPTS ==="; \
        if pip3 install --no-cache-dir \
            --timeout=900 \
            --default-timeout=900 \
            --prefer-binary \
            -r requirements.txt; then \
            echo "=== Installation successful ==="; \
            exit 0; \
        else \
            echo "=== Attempt $ATTEMPT failed ==="; \
            if [ $ATTEMPT -lt $MAX_ATTEMPTS ]; then \
                WAIT_TIME=$((ATTEMPT * 15)); \
                echo "Waiting ${WAIT_TIME} seconds before retry..."; \
                sleep ${WAIT_TIME}; \
            fi; \
            ATTEMPT=$((ATTEMPT + 1)); \
        fi; \
    done; \
    echo "=== All installation attempts failed ==="; \
    exit 1

# Copy backend code
COPY . .

# Create necessary directories
RUN mkdir -p uploads qlora_models faiss_index hf_models

# Expose Flask port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Run Flask app
CMD ["python3", "app.py"]

