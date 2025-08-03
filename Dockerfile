# AI Communication App Dockerfile
# Supports both CPU and GPU inference

ARG CUDA_VERSION=11.8
ARG UBUNTU_VERSION=22.04

# Base image selection - GPU version
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} as gpu-base

# Base image selection - CPU version  
FROM ubuntu:${UBUNTU_VERSION} as cpu-base

# Common base setup
FROM ${BUILD_TARGET:-gpu-base} as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    curl \
    wget \
    git \
    build-essential \
    cmake \
    pkg-config \
    libsndfile1 \
    portaudio19-dev \
    libasound2-dev \
    libpulse-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy Poetry configuration
COPY pyproject.toml poetry.lock* ./

# Configure Poetry
RUN poetry config virtualenvs.create false

# Install dependencies
# For GPU build, install with CUDA support
ARG BUILD_TARGET=gpu-base
RUN if [ "$BUILD_TARGET" = "gpu-base" ]; then \
        poetry install --no-dev && \
        pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118; \
    else \
        poetry install --no-dev && \
        pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu; \
    fi

# Install SeamlessM4T
RUN pip install git+https://github.com/facebookresearch/seamless_communication.git

# Copy application code
COPY src/ ./src/
COPY main.py ./

# Create cache directories
RUN mkdir -p /root/.cache/ai_communication_app \
    /root/.cache/huggingface \
    /root/.cache/torch

# Set environment variables for model caching
ENV MODEL_CACHE_DIR=/root/.cache/ai_communication_app
ENV HF_HOME=/root/.cache/huggingface
ENV TORCH_HOME=/root/.cache/torch

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app /root/.cache
USER appuser

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command
CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "8501"]

# Labels
LABEL maintainer="AI Communication App Team"
LABEL version="0.1.0"
LABEL description="AI English conversation app using SeamlessM4T v2"