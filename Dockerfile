# ─────────────────────────────────────────────────────────────────
#  Stage 1: Build dependencies
# ─────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# System dependencies for DICOM + image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into a prefix
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefix=/install \
        "numpy>=1.26" \
        "pandas>=2.0" \
        "scikit-learn>=1.3" \
        "torch>=2.1" \
        "timm>=1.0" \
        "fastapi>=0.110" \
        "uvicorn[standard]>=0.27" \
        "python-multipart>=0.0.9" \
        "pillow>=10.0" \
        "opencv-python-headless>=4.8" \
        "prometheus-client>=0.20" \
        "pydantic>=2.5" \
        "pydantic-settings>=2.2" \
        "structlog>=24.1" \
        "mlflow>=2.10" \
        "omegaconf>=2.3" \
        "tqdm>=4.66"

# ─────────────────────────────────────────────────────────────────
#  Stage 2: Runtime image
# ─────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="RSNA Knee Abnormality Detection"
LABEL org.opencontainers.image.description="Medical AI inference server for knee MRI multilabel classification"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.source="https://github.com/ZitouniNidhal/knee-abnormality-detection"

# Non-root user for security
RUN groupadd --gid 1001 appuser \
    && useradd --uid 1001 --gid 1001 --no-create-home appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Runtime system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy application source
COPY src/ ./src/
COPY configs/ ./configs/

# Install project in editable mode
RUN pip install --no-cache-dir -e . --no-deps

# Create writable dirs for artifacts
RUN mkdir -p /app/logs /app/artifacts /app/mlruns \
    && chown -R appuser:appuser /app

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV LOG_FORMAT=json
ENV HOST=0.0.0.0
ENV PORT=8000
ENV WORKERS=1

CMD ["python", "-m", "rsna_knee.inference.server"]
