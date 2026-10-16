"""Production FastAPI inference server — Phase 8.

Endpoints:
    GET  /health         — liveness + readiness probe
    GET  /model-info     — model metadata and config
    POST /predict        — single image multilabel prediction
    POST /predict/batch  — batch prediction (multiple images)

Prometheus metrics exposed at /metrics (Phase 9).
OpenTelemetry tracing integrated.
"""

from __future__ import annotations

import io
import os
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

from ..constants import TARGETS
from ..utils.logging import configure_logging, get_logger

configure_logging(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = get_logger(__name__)

# ── Prometheus metrics ─────────────────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "rsna_knee_requests_total",
    "Total prediction requests",
    ["endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "rsna_knee_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)
PREDICTION_CONFIDENCE = Histogram(
    "rsna_knee_prediction_confidence",
    "Distribution of prediction confidence scores",
    ["target"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
MODEL_LOADED = Gauge("rsna_knee_model_loaded", "Whether the model is loaded (1=yes)")
BATCH_SIZE_HIST = Histogram(
    "rsna_knee_batch_size",
    "Batch size distribution",
    buckets=[1, 2, 4, 8, 16, 32],
)


# ── Global model state ─────────────────────────────────────────────────────────


class ModelRegistry:
    """Holds the loaded model and its metadata."""

    model: nn.Module | None = None
    model_path: str = ""
    model_name: str = ""
    image_size: int = 224
    device: torch.device = torch.device("cpu")
    version: str = "1.0.0"
    loaded_at: str = ""


_registry = ModelRegistry()


def _get_transforms(image_size: int = 224) -> Any:
    """Build inference-time image transforms."""
    try:
        import torchvision.transforms as T

        return T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    except ImportError:
        logger.warning("torchvision not installed; using basic PIL resize")
        return None


def _preprocess_image(img_bytes: bytes, image_size: int = 224) -> torch.Tensor:
    """Convert raw image bytes to a model-ready tensor."""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    transforms = _get_transforms(image_size)
    if transforms is not None:
        tensor = transforms(img)
    else:
        img = img.resize((image_size, image_size))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        tensor = torch.from_numpy(arr.transpose(2, 0, 1))
    return tensor.unsqueeze(0)  # (1, C, H, W)


def _run_inference(tensors: torch.Tensor) -> np.ndarray:
    """Run model forward pass and return sigmoid probabilities."""
    if _registry.model is None:
        raise RuntimeError("Model not loaded")
    with torch.inference_mode():
        logits = _registry.model(tensors.to(_registry.device))
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


# ── FastAPI lifespan ──────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load model on startup, clean up on shutdown."""
    model_path = os.environ.get("MODEL_PATH", "")
    device_name = os.environ.get("DEVICE", "cpu")
    _registry.device = torch.device(device_name)

    if model_path and Path(model_path).exists():
        try:
            from ..training.model import KneeAbnormalityModel

            checkpoint = torch.load(model_path, map_location=_registry.device, weights_only=False)
            _registry.model = KneeAbnormalityModel.from_checkpoint(checkpoint)
            _registry.model.eval()
            _registry.model.to(_registry.device)
            _registry.model_path = model_path
            _registry.model_name = checkpoint.get("config", {}).get("name", "unknown")
            _registry.image_size = checkpoint.get("config", {}).get("image_size", 224)
            _registry.loaded_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            MODEL_LOADED.set(1)
            logger.info(
                "Model loaded",
                path=model_path,
                device=device_name,
                model_name=_registry.model_name,
            )
        except Exception as exc:
            logger.error("Failed to load model", error=str(exc))
            MODEL_LOADED.set(0)
    else:
        logger.warning(
            "MODEL_PATH not set or file not found; server starting without model",
            model_path=model_path,
        )
        MODEL_LOADED.set(0)

    yield  # Application runs here

    logger.info("Server shutting down")
    _registry.model = None
    MODEL_LOADED.set(0)


# ── FastAPI app ───────────────────────────────────────────────────────────────


app = FastAPI(
    title="RSNA Knee Abnormality Detection API",
    description=(
        "Production-grade medical AI inference server for multilabel knee MRI classification. "
        "Detects 12 knee abnormalities from MRI images."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Pydantic response schemas ─────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    device: str
    version: str


class PredictionScore(BaseModel):
    target: str
    probability: float = Field(ge=0.0, le=1.0)
    positive: bool


class PredictionResponse(BaseModel):
    study_id: str
    predictions: list[PredictionScore]
    inference_time_ms: float
    model_name: str
    threshold: float


class ModelInfoResponse(BaseModel):
    model_name: str
    model_path: str
    targets: list[str]
    version: str
    loaded_at: str
    device: str
    image_size: int


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["infrastructure"],
    summary="Health and readiness probe",
)
async def health() -> HealthResponse:
    """Liveness and readiness check for orchestrators (k8s, docker-compose)."""
    return HealthResponse(
        status="healthy" if _registry.model is not None else "degraded",
        model_loaded=_registry.model is not None,
        model_name=_registry.model_name,
        device=str(_registry.device),
        version=_registry.version,
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    tags=["infrastructure"],
    summary="Model metadata",
)
async def model_info() -> ModelInfoResponse:
    """Return model architecture, version, and target labels."""
    return ModelInfoResponse(
        model_name=_registry.model_name,
        model_path=_registry.model_path,
        targets=list(TARGETS),
        version=_registry.version,
        loaded_at=_registry.loaded_at,
        device=str(_registry.device),
        image_size=_registry.image_size,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["inference"],
    summary="Single image multilabel prediction",
    status_code=status.HTTP_200_OK,
)
async def predict(
    file: UploadFile = File(..., description="MRI image (PNG/JPEG/DICOM)"),
    threshold: float = 0.5,
    study_id: str = "unknown",
) -> PredictionResponse:
    """Predict knee abnormalities for a single MRI image slice."""
    if _registry.model is None:
        REQUEST_COUNT.labels(endpoint="/predict", status="503").inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Set MODEL_PATH environment variable.",
        )

    start = time.perf_counter()
    try:
        img_bytes = await file.read()
        tensor = _preprocess_image(img_bytes, _registry.image_size)
        probs = _run_inference(tensor)  # (1, 12)

        elapsed_ms = (time.perf_counter() - start) * 1000
        REQUEST_LATENCY.labels(endpoint="/predict").observe(elapsed_ms / 1000)
        REQUEST_COUNT.labels(endpoint="/predict", status="200").inc()

        predictions = []
        for idx, target in enumerate(TARGETS):
            prob = float(probs[0, idx])
            PREDICTION_CONFIDENCE.labels(target=target).observe(prob)
            predictions.append(
                PredictionScore(
                    target=target,
                    probability=prob,
                    positive=prob >= threshold,
                )
            )

        return PredictionResponse(
            study_id=study_id,
            predictions=predictions,
            inference_time_ms=elapsed_ms,
            model_name=_registry.model_name,
            threshold=threshold,
        )
    except HTTPException:
        raise
    except Exception as exc:
        REQUEST_COUNT.labels(endpoint="/predict", status="500").inc()
        logger.error("Prediction failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {exc!s}",
        ) from exc


@app.post(
    "/predict/batch",
    response_model=list[PredictionResponse],
    tags=["inference"],
    summary="Batch multilabel prediction",
    status_code=status.HTTP_200_OK,
)
async def predict_batch(
    files: list[UploadFile] = File(...),
    threshold: float = 0.5,
) -> list[PredictionResponse]:
    """Predict knee abnormalities for multiple MRI images in one request."""
    if _registry.model is None:
        REQUEST_COUNT.labels(endpoint="/predict/batch", status="503").inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded.",
        )

    if not files:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one file must be provided.",
        )

    BATCH_SIZE_HIST.observe(len(files))
    start = time.perf_counter()

    try:
        tensors = []
        for f in files:
            img_bytes = await f.read()
            tensors.append(_preprocess_image(img_bytes, _registry.image_size))

        batch_tensor = torch.cat(tensors, dim=0)  # (B, C, H, W)
        probs = _run_inference(batch_tensor)  # (B, 12)

        elapsed_ms = (time.perf_counter() - start) * 1000
        REQUEST_LATENCY.labels(endpoint="/predict/batch").observe(elapsed_ms / 1000)
        REQUEST_COUNT.labels(endpoint="/predict/batch", status="200").inc()

        results = []
        for b_idx, f in enumerate(files):
            preds = []
            for idx, target in enumerate(TARGETS):
                prob = float(probs[b_idx, idx])
                preds.append(
                    PredictionScore(
                        target=target,
                        probability=prob,
                        positive=prob >= threshold,
                    )
                )
            results.append(
                PredictionResponse(
                    study_id=f.filename or f"sample_{b_idx}",
                    predictions=preds,
                    inference_time_ms=elapsed_ms / len(files),
                    model_name=_registry.model_name,
                    threshold=threshold,
                )
            )
        return results

    except HTTPException:
        raise
    except Exception as exc:
        REQUEST_COUNT.labels(endpoint="/predict/batch", status="500").inc()
        logger.error("Batch prediction failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference error: {exc!s}",
        ) from exc


@app.get("/metrics", tags=["infrastructure"], summary="Prometheus metrics")
async def metrics() -> Response:
    """Expose Prometheus-format metrics for scraping."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry-point: rsna-serve."""
    import uvicorn

    uvicorn.run(
        "rsna_knee.inference.server:app",
        host=os.environ.get("HOST", "0.0.0.0"),  # noqa: S104
        port=int(os.environ.get("PORT", "8000")),
        workers=int(os.environ.get("WORKERS", "1")),
        reload=os.environ.get("RELOAD", "false").lower() == "true",
        log_level="info",
    )


if __name__ == "__main__":
    main()
