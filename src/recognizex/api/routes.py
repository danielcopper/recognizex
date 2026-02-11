"""API route definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, Request, UploadFile, status
from fastapi.responses import JSONResponse

from recognizex.api.middleware import verify_api_key
from recognizex.api.schemas import (
    DetectedFace,
    ErrorResponse,
    HealthResponse,
    ModelInfo,
    ModelsResponse,
)
from recognizex.ml.model_manager import MODEL_REGISTRY, OnnxModelManager

if TYPE_CHECKING:
    from recognizex.config import Settings
    from recognizex.ml.inference import InferencePool

router = APIRouter(prefix="/api/v1", dependencies=[Depends(verify_api_key)])


def _get_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def _get_inference_pool(request: Request) -> InferencePool:
    pool: InferencePool = request.app.state.inference_pool
    return pool


def _get_model_manager(request: Request) -> OnnxModelManager:
    manager: OnnxModelManager = request.app.state.model_manager
    return manager


@router.post(
    "/detect-faces",
    response_model=list[DetectedFace],
    responses={
        status.HTTP_501_NOT_IMPLEMENTED: {"model": ErrorResponse},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse},
    },
    summary="Detect faces in an image",
)
async def detect_faces(file: UploadFile) -> JSONResponse:
    """Detect faces and generate embeddings for an uploaded image."""
    return JSONResponse(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        content={"detail": "Face detection not yet implemented"},
    )


@router.post(
    "/classify-image",
    response_model=None,
    responses={
        status.HTTP_501_NOT_IMPLEMENTED: {"model": ErrorResponse},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse},
    },
    summary="Classify an image with tags",
)
async def classify_image(file: UploadFile) -> JSONResponse:
    """Classify an uploaded image and return ranked tags."""
    return JSONResponse(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        content={"detail": "Image classification not yet implemented"},
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
)
async def health(request: Request) -> HealthResponse:
    """Return service health status."""
    settings = _get_settings(request)
    pool = _get_inference_pool(request)
    manager = _get_model_manager(request)
    return HealthResponse(
        status="ok",
        gpu=settings.device == "cuda",
        models_loaded=manager.get_loaded_models(),
        concurrent_requests=pool.active_count,
        queue_depth=pool.queue_depth,
    )


@router.get(
    "/models",
    response_model=ModelsResponse,
    summary="List available models",
)
async def list_models(request: Request) -> ModelsResponse:
    """Return available models and their status based on current configuration."""
    settings = _get_settings(request)
    active_models = {settings.face_detection_model, settings.face_recognition_model}

    models: list[ModelInfo] = []
    for spec in MODEL_REGISTRY.values():
        if spec.name in active_models:
            model_status = "active"
        elif spec.insightface and not settings.accept_insightface_license:
            model_status = "requires_license"
        else:
            model_status = "available"

        models.append(
            ModelInfo(
                name=spec.name,
                task=spec.task,
                status=model_status,
                license=spec.license,
            )
        )

    return ModelsResponse(models=models)
