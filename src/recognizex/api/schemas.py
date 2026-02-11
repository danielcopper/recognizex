"""Pydantic request/response schemas for the RecognizeX API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FaceAngle(BaseModel):
    """Pose angles for a detected face."""

    roll: float
    yaw: float
    pitch: float


class DetectedFace(BaseModel):
    """A single detected face with bounding box, score, embedding, and pose."""

    x: float = Field(description="Relative bounding box x position (0.0-1.0)")
    y: float = Field(description="Relative bounding box y position (0.0-1.0)")
    width: float = Field(description="Relative bounding box width (0.0-1.0)")
    height: float = Field(description="Relative bounding box height (0.0-1.0)")
    score: float = Field(description="Detection confidence (0.0-1.0)")
    vector: list[float] = Field(description="Face embedding vector (512 dimensions)")
    angle: FaceAngle


class ImageTag(BaseModel):
    """A single classification tag with confidence score."""

    label: str
    confidence: float = Field(ge=0.0, le=1.0)


class ClassifyImageResponse(BaseModel):
    """Response for image classification endpoint."""

    tags: list[ImageTag]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    gpu: bool
    models_loaded: list[str]
    concurrent_requests: int
    queue_depth: int


class ModelInfo(BaseModel):
    """Information about an available model."""

    name: str
    task: str = Field(description="Model task: 'face_detection', 'face_recognition', or 'image_classification'")
    status: str = Field(description="Model status: 'active', 'available', or 'requires_license'")
    license: str


class ModelsResponse(BaseModel):
    """Response for the models listing endpoint."""

    models: list[ModelInfo]


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str
