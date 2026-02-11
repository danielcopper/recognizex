"""Environment-based configuration for RecognizeX."""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from RECOGNIZEX_* environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="RECOGNIZEX_",
        case_sensitive=False,
    )

    # Server
    host: str = "0.0.0.0"  # noqa: S104
    port: int = 8082

    # Authentication (None = disabled)
    api_key: str | None = None

    # ML device
    device: Literal["cpu", "cuda", "openvino"] = "cpu"

    # Model selection
    face_detection_model: str = "retinaface_resnet34"
    face_recognition_model: str = "auraface_v1"
    accept_insightface_license: bool = False

    # ONNX Runtime threading
    intra_op_threads: int = Field(default=0, ge=0)
    inter_op_threads: int = Field(default=1, ge=1)

    # Concurrency
    max_concurrent: int = Field(default=2, ge=1)

    # Input limits
    max_image_pixels: int = Field(default=16_777_216, ge=1)
    max_file_size: int = Field(default=209_715_200, ge=1)

    # Model management
    model_ttl: int = Field(default=300, ge=0)
    gpu_mem_limit: int = Field(default=2_147_483_648, ge=0)


def get_settings() -> Settings:
    """Create and return application settings."""
    return Settings()
