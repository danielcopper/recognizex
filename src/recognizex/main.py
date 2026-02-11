"""FastAPI application entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from recognizex.api.routes import router
from recognizex.config import get_settings
from recognizex.ml.inference import InferencePool

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: initialize on startup, clean up on shutdown."""
    settings = get_settings()
    app.state.settings = settings

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    logger.info(
        "Starting RecognizeX (device=%s, max_concurrent=%s, detection=%s, recognition=%s)",
        settings.device,
        settings.max_concurrent,
        settings.face_detection_model,
        settings.face_recognition_model,
    )

    inference_pool = InferencePool(settings)
    app.state.inference_pool = inference_pool

    logger.info("RecognizeX ready")
    yield

    logger.info("Shutting down RecognizeX")
    inference_pool.shutdown()
    logger.info("RecognizeX shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title="RecognizeX",
        description="Standalone ML inference API for face detection and recognition",
        version="0.1.0",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(router)
    return application


app = create_app()
