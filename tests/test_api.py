"""Tests for the RecognizeX API skeleton."""

from __future__ import annotations

import io
import os
from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

import httpx
import pytest
from fastapi import FastAPI, status

from recognizex.config import get_settings
from recognizex.main import create_app
from recognizex.ml.inference import InferencePool
from recognizex.ml.model_manager import OnnxModelManager


def _init_app_state(app: FastAPI, **env_overrides: str) -> None:
    """Manually initialize app state (ASGITransport does not trigger lifespan)."""
    with patch.dict(os.environ, env_overrides):
        settings = get_settings()
    app.state.settings = settings
    app.state.inference_pool = InferencePool(settings)
    app.state.model_manager = OnnxModelManager(settings)


async def _make_client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as ac:
        yield ac
    pool: InferencePool = app.state.inference_pool
    pool.shutdown()


@pytest.fixture()
def app() -> FastAPI:
    """Create a fresh app instance with default settings."""
    application = create_app()
    _init_app_state(application)
    return application


@pytest.fixture()
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    """Async HTTP client for testing the app."""
    async for ac in _make_client(app):
        yield ac


class TestHealthEndpoint:
    async def test_health_returns_ok(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/v1/health")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "ok"
        assert data["gpu"] is False
        assert isinstance(data["models_loaded"], list)
        assert isinstance(data["concurrent_requests"], int)
        assert isinstance(data["queue_depth"], int)

    async def test_health_gpu_true_when_cuda(self) -> None:
        cuda_app = create_app()
        _init_app_state(cuda_app, RECOGNIZEX_DEVICE="cuda")
        async for ac in _make_client(cuda_app):
            response = await ac.get("/api/v1/health")
            assert response.status_code == status.HTTP_200_OK
            assert response.json()["gpu"] is True


class TestDetectFacesEndpoint:
    async def test_detect_faces_returns_501(self, client: httpx.AsyncClient) -> None:
        fake_image = io.BytesIO(b"fake image data")
        response = await client.post(
            "/api/v1/detect-faces",
            files={"file": ("test.jpg", fake_image, "image/jpeg")},
        )
        assert response.status_code == status.HTTP_501_NOT_IMPLEMENTED
        assert "not yet implemented" in response.json()["detail"].lower()


class TestClassifyImageEndpoint:
    async def test_classify_image_returns_501(self, client: httpx.AsyncClient) -> None:
        fake_image = io.BytesIO(b"fake image data")
        response = await client.post(
            "/api/v1/classify-image",
            files={"file": ("test.jpg", fake_image, "image/jpeg")},
        )
        assert response.status_code == status.HTTP_501_NOT_IMPLEMENTED
        assert "not yet implemented" in response.json()["detail"].lower()


class TestModelsEndpoint:
    async def test_models_returns_list(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/v1/models")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "models" in data
        assert len(data["models"]) >= 2

    async def test_default_models_are_active(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/v1/models")
        models = response.json()["models"]
        active_names = {m["name"] for m in models if m["status"] == "active"}
        assert "retinaface_resnet34" in active_names
        assert "auraface_v1" in active_names

    async def test_insightface_models_require_license(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/v1/models")
        models = response.json()["models"]
        scrfd = next(m for m in models if m["name"] == "scrfd_10g_kps")
        assert scrfd["status"] == "requires_license"
        w600k = next(m for m in models if m["name"] == "w600k_r50")
        assert w600k["status"] == "requires_license"

    async def test_insightface_available_when_accepted(self) -> None:
        app = create_app()
        _init_app_state(app, RECOGNIZEX_ACCEPT_INSIGHTFACE_LICENSE="true")
        async for ac in _make_client(app):
            response = await ac.get("/api/v1/models")
            models = response.json()["models"]
            scrfd = next(m for m in models if m["name"] == "scrfd_10g_kps")
            assert scrfd["status"] == "available"


class TestAuthentication:
    async def test_no_auth_required_by_default(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/v1/health")
        assert response.status_code == status.HTTP_200_OK

    async def test_auth_required_when_api_key_set(self) -> None:
        app = create_app()
        _init_app_state(app, RECOGNIZEX_API_KEY="test-secret-key")
        async for ac in _make_client(app):
            response = await ac.get("/api/v1/health")
            assert response.status_code in (
                status.HTTP_401_UNAUTHORIZED,
                status.HTTP_403_FORBIDDEN,
            )

    async def test_auth_passes_with_correct_key(self) -> None:
        app = create_app()
        _init_app_state(app, RECOGNIZEX_API_KEY="test-secret-key")
        async for ac in _make_client(app):
            response = await ac.get(
                "/api/v1/health",
                headers={"Authorization": "Bearer test-secret-key"},
            )
            assert response.status_code == status.HTTP_200_OK

    async def test_auth_fails_with_wrong_key(self) -> None:
        app = create_app()
        _init_app_state(app, RECOGNIZEX_API_KEY="test-secret-key")
        async for ac in _make_client(app):
            response = await ac.get(
                "/api/v1/health",
                headers={"Authorization": "Bearer wrong-key"},
            )
            assert response.status_code in (
                status.HTTP_401_UNAUTHORIZED,
                status.HTTP_403_FORBIDDEN,
            )
