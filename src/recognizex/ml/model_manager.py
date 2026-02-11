"""Model manager: download, load, cache, and evict ONNX models.

Handles downloading models from HuggingFace, creating and caching ONNX
InferenceSessions, TTL-based eviction, and InsightFace license gating.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from huggingface_hub import hf_hub_download
from onnxruntime import InferenceSession, SessionOptions
from onnxruntime.capi.onnxruntime_pybind11_state import ExecutionMode

if TYPE_CHECKING:
    from recognizex.config import Settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol (kept for test mocking)
# ---------------------------------------------------------------------------


class ModelManager(Protocol):
    """Protocol for model lifecycle management."""

    def ensure_downloaded(self, model_name: str) -> Path:
        """Ensure a model is downloaded and return its file path."""
        ...

    def get_session(self, model_name: str) -> InferenceSession:
        """Return a cached or newly created InferenceSession."""
        ...

    def get_loaded_models(self) -> list[str]:
        """Return names of currently loaded models."""
        ...

    def unload_idle_models(self) -> None:
        """Unload models that have exceeded their TTL."""
        ...

    def shutdown(self) -> None:
        """Clear all cached sessions."""
        ...


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


class ModelTask(StrEnum):
    FACE_DETECTION = "face_detection"
    FACE_RECOGNITION = "face_recognition"


@dataclass(frozen=True)
class ModelSpec:
    """Static metadata for a single ONNX model."""

    name: str
    repo_id: str
    filename: str
    subfolder: str | None
    task: ModelTask
    license: str
    insightface: bool


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "retinaface_resnet34": ModelSpec(
        name="retinaface_resnet34",
        repo_id="danielcopper/recognizex-models",
        filename="retinaface_resnet34.onnx",
        subfolder=None,
        task=ModelTask.FACE_DETECTION,
        license="MIT",
        insightface=False,
    ),
    "retinaface_mobilenetv2": ModelSpec(
        name="retinaface_mobilenetv2",
        repo_id="danielcopper/recognizex-models",
        filename="retinaface_mobilenetv2.onnx",
        subfolder=None,
        task=ModelTask.FACE_DETECTION,
        license="MIT",
        insightface=False,
    ),
    "scrfd_10g_kps": ModelSpec(
        name="scrfd_10g_kps",
        repo_id="fal/AuraFace-v1",
        filename="scrfd_10g_bnkps.onnx",
        subfolder=None,
        task=ModelTask.FACE_DETECTION,
        license="Non-commercial (InsightFace)",
        insightface=True,
    ),
    "auraface_v1": ModelSpec(
        name="auraface_v1",
        repo_id="fal/AuraFace-v1",
        filename="glintr100.onnx",
        subfolder=None,
        task=ModelTask.FACE_RECOGNITION,
        license="Apache-2.0",
        insightface=False,
    ),
    "w600k_r50": ModelSpec(
        name="w600k_r50",
        repo_id="public-data/insightface",
        filename="w600k_r50.onnx",
        subfolder="models/buffalo_l",
        task=ModelTask.FACE_RECOGNITION,
        license="Non-commercial (InsightFace)",
        insightface=True,
    ),
}


# ---------------------------------------------------------------------------
# Concrete implementation
# ---------------------------------------------------------------------------


@dataclass
class _CachedSession:
    session: InferenceSession
    last_used: float


class OnnxModelManager:
    """Downloads, loads, caches, and evicts ONNX inference sessions."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._models_dir = Path(settings.models_dir)
        self._models_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._sessions: dict[str, _CachedSession] = {}
        self._model_paths: dict[str, Path] = {}

        self._providers = self._build_providers()
        self._session_options = self._build_session_options()

    # -- Public API ---------------------------------------------------------

    def ensure_downloaded(self, model_name: str) -> Path:
        """Download a model from HuggingFace if not already present locally."""
        spec = self._get_spec(model_name)
        self._check_license(spec)

        if model_name in self._model_paths:
            path = self._model_paths[model_name]
            if path.exists():
                return path

        downloaded = Path(
            hf_hub_download(
                repo_id=spec.repo_id,
                filename=spec.filename,
                subfolder=spec.subfolder,
                local_dir=str(self._models_dir),
            )
        )
        self._model_paths[model_name] = downloaded
        logger.info("Downloaded %s to %s", model_name, downloaded)
        return downloaded

    def get_session(self, model_name: str) -> InferenceSession:
        """Return a cached InferenceSession, creating one if needed."""
        with self._lock:
            cached = self._sessions.get(model_name)
            if cached is not None:
                cached.last_used = time.monotonic()
                return cached.session

        model_path = self.ensure_downloaded(model_name)
        session = InferenceSession(
            str(model_path),
            sess_options=self._session_options,
            providers=self._providers,
        )

        with self._lock:
            # Double-check: another thread may have created it while we loaded.
            existing = self._sessions.get(model_name)
            if existing is not None:
                existing.last_used = time.monotonic()
                return existing.session
            self._sessions[model_name] = _CachedSession(
                session=session,
                last_used=time.monotonic(),
            )
            logger.info("Loaded session for %s", model_name)
            return session

    def get_loaded_models(self) -> list[str]:
        """Return names of models with active sessions."""
        with self._lock:
            return list(self._sessions.keys())

    def unload_idle_models(self) -> None:
        """Remove sessions that have exceeded the configured TTL."""
        ttl = self._settings.model_ttl
        if ttl == 0:
            return

        now = time.monotonic()
        with self._lock:
            expired = [name for name, cached in self._sessions.items() if (now - cached.last_used) > ttl]
            for name in expired:
                del self._sessions[name]
                logger.info("Evicted idle session for %s", name)

    def shutdown(self) -> None:
        """Clear all cached sessions."""
        with self._lock:
            self._sessions.clear()
            logger.info("All model sessions cleared")

    # -- Internal -----------------------------------------------------------

    @staticmethod
    def _get_spec(model_name: str) -> ModelSpec:
        try:
            return MODEL_REGISTRY[model_name]
        except KeyError:
            raise KeyError(f"Unknown model: {model_name}") from None

    def _check_license(self, spec: ModelSpec) -> None:
        if spec.insightface and not self._settings.accept_insightface_license:
            raise RuntimeError(f"Model '{spec.name}' requires RECOGNIZEX_ACCEPT_INSIGHTFACE_LICENSE=true")

    def _build_providers(self) -> list[str | tuple[str, dict[str, object]]]:
        device = self._settings.device
        if device == "cuda":
            return [
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        "gpu_mem_limit": self._settings.gpu_mem_limit,
                        "arena_extend_strategy": "kSameAsRequested",
                    },
                ),
                "CPUExecutionProvider",
            ]
        if device == "openvino":
            return [
                ("OpenVINOExecutionProvider", {"device_type": "CPU"}),
                "CPUExecutionProvider",
            ]
        return ["CPUExecutionProvider"]

    def _build_session_options(self) -> SessionOptions:
        opts = SessionOptions()
        opts.intra_op_num_threads = self._settings.intra_op_threads
        opts.inter_op_num_threads = self._settings.inter_op_threads
        opts.execution_mode = ExecutionMode.ORT_SEQUENTIAL
        opts.enable_mem_pattern = True
        opts.enable_mem_reuse = True

        if self._settings.device == "openvino":
            # OpenVINO does its own graph optimization
            from onnxruntime import GraphOptimizationLevel

            opts.graph_optimization_level = GraphOptimizationLevel.ORT_DISABLE_ALL
        return opts
