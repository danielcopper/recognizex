"""Model manager stub for download, loading, caching, and TTL eviction.

This module will handle downloading models from HuggingFace, loading ONNX
InferenceSession objects, caching in memory, TTL eviction, and license
checking for InsightFace models.
"""

from __future__ import annotations

from typing import Protocol


class ModelManager(Protocol):
    """Protocol for model lifecycle management."""

    def ensure_model(self, model_name: str) -> str:
        """Ensure a model is downloaded and return its file path.

        Downloads from HuggingFace if not present locally. Checks license
        acceptance for InsightFace models.

        Args:
            model_name: The model identifier (e.g., 'retinaface_resnet34').

        Returns:
            Absolute path to the ONNX model file.

        Raises:
            RuntimeError: If the model requires license acceptance not given.
            OSError: If the download fails.
        """
        ...

    def get_loaded_models(self) -> list[str]:
        """Return names of currently loaded models."""
        ...

    def unload_idle_models(self) -> None:
        """Unload models that have exceeded their TTL."""
        ...
