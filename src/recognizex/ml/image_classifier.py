"""Image classification model stub (Phase 2+).

This will handle object/scene tagging using models like EfficientNet or YOLO via ONNX.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


@dataclass(frozen=True)
class ClassificationResult:
    """A single classification prediction."""

    label: str
    confidence: float


class ImageClassifier(Protocol):
    """Protocol for image classification models."""

    @property
    def model_name(self) -> str:
        """Return the model identifier string."""
        ...

    def classify(self, image: NDArray[np.uint8]) -> list[ClassificationResult]:
        """Classify an image and return ranked tags.

        Args:
            image: HxWx3 RGB uint8 array.

        Returns:
            List of classification results sorted by confidence (descending).
        """
        ...
