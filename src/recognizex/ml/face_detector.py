"""Face detection model stub.

Implementations: RetinaFace (ResNet34, MobileNetV2), SCRFD (opt-in).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


@dataclass(frozen=True)
class RawDetection:
    """Raw face detection result before coordinate normalization.

    Coordinates are in pixel space of the preprocessed input image.
    """

    bbox: NDArray[np.float32]
    score: float
    landmarks: NDArray[np.float32]


class FaceDetector(Protocol):
    """Protocol for face detection models."""

    @property
    def model_name(self) -> str:
        """Return the model identifier string."""
        ...

    def detect(self, image: NDArray[np.uint8]) -> list[RawDetection]:
        """Detect faces in an image.

        Args:
            image: HxWx3 RGB uint8 array.

        Returns:
            List of raw detections with bounding boxes, scores, and landmarks.
        """
        ...
