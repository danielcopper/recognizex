"""Image preprocessing pipeline stub.

This module will handle format detection, decoding, EXIF orientation,
color space conversion, size validation, downscaling, and conversion
to numpy arrays for model input.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class ImagePreprocessor(Protocol):
    """Protocol for image preprocessing."""

    def decode_image(self, image_bytes: bytes) -> NDArray[np.uint8]:
        """Decode raw image bytes into an RGB uint8 numpy array.

        Args:
            image_bytes: Raw file bytes (any supported format).

        Returns:
            HxWx3 RGB uint8 numpy array.

        Raises:
            ValueError: If the image cannot be decoded or exceeds size limits.
        """
        ...

    def preprocess_for_detection(self, image: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Prepare an image for the face detection model.

        Args:
            image: HxWx3 RGB uint8 array.

        Returns:
            Model-specific preprocessed tensor.
        """
        ...

    def preprocess_for_recognition(
        self, image: NDArray[np.uint8], landmarks: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Align and crop a face for the recognition model.

        Args:
            image: HxWx3 RGB uint8 array (original image).
            landmarks: 5x2 float32 array of facial landmarks.

        Returns:
            Preprocessed face tensor.
        """
        ...
