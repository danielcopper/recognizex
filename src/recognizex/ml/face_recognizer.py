"""Face recognition (embedding) model stub.

Implementations: AuraFace v1 (default), ArcFace w600k_r50 (opt-in).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class FaceRecognizer(Protocol):
    """Protocol for face recognition (embedding) models."""

    @property
    def model_name(self) -> str:
        """Return the model identifier string."""
        ...

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimensionality (e.g., 512)."""
        ...

    def get_embeddings(self, face_crops: NDArray[np.float32]) -> NDArray[np.float32]:
        """Generate embeddings for a batch of aligned face crops.

        Args:
            face_crops: Batch of preprocessed face images, shape (N, 3, 112, 112).

        Returns:
            L2-normalized embedding vectors, shape (N, embedding_dim).
        """
        ...
