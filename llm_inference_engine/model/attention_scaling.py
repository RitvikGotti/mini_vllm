"""Scaled dot-product attention score computation."""

import math

import numpy as np
from numpy.typing import NDArray


class AttentionScoreScaler:
    """Scale raw attention scores by the square root of one head's width."""

    def __init__(self, head_dim: int) -> None:
        """Create a score scaler for attention heads of the given width."""
        if head_dim <= 0:
            raise ValueError("head_dim must be positive.")

        self._scale = math.sqrt(head_dim)

    def forward(self, raw_scores: NDArray[np.floating]) -> NDArray[np.floating]:
        """Return raw attention scores divided by the configured scale."""
        if raw_scores.ndim != 2:
            raise ValueError("raw_scores must be a two-dimensional array.")

        if not np.issubdtype(raw_scores.dtype, np.floating):
            raise ValueError("raw_scores must use a floating-point dtype.")

        return raw_scores / self._scale

