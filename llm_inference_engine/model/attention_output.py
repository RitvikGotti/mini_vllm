"""Learned output projection for attention context vectors."""

import numpy as np
from numpy.typing import NDArray

from llm_inference_engine.utils.config import ModelConfig


class AttentionOutputProjection:
    """Project attention context vectors back into model hidden space."""

    def __init__(
        self,
        config: ModelConfig,
        weights: NDArray[np.floating],
    ) -> None:
        """Create an attention output projection from learned weights."""
        expected_shape = (config.hidden_size, config.hidden_size)

        if weights.shape != expected_shape:
            raise ValueError(
                "attention output weights must have shape "
                f"{expected_shape}, got {weights.shape}."
            )

        if not np.issubdtype(weights.dtype, np.floating):
            raise ValueError(
                "attention output weights must use a floating-point dtype."
            )

        self._weights = weights

    def forward(
        self, context: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Return learned projections of two-dimensional context vectors."""
        if context.ndim != 2:
            raise ValueError("context must be a two-dimensional array.")

        if context.shape[1] != self._weights.shape[0]:
            raise ValueError(
                "context feature dimension must match attention output weights."
            )

        if not np.issubdtype(context.dtype, np.floating):
            raise ValueError("context must use a floating-point dtype.")

        return context @ self._weights
