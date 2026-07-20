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
        bias: NDArray[np.floating] | None = None,
    ) -> None:
        """Create an attention output projection from weights and bias."""
        expected_shape = (config.hidden_size, config.hidden_size)
        expected_bias_shape = (config.hidden_size,)

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
        if bias is None:
            self._bias = np.zeros(
                expected_bias_shape,
                dtype=weights.dtype,
            )
        else:
            if bias.shape != expected_bias_shape:
                raise ValueError(
                    "attention output bias must have shape "
                    f"{expected_bias_shape}, got {bias.shape}."
                )

            if not np.issubdtype(bias.dtype, np.floating):
                raise ValueError(
                    "attention output bias must use a floating-point dtype."
                )

            self._bias = bias

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

        return context @ self._weights + self._bias
