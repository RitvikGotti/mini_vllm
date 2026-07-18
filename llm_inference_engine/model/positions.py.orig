"""Learned absolute positional embeddings for transformer inference."""

import numpy as np
from numpy.typing import NDArray

from llm_inference_engine.utils.config import ModelConfig


class PositionEmbedding:
    """Map sequence positions to their learned hidden-state vectors."""

    def __init__(
        self, config: ModelConfig, weights: NDArray[np.floating]
    ) -> None:
        """Create a position lookup from a model config and learned weights."""
        expected_shape = (config.max_sequence_length, config.hidden_size)

        if weights.shape != expected_shape:
            raise ValueError(
                "position weights must have shape "
                f"{expected_shape}, got {weights.shape}."
            )

        if not np.issubdtype(weights.dtype, np.floating):
            raise ValueError("position weights must use a floating-point dtype.")

        self._weights = weights

    def forward(
        self, sequence_length: int, offset: int = 0
    ) -> NDArray[np.floating]:
        """Return vectors for consecutive positions starting at offset."""
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive.")

        if offset < 0:
            raise ValueError("offset must be non-negative.")

        end_position = offset + sequence_length
        if end_position > self._weights.shape[0]:
            raise ValueError("requested positions exceed max_sequence_length.")

        return self._weights[offset:end_position]

