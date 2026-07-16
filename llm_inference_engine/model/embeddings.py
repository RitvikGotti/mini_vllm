"""Token embedding lookup for transformer inference."""

import numpy as np
from numpy.typing import NDArray

from llm_inference_engine.utils.config import ModelConfig


class TokenEmbedding:
    """Map token IDs to their learned hidden-state vectors."""

    def __init__(
        self, config: ModelConfig, weights: NDArray[np.floating]
    ) -> None:
        """Create an embedding lookup from a model config and learned weights."""
        expected_shape = (config.vocab_size, config.hidden_size)

        if weights.shape != expected_shape:
            raise ValueError(
                "embedding weights must have shape "
                f"{expected_shape}, got {weights.shape}."
            )

        if not np.issubdtype(weights.dtype, np.floating):
            raise ValueError("embedding weights must use a floating-point dtype.")

        self._weights = weights

    def forward(self, token_ids: NDArray[np.integer]) -> NDArray[np.floating]:
        """Return one hidden-state vector for each one-dimensional token ID."""
        if token_ids.ndim != 1:
            raise ValueError("token_ids must be a one-dimensional array.")

        if not np.issubdtype(token_ids.dtype, np.integer):
            raise ValueError("token_ids must use an integer dtype.")

        if token_ids.size and (
            np.any(token_ids < 0)
            or np.any(token_ids >= self._weights.shape[0])
        ):
            raise ValueError("token_ids must be valid vocabulary IDs.")

        return self._weights[token_ids]

