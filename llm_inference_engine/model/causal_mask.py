"""Causal masking for decoder self-attention scores."""

import numpy as np
from numpy.typing import NDArray


class CausalMask:
    """Prevent attention queries from accessing future key positions."""

    def forward(
        self,
        scaled_scores: NDArray[np.floating],
        query_offset: int = 0,
    ) -> NDArray[np.floating]:
        """Replace future-key scores with negative infinity."""
        if scaled_scores.ndim != 2:
            raise ValueError("scaled_scores must be a two-dimensional array.")

        if not np.issubdtype(scaled_scores.dtype, np.floating):
            raise ValueError("scaled_scores must use a floating-point dtype.")

        if query_offset < 0:
            raise ValueError("query_offset must be non-negative.")

        query_length, key_length = scaled_scores.shape
        if query_offset + query_length > key_length:
            raise ValueError(
                "query positions must be present in the available key positions."
            )

        query_positions = query_offset + np.arange(query_length)
        key_positions = np.arange(key_length)
        future_positions = key_positions[None, :] > query_positions[:, None]

        masked_scores = scaled_scores.copy()
        masked_scores[future_positions] = -np.inf

        return masked_scores

