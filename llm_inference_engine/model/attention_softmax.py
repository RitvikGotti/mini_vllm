"""Numerically stable softmax for attention score rows."""

import numpy as np
from numpy.typing import NDArray


class AttentionSoftmax:
    """Convert masked attention scores into per-query probability weights."""

    def forward(
        self, masked_scores: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Return numerically stable softmax probabilities for each score row."""
        if masked_scores.ndim != 2:
            raise ValueError("masked_scores must be a two-dimensional array.")

        if not np.issubdtype(masked_scores.dtype, np.floating):
            raise ValueError("masked_scores must use a floating-point dtype.")

        if np.any(np.isnan(masked_scores)) or np.any(np.isposinf(masked_scores)):
            raise ValueError("masked_scores must not contain NaN or positive infinity.")

        if np.any(np.all(np.isneginf(masked_scores), axis=1)):
            raise ValueError("every query row must have at least one available key.")

        row_maximums = np.max(masked_scores, axis=-1, keepdims=True)
        shifted_scores = masked_scores - row_maximums
        exponentials = np.exp(shifted_scores)

        return exponentials / np.sum(exponentials, axis=-1, keepdims=True)

