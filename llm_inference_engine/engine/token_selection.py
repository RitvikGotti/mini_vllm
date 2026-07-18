"""Token-selection strategies for model vocabulary logits."""

import numpy as np
from numpy.typing import NDArray


class GreedyTokenSelector:
    """Choose the highest-logit token at the final sequence position."""

    def select(self, logits: NDArray[np.floating]) -> int:
        """Return the vocabulary ID with the largest final-position logit."""
        if logits.ndim != 2:
            raise ValueError("logits must be a two-dimensional array.")

        if not np.issubdtype(logits.dtype, np.floating):
            raise ValueError("logits must use a floating-point dtype.")

        if logits.shape[0] == 0:
            raise ValueError("logits must contain at least one sequence position.")

        if logits.shape[1] == 0:
            raise ValueError("logits must contain at least one vocabulary token.")

        if not np.all(np.isfinite(logits)):
            raise ValueError("logits must contain only finite values.")

        return int(np.argmax(logits[-1]))
