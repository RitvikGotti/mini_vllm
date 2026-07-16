"""Raw dot-product attention score computation."""

import numpy as np
from numpy.typing import NDArray


class AttentionScore:
    """Compute unnormalized query-to-key compatibility scores."""

    def forward(
        self,
        query: NDArray[np.floating],
        key: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Return one raw dot-product score for every query-key pair."""
        self._validate_states("query", query)
        self._validate_states("key", key)

        if query.shape[1] != key.shape[1]:
            raise ValueError("query and key feature dimensions must match.")

        return query @ key.T

    @staticmethod
    def _validate_states(name: str, states: NDArray[np.floating]) -> None:
        """Validate a two-dimensional floating-point Q or K representation."""
        if states.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional array.")

        if not np.issubdtype(states.dtype, np.floating):
            raise ValueError(f"{name} must use a floating-point dtype.")

