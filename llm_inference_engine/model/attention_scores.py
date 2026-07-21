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

        if query.ndim != key.ndim:
            raise ValueError(
                "query and key must have the same number of dimensions."
            )

        if query.shape[-1] != key.shape[-1]:
            raise ValueError("query and key feature dimensions must match.")

        if query.ndim == 3 and query.shape[0] != key.shape[0]:
            raise ValueError("query and key head counts must match.")

        if query.ndim == 2:
            return query @ key.T

        return query @ key.transpose(0, 2, 1)

    @staticmethod
    def _validate_states(name: str, states: NDArray[np.floating]) -> None:
        """Validate a single-head or multi-head Q or K representation."""
        if states.ndim not in (2, 3):
            raise ValueError(
                f"{name} must be a two- or three-dimensional array."
            )

        if not np.issubdtype(states.dtype, np.floating):
            raise ValueError(f"{name} must use a floating-point dtype.")

