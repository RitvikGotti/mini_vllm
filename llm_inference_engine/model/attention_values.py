"""Weighted value aggregation for attention."""

import numpy as np
from numpy.typing import NDArray


class AttentionValueMixer:
    """Combine value vectors using per-query attention weights."""

    def forward(
        self,
        attention_weights: NDArray[np.floating],
        value: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Return one weighted combination of values for every query."""
        self._validate_matrix("attention_weights", attention_weights)
        self._validate_matrix("value", value)

        if attention_weights.shape[1] != value.shape[0]:
            raise ValueError(
                "attention weight columns must match the number of value rows."
            )

        return attention_weights @ value

    @staticmethod
    def _validate_matrix(name: str, matrix: NDArray[np.floating]) -> None:
        """Validate one two-dimensional floating-point attention matrix."""
        if matrix.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional array.")

        if not np.issubdtype(matrix.dtype, np.floating):
            raise ValueError(f"{name} must use a floating-point dtype.")
