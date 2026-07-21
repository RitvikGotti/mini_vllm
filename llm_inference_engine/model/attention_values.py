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
        self._validate_states("attention_weights", attention_weights)
        self._validate_states("value", value)

        if attention_weights.ndim != value.ndim:
            raise ValueError(
                "attention_weights and value must have the same number of dimensions."
            )

        if attention_weights.shape[-1] != value.shape[-2]:
            raise ValueError(
                "attention weight columns must match the number of value rows."
            )

        if (
            attention_weights.ndim == 3
            and attention_weights.shape[0] != value.shape[0]
        ):
            raise ValueError("attention weight and value head counts must match.")

        return attention_weights @ value

    @staticmethod
    def _validate_states(name: str, states: NDArray[np.floating]) -> None:
        """Validate single-head or multi-head floating-point attention states."""
        if states.ndim not in (2, 3):
            raise ValueError(
                f"{name} must be a two- or three-dimensional array."
            )

        if not np.issubdtype(states.dtype, np.floating):
            raise ValueError(f"{name} must use a floating-point dtype.")
