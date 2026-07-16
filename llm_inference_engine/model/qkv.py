"""Query, key, and value projections for single-head attention."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from llm_inference_engine.utils.config import ModelConfig


@dataclass(frozen=True)
class QueryKeyValue:
    """Hold the three projected representations used by attention."""

    query: NDArray[np.floating]
    key: NDArray[np.floating]
    value: NDArray[np.floating]


class QKVProjection:
    """Project hidden states into separate query, key, and value spaces."""

    def __init__(
        self,
        config: ModelConfig,
        query_weights: NDArray[np.floating],
        key_weights: NDArray[np.floating],
        value_weights: NDArray[np.floating],
    ) -> None:
        """Create Q, K, and V projections from learned weight matrices."""
        expected_shape = (config.hidden_size, config.hidden_size)

        self._validate_weights("query_weights", query_weights, expected_shape)
        self._validate_weights("key_weights", key_weights, expected_shape)
        self._validate_weights("value_weights", value_weights, expected_shape)

        self._query_weights = query_weights
        self._key_weights = key_weights
        self._value_weights = value_weights

    def forward(self, hidden_states: NDArray[np.floating]) -> QueryKeyValue:
        """Return Q, K, and V projections for two-dimensional hidden states."""
        if hidden_states.ndim != 2:
            raise ValueError("hidden_states must be a two-dimensional array.")

        if hidden_states.shape[1] != self._query_weights.shape[0]:
            raise ValueError(
                "hidden_states feature dimension must match projection weights."
            )

        if not np.issubdtype(hidden_states.dtype, np.floating):
            raise ValueError("hidden_states must use a floating-point dtype.")

        return QueryKeyValue(
            query=hidden_states @ self._query_weights,
            key=hidden_states @ self._key_weights,
            value=hidden_states @ self._value_weights,
        )

    @staticmethod
    def _validate_weights(
        name: str,
        weights: NDArray[np.floating],
        expected_shape: tuple[int, int],
    ) -> None:
        """Validate one learned Q, K, or V projection weight matrix."""
        if weights.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {weights.shape}."
            )

        if not np.issubdtype(weights.dtype, np.floating):
            raise ValueError(f"{name} must use a floating-point dtype.")

