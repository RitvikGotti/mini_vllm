"""Position-wise feed-forward network for transformer inference."""

import numpy as np
from numpy.typing import NDArray

from llm_inference_engine.model.activations import (
    ActivationFunction,
    relu,
)
from llm_inference_engine.utils.config import ModelConfig


class FeedForwardNetwork:
    """Expand, activate, and project each token hidden state independently."""

    def __init__(
        self,
        config: ModelConfig,
        input_weights: NDArray[np.floating],
        output_weights: NDArray[np.floating],
        activation: ActivationFunction = relu,
    ) -> None:
        """Create an FFN from learned matrices and an activation function."""
        expected_input_shape = (
            config.hidden_size,
            config.ffn_hidden_size,
        )
        expected_output_shape = (
            config.ffn_hidden_size,
            config.hidden_size,
        )

        self._validate_weights(
            "input_weights",
            input_weights,
            expected_input_shape,
        )
        self._validate_weights(
            "output_weights",
            output_weights,
            expected_output_shape,
        )

        self._input_weights = input_weights
        self._output_weights = output_weights
        self._activation = activation

    def forward(
        self, hidden_states: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Return position-wise expanded, activated, and projected states."""
        if hidden_states.ndim != 2:
            raise ValueError("hidden_states must be a two-dimensional array.")

        if hidden_states.shape[1] != self._input_weights.shape[0]:
            raise ValueError(
                "hidden_states feature dimension must match input_weights."
            )

        if not np.issubdtype(hidden_states.dtype, np.floating):
            raise ValueError("hidden_states must use a floating-point dtype.")

        expanded_states = hidden_states @ self._input_weights
        activated_states = self._activation(expanded_states)

        return activated_states @ self._output_weights

    @staticmethod
    def _validate_weights(
        name: str,
        weights: NDArray[np.floating],
        expected_shape: tuple[int, int],
    ) -> None:
        """Validate one learned feed-forward weight matrix."""
        if weights.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {weights.shape}."
            )

        if not np.issubdtype(weights.dtype, np.floating):
            raise ValueError(f"{name} must use a floating-point dtype.")
