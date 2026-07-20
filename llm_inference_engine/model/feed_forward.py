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
        input_bias: NDArray[np.floating] | None = None,
        output_bias: NDArray[np.floating] | None = None,
    ) -> None:
        """Create an FFN from learned projections, biases, and activation."""
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

        if input_bias is None:
            input_bias = np.zeros(
                config.ffn_hidden_size,
                dtype=input_weights.dtype,
            )
        else:
            self._validate_bias(
                "input_bias",
                input_bias,
                (config.ffn_hidden_size,),
            )

        if output_bias is None:
            output_bias = np.zeros(
                config.hidden_size,
                dtype=output_weights.dtype,
            )
        else:
            self._validate_bias(
                "output_bias",
                output_bias,
                (config.hidden_size,),
            )

        self._input_weights = input_weights
        self._output_weights = output_weights
        self._activation = activation
        self._input_bias = input_bias
        self._output_bias = output_bias

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

        expanded_states = (
            hidden_states @ self._input_weights
            + self._input_bias
        )
        activated_states = self._activation(expanded_states)

        return (
            activated_states @ self._output_weights
            + self._output_bias
        )

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

    @staticmethod
    def _validate_bias(
        name: str,
        bias: NDArray[np.floating],
        expected_shape: tuple[int],
    ) -> None:
        """Validate one learned feed-forward bias vector."""
        if bias.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {bias.shape}."
            )

        if not np.issubdtype(bias.dtype, np.floating):
            raise ValueError(f"{name} must use a floating-point dtype.")
