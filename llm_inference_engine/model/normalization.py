"""Layer normalization for transformer hidden states."""

import numpy as np
from numpy.typing import NDArray

from llm_inference_engine.utils.config import ModelConfig


class LayerNorm:
    """Normalize each token's hidden features with learned scale and bias."""

    def __init__(
        self,
        config: ModelConfig,
        scale: NDArray[np.floating],
        bias: NDArray[np.floating],
        epsilon: float = 1e-5,
    ) -> None:
        """Create layer normalization from learned per-feature parameters."""
        expected_shape = (config.hidden_size,)

        self._validate_parameter("scale", scale, expected_shape)
        self._validate_parameter("bias", bias, expected_shape)

        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")

        self._scale = scale
        self._bias = bias
        self._epsilon = epsilon

    def forward(
        self, hidden_states: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Normalize every token independently across its hidden features."""
        if hidden_states.ndim != 2:
            raise ValueError("hidden_states must be a two-dimensional array.")

        if hidden_states.shape[1] != self._scale.shape[0]:
            raise ValueError(
                "hidden_states feature dimension must match LayerNorm parameters."
            )

        if not np.issubdtype(hidden_states.dtype, np.floating):
            raise ValueError("hidden_states must use a floating-point dtype.")

        mean = np.mean(hidden_states, axis=-1, keepdims=True)
        variance = np.mean(
            (hidden_states - mean) ** 2,
            axis=-1,
            keepdims=True,
        )
        normalized_states = (hidden_states - mean) / np.sqrt(
            variance + self._epsilon
        )

        return self._scale * normalized_states + self._bias

    @staticmethod
    def _validate_parameter(
        name: str,
        parameter: NDArray[np.floating],
        expected_shape: tuple[int],
    ) -> None:
        """Validate one learned LayerNorm parameter vector."""
        if parameter.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {parameter.shape}."
            )

        if not np.issubdtype(parameter.dtype, np.floating):
            raise ValueError(f"{name} must use a floating-point dtype.")
