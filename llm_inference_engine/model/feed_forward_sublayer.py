"""Feed-forward network wrapped in its residual connection."""

import numpy as np
from numpy.typing import NDArray

from llm_inference_engine.model.feed_forward import FeedForwardNetwork
from llm_inference_engine.model.normalization import LayerNorm
from llm_inference_engine.model.residual import ResidualConnection
from llm_inference_engine.utils.config import ModelConfig


class FeedForwardSublayer:
    """Apply a pre-norm position-wise FFN and add its residual input."""

    def __init__(
        self,
        config: ModelConfig,
        input_weights: NDArray[np.floating],
        output_weights: NDArray[np.floating],
        norm_scale: NDArray[np.floating],
        norm_bias: NDArray[np.floating],
        norm_epsilon: float = 1e-5,
    ) -> None:
        """Create a pre-norm FFN sublayer from learned parameters."""
        self._normalization = LayerNorm(
            config,
            norm_scale,
            norm_bias,
            norm_epsilon,
        )
        self._feed_forward = FeedForwardNetwork(
            config,
            input_weights,
            output_weights,
        )
        self._residual = ResidualConnection()

    def forward(
        self, hidden_states: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Return input plus an FFN update from normalized hidden states."""
        normalized_states = self._normalization.forward(hidden_states)
        feed_forward_update = self._feed_forward.forward(normalized_states)

        return self._residual.forward(hidden_states, feed_forward_update)
