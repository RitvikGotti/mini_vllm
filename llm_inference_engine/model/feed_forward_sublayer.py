"""Feed-forward network wrapped in its residual connection."""

import numpy as np
from numpy.typing import NDArray

from llm_inference_engine.model.feed_forward import FeedForwardNetwork
from llm_inference_engine.model.residual import ResidualConnection
from llm_inference_engine.utils.config import ModelConfig


class FeedForwardSublayer:
    """Apply a position-wise FFN and add its residual input."""

    def __init__(
        self,
        config: ModelConfig,
        input_weights: NDArray[np.floating],
        output_weights: NDArray[np.floating],
    ) -> None:
        """Create an FFN sublayer from its learned weight matrices."""
        self._feed_forward = FeedForwardNetwork(
            config,
            input_weights,
            output_weights,
        )
        self._residual = ResidualConnection()

    def forward(
        self, hidden_states: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Return hidden states plus their position-wise FFN update."""
        feed_forward_update = self._feed_forward.forward(hidden_states)

        return self._residual.forward(hidden_states, feed_forward_update)
