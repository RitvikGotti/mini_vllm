"""Single-head attention wrapped in its residual connection."""

import numpy as np
from numpy.typing import NDArray

from llm_inference_engine.model.residual import ResidualConnection
from llm_inference_engine.model.single_head_attention import (
    SingleHeadSelfAttention,
)
from llm_inference_engine.utils.config import ModelConfig


class SingleHeadAttentionSublayer:
    """Apply single-head self-attention and add its residual input."""

    def __init__(
        self,
        config: ModelConfig,
        query_weights: NDArray[np.floating],
        key_weights: NDArray[np.floating],
        value_weights: NDArray[np.floating],
        output_weights: NDArray[np.floating],
    ) -> None:
        """Create an attention sublayer from its learned projection weights."""
        self._attention = SingleHeadSelfAttention(
            config,
            query_weights,
            key_weights,
            value_weights,
            output_weights,
        )
        self._residual = ResidualConnection()

    def forward(
        self, hidden_states: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Return hidden states plus their projected attention update."""
        attention_update = self._attention.forward(hidden_states)

        return self._residual.forward(hidden_states, attention_update)
