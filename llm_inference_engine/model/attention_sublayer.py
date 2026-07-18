"""Single-head attention wrapped in its residual connection."""

import numpy as np
from numpy.typing import NDArray

from llm_inference_engine.model.normalization import LayerNorm
from llm_inference_engine.model.residual import ResidualConnection
from llm_inference_engine.model.single_head_attention import (
    SingleHeadSelfAttention,
)
from llm_inference_engine.utils.config import ModelConfig


class SingleHeadAttentionSublayer:
    """Apply pre-norm single-head attention and add its residual input."""

    def __init__(
        self,
        config: ModelConfig,
        query_weights: NDArray[np.floating],
        key_weights: NDArray[np.floating],
        value_weights: NDArray[np.floating],
        output_weights: NDArray[np.floating],
        norm_scale: NDArray[np.floating],
        norm_bias: NDArray[np.floating],
        norm_epsilon: float = 1e-5,
    ) -> None:
        """Create a pre-norm attention sublayer from learned parameters."""
        self._normalization = LayerNorm(
            config,
            norm_scale,
            norm_bias,
            norm_epsilon,
        )
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
        """Return input plus attention applied to normalized hidden states."""
        normalized_states = self._normalization.forward(hidden_states)
        attention_update = self._attention.forward(normalized_states)

        return self._residual.forward(hidden_states, attention_update)
