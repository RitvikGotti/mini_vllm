"""Single-head pre-norm transformer block composition."""

import numpy as np
from numpy.typing import NDArray

from llm_inference_engine.model.attention_sublayer import (
    SingleHeadAttentionSublayer,
)
from llm_inference_engine.model.feed_forward_sublayer import (
    FeedForwardSublayer,
)
from llm_inference_engine.utils.config import ModelConfig


class SingleHeadTransformerBlock:
    """Run pre-norm attention and feed-forward sublayers in sequence."""

    def __init__(
        self,
        config: ModelConfig,
        query_weights: NDArray[np.floating],
        key_weights: NDArray[np.floating],
        value_weights: NDArray[np.floating],
        attention_output_weights: NDArray[np.floating],
        attention_norm_scale: NDArray[np.floating],
        attention_norm_bias: NDArray[np.floating],
        ffn_input_weights: NDArray[np.floating],
        ffn_output_weights: NDArray[np.floating],
        ffn_norm_scale: NDArray[np.floating],
        ffn_norm_bias: NDArray[np.floating],
        norm_epsilon: float = 1e-5,
    ) -> None:
        """Create one pre-norm transformer block from learned parameters."""
        self._attention_sublayer = SingleHeadAttentionSublayer(
            config,
            query_weights,
            key_weights,
            value_weights,
            attention_output_weights,
            attention_norm_scale,
            attention_norm_bias,
            norm_epsilon,
        )
        self._feed_forward_sublayer = FeedForwardSublayer(
            config,
            ffn_input_weights,
            ffn_output_weights,
            ffn_norm_scale,
            ffn_norm_bias,
            norm_epsilon,
        )

    def forward(
        self, hidden_states: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Return hidden states after both pre-norm residual sublayers."""
        attention_output = self._attention_sublayer.forward(hidden_states)

        return self._feed_forward_sublayer.forward(attention_output)
