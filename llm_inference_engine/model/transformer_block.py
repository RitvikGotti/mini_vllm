"""Pre-norm transformer block composition."""

import numpy as np
from numpy.typing import NDArray

from llm_inference_engine.model.activations import (
    ActivationFunction,
    relu,
)
from llm_inference_engine.model.attention_sublayer import AttentionSublayer
from llm_inference_engine.model.feed_forward_sublayer import (
    FeedForwardSublayer,
)
from llm_inference_engine.utils.config import ModelConfig


class TransformerBlock:
    """Run multi-head attention and feed-forward sublayers in sequence."""

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
        query_bias: NDArray[np.floating] | None = None,
        key_bias: NDArray[np.floating] | None = None,
        value_bias: NDArray[np.floating] | None = None,
        attention_output_bias: NDArray[np.floating] | None = None,
        ffn_input_bias: NDArray[np.floating] | None = None,
        ffn_output_bias: NDArray[np.floating] | None = None,
        ffn_activation: ActivationFunction = relu,
    ) -> None:
        """Create one pre-norm transformer block from learned parameters."""
        self._attention_sublayer = AttentionSublayer(
            config,
            query_weights,
            key_weights,
            value_weights,
            attention_output_weights,
            attention_norm_scale,
            attention_norm_bias,
            norm_epsilon,
            query_bias,
            key_bias,
            value_bias,
            attention_output_bias,
        )
        self._feed_forward_sublayer = FeedForwardSublayer(
            config,
            ffn_input_weights,
            ffn_output_weights,
            ffn_norm_scale,
            ffn_norm_bias,
            norm_epsilon,
            input_bias=ffn_input_bias,
            output_bias=ffn_output_bias,
            activation=ffn_activation,
        )

    def forward(
        self,
        hidden_states: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Return hidden states after both pre-norm residual sublayers."""
        attention_output = self._attention_sublayer.forward(hidden_states)

        return self._feed_forward_sublayer.forward(attention_output)


class SingleHeadTransformerBlock(TransformerBlock):
    """Preserve the original transformer-block API for one head."""

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
        query_bias: NDArray[np.floating] | None = None,
        key_bias: NDArray[np.floating] | None = None,
        value_bias: NDArray[np.floating] | None = None,
        attention_output_bias: NDArray[np.floating] | None = None,
        ffn_input_bias: NDArray[np.floating] | None = None,
        ffn_output_bias: NDArray[np.floating] | None = None,
        ffn_activation: ActivationFunction = relu,
    ) -> None:
        """Create the original block while requiring exactly one head."""
        if config.num_attention_heads != 1:
            raise ValueError(
                "SingleHeadTransformerBlock requires num_attention_heads=1."
            )

        super().__init__(
            config,
            query_weights,
            key_weights,
            value_weights,
            attention_output_weights,
            attention_norm_scale,
            attention_norm_bias,
            ffn_input_weights,
            ffn_output_weights,
            ffn_norm_scale,
            ffn_norm_bias,
            norm_epsilon,
            query_bias,
            key_bias,
            value_bias,
            attention_output_bias,
            ffn_input_bias,
            ffn_output_bias,
            ffn_activation,
        )
