"""Multi-head attention wrapped in its residual connection."""

import numpy as np
from numpy.typing import NDArray

from llm_inference_engine.model.multi_head_attention import (
    MultiHeadSelfAttention,
)
from llm_inference_engine.model.normalization import LayerNorm
from llm_inference_engine.model.residual import ResidualConnection
from llm_inference_engine.utils.config import ModelConfig


class AttentionSublayer:
    """Apply pre-norm multi-head attention and add its residual input."""

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
        query_bias: NDArray[np.floating] | None = None,
        key_bias: NDArray[np.floating] | None = None,
        value_bias: NDArray[np.floating] | None = None,
        output_bias: NDArray[np.floating] | None = None,
    ) -> None:
        """Create a pre-norm attention sublayer from learned parameters."""
        self._normalization = LayerNorm(
            config,
            norm_scale,
            norm_bias,
            norm_epsilon,
        )
        self._attention = MultiHeadSelfAttention(
            config,
            query_weights,
            key_weights,
            value_weights,
            output_weights,
            query_bias,
            key_bias,
            value_bias,
            output_bias,
        )
        self._residual = ResidualConnection()

    def forward(
        self,
        hidden_states: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Return input plus attention applied to normalized hidden states."""
        normalized_states = self._normalization.forward(hidden_states)
        attention_update = self._attention.forward(normalized_states)

        return self._residual.forward(hidden_states, attention_update)


class SingleHeadAttentionSublayer(AttentionSublayer):
    """Preserve the original attention-sublayer API for one head."""

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
        query_bias: NDArray[np.floating] | None = None,
        key_bias: NDArray[np.floating] | None = None,
        value_bias: NDArray[np.floating] | None = None,
        output_bias: NDArray[np.floating] | None = None,
    ) -> None:
        """Create the original sublayer while requiring exactly one head."""
        if config.num_attention_heads != 1:
            raise ValueError(
                "SingleHeadAttentionSublayer requires num_attention_heads=1."
            )

        super().__init__(
            config,
            query_weights,
            key_weights,
            value_weights,
            output_weights,
            norm_scale,
            norm_bias,
            norm_epsilon,
            query_bias,
            key_bias,
            value_bias,
            output_bias,
        )
