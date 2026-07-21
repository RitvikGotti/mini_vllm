"""Integrated multi-head causal self-attention."""

import numpy as np
from numpy.typing import NDArray

from llm_inference_engine.model.attention_heads import (
    AttentionHeadMerger,
    AttentionHeadSplitter,
)
from llm_inference_engine.model.attention_output import (
    AttentionOutputProjection,
)
from llm_inference_engine.model.attention_scaling import AttentionScoreScaler
from llm_inference_engine.model.attention_scores import AttentionScore
from llm_inference_engine.model.attention_softmax import AttentionSoftmax
from llm_inference_engine.model.attention_values import AttentionValueMixer
from llm_inference_engine.model.causal_mask import CausalMask
from llm_inference_engine.model.qkv import QKVProjection
from llm_inference_engine.utils.config import ModelConfig


class MultiHeadSelfAttention:
    """Run complete causal self-attention across every configured head."""

    def __init__(
        self,
        config: ModelConfig,
        query_weights: NDArray[np.floating],
        key_weights: NDArray[np.floating],
        value_weights: NDArray[np.floating],
        output_weights: NDArray[np.floating],
        query_bias: NDArray[np.floating] | None = None,
        key_bias: NDArray[np.floating] | None = None,
        value_bias: NDArray[np.floating] | None = None,
        output_bias: NDArray[np.floating] | None = None,
    ) -> None:
        """Create multi-head attention from its learned projections."""
        self._qkv_projection = QKVProjection(
            config,
            query_weights,
            key_weights,
            value_weights,
            query_bias,
            key_bias,
            value_bias,
        )
        self._head_splitter = AttentionHeadSplitter(config)
        self._score_computer = AttentionScore()
        self._score_scaler = AttentionScoreScaler(config.head_dim)
        self._causal_mask = CausalMask()
        self._softmax = AttentionSoftmax()
        self._value_mixer = AttentionValueMixer()
        self._head_merger = AttentionHeadMerger(config)
        self._output_projection = AttentionOutputProjection(
            config,
            output_weights,
            output_bias,
        )

    def forward(
        self,
        hidden_states: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Return projected multi-head attention output for hidden states."""
        qkv = self._qkv_projection.forward(hidden_states)
        query_heads = self._head_splitter.forward(qkv.query)
        key_heads = self._head_splitter.forward(qkv.key)
        value_heads = self._head_splitter.forward(qkv.value)

        raw_scores = self._score_computer.forward(query_heads, key_heads)
        scaled_scores = self._score_scaler.forward(raw_scores)
        masked_scores = self._causal_mask.forward(scaled_scores)
        attention_weights = self._softmax.forward(masked_scores)

        head_context = self._value_mixer.forward(
            attention_weights,
            value_heads,
        )
        merged_context = self._head_merger.forward(head_context)

        return self._output_projection.forward(merged_context)
