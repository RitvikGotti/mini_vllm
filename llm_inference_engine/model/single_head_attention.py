"""Integrated single-head causal self-attention."""

import numpy as np
from numpy.typing import NDArray

from llm_inference_engine.model.attention_scaling import AttentionScoreScaler
from llm_inference_engine.model.attention_scores import AttentionScore
from llm_inference_engine.model.attention_softmax import AttentionSoftmax
from llm_inference_engine.model.attention_values import AttentionValueMixer
from llm_inference_engine.model.causal_mask import CausalMask
from llm_inference_engine.model.qkv import QKVProjection
from llm_inference_engine.utils.config import ModelConfig


class SingleHeadSelfAttention:
    """Run the complete causal attention calculation for one attention head."""

    def __init__(
        self,
        config: ModelConfig,
        query_weights: NDArray[np.floating],
        key_weights: NDArray[np.floating],
        value_weights: NDArray[np.floating],
    ) -> None:
        """Create single-head attention from its learned Q, K, and V weights."""
        if config.num_attention_heads != 1:
            raise ValueError(
                "SingleHeadSelfAttention requires num_attention_heads=1."
            )

        self._qkv_projection = QKVProjection(
            config,
            query_weights,
            key_weights,
            value_weights,
        )
        self._score_computer = AttentionScore()
        self._score_scaler = AttentionScoreScaler(config.head_dim)
        self._causal_mask = CausalMask()
        self._softmax = AttentionSoftmax()
        self._value_mixer = AttentionValueMixer()

    def forward(
        self, hidden_states: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Return causal single-head context vectors for hidden states."""
        qkv = self._qkv_projection.forward(hidden_states)
        raw_scores = self._score_computer.forward(qkv.query, qkv.key)
        scaled_scores = self._score_scaler.forward(raw_scores)
        masked_scores = self._causal_mask.forward(scaled_scores)
        attention_weights = self._softmax.forward(masked_scores)

        return self._value_mixer.forward(attention_weights, qkv.value)
