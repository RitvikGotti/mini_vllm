"""Integration tests for pre-norm attention with a residual connection."""

import unittest

import numpy as np

from llm_inference_engine.model.attention_sublayer import (
    SingleHeadAttentionSublayer,
)
from llm_inference_engine.utils.config import ModelConfig


class SingleHeadAttentionSublayerTests(unittest.TestCase):
    """Verify normalized attention updates are added to original inputs."""

    def test_forward_uses_pre_norm_and_preserves_residual_input(self) -> None:
        """Attention receives normalized states while residual retains input."""
        config = ModelConfig(
            vocab_size=4,
            hidden_size=2,
            num_layers=1,
            num_attention_heads=1,
            ffn_hidden_size=4,
            max_sequence_length=4,
        )
        identity = np.eye(2, dtype=np.float32)
        output_weights = np.array(
            [[0.0, 1.0], [1.0, 0.0]],
            dtype=np.float32,
        )
        sublayer = SingleHeadAttentionSublayer(
            config,
            query_weights=identity,
            key_weights=identity,
            value_weights=identity,
            output_weights=output_weights,
            norm_scale=np.ones(2, dtype=np.float32),
            norm_bias=np.zeros(2, dtype=np.float32),
        )
        hidden_states = np.array([[1.0, 0.0]], dtype=np.float32)

        output = sublayer.forward(hidden_states)

        np.testing.assert_allclose(
            output,
            np.array([[0.0, 1.0]], dtype=np.float32),
            rtol=1e-4,
            atol=1e-4,
        )
