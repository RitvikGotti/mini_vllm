"""Integration tests for complete multi-head causal self-attention."""

import unittest

import numpy as np

from llm_inference_engine.model.multi_head_attention import (
    MultiHeadSelfAttention,
)
from llm_inference_engine.utils.config import ModelConfig


class MultiHeadSelfAttentionTests(unittest.TestCase):
    """Verify the complete split-to-merge attention path."""

    def test_forward_runs_two_heads_from_qkv_to_output_projection(self) -> None:
        """Each head attends independently before their contexts are merged."""
        config = ModelConfig(
            vocab_size=4,
            hidden_size=2,
            num_layers=1,
            num_attention_heads=2,
            ffn_hidden_size=4,
            max_sequence_length=8,
        )
        identity = np.eye(2, dtype=np.float32)
        attention = MultiHeadSelfAttention(
            config,
            query_weights=identity,
            key_weights=identity,
            value_weights=identity,
            output_weights=identity,
        )
        hidden_states = np.array(
            [[1.0, 0.0], [0.1, 1.0], [1.0, 1.1]],
            dtype=np.float32,
        )

        output = attention.forward(hidden_states)

        expected = np.array(
            [
                [1.0, 0.0],
                [0.5702363, 0.7310586],
                [0.84795254, 0.9096652],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(output, expected, rtol=1e-6, atol=1e-6)
