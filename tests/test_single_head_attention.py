"""Integration tests for single-head causal self-attention."""

import unittest

import numpy as np

from llm_inference_engine.model.single_head_attention import (
    SingleHeadSelfAttention,
)
from llm_inference_engine.utils.config import ModelConfig


class SingleHeadSelfAttentionTests(unittest.TestCase):
    """Verify the complete QKV-to-output attention data flow."""

    def test_forward_runs_the_full_attention_pipeline(self) -> None:
        """Computed scores flow through softmax, values, and output projection."""
        config = ModelConfig(
            vocab_size=4,
            hidden_size=2,
            num_layers=1,
            num_attention_heads=1,
            ffn_hidden_size=4,
            max_sequence_length=4,
        )
        identity = np.eye(2, dtype=np.float32)
        attention = SingleHeadSelfAttention(
            config,
            query_weights=identity,
            key_weights=identity,
            value_weights=identity,
            output_weights=identity,
        )
        hidden_states = np.array(
            [[1.0, 0.0], [0.0, 1.0]],
            dtype=np.float32,
        )

        output = attention.forward(hidden_states)

        expected = np.array(
            [[1.0, 0.0], [0.33023846, 0.66976154]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(output, expected, rtol=1e-6, atol=1e-6)

    def test_forward_applies_attention_output_projection(self) -> None:
        """The weighted context flows through the learned output weights."""
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
        attention = SingleHeadSelfAttention(
            config,
            query_weights=identity,
            key_weights=identity,
            value_weights=identity,
            output_weights=output_weights,
        )

        output = attention.forward(
            np.array([[1.0, 0.0]], dtype=np.float32)
        )

        np.testing.assert_array_equal(
            output,
            np.array([[0.0, 1.0]], dtype=np.float32),
        )

    def test_multiple_attention_heads_raise_error(self) -> None:
        """This milestone must not silently perform multi-head attention."""
        config = ModelConfig(
            vocab_size=4,
            hidden_size=4,
            num_layers=1,
            num_attention_heads=2,
            ffn_hidden_size=8,
            max_sequence_length=4,
        )
        weights = np.eye(4, dtype=np.float32)

        with self.assertRaises(ValueError):
            SingleHeadSelfAttention(
                config,
                query_weights=weights,
                key_weights=weights,
                value_weights=weights,
                output_weights=weights,
            )
