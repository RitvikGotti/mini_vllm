"""Integration tests for the FFN with a residual connection."""

import unittest

import numpy as np

from llm_inference_engine.model.feed_forward_sublayer import (
    FeedForwardSublayer,
)
from llm_inference_engine.utils.config import ModelConfig


class FeedForwardSublayerTests(unittest.TestCase):
    """Verify position-wise FFN updates are added to their inputs."""

    def test_forward_adds_ffn_update_to_original_hidden_states(self) -> None:
        """The residual path preserves input while the FFN adds an update."""
        config = ModelConfig(
            vocab_size=4,
            hidden_size=2,
            num_layers=1,
            num_attention_heads=1,
            ffn_hidden_size=3,
            max_sequence_length=4,
        )
        input_weights = np.array(
            [[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]],
            dtype=np.float32,
        )
        output_weights = np.array(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            dtype=np.float32,
        )
        sublayer = FeedForwardSublayer(
            config,
            input_weights,
            output_weights,
        )
        hidden_states = np.array(
            [[2.0, 1.0], [1.0, 2.0]],
            dtype=np.float32,
        )

        output = sublayer.forward(hidden_states)

        np.testing.assert_array_equal(
            output,
            np.array([[4.0, 5.0], [5.0, 8.0]], dtype=np.float32),
        )
