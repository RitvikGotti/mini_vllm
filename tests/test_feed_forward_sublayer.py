"""Integration tests for the pre-norm FFN residual sublayer."""

import unittest

import numpy as np

from llm_inference_engine.model.feed_forward_sublayer import (
    FeedForwardSublayer,
)
from llm_inference_engine.utils.config import ModelConfig


class FeedForwardSublayerTests(unittest.TestCase):
    """Verify normalized FFN updates are added to original inputs."""

    def test_forward_uses_pre_norm_and_preserves_residual_input(self) -> None:
        """The FFN receives normalized states while residual retains input."""
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
            norm_scale=np.ones(2, dtype=np.float32),
            norm_bias=np.zeros(2, dtype=np.float32),
        )
        hidden_states = np.array(
            [[2.0, 1.0], [1.0, 2.0]],
            dtype=np.float32,
        )

        output = sublayer.forward(hidden_states)

        np.testing.assert_allclose(
            output,
            np.array([[8.0, 9.0], [7.0, 10.0]], dtype=np.float32),
            rtol=1e-4,
            atol=1e-4,
        )
