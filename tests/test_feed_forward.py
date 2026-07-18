"""Tests for the position-wise feed-forward network."""

import unittest

import numpy as np

from llm_inference_engine.model.feed_forward import FeedForwardNetwork
from llm_inference_engine.utils.config import ModelConfig


class FeedForwardNetworkTests(unittest.TestCase):
    """Verify expansion, ReLU activation, and output projection."""

    def setUp(self) -> None:
        """Create a small feed-forward configuration for each test."""
        self.config = ModelConfig(
            vocab_size=4,
            hidden_size=2,
            num_layers=1,
            num_attention_heads=1,
            ffn_hidden_size=3,
            max_sequence_length=4,
        )
        self.input_weights = np.array(
            [[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]],
            dtype=np.float32,
        )
        self.output_weights = np.array(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            dtype=np.float32,
        )

    def test_forward_processes_each_token_independently(self) -> None:
        """Every token follows expansion, ReLU, and projection separately."""
        feed_forward = FeedForwardNetwork(
            self.config,
            self.input_weights,
            self.output_weights,
        )
        hidden_states = np.array(
            [[2.0, 1.0], [1.0, 2.0]],
            dtype=np.float32,
        )

        output = feed_forward.forward(hidden_states)

        np.testing.assert_array_equal(
            output,
            np.array([[2.0, 4.0], [4.0, 6.0]], dtype=np.float32),
        )

    def test_wrong_input_weight_shape_raises_error(self) -> None:
        """The first matrix must expand hidden size to FFN hidden size."""
        with self.assertRaises(ValueError):
            FeedForwardNetwork(
                self.config,
                np.ones((2, 2), dtype=np.float32),
                self.output_weights,
            )

    def test_wrong_output_weight_shape_raises_error(self) -> None:
        """The second matrix must project FFN width back to hidden size."""
        with self.assertRaises(ValueError):
            FeedForwardNetwork(
                self.config,
                self.input_weights,
                np.ones((3, 3), dtype=np.float32),
            )

    def test_wrong_hidden_state_width_raises_error(self) -> None:
        """FFN input must have the configured model hidden width."""
        feed_forward = FeedForwardNetwork(
            self.config,
            self.input_weights,
            self.output_weights,
        )

        with self.assertRaises(ValueError):
            feed_forward.forward(np.ones((2, 3), dtype=np.float32))
