"""Tests for query, key, and value projections."""

import unittest

import numpy as np

from llm_inference_engine.model.qkv import QKVProjection
from llm_inference_engine.utils.config import ModelConfig


class QKVProjectionTests(unittest.TestCase):
    """Verify independent Q, K, and V learned projections."""

    def setUp(self) -> None:
        """Create a small hidden size and visible projection matrices."""
        self.config = ModelConfig(
            vocab_size=4,
            hidden_size=2,
            num_layers=1,
            num_attention_heads=1,
            ffn_hidden_size=4,
            max_sequence_length=4,
        )
        self.query_weights = np.array(
            [[1.0, 0.0], [0.0, 1.0]],
            dtype=np.float32,
        )
        self.key_weights = np.array(
            [[1.0, 0.0], [0.0, 2.0]],
            dtype=np.float32,
        )
        self.value_weights = np.array(
            [[0.0, 1.0], [1.0, 0.0]],
            dtype=np.float32,
        )

    def test_forward_returns_three_independent_projections(self) -> None:
        """The same hidden states receive three different learned transforms."""
        projection = QKVProjection(
            self.config,
            self.query_weights,
            self.key_weights,
            self.value_weights,
        )
        hidden_states = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        qkv = projection.forward(hidden_states)

        np.testing.assert_array_equal(
            qkv.query,
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            qkv.key,
            np.array([[1.0, 4.0], [3.0, 8.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            qkv.value,
            np.array([[2.0, 1.0], [4.0, 3.0]], dtype=np.float32),
        )

    def test_wrong_weight_shape_raises_error(self) -> None:
        """Every projection matrix must match the hidden-state width."""
        with self.assertRaises(ValueError):
            QKVProjection(
                self.config,
                np.ones((2, 2), dtype=np.float32),
                np.ones((3, 2), dtype=np.float32),
                np.ones((2, 2), dtype=np.float32),
            )

    def test_wrong_hidden_state_width_raises_error(self) -> None:
        """Projection input must have the configured hidden-state width."""
        projection = QKVProjection(
            self.config,
            self.query_weights,
            self.key_weights,
            self.value_weights,
        )

        with self.assertRaises(ValueError):
            projection.forward(np.ones((2, 3), dtype=np.float32))

    def test_non_two_dimensional_hidden_states_raise_error(self) -> None:
        """A projection receives one hidden vector per sequence position."""
        projection = QKVProjection(
            self.config,
            self.query_weights,
            self.key_weights,
            self.value_weights,
        )

        with self.assertRaises(ValueError):
            projection.forward(np.ones(2, dtype=np.float32))

