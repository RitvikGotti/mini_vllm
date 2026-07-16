"""Tests for learned positional embedding lookup."""

import unittest

import numpy as np

from llm_inference_engine.model.positions import PositionEmbedding
from llm_inference_engine.utils.config import ModelConfig


class PositionEmbeddingTests(unittest.TestCase):
    """Verify position lookup and sequence-boundary validation."""

    def setUp(self) -> None:
        """Create a small config and visible position matrix for each test."""
        self.config = ModelConfig(
            vocab_size=4,
            hidden_size=3,
            num_layers=1,
            num_attention_heads=1,
            ffn_hidden_size=6,
            max_sequence_length=5,
        )
        self.weights = np.array(
            [
                [0.0, 0.1, 0.2],
                [1.0, 1.1, 1.2],
                [2.0, 2.1, 2.2],
                [3.0, 3.1, 3.2],
                [4.0, 4.1, 4.2],
            ],
            dtype=np.float32,
        )

    def test_forward_looks_up_consecutive_positions(self) -> None:
        """A sequence uses consecutive rows beginning at its offset."""
        embedding = PositionEmbedding(self.config, self.weights)

        position_vectors = embedding.forward(sequence_length=2, offset=1)

        np.testing.assert_array_equal(
            position_vectors,
            np.array([[1.0, 1.1, 1.2], [2.0, 2.1, 2.2]], dtype=np.float32),
        )

    def test_wrong_weight_shape_raises_error(self) -> None:
        """The position matrix must match config sequence and hidden sizes."""
        wrong_shape_weights = np.zeros((4, 3), dtype=np.float32)

        with self.assertRaises(ValueError):
            PositionEmbedding(self.config, wrong_shape_weights)

    def test_non_positive_sequence_length_raises_error(self) -> None:
        """A position lookup must request at least one token position."""
        embedding = PositionEmbedding(self.config, self.weights)

        with self.assertRaises(ValueError):
            embedding.forward(sequence_length=0)

    def test_negative_offset_raises_error(self) -> None:
        """Position offsets cannot precede the start of a sequence."""
        embedding = PositionEmbedding(self.config, self.weights)

        with self.assertRaises(ValueError):
            embedding.forward(sequence_length=1, offset=-1)

    def test_positions_beyond_max_sequence_length_raise_error(self) -> None:
        """A lookup cannot exceed the configured maximum position."""
        embedding = PositionEmbedding(self.config, self.weights)

        with self.assertRaises(ValueError):
            embedding.forward(sequence_length=2, offset=4)

