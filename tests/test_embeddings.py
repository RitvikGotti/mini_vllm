"""Tests for token embedding lookup."""

import unittest

import numpy as np

from llm_inference_engine.model.embeddings import TokenEmbedding
from llm_inference_engine.utils.config import ModelConfig


class TokenEmbeddingTests(unittest.TestCase):
    """Verify embedding shape validation and token-ID lookup."""

    def setUp(self) -> None:
        """Create a small config and visible embedding matrix for each test."""
        self.config = ModelConfig(
            vocab_size=4,
            hidden_size=3,
            num_layers=1,
            num_attention_heads=1,
            ffn_hidden_size=6,
            max_sequence_length=8,
        )
        self.weights = np.array(
            [
                [0.0, 0.1, 0.2],
                [1.0, 1.1, 1.2],
                [2.0, 2.1, 2.2],
                [3.0, 3.1, 3.2],
            ],
            dtype=np.float32,
        )

    def test_forward_looks_up_embedding_rows(self) -> None:
        """Each token ID selects the matching row of the embedding matrix."""
        embedding = TokenEmbedding(self.config, self.weights)

        hidden_states = embedding.forward(np.array([2, 0], dtype=np.int64))

        np.testing.assert_array_equal(
            hidden_states,
            np.array([[2.0, 2.1, 2.2], [0.0, 0.1, 0.2]], dtype=np.float32),
        )

    def test_wrong_weight_shape_raises_error(self) -> None:
        """The embedding matrix must match config vocabulary and hidden sizes."""
        wrong_shape_weights = np.zeros((4, 4), dtype=np.float32)

        with self.assertRaises(ValueError):
            TokenEmbedding(self.config, wrong_shape_weights)

    def test_out_of_range_token_id_raises_error(self) -> None:
        """A lookup ID must identify a row in the embedding matrix."""
        embedding = TokenEmbedding(self.config, self.weights)

        with self.assertRaises(ValueError):
            embedding.forward(np.array([4], dtype=np.int64))

    def test_non_one_dimensional_token_ids_raise_error(self) -> None:
        """A single embedding lookup sequence must be one-dimensional."""
        embedding = TokenEmbedding(self.config, self.weights)

        with self.assertRaises(ValueError):
            embedding.forward(np.array([[0, 1]], dtype=np.int64))

