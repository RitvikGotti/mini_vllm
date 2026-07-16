"""Tests for input embedding composition."""

import unittest

import numpy as np

from llm_inference_engine.model.embeddings import TokenEmbedding
from llm_inference_engine.model.input_embeddings import InputEmbedding
from llm_inference_engine.model.positions import PositionEmbedding
from llm_inference_engine.utils.config import ModelConfig


class InputEmbeddingTests(unittest.TestCase):
    """Verify token and positional vectors are combined correctly."""

    def setUp(self) -> None:
        """Create small, visible lookup tables for each test."""
        config = ModelConfig(
            vocab_size=3,
            hidden_size=2,
            num_layers=1,
            num_attention_heads=1,
            ffn_hidden_size=4,
            max_sequence_length=4,
        )
        token_embedding = TokenEmbedding(
            config,
            np.array(
                [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
                dtype=np.float32,
            ),
        )
        position_embedding = PositionEmbedding(
            config,
            np.array(
                [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]],
                dtype=np.float32,
            ),
        )
        self.embedding = InputEmbedding(token_embedding, position_embedding)

    def test_forward_adds_token_and_position_vectors(self) -> None:
        """Each hidden state is the elementwise sum of its two vectors."""
        hidden_states = self.embedding.forward(
            np.array([1, 0], dtype=np.int64)
        )

        np.testing.assert_allclose(
            hidden_states,
            np.array([[2.1, 20.2], [1.3, 10.4]], dtype=np.float32),
        )

    def test_forward_applies_position_offset(self) -> None:
        """The position offset selects later vectors in the position table."""
        hidden_states = self.embedding.forward(
            np.array([2], dtype=np.int64),
            position_offset=2,
        )

        np.testing.assert_allclose(
            hidden_states,
            np.array([[3.5, 30.6]], dtype=np.float32),
        )

