"""Tests for raw attention score computation."""

import unittest

import numpy as np

from llm_inference_engine.model.attention_heads import AttentionHeadSplitter
from llm_inference_engine.model.attention_scores import AttentionScore
from llm_inference_engine.utils.config import ModelConfig


class AttentionScoreTests(unittest.TestCase):
    """Verify query-key dot products and input validation."""

    def setUp(self) -> None:
        """Create a score computer for each test."""
        self.score_computer = AttentionScore()

    def test_forward_computes_one_score_per_query_key_pair(self) -> None:
        """The output row and column counts follow query and key lengths."""
        query = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        key = np.array(
            [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]],
            dtype=np.float32,
        )

        scores = self.score_computer.forward(query, key)

        np.testing.assert_array_equal(
            scores,
            np.array([[1.0, 4.0, 3.0], [3.0, 8.0, 7.0]], dtype=np.float32),
        )

    def test_mismatched_feature_dimensions_raise_error(self) -> None:
        """Query and key vectors need the same width for a dot product."""
        query = np.ones((2, 2), dtype=np.float32)
        key = np.ones((3, 3), dtype=np.float32)

        with self.assertRaises(ValueError):
            self.score_computer.forward(query, key)

    def test_non_two_dimensional_query_raises_error(self) -> None:
        """A score computation receives one query vector per position."""
        query = np.ones(2, dtype=np.float32)
        key = np.ones((2, 2), dtype=np.float32)

        with self.assertRaises(ValueError):
            self.score_computer.forward(query, key)

    def test_non_floating_key_raises_error(self) -> None:
        """Attention states must contain continuous floating-point features."""
        query = np.ones((2, 2), dtype=np.float32)
        key = np.ones((2, 2), dtype=np.int64)

        with self.assertRaises(ValueError):
            self.score_computer.forward(query, key)

    def test_forward_computes_scores_independently_for_each_head(self) -> None:
        """Every attention head receives its own query-by-key score matrix."""
        config = ModelConfig(
            vocab_size=4,
            hidden_size=2,
            num_layers=1,
            num_attention_heads=2,
            ffn_hidden_size=4,
            max_sequence_length=8,
        )
        projected_states = np.array(
            [[1.0, 0.0], [0.1, 1.0], [1.0, 1.1]],
            dtype=np.float32,
        )
        split_states = AttentionHeadSplitter(config).forward(
            projected_states
        )

        scores = self.score_computer.forward(
            split_states,
            split_states,
        )

        expected = np.array(
            [
                [
                    [1.0, 0.1, 1.0],
                    [0.1, 0.01, 0.1],
                    [1.0, 0.1, 1.0],
                ],
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.1],
                    [0.0, 1.1, 1.21],
                ],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(scores, expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(scores.shape, (2, 3, 3))

    def test_mismatched_head_counts_raise_error(self) -> None:
        """Multi-head Q and K tensors need the same number of heads."""
        query = np.ones((2, 3, 1), dtype=np.float32)
        key = np.ones((3, 3, 1), dtype=np.float32)

        with self.assertRaises(ValueError):
            self.score_computer.forward(query, key)
