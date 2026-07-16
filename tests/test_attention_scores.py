"""Tests for raw attention score computation."""

import unittest

import numpy as np

from llm_inference_engine.model.attention_scores import AttentionScore


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

