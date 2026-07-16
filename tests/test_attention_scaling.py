"""Tests for scaled attention scores."""

import unittest

import numpy as np

from llm_inference_engine.model.attention_scaling import AttentionScoreScaler


class AttentionScoreScalerTests(unittest.TestCase):
    """Verify score scaling and input validation."""

    def test_forward_divides_scores_by_square_root_of_head_dim(self) -> None:
        """A four-dimensional head scales every score by two."""
        scaler = AttentionScoreScaler(head_dim=4)
        raw_scores = np.array(
            [[2.0, 4.0], [6.0, 8.0]],
            dtype=np.float32,
        )

        scaled_scores = scaler.forward(raw_scores)

        np.testing.assert_array_equal(
            scaled_scores,
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        )

    def test_non_positive_head_dim_raises_error(self) -> None:
        """A head must have at least one feature."""
        with self.assertRaises(ValueError):
            AttentionScoreScaler(head_dim=0)

    def test_non_two_dimensional_scores_raise_error(self) -> None:
        """Scaling operates on a query-by-key score matrix."""
        scaler = AttentionScoreScaler(head_dim=2)

        with self.assertRaises(ValueError):
            scaler.forward(np.ones(2, dtype=np.float32))

    def test_non_floating_scores_raise_error(self) -> None:
        """Scores must be continuous values before softmax normalization."""
        scaler = AttentionScoreScaler(head_dim=2)

        with self.assertRaises(ValueError):
            scaler.forward(np.ones((2, 2), dtype=np.int64))

