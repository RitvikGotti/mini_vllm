"""Tests for attention softmax."""

import unittest

import numpy as np

from llm_inference_engine.model.attention_softmax import AttentionSoftmax


class AttentionSoftmaxTests(unittest.TestCase):
    """Verify stable attention-probability normalization."""

    def setUp(self) -> None:
        """Create a row-wise attention softmax for each test."""
        self.softmax = AttentionSoftmax()

    def test_forward_converts_masked_scores_to_attention_weights(self) -> None:
        """Each row sums to one and masked positions receive zero weight."""
        masked_scores = np.array(
            [[1.0, 2.0, -np.inf], [0.0, -np.inf, -np.inf]],
            dtype=np.float32,
        )

        weights = self.softmax.forward(masked_scores)

        expected = np.array(
            [[0.26894143, 0.7310586, 0.0], [1.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(weights, expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(np.sum(weights, axis=1), [1.0, 1.0])

    def test_forward_handles_large_scores_without_overflow(self) -> None:
        """Subtracting the row maximum preserves finite probabilities."""
        masked_scores = np.array([[1000.0, 1001.0]], dtype=np.float32)

        weights = self.softmax.forward(masked_scores)

        np.testing.assert_allclose(
            weights,
            np.array([[0.26894143, 0.7310586]], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_all_masked_query_row_raises_error(self) -> None:
        """Softmax needs at least one available key for every query."""
        with self.assertRaises(ValueError):
            self.softmax.forward(
                np.array([[-np.inf, -np.inf]], dtype=np.float32)
            )

    def test_non_two_dimensional_scores_raise_error(self) -> None:
        """Attention softmax expects a query-by-key score matrix."""
        with self.assertRaises(ValueError):
            self.softmax.forward(np.ones(2, dtype=np.float32))

    def test_non_floating_scores_raise_error(self) -> None:
        """Probabilities require floating-point score values."""
        with self.assertRaises(ValueError):
            self.softmax.forward(np.ones((2, 2), dtype=np.int64))

