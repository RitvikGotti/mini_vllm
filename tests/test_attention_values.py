"""Tests for attention-weighted value aggregation."""

import unittest

import numpy as np

from llm_inference_engine.model.attention_values import AttentionValueMixer


class AttentionValueMixerTests(unittest.TestCase):
    """Verify attention weights combine value vectors correctly."""

    def setUp(self) -> None:
        """Create a value mixer for each test."""
        self.mixer = AttentionValueMixer()

    def test_forward_computes_weighted_value_sums(self) -> None:
        """Each query receives its own weighted combination of value rows."""
        attention_weights = np.array(
            [[1.0, 0.0, 0.0], [0.25, 0.75, 0.0]],
            dtype=np.float32,
        )
        value = np.array(
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
            dtype=np.float32,
        )

        context = self.mixer.forward(attention_weights, value)

        np.testing.assert_allclose(
            context,
            np.array([[1.0, 10.0], [1.75, 17.5]], dtype=np.float32),
        )

    def test_mismatched_key_and_value_lengths_raise_error(self) -> None:
        """Every attention-weight column needs a corresponding value row."""
        attention_weights = np.ones((2, 3), dtype=np.float32)
        value = np.ones((2, 2), dtype=np.float32)

        with self.assertRaises(ValueError):
            self.mixer.forward(attention_weights, value)

    def test_non_two_dimensional_weights_raise_error(self) -> None:
        """Weights must form a query-by-key matrix."""
        attention_weights = np.ones(3, dtype=np.float32)
        value = np.ones((3, 2), dtype=np.float32)

        with self.assertRaises(ValueError):
            self.mixer.forward(attention_weights, value)

    def test_non_floating_values_raise_error(self) -> None:
        """Value vectors must contain continuous floating-point features."""
        attention_weights = np.ones((2, 2), dtype=np.float32)
        value = np.ones((2, 2), dtype=np.int64)

        with self.assertRaises(ValueError):
            self.mixer.forward(attention_weights, value)
