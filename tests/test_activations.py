"""Tests for transformer feed-forward activation functions."""

import unittest

import numpy as np

from llm_inference_engine.model.activations import gelu_new, relu


class ActivationFunctionTests(unittest.TestCase):
    """Verify activation functions transform values independently."""

    def test_relu_zeros_negative_values(self) -> None:
        """ReLU preserves positive values and replaces negatives with zero."""
        values = np.array([-1.0, 0.0, 1.0], dtype=np.float32)

        output = relu(values)

        np.testing.assert_array_equal(
            output,
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
        )

    def test_gelu_new_smoothly_gates_values(self) -> None:
        """GPT-2 GELU preserves a small negative contribution."""
        values = np.array([-1.0, 0.0, 1.0], dtype=np.float32)

        output = gelu_new(values)

        np.testing.assert_allclose(
            output,
            np.array([-0.158808, 0.0, 0.841192], dtype=np.float32),
            rtol=1e-5,
            atol=1e-5,
        )
