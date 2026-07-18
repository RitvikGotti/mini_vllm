"""Tests for transformer residual connections."""

import unittest

import numpy as np

from llm_inference_engine.model.residual import ResidualConnection


class ResidualConnectionTests(unittest.TestCase):
    """Verify aligned hidden-state addition and validation."""

    def setUp(self) -> None:
        """Create a residual connection for each test."""
        self.connection = ResidualConnection()

    def test_forward_adds_sublayer_update_to_original_states(self) -> None:
        """Each token feature retains its input and receives the update."""
        residual = np.array(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=np.float32,
        )
        sublayer_output = np.array(
            [[0.1, 0.2], [0.3, 0.4]],
            dtype=np.float32,
        )

        output = self.connection.forward(residual, sublayer_output)

        np.testing.assert_allclose(
            output,
            np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float32),
        )

    def test_mismatched_shapes_raise_error(self) -> None:
        """Residual addition requires position-and-feature alignment."""
        residual = np.ones((2, 2), dtype=np.float32)
        sublayer_output = np.ones((2, 3), dtype=np.float32)

        with self.assertRaises(ValueError):
            self.connection.forward(residual, sublayer_output)

    def test_non_two_dimensional_residual_raises_error(self) -> None:
        """A residual path carries one hidden vector per token position."""
        residual = np.ones(2, dtype=np.float32)
        sublayer_output = np.ones((1, 2), dtype=np.float32)

        with self.assertRaises(ValueError):
            self.connection.forward(residual, sublayer_output)

    def test_non_floating_sublayer_output_raises_error(self) -> None:
        """Transformer hidden states must remain floating-point values."""
        residual = np.ones((2, 2), dtype=np.float32)
        sublayer_output = np.ones((2, 2), dtype=np.int64)

        with self.assertRaises(ValueError):
            self.connection.forward(residual, sublayer_output)
