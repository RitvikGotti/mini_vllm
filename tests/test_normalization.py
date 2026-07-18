"""Tests for transformer layer normalization."""

import unittest

import numpy as np

from llm_inference_engine.model.normalization import LayerNorm
from llm_inference_engine.utils.config import ModelConfig


class LayerNormTests(unittest.TestCase):
    """Verify per-token normalization and learned affine parameters."""

    def setUp(self) -> None:
        """Create a three-feature model configuration for each test."""
        self.config = ModelConfig(
            vocab_size=4,
            hidden_size=3,
            num_layers=1,
            num_attention_heads=1,
            ffn_hidden_size=6,
            max_sequence_length=4,
        )
        self.scale = np.ones(3, dtype=np.float32)
        self.bias = np.zeros(3, dtype=np.float32)

    def test_forward_normalizes_each_token_independently(self) -> None:
        """Scaled versions of a feature pattern normalize almost identically."""
        layer_norm = LayerNorm(self.config, self.scale, self.bias)
        hidden_states = np.array(
            [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]],
            dtype=np.float32,
        )

        output = layer_norm.forward(hidden_states)

        expected = np.array(
            [
                [-1.2247356, 0.0, 1.2247356],
                [-1.2247438, 0.0, 1.2247438],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(output, expected, rtol=1e-6, atol=1e-6)

    def test_forward_applies_learned_scale_and_bias(self) -> None:
        """Scale and bias modify each normalized feature independently."""
        layer_norm = LayerNorm(
            self.config,
            np.array([2.0, 1.0, 0.5], dtype=np.float32),
            np.array([1.0, 0.0, -1.0], dtype=np.float32),
        )

        output = layer_norm.forward(
            np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        )

        np.testing.assert_allclose(
            output,
            np.array([[-1.4494712, 0.0, -0.3876322]], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_constant_hidden_state_remains_finite(self) -> None:
        """Epsilon safely handles a token vector with zero variance."""
        layer_norm = LayerNorm(self.config, self.scale, self.bias)

        output = layer_norm.forward(
            np.array([[2.0, 2.0, 2.0]], dtype=np.float32)
        )

        np.testing.assert_array_equal(
            output,
            np.zeros((1, 3), dtype=np.float32),
        )

    def test_wrong_parameter_shape_raises_error(self) -> None:
        """LayerNorm needs one scale and bias value per hidden feature."""
        with self.assertRaises(ValueError):
            LayerNorm(
                self.config,
                np.ones(2, dtype=np.float32),
                self.bias,
            )

    def test_non_positive_epsilon_raises_error(self) -> None:
        """Epsilon must keep the normalization denominator nonzero."""
        with self.assertRaises(ValueError):
            LayerNorm(
                self.config,
                self.scale,
                self.bias,
                epsilon=0.0,
            )
