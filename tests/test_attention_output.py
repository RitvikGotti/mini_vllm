"""Tests for the learned attention output projection."""

import unittest

import numpy as np

from llm_inference_engine.model.attention_output import (
    AttentionOutputProjection,
)
from llm_inference_engine.utils.config import ModelConfig


class AttentionOutputProjectionTests(unittest.TestCase):
    """Verify context projection and tensor-shape validation."""

    def setUp(self) -> None:
        """Create a two-feature model configuration for each test."""
        self.config = ModelConfig(
            vocab_size=4,
            hidden_size=2,
            num_layers=1,
            num_attention_heads=1,
            ffn_hidden_size=4,
            max_sequence_length=4,
        )

    def test_forward_applies_learned_output_projection(self) -> None:
        """Each context vector is multiplied by the output weight matrix."""
        projection = AttentionOutputProjection(
            self.config,
            np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32),
        )
        context = np.array(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=np.float32,
        )

        output = projection.forward(context)

        np.testing.assert_array_equal(
            output,
            np.array([[2.0, 6.0], [6.0, 12.0]], dtype=np.float32),
        )

    def test_wrong_weight_shape_raises_error(self) -> None:
        """Output weights must match the model hidden width."""
        with self.assertRaises(ValueError):
            AttentionOutputProjection(
                self.config,
                np.ones((2, 3), dtype=np.float32),
            )

    def test_wrong_context_width_raises_error(self) -> None:
        """Context vectors must match the learned projection input width."""
        projection = AttentionOutputProjection(
            self.config,
            np.eye(2, dtype=np.float32),
        )

        with self.assertRaises(ValueError):
            projection.forward(np.ones((2, 3), dtype=np.float32))

    def test_non_two_dimensional_context_raises_error(self) -> None:
        """Projection receives one context vector per sequence position."""
        projection = AttentionOutputProjection(
            self.config,
            np.eye(2, dtype=np.float32),
        )

        with self.assertRaises(ValueError):
            projection.forward(np.ones(2, dtype=np.float32))
