"""Tests for sequential transformer-block composition."""

import unittest

import numpy as np

from llm_inference_engine.model.transformer_stack import TransformerStack
from llm_inference_engine.utils.config import ModelConfig


class _AddOneBlock:
    """Test block that adds one to every hidden-state value."""

    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        """Return hidden states with one added."""
        return hidden_states + 1.0


class _DoubleBlock:
    """Test block that doubles every hidden-state value."""

    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        """Return doubled hidden states."""
        return hidden_states * 2.0


class TransformerStackTests(unittest.TestCase):
    """Verify transformer blocks run sequentially and in order."""

    def setUp(self) -> None:
        """Create a two-layer configuration."""
        self.config = ModelConfig(
            vocab_size=4,
            hidden_size=2,
            num_layers=2,
            num_attention_heads=1,
            ffn_hidden_size=4,
            max_sequence_length=8,
        )

    def test_forward_passes_each_layers_output_to_the_next_layer(self) -> None:
        """Adding one and then doubling must turn one into four."""
        stack = TransformerStack(
            self.config,
            [_AddOneBlock(), _DoubleBlock()],
        )

        output = stack.forward(
            np.ones((1, 2), dtype=np.float32)
        )

        np.testing.assert_array_equal(
            output,
            np.full((1, 2), 4.0, dtype=np.float32),
        )

    def test_wrong_number_of_blocks_raises_error(self) -> None:
        """The provided block count must match the architecture config."""
        with self.assertRaises(ValueError):
            TransformerStack(self.config, [_AddOneBlock()])
