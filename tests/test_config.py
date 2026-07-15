"""Tests for model architecture configuration."""

import unittest

from llm_inference_engine.utils.config import ModelConfig


class ModelConfigTests(unittest.TestCase):
    """Verify ModelConfig validation and derived values."""

    def test_valid_config_derives_head_dim(self) -> None:
        """A valid configuration splits hidden dimensions evenly across heads."""
        config = ModelConfig(
            vocab_size=1_000,
            hidden_size=128,
            num_layers=2,
            num_attention_heads=4,
            ffn_hidden_size=512,
            max_sequence_length=64,
        )

        self.assertEqual(config.head_dim, 32)

    def test_uneven_hidden_size_and_head_count_raises_error(self) -> None:
        """Attention heads must divide the hidden dimension evenly."""
        with self.assertRaises(ValueError):
            ModelConfig(
                vocab_size=1_000,
                hidden_size=130,
                num_layers=2,
                num_attention_heads=4,
                ffn_hidden_size=512,
                max_sequence_length=64,
            )

    def test_non_positive_dimension_raises_error(self) -> None:
        """Every architecture dimension must be positive."""
        with self.assertRaises(ValueError):
            ModelConfig(
                vocab_size=0,
                hidden_size=128,
                num_layers=2,
                num_attention_heads=4,
                ffn_hidden_size=512,
                max_sequence_length=64,
            )
