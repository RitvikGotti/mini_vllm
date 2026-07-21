"""Tests for adapting DistilGPT-2 checkpoint configuration values."""

import unittest

from llm_inference_engine.loaders.distilgpt2_config import (
    DistilGPT2ConfigAdapter,
)


class DistilGPT2ConfigAdapterTests(unittest.TestCase):
    """Verify external GPT-2 names become internal architecture values."""

    def setUp(self) -> None:
        """Create the architecture values published for DistilGPT-2."""
        self.checkpoint_config: dict[str, object] = {
            "model_type": "gpt2",
            "vocab_size": 50_257,
            "n_positions": 1_024,
            "n_embd": 768,
            "n_layer": 6,
            "n_head": 12,
        }

    def test_from_dict_maps_distilgpt2_architecture(self) -> None:
        """Published checkpoint dimensions map to our internal names."""
        config = DistilGPT2ConfigAdapter.from_dict(self.checkpoint_config)

        self.assertEqual(config.vocab_size, 50_257)
        self.assertEqual(config.hidden_size, 768)
        self.assertEqual(config.num_layers, 6)
        self.assertEqual(config.num_attention_heads, 12)
        self.assertEqual(config.head_dim, 64)
        self.assertEqual(config.ffn_hidden_size, 3_072)
        self.assertEqual(config.max_sequence_length, 1_024)

    def test_from_dict_preserves_explicit_inner_size(self) -> None:
        """A checkpoint-provided MLP width overrides GPT-2's default rule."""
        self.checkpoint_config["n_inner"] = 2_048

        config = DistilGPT2ConfigAdapter.from_dict(self.checkpoint_config)

        self.assertEqual(config.ffn_hidden_size, 2_048)

    def test_missing_required_dimension_raises_error(self) -> None:
        """The adapter never guesses a required checkpoint dimension."""
        del self.checkpoint_config["n_layer"]

        with self.assertRaisesRegex(ValueError, "n_layer"):
            DistilGPT2ConfigAdapter.from_dict(self.checkpoint_config)

    def test_wrong_model_type_raises_error(self) -> None:
        """A GPT-2 tensor mapper must not silently accept another architecture."""
        self.checkpoint_config["model_type"] = "bert"

        with self.assertRaisesRegex(ValueError, "model_type='gpt2'"):
            DistilGPT2ConfigAdapter.from_dict(self.checkpoint_config)

    def test_boolean_dimension_raises_error(self) -> None:
        """Boolean values must not pass Python's integer type relationship."""
        self.checkpoint_config["n_head"] = True

        with self.assertRaisesRegex(ValueError, "n_head"):
            DistilGPT2ConfigAdapter.from_dict(self.checkpoint_config)
