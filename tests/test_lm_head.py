"""Tests for the language-model vocabulary projection."""

import unittest

import numpy as np

from llm_inference_engine.model.lm_head import LanguageModelHead
from llm_inference_engine.utils.config import ModelConfig


class LanguageModelHeadTests(unittest.TestCase):
    """Verify hidden states become vocabulary logits."""

    def setUp(self) -> None:
        """Create a two-feature, four-token configuration for each test."""
        self.config = ModelConfig(
            vocab_size=4,
            hidden_size=2,
            num_layers=1,
            num_attention_heads=1,
            ffn_hidden_size=4,
            max_sequence_length=4,
        )
        self.weights = np.array(
            [
                [1.0, 0.0, -1.0, 0.5],
                [0.0, 2.0, 1.0, -0.5],
            ],
            dtype=np.float32,
        )

    def test_forward_returns_one_logit_per_vocabulary_token(self) -> None:
        """Every sequence row is projected from hidden width to vocabulary size."""
        lm_head = LanguageModelHead(self.config, self.weights)
        hidden_states = np.array(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=np.float32,
        )

        logits = lm_head.forward(hidden_states)

        np.testing.assert_array_equal(
            logits,
            np.array(
                [[1.0, 4.0, 1.0, -0.5], [3.0, 8.0, 1.0, -0.5]],
                dtype=np.float32,
            ),
        )

    def test_wrong_weight_shape_raises_error(self) -> None:
        """LM-head weights must map hidden size to vocabulary size."""
        with self.assertRaises(ValueError):
            LanguageModelHead(
                self.config,
                np.ones((2, 3), dtype=np.float32),
            )

    def test_wrong_hidden_state_width_raises_error(self) -> None:
        """LM-head input must have the configured model hidden width."""
        lm_head = LanguageModelHead(self.config, self.weights)

        with self.assertRaises(ValueError):
            lm_head.forward(np.ones((2, 3), dtype=np.float32))

    def test_non_two_dimensional_hidden_states_raise_error(self) -> None:
        """The LM head receives one hidden vector per sequence position."""
        lm_head = LanguageModelHead(self.config, self.weights)

        with self.assertRaises(ValueError):
            lm_head.forward(np.ones(2, dtype=np.float32))
