"""Tests for choosing the next token from model logits."""

import unittest

import numpy as np

from llm_inference_engine.engine.token_selection import GreedyTokenSelector


class GreedyTokenSelectorTests(unittest.TestCase):
    """Verify greedy selection uses the final prompt position."""

    def test_select_returns_highest_logit_token_from_final_position(self) -> None:
        """Earlier rows are ignored when choosing the next token."""
        logits = np.array(
            [
                [10.0, 0.0, 0.0, 0.0],
                [0.0, 8.0, 0.0, 0.0],
                [-0.5, 0.4, -0.2, 1.0],
            ],
            dtype=np.float32,
        )

        token_id = GreedyTokenSelector().select(logits)

        self.assertEqual(token_id, 3)

    def test_non_two_dimensional_logits_raise_error(self) -> None:
        """The selector requires sequence-by-vocabulary logits."""
        with self.assertRaises(ValueError):
            GreedyTokenSelector().select(
                np.array([0.1, 0.2, 0.3], dtype=np.float32)
            )

    def test_empty_sequence_logits_raise_error(self) -> None:
        """A next token cannot be selected without a prompt position."""
        with self.assertRaises(ValueError):
            GreedyTokenSelector().select(
                np.empty((0, 4), dtype=np.float32)
            )

    def test_non_finite_logits_raise_error(self) -> None:
        """NaN and infinity must not silently determine the next token."""
        with self.assertRaises(ValueError):
            GreedyTokenSelector().select(
                np.array([[0.1, np.nan, 0.3]], dtype=np.float32)
            )
