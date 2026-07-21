"""Tests for causal attention masking."""

import unittest

import numpy as np

from llm_inference_engine.model.causal_mask import CausalMask


class CausalMaskTests(unittest.TestCase):
    """Verify future keys are unavailable to decoder attention queries."""

    def setUp(self) -> None:
        """Create a causal mask for each test."""
        self.mask = CausalMask()

    def test_forward_masks_future_positions(self) -> None:
        """Each query can retain only its own and earlier key positions."""
        scaled_scores = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            dtype=np.float32,
        )

        masked_scores = self.mask.forward(scaled_scores)

        expected = np.array(
            [
                [1.0, -np.inf, -np.inf],
                [4.0, 5.0, -np.inf],
                [7.0, 8.0, 9.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(masked_scores, expected)

    def test_forward_uses_query_offset_for_cached_decoding(self) -> None:
        """Offset queries retain keys through their absolute positions."""
        scaled_scores = np.array(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            dtype=np.float32,
        )

        masked_scores = self.mask.forward(scaled_scores, query_offset=2)

        expected = np.array(
            [[1.0, 2.0, 3.0, -np.inf], [5.0, 6.0, 7.0, 8.0]],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(masked_scores, expected)

    def test_negative_query_offset_raises_error(self) -> None:
        """A sequence cannot begin before absolute position zero."""
        with self.assertRaises(ValueError):
            self.mask.forward(np.ones((2, 2), dtype=np.float32), query_offset=-1)

    def test_unavailable_query_positions_raise_error(self) -> None:
        """Every query needs a corresponding key position."""
        with self.assertRaises(ValueError):
            self.mask.forward(np.ones((2, 2), dtype=np.float32), query_offset=1)

    def test_non_floating_scores_raise_error(self) -> None:
        """Scores must remain floating-point so they can hold negative infinity."""
        with self.assertRaises(ValueError):
            self.mask.forward(np.ones((2, 2), dtype=np.int64))

    def test_forward_masks_future_positions_for_every_head(self) -> None:
        """Every head receives the same causal visibility rule."""
        scaled_scores = np.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ],
            dtype=np.float32,
        )

        masked_scores = self.mask.forward(scaled_scores)

        expected = np.array(
            [
                [[1.0, -np.inf], [3.0, 4.0]],
                [[5.0, -np.inf], [7.0, 8.0]],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(masked_scores, expected)
