"""Tests for splitting and merging multi-head attention tensors."""

import unittest

import numpy as np

from llm_inference_engine.model.attention_heads import (
    AttentionHeadMerger,
    AttentionHeadSplitter,
)
from llm_inference_engine.utils.config import ModelConfig


class AttentionHeadShapeTests(unittest.TestCase):
    """Verify hidden features move to and from separate heads."""

    def setUp(self) -> None:
        """Create the two-head version of the running example."""
        self.config = ModelConfig(
            vocab_size=4,
            hidden_size=2,
            num_layers=1,
            num_attention_heads=2,
            ffn_hidden_size=4,
            max_sequence_length=8,
        )
        self.projected_states = np.array(
            [[1.0, 0.0], [0.1, 1.0], [1.0, 1.1]],
            dtype=np.float32,
        )

    def test_splitter_assigns_hidden_features_to_heads(self) -> None:
        """Each head receives its portion from every sequence token."""
        head_states = AttentionHeadSplitter(self.config).forward(
            self.projected_states
        )

        expected = np.array(
            [
                [[1.0], [0.1], [1.0]],
                [[0.0], [1.0], [1.1]],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(head_states, expected)
        self.assertEqual(head_states.shape, (2, 3, 1))

    def test_merger_restores_original_hidden_feature_order(self) -> None:
        """Merging split heads must reproduce every original value."""
        splitter = AttentionHeadSplitter(self.config)
        merger = AttentionHeadMerger(self.config)

        merged_states = merger.forward(
            splitter.forward(self.projected_states)
        )

        np.testing.assert_array_equal(
            merged_states,
            self.projected_states,
        )
        self.assertEqual(merged_states.shape, (3, 2))

    def test_splitter_rejects_wrong_hidden_width(self) -> None:
        """Projected states must contain the configured hidden width."""
        with self.assertRaises(ValueError):
            AttentionHeadSplitter(self.config).forward(
                np.ones((3, 3), dtype=np.float32)
            )

    def test_merger_rejects_wrong_head_count(self) -> None:
        """Head states must contain the configured number of heads."""
        with self.assertRaises(ValueError):
            AttentionHeadMerger(self.config).forward(
                np.ones((1, 3, 1), dtype=np.float32)
            )
