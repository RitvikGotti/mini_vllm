"""Integration tests for one single-head pre-norm transformer block."""

import unittest

import numpy as np

from llm_inference_engine.model.transformer_block import (
    SingleHeadTransformerBlock,
)
from llm_inference_engine.utils.config import ModelConfig


class SingleHeadTransformerBlockTests(unittest.TestCase):
    """Verify both pre-norm residual sublayers run in sequence."""

    def test_forward_runs_complete_pre_norm_transformer_block(self) -> None:
        """One call normalizes before attention and FFN updates."""
        config = ModelConfig(
            vocab_size=4,
            hidden_size=2,
            num_layers=1,
            num_attention_heads=1,
            ffn_hidden_size=4,
            max_sequence_length=8,
        )
        identity = np.eye(2, dtype=np.float32)
        block = SingleHeadTransformerBlock(
            config,
            query_weights=identity,
            key_weights=identity,
            value_weights=identity,
            attention_output_weights=identity,
            attention_norm_scale=np.ones(2, dtype=np.float32),
            attention_norm_bias=np.zeros(2, dtype=np.float32),
            ffn_input_weights=np.array(
                [[1.0, -1.0, 0.5, 0.0], [0.0, 1.0, 0.5, -1.0]],
                dtype=np.float32,
            ),
            ffn_output_weights=np.array(
                [
                    [0.5, 0.0],
                    [0.0, 0.5],
                    [0.5, 0.5],
                    [0.0, 0.5],
                ],
                dtype=np.float32,
            ),
            ffn_norm_scale=np.ones(2, dtype=np.float32),
            ffn_norm_bias=np.zeros(2, dtype=np.float32),
        )
        initial_hidden_states = np.array(
            [[1.0, 0.0], [0.1, 1.0], [1.0, 1.1]],
            dtype=np.float32,
        )

        output = block.forward(initial_hidden_states)

        expected = np.array(
            [
                [2.499979, -0.499981],
                [-0.788349, 2.888347],
                [0.058784, 3.041211],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(output, expected, rtol=1e-4, atol=1e-4)
