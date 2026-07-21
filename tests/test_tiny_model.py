"""End-to-end tests for the connected tiny transformer model."""

from dataclasses import replace
import unittest

import numpy as np

from llm_inference_engine.model.tiny_model import (
    TinyTransformerModel,
    TinyTransformerWeights,
    TransformerLayerWeights,
)
from llm_inference_engine.utils.config import ModelConfig


class TinyTransformerModelTests(unittest.TestCase):
    """Verify token IDs flow through the complete tiny model to logits."""

    def setUp(self) -> None:
        """Create the running-example configuration and learned tensors."""
        self.config = ModelConfig(
            vocab_size=4,
            hidden_size=2,
            num_layers=1,
            num_attention_heads=1,
            ffn_hidden_size=4,
            max_sequence_length=8,
        )
        identity = np.eye(2, dtype=np.float32)
        position_embedding = np.zeros((8, 2), dtype=np.float32)
        position_embedding[:3] = np.array(
            [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]],
            dtype=np.float32,
        )
        self.layer_weights = TransformerLayerWeights(
            query=identity,
            key=identity,
            value=identity,
            attention_output=identity,
            attention_norm_scale=np.ones(2, dtype=np.float32),
            attention_norm_bias=np.zeros(2, dtype=np.float32),
            ffn_input=np.array(
                [[1.0, -1.0, 0.5, 0.0], [0.0, 1.0, 0.5, -1.0]],
                dtype=np.float32,
            ),
            ffn_output=np.array(
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
        self.weights = TinyTransformerWeights(
            token_embedding=np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [0.5, 0.5],
                ],
                dtype=np.float32,
            ),
            position_embedding=position_embedding,
            layers=(self.layer_weights,),
            final_norm_scale=np.ones(2, dtype=np.float32),
            final_norm_bias=np.zeros(2, dtype=np.float32),
            lm_head=np.array(
                [[0.5, -0.2, 0.1, 0.0], [0.0, 0.2, -0.1, 1.0]],
                dtype=np.float32,
            ),
        )

    def test_forward_connects_token_ids_to_vocabulary_logits(self) -> None:
        """The final prompt row assigns the highest logit to token ID three."""
        model = TinyTransformerModel(self.config, self.weights)

        logits = model.forward(np.array([0, 1, 2], dtype=np.int64))

        self.assertEqual(logits.shape, (3, 4))
        self.assertEqual(int(np.argmax(logits[-1])), 3)

    def test_layer_count_mismatch_raises_error(self) -> None:
        """The supplied layer weights must match the configured layer count."""
        config = ModelConfig(
            vocab_size=4,
            hidden_size=2,
            num_layers=2,
            num_attention_heads=1,
            ffn_hidden_size=4,
            max_sequence_length=8,
        )

        with self.assertRaises(ValueError):
            TinyTransformerModel(config, self.weights)

    def test_forward_supports_multiple_transformer_layers(self) -> None:
        """The model can run a separately weighted block for each layer."""
        config = ModelConfig(
            vocab_size=4,
            hidden_size=2,
            num_layers=2,
            num_attention_heads=1,
            ffn_hidden_size=4,
            max_sequence_length=8,
        )
        weights = replace(
            self.weights,
            layers=(self.layer_weights, self.layer_weights),
        )
        model = TinyTransformerModel(config, weights)

        logits = model.forward(np.array([0, 1, 2], dtype=np.int64))

        self.assertEqual(logits.shape, (3, 4))

    def test_forward_routes_activation_to_every_layer(self) -> None:
        """The model-level FFN activation reaches the transformer block."""
        activation_calls = 0

        def recording_relu(values: np.ndarray) -> np.ndarray:
            """Count calls while preserving the running example's ReLU."""
            nonlocal activation_calls
            activation_calls += 1
            return np.maximum(values, 0.0)

        config = replace(self.config, num_layers=2)
        weights = replace(
            self.weights,
            layers=(self.layer_weights, self.layer_weights),
        )
        model = TinyTransformerModel(
            config,
            weights,
            ffn_activation=recording_relu,
        )

        model.forward(np.array([0, 1, 2], dtype=np.int64))

        self.assertEqual(activation_calls, 2)

    def test_forward_routes_layer_biases_into_attention(self) -> None:
        """A layer output bias changes the model's selected next token."""
        biased_layer = replace(
            self.layer_weights,
            attention_output_bias=np.array(
                [5.0, 0.0],
                dtype=np.float32,
            ),
        )
        weights = replace(self.weights, layers=(biased_layer,))
        model = TinyTransformerModel(self.config, weights)

        logits = model.forward(np.array([0, 1, 2], dtype=np.int64))

        self.assertEqual(int(np.argmax(logits[-1])), 0)
