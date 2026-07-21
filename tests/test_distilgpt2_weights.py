"""Tests for mapping DistilGPT-2 checkpoint tensor names and layouts."""

import unittest

import numpy as np

from llm_inference_engine.loaders.distilgpt2_weights import (
    DistilGPT2WeightMapper,
)
from llm_inference_engine.model.activations import gelu_new
from llm_inference_engine.model.tiny_model import TinyTransformerModel
from llm_inference_engine.utils.config import ModelConfig


class DistilGPT2WeightMapperTests(unittest.TestCase):
    """Verify GPT-2 tensors become weights accepted by our model."""

    def setUp(self) -> None:
        """Create a tiny checkpoint with real GPT-2 tensor names."""
        self.config = ModelConfig(
            vocab_size=4,
            hidden_size=2,
            num_layers=1,
            num_attention_heads=2,
            ffn_hidden_size=4,
            max_sequence_length=5,
        )
        identity = np.eye(2, dtype=np.float32)
        self.query = identity
        self.key = 2.0 * identity
        self.value = 3.0 * identity
        self.checkpoint = {
            "transformer.wte.weight": np.array(
                [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]],
                dtype=np.float32,
            ),
            "transformer.wpe.weight": np.zeros((5, 2), dtype=np.float32),
            "transformer.h.0.ln_1.weight": np.ones(2, dtype=np.float32),
            "transformer.h.0.ln_1.bias": np.zeros(2, dtype=np.float32),
            "transformer.h.0.attn.c_attn.weight": np.concatenate(
                (self.query, self.key, self.value),
                axis=1,
            ),
            "transformer.h.0.attn.c_attn.bias": np.array(
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                dtype=np.float32,
            ),
            "transformer.h.0.attn.c_proj.weight": identity,
            "transformer.h.0.attn.c_proj.bias": np.zeros(2, dtype=np.float32),
            "transformer.h.0.ln_2.weight": np.ones(2, dtype=np.float32),
            "transformer.h.0.ln_2.bias": np.zeros(2, dtype=np.float32),
            "transformer.h.0.mlp.c_fc.weight": np.array(
                [[1.0, -1.0, 0.5, 0.0], [0.0, 1.0, 0.5, -1.0]],
                dtype=np.float32,
            ),
            "transformer.h.0.mlp.c_fc.bias": np.zeros(4, dtype=np.float32),
            "transformer.h.0.mlp.c_proj.weight": np.array(
                [[0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [0.0, 0.5]],
                dtype=np.float32,
            ),
            "transformer.h.0.mlp.c_proj.bias": np.zeros(2, dtype=np.float32),
            "transformer.ln_f.weight": np.ones(2, dtype=np.float32),
            "transformer.ln_f.bias": np.zeros(2, dtype=np.float32),
            "lm_head.weight": np.array(
                [[0.5, 0.0], [-0.2, 0.2], [0.1, -0.1], [0.0, 1.0]],
                dtype=np.float32,
            ),
        }
        self.mapper = DistilGPT2WeightMapper(self.config)

    def test_map_weights_splits_qkv_and_orients_lm_head(self) -> None:
        """Combined QKV columns split while the LM head turns input-first."""
        weights = self.mapper.map_weights(self.checkpoint)
        layer = weights.layers[0]

        np.testing.assert_array_equal(layer.query, self.query)
        np.testing.assert_array_equal(layer.key, self.key)
        np.testing.assert_array_equal(layer.value, self.value)
        np.testing.assert_allclose(layer.query_bias, [0.1, 0.2])
        np.testing.assert_allclose(layer.key_bias, [0.3, 0.4])
        np.testing.assert_allclose(layer.value_bias, [0.5, 0.6])
        np.testing.assert_array_equal(
            weights.lm_head,
            self.checkpoint["lm_head.weight"].T,
        )

    def test_map_weights_uses_tied_embedding_when_lm_head_is_absent(self) -> None:
        """GPT-2's token embedding can serve as its tied output weights."""
        checkpoint = dict(self.checkpoint)
        del checkpoint["lm_head.weight"]

        weights = self.mapper.map_weights(checkpoint)

        np.testing.assert_array_equal(
            weights.lm_head,
            checkpoint["transformer.wte.weight"].T,
        )

    def test_map_weights_rejects_incorrect_tensor_shape(self) -> None:
        """A malformed checkpoint fails at the named tensor boundary."""
        checkpoint = dict(self.checkpoint)
        checkpoint["transformer.h.0.attn.c_attn.weight"] = np.zeros(
            (2, 5),
            dtype=np.float32,
        )

        with self.assertRaisesRegex(ValueError, "attn.c_attn.weight"):
            self.mapper.map_weights(checkpoint)

    def test_map_weights_rejects_missing_layer_tensor(self) -> None:
        """Every configured layer must supply all required parameters."""
        checkpoint = dict(self.checkpoint)
        del checkpoint["transformer.h.0.mlp.c_proj.bias"]

        with self.assertRaisesRegex(ValueError, "mlp.c_proj.bias"):
            self.mapper.map_weights(checkpoint)

    def test_mapped_weights_run_through_existing_model(self) -> None:
        """Mapped tensors require no hand-written transfer into the model."""
        weights = self.mapper.map_weights(self.checkpoint)
        model = TinyTransformerModel(
            self.config,
            weights,
            ffn_activation=gelu_new,
        )

        logits = model.forward(np.array([0, 1, 2], dtype=np.int64))

        self.assertEqual(logits.shape, (3, 4))
        self.assertTrue(np.all(np.isfinite(logits)))
