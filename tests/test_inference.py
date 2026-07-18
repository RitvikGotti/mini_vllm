"""Tests for the connected text-to-next-token inference path."""

import unittest

import numpy as np

from llm_inference_engine.engine.inference import TinyInferenceEngine
from llm_inference_engine.engine.token_selection import GreedyTokenSelector
from llm_inference_engine.model.tiny_model import (
    TinyTransformerModel,
    TinyTransformerWeights,
    TransformerLayerWeights,
)
from llm_inference_engine.model.tokenizer import WhitespaceTokenizer
from llm_inference_engine.utils.config import ModelConfig


class TinyInferenceEngineTests(unittest.TestCase):
    """Verify the running example flows from prompt text to next-token text."""

    def setUp(self) -> None:
        """Create the running-example tokenizer, model, and selector."""
        config = ModelConfig(
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
        layer_weights = TransformerLayerWeights(
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
        weights = TinyTransformerWeights(
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
            layers=(layer_weights,),
            final_norm_scale=np.ones(2, dtype=np.float32),
            final_norm_bias=np.zeros(2, dtype=np.float32),
            lm_head=np.array(
                [[0.5, -0.2, 0.1, 0.0], [0.0, 0.2, -0.1, 1.0]],
                dtype=np.float32,
            ),
        )
        tokenizer = WhitespaceTokenizer(
            {"the": 0, "cat": 1, "sat": 2, "on": 3}
        )
        model = TinyTransformerModel(config, weights)
        self.engine = TinyInferenceEngine(
            tokenizer,
            model,
            GreedyTokenSelector(),
        )

    def test_predict_next_token_connects_text_to_text(self) -> None:
        """The complete running example predicts on after the cat sat."""
        next_token = self.engine.predict_next_token("the cat sat")

        self.assertEqual(next_token, "on")

    def test_empty_prompt_raises_error(self) -> None:
        """The model needs at least one token before predicting another."""
        with self.assertRaises(ValueError):
            self.engine.predict_next_token("   ")

    def test_generate_repeats_next_token_prediction(self) -> None:
        """Each selected token is appended before the next model run."""
        generated_text = self.engine.generate(
            "the cat sat",
            max_new_tokens=2,
        )

        self.assertEqual(generated_text, "the cat sat on on")

    def test_negative_max_new_tokens_raises_error(self) -> None:
        """The generation limit cannot request a negative token count."""
        with self.assertRaises(ValueError):
            self.engine.generate("the cat sat", max_new_tokens=-1)
