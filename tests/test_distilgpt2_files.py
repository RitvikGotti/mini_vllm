"""Tests for loading DistilGPT-2 artifacts from real local file formats."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from safetensors.numpy import save_file

from llm_inference_engine.loaders.distilgpt2_files import (
    DistilGPT2FileLoader,
)


class DistilGPT2FileLoaderTests(unittest.TestCase):
    """Verify config JSON and safetensors become a runnable model."""

    def setUp(self) -> None:
        """Create a tiny DistilGPT-2-shaped model directory."""
        temporary_directory = TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.model_directory = Path(temporary_directory.name)
        self.config_data: dict[str, object] = {
            "model_type": "gpt2",
            "vocab_size": 4,
            "n_positions": 5,
            "n_embd": 2,
            "n_layer": 1,
            "n_head": 2,
            "n_inner": 4,
            "activation_function": "gelu_new",
            "layer_norm_epsilon": 1e-5,
        }
        self._write_config()
        self._write_tokenizer_files()
        save_file(
            self._checkpoint_tensors(),
            self.model_directory / "model.safetensors",
        )
        self.loader = DistilGPT2FileLoader()

    def test_load_builds_model_from_json_and_safetensors(self) -> None:
        """Loaded files supply architecture, weights, and runtime settings."""
        loaded = self.loader.load(self.model_directory)
        model = loaded.create_model()

        logits = model.forward(np.array([0, 1, 2], dtype=np.int64))

        self.assertEqual(loaded.config.num_attention_heads, 2)
        self.assertEqual(loaded.norm_epsilon, 1e-5)
        self.assertEqual(logits.shape, (3, 4))
        self.assertTrue(np.all(np.isfinite(logits)))

    def test_load_rejects_unsupported_activation(self) -> None:
        """The loader refuses checkpoint behavior our model cannot reproduce."""
        self.config_data["activation_function"] = "relu"
        self._write_config()

        with self.assertRaisesRegex(ValueError, "gelu_new"):
            self.loader.load(self.model_directory)

    def test_load_inference_engine_connects_directory_to_text(self) -> None:
        """All four model files connect the running prompt to its next token."""
        engine = self.loader.load_inference_engine(self.model_directory)

        next_token = engine.predict_next_token("the cat sat")
        generated = engine.generate("the cat sat", max_new_tokens=1)

        self.assertEqual(next_token, " on")
        self.assertEqual(generated, "the cat sat on")

    def test_load_rejects_missing_safetensors_file(self) -> None:
        """A model cannot be built without its learned tensors."""
        (self.model_directory / "model.safetensors").unlink()

        with self.assertRaisesRegex(ValueError, "checkpoint file"):
            self.loader.load(self.model_directory)

    def _write_config(self) -> None:
        """Write the current tiny checkpoint configuration."""
        with (self.model_directory / "config.json").open(
            "w",
            encoding="utf-8",
        ) as config_file:
            json.dump(self.config_data, config_file)

    def _write_tokenizer_files(self) -> None:
        """Write tiny GPT-2 vocabulary and merge assets."""
        space_marker = "\u0120"
        vocabulary = {
            "the": 0,
            f"{space_marker}cat": 1,
            f"{space_marker}sat": 2,
            f"{space_marker}on": 3,
        }
        with (self.model_directory / "vocab.json").open(
            "w",
            encoding="utf-8",
        ) as vocabulary_file:
            json.dump(vocabulary, vocabulary_file, ensure_ascii=False)

        merges = [
            ("t", "h"),
            ("th", "e"),
            (space_marker, "c"),
            (f"{space_marker}c", "a"),
            (f"{space_marker}ca", "t"),
            (space_marker, "s"),
            (f"{space_marker}s", "a"),
            (f"{space_marker}sa", "t"),
            (space_marker, "o"),
            (f"{space_marker}o", "n"),
        ]
        merge_text = "#version: 0.2\n" + "\n".join(
            f"{first} {second}" for first, second in merges
        )
        (self.model_directory / "merges.txt").write_text(
            merge_text,
            encoding="utf-8",
        )

    @staticmethod
    def _checkpoint_tensors() -> dict[str, np.ndarray]:
        """Return one complete tiny checkpoint using GPT-2 tensor names."""
        identity = np.eye(2, dtype=np.float32)

        return {
            "transformer.wte.weight": np.array(
                [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]],
                dtype=np.float32,
            ),
            "transformer.wpe.weight": np.array(
                [
                    [0.0, 0.0],
                    [0.1, 0.0],
                    [0.0, 0.1],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            "transformer.h.0.ln_1.weight": np.ones(2, dtype=np.float32),
            "transformer.h.0.ln_1.bias": np.zeros(2, dtype=np.float32),
            "transformer.h.0.attn.c_attn.weight": np.concatenate(
                (identity, identity, identity),
                axis=1,
            ),
            "transformer.h.0.attn.c_attn.bias": np.zeros(6, dtype=np.float32),
            "transformer.h.0.attn.c_proj.weight": identity.copy(),
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
