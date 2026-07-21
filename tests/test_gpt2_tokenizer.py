"""Tests for GPT-2 byte-level BPE tokenization and inference use."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from llm_inference_engine.engine.inference import TinyInferenceEngine
from llm_inference_engine.engine.token_selection import GreedyTokenSelector
from llm_inference_engine.model.gpt2_tokenizer import GPT2BPETokenizer


class _AlwaysOnModel:
    """Return logits whose greedy choice is the token representing ' on'."""

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """Return one four-token logit row per input token."""
        logits = np.zeros((len(token_ids), 4), dtype=np.float32)
        logits[:, 3] = 1.0
        return logits


class GPT2BPETokenizerTests(unittest.TestCase):
    """Verify GPT-2 pre-tokenization, byte mapping, BPE, and decoding."""

    def setUp(self) -> None:
        """Create tiny real-format vocabulary and merge files."""
        temporary_directory = TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.model_directory = Path(temporary_directory.name)

        space_marker = "\u0120"
        encoded_cafe = "caf\u00c3\u00a9"
        vocabulary = {
            "the": 0,
            f"{space_marker}cat": 1,
            f"{space_marker}sat": 2,
            f"{space_marker}on": 3,
            "!": 4,
            encoded_cafe: 5,
            "<|endoftext|>": 6,
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
            ("c", "a"),
            ("ca", "f"),
            ("caf", "\u00c3"),
            ("caf\u00c3", "\u00a9"),
        ]
        merge_text = "#version: 0.2\n" + "\n".join(
            f"{first} {second}" for first, second in merges
        )
        (self.model_directory / "merges.txt").write_text(
            merge_text,
            encoding="utf-8",
        )
        self.tokenizer = GPT2BPETokenizer.from_directory(
            self.model_directory
        )

    def test_encode_applies_learned_merges_to_running_example(self) -> None:
        """Words and their leading spaces become learned GPT-2 tokens."""
        token_ids = self.tokenizer.encode("the cat sat")

        self.assertEqual(token_ids, [0, 1, 2])

    def test_decode_restores_spaces_stored_inside_tokens(self) -> None:
        """The visible space marker reverses to its original byte."""
        text = self.tokenizer.decode([0, 1, 2, 3])

        self.assertEqual(text, "the cat sat on")
        self.assertEqual(self.tokenizer.decode([3]), " on")

    def test_byte_mapping_round_trips_utf8_text(self) -> None:
        """Non-ASCII text survives reversible byte-to-Unicode mapping."""
        token_ids = self.tokenizer.encode("caf\u00e9")

        self.assertEqual(token_ids, [5])
        self.assertEqual(self.tokenizer.decode(token_ids), "caf\u00e9")

    def test_end_of_text_is_kept_as_one_special_token(self) -> None:
        """GPT-2's special marker bypasses ordinary BPE splitting."""
        token_ids = self.tokenizer.encode("the<|endoftext|>")

        self.assertEqual(token_ids, [0, 6])

    def test_missing_bpe_vocabulary_token_raises_error(self) -> None:
        """Every final BPE symbol must exist in vocab.json."""
        with self.assertRaisesRegex(ValueError, "missing from vocabulary"):
            self.tokenizer.encode("dog")

    def test_inference_engine_accepts_gpt2_tokenizer_protocol(self) -> None:
        """Generation preserves spaces carried by GPT-2 output tokens."""
        engine = TinyInferenceEngine(
            self.tokenizer,
            _AlwaysOnModel(),
            GreedyTokenSelector(),
        )

        next_token = engine.predict_next_token("the cat sat")
        generated = engine.generate("the cat sat", max_new_tokens=1)

        self.assertEqual(next_token, " on")
        self.assertEqual(generated, "the cat sat on")
