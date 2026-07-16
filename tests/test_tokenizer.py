"""Tests for the early whitespace tokenizer."""

import unittest

from llm_inference_engine.model.tokenizer import WhitespaceTokenizer


class WhitespaceTokenizerTests(unittest.TestCase):
    """Verify conversion between whitespace-delimited tokens and IDs."""

    def setUp(self) -> None:
        """Create a small fixed vocabulary for each test."""
        self.tokenizer = WhitespaceTokenizer(
            {"hello": 0, "world": 1, "!": 2}
        )

    def test_encode_returns_ids_for_known_tokens(self) -> None:
        """Encoding maps each known token to its vocabulary ID."""
        token_ids = self.tokenizer.encode("hello world !")

        self.assertEqual(token_ids, [0, 1, 2])

    def test_decode_returns_text_for_known_ids(self) -> None:
        """Decoding maps each known ID back to its token."""
        text = self.tokenizer.decode([0, 1, 2])

        self.assertEqual(text, "hello world !")

    def test_encode_unknown_token_raises_error(self) -> None:
        """Encoding text outside the fixed vocabulary fails explicitly."""
        with self.assertRaises(ValueError):
            self.tokenizer.encode("hello Codex")

    def test_decode_unknown_id_raises_error(self) -> None:
        """Decoding an ID outside the fixed vocabulary fails explicitly."""
        with self.assertRaises(ValueError):
            self.tokenizer.decode([99])

    def test_duplicate_vocabulary_ids_raise_error(self) -> None:
        """Each vocabulary ID must identify exactly one token."""
        with self.assertRaises(ValueError):
            WhitespaceTokenizer({"hello": 0, "world": 0})

