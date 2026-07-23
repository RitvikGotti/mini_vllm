"""Tests for the local inference command-line interface."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import unittest
from unittest.mock import patch

from llm_inference_engine.cli import main


class CommandLineInterfaceTests(unittest.TestCase):
    """Check argument handling without loading a real checkpoint."""

    @patch("llm_inference_engine.cli.DistilGPT2FileLoader")
    def test_default_command_generates_one_token(self, loader_class) -> None:
        engine = loader_class.return_value.load_inference_engine.return_value
        engine.generate.return_value = "the cat sat in"
        output = StringIO()

        with redirect_stdout(output):
            exit_code = main(["the cat sat"])

        self.assertEqual(exit_code, 0)
        loader_class.return_value.load_inference_engine.assert_called_once_with(
            Path("models/distilgpt2")
        )
        engine.generate.assert_called_once_with("the cat sat", 1)
        self.assertEqual(output.getvalue(), "the cat sat in\n")

    @patch("llm_inference_engine.cli.DistilGPT2FileLoader")
    def test_command_accepts_model_directory_and_token_count(
        self,
        loader_class,
    ) -> None:
        engine = loader_class.return_value.load_inference_engine.return_value
        engine.generate.return_value = "hello there friend"
        output = StringIO()

        with redirect_stdout(output):
            exit_code = main(
                [
                    "hello",
                    "--model-directory",
                    "checkpoints/model",
                    "--max-new-tokens",
                    "2",
                ]
            )

        self.assertEqual(exit_code, 0)
        loader_class.return_value.load_inference_engine.assert_called_once_with(
            Path("checkpoints/model")
        )
        engine.generate.assert_called_once_with("hello", 2)
        self.assertEqual(output.getvalue(), "hello there friend\n")

    @patch("llm_inference_engine.cli.DistilGPT2FileLoader")
    def test_negative_token_count_is_rejected(self, loader_class) -> None:
        errors = StringIO()

        with redirect_stderr(errors):
            with self.assertRaises(SystemExit) as raised_error:
                main(["hello", "--max-new-tokens", "-1"])

        self.assertEqual(raised_error.exception.code, 2)
        self.assertIn("must be non-negative", errors.getvalue())
        loader_class.assert_not_called()


if __name__ == "__main__":
    unittest.main()
