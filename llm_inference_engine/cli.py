"""Command-line interface for local language-model inference."""

from argparse import ArgumentParser, ArgumentTypeError
from collections.abc import Sequence
from pathlib import Path

from llm_inference_engine.loaders.distilgpt2_files import (
    DistilGPT2FileLoader,
)


def _non_negative_integer(value: str) -> int:
    """Parse a command-line integer that may be zero but not negative."""
    try:
        parsed_value = int(value)
    except ValueError as error:
        raise ArgumentTypeError("must be an integer") from error

    if parsed_value < 0:
        raise ArgumentTypeError("must be non-negative")

    return parsed_value


def _build_parser() -> ArgumentParser:
    """Describe the supported local-inference command-line arguments."""
    parser = ArgumentParser(
        prog="llm-infer",
        description="Generate text with a local DistilGPT-2 checkpoint.",
    )
    parser.add_argument("prompt", help="Text the model should continue.")
    parser.add_argument(
        "--model-directory",
        type=Path,
        default=Path("models/distilgpt2"),
        help="Checkpoint directory (default: models/distilgpt2).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=_non_negative_integer,
        default=1,
        help="Number of greedily selected tokens to append (default: 1).",
    )

    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    """Load the checkpoint, generate text, print it, and report success."""
    parser = _build_parser()
    parsed_arguments = parser.parse_args(arguments)

    try:
        engine = DistilGPT2FileLoader().load_inference_engine(
            parsed_arguments.model_directory
        )
        generated_text = engine.generate(
            parsed_arguments.prompt,
            parsed_arguments.max_new_tokens,
        )
    except ValueError as error:
        parser.error(str(error))

    print(generated_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
