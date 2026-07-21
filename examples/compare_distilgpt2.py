"""Numerically compare this engine with Hugging Face DistilGPT-2."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from llm_inference_engine.loaders.distilgpt2_files import (
    DistilGPT2FileLoader,
)
from llm_inference_engine.model.gpt2_tokenizer import GPT2BPETokenizer


def _parse_arguments() -> tuple[Path, str, float, float]:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_directory",
        type=Path,
        help="Directory containing config, tokenizer, and safetensors files.",
    )
    parser.add_argument("--prompt", default="the cat sat")
    parser.add_argument("--absolute-tolerance", type=float, default=2e-4)
    parser.add_argument("--relative-tolerance", type=float, default=2e-4)
    arguments = parser.parse_args()

    return (
        arguments.model_directory,
        arguments.prompt,
        arguments.absolute_tolerance,
        arguments.relative_tolerance,
    )


def main() -> None:
    """Run both implementations and fail if their outputs disagree."""
    try:
        import torch
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
    except ImportError as error:
        raise SystemExit(
            'Install reference dependencies with: pip install -e ".[reference]"'
        ) from error

    model_directory, prompt, absolute_tolerance, relative_tolerance = (
        _parse_arguments()
    )

    our_tokenizer = GPT2BPETokenizer.from_directory(model_directory)
    reference_tokenizer = GPT2Tokenizer(
        vocab_file=str(model_directory / "vocab.json"),
        merges_file=str(model_directory / "merges.txt"),
    )
    our_ids = our_tokenizer.encode(prompt)
    reference_ids = reference_tokenizer.encode(
        prompt,
        add_special_tokens=False,
    )

    print(f"Prompt: {prompt!r}")
    print(f"Our token IDs:       {our_ids}")
    print(f"Reference token IDs: {reference_ids}")
    if our_ids != reference_ids:
        raise SystemExit("comparison failed: token IDs differ")

    loaded_model = DistilGPT2FileLoader().load(model_directory)
    our_logits = loaded_model.create_model().forward(
        np.asarray(our_ids, dtype=np.int64)
    )

    reference_model = GPT2LMHeadModel.from_pretrained(
        model_directory,
        local_files_only=True,
        use_safetensors=True,
    )
    reference_model.eval()
    with torch.no_grad():
        reference_logits = reference_model(
            torch.tensor([reference_ids], dtype=torch.long)
        ).logits[0].cpu().numpy()

    if our_logits.shape != reference_logits.shape:
        raise SystemExit(
            "comparison failed: logits shapes differ: "
            f"{our_logits.shape} != {reference_logits.shape}"
        )

    difference = np.abs(
        our_logits.astype(np.float64)
        - reference_logits.astype(np.float64)
    )
    logits_match = np.allclose(
        our_logits,
        reference_logits,
        atol=absolute_tolerance,
        rtol=relative_tolerance,
    )
    our_next_id = int(np.argmax(our_logits[-1]))
    reference_next_id = int(np.argmax(reference_logits[-1]))

    print(f"Logits shape: {our_logits.shape}")
    print(f"Maximum absolute difference: {difference.max():.9g}")
    print(f"Mean absolute difference:    {difference.mean():.9g}")
    print(f"All logits close: {logits_match}")
    print(
        "Our next token:       "
        f"id={our_next_id}, text={our_tokenizer.decode([our_next_id])!r}"
    )
    print(
        "Reference next token: "
        f"id={reference_next_id}, "
        f"text={reference_tokenizer.decode([reference_next_id])!r}"
    )

    top_ids = np.argsort(reference_logits[-1])[-5:][::-1]
    print("Reference top five tokens:")
    for token_id in top_ids:
        print(
            f"  id={int(token_id):5d} "
            f"logit={reference_logits[-1, token_id]: .6f} "
            f"text={reference_tokenizer.decode([int(token_id)])!r}"
        )

    if not logits_match:
        raise SystemExit("comparison failed: logits exceed tolerance")
    if our_next_id != reference_next_id:
        raise SystemExit("comparison failed: greedy next tokens differ")

    print("Comparison passed.")


if __name__ == "__main__":
    main()
