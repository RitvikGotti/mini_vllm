# LLM Inference Engine

A from-scratch learning project for building a small LLM inference engine incrementally.

## Current status

The engine now implements the complete inference path in NumPy: tokenization,
embeddings, causal multi-head attention, feed-forward layers, normalization,
the language-model head, greedy next-token selection, and autoregressive
generation. It can load the official DistilGPT-2 configuration, tokenizer
files, and safetensors checkpoint from a local directory.

## Run locally

Create and activate an isolated Python environment, then install the project:

```shell
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Download the official DistilGPT-2 files once:

```shell
mkdir -p models/distilgpt2
BASE=https://huggingface.co/distilbert/distilgpt2/resolve/main

for FILE in config.json vocab.json merges.txt model.safetensors
do
    curl -L --fail "$BASE/$FILE" -o "models/distilgpt2/$FILE"
done
```

Generate one next token:

```shell
llm-infer "the cat sat"
```

The official checkpoint currently produces:

```text
the cat sat in
```

Generate more than one token with, for example,
`llm-infer "the cat sat" --max-new-tokens 5`.

## Verify against Hugging Face

The downloaded model files belong in `models/distilgpt2/`, which Git ignores.
Install the optional reference tools and compare every output logit:

```shell
python -m pip install -e ".[reference]"
python -m examples.compare_distilgpt2 models/distilgpt2 \
    --prompt "the cat sat"
```

For the official checkpoint, both implementations encode the prompt as
`[1169, 3797, 3332]` and greedily select token `287`, which decodes to `" in"`.
The educational tiny weights used earlier can still be designed to demonstrate
`"the cat sat" -> " on"`; a real pretrained checkpoint is not hardcoded to that
example.

