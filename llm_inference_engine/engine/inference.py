"""End-to-end next-token inference for the educational tiny model."""

import numpy as np

from llm_inference_engine.engine.token_selection import GreedyTokenSelector
from llm_inference_engine.model.tiny_model import TinyTransformerModel
from llm_inference_engine.model.tokenizer import WhitespaceTokenizer


class TinyInferenceEngine:
    """Connect text tokenization, model execution, and token selection."""

    def __init__(
        self,
        tokenizer: WhitespaceTokenizer,
        model: TinyTransformerModel,
        token_selector: GreedyTokenSelector,
    ) -> None:
        """Create an inference engine from its three main components."""
        self._tokenizer = tokenizer
        self._model = model
        self._token_selector = token_selector

    def predict_next_token(self, prompt: str) -> str:
        """Return the greedily selected next token for a text prompt."""
        token_ids = self._encode_prompt(prompt)
        next_token_id = self._predict_next_token_id(token_ids)

        return self._tokenizer.decode([next_token_id])

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        """Greedily append up to max_new_tokens to a text prompt."""
        if not isinstance(max_new_tokens, int):
            raise ValueError("max_new_tokens must be an integer.")

        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative.")

        token_ids = self._encode_prompt(prompt)

        for _ in range(max_new_tokens):
            next_token_id = self._predict_next_token_id(token_ids)
            token_ids.append(next_token_id)

        return self._tokenizer.decode(token_ids)

    def _encode_prompt(self, prompt: str) -> list[int]:
        """Encode a prompt and require at least one input token."""
        token_ids = self._tokenizer.encode(prompt)

        if not token_ids:
            raise ValueError("prompt must contain at least one token.")

        return token_ids

    def _predict_next_token_id(self, token_ids: list[int]) -> int:
        """Run the model and greedily choose the next token ID."""
        logits = self._model.forward(
            np.asarray(token_ids, dtype=np.int64)
        )

        return self._token_selector.select(logits)
