"""Language-model head for vocabulary-logit projection."""

import numpy as np
from numpy.typing import NDArray

from llm_inference_engine.utils.config import ModelConfig


class LanguageModelHead:
    """Project hidden states into one logit per vocabulary token."""

    def __init__(
        self,
        config: ModelConfig,
        weights: NDArray[np.floating],
    ) -> None:
        """Create an LM head from its learned projection weights."""
        expected_shape = (config.hidden_size, config.vocab_size)

        if weights.shape != expected_shape:
            raise ValueError(
                f"LM-head weights must have shape {expected_shape}, "
                f"got {weights.shape}."
            )

        if not np.issubdtype(weights.dtype, np.floating):
            raise ValueError("LM-head weights must use a floating-point dtype.")

        self._weights = weights

    def forward(
        self, hidden_states: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Return vocabulary logits for every sequence hidden state."""
        if hidden_states.ndim != 2:
            raise ValueError("hidden_states must be a two-dimensional array.")

        if hidden_states.shape[1] != self._weights.shape[0]:
            raise ValueError(
                "hidden_states feature dimension must match LM-head weights."
            )

        if not np.issubdtype(hidden_states.dtype, np.floating):
            raise ValueError("hidden_states must use a floating-point dtype.")

        return hidden_states @ self._weights
