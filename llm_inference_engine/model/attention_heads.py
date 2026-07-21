"""Split hidden features across attention heads and merge them again."""

import numpy as np
from numpy.typing import NDArray

from llm_inference_engine.utils.config import ModelConfig


class AttentionHeadSplitter:
    """Reshape projected token states into separate attention heads."""

    def __init__(self, config: ModelConfig) -> None:
        """Store the configured head count and dimensions."""
        self._hidden_size = config.hidden_size
        self._num_heads = config.num_attention_heads
        self._head_dim = config.head_dim

    def forward(
        self,
        projected_states: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Return states shaped as heads by sequence by head dimension."""
        if projected_states.ndim != 2:
            raise ValueError(
                "projected_states must be a two-dimensional array."
            )

        if projected_states.shape[1] != self._hidden_size:
            raise ValueError(
                "projected_states feature dimension must match hidden_size."
            )

        if not np.issubdtype(projected_states.dtype, np.floating):
            raise ValueError(
                "projected_states must use a floating-point dtype."
            )

        sequence_length = projected_states.shape[0]
        states_by_token_and_head = projected_states.reshape(
            sequence_length,
            self._num_heads,
            self._head_dim,
        )

        return states_by_token_and_head.transpose(1, 0, 2)


class AttentionHeadMerger:
    """Restore separate attention heads to the model hidden width."""

    def __init__(self, config: ModelConfig) -> None:
        """Store the configured head count and dimensions."""
        self._hidden_size = config.hidden_size
        self._num_heads = config.num_attention_heads
        self._head_dim = config.head_dim

    def forward(
        self,
        head_states: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Return states shaped as sequence by complete hidden dimension."""
        if head_states.ndim != 3:
            raise ValueError("head_states must be a three-dimensional array.")

        if head_states.shape[0] != self._num_heads:
            raise ValueError(
                "head_states first dimension must match num_attention_heads."
            )

        if head_states.shape[2] != self._head_dim:
            raise ValueError(
                "head_states feature dimension must match head_dim."
            )

        if not np.issubdtype(head_states.dtype, np.floating):
            raise ValueError("head_states must use a floating-point dtype.")

        sequence_length = head_states.shape[1]
        states_by_token = head_states.transpose(1, 0, 2)

        return states_by_token.reshape(sequence_length, self._hidden_size)
