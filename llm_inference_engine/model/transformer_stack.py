"""Sequential composition of transformer blocks."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from llm_inference_engine.model.transformer_block import (
    SingleHeadTransformerBlock,
)
from llm_inference_engine.utils.config import ModelConfig


class TransformerStack:
    """Pass hidden states through every transformer block in order."""

    def __init__(
        self,
        config: ModelConfig,
        blocks: Sequence[SingleHeadTransformerBlock],
    ) -> None:
        """Create a stack containing exactly the configured number of layers."""
        if len(blocks) != config.num_layers:
            raise ValueError(
                "number of transformer blocks must match config.num_layers."
            )

        self._blocks = tuple(blocks)

    def forward(
        self,
        hidden_states: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Return hidden states after every transformer block."""
        for block in self._blocks:
            hidden_states = block.forward(hidden_states)

        return hidden_states
