"""Single-head compatibility wrapper for multi-head self-attention."""

import numpy as np
from numpy.typing import NDArray

from llm_inference_engine.model.multi_head_attention import (
    MultiHeadSelfAttention,
)
from llm_inference_engine.utils.config import ModelConfig


class SingleHeadSelfAttention(MultiHeadSelfAttention):
    """Run multi-head attention with exactly one configured head."""

    def __init__(
        self,
        config: ModelConfig,
        query_weights: NDArray[np.floating],
        key_weights: NDArray[np.floating],
        value_weights: NDArray[np.floating],
        output_weights: NDArray[np.floating],
        query_bias: NDArray[np.floating] | None = None,
        key_bias: NDArray[np.floating] | None = None,
        value_bias: NDArray[np.floating] | None = None,
        output_bias: NDArray[np.floating] | None = None,
    ) -> None:
        """Create attention while preserving the original single-head API."""
        if config.num_attention_heads != 1:
            raise ValueError(
                "SingleHeadSelfAttention requires num_attention_heads=1."
            )

        super().__init__(
            config,
            query_weights,
            key_weights,
            value_weights,
            output_weights,
            query_bias,
            key_bias,
            value_bias,
            output_bias,
        )
