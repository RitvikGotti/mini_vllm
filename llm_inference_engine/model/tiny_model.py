"""Connected one-layer, single-head transformer language model."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from llm_inference_engine.model.embeddings import TokenEmbedding
from llm_inference_engine.model.input_embeddings import InputEmbedding
from llm_inference_engine.model.lm_head import LanguageModelHead
from llm_inference_engine.model.normalization import LayerNorm
from llm_inference_engine.model.positions import PositionEmbedding
from llm_inference_engine.model.transformer_block import (
    SingleHeadTransformerBlock,
)
from llm_inference_engine.utils.config import ModelConfig


@dataclass(frozen=True)
class TinyTransformerWeights:
    """Collect the learned tensors required by the one-layer tiny model."""

    token_embedding: NDArray[np.floating]
    position_embedding: NDArray[np.floating]
    query: NDArray[np.floating]
    key: NDArray[np.floating]
    value: NDArray[np.floating]
    attention_output: NDArray[np.floating]
    attention_norm_scale: NDArray[np.floating]
    attention_norm_bias: NDArray[np.floating]
    ffn_input: NDArray[np.floating]
    ffn_output: NDArray[np.floating]
    ffn_norm_scale: NDArray[np.floating]
    ffn_norm_bias: NDArray[np.floating]
    final_norm_scale: NDArray[np.floating]
    final_norm_bias: NDArray[np.floating]
    lm_head: NDArray[np.floating]


class TinyTransformerModel:
    """Run token IDs through one single-head transformer layer to logits."""

    def __init__(
        self,
        config: ModelConfig,
        weights: TinyTransformerWeights,
        norm_epsilon: float = 1e-5,
    ) -> None:
        """Create the connected tiny model from configuration and weights."""
        if config.num_layers != 1:
            raise ValueError("TinyTransformerModel requires num_layers=1.")

        if config.num_attention_heads != 1:
            raise ValueError(
                "TinyTransformerModel requires num_attention_heads=1."
            )

        token_embedding = TokenEmbedding(config, weights.token_embedding)
        position_embedding = PositionEmbedding(
            config,
            weights.position_embedding,
        )
        self._input_embedding = InputEmbedding(
            token_embedding,
            position_embedding,
        )
        self._transformer_block = SingleHeadTransformerBlock(
            config,
            query_weights=weights.query,
            key_weights=weights.key,
            value_weights=weights.value,
            attention_output_weights=weights.attention_output,
            attention_norm_scale=weights.attention_norm_scale,
            attention_norm_bias=weights.attention_norm_bias,
            ffn_input_weights=weights.ffn_input,
            ffn_output_weights=weights.ffn_output,
            ffn_norm_scale=weights.ffn_norm_scale,
            ffn_norm_bias=weights.ffn_norm_bias,
            norm_epsilon=norm_epsilon,
        )
        self._final_norm = LayerNorm(
            config,
            weights.final_norm_scale,
            weights.final_norm_bias,
            norm_epsilon,
        )
        self._lm_head = LanguageModelHead(config, weights.lm_head)

    def forward(
        self,
        token_ids: NDArray[np.integer],
        position_offset: int = 0,
    ) -> NDArray[np.floating]:
        """Return vocabulary logits for every input token position."""
        hidden_states = self._input_embedding.forward(
            token_ids,
            position_offset,
        )
        hidden_states = self._transformer_block.forward(hidden_states)
        hidden_states = self._final_norm.forward(hidden_states)

        return self._lm_head.forward(hidden_states)
