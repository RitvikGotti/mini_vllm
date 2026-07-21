"""Map Hugging Face DistilGPT-2 tensors into engine weight containers."""

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from llm_inference_engine.model.tiny_model import (
    TinyTransformerWeights,
    TransformerLayerWeights,
)
from llm_inference_engine.utils.config import ModelConfig


CheckpointTensors = Mapping[str, NDArray[np.floating]]


class DistilGPT2WeightMapper:
    """Translate named DistilGPT-2 tensors into the engine's layout."""

    def __init__(self, config: ModelConfig) -> None:
        """Store the architecture that every checkpoint tensor must match."""
        self._config = config

    def map_weights(
        self,
        checkpoint: CheckpointTensors,
    ) -> TinyTransformerWeights:
        """Validate and map all tensors needed for causal LM inference."""
        hidden_size = self._config.hidden_size
        token_embedding = self._required_tensor(
            checkpoint,
            "transformer.wte.weight",
            (self._config.vocab_size, hidden_size),
        )
        position_embedding = self._required_tensor(
            checkpoint,
            "transformer.wpe.weight",
            (self._config.max_sequence_length, hidden_size),
        )

        layers = tuple(
            self._map_layer(checkpoint, layer_index)
            for layer_index in range(self._config.num_layers)
        )
        final_norm_scale = self._required_tensor(
            checkpoint,
            "transformer.ln_f.weight",
            (hidden_size,),
        )
        final_norm_bias = self._required_tensor(
            checkpoint,
            "transformer.ln_f.bias",
            (hidden_size,),
        )

        if "lm_head.weight" in checkpoint:
            checkpoint_lm_head = self._required_tensor(
                checkpoint,
                "lm_head.weight",
                (self._config.vocab_size, hidden_size),
            )
            lm_head = checkpoint_lm_head.T
        else:
            lm_head = token_embedding.T

        return TinyTransformerWeights(
            token_embedding=token_embedding,
            position_embedding=position_embedding,
            layers=layers,
            final_norm_scale=final_norm_scale,
            final_norm_bias=final_norm_bias,
            lm_head=lm_head,
        )

    def _map_layer(
        self,
        checkpoint: CheckpointTensors,
        layer_index: int,
    ) -> TransformerLayerWeights:
        """Map one numbered GPT-2 transformer block."""
        hidden_size = self._config.hidden_size
        ffn_hidden_size = self._config.ffn_hidden_size
        prefix = f"transformer.h.{layer_index}"

        combined_qkv = self._required_tensor(
            checkpoint,
            f"{prefix}.attn.c_attn.weight",
            (hidden_size, 3 * hidden_size),
        )
        query, key, value = np.split(combined_qkv, 3, axis=1)

        combined_qkv_bias = self._required_tensor(
            checkpoint,
            f"{prefix}.attn.c_attn.bias",
            (3 * hidden_size,),
        )
        query_bias, key_bias, value_bias = np.split(
            combined_qkv_bias,
            3,
        )

        return TransformerLayerWeights(
            query=query,
            key=key,
            value=value,
            attention_output=self._required_tensor(
                checkpoint,
                f"{prefix}.attn.c_proj.weight",
                (hidden_size, hidden_size),
            ),
            attention_norm_scale=self._required_tensor(
                checkpoint,
                f"{prefix}.ln_1.weight",
                (hidden_size,),
            ),
            attention_norm_bias=self._required_tensor(
                checkpoint,
                f"{prefix}.ln_1.bias",
                (hidden_size,),
            ),
            ffn_input=self._required_tensor(
                checkpoint,
                f"{prefix}.mlp.c_fc.weight",
                (hidden_size, ffn_hidden_size),
            ),
            ffn_output=self._required_tensor(
                checkpoint,
                f"{prefix}.mlp.c_proj.weight",
                (ffn_hidden_size, hidden_size),
            ),
            ffn_norm_scale=self._required_tensor(
                checkpoint,
                f"{prefix}.ln_2.weight",
                (hidden_size,),
            ),
            ffn_norm_bias=self._required_tensor(
                checkpoint,
                f"{prefix}.ln_2.bias",
                (hidden_size,),
            ),
            query_bias=query_bias,
            key_bias=key_bias,
            value_bias=value_bias,
            attention_output_bias=self._required_tensor(
                checkpoint,
                f"{prefix}.attn.c_proj.bias",
                (hidden_size,),
            ),
            ffn_input_bias=self._required_tensor(
                checkpoint,
                f"{prefix}.mlp.c_fc.bias",
                (ffn_hidden_size,),
            ),
            ffn_output_bias=self._required_tensor(
                checkpoint,
                f"{prefix}.mlp.c_proj.bias",
                (hidden_size,),
            ),
        )

    @staticmethod
    def _required_tensor(
        checkpoint: CheckpointTensors,
        name: str,
        expected_shape: tuple[int, ...],
    ) -> NDArray[np.floating]:
        """Return one required floating tensor with its expected shape."""
        if name not in checkpoint:
            raise ValueError(f"checkpoint is missing required tensor '{name}'.")

        tensor = checkpoint[name]
        if tensor.shape != expected_shape:
            raise ValueError(
                f"checkpoint tensor '{name}' must have shape "
                f"{expected_shape}, got {tensor.shape}."
            )

        if not np.issubdtype(tensor.dtype, np.floating):
            raise ValueError(
                f"checkpoint tensor '{name}' must use a floating-point dtype."
            )

        return tensor
