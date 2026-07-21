"""Adapt a Hugging Face DistilGPT-2 configuration to ModelConfig."""

from collections.abc import Mapping

from llm_inference_engine.utils.config import ModelConfig


class DistilGPT2ConfigAdapter:
    """Translate DistilGPT-2 checkpoint dimensions into engine dimensions."""

    @classmethod
    def from_dict(
        cls,
        checkpoint_config: Mapping[str, object],
    ) -> ModelConfig:
        """Return a validated ModelConfig from checkpoint JSON values."""
        model_type = checkpoint_config.get("model_type")
        if model_type != "gpt2":
            raise ValueError("DistilGPT2ConfigAdapter requires model_type='gpt2'.")

        hidden_size = cls._required_positive_int(
            checkpoint_config,
            "n_embd",
        )
        configured_inner_size = checkpoint_config.get("n_inner")
        if configured_inner_size is None:
            ffn_hidden_size = 4 * hidden_size
        else:
            ffn_hidden_size = cls._positive_int(
                "n_inner",
                configured_inner_size,
            )

        return ModelConfig(
            vocab_size=cls._required_positive_int(
                checkpoint_config,
                "vocab_size",
            ),
            hidden_size=hidden_size,
            num_layers=cls._required_positive_int(
                checkpoint_config,
                "n_layer",
            ),
            num_attention_heads=cls._required_positive_int(
                checkpoint_config,
                "n_head",
            ),
            ffn_hidden_size=ffn_hidden_size,
            max_sequence_length=cls._required_positive_int(
                checkpoint_config,
                "n_positions",
            ),
        )

    @classmethod
    def _required_positive_int(
        cls,
        checkpoint_config: Mapping[str, object],
        name: str,
    ) -> int:
        """Read one required positive integer checkpoint field."""
        if name not in checkpoint_config:
            raise ValueError(f"checkpoint config is missing '{name}'.")

        return cls._positive_int(name, checkpoint_config[name])

    @staticmethod
    def _positive_int(name: str, value: object) -> int:
        """Validate and return one positive integer configuration value."""
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"checkpoint config '{name}' must be a positive integer."
            )

        return value
