"""Model architecture configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Describe the fixed architecture of a transformer language model.

    This configuration specifies tensor dimensions, not learned model weights.
    """

    vocab_size: int
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    ffn_hidden_size: int
    max_sequence_length: int

    def __post_init__(self) -> None:
        """Validate that the architecture dimensions are usable."""
        dimensions = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "ffn_hidden_size": self.ffn_hidden_size,
            "max_sequence_length": self.max_sequence_length,
        }
        for name, value in dimensions.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads."
            )

    @property
    def head_dim(self) -> int:
        """Return the number of hidden dimensions assigned to each attention head."""
        return self.hidden_size // self.num_attention_heads

