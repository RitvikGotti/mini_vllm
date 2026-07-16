"""Input embedding composition for transformer inference."""

import numpy as np
from numpy.typing import NDArray

from llm_inference_engine.model.embeddings import TokenEmbedding
from llm_inference_engine.model.positions import PositionEmbedding


class InputEmbedding:
    """Combine token and positional vectors into initial hidden states."""

    def __init__(
        self,
        token_embedding: TokenEmbedding,
        position_embedding: PositionEmbedding,
    ) -> None:
        """Create an input embedding layer from token and position lookups."""
        self._token_embedding = token_embedding
        self._position_embedding = position_embedding

    def forward(
        self, token_ids: NDArray[np.integer], position_offset: int = 0
    ) -> NDArray[np.floating]:
        """Return token vectors plus their corresponding position vectors."""
        token_vectors = self._token_embedding.forward(token_ids)
        position_vectors = self._position_embedding.forward(
            sequence_length=token_ids.size,
            offset=position_offset,
        )

        if token_vectors.shape != position_vectors.shape:
            raise ValueError("token and position vectors must have the same shape.")

        return token_vectors + position_vectors

