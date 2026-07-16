"""A minimal tokenizer for early inference-engine milestones."""

from collections.abc import Mapping


class WhitespaceTokenizer:
    """Convert whitespace-delimited tokens to and from fixed vocabulary IDs."""

    def __init__(self, vocabulary: Mapping[str, int]) -> None:
        """Create a tokenizer from a token-to-ID vocabulary mapping."""
        if not vocabulary:
            raise ValueError("vocabulary must not be empty.")

        self._token_to_id = dict(vocabulary)
        self._id_to_token = {token_id: token for token, token_id in vocabulary.items()}

        if len(self._id_to_token) != len(self._token_to_id):
            raise ValueError("vocabulary token IDs must be unique.")

        if any(token_id < 0 for token_id in self._id_to_token):
            raise ValueError("vocabulary token IDs must be non-negative.")

    def encode(self, text: str) -> list[int]:
        """Return vocabulary IDs for whitespace-delimited tokens in text."""
        tokens = text.split()
        token_ids: list[int] = []

        for token in tokens:
            try:
                token_ids.append(self._token_to_id[token])
            except KeyError as error:
                raise ValueError(f"unknown token: {token!r}") from error

        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """Return whitespace-delimited text for vocabulary token IDs."""
        tokens: list[str] = []

        for token_id in token_ids:
            try:
                tokens.append(self._id_to_token[token_id])
            except KeyError as error:
                raise ValueError(f"unknown token ID: {token_id}") from error

        return " ".join(tokens)


