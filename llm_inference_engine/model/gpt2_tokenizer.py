"""GPT-2 byte-level byte-pair encoding for pretrained inference."""

from collections.abc import Mapping, Sequence
import json
from pathlib import Path

import regex


_END_OF_TEXT = "<|endoftext|>"
_TOKEN_PATTERN = regex.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+|"
    r" ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)


class GPT2BPETokenizer:
    """Encode text with GPT-2's byte mapping and ranked BPE merges."""

    def __init__(
        self,
        vocabulary: Mapping[str, int],
        merges: Sequence[tuple[str, str]],
    ) -> None:
        """Create a tokenizer from GPT-2 vocabulary and merge data."""
        if not vocabulary:
            raise ValueError("vocabulary must not be empty.")

        self._token_to_id = dict(vocabulary)
        self._id_to_token = {
            token_id: token for token, token_id in self._token_to_id.items()
        }
        if len(self._id_to_token) != len(self._token_to_id):
            raise ValueError("vocabulary token IDs must be unique.")

        for token, token_id in self._token_to_id.items():
            if (
                not isinstance(token, str)
                or isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
            ):
                raise ValueError(
                    "vocabulary must map strings to non-negative integer IDs."
                )

        self._bpe_ranks: dict[tuple[str, str], int] = {}
        for rank, pair in enumerate(merges):
            if len(pair) != 2 or pair in self._bpe_ranks:
                raise ValueError("BPE merge pairs must be unique token pairs.")
            self._bpe_ranks[pair] = rank

        self._byte_encoder = self._build_byte_encoder()
        self._byte_decoder = {
            encoded_byte: byte
            for byte, encoded_byte in self._byte_encoder.items()
        }
        self._bpe_cache: dict[str, tuple[str, ...]] = {}
        self._end_of_text_id = self._token_to_id.get(_END_OF_TEXT)

    @classmethod
    def from_files(
        cls,
        vocabulary_path: str | Path,
        merges_path: str | Path,
    ) -> "GPT2BPETokenizer":
        """Create a tokenizer from vocab.json and merges.txt files."""
        vocabulary = cls._load_vocabulary(Path(vocabulary_path))
        merges = cls._load_merges(Path(merges_path))
        return cls(vocabulary, merges)

    @classmethod
    def from_directory(
        cls,
        model_directory: str | Path,
    ) -> "GPT2BPETokenizer":
        """Load GPT-2 tokenizer assets from one model directory."""
        directory = Path(model_directory)
        return cls.from_files(
            directory / "vocab.json",
            directory / "merges.txt",
        )

    def encode(self, text: str) -> list[int]:
        """Return GPT-2 vocabulary IDs for text."""
        if not isinstance(text, str):
            raise ValueError("text must be a string.")

        if self._end_of_text_id is None:
            return self._encode_ordinary_text(text)

        token_ids: list[int] = []
        sections = text.split(_END_OF_TEXT)
        for index, section in enumerate(sections):
            token_ids.extend(self._encode_ordinary_text(section))
            if index < len(sections) - 1:
                token_ids.append(self._end_of_text_id)

        return token_ids

    def decode(self, token_ids: Sequence[int]) -> str:
        """Return UTF-8 text represented by GPT-2 token IDs."""
        encoded_text_parts: list[str] = []
        for token_id in token_ids:
            try:
                encoded_text_parts.append(self._id_to_token[token_id])
            except KeyError as error:
                raise ValueError(f"unknown token ID: {token_id}") from error

        encoded_text = "".join(encoded_text_parts)
        try:
            byte_values = bytes(
                self._byte_decoder[character] for character in encoded_text
            )
        except KeyError as error:
            raise ValueError(
                "vocabulary contains an invalid byte token."
            ) from error

        return byte_values.decode("utf-8", errors="replace")

    def _encode_ordinary_text(self, text: str) -> list[int]:
        """Encode text that contains no recognized special token."""
        token_ids: list[int] = []
        for text_piece in _TOKEN_PATTERN.findall(text):
            encoded_piece = "".join(
                self._byte_encoder[byte]
                for byte in text_piece.encode("utf-8")
            )
            for bpe_token in self._apply_bpe(encoded_piece):
                try:
                    token_ids.append(self._token_to_id[bpe_token])
                except KeyError as error:
                    raise ValueError(
                        f"BPE token is missing from vocabulary: {bpe_token!r}"
                    ) from error

        return token_ids

    def _apply_bpe(self, token: str) -> tuple[str, ...]:
        """Merge adjacent symbols according to learned BPE rank order."""
        cached = self._bpe_cache.get(token)
        if cached is not None:
            return cached

        word = tuple(token)
        while len(word) > 1:
            pairs = set(zip(word, word[1:]))
            best_pair = min(
                pairs,
                key=lambda pair: self._bpe_ranks.get(pair, float("inf")),
            )
            if best_pair not in self._bpe_ranks:
                break

            first, second = best_pair
            merged_word: list[str] = []
            index = 0
            while index < len(word):
                if (
                    index < len(word) - 1
                    and word[index] == first
                    and word[index + 1] == second
                ):
                    merged_word.append(first + second)
                    index += 2
                else:
                    merged_word.append(word[index])
                    index += 1

            word = tuple(merged_word)

        self._bpe_cache[token] = word
        return word

    @staticmethod
    def _build_byte_encoder() -> dict[int, str]:
        """Map all bytes to reversible visible Unicode characters."""
        visible_bytes = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("\u00a1"), ord("\u00ac") + 1))
            + list(range(ord("\u00ae"), ord("\u00ff") + 1))
        )
        encoded_values = list(visible_bytes)
        visible_byte_set = set(visible_bytes)
        extra_index = 0

        for byte in range(256):
            if byte not in visible_byte_set:
                visible_bytes.append(byte)
                encoded_values.append(256 + extra_index)
                extra_index += 1

        return {
            byte: chr(encoded_value)
            for byte, encoded_value in zip(visible_bytes, encoded_values)
        }

    @staticmethod
    def _load_vocabulary(vocabulary_path: Path) -> dict[str, int]:
        """Read the token-to-ID object stored in vocab.json."""
        try:
            with vocabulary_path.open(encoding="utf-8") as vocabulary_file:
                vocabulary = json.load(vocabulary_file)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"could not read tokenizer vocabulary: {vocabulary_path}"
            ) from error

        if not isinstance(vocabulary, dict):
            raise ValueError("tokenizer vocabulary must contain a JSON object.")

        return vocabulary

    @staticmethod
    def _load_merges(merges_path: Path) -> list[tuple[str, str]]:
        """Read ranked BPE token pairs from merges.txt."""
        try:
            lines = merges_path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise ValueError(
                f"could not read tokenizer merges: {merges_path}"
            ) from error

        merges: list[tuple[str, str]] = []
        for line_number, line in enumerate(lines, start=1):
            stripped_line = line.strip()
            if not stripped_line or stripped_line.startswith("#version:"):
                continue

            pair = tuple(stripped_line.split())
            if len(pair) != 2:
                raise ValueError(
                    f"invalid BPE merge on line {line_number}: {line!r}"
                )
            merges.append((pair[0], pair[1]))

        return merges
