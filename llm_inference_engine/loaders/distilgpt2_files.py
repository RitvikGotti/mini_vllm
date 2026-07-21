"""Load DistilGPT-2 configuration and weights from local files."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from safetensors.numpy import load_file

from llm_inference_engine.engine.inference import TinyInferenceEngine
from llm_inference_engine.engine.token_selection import GreedyTokenSelector
from llm_inference_engine.loaders.distilgpt2_config import (
    DistilGPT2ConfigAdapter,
)
from llm_inference_engine.loaders.distilgpt2_weights import (
    DistilGPT2WeightMapper,
)
from llm_inference_engine.model.activations import gelu_new
from llm_inference_engine.model.gpt2_tokenizer import GPT2BPETokenizer
from llm_inference_engine.model.tiny_model import (
    TinyTransformerModel,
    TinyTransformerWeights,
)
from llm_inference_engine.utils.config import ModelConfig


@dataclass(frozen=True)
class LoadedDistilGPT2:
    """Hold validated model artifacts loaded from a checkpoint directory."""

    config: ModelConfig
    weights: TinyTransformerWeights
    norm_epsilon: float

    def create_model(self) -> TinyTransformerModel:
        """Build an inference model with DistilGPT-2 runtime settings."""
        return TinyTransformerModel(
            self.config,
            self.weights,
            norm_epsilon=self.norm_epsilon,
            ffn_activation=gelu_new,
        )


class DistilGPT2FileLoader:
    """Read and validate a local Hugging Face DistilGPT-2 directory."""

    def load(self, model_directory: str | Path) -> LoadedDistilGPT2:
        """Load config.json and model.safetensors from one directory."""
        directory = Path(model_directory)
        if not directory.is_dir():
            raise ValueError(f"model directory does not exist: {directory}")

        checkpoint_config = self._load_config_json(directory / "config.json")
        config = DistilGPT2ConfigAdapter.from_dict(checkpoint_config)
        self._validate_activation(checkpoint_config)
        norm_epsilon = self._read_norm_epsilon(checkpoint_config)
        tensors = self._load_safetensors(directory / "model.safetensors")
        weights = DistilGPT2WeightMapper(config).map_weights(tensors)

        return LoadedDistilGPT2(
            config=config,
            weights=weights,
            norm_epsilon=norm_epsilon,
        )

    def load_inference_engine(
        self,
        model_directory: str | Path,
    ) -> TinyInferenceEngine:
        """Build text-to-text inference from one local model directory."""
        loaded_model = self.load(model_directory)
        tokenizer = GPT2BPETokenizer.from_directory(model_directory)

        return TinyInferenceEngine(
            tokenizer,
            loaded_model.create_model(),
            GreedyTokenSelector(),
        )

    @staticmethod
    def _load_config_json(config_path: Path) -> dict[str, object]:
        """Read a JSON object containing checkpoint configuration values."""
        try:
            with config_path.open(encoding="utf-8") as config_file:
                checkpoint_config = json.load(config_file)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"could not read config file: {config_path}"
            ) from error

        if not isinstance(checkpoint_config, dict):
            raise ValueError("checkpoint config must contain a JSON object.")

        return checkpoint_config

    @staticmethod
    def _load_safetensors(
        checkpoint_path: Path,
    ) -> dict[str, NDArray[np.floating]]:
        """Read named NumPy tensors from a safetensors checkpoint."""
        if not checkpoint_path.is_file():
            raise ValueError(
                f"checkpoint file does not exist: {checkpoint_path}"
            )

        try:
            return load_file(checkpoint_path)
        except (OSError, ValueError) as error:
            raise ValueError(
                f"could not read safetensors checkpoint: {checkpoint_path}"
            ) from error

    @staticmethod
    def _validate_activation(checkpoint_config: dict[str, object]) -> None:
        """Require the GPT-2 GELU variant implemented by this engine."""
        activation = checkpoint_config.get("activation_function", "gelu_new")
        if activation != "gelu_new":
            raise ValueError(
                "DistilGPT-2 checkpoint activation_function must be 'gelu_new'."
            )

    @staticmethod
    def _read_norm_epsilon(checkpoint_config: dict[str, object]) -> float:
        """Return DistilGPT-2's positive LayerNorm epsilon."""
        value = checkpoint_config.get("layer_norm_epsilon", 1e-5)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("layer_norm_epsilon must be a positive number.")

        norm_epsilon = float(value)
        if norm_epsilon <= 0.0:
            raise ValueError("layer_norm_epsilon must be a positive number.")

        return norm_epsilon
