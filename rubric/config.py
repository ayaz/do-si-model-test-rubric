"""Load the model list / thresholds config and the API key from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ENV_KEY = "DIGITALOCEAN_INFERENCE_KEY"
DEFAULT_CONFIG_PATH = "models.yaml"


@dataclass
class Config:
    models: list[str]
    thresholds: dict[str, float] = field(default_factory=dict)


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Read models.yaml and return the list of models plus any threshold overrides."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}. Copy the shipped models.yaml and edit it."
        )

    data = yaml.safe_load(path.read_text()) or {}

    models = data.get("models") or []
    if not isinstance(models, list) or not all(isinstance(m, str) for m in models):
        raise ValueError("`models` in the config must be a list of model-ID strings.")
    if not models:
        raise ValueError("No models listed in the config — add at least one model ID.")

    thresholds = data.get("thresholds") or {}
    if not isinstance(thresholds, dict):
        raise ValueError("`thresholds` in the config must be a mapping if present.")

    return Config(models=models, thresholds={k: float(v) for k, v in thresholds.items()})


def get_api_key() -> str:
    """Read the DO model access key from the environment, with a clear error if missing."""
    key = os.environ.get(ENV_KEY)
    if not key:
        raise RuntimeError(
            f"Environment variable {ENV_KEY} is not set. "
            f"Export it or put it in a .env file (see .env.example)."
        )
    return key
