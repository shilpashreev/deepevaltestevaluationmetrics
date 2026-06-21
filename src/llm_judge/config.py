"""Configuration loading and management for LLM-as-Judge.

Supports YAML file configuration with environment variable overrides.
Environment variables follow the pattern ``LLM_JUDGE_<FIELD>`` (upper-case).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Environment variable prefix
# ---------------------------------------------------------------------------
_ENV_PREFIX = "LLM_JUDGE_"


@dataclass
class Config:
    """Framework configuration.

    Attributes:
        provider: LLM provider name (openai, anthropic, google).
        model: Model identifier to use for judging.
        api_key: API key; when *None* the provider's default env var is used.
        criteria: List of criterion names to evaluate. Empty means all.
        pass_threshold: Minimum aggregate score for a test case to pass.
        concurrency: Maximum number of concurrent LLM calls.
        output_dir: Directory for writing result artifacts.
        report_format: Output report format (console, json, html).
        temperature: Sampling temperature for the judge model.
        max_retries: Number of retries on transient LLM errors.
    """

    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    criteria: list[str] = field(default_factory=list)
    pass_threshold: float = 0.7
    concurrency: int = 5
    output_dir: str = "./results"
    report_format: str = "console"
    temperature: float = 0.0
    max_retries: int = 3


# ---------------------------------------------------------------------------
# Mapping from Config field -> (env var name, type converter)
# ---------------------------------------------------------------------------
_ENV_MAP: dict[str, tuple[str, type]] = {
    "provider": (f"{_ENV_PREFIX}PROVIDER", str),
    "model": (f"{_ENV_PREFIX}MODEL", str),
    "api_key": (f"{_ENV_PREFIX}API_KEY", str),
    "pass_threshold": (f"{_ENV_PREFIX}PASS_THRESHOLD", float),
    "concurrency": (f"{_ENV_PREFIX}CONCURRENCY", int),
    "output_dir": (f"{_ENV_PREFIX}OUTPUT_DIR", str),
    "report_format": (f"{_ENV_PREFIX}REPORT_FORMAT", str),
    "temperature": (f"{_ENV_PREFIX}TEMPERATURE", float),
    "max_retries": (f"{_ENV_PREFIX}MAX_RETRIES", int),
}


def _apply_env_overrides(cfg: Config) -> None:
    """Overlay environment variable values onto *cfg* in-place."""
    for field_name, (env_var, conv) in _ENV_MAP.items():
        value = os.environ.get(env_var)
        if value is not None:
            setattr(cfg, field_name, conv(value))

    # Criteria can be set as a comma-separated string
    criteria_env = os.environ.get(f"{_ENV_PREFIX}CRITERIA")
    if criteria_env is not None:
        cfg.criteria = [c.strip() for c in criteria_env.split(",") if c.strip()]


def _dict_to_config(data: dict[str, Any]) -> Config:
    """Create a ``Config`` from a raw dictionary (e.g. parsed YAML).

    Unknown keys are silently ignored so the YAML file can contain
    comments or future fields without breaking older versions.
    """
    known_fields = {f.name for f in Config.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    filtered = {k: v for k, v in data.items() if k in known_fields}
    return Config(**filtered)


def load_config(path: str | None = None) -> Config:
    """Load configuration from a YAML file with environment variable overrides.

    Resolution order (last wins):
        1. Dataclass defaults
        2. YAML file values (if *path* is provided and file exists)
        3. ``LLM_JUDGE_*`` environment variables

    Args:
        path: Optional filesystem path to a YAML configuration file.

    Returns:
        A fully-resolved ``Config`` instance.

    Raises:
        FileNotFoundError: If *path* is given but the file does not exist.
        yaml.YAMLError: If the YAML file is malformed.
    """
    if path is not None:
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with config_path.open("r", encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}

        cfg = _dict_to_config(raw)
    else:
        cfg = Config()

    _apply_env_overrides(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Module-level convenience instance
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = Config()
