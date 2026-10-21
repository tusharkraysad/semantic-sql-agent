"""
Settings
========
Everything tunable lives here and is read from environment variables, so the
same image runs on a laptop, in Docker Compose and in Kubernetes with nothing
but different env values. Defaults work offline with no API key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # -- paths ---------------------------------------------------------------
    data_dir: str = field(default_factory=lambda: _env("DATA_DIR", "data"))
    artifact_root: str = field(default_factory=lambda: _env("ARTIFACT_DIR", "artifacts"))
    overrides_path: str = field(default_factory=lambda: _env("SEMANTIC_OVERRIDES", "config/semantic_overrides.yaml"))
    state_dir: str = field(default_factory=lambda: _env("STATE_DIR", "state"))
    # Build (and generate demo data if missing) on startup when no matching artifacts exist.
    # On for local and free hosting; off in Kubernetes, where a Job builds artifacts.
    auto_build: bool = field(default_factory=lambda: _env_bool("AUTO_BUILD", False))
    generate_if_missing: bool = field(default_factory=lambda: _env_bool("GENERATE_DEMO_DATA", False))

    # -- models (OpenAI-compatible; "none" runs fully deterministic) ---------
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "none"))
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", ""))
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", ""))
    planner_model: str = field(default_factory=lambda: _env("PLANNER_MODEL", ""))
    writer_model: str = field(default_factory=lambda: _env("WRITER_MODEL", ""))
    llm_timeout_s: int = field(default_factory=lambda: _env_int("LLM_TIMEOUT_S", 45))
    max_tokens_planner: int = field(default_factory=lambda: _env_int("MAX_TOKENS_PLANNER", 1200))
    max_tokens_writer: int = field(default_factory=lambda: _env_int("MAX_TOKENS_WRITER", 700))
    max_llm_calls_per_question: int = field(default_factory=lambda: _env_int("MAX_LLM_CALLS", 3))

    # -- semantic layer tunables ---------------------------------------------
    value_fuzzy_threshold: float = field(default_factory=lambda: _env_float("VALUE_FUZZY_THRESHOLD", 0.86))
    value_ambiguity_margin: float = 0.03
    max_values_indexed_per_column: int = 5000
    min_join_value_alignment: float = 0.35
    repeated_base_constant_ratio: float = 0.95
    repeated_base_min_rows_per_group: float = 2.0
    repeated_base_min_distinct_per_group: float = 0.5
    confidence_floor: float = field(default_factory=lambda: _env_float("CONFIDENCE_FLOOR", 0.55))

    # -- retrieval weights ---------------------------------------------------
    dense_weight: float = 1.0
    lexical_weight: float = 0.6
    semantic_boost: float = 1.2
    value_boost: float = 1.5
    table_prior_weight: float = 0.4
    top_k_columns: int = 14

    # -- execution / service -------------------------------------------------
    max_result_rows: int = field(default_factory=lambda: _env_int("MAX_RESULT_ROWS", 200))
    query_timeout_s: int = field(default_factory=lambda: _env_int("QUERY_TIMEOUT_S", 30))
    max_concurrent_queries: int = field(default_factory=lambda: _env_int("MAX_CONCURRENT_QUERIES", 4))
    cache_size: int = field(default_factory=lambda: _env_int("CACHE_SIZE", 256))
    small_base_ratio: float = 0.05

    @property
    def llm_enabled(self) -> bool:
        return self.llm_provider.lower() not in {"", "none", "off"} and bool(self.llm_base_url)

    @property
    def decisions_path(self) -> str:
        return os.path.join(self.state_dir, "decisions.jsonl")

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.state_dir, "checkpoints.sqlite")


def get_settings() -> Settings:
    return Settings()
