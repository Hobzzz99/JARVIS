"""Centralised runtime configuration for JARVIS.

Every tunable lives here so agents, tools and the API read from one place
instead of scattering ``os.getenv`` calls across the codebase. Values are
resolved from the environment (``.env`` is loaded automatically from the
project root) and fall back to safe defaults, so the system boots even with
no credentials at all.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
MEMORY_DIR = PROJECT_ROOT / "memory"

_PLACEHOLDERS = {
    "",
    "your_gemini_api_key",
    "your_newsapi_key",
    "your_key_here",
    "changeme",
    "none",
}


def _load_env() -> None:
    """Load ``.env`` from the project root, wherever the process was started.

    Setting ``JARVIS_SKIP_DOTENV=1`` disables this entirely — the test suite
    relies on it so a developer's real credentials can never bleed into a run.
    """
    if os.getenv("JARVIS_SKIP_DOTENV", "").strip().lower() in {"1", "true", "yes"}:
        return
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
    else:  # fall back to any .env discoverable from the CWD
        load_dotenv(override=False)


def _str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _is_real_secret(value: str | None) -> bool:
    """A key counts as configured only if it is present and not a placeholder."""
    return bool(value) and value.strip().lower() not in _PLACEHOLDERS


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the runtime configuration."""

    # --- credentials -------------------------------------------------------
    gemini_api_key: str = ""
    news_api_key: str = ""

    # --- models ------------------------------------------------------------
    gemini_model: str = "gemini-flash-latest"
    hf_classifier_model: str = "facebook/bart-large-mnli"
    hf_summarizer_model: str = "facebook/bart-large-cnn"
    hf_embedding_model: str = "all-MiniLM-L6-v2"

    # --- routing flags -----------------------------------------------------
    hf_ranking: bool = True
    hf_summarizer: bool = False
    preload_models: bool = True

    # --- retrieval tuning --------------------------------------------------
    max_news_queries: int = 3
    max_paper_queries: int = 2
    news_page_size: int = 10
    papers_per_query: int = 4
    dedup_threshold: float = 0.85
    max_ranked_articles: int = 8
    request_timeout: int = 15

    # --- LLM resilience ----------------------------------------------------
    llm_max_retries: int = 2
    llm_retry_backoff: float = 1.5

    # --- server ------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: tuple[str, ...] = ("http://localhost:5173", "http://127.0.0.1:5173")
    log_level: str = "INFO"

    # --- paths -------------------------------------------------------------
    project_root: Path = field(default=PROJECT_ROOT)
    memory_dir: Path = field(default=MEMORY_DIR)

    @property
    def gemini_enabled(self) -> bool:
        """True when a usable Gemini key is configured."""
        return _is_real_secret(self.gemini_api_key)

    @property
    def news_api_enabled(self) -> bool:
        """True when a usable NewsAPI key is configured."""
        return _is_real_secret(self.news_api_key)

    @property
    def offline_mode(self) -> bool:
        """True when no external API is reachable and JARVIS runs fully local."""
        return not self.gemini_enabled and not self.news_api_enabled

    def masked_gemini_key(self) -> str:
        return _mask(self.gemini_api_key)

    def masked_news_key(self) -> str:
        return _mask(self.news_api_key)


def _mask(secret: str) -> str:
    """Render a credential safe for logs: ``AIzaSy...Xe9g``."""
    if not _is_real_secret(secret):
        return "not-configured"
    if len(secret) <= 10:
        return "*" * len(secret)
    return f"{secret[:6]}...{secret[-4:]}"


def _build_settings() -> Settings:
    _load_env()
    origins = tuple(
        origin.strip()
        for origin in _str("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
        if origin.strip()
    )
    return Settings(
        gemini_api_key=_str("GEMINI_API_KEY"),
        news_api_key=_str("NEWS_API_KEY"),
        gemini_model=_str("GEMINI_MODEL", "gemini-flash-latest"),
        hf_classifier_model=_str("HF_CLASSIFIER_MODEL", "facebook/bart-large-mnli"),
        hf_summarizer_model=_str("HF_SUMMARIZER_MODEL", "facebook/bart-large-cnn"),
        hf_embedding_model=_str("HF_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        hf_ranking=_bool("HF_RANKING", True),
        hf_summarizer=_bool("HF_SUMMARIZER", False),
        preload_models=_bool("PRELOAD_MODELS", True),
        max_news_queries=_int("MAX_NEWS_QUERIES", 3),
        max_paper_queries=_int("MAX_PAPER_QUERIES", 2),
        news_page_size=_int("NEWS_PAGE_SIZE", 10),
        papers_per_query=_int("PAPERS_PER_QUERY", 4),
        dedup_threshold=_float("DEDUP_THRESHOLD", 0.85),
        max_ranked_articles=_int("MAX_RANKED_ARTICLES", 8),
        request_timeout=_int("REQUEST_TIMEOUT", 15),
        llm_max_retries=_int("LLM_MAX_RETRIES", 2),
        llm_retry_backoff=_float("LLM_RETRY_BACKOFF", 1.5),
        api_host=_str("API_HOST", "0.0.0.0"),
        api_port=_int("API_PORT", 8000),
        cors_origins=origins or ("http://localhost:5173",),
        log_level=_str("LOG_LEVEL", "INFO").upper(),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return _build_settings()


def reload_settings() -> Settings:
    """Rebuild settings from the environment — used by tests."""
    get_settings.cache_clear()
    return get_settings()


_LOGGING_CONFIGURED = False


def configure_logging(level: str | None = None) -> None:
    """Install a single consistent log format for every entry point."""
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return
    logging.basicConfig(
        level=level or get_settings().log_level,
        format="%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Third-party libraries are noisy at INFO; keep the JARVIS log readable.
    for noisy in ("httpx", "urllib3", "arxiv", "sentence_transformers", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _LOGGING_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger — the only logging entry point agents use."""
    configure_logging()
    return logging.getLogger(name)
