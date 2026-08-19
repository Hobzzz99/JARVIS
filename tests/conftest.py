"""Shared pytest fixtures.

Every test runs against an isolated temporary memory directory so the suite
never reads or writes the developer's real briefing archive.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import config


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Neutralise credentials and redirect memory to a temp directory."""
    # Never read the developer's real .env — tests define their own environment.
    monkeypatch.setenv("JARVIS_SKIP_DOTENV", "1")

    for name in (
        "GEMINI_API_KEY",
        "NEWS_API_KEY",
        "HF_RANKING",
        "HF_SUMMARIZER",
        "PRELOAD_MODELS",
        "CORS_ORIGINS",
    ):
        monkeypatch.delenv(name, raising=False)

    # Offline by default: tests must never make a real network call.
    monkeypatch.setenv("HF_RANKING", "false")
    monkeypatch.setenv("PRELOAD_MODELS", "false")
    monkeypatch.setattr(config, "MEMORY_DIR", tmp_path / "memory")
    monkeypatch.setattr(config.Settings, "memory_dir", tmp_path / "memory", raising=False)

    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


@pytest.fixture()
def memory_tool(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A freshly imported memory_tool bound to the temp memory directory."""
    from jarvis_mcp.tools import memory_tool as module

    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(module, "MEMORY_DIR", memory_dir)
    monkeypatch.setattr(module, "PREFERENCES_FILE", memory_dir / "user_preferences.json")
    monkeypatch.setattr(module, "HISTORY_FILE", memory_dir / "briefing_history.json")
    monkeypatch.setattr(module, "CONVERSATION_FILE", memory_dir / "conversation_memory.json")
    monkeypatch.setattr(module, "LOG_FILE", memory_dir / "workflow_logs.json")
    module.init_memory()
    return module


@pytest.fixture()
def reload_module():
    """Reimport a module so it picks up patched environment variables."""

    def _reload(name: str):
        return importlib.reload(importlib.import_module(name))

    return _reload
