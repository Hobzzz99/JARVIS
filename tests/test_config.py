"""Configuration resolution and secret handling."""

from __future__ import annotations

import pytest

import config


def test_placeholder_keys_are_not_treated_as_configured(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GEMINI_API_KEY", "your_gemini_api_key")
    monkeypatch.setenv("NEWS_API_KEY", "")
    settings = config.reload_settings()

    assert settings.gemini_enabled is False
    assert settings.news_api_enabled is False
    assert settings.offline_mode is True


def test_real_keys_enable_engines(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyRealLookingKeyValue123")
    monkeypatch.setenv("NEWS_API_KEY", "abcdef0123456789abcdef0123456789")
    settings = config.reload_settings()

    assert settings.gemini_enabled is True
    assert settings.news_api_enabled is True
    assert settings.offline_mode is False


def test_masked_key_never_leaks_the_secret(monkeypatch: pytest.MonkeyPatch):
    secret = "AIzaSyVerySecretValue0987654321"
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    masked = config.reload_settings().masked_gemini_key()

    assert secret not in masked
    assert masked.startswith("AIzaSy")
    assert masked.endswith("4321")


def test_missing_key_masks_to_a_readable_label(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert config.reload_settings().masked_gemini_key() == "not-configured"


def test_cors_origins_are_parsed_from_a_comma_separated_list(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CORS_ORIGINS", "http://a.test , http://b.test")
    assert config.reload_settings().cors_origins == ("http://a.test", "http://b.test")


def test_malformed_numeric_env_falls_back_to_the_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MAX_NEWS_QUERIES", "not-a-number")
    assert config.reload_settings().max_news_queries == 3
