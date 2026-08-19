"""NewsAPI tool: sample-data fallback, normalisation and failure handling."""

from __future__ import annotations

import pytest
import requests

import config
from jarvis_mcp.tools import news_tool


def _api_response(payload: dict, status: int = 200):
    class _Response:
        status_code = status

        @staticmethod
        def raise_for_status():
            if status >= 400:
                raise requests.HTTPError(f"HTTP {status}")

        @staticmethod
        def json():
            return payload

    return _Response()


def test_missing_key_returns_clearly_labelled_sample_data(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("NEWS_API_KEY", raising=False)
    config.reload_settings()

    articles = news_tool.fetch_ai_news()

    assert articles, "sample fallback should never be empty"
    assert all(article["is_sample"] is True for article in articles)
    assert all(article["title"].startswith("[SAMPLE]") for article in articles)


def test_live_response_is_normalised(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NEWS_API_KEY", "abcdef0123456789abcdef0123456789")
    config.reload_settings()
    monkeypatch.setattr(
        news_tool.requests,
        "get",
        lambda *a, **k: _api_response(
            {
                "status": "ok",
                "articles": [
                    {
                        "title": "Real headline",
                        "description": "Real description",
                        "url": "https://example.test/a",
                        "source": {"name": "Example Wire"},
                        "publishedAt": "2026-08-19T10:00:00Z",
                    }
                ],
            }
        ),
    )

    articles = news_tool.fetch_ai_news(query="agents")

    assert len(articles) == 1
    assert articles[0]["source"] == "Example Wire"
    assert articles[0]["is_sample"] is False


def test_removed_and_incomplete_articles_are_dropped(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NEWS_API_KEY", "abcdef0123456789abcdef0123456789")
    config.reload_settings()
    monkeypatch.setattr(
        news_tool.requests,
        "get",
        lambda *a, **k: _api_response(
            {
                "status": "ok",
                "articles": [
                    {"title": "[Removed]", "description": "gone", "source": {"name": "X"}},
                    {"title": "No description", "description": None, "source": {"name": "X"}},
                    {
                        "title": "Keeper",
                        "description": "Has both fields",
                        "url": "https://example.test/k",
                        "source": {"name": "X"},
                    },
                ],
            }
        ),
    )

    assert [a["title"] for a in news_tool.fetch_ai_news()] == ["Keeper"]


def test_network_failure_degrades_to_an_empty_list(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NEWS_API_KEY", "abcdef0123456789abcdef0123456789")
    config.reload_settings()

    def _boom(*_args, **_kwargs):
        raise requests.ConnectionError("network down")

    monkeypatch.setattr(news_tool.requests, "get", _boom)

    assert news_tool.fetch_ai_news() == []


def test_api_level_error_status_degrades_to_an_empty_list(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NEWS_API_KEY", "abcdef0123456789abcdef0123456789")
    config.reload_settings()
    monkeypatch.setattr(
        news_tool.requests,
        "get",
        lambda *a, **k: _api_response({"status": "error", "message": "rateLimited"}),
    )

    assert news_tool.fetch_ai_news() == []


def test_api_key_is_sent_as_a_header_not_in_the_url(monkeypatch: pytest.MonkeyPatch):
    """Keys in query strings leak into proxy and server access logs."""
    secret = "abcdef0123456789abcdef0123456789"
    monkeypatch.setenv("NEWS_API_KEY", secret)
    config.reload_settings()
    captured: dict = {}

    def _capture(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _api_response({"status": "ok", "articles": []})

    monkeypatch.setattr(news_tool.requests, "get", _capture)
    news_tool.fetch_ai_news()

    assert secret not in captured["url"]
    assert secret not in str(captured["params"])
    assert captured["headers"]["X-Api-Key"] == secret
