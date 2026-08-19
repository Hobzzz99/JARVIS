"""HTTP contract tests driven through FastAPI's TestClient."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import config
from api.main import app
from llm.gemini_client import GeminiUnavailableError


@pytest.fixture()
def client(memory_tool):  # noqa: ARG001 - fixture redirects memory to tmp_path
    with TestClient(app) as test_client:
        yield test_client


def test_root_reports_capability_flags(client):
    body = client.get("/").json()

    assert body["status"] == "Jarvis online"
    assert body["gemini_enabled"] is False
    assert body["offline_mode"] is True


def test_health_probe(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_preferences_round_trip_through_the_api(client):
    payload = {
        "name": "Grace",
        "interests": ["compilers", "distributed systems"],
        "favorite_sources": ["ACM"],
    }
    put_response = client.put("/preferences", json=payload)

    assert put_response.status_code == 200
    assert put_response.json()["name"] == "Grace"
    assert client.get("/preferences").json()["interests"] == payload["interests"]


def test_preferences_rejects_an_invalid_payload(client):
    assert client.put("/preferences", json={"name": ""}).status_code == 422


def test_chat_requires_a_non_empty_message(client):
    assert client.post("/chat", json={"message": ""}).status_code == 422


def test_chat_uses_gemini_when_available(client, monkeypatch: pytest.MonkeyPatch):
    import api.main as api_main
    import llm.gemini_client as gemini

    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyRealLookingKeyValue123")
    config.reload_settings()
    monkeypatch.setattr(gemini, "gemini_complete", lambda *a, **k: "Certainly, Sir.")
    assert api_main  # imported for clarity; the route imports lazily

    body = client.post("/chat", json={"message": "status report"}).json()

    assert body["response"] == "Certainly, Sir."
    assert body["source"] == "gemini"


def test_chat_falls_back_locally_without_a_key(client, monkeypatch: pytest.MonkeyPatch):
    import llm.gemini_client as gemini

    def _offline(*_a, **_k):
        raise GeminiUnavailableError("GEMINI_API_KEY is missing")

    monkeypatch.setattr(gemini, "gemini_complete", _offline)

    body = client.post("/chat", json={"message": "what are my interests?"}).json()

    assert body["source"] == "fallback"
    assert "focus topics" in body["response"].lower()


def test_briefing_returns_the_structured_payload(client, monkeypatch: pytest.MonkeyPatch):
    import api.main as api_main

    monkeypatch.setattr(
        api_main,
        "run_daily_briefing_data",
        lambda: {
            "output": "BRIEFING",
            "articles": [{"title": "A"}],
            "papers": [{"title": "P"}],
            "insights": "Insight",
            "insights_source": "offline",
            "focus_topics": ["AI"],
            "telemetry": [{"node": "planner", "seconds": 0.1, "status": "ok"}],
            "duration_seconds": 1.23,
        },
    )

    body = client.post("/briefing").json()

    assert body["briefing"] == "BRIEFING"
    assert body["duration_seconds"] == 1.23
    assert body["telemetry"][0]["node"] == "planner"


def test_briefing_failure_returns_a_clean_500(client, monkeypatch: pytest.MonkeyPatch):
    import api.main as api_main

    def _boom():
        raise RuntimeError("pipeline down")

    monkeypatch.setattr(api_main, "run_daily_briefing_data", _boom)
    response = client.post("/briefing")

    assert response.status_code == 500
    assert "pipeline down" in response.json()["detail"]


def test_history_and_logs_are_bounded(client, memory_tool):
    memory_tool.save_briefing("archived", "2026-01-01")

    assert len(client.get("/history?limit=500").json()["history"]) <= 100
    assert client.get("/logs?limit=0").status_code == 200
