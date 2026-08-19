"""Agent behaviour — especially the fallback paths that keep a run alive."""

from __future__ import annotations

import pytest

import config
from agents import planner_agent, ranking_agent, research_agent, retrieval_agent, summary_agent
from llm.gemini_client import GeminiUnavailableError

PREFERENCES = {
    "name": "Tester",
    "interests": ["Agentic AI", "MCP"],
    "favorite_sources": ["Hugging Face"],
}


# --------------------------------------------------------------------------- #
# Planner
# --------------------------------------------------------------------------- #
def test_planner_extracts_json_from_a_fenced_response(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(planner_agent, "load_preferences", lambda: PREFERENCES)
    monkeypatch.setattr(planner_agent, "get_recent_briefings", lambda n=3: [])
    monkeypatch.setattr(
        planner_agent,
        "gemini_complete",
        lambda *a, **k: (
            'Here you go:\n```json\n{"news_queries": ["a"], '
            '"paper_queries": ["b"], "focus_topics": ["c"]}\n```\nHope that helps!'
        ),
    )

    plan = planner_agent.run_planner_agent()

    assert plan == {"news_queries": ["a"], "paper_queries": ["b"], "focus_topics": ["c"]}


def test_planner_falls_back_to_preferences_when_the_llm_is_offline(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(planner_agent, "load_preferences", lambda: PREFERENCES)
    monkeypatch.setattr(planner_agent, "get_recent_briefings", lambda n=3: [])

    def _offline(*_a, **_k):
        raise GeminiUnavailableError("no key")

    monkeypatch.setattr(planner_agent, "gemini_complete", _offline)

    plan = planner_agent.run_planner_agent()

    assert plan["news_queries"] == PREFERENCES["interests"][:3]
    assert plan["focus_topics"] == PREFERENCES["interests"]


def test_planner_repairs_a_partially_valid_plan(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(planner_agent, "load_preferences", lambda: PREFERENCES)
    monkeypatch.setattr(planner_agent, "get_recent_briefings", lambda n=3: [])
    monkeypatch.setattr(
        planner_agent,
        "gemini_complete",
        lambda *a, **k: '{"news_queries": ["good"], "paper_queries": [], "focus_topics": null}',
    )

    plan = planner_agent.run_planner_agent()

    assert plan["news_queries"] == ["good"]
    assert plan["paper_queries"] == ["LLM agents", "agentic AI systems"]
    assert plan["focus_topics"] == PREFERENCES["interests"]


def test_planner_rejects_a_response_with_no_json():
    with pytest.raises(ValueError):
        planner_agent._extract_json("I am afraid I cannot do that, Dave.")


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
def test_retrieval_removes_exact_title_duplicates(monkeypatch: pytest.MonkeyPatch):
    config.reload_settings()
    articles = [
        {"title": "Same Story", "description": "one"},
        {"title": "same story", "description": "two"},
        {"title": "Different", "description": "three"},
    ]
    monkeypatch.setattr(retrieval_agent, "fetch_ai_news", lambda **_: articles)
    monkeypatch.setattr(retrieval_agent, "fetch_ai_papers", lambda **_: [])
    # Semantic dedup needs a 400MB model download; exercise the title pass alone.
    monkeypatch.setattr(
        retrieval_agent, "deduplicate_articles", lambda items, threshold=None: items
    )

    result = retrieval_agent.run_retrieval_agent({"news_queries": ["q"], "paper_queries": []})

    assert [a["title"] for a in result["articles"]] == ["Same Story", "Different"]


def test_retrieval_survives_a_failing_dedup_model(monkeypatch: pytest.MonkeyPatch):
    config.reload_settings()
    monkeypatch.setattr(
        retrieval_agent, "fetch_ai_news", lambda **_: [{"title": "A"}, {"title": "B"}]
    )
    monkeypatch.setattr(retrieval_agent, "fetch_ai_papers", lambda **_: [])

    def _model_missing(*_a, **_k):
        raise OSError("model weights unavailable")

    monkeypatch.setattr(retrieval_agent, "deduplicate_articles", _model_missing)

    result = retrieval_agent.run_retrieval_agent({"news_queries": ["q"]})

    assert len(result["articles"]) == 2


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #
def test_ranking_sorts_by_classifier_confidence(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HF_RANKING", "true")
    config.reload_settings()
    monkeypatch.setattr(
        ranking_agent,
        "hf_classify_batch",
        lambda texts, labels: [{"MCP": 0.2}, {"MCP": 0.9}, {"MCP": 0.5}],
    )

    retrieved = {"articles": [{"title": "low"}, {"title": "high"}, {"title": "mid"}], "papers": []}
    ranked = ranking_agent.run_ranking_agent(retrieved, ["MCP"])

    assert [a["title"] for a in ranked["articles"]] == ["high", "mid", "low"]
    assert ranked["articles"][0]["relevance_score"] == 0.9
    assert ranked["articles"][0]["top_topic"] == "MCP"


def test_ranking_assigns_neutral_scores_when_the_classifier_fails(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HF_RANKING", "true")
    config.reload_settings()

    def _boom(*_a, **_k):
        raise RuntimeError("model not loaded")

    monkeypatch.setattr(ranking_agent, "hf_classify_batch", _boom)

    ranked = ranking_agent.run_ranking_agent({"articles": [{"title": "x"}], "papers": []}, ["AI"])

    assert ranked["articles"][0]["relevance_score"] == ranking_agent.NEUTRAL_SCORE
    assert ranked["articles"][0]["top_topic"] == "AI"


def test_ranking_is_skipped_when_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HF_RANKING", "false")
    config.reload_settings()

    def _must_not_run(*_a, **_k):
        raise AssertionError("classifier should not be called when HF_RANKING=false")

    monkeypatch.setattr(ranking_agent, "hf_classify_batch", _must_not_run)

    ranked = ranking_agent.run_ranking_agent({"articles": [{"title": "x"}], "papers": []}, ["AI"])

    assert ranked["articles"][0]["relevance_score"] == ranking_agent.NEUTRAL_SCORE


def test_ranking_caps_the_article_count(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MAX_RANKED_ARTICLES", "3")
    monkeypatch.setenv("HF_RANKING", "false")
    config.reload_settings()

    retrieved = {"articles": [{"title": str(i)} for i in range(10)], "papers": []}

    assert len(ranking_agent.run_ranking_agent(retrieved, ["AI"])["articles"]) == 3


# --------------------------------------------------------------------------- #
# Research + Summary
# --------------------------------------------------------------------------- #
def test_research_marks_offline_insights_when_gemini_is_down(monkeypatch: pytest.MonkeyPatch):
    def _offline(*_a, **_k):
        raise GeminiUnavailableError("no key")

    monkeypatch.setattr(research_agent, "gemini_complete", _offline)

    enriched = research_agent.run_research_agent({"articles": [], "papers": []})

    assert enriched["insights_source"] == "offline"
    assert "Trend:" in enriched["insights"]


def test_summary_falls_back_to_a_template_when_every_model_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    config.reload_settings()

    def _offline(*_a, **_k):
        raise GeminiUnavailableError("no key")

    def _no_local_model(*_a, **_k):
        raise OSError("BART weights unavailable")

    monkeypatch.setattr(summary_agent, "gemini_complete", _offline)
    monkeypatch.setattr(summary_agent, "hf_summarize", _no_local_model)

    briefing = summary_agent.run_summary_agent(
        {"articles": [{"title": "Headline", "description": "Body"}], "insights": "Insight text"}
    )

    assert "Template Fallback" in briefing
    assert "Headline" in briefing
    assert "Insight text" in briefing


def test_summary_uses_local_bart_when_explicitly_routed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HF_SUMMARIZER", "true")
    config.reload_settings()

    def _must_not_run(*_a, **_k):
        raise AssertionError("Gemini must not be called when HF_SUMMARIZER=true")

    monkeypatch.setattr(summary_agent, "gemini_complete", _must_not_run)
    monkeypatch.setattr(summary_agent, "hf_summarize", lambda *_a, **_k: "local summary")

    briefing = summary_agent.run_summary_agent({"articles": [{"title": "T"}], "insights": ""})

    assert "local summary" in briefing
