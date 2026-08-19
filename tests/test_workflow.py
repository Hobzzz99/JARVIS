"""End-to-end workflow wiring, with every agent stubbed."""

from __future__ import annotations

import pytest

from workflows import daily_briefing


@pytest.fixture()
def stub_agents(monkeypatch: pytest.MonkeyPatch):
    """Replace all six agents with deterministic doubles."""
    calls: list[str] = []

    def _record(name, value):
        def _handler(*_args, **_kwargs):
            calls.append(name)
            return value

        return _handler

    monkeypatch.setattr(
        daily_briefing,
        "run_planner_agent",
        _record("planner", {"news_queries": ["q"], "paper_queries": ["p"], "focus_topics": ["AI"]}),
    )
    monkeypatch.setattr(
        daily_briefing,
        "run_retrieval_agent",
        _record("retrieval", {"articles": [{"title": "A"}], "papers": [{"title": "P"}]}),
    )
    monkeypatch.setattr(
        daily_briefing,
        "run_ranking_agent",
        _record(
            "ranking",
            {"articles": [{"title": "A", "relevance_score": 0.9}], "papers": [{"title": "P"}]},
        ),
    )
    monkeypatch.setattr(
        daily_briefing,
        "run_research_agent",
        _record(
            "research",
            {
                "articles": [{"title": "A", "relevance_score": 0.9}],
                "papers": [{"title": "P"}],
                "insights": "Insight",
                "insights_source": "offline",
            },
        ),
    )
    monkeypatch.setattr(daily_briefing, "run_summary_agent", _record("summary", "Briefing body"))
    monkeypatch.setattr(daily_briefing, "run_delivery_agent", _record("delivery", "FORMATTED"))
    return calls


def test_agents_execute_in_pipeline_order(stub_agents):
    daily_briefing.run_briefing_workflow(echo=False)

    assert stub_agents == ["planner", "retrieval", "ranking", "research", "summary", "delivery"]


def test_workflow_returns_the_full_structured_result(stub_agents):
    result = daily_briefing.run_briefing_workflow(echo=False)

    assert result["output"] == "FORMATTED"
    assert result["articles"] == [{"title": "A", "relevance_score": 0.9}]
    assert result["papers"] == [{"title": "P"}]
    assert result["insights"] == "Insight"
    assert result["focus_topics"] == ["AI"]
    assert result["duration_seconds"] >= 0


def test_every_node_reports_timing_telemetry(stub_agents):
    telemetry = daily_briefing.run_briefing_workflow(echo=False)["telemetry"]

    assert [entry["node"] for entry in telemetry] == [
        "planner",
        "retrieval",
        "ranking",
        "research",
        "summary",
        "delivery",
    ]
    assert all(entry["status"] == "ok" for entry in telemetry)
    assert all(entry["seconds"] >= 0 for entry in telemetry)


def test_a_failing_node_propagates_instead_of_returning_a_partial_briefing(
    monkeypatch: pytest.MonkeyPatch, stub_agents
):
    def _boom(*_a, **_k):
        raise RuntimeError("retrieval exploded")

    monkeypatch.setattr(daily_briefing, "run_retrieval_agent", _boom)

    with pytest.raises(RuntimeError, match="retrieval exploded"):
        daily_briefing.run_briefing_workflow(echo=False)


def test_graph_compiles_with_all_six_nodes():
    assert len(daily_briefing.NODES) == 6
    assert daily_briefing.build_workflow() is not None
