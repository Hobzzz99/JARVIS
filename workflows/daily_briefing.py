"""LangGraph workflow — the daily briefing pipeline.

Six agents run as a linear ``StateGraph``:

    planner -> retrieval -> ranking -> research -> summary -> delivery

Every node is wrapped by :func:`_timed_node`, which records wall-clock duration
and node status into the shared state. That telemetry is surfaced by the API,
so the dashboard can show where a run actually spent its time instead of
replaying a canned animation.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from agents.delivery_agent import run_delivery_agent
from agents.planner_agent import run_planner_agent
from agents.ranking_agent import run_ranking_agent
from agents.research_agent import run_research_agent
from agents.retrieval_agent import run_retrieval_agent
from agents.summary_agent import run_summary_agent
from config import get_logger

logger = get_logger("jarvis.workflow")


class BriefingState(TypedDict):
    """Shared state threaded through every node of the graph."""

    plan: dict
    retrieved: dict
    ranked: dict
    enriched: dict
    briefing: str
    output: str
    telemetry: list[dict[str, Any]]
    echo: bool


def _initial_state(echo: bool) -> BriefingState:
    return BriefingState(
        plan={},
        retrieved={},
        ranked={},
        enriched={},
        briefing="",
        output="",
        telemetry=[],
        echo=echo,
    )


def _timed_node(
    name: str,
    handler: Callable[[BriefingState], BriefingState],
) -> Callable[[BriefingState], BriefingState]:
    """Wrap a node so its duration and outcome land in ``state["telemetry"]``."""

    def wrapper(state: BriefingState) -> BriefingState:
        started = time.perf_counter()
        status = "ok"
        try:
            state = handler(state)
        except Exception as exc:  # noqa: BLE001 - record then re-raise for the caller
            status = f"error: {exc}"
            raise
        finally:
            elapsed = round(time.perf_counter() - started, 3)
            state.setdefault("telemetry", []).append(
                {"node": name, "seconds": elapsed, "status": status}
            )
            logger.info("Node '%s' finished in %.2fs (%s)", name, elapsed, status)
        return state

    return wrapper


def planner_node(state: BriefingState) -> BriefingState:
    state["plan"] = run_planner_agent()
    return state


def retrieval_node(state: BriefingState) -> BriefingState:
    state["retrieved"] = run_retrieval_agent(state["plan"])
    return state


def ranking_node(state: BriefingState) -> BriefingState:
    state["ranked"] = run_ranking_agent(state["retrieved"], state["plan"].get("focus_topics", []))
    return state


def research_node(state: BriefingState) -> BriefingState:
    state["enriched"] = run_research_agent(state["ranked"])
    return state


def summary_node(state: BriefingState) -> BriefingState:
    state["briefing"] = run_summary_agent(state["enriched"])
    return state


def delivery_node(state: BriefingState) -> BriefingState:
    state["output"] = run_delivery_agent(state["briefing"], echo=state.get("echo", True))
    return state


NODES: tuple[tuple[str, Callable[[BriefingState], BriefingState]], ...] = (
    ("planner", planner_node),
    ("retrieval", retrieval_node),
    ("ranking", ranking_node),
    ("research", research_node),
    ("summary", summary_node),
    ("delivery", delivery_node),
)


def build_workflow():
    """Compile the six-node briefing StateGraph."""
    graph = StateGraph(BriefingState)
    for name, handler in NODES:
        graph.add_node(name, _timed_node(name, handler))

    graph.set_entry_point(NODES[0][0])
    for (current, _), (following, _) in zip(NODES, NODES[1:], strict=False):
        graph.add_edge(current, following)
    graph.add_edge(NODES[-1][0], END)

    return graph.compile()


def run_briefing_workflow(echo: bool = True) -> dict[str, Any]:
    """Execute the full pipeline once.

    Args:
        echo: Print the formatted briefing to stdout (CLI runs only).

    Returns:
        ``output``, ``articles``, ``papers``, ``insights``, ``focus_topics``,
        ``telemetry`` and total ``duration_seconds``.
    """
    started = time.perf_counter()
    logger.info("Daily briefing workflow starting")
    result = build_workflow().invoke(_initial_state(echo))
    duration = round(time.perf_counter() - started, 2)
    logger.info("Daily briefing workflow completed in %.2fs", duration)

    ranked = result.get("ranked", {})
    enriched = result.get("enriched", {})
    return {
        "output": result.get("output", ""),
        "articles": ranked.get("articles", []),
        "papers": ranked.get("papers", []),
        "insights": enriched.get("insights", ""),
        "insights_source": enriched.get("insights_source", "unknown"),
        "focus_topics": result.get("plan", {}).get("focus_topics", []),
        "telemetry": result.get("telemetry", []),
        "duration_seconds": duration,
    }


def run_daily_briefing(echo: bool = True) -> str:
    """Run the pipeline and return only the formatted briefing text."""
    return run_briefing_workflow(echo=echo)["output"]


def run_daily_briefing_data(echo: bool = False) -> dict[str, Any]:
    """Run the pipeline and return the full structured result (used by the API)."""
    return run_briefing_workflow(echo=echo)


if __name__ == "__main__":
    run_daily_briefing()
