"""Research Agent — synthesises cross-cutting insight from the ranked feed.

Where the summary agent reports *what happened*, this agent answers *what it
means*: the dominant trend across today's news and papers, why it matters to
an engineer building systems, and one concrete action.
"""

from __future__ import annotations

from typing import Any

from config import get_logger
from llm.gemini_client import gemini_complete

logger = get_logger("jarvis.agents.research")

_MAX_ARTICLES = 5
_MAX_PAPERS = 3
_SNIPPET_CHARS = 180

PROMPT_TEMPLATE = """\
You are a senior AI research analyst.

Today's top news:
{articles}

Today's research papers:
{papers}

Provide:
1. The single biggest trend across all of these (2 sentences)
2. Why it matters specifically for an AI engineer building systems (2 sentences)
3. One concrete takeaway or action (1 sentence)

Be direct and technical. No fluff.
"""

# Used when Gemini is offline — generic but honest, and clearly marked.
OFFLINE_INSIGHTS = (
    "**Trend:** Agentic orchestration is consolidating around graph-structured state "
    "machines while inference shifts toward local, task-specialised models.\n"
    "**Why it matters:** Engineers increasingly design hybrid pipelines that route cheap, "
    "high-volume work (classification, embedding, dedup) to local models and reserve "
    "hosted LLMs for reasoning, cutting both latency and spend.\n"
    "**Takeaway:** Identify the highest-volume LLM call in your pipeline and test whether "
    "a local zero-shot or embedding model can serve it.\n\n"
    "_(Generated offline — configure GEMINI_API_KEY for live analysis.)_"
)


def _format_articles(articles: list[dict[str, Any]]) -> str:
    return (
        "\n".join(
            f"- {a.get('title', '')} [{a.get('top_topic', 'AI')}]: "
            f"{a.get('description', '')[:_SNIPPET_CHARS]}"
            for a in articles
        )
        or "No articles today."
    )


def _format_papers(papers: list[dict[str, Any]]) -> str:
    return (
        "\n".join(
            f"- {p.get('title', '')}: {p.get('summary', '')[:_SNIPPET_CHARS]}" for p in papers
        )
        or "No papers today."
    )


def run_research_agent(ranked: dict[str, Any]) -> dict[str, Any]:
    """Attach an ``insights`` field to the ranked payload.

    Args:
        ranked: Output of the ranking agent.

    Returns:
        The same dict, plus ``insights`` (str) and ``insights_source``
        (``"gemini"`` or ``"offline"``).
    """
    logger.info("Synthesising research insights")
    prompt = PROMPT_TEMPLATE.format(
        articles=_format_articles(ranked.get("articles", [])[:_MAX_ARTICLES]),
        papers=_format_papers(ranked.get("papers", [])[:_MAX_PAPERS]),
    )

    try:
        ranked["insights"] = gemini_complete(prompt)
        ranked["insights_source"] = "gemini"
    except Exception as exc:  # noqa: BLE001 - degrade rather than abort the run
        logger.warning("Research enrichment unavailable (%s) — using offline insights", exc)
        ranked["insights"] = OFFLINE_INSIGHTS
        ranked["insights_source"] = "offline"

    return ranked
