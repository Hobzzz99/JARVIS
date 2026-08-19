"""arXiv MCP tool — latest AI research preprints.

arXiv requires no API key and no account, which makes it the zero-cost
research feed behind the briefing pipeline.
"""

from __future__ import annotations

from typing import Any

import arxiv

from config import get_logger, get_settings

logger = get_logger("jarvis.tools.arxiv")

# Abstracts run long; the research agent only needs the opening of each.
_SUMMARY_CHARS = 400
_MAX_AUTHORS = 3


def fetch_ai_papers(
    query: str = "LLM agents",
    max_results: int | None = None,
) -> list[dict[str, Any]]:
    """MCP Tool: fetch the most recently submitted papers matching ``query``.

    Args:
        query: arXiv search expression (e.g. ``"LLM agents"``).
        max_results: Papers to return (defaults to ``PAPERS_PER_QUERY``).

    Returns:
        Normalised paper dicts. Empty list on any transport or parse failure —
        a missing research feed degrades the briefing rather than aborting it.
    """
    limit = max_results or get_settings().papers_per_query
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=limit,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )
        papers = [
            {
                "title": result.title.strip(),
                "summary": result.summary.strip()[:_SUMMARY_CHARS],
                "url": result.entry_id,
                "authors": [author.name for author in result.authors[:_MAX_AUTHORS]],
                "published": str(result.published.date()),
            }
            for result in client.results(search)
        ]
    except Exception as exc:  # noqa: BLE001 - arxiv wraps many transport errors
        logger.error("arXiv request failed for query %r: %s", query, exc)
        return []

    logger.info("Retrieved %d paper(s) for query %r", len(papers), query)
    return papers
