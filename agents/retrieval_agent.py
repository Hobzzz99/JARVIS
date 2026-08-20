"""Retrieval Agent — executes the planner's queries against the MCP tools.

Runs the news and arXiv fetches concurrently, then applies two-stage
deduplication: exact title matching first (cheap), semantic embedding
similarity second (catches the same story reported by different outlets).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from config import get_logger, get_settings
from jarvis_mcp.tools.arxiv_tool import fetch_ai_papers
from jarvis_mcp.tools.news_tool import fetch_ai_news
from llm.hf_client import deduplicate_articles

logger = get_logger("jarvis.agents.retrieval")

# Retrieval is network-bound, so a handful of parallel workers is plenty.
_MAX_WORKERS = 5


def _broaden(queries: list[str]) -> list[str]:
    """Derive short fallback queries from over-specific ones.

    Takes the first two words of each original query — enough to stay on topic
    while restoring recall — and always includes a catch-all so the retry
    cannot itself come back empty.
    """
    broadened = {" ".join(q.split()[:2]) for q in queries if q.split()}
    broadened.discard("")
    return [*sorted(broadened)[:2], "artificial intelligence"]


def _dedupe_by_title(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop exact title repeats while preserving order."""
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for article in articles:
        title = str(article.get("title", "")).strip().lower()
        if title and title not in seen:
            seen.add(title)
            unique.append(article)
    return unique


def run_retrieval_agent(plan: dict[str, list[str]]) -> dict[str, list[dict[str, Any]]]:
    """Fetch news and papers for every query in the plan.

    Args:
        plan: Output of the planner agent.

    Returns:
        ``{"articles": [...], "papers": [...]}`` capped at the configured limits.
    """
    settings = get_settings()
    news_queries = (plan.get("news_queries") or ["artificial intelligence"])[
        : settings.max_news_queries
    ]
    paper_queries = (plan.get("paper_queries") or ["LLM agents"])[: settings.max_paper_queries]

    logger.info(
        "Retrieving %d news queries and %d paper queries", len(news_queries), len(paper_queries)
    )

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        news_futures = [pool.submit(fetch_ai_news, query=q) for q in news_queries]
        paper_futures = [pool.submit(fetch_ai_papers, query=q) for q in paper_queries]
        articles = [item for future in news_futures for item in future.result()]
        papers = [item for future in paper_futures for item in future.result()]

    # A planner can produce queries so specific that every one returns nothing,
    # leaving the briefing with no news at all. Retry once with broad terms
    # rather than shipping an empty feed.
    if not articles:
        broad = _broaden(news_queries)
        logger.warning("All news queries returned nothing — retrying with %s", broad)
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            retries = [pool.submit(fetch_ai_news, query=q) for q in broad]
            articles = [item for future in retries for item in future.result()]

    raw_count = len(articles)
    articles = _dedupe_by_title(articles)

    try:
        articles = deduplicate_articles(articles, threshold=settings.dedup_threshold)
    except Exception as exc:  # noqa: BLE001 - embedding model may be unavailable
        logger.warning("Semantic dedup unavailable (%s) — keeping title-level dedup only", exc)

    papers = _dedupe_by_title(papers)
    logger.info(
        "Retrieved %d articles (%d after dedup) and %d papers",
        raw_count,
        len(articles),
        len(papers),
    )
    return {
        "articles": articles[: settings.max_ranked_articles + 2],
        "papers": papers[: settings.papers_per_query + 1],
    }
