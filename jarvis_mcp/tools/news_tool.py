"""NewsAPI MCP tool — live AI headlines.

Falls back to a small set of **explicitly labelled sample articles** when no
``NEWS_API_KEY`` is configured, so the pipeline is demonstrable end-to-end
without credentials. Sample items are marked ``is_sample: True`` and carry a
``[SAMPLE]`` source prefix — they are never presented as real reporting.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import requests

from config import get_logger, get_settings

logger = get_logger("jarvis.tools.news")

NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"

# Neutral, non-attributed placeholders used only in offline demo mode.
_SAMPLE_ARTICLES: list[dict[str, Any]] = [
    {
        "title": "[SAMPLE] Multi-agent orchestration frameworks converge on graph-based state",
        "description": (
            "Sample record used when NEWS_API_KEY is unset. Illustrates how the ranking "
            "and summarisation agents consume a retrieved article."
        ),
        "url": "https://newsapi.org/",
    },
    {
        "title": "[SAMPLE] Local-first inference narrows the gap with hosted models",
        "description": (
            "Sample record used when NEWS_API_KEY is unset. Demonstrates semantic "
            "deduplication against a near-duplicate headline."
        ),
        "url": "https://huggingface.co/models",
    },
    {
        "title": "[SAMPLE] On-device inference closes the distance to cloud-hosted models",
        "description": (
            "Sample record used when NEWS_API_KEY is unset. Deliberately near-identical "
            "to the previous item so the dedup step has something to remove."
        ),
        "url": "https://huggingface.co/models",
    },
    {
        "title": "[SAMPLE] Standardised tool protocols reshape how agents call external systems",
        "description": (
            "Sample record used when NEWS_API_KEY is unset. Exercises the zero-shot "
            "relevance classifier against protocol-oriented focus topics."
        ),
        "url": "https://modelcontextprotocol.io/",
    },
    {
        "title": "[SAMPLE] Retrieval pipelines shift from keyword search to embedding recall",
        "description": (
            "Sample record used when NEWS_API_KEY is unset. Provides a fifth item so the "
            "summary agent receives a full briefing-sized batch."
        ),
        "url": "https://arxiv.org/list/cs.AI/recent",
    },
]


def check_news_key() -> bool:
    """Return True when a real (non-placeholder) NewsAPI key is configured."""
    return get_settings().news_api_enabled


def _sample_articles() -> list[dict[str, Any]]:
    """Build the offline demo payload with a fresh timestamp."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return [
        {
            **article,
            "source": "[SAMPLE] Offline demo data",
            "published": now,
            "is_sample": True,
        }
        for article in _SAMPLE_ARTICLES
    ]


def _normalise(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Project a NewsAPI record onto the internal article schema."""
    title = raw.get("title") or ""
    description = raw.get("description") or ""
    if not title or not description or "[Removed]" in title:
        return None
    return {
        "title": title,
        "description": description,
        "url": raw.get("url", ""),
        "source": (raw.get("source") or {}).get("name", "Unknown"),
        "published": raw.get("publishedAt", ""),
        "is_sample": False,
    }


def fetch_ai_news(
    query: str = "artificial intelligence",
    page_size: int | None = None,
) -> list[dict[str, Any]]:
    """MCP Tool: fetch recent English-language articles matching ``query``.

    Args:
        query: Free-text NewsAPI search expression.
        page_size: Maximum articles to request (defaults to ``NEWS_PAGE_SIZE``).

    Returns:
        Normalised article dicts. Empty on API failure; sample data when the
        key is unset.
    """
    settings = get_settings()
    if not settings.news_api_enabled:
        logger.warning("NEWS_API_KEY not configured — returning labelled sample articles")
        return _sample_articles()

    params = {
        "q": query,
        "sortBy": "publishedAt",
        "pageSize": page_size or settings.news_page_size,
        "language": "en",
    }
    try:
        response = requests.get(
            NEWSAPI_ENDPOINT,
            params=params,
            headers={"X-Api-Key": settings.news_api_key},
            timeout=settings.request_timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        logger.error("NewsAPI request failed for query %r: %s", query, exc)
        return []
    except ValueError as exc:
        logger.error("NewsAPI returned invalid JSON for query %r: %s", query, exc)
        return []

    if payload.get("status") != "ok":
        logger.error("NewsAPI error for query %r: %s", query, payload.get("message"))
        return []

    articles = [item for raw in payload.get("articles", []) if (item := _normalise(raw))]
    logger.info("Retrieved %d article(s) for query %r", len(articles), query)
    return articles
