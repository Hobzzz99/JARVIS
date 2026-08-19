"""Ranking Agent — scores retrieved articles against the operator's focus topics.

Uses zero-shot natural-language inference (``facebook/bart-large-mnli``)
running locally, so relevance ranking costs nothing and needs no network. The
whole batch is classified in a single forward pass rather than one call per
article.
"""

from __future__ import annotations

from typing import Any

from config import get_logger, get_settings
from llm.hf_client import hf_classify_batch

logger = get_logger("jarvis.agents.ranking")

# Applied when the classifier is disabled or fails — keeps ordering stable.
NEUTRAL_SCORE = 0.5


def _apply_neutral_scores(articles: list[dict[str, Any]], focus_topics: list[str]) -> None:
    default_topic = focus_topics[0] if focus_topics else "AI"
    for article in articles:
        article.setdefault("relevance_score", NEUTRAL_SCORE)
        article.setdefault("top_topic", default_topic)


def run_ranking_agent(
    retrieved: dict[str, list[dict[str, Any]]],
    focus_topics: list[str],
) -> dict[str, Any]:
    """Rank articles by relevance and keep the strongest ones.

    Args:
        retrieved: Output of the retrieval agent.
        focus_topics: Candidate labels from the planner.

    Returns:
        ``{"articles": [...], "papers": [...]}`` with articles sorted by
        ``relevance_score`` descending and annotated with ``top_topic``.
    """
    settings = get_settings()
    articles = list(retrieved.get("articles", []))
    papers = list(retrieved.get("papers", []))

    if not (settings.hf_ranking and focus_topics and articles):
        logger.info(
            "Ranking skipped (hf_ranking=%s, %d articles)", settings.hf_ranking, len(articles)
        )
        _apply_neutral_scores(articles, focus_topics)
        return {"articles": articles[: settings.max_ranked_articles], "papers": papers}

    logger.info("Scoring %d articles against %d focus topics", len(articles), len(focus_topics))
    texts = [f"{a.get('title', '')}. {a.get('description', '')}" for a in articles]

    try:
        for article, scores in zip(articles, hf_classify_batch(texts, focus_topics), strict=True):
            if scores:
                article["relevance_score"] = round(max(scores.values()), 4)
                article["top_topic"] = max(scores, key=scores.get)
    except Exception as exc:  # noqa: BLE001 - model load or inference may fail
        logger.warning("Zero-shot ranking failed (%s) — falling back to neutral scores", exc)

    _apply_neutral_scores(articles, focus_topics)
    articles.sort(key=lambda a: a.get("relevance_score", 0.0), reverse=True)

    if articles:
        top = articles[0]
        logger.info(
            "Top article (%.0f%% / %s): %s",
            top["relevance_score"] * 100,
            top["top_topic"],
            top.get("title", "")[:70],
        )

    return {"articles": articles[: settings.max_ranked_articles], "papers": papers}
