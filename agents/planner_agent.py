"""Planner Agent — decides *what* today's briefing should cover.

Reads operator preferences plus the recent briefing archive and asks Gemini to
emit a search plan that deliberately avoids repeating recent coverage. If
Gemini is unavailable or returns malformed JSON, a deterministic plan derived
from stored preferences takes over so the pipeline never stalls.
"""

from __future__ import annotations

import json
import re
from typing import Any

from config import get_logger
from jarvis_mcp.tools.memory_tool import get_recent_briefings, load_preferences
from llm.gemini_client import gemini_complete

logger = get_logger("jarvis.agents.planner")

PLAN_KEYS = ("news_queries", "paper_queries", "focus_topics")

PROMPT_TEMPLATE = """\
You are the planning agent for an AI intelligence briefing system.

Operator interests: {interests}
Preferred sources: {sources}

Recent briefings (do NOT repeat these angles):
{recent}

Choose search queries that surface *new* developments relevant to the operator.

CRITICAL — query format. `news_queries` are sent to a keyword search engine
that requires EVERY term to appear in an article. Long descriptive phrases
match nothing.

- Each news query: 1-3 words maximum. Keywords, not sentences.
- Good: "AI agents", "Gemini", "open source LLM", "AI regulation"
- Bad:  "Google DeepMind agentic AI coding frameworks announcement"
- Do not append words like "news", "announcement", "developments" or "latest".

`paper_queries` search arXiv and may be slightly longer (2-5 words), but should
still read as topic keywords rather than questions.

`focus_topics` are classification labels for ranking — short noun phrases.

Return ONLY a valid JSON object, no markdown fences and no commentary:
{{
  "news_queries": ["query1", "query2", "query3"],
  "paper_queries": ["query1", "query2"],
  "focus_topics": ["topic1", "topic2", "topic3"]
}}
"""

# NewsAPI ANDs every term, so recall collapses past a few words. Anything the
# model returns above this is trimmed to its leading terms before searching.
MAX_NEWS_QUERY_WORDS = 4


def _extract_json(raw: str) -> dict[str, Any]:
    """Pull the first JSON object out of an LLM response.

    Models wrap JSON in prose or ```json fences even when told not to, so the
    outermost brace pair is extracted before parsing.
    """
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in response")
    return json.loads(cleaned[start : end + 1])


def _fallback_plan(preferences: dict[str, Any]) -> dict[str, list[str]]:
    """Deterministic plan built straight from stored preferences."""
    interests = preferences.get("interests") or ["artificial intelligence"]
    return {
        "news_queries": interests[:3],
        "paper_queries": ["LLM agents", "agentic AI systems"],
        "focus_topics": interests,
    }


def _trim_query(query: str, max_words: int) -> str:
    """Keep only the leading terms of an over-long search query.

    Models routinely ignore the "1-3 words" instruction and return a sentence.
    Truncating is better than searching verbatim, because a keyword engine that
    ANDs every term returns nothing for a seven-word phrase.
    """
    words = query.split()
    return " ".join(words[:max_words]) if len(words) > max_words else query


def _validate(plan: Any, preferences: dict[str, Any]) -> dict[str, list[str]]:
    """Coerce an LLM plan into the expected shape, filling gaps from preferences."""
    fallback = _fallback_plan(preferences)
    if not isinstance(plan, dict):
        return fallback
    validated: dict[str, list[str]] = {}
    for key in PLAN_KEYS:
        value = plan.get(key)
        items = (
            [str(item).strip() for item in value if str(item).strip()]
            if isinstance(value, list)
            else []
        )
        if key == "news_queries":
            items = [_trim_query(item, MAX_NEWS_QUERY_WORDS) for item in items]
        validated[key] = items or fallback[key]
    return validated


def run_planner_agent() -> dict[str, list[str]]:
    """Produce the search plan for today's briefing.

    Returns:
        Dict with ``news_queries``, ``paper_queries`` and ``focus_topics``.
    """
    logger.info("Planning today's briefing")
    preferences = load_preferences()
    recent = get_recent_briefings(n=3)
    recent_text = "\n".join(entry.get("briefing", "")[:200] for entry in recent) or "None yet."

    prompt = PROMPT_TEMPLATE.format(
        interests=json.dumps(preferences.get("interests", [])),
        sources=json.dumps(preferences.get("favorite_sources", [])),
        recent=recent_text,
    )

    try:
        plan = _validate(_extract_json(gemini_complete(prompt)), preferences)
        logger.info("Plan ready — focus topics: %s", ", ".join(plan["focus_topics"]))
        return plan
    except Exception as exc:  # noqa: BLE001 - any planner failure degrades gracefully
        logger.warning("Planner LLM unavailable (%s) — using preference-derived plan", exc)
        return _fallback_plan(preferences)
