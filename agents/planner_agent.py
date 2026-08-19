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

Return ONLY a valid JSON object, no markdown fences and no commentary:
{{
  "news_queries": ["query1", "query2", "query3"],
  "paper_queries": ["query1", "query2"],
  "focus_topics": ["topic1", "topic2", "topic3"]
}}
"""


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
