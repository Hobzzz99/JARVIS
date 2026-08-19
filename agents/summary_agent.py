"""Summary Agent — turns the enriched feed into the final briefing text.

Three generation strategies, tried in order of quality:

1. **Gemini** (default) — structured, opinionated briefing.
2. **Local BART** — used when ``HF_SUMMARIZER=true`` or Gemini fails.
3. **Template** — deterministic bullet list; the last line of defence so a
   briefing is always produced.
"""

from __future__ import annotations

from typing import Any

from config import get_logger, get_settings
from llm.gemini_client import gemini_complete
from llm.hf_client import hf_summarize

logger = get_logger("jarvis.agents.summary")

_MAX_ARTICLES = 5
_SNIPPET_CHARS = 120

PROMPT_TEMPLATE = """\
You are Jarvis, an AI intelligence assistant briefing a working AI engineer.

Today's top articles:
{articles}

Research insights:
{insights}

Generate a sharp daily briefing in exactly this format:

**[Bold headline summarising today's biggest AI development]**

• [Article 1 — one sentence]
• [Article 2 — one sentence]
• [Article 3 — one sentence]
• [Article 4 — one sentence]
• [Article 5 — one sentence]

**Why It Matters:**
[2 sentences for an AI engineer]

**Today's Signal:**
[1 sentence — the single most important trend right now]

Be sharp. Be technical. No generic filler.
"""


def _bullet_list(articles: list[dict[str, Any]]) -> str:
    return (
        "\n".join(
            f"• {a.get('title', '')} — {a.get('description', '')[:_SNIPPET_CHARS]}"
            for a in articles
        )
        or "• No articles retrieved today."
    )


def _flatten(articles: list[dict[str, Any]]) -> str:
    return " ".join(f"{a.get('title', '')}. {a.get('description', '')}" for a in articles)


def _local_summary(articles: list[dict[str, Any]], insights: str, label: str) -> str:
    """Summarise locally with BART, falling back to a plain template."""
    try:
        summary = hf_summarize(_flatten(articles))
        return f"**Daily AI Briefing ({label})**\n\n{summary}\n\n**Why It Matters:**\n{insights}"
    except Exception as exc:  # noqa: BLE001 - local model may be missing entirely
        logger.warning("Local summarisation failed (%s) — using template briefing", exc)
        return (
            f"**Daily AI Briefing (Template Fallback)**\n\n"
            f"{_bullet_list(articles)}\n\n**Why It Matters:**\n{insights}"
        )


def run_summary_agent(enriched: dict[str, Any]) -> str:
    """Generate the briefing body.

    Args:
        enriched: Output of the research agent.

    Returns:
        Markdown-flavoured briefing text.
    """
    settings = get_settings()
    articles = enriched.get("articles", [])[:_MAX_ARTICLES]
    insights = enriched.get("insights", "")

    if settings.hf_summarizer:
        logger.info("Generating briefing with local BART (HF_SUMMARIZER=true)")
        return _local_summary(articles, insights, "Local BART")

    logger.info("Generating briefing with Gemini")
    prompt = PROMPT_TEMPLATE.format(articles=_bullet_list(articles), insights=insights)
    try:
        return gemini_complete(prompt)
    except Exception as exc:  # noqa: BLE001 - fall through to the local path
        logger.warning("Gemini summarisation unavailable (%s) — falling back to local", exc)
        return _local_summary(articles, insights, "Local BART Fallback")
