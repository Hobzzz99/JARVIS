"""Delivery Agent — formats, persists and emits the finished briefing.

The terminal banner uses box-drawing characters, which legacy Windows consoles
(cp1252) cannot encode; printing is guarded so a console encoding problem can
never fail a briefing that was otherwise generated successfully.
"""

from __future__ import annotations

from datetime import date

from config import get_logger
from jarvis_mcp.tools.memory_tool import load_preferences, log_event, save_briefing

logger = get_logger("jarvis.agents.delivery")

_RULE = "━" * 60

TEMPLATE = """\
{rule}
JARVIS — AI INTELLIGENCE BRIEF
{today} // Operator: {operator}
{rule}

{briefing}

{rule}
"""


def _safe_print(text: str) -> None:
    """Print ``text``, degrading to ASCII if the console cannot encode it."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


def run_delivery_agent(briefing: str, echo: bool = True) -> str:
    """Format the briefing, archive it, and log the delivery.

    Args:
        briefing: Body text from the summary agent.
        echo: Print the formatted briefing to stdout (CLI runs only).

    Returns:
        The fully formatted briefing.
    """
    operator = load_preferences().get("name", "Operator")
    today = str(date.today())

    output = TEMPLATE.format(rule=_RULE, today=today, operator=operator, briefing=briefing.strip())

    save_briefing(briefing, today)
    log_event(f"Daily briefing delivered for {today}")
    logger.info("Briefing delivered for %s (operator: %s)", today, operator)

    if echo:
        _safe_print(output)
    return output
