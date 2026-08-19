"""Persistent memory MCP tools.

JARVIS keeps four JSON stores under ``memory/``: operator preferences, the
briefing archive, conversation history and a workflow event log.

Two properties matter here and are easy to get wrong:

* **Location independence** — paths resolve from the project root, not the
  current working directory, so ``uvicorn`` and ``pytest`` see the same files.
* **Crash safety** — every write goes to a temp file and is atomically
  replaced, so an interrupted write can never leave truncated JSON behind.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Any

from config import get_logger, get_settings

logger = get_logger("jarvis.memory")

MEMORY_DIR: Path = get_settings().memory_dir

PREFERENCES_FILE = MEMORY_DIR / "user_preferences.json"
HISTORY_FILE = MEMORY_DIR / "briefing_history.json"
CONVERSATION_FILE = MEMORY_DIR / "conversation_memory.json"
LOG_FILE = MEMORY_DIR / "workflow_logs.json"

DEFAULT_PREFERENCES: dict[str, Any] = {
    "name": "Operator",
    "interests": ["AI Engineering", "MCP", "Coding Agents", "LLMs", "Agentic AI"],
    "favorite_sources": ["Google DeepMind", "Hugging Face", "OpenAI", "Anthropic"],
}

# Serialises read-modify-write cycles; FastAPI serves requests from a thread pool.
_write_lock = threading.Lock()


def _read_json(path: Path, default: Any) -> Any:
    """Read JSON, healing a missing or corrupted file by rewriting the default."""
    if not path.exists():
        _write_json(path, default)
        return default
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Memory file %s unreadable (%s) — reinitialising", path.name, exc)
        _write_json(path, default)
        return default


def _write_json(path: Path, payload: Any) -> None:
    """Atomically write ``payload`` as JSON to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # delete=False is required so the file survives close() and can be renamed;
    # the `with` block below still closes it, and the except clause cleans up.
    handle = tempfile.NamedTemporaryFile(  # noqa: SIM115
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    try:
        with handle as tmp:
            json.dump(payload, tmp, indent=2, ensure_ascii=False)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def init_memory() -> None:
    """Create any missing memory file with its default contents."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    for path, default in (
        (PREFERENCES_FILE, DEFAULT_PREFERENCES),
        (HISTORY_FILE, []),
        (CONVERSATION_FILE, []),
        (LOG_FILE, []),
    ):
        if not path.exists():
            _write_json(path, default)
            logger.info("Initialised memory store %s", path.name)


def load_preferences() -> dict[str, Any]:
    """MCP Tool: load operator preferences (name, interests, favourite sources)."""
    prefs = _read_json(PREFERENCES_FILE, DEFAULT_PREFERENCES)
    if not isinstance(prefs, dict):
        logger.warning("Preferences file had unexpected shape — using defaults")
        return dict(DEFAULT_PREFERENCES)
    # Merge over defaults so a partially-written file never breaks an agent.
    return {**DEFAULT_PREFERENCES, **prefs}


def save_preferences(preferences: dict[str, Any]) -> dict[str, Any]:
    """MCP Tool: persist operator preferences and return the stored document."""
    merged = {**DEFAULT_PREFERENCES, **preferences}
    with _write_lock:
        _write_json(PREFERENCES_FILE, merged)
    logger.info("Preferences updated for operator '%s'", merged.get("name"))
    return merged


def save_briefing(briefing: str, briefing_date: str | None = None) -> bool:
    """MCP Tool: append a briefing to the archive."""
    entry = {
        "date": briefing_date or str(date.today()),
        "briefing": briefing,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    with _write_lock:
        history = _read_json(HISTORY_FILE, [])
        if not isinstance(history, list):
            history = []
        history.append(entry)
        _write_json(HISTORY_FILE, history)
    logger.info("Briefing archived for %s (%d total)", entry["date"], len(history))
    return True


def get_recent_briefings(n: int = 3) -> list[dict[str, Any]]:
    """MCP Tool: return the last ``n`` briefings, oldest first."""
    history = _read_json(HISTORY_FILE, [])
    if not isinstance(history, list):
        return []
    return history[-n:] if n > 0 else []


def log_event(event: str) -> bool:
    """MCP Tool: append a timestamped entry to the workflow log."""
    with _write_lock:
        logs = _read_json(LOG_FILE, [])
        if not isinstance(logs, list):
            logs = []
        logs.append({"timestamp": datetime.now().isoformat(timespec="seconds"), "event": event})
        # Keep the log bounded so it never grows without limit.
        _write_json(LOG_FILE, logs[-500:])
    return True


def get_workflow_logs(n: int = 50) -> list[dict[str, Any]]:
    """MCP Tool: return the last ``n`` workflow log entries."""
    logs = _read_json(LOG_FILE, [])
    if not isinstance(logs, list):
        return []
    return logs[-n:] if n > 0 else []


init_memory()
