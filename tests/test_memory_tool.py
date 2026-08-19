"""Persistent memory tools: initialisation, round-tripping and self-healing."""

from __future__ import annotations

import json


def test_init_creates_every_store(memory_tool):
    for path in (
        memory_tool.PREFERENCES_FILE,
        memory_tool.HISTORY_FILE,
        memory_tool.CONVERSATION_FILE,
        memory_tool.LOG_FILE,
    ):
        assert path.exists(), f"{path.name} was not created"


def test_preferences_round_trip(memory_tool):
    memory_tool.save_preferences(
        {"name": "Ada", "interests": ["compilers"], "favorite_sources": ["ACM"]}
    )
    stored = memory_tool.load_preferences()

    assert stored["name"] == "Ada"
    assert stored["interests"] == ["compilers"]


def test_partial_preferences_are_merged_over_defaults(memory_tool):
    memory_tool.save_preferences({"name": "Ada"})
    stored = memory_tool.load_preferences()

    assert stored["name"] == "Ada"
    assert stored["interests"] == memory_tool.DEFAULT_PREFERENCES["interests"]


def test_briefings_append_in_order_and_read_back_newest_last(memory_tool):
    memory_tool.save_briefing("first", "2026-01-01")
    memory_tool.save_briefing("second", "2026-01-02")
    memory_tool.save_briefing("third", "2026-01-03")

    recent = memory_tool.get_recent_briefings(n=2)

    assert [entry["briefing"] for entry in recent] == ["second", "third"]
    assert recent[-1]["date"] == "2026-01-03"


def test_corrupted_json_is_healed_rather_than_raising(memory_tool):
    memory_tool.HISTORY_FILE.write_text("{ this is not valid json", encoding="utf-8")

    assert memory_tool.get_recent_briefings() == []
    assert json.loads(memory_tool.HISTORY_FILE.read_text(encoding="utf-8")) == []


def test_workflow_log_is_bounded(memory_tool):
    for index in range(520):
        memory_tool.log_event(f"event-{index}")

    logs = memory_tool.get_workflow_logs(n=1000)

    assert len(logs) == 500
    assert logs[-1]["event"] == "event-519"


def test_writes_are_atomic_and_leave_no_temp_files(memory_tool):
    memory_tool.save_briefing("only briefing", "2026-02-02")
    leftovers = [p.name for p in memory_tool.MEMORY_DIR.iterdir() if p.name.endswith(".tmp")]

    assert leftovers == []
