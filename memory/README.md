# Memory stores

JARVIS persists state as four JSON documents in this directory. They are
**git-ignored** — each contains machine-local runtime data — and are created
automatically on first import of `jarvis_mcp.tools.memory_tool`.

| File | Shape | Purpose |
|---|---|---|
| `user_preferences.json` | object | Operator name, interests and favourite sources. Drives planning and ranking. |
| `briefing_history.json` | array | Every generated briefing, newest last. The planner reads the tail to avoid repeating topics. |
| `conversation_memory.json` | array | Reserved for multi-turn chat history. |
| `workflow_logs.json` | array | Timestamped workflow events, capped at the most recent 500 entries. |

## Guarantees

* **Atomic writes** — every write lands in a temp file and is `os.replace`d
  into position, so an interrupted write cannot truncate a store.
* **Self-healing** — a missing or corrupted file is logged and reinitialised
  with its default contents rather than raising.
* **Location independence** — paths resolve from the project root, so the API
  server, the CLI and the test suite all agree on where memory lives.

To reset an operator profile, delete `user_preferences.json` and restart; the
defaults in `memory_tool.DEFAULT_PREFERENCES` are written back.
