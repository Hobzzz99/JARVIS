# Architecture

This document explains *why* JARVIS is built the way it is. For setup and
usage, see the [README](../README.md).

---

## 1. The core constraint: no external dependency may be load-bearing

JARVIS runs on free tiers. Free tiers rate-limit, expire, and get retired —
during development, Google retired `gemini-2.0-flash` mid-project. A system
whose only output path runs through one API is a system that produces nothing
the day that API changes.

So the design rule is: **every stage degrades instead of failing.**

| Stage | Primary | Fallback | Last resort |
|---|---|---|---|
| Planning | Gemini | Preference-derived query plan | — |
| News retrieval | NewsAPI | Labelled sample articles | Empty list |
| Research retrieval | arXiv | — | Empty list |
| Deduplication | MiniLM embeddings | Exact title matching | — |
| Ranking | BART-MNLI zero-shot | Neutral scores, order preserved | — |
| Insight synthesis | Gemini | Static offline analysis | — |
| Summarisation | Gemini | Local BART-CNN | Template bullet list |
| Chat | Gemini | Rule-based responder | — |

A run with **zero credentials and zero network** still produces a briefing.
This is verified by tests, not just asserted — see
`test_summary_falls_back_to_a_template_when_every_model_fails`.

The tradeoff is honesty about quality: degraded output is labelled as such
(`Local BART Fallback`, `[SAMPLE]`, `insights_source: "offline"`) and the
dashboard shows a banner. A fallback that silently pretends to be the real
thing is worse than an error.

---

## 2. Why six agents instead of one prompt

A single "summarise today's AI news" prompt would be shorter. It would also be
worse in three specific ways:

**Retrieval quality is not a language problem.** Deciding *what to search for*
benefits from an LLM. Deciding *which of 29 retrieved articles are duplicates*
does not — that is an embedding-similarity problem, solved better and ~1000x
cheaper by a 22 MB local model than by tokens in a context window.

**Separation enables routing.** Because ranking is its own stage, it can run
on a local classifier while planning runs on a hosted LLM. One monolithic
prompt forces every sub-task onto the most expensive engine.

**Failure isolation.** When Gemini returns 404, only the three stages that
need it degrade. Retrieval, dedup and ranking are unaffected. In a monolith,
one API failure means no output at all.

The pipeline is linear (`planner → retrieval → ranking → research → summary →
delivery`) because the data genuinely flows one way. LangGraph is used for the
state machine, timing instrumentation and the ability to add conditional edges
later — not because the current graph needs cycles.

---

## 3. Why MCP rather than direct function calls

The agents could import `fetch_ai_news` directly — and internally, they do.
The MCP server in `jarvis_mcp/server.py` wraps the same functions in the
Model Context Protocol so that *external* clients (Claude Desktop, Claude
Code, any MCP host) can drive JARVIS's capabilities without importing its
code.

This is the meaningful distinction: MCP is an integration boundary, not an
internal abstraction. Tools are defined once in `TOOL_REGISTRY` — a single
mapping of name to `(callable, schema)` — so the protocol surface and the
Python surface cannot drift apart.

Tool handlers are synchronous and I/O-bound, so `call_tool` dispatches them
through `asyncio.to_thread` rather than blocking the protocol event loop.

---

## 4. Local models: what runs where, and why

| Task | Model | Size | Why local |
|---|---|---|---|
| Semantic dedup | `all-MiniLM-L6-v2` | ~22 MB | Runs on ~30 short strings per briefing. An API round trip per pair would dominate runtime. |
| Relevance ranking | `facebook/bart-large-mnli` | ~1.6 GB | Zero-shot NLI gives calibrated per-topic scores. Doing this by prompt costs tokens and returns less consistent numbers. |
| Summarisation fallback | `facebook/bart-large-cnn` | ~1.6 GB | The offline safety net. Only loads when Gemini is unavailable or explicitly bypassed. |

Models are lazily loaded behind a `threading.Lock` and cached for the process
lifetime. The lock matters: FastAPI's startup warm-up thread and an inbound
request thread can both reach a getter simultaneously, and loading BART twice
concurrently would double a 1.6 GB allocation.

Dedup computes the **full pairwise similarity matrix in one operation**
(`util.cos_sim(embeddings, embeddings)`) rather than looping per pair, and
ranking classifies the **whole batch in a single forward pass** — roughly
3-5x faster than per-article calls.

---

## 5. Persistence: why flat JSON is the right call here

The briefing archive holds tens of documents and is read by exactly one
process. A database would add an operational dependency to a project whose
selling point is that it runs with no infrastructure.

What flat JSON still has to get right:

- **Atomic writes.** Every write goes to a `NamedTemporaryFile` in the same
  directory, is `fsync`ed, then `os.replace`d into position. A crash mid-write
  leaves the previous version intact, never a truncated file.
- **Self-healing reads.** A corrupted or missing store is logged and
  reinitialised with defaults rather than raising into an agent.
- **Serialised mutations.** A module-level lock wraps read-modify-write cycles,
  because FastAPI serves requests from a thread pool.
- **Location independence.** Paths resolve from the project root via
  `config.PROJECT_ROOT`, not the working directory — otherwise `uvicorn`,
  `python -m jarvis_mcp.server` and `pytest` would each see a different
  `memory/`.

If the archive grew into the thousands, the migration path is a vector store
(embeddings already exist for dedup) so the planner could retrieve
*semantically related* past briefings instead of just the last three.

---

## 6. Configuration and secret handling

All configuration resolves through one frozen `Settings` dataclass built by
`config.get_settings()`, cached with `lru_cache`. No module calls `os.getenv`
at point of use.

Two details worth noting:

**Placeholders are not credentials.** A `.env` containing
`GEMINI_API_KEY=your_gemini_api_key` is the most common misconfiguration.
`_is_real_secret()` treats known placeholder strings as absent, so the system
reports "offline" and degrades cleanly instead of failing an API call with a
confusing 400.

**Secrets never reach logs.** `masked_gemini_key()` renders
`AIzaSy...Xe9g`; a test asserts the raw value never appears in the masked
output. The NewsAPI key is sent as an `X-Api-Key` header rather than a query
parameter, because query strings land in proxy and server access logs — this
is also covered by a test.

---

## 7. Observability

Every workflow node is wrapped by `_timed_node`, which records duration and
outcome into the shared state. That telemetry is returned by `POST /briefing`
and rendered in the dashboard as a per-agent breakdown.

This replaced a set of hardcoded `setTimeout` calls that animated fake
progress messages. Real timings are more useful and more honest — on a cold
start the summary node visibly dominates because it is downloading BART, which
is exactly the kind of thing you want a dashboard to reveal.

---

## 8. Known limitations

- **Cold start is slow.** The first briefing that needs local models downloads
  1-2 GB. `PRELOAD_MODELS=true` warms them in a background thread at API
  startup to move that cost off the first request.
- **Single-turn chat.** `conversation_memory.json` exists but `POST /chat`
  does not yet thread history through it.
- **Category filters are keyword-based.** The sidebar filters client-side on
  substring matches, not embeddings. Honest but crude.
- **No authentication.** The API is designed for localhost. Exposing it
  publicly would require auth and per-client rate limiting.
- **NewsAPI free tier** allows 100 requests/day and returns articles up to a
  month old.
