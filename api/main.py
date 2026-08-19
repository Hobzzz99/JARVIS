"""FastAPI application — HTTP surface for the JARVIS pipeline.

Endpoints
---------
``GET  /``             service metadata and capability flags
``GET  /health``       liveness probe
``POST /briefing``     run the full six-agent workflow
``GET  /history``      recent briefings from the archive
``GET  /logs``         recent workflow events
``GET  /preferences``  read operator preferences
``PUT  /preferences``  update operator preferences
``POST /chat``         single-turn conversation with the JARVIS persona
"""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import configure_logging, get_logger, get_settings
from jarvis_mcp.tools.memory_tool import (
    get_recent_briefings,
    get_workflow_logs,
    load_preferences,
    save_preferences,
)
from workflows.daily_briefing import run_daily_briefing_data

configure_logging()
logger = get_logger("jarvis.api")

CHAT_PERSONA = (
    "You are JARVIS, a highly advanced, witty AI assistant in the style of Tony Stark's "
    "assistant. Address the operator as 'Sir'. Be technically sharp and lightly dry. "
    "Keep responses under four sentences.\n"
    "Operator name: {name}. Operator interests: {interests}."
)


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class Preferences(BaseModel):
    """Operator profile that steers planning and ranking."""

    name: str = Field(default="Operator", min_length=1, max_length=80)
    interests: list[str] = Field(default_factory=list, max_length=25)
    favorite_sources: list[str] = Field(default_factory=list, max_length=25)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    response: str
    source: str = Field(description="'gemini' when the LLM answered, 'fallback' otherwise")


class BriefingResponse(BaseModel):
    briefing: str
    articles: list[dict[str, Any]]
    papers: list[dict[str, Any]]
    insights: str
    focus_topics: list[str]
    telemetry: list[dict[str, Any]]
    duration_seconds: float


class StatusResponse(BaseModel):
    status: str
    llm: str
    gemini_enabled: bool
    news_api_enabled: bool
    hf_ranking: bool
    hf_summarizer: bool
    offline_mode: bool


# --------------------------------------------------------------------------- #
# Lifespan
# --------------------------------------------------------------------------- #
def _preload_models() -> None:
    """Warm the local HuggingFace models so the first request is not slow."""
    try:
        from llm.hf_client import get_classifier, get_embedder

        get_embedder()
        get_classifier()
        logger.info("HuggingFace warm-up complete")
    except Exception as exc:  # noqa: BLE001 - warm-up is best-effort
        logger.warning("HuggingFace warm-up failed: %s", exc)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Report configuration on boot and warm models in the background."""
    settings = get_settings()
    logger.info("JARVIS API starting")
    logger.info("  Gemini  : %s (%s)", settings.masked_gemini_key(), settings.gemini_model)
    logger.info("  NewsAPI : %s", settings.masked_news_key())
    logger.info("  Local AI: ranking=%s summarizer=%s", settings.hf_ranking, settings.hf_summarizer)
    if settings.offline_mode:
        logger.warning("No API keys configured — running fully offline on local models")

    if settings.preload_models:
        threading.Thread(target=_preload_models, name="hf-warmup", daemon=True).start()

    yield
    logger.info("JARVIS API shutting down")


app = FastAPI(
    title="JARVIS AI Assistant",
    description="Multi-agent daily AI intelligence briefing system (LangGraph + MCP + Gemini).",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(get_settings().cors_origins),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "OPTIONS"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/", response_model=StatusResponse, tags=["system"])
def root() -> StatusResponse:
    """Service metadata — which engines are live and which are offline."""
    settings = get_settings()
    return StatusResponse(
        status="Jarvis online",
        llm=settings.gemini_model,
        gemini_enabled=settings.gemini_enabled,
        news_api_enabled=settings.news_api_enabled,
        hf_ranking=settings.hf_ranking,
        hf_summarizer=settings.hf_summarizer,
        offline_mode=settings.offline_mode,
    )


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    """Liveness probe for containers and uptime checks."""
    return {"status": "ok"}


@app.post("/briefing", response_model=BriefingResponse, tags=["briefing"])
def create_briefing() -> BriefingResponse:
    """Run the full six-agent workflow and return the briefing plus its sources."""
    try:
        data = run_daily_briefing_data()
    except Exception as exc:  # noqa: BLE001 - surface as a clean 500
        logger.exception("Briefing workflow failed")
        raise HTTPException(status_code=500, detail=f"Workflow failed: {exc}") from exc

    return BriefingResponse(
        briefing=data["output"],
        articles=data["articles"],
        papers=data["papers"],
        insights=data["insights"],
        focus_topics=data["focus_topics"],
        telemetry=data["telemetry"],
        duration_seconds=data["duration_seconds"],
    )


@app.get("/history", tags=["briefing"])
def history(limit: int = 10) -> dict[str, list[dict[str, Any]]]:
    """Return the most recent archived briefings."""
    return {"history": get_recent_briefings(n=max(1, min(limit, 100)))}


@app.get("/logs", tags=["system"])
def logs(limit: int = 50) -> dict[str, list[dict[str, Any]]]:
    """Return recent workflow events for observability."""
    return {"logs": get_workflow_logs(n=max(1, min(limit, 500)))}


@app.get("/preferences", response_model=Preferences, tags=["preferences"])
def read_preferences() -> Preferences:
    """Read the stored operator profile."""
    return Preferences(**load_preferences())


@app.put("/preferences", response_model=Preferences, tags=["preferences"])
def update_preferences(preferences: Preferences) -> Preferences:
    """Validate and persist the operator profile."""
    return Preferences(**save_preferences(preferences.model_dump()))


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat(request: ChatRequest) -> ChatResponse:
    """Single-turn chat with the JARVIS persona.

    Falls back to a local rule-based responder when Gemini is unavailable, so
    the dashboard's voice interface still works without any API key.
    """
    from llm.gemini_client import gemini_complete

    preferences = load_preferences()
    persona = CHAT_PERSONA.format(
        name=preferences.get("name", "Operator"),
        interests=", ".join(preferences.get("interests", [])) or "AI engineering",
    )

    try:
        return ChatResponse(
            response=gemini_complete(request.message, system=persona), source="gemini"
        )
    except Exception as exc:  # noqa: BLE001 - any LLM failure routes to the fallback
        logger.warning("Chat fell back to local responder: %s", exc)
        return ChatResponse(
            response=_fallback_reply(request.message, preferences, exc), source="fallback"
        )


def _fallback_reply(message: str, preferences: dict[str, Any], error: Exception) -> str:
    """Rule-based responder used when the LLM core is offline."""
    lowered = message.lower()
    name = preferences.get("name", "Operator")
    interests = ", ".join(preferences.get("interests", [])) or "AI engineering"

    if any(word in lowered for word in ("interest", "topic", "focus")):
        return f"My telemetry lists your focus topics as: {interests}, Sir."
    if "who am i" in lowered or "my name" in lowered:
        return f"You are {name}, Sir — operator of this unit."
    if any(lowered.startswith(greeting) for greeting in ("hello", "hi", "hey")):
        return (
            "Hello, Sir. I am running on local auxiliary power — my Gemini reasoning core "
            "is offline. How may I assist?"
        )

    detail = str(error).lower()
    if not get_settings().gemini_enabled:
        return (
            "Standing by, Sir. My primary reasoning core is offline: no GEMINI_API_KEY is "
            "configured. Add one to .env and restart the server."
        )
    if any(marker in detail for marker in ("quota", "429", "rate limit")):
        return (
            "Standing by, Sir. My Gemini core reports a rate limit or exhausted quota. "
            "Operating on fallback subroutines until it clears."
        )
    return (
        f"Standing by, Sir. My reasoning core hit a connection error ({str(error)[:80]}). "
        "Operating on auxiliary subroutines."
    )


if __name__ == "__main__":
    import uvicorn

    config = get_settings()
    uvicorn.run("api.main:app", host=config.api_host, port=config.api_port, reload=True)
