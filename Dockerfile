# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# JARVIS backend — FastAPI + LangGraph + local HuggingFace models
#
# Note on size: torch and the transformer weights make this image large by
# nature. Weights are NOT baked in; they download on first use into the
# `hf-cache` volume declared in docker-compose.yml, so rebuilds stay fast.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/jarvis/.cache/huggingface

WORKDIR /app

# curl is used by the container healthcheck below.
RUN apt-get update \
    && apt-get install --no-install-recommends -y curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first so application edits do not invalidate the layer cache.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py ./
COPY agents/ ./agents/
COPY api/ ./api/
COPY jarvis_mcp/ ./jarvis_mcp/
COPY llm/ ./llm/
COPY workflows/ ./workflows/

# Run as a non-root user; the model cache and memory dir must be writable.
RUN useradd --create-home --uid 1000 jarvis \
    && mkdir -p /app/memory /home/jarvis/.cache/huggingface \
    && chown -R jarvis:jarvis /app /home/jarvis
USER jarvis

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
