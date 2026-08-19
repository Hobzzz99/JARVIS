"""Google Gemini client — the reasoning engine behind the planner,
research and summary agents.

Built on ``google-genai``, the current Google GenAI SDK. (The older
``google-generativeai`` package reached end of life and is deliberately not
used here.)

The wrapper adds the three things the raw SDK does not: explicit
"is this configured?" detection, bounded retries with exponential backoff for
transient 429/503 responses, and a single typed error the agents catch to fall
back to local models.
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Any

from config import get_logger, get_settings

logger = get_logger("jarvis.llm.gemini")

DEFAULT_SYSTEM_PROMPT = "You are Jarvis, an AI intelligence assistant."

# Substrings marking an error as worth retrying rather than failing over.
_TRANSIENT_MARKERS = (
    "429",
    "500",
    "503",
    "rate limit",
    "resource_exhausted",
    "quota",
    "deadline",
    "unavailable",
    "timeout",
)


class GeminiUnavailableError(RuntimeError):
    """Raised when Gemini cannot serve a request (no key, quota, network)."""


def check_gemini_key() -> bool:
    """Return True when a real (non-placeholder) Gemini API key is configured."""
    return get_settings().gemini_enabled


@lru_cache(maxsize=4)
def _client(api_key: str) -> Any:
    """Return a cached SDK client. Keyed by credential so rotation is picked up."""
    from google import genai

    return genai.Client(api_key=api_key)


def _is_transient(error: Exception) -> bool:
    message = str(error).lower()
    return any(marker in message for marker in _TRANSIENT_MARKERS)


def gemini_complete(
    prompt: str,
    system: str = DEFAULT_SYSTEM_PROMPT,
    model: str | None = None,
) -> str:
    """Send a single-turn completion request to Gemini.

    Args:
        prompt: The user-facing instruction.
        system: System instruction framing the model's role.
        model: Optional model override; defaults to ``GEMINI_MODEL``.

    Returns:
        The generated text, stripped.

    Raises:
        GeminiUnavailableError: No usable key, or every retry was exhausted.
    """
    settings = get_settings()
    if not settings.gemini_enabled:
        raise GeminiUnavailableError(
            "GEMINI_API_KEY is missing or still a placeholder. "
            "Add a real key to .env to enable cloud reasoning."
        )

    from google.genai import types

    model_name = model or settings.gemini_model
    last_error: Exception | None = None

    for attempt in range(settings.llm_max_retries + 1):
        try:
            response = _client(settings.gemini_api_key).models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    # These are single-turn text completions — no tools are ever
                    # passed, so the SDK's automatic function-calling loop is
                    # pure overhead (and warns when left enabled).
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True,
                        maximum_remote_calls=None,
                    ),
                ),
            )
            text = (response.text or "").strip()
            if not text:
                raise GeminiUnavailableError("Gemini returned an empty response.")
            logger.debug("Gemini %s responded with %d chars", model_name, len(text))
            return text
        except Exception as exc:  # noqa: BLE001 - SDK raises a wide error surface
            last_error = exc
            if attempt < settings.llm_max_retries and _is_transient(exc):
                delay = settings.llm_retry_backoff ** (attempt + 1)
                logger.warning(
                    "Gemini call failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1,
                    settings.llm_max_retries + 1,
                    exc,
                    delay,
                )
                time.sleep(delay)
                continue
            break

    raise GeminiUnavailableError(f"Gemini request failed: {last_error}") from last_error
