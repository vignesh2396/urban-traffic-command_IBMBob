"""
LLM Helpers — UrbanPulse
Thin retry wrapper around the google.genai SDK to handle 429 rate limits
with exponential back-off.
"""

from __future__ import annotations

import logging
import time

from google import genai
from google.genai import errors as genai_errors

from utils.config import settings

logger = logging.getLogger("urbanpulse")

_client = genai.Client(api_key=settings.GEMINI_API_KEY)


def generate_content(prompt: str, retries: int = 4, base_delay: float = 15.0) -> str:
    """
    Call Gemini generate_content with automatic retry on 429 RESOURCE_EXHAUSTED.

    Args:
        prompt:      The prompt string to send.
        retries:     Maximum number of retry attempts after the first failure.
        base_delay:  Initial wait in seconds; doubles each retry (15 → 30 → 60 → 120).

    Returns:
        The model's response text.

    Raises:
        The last exception if all retries are exhausted.
    """
    delay = base_delay
    last_exc: Exception | None = None

    for attempt in range(retries + 1):
        try:
            response = _client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=prompt,
            )
            return response.text
        except Exception as exc:
            last_exc = exc
            err_str = str(exc)
            is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str

            if is_rate_limit and attempt < retries:
                logger.warning(
                    "Gemini rate limit hit (attempt %d/%d). Retrying in %.0fs…",
                    attempt + 1,
                    retries + 1,
                    delay,
                )
                time.sleep(delay)
                delay *= 2
            else:
                raise

    raise last_exc  # type: ignore[misc]
