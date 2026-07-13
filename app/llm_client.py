"""Shared OpenAI client for structured and free-text completions."""

from __future__ import annotations

import time
from typing import Any

from openai import AsyncOpenAI, APITimeoutError, OpenAIError

from config import Settings, get_settings
from json_utils import parse_json_object
from logger import get_logger
from prompts import HIPPO_SYSTEM


logger = get_logger(__name__)


class LLMError(Exception):
    """Raised when an LLM request fails."""


class LLMClient:
    """Thin wrapper around the OpenAI chat completions API."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the LLM client with application settings."""
        self._settings = settings or get_settings()
        self._client = AsyncOpenAI(api_key=self._settings.openai_api_key)

    async def complete_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 256,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Request structured JSON output from the language model.

        Args:
            prompt: User prompt content.
            system: Optional system prompt override.

        Returns:
            Parsed JSON object from the model response.

        Raises:
            LLMError: When the request fails or the response is not valid JSON.
        """
        system_content = system or HIPPO_SYSTEM
        started = time.perf_counter()
        resolved_model = model or self._settings.openai_fast_model
        try:
            response = await self._client.chat.completions.create(
                model=resolved_model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=max_tokens,
                timeout=self._settings.llm_timeout_seconds,
            )
        except APITimeoutError as exc:
            logger.error(
                "LLM JSON request timed out.",
                error_type="APITimeoutError",
                recovery_action="raise LLMError",
                exc=exc,
            )
            raise LLMError("Request timed out.") from exc
        except OpenAIError as exc:
            logger.error(
                "LLM JSON request failed.",
                error_type=type(exc).__name__,
                recovery_action="raise LLMError",
                exc=exc,
            )
            raise LLMError("Request failed.") from exc

        elapsed_ms = (time.perf_counter() - started) * 1000
        usage = response.usage
        logger.log_event(
            "llm_json_completed",
            latency_ms=round(elapsed_ms, 2),
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
        )

        content = response.choices[0].message.content
        if not content:
            raise LLMError("Empty response from language model.")

        try:
            return parse_json_object(content)
        except ValueError as exc:
            logger.error(
                "Invalid JSON from language model.",
                error_type="JSONDecodeError",
                recovery_action="raise LLMError",
                exc=exc,
            )
            raise LLMError("Invalid JSON from language model.") from exc

    async def complete_text(
        self,
        prompt: str,
        *,
        temperature: float = 0.4,
        max_tokens: int = 120,
        model: str | None = None,
    ) -> str:
        """Generate a natural-language response from the language model.

        Args:
            prompt: User prompt content.
            temperature: Sampling temperature for the completion.

        Returns:
            Trimmed natural-language response text.

        Raises:
            LLMError: When the request fails or returns empty content.
        """
        started = time.perf_counter()
        resolved_model = model or self._settings.openai_fast_model
        try:
            response = await self._client.chat.completions.create(
                model=resolved_model,
                messages=[
                    {"role": "system", "content": HIPPO_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=self._settings.llm_timeout_seconds,
            )
        except APITimeoutError as exc:
            logger.error(
                "LLM text request timed out.",
                error_type="APITimeoutError",
                recovery_action="raise LLMError",
                exc=exc,
            )
            raise LLMError("Request timed out.") from exc
        except OpenAIError as exc:
            logger.error(
                "LLM text request failed.",
                error_type=type(exc).__name__,
                recovery_action="raise LLMError",
                exc=exc,
            )
            raise LLMError("Request failed.") from exc

        elapsed_ms = (time.perf_counter() - started) * 1000
        usage = response.usage
        logger.log_event(
            "llm_text_completed",
            latency_ms=round(elapsed_ms, 2),
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
        )

        content = response.choices[0].message.content
        if not content:
            raise LLMError("Empty response from language model.")

        return content.strip()
