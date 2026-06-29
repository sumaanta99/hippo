"""Tests for the OpenAI LLM client wrapper."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from openai import APITimeoutError

from llm_client import LLMClient, LLMError


@pytest.mark.asyncio
async def test_complete_json_success(test_settings) -> None:
    """Successful JSON completions should parse structured payloads."""
    llm = LLMClient(test_settings)
    mock_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"intent":"SAVE_MEMORY"}'))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    llm._client.chat.completions.create = AsyncMock(return_value=mock_response)
    payload = await llm.complete_json("classify this")
    assert payload["intent"] == "SAVE_MEMORY"


@pytest.mark.asyncio
async def test_complete_json_timeout(test_settings) -> None:
    """Timeouts should raise LLMError."""
    llm = LLMClient(test_settings)
    llm._client.chat.completions.create = AsyncMock(
        side_effect=APITimeoutError("timed out")
    )
    with pytest.raises(LLMError, match="timed out"):
        await llm.complete_json("classify this")


@pytest.mark.asyncio
async def test_complete_json_recovers_embedded_object(test_settings) -> None:
    """Malformed wrappers around JSON objects should still parse."""
    llm = LLMClient(test_settings)
    mock_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content='Here you go: {"intent":"QUERY_MEMORY"}')
            )
        ],
        usage=None,
    )
    llm._client.chat.completions.create = AsyncMock(return_value=mock_response)
    payload = await llm.complete_json("query this")
    assert payload["intent"] == "QUERY_MEMORY"


@pytest.mark.asyncio
async def test_complete_text_success(test_settings) -> None:
    """Text completions should return trimmed content."""
    llm = LLMClient(test_settings)
    mock_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="  Hello there.  "))],
        usage=None,
    )
    llm._client.chat.completions.create = AsyncMock(return_value=mock_response)
    text = await llm.complete_text("say hello")
    assert text == "Hello there."
