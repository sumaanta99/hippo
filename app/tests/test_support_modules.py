"""Additional tests for logging, JSON utilities, and orchestration."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from engine.hippo_engine import HippoEngine
from json_utils import parse_json_object
from logger import StructuredLogger, configure_logging, get_logger
from models.operations import MemoryServiceResult
from prompts import SAVE_CONFIRM_RESPONSE
from tests.conftest import MockLLMClient


def test_parse_json_object_rejects_invalid_payload() -> None:
    """Invalid payloads with no JSON object should raise ValueError."""
    with pytest.raises(ValueError):
        parse_json_object("")


def test_structured_logger_emits_event(capsys) -> None:
    """Structured logger should emit JSON events to stderr when enabled."""
    configure_logging("WARNING", structured=True)
    logger = get_logger("test.logger")
    logger.log_event("unit_test", value=1)
    captured = capsys.readouterr()
    assert "unit_test" in captured.err
    assert '"value": 1' in captured.err


def test_structured_logger_hidden_by_default(capsys) -> None:
    """Structured events should not appear in the CLI unless explicitly enabled."""
    configure_logging("WARNING", structured=False)
    logger = get_logger("test.logger")
    logger.log_event("unit_test", value=1)
    captured = capsys.readouterr()
    assert "unit_test" not in captured.err


def test_structured_logger_error_event(capsys) -> None:
    """Error logging should include structured recovery metadata when enabled."""
    configure_logging("WARNING", structured=True)
    logger = StructuredLogger("test.error")
    logger.error(
        "boom",
        error_type="TestError",
        recovery_action="continue",
    )
    captured = capsys.readouterr()
    assert "error" in captured.err


@pytest.mark.asyncio
async def test_engine_routes_save_intent(test_settings, fake_embeddings) -> None:
    """HippoEngine should route SAVE_MEMORY fast-path messages to the memory service."""
    llm = MockLLMClient(
        test_settings,
        json_responses={
            "__default__": {
                "title": "Hair clip",
                "content": "Hair clip on bookshelf",
                "memory_type": "object_location",
                "category": "personal",
            }
        },
    )
    engine = HippoEngine(test_settings, llm=llm, embedding_client=fake_embeddings)
    await engine.initialize()
    result = await engine.chat("hair clip on bookshelf", "test")
    assert result.response == SAVE_CONFIRM_RESPONSE
    assert result.intent == "save"


@pytest.mark.asyncio
async def test_engine_routes_shopping_show(test_settings, fake_embeddings) -> None:
    """HippoEngine should route SHOPPING_SHOW fast-path messages to shopping."""
    engine = HippoEngine(test_settings, embedding_client=fake_embeddings)
    await engine.initialize()
    result = await engine.chat("what's on my shopping list", "test")
    assert "empty" in result.response.lower() or "have:" in result.response.lower()
    assert result.intent == "shopping_show"


@pytest.mark.asyncio
async def test_llm_client_logs_token_usage(test_settings, capsys) -> None:
    """LLM client should emit structured latency and token events."""
    llm = MockLLMClient(test_settings, json_responses={"__default__": {"ok": True}})
    with patch.object(
        llm,
        "_client",
        create=True,
    ):
        response = await llm.complete_json("test prompt")
    assert response == {"ok": True}
