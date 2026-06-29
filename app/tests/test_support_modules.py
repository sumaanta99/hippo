"""Additional tests for logging, JSON utilities, and orchestration."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from classifier import ClassificationResult
from config import Intent
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
async def test_engine_routes_save_intent(test_settings) -> None:
    """HippoEngine should route SAVE_MEMORY intents to the memory service."""
    engine = HippoEngine(test_settings)
    await engine.initialize()
    engine._classifier.classify_intent = AsyncMock(
        return_value=ClassificationResult(
            intent=Intent.SAVE_MEMORY,
            confidence=0.95,
            reasoning="save",
        )
    )
    engine._session("test").memory_service.save_memory = AsyncMock(
        return_value=MemoryServiceResult(text=SAVE_CONFIRM_RESPONSE)
    )
    result = await engine.chat("hair clip on bookshelf", "test")
    assert result.response == SAVE_CONFIRM_RESPONSE
    engine._session("test").memory_service.save_memory.assert_awaited_once()


@pytest.mark.asyncio
async def test_engine_routes_shopping_show(test_settings) -> None:
    """HippoEngine should route SHOPPING_SHOW intents to the shopping service."""
    from models.operations import ShoppingServiceResult

    engine = HippoEngine(test_settings)
    await engine.initialize()
    engine._classifier.classify_intent = AsyncMock(
        return_value=ClassificationResult(
            intent=Intent.SHOPPING_SHOW,
            confidence=0.95,
            reasoning="show",
        )
    )
    engine._session("test").shopping_service.show_list = AsyncMock(
        return_value=ShoppingServiceResult(text="You have: eggs.")
    )
    result = await engine.chat("what's on my list", "test")
    assert result.response == "You have: eggs."


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
