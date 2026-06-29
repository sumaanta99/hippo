"""Edge-case and resilience tests."""

from __future__ import annotations

import pytest

from classifier import ClassificationError, IntentClassifier
from engine.hippo_engine import HippoEngine
from json_utils import parse_json_object
from llm_client import LLMError
from prompts import API_FAILURE_RESPONSE, INPUT_TOO_LONG_RESPONSE, UNKNOWN_RESPONSE
from tests.conftest import MockLLMClient


@pytest.mark.asyncio
async def test_empty_input_returns_empty_response(test_settings) -> None:
    """Empty input should return an empty ChatResponse without raising."""
    engine = HippoEngine(test_settings)
    await engine.initialize()
    result = await engine.chat("", "session-1")
    assert result.response == ""
    assert result.session_id == "session-1"


@pytest.mark.asyncio
async def test_input_too_long_rejected(test_settings) -> None:
    """Very long input should be rejected with a friendly message."""
    test_settings.max_input_length = 100
    engine = HippoEngine(test_settings)
    await engine.initialize()
    result = await engine.chat("a" * 101, "session-1")
    assert result.response == INPUT_TOO_LONG_RESPONSE


@pytest.mark.asyncio
async def test_classifier_api_failure_returns_fallback(test_settings) -> None:
    """Classifier API failures should return the API failure response."""

    class FailingLLM(MockLLMClient):
        async def complete_json(self, prompt: str, *, system: str | None = None) -> dict:
            raise LLMError("Request timed out.")

    engine = HippoEngine(test_settings)
    engine._classifier = IntentClassifier(FailingLLM(test_settings), test_settings)
    await engine.initialize()
    result = await engine.chat("buy eggs", "session-1")
    assert result.response == API_FAILURE_RESPONSE


@pytest.mark.asyncio
async def test_malformed_json_is_recovered() -> None:
    """Partial JSON payloads should be parsed safely."""
    payload = parse_json_object('Sure! {"intent": "SAVE_MEMORY", "confidence": 0.9}')
    assert payload["intent"] == "SAVE_MEMORY"
    assert payload["confidence"] == 0.9


def test_malformed_json_raises_cleanly() -> None:
    """Unrecoverable JSON should raise ValueError."""
    with pytest.raises(ValueError):
        parse_json_object("not json at all")


@pytest.mark.asyncio
async def test_unknown_without_matches_returns_rephrase(test_settings) -> None:
    """Unknown input with no memory match should ask the user to rephrase."""
    llm = MockLLMClient(
        test_settings,
        json_handler=lambda prompt, **_: {
            "intent": "UNKNOWN",
            "confidence": 0.2,
            "reasoning": "gibberish",
        },
    )
    engine = HippoEngine(test_settings)
    engine._classifier = IntentClassifier(llm, test_settings)
    await engine.initialize()
    result = await engine.chat("xyzabc-nomatch-12345", "session-1")
    assert result.response == UNKNOWN_RESPONSE


@pytest.mark.asyncio
async def test_duplicate_classification_is_idempotent(test_settings) -> None:
    """Repeated classification of the same message should be stable."""
    llm = MockLLMClient(
        test_settings,
        json_handler=lambda prompt, **_: {
            "intent": "SHOPPING_ADD",
            "confidence": 0.95,
            "reasoning": "buy request",
        },
    )
    classifier = IntentClassifier(llm, test_settings)
    first = await classifier.classify("buy eggs")
    second = await classifier.classify("buy eggs")
    assert first == second


@pytest.mark.asyncio
async def test_service_failure_returns_api_fallback(test_settings) -> None:
    """Service-level LLM failures should not crash the engine."""

    def handler(prompt: str, *, system: str | None = None) -> dict:
        if "Classify the user's message" in prompt:
            return {
                "intent": "SAVE_MEMORY",
                "confidence": 0.95,
                "reasoning": "save",
            }
        raise LLMError("Request failed.")

    llm = MockLLMClient(test_settings, json_handler=handler)
    engine = HippoEngine(test_settings)
    engine._classifier = IntentClassifier(llm, test_settings)
    services = engine._session("session-1")
    services.memory_service._llm = llm
    await engine.initialize()
    result = await engine.chat("hair clip on bookshelf", "session-1")
    assert result.response == API_FAILURE_RESPONSE
