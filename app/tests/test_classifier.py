"""Tests for intent classification."""

from __future__ import annotations

import pytest

from classifier import ClassificationError, IntentClassifier
from config import Intent
from llm_client import LLMError
from tests.conftest import MockLLMClient


INTENT_EXAMPLES: dict[Intent, list[str]] = {
    Intent.SAVE_MEMORY: [
        "hair clip on the bookshelf",
        "rent agreement in the cupboard",
        "gas agency number 98XXXXXXXX",
        "rajesh is our electrician",
    ],
    Intent.QUERY_MEMORY: [
        "where's my passport",
        "what's my gas number",
        "who's my plumber",
        "pm resources",
        "passport",
    ],
    Intent.UPDATE_MEMORY: [
        "passport is now in the locker",
        "changed my mind, it's in the drawer",
        "updated: hair clip is in the shelf",
    ],
    Intent.DELETE_MEMORY: [
        "forget passport",
        "delete hair clip",
        "remove eggs from my memory",
    ],
    Intent.SHOPPING_ADD: [
        "buy eggs",
        "need milk",
        "add detergent",
        "eggs, milk, bread",
    ],
    Intent.SHOPPING_REMOVE: [
        "remove eggs",
        "forget eggs from shopping",
        "no more milk",
    ],
    Intent.SHOPPING_SHOW: [
        "what's on my shopping list",
        "show my list",
        "shopping list?",
    ],
    Intent.GENERAL_CHAT: [
        "hey hippo, how are you",
        "good morning",
        "thanks",
        "lol",
    ],
    Intent.UNKNOWN: [
        "xyzabc",
        "asdfghjkl qwerty",
    ],
}


def _extract_user_message(prompt: str) -> str:
    """Extract the user message portion from a classification prompt."""
    if "<<<USER_CONTENT>>>" in prompt:
        after = prompt.split("<<<USER_CONTENT>>>", 1)[1]
        return after.split("<<<END_USER_CONTENT>>>", 1)[0].strip().lower()

    if "User message:" not in prompt:
        return prompt.strip().lower()
    after = prompt.split("User message:", 1)[1]
    return after.split("\n\n", 1)[0].strip().lower()


def _example_matches(message: str, example: str) -> bool:
    """Return True when a user message matches a classification example."""
    normalized_message = message.strip().lower()
    normalized_example = example.strip().lower()
    if normalized_example in normalized_message or normalized_message in normalized_example:
        return True
    message_tokens = set(normalized_message.split())
    example_tokens = set(normalized_example.split())
    if not message_tokens:
        return False
    overlap = message_tokens & example_tokens
    return len(overlap) >= len(message_tokens) - 1


def _classification_payload(intent: Intent, example: str) -> dict:
    """Build a canned classification payload."""
    confidence = 0.35 if intent == Intent.UNKNOWN else 0.92
    return {
        "intent": intent.value,
        "confidence": confidence,
        "reasoning": f"Matched example: {example}",
    }


def _json_handler(prompt: str, *, system: str | None = None) -> dict:
    """Return canned classification payloads based on prompt content."""
    message = _extract_user_message(prompt)
    ranked_examples: list[tuple[int, Intent, str]] = []
    for intent, examples in INTENT_EXAMPLES.items():
        for example in examples:
            ranked_examples.append((len(example), intent, example))

    ranked_examples.sort(key=lambda item: item[0], reverse=True)

    for _, intent, example in ranked_examples:
        example_lower = example.lower()
        if message == example_lower:
            return _classification_payload(intent, example)

    for _, intent, example in ranked_examples:
        example_lower = example.lower()
        if example_lower in message or message in example_lower:
            return _classification_payload(intent, example)

    for _, intent, example in ranked_examples:
        if _example_matches(message, example):
            return _classification_payload(intent, example)

    return {
        "intent": Intent.UNKNOWN.value,
        "confidence": 0.3,
        "reasoning": "No example matched.",
    }


@pytest.fixture
def classifier(test_settings) -> IntentClassifier:
    """Provide a classifier backed by a mock LLM."""
    llm = MockLLMClient(test_settings, json_handler=_json_handler)
    return IntentClassifier(llm, test_settings)


@pytest.mark.parametrize(
    ("message", "expected_intent"),
    [
        ("hair clip on bookshelf", Intent.SAVE_MEMORY),
        ("where's my passport", Intent.QUERY_MEMORY),
        ("passport is now in the locker", Intent.UPDATE_MEMORY),
        ("forget passport", Intent.DELETE_MEMORY),
        ("buy eggs", Intent.SHOPPING_ADD),
        ("remove eggs", Intent.SHOPPING_REMOVE),
        ("what's on my shopping list", Intent.SHOPPING_SHOW),
        ("hey hippo, how are you", Intent.GENERAL_CHAT),
        ("xyzabc", Intent.UNKNOWN),
    ],
)
@pytest.mark.asyncio
async def test_classify_core_intents(
    classifier: IntentClassifier,
    message: str,
    expected_intent: Intent,
) -> None:
    """Each primary intent should classify correctly with high confidence."""
    result = await classifier.classify(message)
    assert result["intent"] == expected_intent.value
    if expected_intent == Intent.UNKNOWN:
        assert result["confidence"] < 0.5
    else:
        assert result["confidence"] >= 0.8


@pytest.mark.parametrize("message", INTENT_EXAMPLES[Intent.SAVE_MEMORY])
@pytest.mark.asyncio
async def test_save_memory_examples(classifier: IntentClassifier, message: str) -> None:
    """SAVE_MEMORY examples should classify as save intents."""
    result = await classifier.classify(message)
    assert result["intent"] == Intent.SAVE_MEMORY.value


@pytest.mark.parametrize("message", INTENT_EXAMPLES[Intent.QUERY_MEMORY])
@pytest.mark.asyncio
async def test_query_memory_examples(classifier: IntentClassifier, message: str) -> None:
    """QUERY_MEMORY examples should classify as query intents."""
    result = await classifier.classify(message)
    assert result["intent"] == Intent.QUERY_MEMORY.value


@pytest.mark.asyncio
async def test_plural_pm_resources_is_query(classifier: IntentClassifier) -> None:
    """Plural resource queries should be treated as memory lookups."""
    result = await classifier.classify("pm resources")
    assert result["intent"] == Intent.QUERY_MEMORY.value


@pytest.mark.asyncio
async def test_empty_input_returns_unknown(test_settings) -> None:
    """Empty input should classify as UNKNOWN with zero confidence."""
    classifier = IntentClassifier(MockLLMClient(test_settings), test_settings)
    result = await classifier.classify_intent("", "test_user")
    assert result.intent == Intent.UNKNOWN
    assert result.confidence == 0.0


@pytest.mark.asyncio
async def test_ambiguous_input_low_confidence(test_settings) -> None:
    """Ambiguous input should map to UNKNOWN with low confidence."""
    classifier = IntentClassifier(MockLLMClient(test_settings), test_settings)
    result = await classifier.classify("xyzabc")
    assert result["intent"] == Intent.UNKNOWN.value
    assert result["confidence"] < 0.5


@pytest.mark.asyncio
async def test_classifier_is_idempotent(test_settings) -> None:
    """Repeated classification requests should return the same result."""
    classifier = IntentClassifier(MockLLMClient(test_settings), test_settings)
    first = await classifier.classify("buy eggs")
    second = await classifier.classify("buy eggs")
    assert first == second


@pytest.mark.asyncio
async def test_classifier_api_failure_raises(test_settings) -> None:
    """LLM failures should surface as ClassificationError."""

    class FailingLLM(MockLLMClient):
        async def complete_json(
            self,
            prompt: str,
            *,
            system: str | None = None,
            max_tokens: int = 256,
            model: str | None = None,
        ) -> dict:
            _ = max_tokens, model
            raise LLMError("Request timed out.")

    classifier = IntentClassifier(FailingLLM(test_settings), test_settings)
    with pytest.raises(ClassificationError):
        await classifier.classify_intent("xyzabc nonsense phrase here", "test_user")
