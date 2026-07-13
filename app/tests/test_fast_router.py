"""Tests for rule-based fast intent routing."""

from __future__ import annotations

import pytest

from config import Intent
from fast_router import try_fast_classify


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("where's my passport", Intent.QUERY_MEMORY),
        ("what's on my shopping list", Intent.SHOPPING_SHOW),
        ("buy eggs", Intent.SHOPPING_ADD),
        ("add water to shopping list", Intent.SHOPPING_ADD),
        ("remove eggs", Intent.SHOPPING_REMOVE),
        ("forget passport", Intent.DELETE_MEMORY),
        ("passport is now in the locker", Intent.UPDATE_MEMORY),
        ("my passport is in the blue drawer", Intent.SAVE_MEMORY),
        ("Send the client deck by Friday EOD", Intent.SAVE_MEMORY),
        ("hey hippo, how are you", Intent.GENERAL_CHAT),
        ("pm resources", Intent.QUERY_MEMORY),
    ],
)
def test_fast_router_common_intents(message: str, expected: Intent) -> None:
    """Common messages should classify without the LLM."""
    result = try_fast_classify(message)
    assert result is not None
    intent, confidence, _reason = result
    assert intent == expected
    assert confidence >= 0.9


def test_fast_router_returns_none_for_ambiguous_input() -> None:
    """Ambiguous text should fall through to the LLM classifier."""
    assert try_fast_classify("xyzabc nonsense phrase here") is None
