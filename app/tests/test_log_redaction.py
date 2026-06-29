"""Tests for log redaction."""

from __future__ import annotations

from log_redaction import redact_for_log


def test_redact_for_log_structured() -> None:
    """Structured logs should not include raw user text."""
    assert redact_for_log("where is my passport", structured=True) == "whe…rt (len=20)"


def test_redact_for_log_plain() -> None:
    """Non-structured logs may keep the original query."""
    assert redact_for_log("where is my passport", structured=False) == "where is my passport"
