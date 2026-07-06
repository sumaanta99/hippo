"""Tests for phone number log redaction."""

from __future__ import annotations

from log_redaction import redact_for_log, redact_phone_for_log


def test_redact_for_log_structured() -> None:
    """Structured mode should redact middle content."""
    result = redact_for_log("gas agency number 9876543210", structured=True)
    assert "9876543210" not in result
    assert "len=" in result


def test_redact_for_log_plain() -> None:
    """Plain mode should return text unchanged."""
    text = "gas agency number 9876543210"
    assert redact_for_log(text, structured=False) == text


def test_redact_phone_for_log() -> None:
    """Phone numbers should be partially redacted."""
    result = redact_phone_for_log("whatsapp:+14155551234")
    assert "14155551234" not in result
    assert result.startswith("+14")
