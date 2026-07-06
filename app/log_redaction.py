"""Redact sensitive values before they reach production logs."""

from __future__ import annotations


def redact_phone_for_log(phone: str) -> str:
    """Return a redacted phone number safe for production logs."""
    cleaned = phone.strip().removeprefix("whatsapp:").lstrip("+")
    if not cleaned:
        return "<empty>"
    if len(cleaned) <= 4:
        return "<phone redacted>"
    return f"+{cleaned[:2]}…{cleaned[-2:]}"


def redact_for_log(text: str, *, structured: bool) -> str:
    """Return a safe log representation of user-provided text."""
    cleaned = text.strip()
    if not structured:
        return cleaned
    if not cleaned:
        return "<empty>"
    if len(cleaned) <= 8:
        return f"<redacted len={len(cleaned)}>"
    return f"{cleaned[:3]}…{cleaned[-2:]} (len={len(cleaned)})"
