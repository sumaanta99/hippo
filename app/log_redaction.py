"""Redact sensitive values before they reach production logs."""

from __future__ import annotations


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
