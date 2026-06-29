"""Tests for session token signing."""

from __future__ import annotations

from session_auth import (
    create_session_id,
    sign_session_token,
    verify_session_token,
)


def test_sign_and_verify_session_token() -> None:
    """Valid tokens should verify for their session id."""
    session_id = create_session_id()
    secret = "test-secret"
    token = sign_session_token(session_id, secret)
    assert verify_session_token(session_id, token, secret)


def test_reject_wrong_session_or_token() -> None:
    """Tokens should not verify for a different session id."""
    secret = "test-secret"
    token = sign_session_token("web-a", secret)
    assert not verify_session_token("web-b", token, secret)
    assert not verify_session_token("web-a", "bad-token", secret)
