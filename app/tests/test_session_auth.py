"""Tests for session token signing."""

from __future__ import annotations

import time

import pytest

from session_auth import (
    TOKEN_VERSION,
    create_session_id,
    is_token_expired,
    sign_session_token,
    verify_session_token,
)


def test_sign_and_verify_session_token() -> None:
    """Valid tokens should verify for their session id."""
    session_id = create_session_id()
    secret = "test-secret"
    token, expires_at = sign_session_token(session_id, secret, ttl_seconds=3600)
    assert token.startswith(f"{TOKEN_VERSION}.")
    assert expires_at > int(time.time())
    assert verify_session_token(session_id, token, secret)


def test_reject_wrong_session_or_token() -> None:
    """Tokens should not verify for a different session id."""
    secret = "test-secret"
    token, _ = sign_session_token("web-a", secret, ttl_seconds=3600)
    assert not verify_session_token("web-b", token, secret)
    assert not verify_session_token("web-a", "bad-token", secret)


def test_expired_token_is_rejected() -> None:
    """Expired tokens should fail verification outside the grace window."""
    session_id = create_session_id()
    secret = "test-secret"
    token, _ = sign_session_token(
        session_id,
        secret,
        ttl_seconds=60,
        issued_at=int(time.time()) - 120,
    )
    assert not verify_session_token(session_id, token, secret)


def test_expired_token_can_refresh_within_grace() -> None:
    """Expired tokens should verify when refresh grace is enabled."""
    session_id = create_session_id()
    secret = "test-secret"
    token, _ = sign_session_token(
        session_id,
        secret,
        ttl_seconds=60,
        issued_at=int(time.time()) - 120,
    )
    assert verify_session_token(
        session_id,
        token,
        secret,
        allow_expired_grace_seconds=3600,
    )


def test_legacy_token_still_verifies() -> None:
    """Legacy non-expiring tokens should remain valid during migration."""
    import hashlib
    import hmac

    session_id = "web-legacy"
    secret = "test-secret"
    legacy = hmac.new(
        secret.encode("utf-8"),
        session_id.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    assert verify_session_token(session_id, legacy, secret)


def test_is_token_expired() -> None:
    """Expired versioned tokens should report expiry after grace."""
    token, _ = sign_session_token(
        "web-a",
        "secret",
        ttl_seconds=10,
        issued_at=int(time.time()) - 100,
    )
    assert is_token_expired(token, grace_seconds=0)
    assert not is_token_expired(token, grace_seconds=3600)
