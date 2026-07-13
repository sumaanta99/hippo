"""HMAC session tokens for web studio clients."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
import uuid

TOKEN_VERSION = "v1"
DEFAULT_SESSION_TOKEN_TTL_SECONDS = 86_400
DEFAULT_SESSION_TOKEN_REFRESH_GRACE_SECONDS = 3_600


def create_session_id() -> str:
    """Generate a server-issued session identifier."""
    return f"web-{uuid.uuid4()}"


def _sign_payload(payload: str, secret: str) -> str:
    return hmac.new(
        secret.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def sign_session_token(
    session_id: str,
    secret: str,
    *,
    ttl_seconds: int = DEFAULT_SESSION_TOKEN_TTL_SECONDS,
    issued_at: int | None = None,
) -> tuple[str, int]:
    """Return a versioned token and its expiry timestamp."""
    now = issued_at if issued_at is not None else int(time.time())
    expires_at = now + ttl_seconds
    payload = f"{TOKEN_VERSION}:{session_id}:{expires_at}"
    signature = _sign_payload(payload, secret)
    return f"{TOKEN_VERSION}.{expires_at}.{signature}", expires_at


def _verify_versioned_token(
    session_id: str,
    token: str,
    secret: str,
    *,
    allow_expired_grace_seconds: int = 0,
) -> bool:
    parts = token.split(".")
    if len(parts) != 3 or parts[0] != TOKEN_VERSION:
        return False

    try:
        expires_at = int(parts[1])
    except ValueError:
        return False

    now = int(time.time())
    if now > expires_at + allow_expired_grace_seconds:
        return False

    payload = f"{TOKEN_VERSION}:{session_id}:{expires_at}"
    expected = _sign_payload(payload, secret)
    return secrets.compare_digest(parts[2], expected)


def _verify_legacy_token(session_id: str, token: str, secret: str) -> bool:
    """Verify legacy non-expiring tokens issued before v1."""
    expected = _sign_payload(session_id, secret)
    return secrets.compare_digest(token, expected)


def verify_session_token(
    session_id: str,
    token: str,
    secret: str,
    *,
    allow_expired_grace_seconds: int = 0,
) -> bool:
    """Return True when the token matches the session id."""
    if not session_id or not token or not secret:
        return False

    if token.startswith(f"{TOKEN_VERSION}."):
        return _verify_versioned_token(
            session_id,
            token,
            secret,
            allow_expired_grace_seconds=allow_expired_grace_seconds,
        )

    return _verify_legacy_token(session_id, token, secret)


def is_token_expired(
    token: str,
    *,
    grace_seconds: int = DEFAULT_SESSION_TOKEN_REFRESH_GRACE_SECONDS,
) -> bool:
    """Return True when a versioned token is past its refresh grace window."""
    if not token.startswith(f"{TOKEN_VERSION}."):
        return False

    parts = token.split(".")
    if len(parts) != 3:
        return True

    try:
        expires_at = int(parts[1])
    except ValueError:
        return True

    return int(time.time()) > expires_at + grace_seconds
