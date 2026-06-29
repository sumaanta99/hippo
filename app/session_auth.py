"""HMAC session tokens for web studio clients."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import uuid


def create_session_id() -> str:
    """Generate a server-issued session identifier."""
    return f"web-{uuid.uuid4()}"


def sign_session_token(session_id: str, secret: str) -> str:
    """Return a hex HMAC token bound to a session id."""
    digest = hmac.new(
        secret.encode("utf-8"),
        session_id.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return digest


def verify_session_token(session_id: str, token: str, secret: str) -> bool:
    """Return True when the token matches the session id."""
    if not session_id or not token or not secret:
        return False

    expected = sign_session_token(session_id, secret)
    return secrets.compare_digest(token, expected)
