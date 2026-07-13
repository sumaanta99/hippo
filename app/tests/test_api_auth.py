"""Tests for API auth and session lifecycle routes."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from api.server import create_app
from config import Settings
from models.responses import ChatResponse
from session_auth import sign_session_token


@pytest.fixture
def api_settings(temp_db_path) -> Settings:
    return Settings(
        openai_api_key="test-key",
        database_path=str(temp_db_path),
        session_secret="test-session-secret",
        session_token_ttl_seconds=3600,
    )


@pytest.fixture
def client(api_settings: Settings, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("api.server.get_settings", lambda: api_settings)
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_create_session_returns_expiring_token(client: TestClient) -> None:
    """POST /sessions should return a versioned token and expiry."""
    response = client.post("/sessions")
    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"].startswith("web-")
    assert payload["token"].startswith("v1.")
    assert payload["expires_at"] > 0


def test_chat_requires_bearer_token(client: TestClient) -> None:
    """Protected routes should reject missing auth."""
    response = client.post(
        "/chat",
        json={"message": "hello", "session_id": "web-test"},
    )
    assert response.status_code == 401


def test_refresh_session_reissues_token(client: TestClient) -> None:
    """POST /sessions/refresh should mint a new token for the same session."""
    created = client.post("/sessions").json()
    session_id = created["session_id"]
    token = created["token"]

    refreshed = client.post(
        "/sessions/refresh",
        json={"session_id": session_id},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert refreshed.status_code == 200
    body = refreshed.json()
    assert body["session_id"] == session_id
    assert body["expires_at"] >= created["expires_at"]
    assert body["token"].startswith("v1.")


def test_refresh_rejects_unknown_session_token(client: TestClient) -> None:
    """Refresh should reject tokens for a different session id."""
    created = client.post("/sessions").json()
    token, _ = sign_session_token("web-other", "test-session-secret", ttl_seconds=3600)

    response = client.post(
        "/sessions/refresh",
        json={"session_id": created["session_id"]},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 401


def test_chat_accepts_valid_session(client: TestClient) -> None:
    """Authenticated chat requests should reach the engine."""
    created = client.post("/sessions").json()
    with patch("api.server.get_engine") as mock_get_engine:
        engine = AsyncMock()
        engine.chat = AsyncMock(
            return_value=ChatResponse(
                response="ok",
                intent="general_chat",
                confidence=1.0,
                session_id=created["session_id"],
            )
        )
        mock_get_engine.return_value = engine

        response = client.post(
            "/chat",
            json={"message": "hello", "session_id": created["session_id"]},
            headers={"Authorization": f"Bearer {created['token']}"},
        )

    assert response.status_code == 200
    engine.chat.assert_awaited_once()
