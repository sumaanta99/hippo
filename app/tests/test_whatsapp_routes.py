"""Tests for WhatsApp webhook HTTP routes."""

from __future__ import annotations

import hashlib
import hmac
import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from api.server import create_app
from config import Settings, WhatsAppProvider


WEBHOOK_SECRET = "test-webhook-secret"


@pytest.fixture
def whatsapp_settings(temp_db_path) -> Settings:
    return Settings(
        openai_api_key="test-key",
        database_path=str(temp_db_path),
        whatsapp_provider=WhatsAppProvider.META,
        whatsapp_webhook_secret=WEBHOOK_SECRET,
        whatsapp_phone_number_id="123456789",
        whatsapp_access_token="test-token",
        session_secret="test-session-secret",
    )


@pytest.fixture
def client(whatsapp_settings: Settings, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("api.server.get_settings", lambda: whatsapp_settings)
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def _meta_body(from_number: str, text: str) -> bytes:
    payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "messages": [
                                {
                                    "from": from_number,
                                    "id": "wamid.route-test",
                                    "type": "text",
                                    "text": {"body": text},
                                }
                            ]
                        },
                        "field": "messages",
                    }
                ]
            }
        ],
    }
    return json.dumps(payload).encode()


def _meta_signature(body: bytes) -> str:
    digest = hmac.new(WEBHOOK_SECRET.encode(), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def test_whatsapp_webhook_verify_challenge(client: TestClient) -> None:
    """GET /whatsapp/webhook should return the Meta verification challenge."""
    response = client.get(
        "/whatsapp/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": WEBHOOK_SECRET,
            "hub.challenge": "challenge-token-123",
        },
    )
    assert response.status_code == 200
    assert response.text == "challenge-token-123"


def test_whatsapp_webhook_verify_rejects_bad_token(client: TestClient) -> None:
    """GET /whatsapp/webhook should reject an invalid verify token."""
    response = client.get(
        "/whatsapp/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "wrong-token",
            "hub.challenge": "challenge-token-123",
        },
    )
    assert response.status_code == 403


def test_whatsapp_webhook_post_processes_message(client: TestClient) -> None:
    """POST /whatsapp/webhook should accept a signed Meta payload."""
    body = _meta_body("14155551234", "hi")
    with patch(
        "api.server.process_whatsapp_message",
        new_callable=AsyncMock,
    ) as mock_process:
        response = client.post(
            "/whatsapp/webhook",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": _meta_signature(body),
            },
        )
    assert response.status_code == 200
    mock_process.assert_awaited_once()


def test_whatsapp_webhook_post_rejects_bad_signature(client: TestClient) -> None:
    """POST /whatsapp/webhook should reject an invalid signature."""
    body = _meta_body("14155551234", "hi")
    response = client.post(
        "/whatsapp/webhook",
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-Hub-Signature-256": "sha256=invalid",
        },
    )
    assert response.status_code == 403


def test_whatsapp_webhook_post_ignores_invalid_json(client: TestClient) -> None:
    """Malformed JSON should ack 200 so Meta does not retry."""
    body = b"not-json"
    with patch(
        "api.server.process_whatsapp_message",
        new_callable=AsyncMock,
    ) as mock_process:
        response = client.post(
            "/whatsapp/webhook",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": _meta_signature(body),
            },
        )
    assert response.status_code == 200
    mock_process.assert_not_called()
