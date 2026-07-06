"""Unit tests for WhatsApp webhook handler."""

from __future__ import annotations

import hashlib
import hmac
import json
from unittest.mock import AsyncMock, patch

import pytest

from config import WhatsAppProvider
from handlers.whatsapp import (
    IncomingWhatsAppMessage,
    format_whatsapp_response,
    parse_meta_payload,
    parse_twilio_payload,
    process_whatsapp_message,
    session_id_for_phone,
    verify_meta_signature,
    verify_twilio_signature,
)
from models.responses import ChatResponse


WEBHOOK_SECRET = "test-webhook-secret"


def _meta_payload(from_number: str, body: str, message_id: str = "wamid.abc123") -> dict:
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "WABA_ID",
                "changes": [
                    {
                        "value": {
                            "messaging_product": "whatsapp",
                            "messages": [
                                {
                                    "from": from_number,
                                    "id": message_id,
                                    "timestamp": "1710000000",
                                    "type": "text",
                                    "text": {"body": body},
                                }
                            ],
                        },
                        "field": "messages",
                    }
                ],
            }
        ],
    }


def _sign_meta_body(body: bytes, secret: str) -> str:
    digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def test_session_id_mapping_is_deterministic() -> None:
    """Same phone + secret should always produce the same session id."""
    phone = "+14155551234"
    first = session_id_for_phone(phone, WEBHOOK_SECRET)
    second = session_id_for_phone(phone, WEBHOOK_SECRET)
    assert first == second
    assert first.startswith("whatsapp-")


def test_session_id_mapping_is_isolated_per_phone() -> None:
    """Different phone numbers should map to different session ids."""
    a = session_id_for_phone("+14155551234", WEBHOOK_SECRET)
    b = session_id_for_phone("+14155559999", WEBHOOK_SECRET)
    assert a != b


def test_session_id_normalizes_phone_formats() -> None:
    """Provider prefixes and formatting should not change the session id."""
    plain = session_id_for_phone("+14155551234", WEBHOOK_SECRET)
    twilio = session_id_for_phone("whatsapp:+14155551234", WEBHOOK_SECRET)
    assert plain == twilio


def test_parse_meta_payload_extracts_message() -> None:
    """Meta webhook JSON should yield sender, text, and message id."""
    payload = _meta_payload("14155551234", "remember my coffee order")
    incoming = parse_meta_payload(payload)
    assert incoming is not None
    assert incoming.sender_phone == "14155551234"
    assert incoming.message_text == "remember my coffee order"
    assert incoming.message_id == "wamid.abc123"


def test_parse_meta_payload_ignores_non_text() -> None:
    """Non-text Meta messages should be ignored."""
    payload = _meta_payload("14155551234", "hello")
    payload["entry"][0]["changes"][0]["value"]["messages"][0]["type"] = "image"
    assert parse_meta_payload(payload) is None


def test_parse_twilio_payload_extracts_message() -> None:
    """Twilio form payload should yield sender, text, and sid."""
    form = {
        "From": "whatsapp:+14155551234",
        "Body": "add milk to shopping",
        "MessageSid": "SM1234567890",
    }
    incoming = parse_twilio_payload(form)
    assert incoming is not None
    assert incoming.sender_phone == "whatsapp:+14155551234"
    assert incoming.message_text == "add milk to shopping"
    assert incoming.message_id == "SM1234567890"


def test_verify_meta_signature_accepts_valid() -> None:
    """Valid Meta signature should pass verification."""
    body = json.dumps(_meta_payload("1", "hi")).encode()
    signature = _sign_meta_body(body, WEBHOOK_SECRET)
    assert verify_meta_signature(body, signature, WEBHOOK_SECRET)


def test_verify_meta_signature_rejects_invalid() -> None:
    """Invalid Meta signature should fail verification."""
    body = b'{"test": true}'
    assert not verify_meta_signature(body, "sha256=deadbeef", WEBHOOK_SECRET)


def test_verify_twilio_signature_accepts_valid() -> None:
    """Valid Twilio signature should pass verification."""
    import base64

    url = "https://hippo-api.onrender.com/whatsapp/webhook"
    params = {"From": "whatsapp:+1", "Body": "hi", "MessageSid": "SM1"}
    auth_token = "twilio-auth-token"
    data = url + "Body=hi&From=whatsapp%3A%2B1&MessageSid=SM1"
    digest = hmac.new(auth_token.encode(), data.encode(), hashlib.sha1).digest()
    signature = base64.b64encode(digest).decode()
    assert verify_twilio_signature(url, params, signature, auth_token)


def test_format_whatsapp_response_passes_through_short_text() -> None:
    """Short responses should pass through unchanged."""
    chat = ChatResponse(response="I'll remember that.", intent="save", session_id="s1")
    assert format_whatsapp_response(chat) == "I'll remember that."


def test_format_whatsapp_response_truncates_long_lists() -> None:
    """Long list responses should summarize and link to the studio."""
    lines = ["Found it."] + [f"- item {i}" for i in range(20)]
    chat = ChatResponse(
        response="\n".join(lines),
        intent="recall",
        session_id="s1",
    )
    formatted = format_whatsapp_response(chat, studio_url="https://example.com")
    assert "Full list available at https://example.com" in formatted
    assert "item 0" in formatted
    assert "item 19" not in formatted


@pytest.mark.asyncio
async def test_process_whatsapp_message_calls_engine_and_sends(test_settings) -> None:
    """Inbound message should route through HippoEngine with mapped session id."""
    test_settings.whatsapp_webhook_secret = WEBHOOK_SECRET
    test_settings.whatsapp_provider = WhatsAppProvider.META
    test_settings.whatsapp_phone_number_id = "123456"
    test_settings.whatsapp_access_token = "token"

    incoming = IncomingWhatsAppMessage(
        sender_phone="+14155551234",
        message_text="what's my coffee order",
        message_id="wamid.test",
    )
    expected_session = session_id_for_phone(incoming.sender_phone, WEBHOOK_SECRET)
    mock_engine = AsyncMock()
    mock_engine.chat = AsyncMock(
        return_value=ChatResponse(
            response="Your coffee order is a flat white.",
            intent="recall",
            session_id=expected_session,
            latency_ms=42.0,
        )
    )

    with patch(
        "handlers.whatsapp.send_whatsapp_message",
        new_callable=AsyncMock,
    ) as mock_send:
        result = await process_whatsapp_message(incoming, mock_engine, test_settings)

    mock_engine.chat.assert_awaited_once_with(
        "what's my coffee order",
        expected_session,
    )
    mock_send.assert_awaited_once()
    assert result.intent == "recall"
    assert mock_send.call_args[0][1] == "Your coffee order is a flat white."
