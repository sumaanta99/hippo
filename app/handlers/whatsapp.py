"""WhatsApp webhook handler — receive, verify, route, and reply via Meta or Twilio."""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import httpx

from config import Settings, WhatsAppProvider, get_settings
from engine.hippo_engine import HippoEngine
from logger import get_logger
from log_redaction import redact_phone_for_log
from models.responses import ChatResponse

logger = get_logger(__name__)

META_GRAPH_API_VERSION = "v21.0"
WHATSAPP_MAX_RESPONSE_CHARS = 1500
WHATSAPP_LIST_PREVIEW_ITEMS = 8


@dataclass(frozen=True)
class IncomingWhatsAppMessage:
    """Normalized inbound WhatsApp message."""

    sender_phone: str
    message_text: str
    message_id: str


def session_id_for_phone(phone_number: str, secret: str) -> str:
    """Map a phone number to a deterministic, isolated session id.

    Uses HMAC-SHA256 so the mapping is stable per user but not reversible
    without the secret. No database lookup required.
    """
    normalized = _normalize_phone(phone_number)
    digest = hmac.new(
        secret.encode("utf-8"),
        normalized.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()[:16]
    return f"whatsapp-{digest}"


def _normalize_phone(phone: str) -> str:
    """Strip provider prefixes and keep digits only."""
    cleaned = phone.strip().removeprefix("whatsapp:").lstrip("+")
    return "".join(ch for ch in cleaned if ch.isdigit())


def parse_meta_payload(body: dict[str, Any]) -> IncomingWhatsAppMessage | None:
    """Extract the first text message from a Meta Cloud API webhook payload."""
    for entry in body.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            for message in value.get("messages", []):
                if message.get("type") != "text":
                    continue
                text_body = message.get("text", {}).get("body", "")
                sender = message.get("from", "")
                message_id = message.get("id", "")
                if sender and text_body.strip():
                    return IncomingWhatsAppMessage(
                        sender_phone=sender,
                        message_text=text_body.strip(),
                        message_id=message_id,
                    )
    return None


def parse_twilio_payload(form: dict[str, str]) -> IncomingWhatsAppMessage | None:
    """Extract a text message from a Twilio WhatsApp webhook form body."""
    sender = form.get("From", "")
    text_body = form.get("Body", "")
    message_id = form.get("MessageSid", "")
    if sender and text_body.strip():
        return IncomingWhatsAppMessage(
            sender_phone=sender,
            message_text=text_body.strip(),
            message_id=message_id,
        )
    return None


def verify_meta_signature(body: bytes, signature: str | None, secret: str) -> bool:
    """Verify Meta X-Hub-Signature-256 header."""
    if not signature or not signature.startswith("sha256="):
        return False
    expected = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    received = signature.removeprefix("sha256=")
    return secrets.compare_digest(expected, received)


def verify_twilio_signature(
    url: str,
    params: dict[str, str],
    signature: str | None,
    auth_token: str,
) -> bool:
    """Verify Twilio X-Twilio-Signature header."""
    if not signature:
        return False
    sorted_params = sorted(params.items())
    data = url + urlencode(sorted_params, doseq=True)
    digest = hmac.new(
        auth_token.encode("utf-8"),
        data.encode("utf-8"),
        hashlib.sha1,
    ).digest()
    expected = base64.b64encode(digest).decode("utf-8")
    return secrets.compare_digest(expected, signature)


def format_whatsapp_response(
    chat_response: ChatResponse,
    *,
    studio_url: str = "https://hippostudio.netlify.app",
) -> str:
    """Format a ChatResponse as plain text suitable for WhatsApp."""
    text = chat_response.response.strip()
    if not text:
        return "I didn't quite catch that. Can you rephrase?"

    lines = text.splitlines()
    if len(lines) > WHATSAPP_LIST_PREVIEW_ITEMS + 1:
        preview = "\n".join(lines[: WHATSAPP_LIST_PREVIEW_ITEMS + 1])
        remaining = len(lines) - WHATSAPP_LIST_PREVIEW_ITEMS - 1
        return (
            f"{preview}\n"
            f"... and {remaining} more items.\n"
            f"Full list available at {studio_url}"
        )

    if len(text) <= WHATSAPP_MAX_RESPONSE_CHARS:
        return text

    return text[: WHATSAPP_MAX_RESPONSE_CHARS - 50].rstrip() + f"\n\nFull details at {studio_url}"


async def send_whatsapp_message(
    to_phone: str,
    text: str,
    settings: Settings | None = None,
) -> None:
    """Send a plain-text WhatsApp reply via the configured provider."""
    resolved = settings or get_settings()
    provider = resolved.whatsapp_provider

    if provider == WhatsAppProvider.META:
        await _send_meta_message(to_phone, text, resolved)
    elif provider == WhatsAppProvider.TWILIO:
        await _send_twilio_message(to_phone, text, resolved)
    else:
        raise ValueError(f"Unsupported WhatsApp provider: {provider}")


async def _send_meta_message(to_phone: str, text: str, settings: Settings) -> None:
    phone_number_id = settings.whatsapp_phone_number_id
    access_token = settings.whatsapp_access_token
    if not phone_number_id or not access_token:
        raise RuntimeError("WHATSAPP_PHONE_NUMBER_ID and WHATSAPP_ACCESS_TOKEN required.")

    url = f"https://graph.facebook.com/{META_GRAPH_API_VERSION}/{phone_number_id}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "to": _normalize_phone(to_phone),
        "type": "text",
        "text": {"body": text},
    }
    headers = {"Authorization": f"Bearer {access_token}"}

    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()


async def _send_twilio_message(to_phone: str, text: str, settings: Settings) -> None:
    account_sid = settings.twilio_account_sid
    auth_token = settings.whatsapp_access_token
    from_number = settings.whatsapp_phone_number_id
    if not account_sid or not auth_token or not from_number:
        raise RuntimeError(
            "TWILIO_ACCOUNT_SID, WHATSAPP_ACCESS_TOKEN, and "
            "WHATSAPP_PHONE_NUMBER_ID required for Twilio."
        )

    to = to_phone if to_phone.startswith("whatsapp:") else f"whatsapp:+{_normalize_phone(to_phone)}"
    from_addr = from_number if from_number.startswith("whatsapp:") else f"whatsapp:{from_number}"

    url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
    payload = {"From": from_addr, "To": to, "Body": text}

    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.post(url, data=payload, auth=(account_sid, auth_token))
        response.raise_for_status()


async def process_whatsapp_message(
    incoming: IncomingWhatsAppMessage,
    engine: HippoEngine,
    settings: Settings | None = None,
) -> ChatResponse:
    """Route an inbound WhatsApp message through HippoEngine and send the reply."""
    resolved = settings or get_settings()
    secret = resolved.whatsapp_webhook_secret or resolved.session_secret or "dev-secret"
    session_id = session_id_for_phone(incoming.sender_phone, secret)
    redacted_phone = redact_phone_for_log(incoming.sender_phone)

    logger.log_event(
        "whatsapp_message_received",
        phone=redacted_phone,
        message_id=incoming.message_id,
        session_id=session_id,
    )

    chat_response = await engine.chat(incoming.message_text, session_id)
    reply_text = format_whatsapp_response(
        chat_response,
        studio_url=resolved.whatsapp_studio_url,
    )

    memory_ids = [
        m.id
        for m in (
            chat_response.memories_created
            + chat_response.memories_updated
            + chat_response.memories_deleted
        )
    ]

    logger.log_event(
        "whatsapp_response_sent",
        phone=redacted_phone,
        session_id=session_id,
        intent=chat_response.intent,
        success=True,
        memory_ids=memory_ids or None,
        latency_ms=round(chat_response.latency_ms, 2),
    )

    await send_whatsapp_message(incoming.sender_phone, reply_text, resolved)
    return chat_response
