"""FastAPI server scaffold — reuses HippoEngine for all chat logic."""

from __future__ import annotations

import json
import secrets
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any

from pydantic import BaseModel, Field

from analytics_store import AnalyticsPayload, UsageAnalyticsStore, UsageReport
from config import WhatsAppProvider, get_settings
from engine.hippo_engine import HippoEngine
from handlers.whatsapp import (
    parse_meta_payload,
    parse_twilio_payload,
    process_whatsapp_message,
    verify_meta_signature,
    verify_twilio_signature,
)
from log_redaction import redact_phone_for_log
from logger import get_logger
from models.responses import ChatResponse, MemorySnapshot, SessionStats
from session_auth import create_session_id, sign_session_token, verify_session_token

try:
    from fastapi import Depends, FastAPI, Header, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse, PlainTextResponse, Response
except ImportError:  # pragma: no cover - optional dependency
    Depends = None  # type: ignore[misc, assignment]
    FastAPI = None  # type: ignore[misc, assignment]
    HTTPException = None  # type: ignore[misc, assignment]
    CORSMiddleware = None  # type: ignore[misc, assignment]
    BaseHTTPMiddleware = None  # type: ignore[misc, assignment]
    JSONResponse = None  # type: ignore[misc, assignment]
    PlainTextResponse = None  # type: ignore[misc, assignment]
    Response = None  # type: ignore[misc, assignment]
    Request = None  # type: ignore[misc, assignment]


SESSION_ID_PATTERN = r"^[a-zA-Z0-9_-]{1,128}$"
RATE_LIMIT_PATHS = frozenset({"/chat", "/analytics", "/sessions"})
RATE_LIMIT_MAX_REQUESTS = 60
RATE_LIMIT_WINDOW_SECONDS = 60


class ChatRequest(BaseModel):
    """Incoming chat request body."""

    message: str = Field(max_length=5000)
    session_id: str = Field(min_length=1, max_length=128, pattern=SESSION_ID_PATTERN)


class SessionResponse(BaseModel):
    """Server-issued session credentials."""

    session_id: str
    token: str


_engine: HippoEngine | None = None
_analytics: UsageAnalyticsStore | None = None
_rate_buckets: dict[str, list[float]] = defaultdict(list)
_session_rate_buckets: dict[str, list[float]] = defaultdict(list)
_webhook_logger = get_logger(__name__)


def get_engine() -> HippoEngine:
    """Return the shared engine instance."""
    if _engine is None:
        raise RuntimeError("Engine not initialized.")
    return _engine


def get_analytics() -> UsageAnalyticsStore:
    """Return the shared analytics store."""
    if _analytics is None:
        raise RuntimeError("Analytics store not initialized.")
    return _analytics


def _parse_cors_origins(raw: str) -> list[str]:
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    return origins or ["*"]


def _is_production(settings: Any) -> bool:
    return settings.hippo_env.lower() == "production"


def _session_auth_enabled(settings: Any) -> bool:
    return bool(settings.session_secret)


def _verify_session_access(session_id: str, authorization: str | None) -> None:
    """Ensure the caller owns the session id."""
    settings = get_settings()
    if not _session_auth_enabled(settings):
        return

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized.")

    token = authorization.removeprefix("Bearer ").strip()
    if not verify_session_token(session_id, token, settings.session_secret or ""):
        raise HTTPException(status_code=401, detail="Unauthorized.")


def _enforce_session_rate_limit(session_id: str) -> None:
    """Limit chat volume per authenticated session."""
    settings = get_settings()
    now = time.time()
    bucket = _session_rate_buckets[session_id]
    bucket[:] = [
        timestamp
        for timestamp in bucket
        if now - timestamp < RATE_LIMIT_WINDOW_SECONDS
    ]
    if len(bucket) >= settings.chat_rate_limit_per_session:
        raise HTTPException(
            status_code=429,
            detail="Too many requests for this session. Try again shortly.",
        )
    bucket.append(now)


def _require_analytics_admin(
    authorization: str | None = Header(default=None),
) -> None:
    settings = get_settings()
    admin_key = settings.analytics_admin_key
    if not admin_key:
        raise HTTPException(status_code=503, detail="Analytics admin key not configured.")

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized.")

    token = authorization.removeprefix("Bearer ").strip()
    if not secrets.compare_digest(token, admin_key):
        raise HTTPException(status_code=401, detail="Unauthorized.")


def _validate_session_id(session_id: str) -> str:
    """Reject malformed session identifiers."""
    if len(session_id) > 128 or not session_id.replace("-", "").replace("_", "").isalnum():
        raise HTTPException(status_code=400, detail="Invalid session id.")
    return session_id


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiter for public write endpoints."""

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        if request.url.path not in RATE_LIMIT_PATHS:
            return await call_next(request)

        client = request.client.host if request.client else "unknown"
        now = time.time()
        bucket = _rate_buckets[client]
        bucket[:] = [
            timestamp
            for timestamp in bucket
            if now - timestamp < RATE_LIMIT_WINDOW_SECONDS
        ]

        if len(bucket) >= RATE_LIMIT_MAX_REQUESTS:
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Try again shortly."},
            )

        bucket.append(now)
        return await call_next(request)


def create_app() -> Any:
    """Create the FastAPI application."""
    if FastAPI is None:
        raise ImportError(
            "FastAPI is not installed. Add fastapi and uvicorn to requirements.txt."
        )

    settings = get_settings()
    production = _is_production(settings)
    cors_origins = _parse_cors_origins(settings.cors_origins)
    allow_credentials = "*" not in cors_origins

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        global _engine, _analytics
        _engine = HippoEngine(settings)
        await _engine.initialize()
        _analytics = UsageAnalyticsStore(settings)
        await _analytics.initialize()
        yield

    app = FastAPI(
        title="Hippo Memory API",
        version="1.0.0",
        lifespan=lifespan,
        docs_url=None if production else "/docs",
        redoc_url=None if production else "/redoc",
        openapi_url=None if production else "/openapi.json",
    )

    app.add_middleware(RateLimitMiddleware)

    if CORSMiddleware is not None:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=allow_credentials,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type"],
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Health check for deployment and local dev."""
        return {"status": "ok"}

    @app.post("/sessions", response_model=SessionResponse)
    async def create_session() -> SessionResponse:
        """Issue a server-signed session id and token."""
        session_id = create_session_id()
        secret = settings.session_secret
        token = sign_session_token(session_id, secret) if secret else ""
        return SessionResponse(session_id=session_id, token=token)

    @app.post("/chat", response_model=ChatResponse)
    async def chat(
        request: ChatRequest,
        authorization: str | None = Header(default=None),
    ) -> ChatResponse:
        """Process a natural-language message for a session."""
        _verify_session_access(request.session_id, authorization)
        _enforce_session_rate_limit(request.session_id)
        return await get_engine().chat(request.message, request.session_id)

    @app.post("/analytics")
    async def record_analytics(
        payload: AnalyticsPayload,
        authorization: str | None = Header(default=None),
    ) -> dict[str, bool]:
        """Record private studio usage metrics."""
        _verify_session_access(payload.session_id, authorization)
        await get_analytics().record_event(payload)
        return {"ok": True}

    @app.get("/admin/analytics", response_model=UsageReport)
    async def analytics_report(_: None = Depends(_require_analytics_admin)) -> UsageReport:
        """Return private usage metrics. Requires Bearer ANALYTICS_ADMIN_KEY."""
        return await get_analytics().get_report()

    @app.get("/memories/{session_id}", response_model=list[MemorySnapshot])
    async def list_memories(
        session_id: str,
        authorization: str | None = Header(default=None),
    ) -> list[MemorySnapshot]:
        """List active memories for a session."""
        validated = _validate_session_id(session_id)
        _verify_session_access(validated, authorization)
        return await get_engine().get_memories(validated)

    @app.delete("/memories/{memory_id}")
    async def delete_memory(
        memory_id: str,
        session_id: str,
        authorization: str | None = Header(default=None),
    ) -> dict[str, bool]:
        """Archive a memory by id."""
        validated = _validate_session_id(session_id)
        _verify_session_access(validated, authorization)
        deleted = await get_engine().delete_memory(memory_id, validated)
        if not deleted:
            raise HTTPException(status_code=404, detail="Memory not found.")
        return {"deleted": True}

    @app.get("/stats/{session_id}", response_model=SessionStats)
    async def stats(
        session_id: str,
        authorization: str | None = Header(default=None),
    ) -> SessionStats:
        """Return session statistics."""
        validated = _validate_session_id(session_id)
        _verify_session_access(validated, authorization)
        return await get_engine().get_stats(validated)

    @app.delete("/sessions/{session_id}")
    async def clear_session(
        session_id: str,
        authorization: str | None = Header(default=None),
    ) -> dict[str, str]:
        """Clear all memories and shopping items for a session."""
        validated = _validate_session_id(session_id)
        _verify_session_access(validated, authorization)
        await get_engine().clear_session(validated)
        return {"status": "cleared"}

    @app.get("/whatsapp/webhook")
    async def whatsapp_webhook_verify(
        request: Request,
    ) -> PlainTextResponse:
        """Meta webhook verification challenge (GET)."""
        params = request.query_params
        mode = params.get("hub.mode")
        token = params.get("hub.verify_token")
        challenge = params.get("hub.challenge", "")

        secret = settings.whatsapp_webhook_secret
        if mode == "subscribe" and secret and token == secret:
            return PlainTextResponse(content=challenge)
        raise HTTPException(status_code=403, detail="Verification failed.")

    @app.post("/whatsapp/webhook")
    async def whatsapp_webhook(request: Request) -> Response:
        """Receive inbound WhatsApp messages from Meta or Twilio."""
        provider = settings.whatsapp_provider
        secret = settings.whatsapp_webhook_secret

        if provider == WhatsAppProvider.META:
            body = await request.body()
            if secret and not verify_meta_signature(
                body,
                request.headers.get("X-Hub-Signature-256"),
                secret,
            ):
                raise HTTPException(status_code=403, detail="Invalid signature.")

            payload = json.loads(body)
            incoming = parse_meta_payload(payload)
        else:
            form = {key: str(value) for key, value in (await request.form()).multi_items()}
            if secret:
                auth_token = settings.whatsapp_access_token or secret
                if not verify_twilio_signature(
                    str(request.url),
                    form,
                    request.headers.get("X-Twilio-Signature"),
                    auth_token,
                ):
                    raise HTTPException(status_code=403, detail="Invalid signature.")
            incoming = parse_twilio_payload(form)

        if incoming is None:
            return Response(status_code=200)

        try:
            await process_whatsapp_message(incoming, get_engine(), settings)
        except Exception as exc:
            _webhook_logger.error(
                "WhatsApp message processing failed.",
                error_type=type(exc).__name__,
                phone=redact_phone_for_log(incoming.sender_phone),
                message_id=incoming.message_id,
                exc=exc,
            )
            raise HTTPException(status_code=500, detail="Processing failed.") from exc

        return Response(status_code=200)

    return app
