"""FastAPI server scaffold — reuses HippoEngine for all chat logic."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from pydantic import BaseModel, Field

from analytics_store import AnalyticsPayload, UsageAnalyticsStore, UsageReport
from config import get_settings
from engine.hippo_engine import HippoEngine
from models.responses import ChatResponse, MemorySnapshot, SessionStats

try:
    from fastapi import Depends, FastAPI, Header, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
except ImportError:  # pragma: no cover - optional dependency
    Depends = None  # type: ignore[misc, assignment]
    FastAPI = None  # type: ignore[misc, assignment]
    HTTPException = None  # type: ignore[misc, assignment]
    CORSMiddleware = None  # type: ignore[misc, assignment]


class ChatRequest(BaseModel):
    """Incoming chat request body."""

    message: str
    session_id: str = Field(min_length=1)


class DeleteMemoryRequest(BaseModel):
    """Request to delete a memory by id."""

    memory_id: str
    session_id: str = Field(min_length=1)


_engine: HippoEngine | None = None
_analytics: UsageAnalyticsStore | None = None


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
    if token != admin_key:
        raise HTTPException(status_code=401, detail="Unauthorized.")


def create_app() -> Any:
    """Create the FastAPI application."""
    if FastAPI is None:
        raise ImportError(
            "FastAPI is not installed. Add fastapi and uvicorn to requirements.txt."
        )

    settings = get_settings()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        global _engine, _analytics
        _engine = HippoEngine(settings)
        await _engine.initialize()
        _analytics = UsageAnalyticsStore(settings)
        await _analytics.initialize()
        yield

    app = FastAPI(title="Hippo Memory API", version="1.0.0", lifespan=lifespan)

    if CORSMiddleware is not None:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=_parse_cors_origins(settings.cors_origins),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Health check for deployment and local dev."""
        return {"status": "ok"}

    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        """Process a natural-language message for a session."""
        return await get_engine().chat(request.message, request.session_id)

    @app.post("/analytics")
    async def record_analytics(payload: AnalyticsPayload) -> dict[str, bool]:
        """Record private studio usage metrics."""
        await get_analytics().record_event(payload)
        return {"ok": True}

    @app.get("/admin/analytics", response_model=UsageReport)
    async def analytics_report(_: None = Depends(_require_analytics_admin)) -> UsageReport:
        """Return private usage metrics. Requires Bearer ANALYTICS_ADMIN_KEY."""
        return await get_analytics().get_report()

    @app.get("/memories/{session_id}", response_model=list[MemorySnapshot])
    async def list_memories(session_id: str) -> list[MemorySnapshot]:
        """List active memories for a session."""
        return await get_engine().get_memories(session_id)

    @app.delete("/memories/{memory_id}")
    async def delete_memory(memory_id: str, session_id: str) -> dict[str, bool]:
        """Archive a memory by id."""
        deleted = await get_engine().delete_memory(memory_id, session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Memory not found.")
        return {"deleted": True}

    @app.get("/stats/{session_id}", response_model=SessionStats)
    async def stats(session_id: str) -> SessionStats:
        """Return session statistics."""
        return await get_engine().get_stats(session_id)

    @app.delete("/sessions/{session_id}")
    async def clear_session(session_id: str) -> dict[str, str]:
        """Clear all memories and shopping items for a session."""
        await get_engine().clear_session(session_id)
        return {"status": "cleared"}

    return app
