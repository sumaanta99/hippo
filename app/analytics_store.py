"""SQLite-backed usage analytics for the web studio."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Literal

import aiosqlite
from pydantic import BaseModel, ConfigDict, Field

from config import Settings, get_database_path, get_settings


AnalyticsEventType = Literal["session_start", "chat_success", "chat_error"]


class AnalyticsPayload(BaseModel):
    """Incoming analytics event from the web client."""

    model_config = ConfigDict(populate_by_name=True)

    type: AnalyticsEventType
    session_id: str = Field(alias="sessionId", min_length=1)
    timestamp: str | None = None
    intent: str | None = None
    confidence: float | None = None
    latency_ms: float | None = Field(default=None, alias="latencyMs")
    message_length: int | None = Field(default=None, alias="messageLength")
    memories_created: int | None = Field(default=None, alias="memoriesCreated")
    memories_updated: int | None = Field(default=None, alias="memoriesUpdated")
    memories_deleted: int | None = Field(default=None, alias="memoriesDeleted")
    search_result_count: int | None = Field(default=None, alias="searchResultCount")
    error: str | None = None


class SessionMetrics(BaseModel):
    """Aggregated metrics for one browser session."""

    first_seen: str
    last_seen: str
    message_count: int
    error_count: int
    intents: dict[str, int]
    total_latency_ms: float
    memories_created: int
    memories_updated: int
    memories_deleted: int


class UsageReport(BaseModel):
    """Private analytics summary."""

    updated_at: str
    summary: dict[str, Any]
    sessions: dict[str, SessionMetrics]
    agent_insights: dict[str, Any] = Field(default_factory=dict)


class UsageAnalyticsStore:
    """Persist and aggregate studio usage metrics."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._db_path = get_database_path(self._settings)

    async def initialize(self) -> None:
        """Create analytics tables if they do not exist."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS usage_sessions (
                    session_id TEXT PRIMARY KEY,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    message_count INTEGER NOT NULL DEFAULT 0,
                    error_count INTEGER NOT NULL DEFAULT 0,
                    total_latency_ms REAL NOT NULL DEFAULT 0,
                    memories_created INTEGER NOT NULL DEFAULT 0,
                    memories_updated INTEGER NOT NULL DEFAULT 0,
                    memories_deleted INTEGER NOT NULL DEFAULT 0,
                    intents_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            await db.commit()

    async def record_event(self, payload: AnalyticsPayload) -> None:
        """Record a session or chat analytics event."""
        timestamp = payload.timestamp or datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            row = await (
                await db.execute(
                    "SELECT * FROM usage_sessions WHERE session_id = ?",
                    (payload.session_id,),
                )
            ).fetchone()

            if row is None:
                await db.execute(
                    """
                    INSERT INTO usage_sessions (
                        session_id, first_seen, last_seen
                    ) VALUES (?, ?, ?)
                    """,
                    (payload.session_id, timestamp, timestamp),
                )
            else:
                await db.execute(
                    """
                    UPDATE usage_sessions
                    SET last_seen = ?
                    WHERE session_id = ?
                    """,
                    (timestamp, payload.session_id),
                )

            if payload.type == "session_start":
                await db.commit()
                return

            row = await (
                await db.execute(
                    "SELECT * FROM usage_sessions WHERE session_id = ?",
                    (payload.session_id,),
                )
            ).fetchone()
            if row is None:
                await db.commit()
                return

            intents = json.loads(row["intents_json"] or "{}")
            message_count = int(row["message_count"])
            error_count = int(row["error_count"])
            total_latency_ms = float(row["total_latency_ms"])
            memories_created = int(row["memories_created"])
            memories_updated = int(row["memories_updated"])
            memories_deleted = int(row["memories_deleted"])

            if payload.type == "chat_error":
                message_count += 1
                error_count += 1
            elif payload.type == "chat_success":
                message_count += 1
                total_latency_ms += float(payload.latency_ms or 0)
                memories_created += int(payload.memories_created or 0)
                memories_updated += int(payload.memories_updated or 0)
                memories_deleted += int(payload.memories_deleted or 0)
                if payload.intent:
                    intents[payload.intent] = intents.get(payload.intent, 0) + 1

            await db.execute(
                """
                UPDATE usage_sessions
                SET message_count = ?,
                    error_count = ?,
                    total_latency_ms = ?,
                    memories_created = ?,
                    memories_updated = ?,
                    memories_deleted = ?,
                    intents_json = ?,
                    last_seen = ?
                WHERE session_id = ?
                """,
                (
                    message_count,
                    error_count,
                    total_latency_ms,
                    memories_created,
                    memories_updated,
                    memories_deleted,
                    json.dumps(intents),
                    timestamp,
                    payload.session_id,
                ),
            )
            await db.commit()

    async def get_report(self) -> UsageReport:
        """Build a usage report from stored session metrics."""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            rows = await (await db.execute("SELECT * FROM usage_sessions")).fetchall()

        sessions: dict[str, SessionMetrics] = {}
        total_messages = 0
        total_errors = 0
        total_latency_ms = 0.0
        latency_samples = 0
        total_memories_created = 0
        intent_breakdown: dict[str, int] = {}

        for row in rows:
            intents = json.loads(row["intents_json"] or "{}")
            message_count = int(row["message_count"])
            session = SessionMetrics(
                first_seen=row["first_seen"],
                last_seen=row["last_seen"],
                message_count=message_count,
                error_count=int(row["error_count"]),
                intents=intents,
                total_latency_ms=float(row["total_latency_ms"]),
                memories_created=int(row["memories_created"]),
                memories_updated=int(row["memories_updated"]),
                memories_deleted=int(row["memories_deleted"]),
            )
            sessions[row["session_id"]] = session

            total_messages += message_count
            total_errors += session.error_count
            total_memories_created += session.memories_created
            if message_count > 0:
                total_latency_ms += session.total_latency_ms
                latency_samples += message_count
            for intent, count in intents.items():
                intent_breakdown[intent] = intent_breakdown.get(intent, 0) + count

        updated_at = datetime.now(timezone.utc).isoformat()
        if rows:
            updated_at = max(row["last_seen"] for row in rows)

        return UsageReport(
            updated_at=updated_at,
            summary={
                "uniqueUsers": len(sessions),
                "totalMessages": total_messages,
                "totalErrors": total_errors,
                "avgLatencyMs": round(total_latency_ms / latency_samples)
                if latency_samples
                else 0,
                "totalMemoriesCreated": total_memories_created,
                "intentBreakdown": intent_breakdown,
            },
            sessions=sessions,
        )
