"""SQLite persistence for chat turns and agent traces."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

import aiosqlite
from pydantic import BaseModel, Field

from config import Settings, get_database_path, get_settings


class ChatTurn(BaseModel):
    """One user-facing chat turn."""

    id: str
    session_id: str
    user_message: str
    assistant_response: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    agent_trace: list[dict[str, Any]] | None = None
    created_at: str


class ConversationStore:
    """Persist chat turns per session."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._db_path = get_database_path(self._settings)

    async def initialize(self) -> None:
        """Create chat turn tables if they do not exist."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_turns (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    user_message TEXT NOT NULL,
                    assistant_response TEXT NOT NULL,
                    tool_calls_json TEXT NOT NULL DEFAULT '[]',
                    agent_trace_json TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chat_turns_session
                ON chat_turns (session_id, created_at)
                """
            )
            await db.commit()

    async def append_turn(
        self,
        *,
        session_id: str,
        user_message: str,
        assistant_response: str,
        tool_calls: list[dict[str, Any]] | None = None,
        agent_trace: list[dict[str, Any]] | None = None,
        turn_id: str | None = None,
    ) -> str:
        """Persist one chat turn and return its id."""
        resolved_id = turn_id or uuid.uuid4().hex
        timestamp = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO chat_turns (
                    id, session_id, user_message, assistant_response,
                    tool_calls_json, agent_trace_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    resolved_id,
                    session_id,
                    user_message,
                    assistant_response,
                    json.dumps(tool_calls or []),
                    json.dumps(agent_trace) if agent_trace is not None else None,
                    timestamp,
                ),
            )
            await db.commit()
        return resolved_id

    async def get_turn(self, turn_id: str) -> ChatTurn | None:
        """Fetch a single chat turn by id."""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            row = await (
                await db.execute("SELECT * FROM chat_turns WHERE id = ?", (turn_id,))
            ).fetchone()
        if row is None:
            return None
        return _row_to_turn(row)

    async def get_recent_history(
        self,
        session_id: str,
        *,
        limit: int = 10,
    ) -> list[dict[str, str]]:
        """Return recent user/assistant messages for agent context."""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            rows = await (
                await db.execute(
                    """
                    SELECT user_message, assistant_response
                    FROM chat_turns
                    WHERE session_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (session_id, limit),
                )
            ).fetchall()

        history: list[dict[str, str]] = []
        for row in reversed(rows):
            history.append({"role": "user", "content": row["user_message"]})
            history.append({"role": "assistant", "content": row["assistant_response"]})
        return history

    async def list_recent_tool_calls(self, *, limit: int = 50) -> list[dict[str, Any]]:
        """Return recent tool call audits for admin analytics."""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            rows = await (
                await db.execute(
                    """
                    SELECT id, session_id, user_message, tool_calls_json, created_at
                    FROM chat_turns
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
            ).fetchall()

        return [
            {
                "message_id": row["id"],
                "session_id": row["session_id"],
                "user_message": row["user_message"],
                "tool_calls": json.loads(row["tool_calls_json"] or "[]"),
                "created_at": row["created_at"],
            }
            for row in rows
        ]


def _row_to_turn(row: aiosqlite.Row) -> ChatTurn:
    return ChatTurn(
        id=row["id"],
        session_id=row["session_id"],
        user_message=row["user_message"],
        assistant_response=row["assistant_response"],
        tool_calls=json.loads(row["tool_calls_json"] or "[]"),
        agent_trace=json.loads(row["agent_trace_json"])
        if row["agent_trace_json"]
        else None,
        created_at=row["created_at"],
    )
