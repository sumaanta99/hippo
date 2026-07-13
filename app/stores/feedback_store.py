"""SQLite persistence and retrieval for user feedback."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

import aiosqlite
from pydantic import BaseModel, Field

from config import Settings, get_database_path, get_settings
from embeddings import EmbeddingClient, EmbeddingError, cosine_similarity


FeedbackRating = Literal["helpful", "not_helpful"]


class FeedbackRecord(BaseModel):
    """One feedback submission tied to a chat turn."""

    id: str
    session_id: str
    message_id: str
    rating: FeedbackRating
    note: str | None = None
    user_message: str
    assistant_response: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    created_at: str


class FeedbackStore:
    """Store feedback and retrieve similar correction examples."""

    def __init__(
        self,
        settings: Settings | None = None,
        embedding_client: EmbeddingClient | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._db_path = get_database_path(self._settings)
        self._embeddings = embedding_client or EmbeddingClient(self._settings)

    async def initialize(self) -> None:
        """Create feedback tables if they do not exist."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    rating TEXT NOT NULL,
                    note TEXT,
                    user_message TEXT NOT NULL,
                    assistant_response TEXT NOT NULL,
                    tool_calls_json TEXT NOT NULL DEFAULT '[]',
                    embedding_json TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_feedback_rating
                ON feedback (rating, created_at)
                """
            )
            await db.commit()

    async def record_feedback(
        self,
        *,
        session_id: str,
        message_id: str,
        rating: FeedbackRating,
        note: str | None,
        user_message: str,
        assistant_response: str,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> str:
        """Persist feedback and return its id."""
        feedback_id = uuid.uuid4().hex
        timestamp = datetime.now(timezone.utc).isoformat()
        embedding = await self._embeddings.embed_one(user_message)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO feedback (
                    id, session_id, message_id, rating, note,
                    user_message, assistant_response, tool_calls_json,
                    embedding_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    feedback_id,
                    session_id,
                    message_id,
                    rating,
                    note,
                    user_message,
                    assistant_response,
                    json.dumps(tool_calls or []),
                    json.dumps(embedding),
                    timestamp,
                ),
            )
            await db.commit()
        return feedback_id

    async def get_similar_corrections(
        self,
        query: str,
        *,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """Return top similar not_helpful examples for in-context correction."""
        try:
            query_vector = await self._embeddings.embed_one(query)
        except EmbeddingError:
            return []
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            rows = await (
                await db.execute(
                    """
                    SELECT user_message, tool_calls_json, note, embedding_json
                    FROM feedback
                    WHERE rating = 'not_helpful' AND note IS NOT NULL AND note != ''
                    """
                )
            ).fetchall()

        scored: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            stored = json.loads(row["embedding_json"] or "[]")
            if not stored:
                continue
            score = cosine_similarity(query_vector, stored)
            scored.append(
                (
                    score,
                    {
                        "user_message": row["user_message"],
                        "tool_calls_made": json.loads(row["tool_calls_json"] or "[]"),
                        "note": row["note"],
                        "what_should_have_happened": row["note"],
                    },
                )
            )

        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored[:limit]]

    async def get_summary(self) -> dict[str, Any]:
        """Aggregate feedback counts for admin analytics."""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            rows = await (
                await db.execute(
                    """
                    SELECT rating, COUNT(*) AS count
                    FROM feedback
                    GROUP BY rating
                    """
                )
            ).fetchall()
            recent = await (
                await db.execute(
                    """
                    SELECT id, session_id, message_id, rating, note,
                           user_message, tool_calls_json, created_at
                    FROM feedback
                    ORDER BY created_at DESC
                    LIMIT 20
                    """
                )
            ).fetchall()

        counts = {row["rating"]: int(row["count"]) for row in rows}
        return {
            "helpfulCount": counts.get("helpful", 0),
            "notHelpfulCount": counts.get("not_helpful", 0),
            "totalFeedback": sum(counts.values()),
            "recentFeedback": [
                {
                    "id": row["id"],
                    "session_id": row["session_id"],
                    "message_id": row["message_id"],
                    "rating": row["rating"],
                    "note": row["note"],
                    "user_message": row["user_message"],
                    "tool_calls": json.loads(row["tool_calls_json"] or "[]"),
                    "created_at": row["created_at"],
                }
                for row in recent
            ],
        }
