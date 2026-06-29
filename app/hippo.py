"""Backward-compatible facade over HippoEngine."""

from __future__ import annotations

from engine.hippo_engine import HippoEngine, HippoEngineError
from models.responses import ChatResponse

HippoError = HippoEngineError


class Hippo:
    """Legacy wrapper — prefer :class:`engine.HippoEngine` for new code."""

    def __init__(self, settings=None) -> None:
        """Initialize the legacy Hippo facade."""
        self._engine = HippoEngine(settings)
        self._settings = self._engine._settings
        self._default_session = self._settings.user_id

    async def initialize(self) -> None:
        """Prepare storage backends."""
        await self._engine.initialize()

    async def process_message(self, message: str) -> str | None:
        """Process a message using the default session id (legacy CLI API)."""
        result = await self._engine.chat(message, self._default_session)
        if not message.strip():
            return None
        return result.response

    async def chat(self, message: str, session_id: str | None = None) -> ChatResponse:
        """Process a message and return a structured response."""
        session = session_id or self._default_session
        return await self._engine.chat(message, session)
