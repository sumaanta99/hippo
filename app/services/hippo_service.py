"""General conversation handler for Hippo."""

from __future__ import annotations

from config import Settings, get_settings
from llm_client import LLMClient, LLMError
from logger import get_logger
from prompts import API_FAILURE_RESPONSE, GENERAL_CHAT_PROMPT
from prompts.safety import wrap_user_content


logger = get_logger(__name__)


class HippoService:
    """Handles casual conversation without touching memory storage."""

    def __init__(
        self,
        llm: LLMClient,
        settings: Settings | None = None,
    ) -> None:
        """Initialize the Hippo conversation service."""
        self._llm = llm
        self._settings = settings or get_settings()

    async def handle_general_chat(self, user_input: str) -> str:
        """Respond warmly to greetings and chit-chat without saving anything."""
        message = user_input.strip()
        prompt = GENERAL_CHAT_PROMPT.format(message=wrap_user_content(message))
        try:
            response = await self._llm.complete_text(prompt, temperature=0.5)
        except LLMError as exc:
            logger.warning("General chat failed: %s", exc)
            return API_FAILURE_RESPONSE

        logger.log_event("general_chat_completed")
        return response
