"""Intent classification using the LLM."""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field

from config import Intent, Settings, get_settings
from llm_client import LLMClient, LLMError
from logger import get_logger
from prompts.classification import CLASSIFICATION_PROMPT, CLASSIFICATION_SYSTEM
from prompts.safety import wrap_user_content


logger = get_logger(__name__)


class ClassificationError(Exception):
    """Raised when intent classification fails."""


class ClassificationResult(BaseModel):
    """Structured output from intent classification."""

    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""


class IntentClassifier:
    """Detect user intent from natural language input."""

    def __init__(
        self,
        llm: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize the classifier."""
        self._settings = settings or get_settings()
        self._llm = llm or LLMClient(self._settings)

    async def classify_intent(
        self,
        user_input: str,
        user_id: str,
    ) -> ClassificationResult:
        """Classify user input into one of nine supported intents.

        Args:
            user_input: Raw user message text.
            user_id: Active user identifier.

        Returns:
            Structured classification result with intent and confidence.

        Raises:
            ClassificationError: When the LLM request fails.
        """
        _ = user_id
        message = user_input.strip()
        if not message:
            return ClassificationResult(
                intent=Intent.UNKNOWN,
                confidence=0.0,
                reasoning="Empty message.",
            )

        from fast_router import try_fast_classify

        fast = try_fast_classify(message)
        if fast is not None:
            intent, confidence, reasoning = fast
            result = ClassificationResult(
                intent=intent,
                confidence=confidence,
                reasoning=reasoning,
            )
            logger.log_event(
                "intent_classified",
                intent=result.intent.value,
                confidence=result.confidence,
                latency_ms=0.0,
                reasoning=result.reasoning,
                fast_path=True,
            )
            return result

        prompt = CLASSIFICATION_PROMPT.format(message=wrap_user_content(message))
        started = time.perf_counter()

        try:
            payload = await self._llm.complete_json(
                prompt,
                system=CLASSIFICATION_SYSTEM,
                max_tokens=120,
            )
        except LLMError as exc:
            logger.error(
                "Classification failed.",
                error_type="LLMError",
                recovery_action="raise ClassificationError",
                exc=exc,
            )
            raise ClassificationError(str(exc)) from exc

        elapsed_ms = (time.perf_counter() - started) * 1000
        result = _parse_classification(payload)
        logger.log_event(
            "intent_classified",
            intent=result.intent.value,
            confidence=result.confidence,
            latency_ms=round(elapsed_ms, 2),
            reasoning=result.reasoning,
        )
        return result

    async def classify(self, message: str) -> dict[str, Any]:
        """Classify a message and return a plain dictionary for callers and tests.

        Args:
            message: Raw user message text.

        Returns:
            Dictionary with intent, confidence, and reasoning keys.
        """
        result = await self.classify_intent(message, self._settings.user_id)
        return {
            "intent": result.intent.value,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
        }


def _parse_classification(payload: dict[str, Any]) -> ClassificationResult:
    """Parse and validate a classification response from the LLM."""
    raw_intent = str(payload.get("intent", Intent.UNKNOWN.value)).strip().upper()
    try:
        intent = Intent(raw_intent)
    except ValueError:
        intent = Intent.UNKNOWN

    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))
    reasoning = str(payload.get("reasoning", "")).strip()

    if intent == Intent.UNKNOWN and confidence >= 0.5:
        confidence = min(confidence, 0.49)

    return ClassificationResult(
        intent=intent,
        confidence=confidence,
        reasoning=reasoning,
    )
