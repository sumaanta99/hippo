"""Rule-based intent routing to skip the classifier LLM on common messages."""

from __future__ import annotations

import re

from config import Intent

_GREETING_PREFIXES = (
    "hi ",
    "hello ",
    "hey ",
    "good morning",
    "good afternoon",
    "good evening",
    "good night",
)
_GREETING_EXACT = frozenset(
    {
        "hi",
        "hello",
        "hey",
        "yo",
        "thanks",
        "thank you",
        "thx",
        "lol",
        "ok",
        "okay",
        "cool",
        "nice",
    }
)


def _is_greeting(message: str) -> bool:
    lowered = message.lower().strip("!.? ")
    if lowered in _GREETING_EXACT or "how are you" in lowered:
        return True
    return any(lowered.startswith(prefix) for prefix in _GREETING_PREFIXES)

_SHOPPING_SHOW = re.compile(
    r"(?:what(?:'s|\s+is)\s+on\s+(?:my\s+)?(?:shopping\s+)?list|"
    r"show\s+(?:my\s+)?(?:shopping\s+)?list|"
    r"^shopping\s+list\s*\?*$)",
    re.IGNORECASE,
)

_SHOPPING_REMOVE = re.compile(
    r"^(?:remove|drop|take\s+off|bought|got|picked\s+up)(?:\s+the)?\s+\w|"
    r"(?:no\s+more|don't\s+need)\s+\w|"
    r"forget\s+.+\s+from\s+(?:my\s+)?(?:shopping\s+)?list",
    re.IGNORECASE,
)

_SHOPPING_ADD = re.compile(
    r"^(?:buy|need|get|pick\s+up|add)\b|"
    r"\b(?:buy|need|get)\s+",
    re.IGNORECASE,
)

_DELETE = re.compile(
    r"^(?:forget|delete|remove)\s+(?!eggs|milk|bread|butter|cheese|detergent\b)",
    re.IGNORECASE,
)

_UPDATE = re.compile(
    r"\b(?:now\s+in|changed\s+my\s+mind|updated:|is\s+now\s+in|is\s+now\s+on|"
    r"moved\s+(?:it\s+)?to)\b",
    re.IGNORECASE,
)

_QUERY = re.compile(
    r"^(?:where(?:'s|\s+is|\s+are|\s+did)?|what(?:'s|\s+is|\s+are)?|"
    r"who(?:'s|\s+is|\s+are)?|when(?:'s|\s+is|\s+are)?|how\s+(?:do|did|can)\s+i|"
    r"do\s+i\s+have|any\s+(?:upcoming|meetings?))\b|"
    r"\?\s*$",
    re.IGNORECASE,
)

_SAVE = re.compile(
    r"\b(?:is\s+(?:our|my|the|a|an|in|on|at|inside|under)|are\s+(?:in|on|at)|"
    r"remind\s+me|follow\s+up|message\s+\w+|"
    r"send\s+.+\s+by\s+|"
    r"\bby\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|eod|end\s+of\s+day)\b|"
    r"\b(?:in|on|at|inside|under)\s+(?:the\s+)?\w+|"
    r"\b(?:on|in|at|inside|under)\s+\w+|"
    r"\bnumber\b|"
    r"\b\d{8,}\b)",
    re.IGNORECASE,
)


def try_fast_classify(message: str) -> tuple[Intent, float, str] | None:
    """Return (intent, confidence, reasoning) when rules match, else None."""
    cleaned = message.strip()
    if not cleaned:
        return Intent.UNKNOWN, 0.0, "Empty message."

    if _is_greeting(cleaned):
        return _result(Intent.GENERAL_CHAT, "Greeting or casual chat.")

    if _SHOPPING_REMOVE.search(cleaned):
        return _result(Intent.SHOPPING_REMOVE, "Shopping list removal.")

    if _SHOPPING_ADD.search(cleaned):
        return _result(Intent.SHOPPING_ADD, "Shopping list addition.")

    if _SHOPPING_SHOW.search(cleaned):
        return _result(Intent.SHOPPING_SHOW, "Shopping list request.")

    if _DELETE.search(cleaned):
        return _result(Intent.DELETE_MEMORY, "Delete memory request.")

    if _QUERY.search(cleaned) or cleaned.endswith("?"):
        return _result(Intent.QUERY_MEMORY, "Question-shaped memory lookup.")

    if _UPDATE.search(cleaned):
        return _result(Intent.UPDATE_MEMORY, "Memory update request.")

    if _SAVE.search(cleaned):
        return _result(Intent.SAVE_MEMORY, "Declarative save statement.")

    lowered = cleaned.lower()
    tokens = [token for token in re.findall(r"[a-z0-9']+", lowered) if len(token) > 1]
    if 2 <= len(tokens) <= 3 and "?" not in cleaned and not _SHOPPING_ADD.search(cleaned):
        return _result(Intent.QUERY_MEMORY, "Short lookup phrase.")

    return None


def _result(intent: Intent, reasoning: str) -> tuple[Intent, float, str]:
    return intent, 0.95, reasoning
