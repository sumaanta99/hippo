"""Safe JSON parsing utilities for LLM responses."""

from __future__ import annotations

import json
import re
from typing import Any


_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_object(content: str) -> dict[str, Any]:
    """Parse a JSON object from LLM output, recovering partial payloads when possible.

    Args:
        content: Raw text returned by the language model.

    Returns:
        A parsed JSON object dictionary.

    Raises:
        ValueError: When no valid JSON object can be extracted.
    """
    stripped = content.strip()
    if not stripped:
        raise ValueError("Empty JSON content.")

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = _JSON_OBJECT_PATTERN.search(stripped)
    if match is not None:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    raise ValueError("Unable to parse JSON object from LLM response.")
