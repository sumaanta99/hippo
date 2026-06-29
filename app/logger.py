"""Structured JSON logging for Hippo Terminal."""

from __future__ import annotations

import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from typing import Any


_CONFIGURED = False
_STRUCTURED_STDERR = False

_QUIET_LOGGER_NAMES = (
    "httpx",
    "httpcore",
    "openai",
    "openai._base_client",
)


def configure_logging(level: str = "WARNING", *, structured: bool = False) -> None:
    """Configure root logging and optional structured event output.

    Args:
        level: Logging level name such as INFO or DEBUG.
        structured: When True, emit JSON events to stderr. Off by default for a
            clean interactive CLI.
    """
    global _CONFIGURED, _STRUCTURED_STDERR
    _STRUCTURED_STDERR = structured

    if not _CONFIGURED:
        logging.basicConfig(
            level=getattr(logging, level.upper(), logging.WARNING),
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
        _CONFIGURED = True
    else:
        logging.getLogger().setLevel(getattr(logging, level.upper(), logging.WARNING))

    for name in _QUIET_LOGGER_NAMES:
        logging.getLogger(name).setLevel(logging.WARNING)


class StructuredLogger:
    """Emit structured JSON log events when enabled."""

    def __init__(self, name: str) -> None:
        """Initialize a structured logger for a module."""
        self._name = name
        self._logger = logging.getLogger(name)

    def log_event(self, event: str, **fields: Any) -> None:
        """Record a structured event (stderr only when structured logging is on).

        Args:
            event: Event name such as intent_classified or response_completed.
            **fields: Additional structured fields for the event payload.
        """
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "logger": self._name,
            "event": event,
            **fields,
        }
        serialized = json.dumps(payload, default=str)
        if _STRUCTURED_STDERR:
            print(serialized, file=sys.stderr, flush=True)
        else:
            self._logger.debug(serialized)

    def info(self, message: str, *args: Any) -> None:
        """Write a standard info log line."""
        self._logger.info(message, *args)

    def warning(self, message: str, *args: Any) -> None:
        """Write a standard warning log line."""
        self._logger.warning(message, *args)

    def error(
        self,
        message: str,
        *,
        error_type: str | None = None,
        recovery_action: str | None = None,
        exc: BaseException | None = None,
        **fields: Any,
    ) -> None:
        """Write an error log line and structured error event.

        Args:
            message: Human-readable error summary.
            error_type: Optional error classification label.
            recovery_action: Optional description of the fallback taken.
            exc: Optional exception used to capture traceback details.
            **fields: Additional structured fields to include in the event.
        """
        self._logger.error(message)
        event_fields: dict[str, Any] = {"message": message, **fields}
        if error_type is not None:
            event_fields["error_type"] = error_type
        if recovery_action is not None:
            event_fields["recovery_action"] = recovery_action
        if exc is not None:
            event_fields["traceback"] = traceback.format_exc()
        self.log_event("error", **event_fields)


def get_logger(name: str) -> StructuredLogger:
    """Return a structured logger for the given module name."""
    return StructuredLogger(name)
