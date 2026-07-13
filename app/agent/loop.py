"""Async Anthropic agent loop for Hippo."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from agent.executor import ToolEffects, ToolExecutionError, ToolExecutor
from agent.prompts import build_system_prompt
from agent.tools import AGENT_TOOLS
from config import Settings, get_settings
from engine.session import SessionServices


logger = logging.getLogger(__name__)

MAX_ITERATIONS = 5

MessagesCreateFn = Callable[..., Awaitable[Any]]


class AgentLoopError(Exception):
    """Raised when the agent loop cannot complete."""


@dataclass
class AgentRunResult:
    """Result of one agent loop invocation."""

    final_text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    effects: ToolEffects = field(default_factory=ToolEffects)
    iterations: int = 0
    agent_trace: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0


def _extract_text(content: list[Any]) -> str:
    parts: list[str] = []
    for block in content:
        block_type = getattr(block, "type", None) or block.get("type")
        if block_type == "text":
            text = getattr(block, "text", None) or block.get("text", "")
            parts.append(str(text))
    return "\n".join(parts).strip()


def _content_blocks_to_dicts(content: list[Any]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for block in content:
        if hasattr(block, "model_dump"):
            serialized.append(block.model_dump())
        elif isinstance(block, dict):
            serialized.append(block)
        else:
            serialized.append(
                {
                    "type": getattr(block, "type", "unknown"),
                    "text": getattr(block, "text", None),
                    "id": getattr(block, "id", None),
                    "name": getattr(block, "name", None),
                    "input": getattr(block, "input", None),
                }
            )
    return serialized


class AgentLoop:
    """Run a tool-calling loop with the Anthropic Messages API."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        messages_create: MessagesCreateFn | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._messages_create = messages_create

    async def run(
        self,
        *,
        user_message: str,
        services: SessionServices,
        history: list[dict[str, str]] | None = None,
        corrections: list[dict[str, Any]] | None = None,
    ) -> AgentRunResult:
        """Execute the agent loop for one user turn."""
        if self._messages_create is None and not self._settings.anthropic_api_key:
            raise AgentLoopError("ANTHROPIC_API_KEY is not configured.")

        started = time.perf_counter()
        system_prompt = build_system_prompt(corrections)
        messages: list[dict[str, Any]] = []

        for turn in history or []:
            role = turn.get("role")
            content = turn.get("content", "")
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": user_message})

        executor = ToolExecutor(services)
        audit: list[dict[str, Any]] = []
        trace: list[dict[str, Any]] = []
        final_text = ""
        iterations = 0

        for iteration in range(1, MAX_ITERATIONS + 1):
            iterations = iteration
            response = await self._create_messages(
                system=system_prompt,
                messages=messages,
            )
            trace.append(
                {
                    "iteration": iteration,
                    "stop_reason": response.stop_reason,
                    "content": _content_blocks_to_dicts(response.content),
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": _content_blocks_to_dicts(response.content),
                }
            )

            tool_use_blocks = [
                block
                for block in response.content
                if getattr(block, "type", None) == "tool_use"
                or (isinstance(block, dict) and block.get("type") == "tool_use")
            ]

            if response.stop_reason != "tool_use" or not tool_use_blocks:
                final_text = _extract_text(response.content)
                break

            tool_results: list[dict[str, Any]] = []
            for block in tool_use_blocks:
                block_id = getattr(block, "id", None) or block.get("id")
                name = getattr(block, "name", None) or block.get("name")
                tool_input = getattr(block, "input", None) or block.get("input") or {}
                entry: dict[str, Any] = {
                    "iteration": iteration,
                    "tool_use_id": block_id,
                    "name": name,
                    "input": tool_input,
                }
                try:
                    result = await executor.execute(str(name), dict(tool_input))
                    entry["result"] = result.data
                    entry["success"] = result.success
                    if result.error:
                        entry["error_message"] = result.error
                    payload = {
                        "success": result.success,
                        "data": result.data,
                        "error": result.error,
                    }
                except ToolExecutionError as exc:
                    entry["result"] = {"success": False, "error": str(exc)}
                    entry["success"] = False
                    entry["error_message"] = str(exc)
                    payload = {"success": False, "error": str(exc)}

                audit.append(entry)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block_id,
                        "content": json.dumps(payload),
                        **({"is_error": True} if not entry.get("success", True) else {}),
                    }
                )

            messages.append({"role": "user", "content": tool_results})
        else:
            raise AgentLoopError(
                f"Agent exceeded {MAX_ITERATIONS} iterations without a final answer."
            )

        elapsed_ms = (time.perf_counter() - started) * 1000
        return AgentRunResult(
            final_text=final_text,
            tool_calls=audit,
            effects=executor.effects,
            iterations=iterations,
            agent_trace=trace,
            latency_ms=elapsed_ms,
        )

    async def _create_messages(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
    ) -> Any:
        if self._messages_create is not None:
            return await self._messages_create(
                model=self._settings.anthropic_model,
                max_tokens=1024,
                system=system,
                messages=messages,
                tools=AGENT_TOOLS,
            )

        try:
            import anthropic
        except ImportError as exc:
            raise AgentLoopError("anthropic package is not installed.") from exc

        client = anthropic.AsyncAnthropic(api_key=self._settings.anthropic_api_key)
        try:
            return await client.messages.create(
                model=self._settings.anthropic_model,
                max_tokens=1024,
                system=system,
                messages=messages,
                tools=AGENT_TOOLS,
            )
        except anthropic.APIError as exc:
            raise AgentLoopError("Anthropic request failed.") from exc
