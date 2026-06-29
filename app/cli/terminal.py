"""Interactive terminal client for Hippo."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from dotenv import load_dotenv

from engine.hippo_engine import HippoEngine, HippoEngineError
from prompts import API_FAILURE_RESPONSE

APP_DIR = Path(__file__).resolve().parent.parent
WELCOME = "Hippo — your external memory. Type naturally. 'quit' to exit."
PROMPT = "you> "
HIPPO_PREFIX = "hippo> "


async def read_input(prompt: str) -> str:
    """Read user input without blocking the event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: input(prompt))


async def run_terminal(session_id: str | None = None) -> None:
    """Run the interactive Hippo terminal loop."""
    load_dotenv(APP_DIR.parent / ".env")
    load_dotenv(APP_DIR / ".env")

    engine = HippoEngine()
    await engine.initialize()

    resolved_session = session_id or f"cli-{uuid.uuid4().hex[:8]}"
    print(WELCOME)

    while True:
        try:
            user_message = await read_input(PROMPT)
        except (EOFError, KeyboardInterrupt):
            print()
            break

        command = user_message.strip().lower()
        if command in {"quit", "exit", "q"}:
            break
        if not user_message.strip():
            continue

        try:
            result = await engine.chat(
                user_message,
                resolved_session,
                on_status=lambda msg: print(f"{HIPPO_PREFIX}{msg}", flush=True),
            )
        except HippoEngineError:
            print(f"{HIPPO_PREFIX}{API_FAILURE_RESPONSE}")
            continue

        if not result.response:
            continue

        print(f"{HIPPO_PREFIX}{result.response}")

    print(f"{HIPPO_PREFIX}See you.")
