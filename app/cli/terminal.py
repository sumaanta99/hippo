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
FEEDBACK_HELP = "Usage: /feedback good|bad [optional note]"


async def read_input(prompt: str) -> str:
    """Read user input without blocking the event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: input(prompt))


def _parse_feedback_command(command: str) -> tuple[str, str | None] | None:
    """Parse `/feedback good|bad [note]` commands."""
    parts = command.strip().split(maxsplit=2)
    if len(parts) < 2:
        return None
    rating_token = parts[1].lower()
    note = parts[2].strip() if len(parts) == 3 else None
    if rating_token in {"good", "helpful", "+"}:
        return "helpful", note
    if rating_token in {"bad", "not_helpful", "-"}:
        return "not_helpful", note
    return None


async def run_terminal(session_id: str | None = None) -> None:
    """Run the interactive Hippo terminal loop."""
    load_dotenv(APP_DIR.parent / ".env")
    load_dotenv(APP_DIR / ".env")

    engine = HippoEngine()
    await engine.initialize()

    resolved_session = session_id or f"cli-{uuid.uuid4().hex[:8]}"
    last_message_id: str | None = None
    print(WELCOME)
    print(f"{HIPPO_PREFIX}Tip: after a reply, `/feedback good` or `/feedback bad note`")

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

        if command.startswith("/feedback"):
            parsed = _parse_feedback_command(user_message)
            if parsed is None or last_message_id is None:
                print(f"{HIPPO_PREFIX}{FEEDBACK_HELP}")
                continue
            rating, note = parsed
            try:
                feedback_id = await engine.submit_feedback(
                    session_id=resolved_session,
                    message_id=last_message_id,
                    rating=rating,  # type: ignore[arg-type]
                    note=note,
                )
            except HippoEngineError as exc:
                print(f"{HIPPO_PREFIX}{exc}")
                continue
            print(f"{HIPPO_PREFIX}Thanks — feedback recorded ({feedback_id[:8]}).")
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

        last_message_id = result.message_id
        print(f"{HIPPO_PREFIX}{result.response}")

    print(f"{HIPPO_PREFIX}See you.")
