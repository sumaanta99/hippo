"""One-time duplicate memory cleanup."""

from __future__ import annotations

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from memory import MemoryStore


async def run_cleanup() -> None:
    """Merge duplicate memories in the local database."""
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    load_dotenv(Path(__file__).resolve().parent / ".env")

    store = MemoryStore()
    await store.initialize()

    print("Before cleanup:")
    for memory in await store.list_active():
        print(f"  - {memory.title}: {memory.content}")

    actions = await store.merge_duplicates()
    if not actions:
        print("\nNo duplicate memories found.")
        return

    print("\nMerged:")
    for action in actions:
        print(f"  - {action}")

    print("\nAfter cleanup:")
    for memory in await store.list_active():
        print(f"  - {memory.title}: {memory.content}")


def main() -> None:
    """Run duplicate memory cleanup once."""
    asyncio.run(run_cleanup())


if __name__ == "__main__":
    main()
