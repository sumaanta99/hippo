"""CLI entry point — delegates to the terminal client."""

from __future__ import annotations

import asyncio
import sys

from cli.terminal import run_terminal


def main() -> None:
    """Start the Hippo Terminal CLI."""
    try:
        asyncio.run(run_terminal())
    except KeyboardInterrupt:
        print("\nhippo> See you.")
        sys.exit(0)


if __name__ == "__main__":
    main()
