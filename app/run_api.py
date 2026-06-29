"""Run the Hippo FastAPI server."""

from __future__ import annotations

import os
from pathlib import Path

import uvicorn

APP_DIR = Path(__file__).resolve().parent


def main() -> None:
    """Start uvicorn with the Hippo API app."""
    os.chdir(APP_DIR)
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "true").lower() in {"1", "true", "yes"}
    uvicorn.run(
        "api.server:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    main()
