"""Hippo memory engine — interface-agnostic orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.hippo_engine import HippoEngine, HippoEngineError

__all__ = ["HippoEngine", "HippoEngineError"]


def __getattr__(name: str):
    if name in __all__:
        from engine import hippo_engine

        return getattr(hippo_engine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
