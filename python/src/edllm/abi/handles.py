"""Kinded opaque handles — host-owned objects addressed by uint64.

Kinds exist because backend_runner.py threads FOUR distinct opaque objects with
different lifetimes (weights, kv_cache_state, state_cache_manager, kernels). A
single untyped handle makes a mix-up a wrong pointer; a kinded one makes it a
raise. Tokens are monotonic and never reused: once a compiled consumer holds the
integer, reuse is a use-after-free.
"""
from __future__ import annotations

from enum import IntEnum
from typing import Any

NULL_HANDLE = 0


class HandleKind(IntEnum):
    WEIGHTS = 1
    KV_CACHE_STATE = 2
    STATE_CACHE_MANAGER = 3
    KERNEL_TABLE = 4
    POLICY = 5
    BUFFER = 6
    GRAPH = 7


class HandleRegistry:
    """Session-scoped. Not thread-safe by design: sharing must be explicit."""

    __slots__ = ("_entries", "_next")

    def __init__(self) -> None:
        self._entries: dict[int, tuple[HandleKind, Any]] = {}
        self._next = NULL_HANDLE + 1

    def register(self, obj: Any, *, kind: HandleKind) -> int:
        token = self._next
        self._next += 1
        self._entries[token] = (kind, obj)
        return token

    def resolve(self, token: int, *, kind: HandleKind | None = None) -> Any:
        try:
            actual_kind, obj = self._entries[token]
        except KeyError:
            raise KeyError(
                f"handle {token} is not live (never registered, or released)"
            ) from None
        if kind is not None and actual_kind != kind:
            raise TypeError(
                f"handle {token} has kind {actual_kind.name}, caller expected "
                f"{kind.name}"
            )
        return obj

    def release(self, token: int) -> None:
        try:
            del self._entries[token]
        except KeyError:
            raise KeyError(
                f"handle {token} is not live; double release is a caller bug"
            ) from None

    @property
    def live_count(self) -> int:
        return len(self._entries)
