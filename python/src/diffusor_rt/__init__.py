"""Canonical public API for DiffusorRT.

The implementation package remains ``edllm`` during the pre-1.0 compatibility
window. This module exposes the same lazy public API without loading optional
backend dependencies at import time.
"""

from __future__ import annotations

import edllm as _implementation
from typing import Any


__version__ = _implementation.__version__
__all__ = list(_implementation.__all__)


def __getattr__(name: str) -> Any:
    try:
        value = getattr(_implementation, name)
    except AttributeError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
