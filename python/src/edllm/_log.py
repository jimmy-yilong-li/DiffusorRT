"""Logging configuration for DiffusorRT. (author:GLM)

Priority: explicit ``level`` > ``DIFFUSORRT_LOG_LEVEL`` env > ``default``.

- The library never configures anything. An embedding application that does
  nothing sees WARNING and above through Python's ``lastResort`` handler, and
  one that calls ``configure_logging()`` gets the env var with a WARNING
  default.
- The CLI passes ``default="INFO"`` so progress and stage events are visible
  without a flag, and ``"DEBUG"`` when ``-v`` is given.
- The handler is attached to the ``edllm`` logger only, with
  ``propagate=False`` — a host application with its own ``basicConfig`` does
  not see every line twice.
- The rendered form is a contract, not decoration: ``pull`` and ``compile``
  emit their stage events through it and ``scripts/installed_wheel_gate.py``
  parses them. ``tests/cli/test_log_contract.py`` pins both the format and
  the CLI's default level.

The logger namespace is ``edllm.*`` — the Python package name, not the
console script name.
"""
from __future__ import annotations

import logging
import os
import sys

_FORMAT = "diffusorrt: %(asctime)s [%(levelname)s] %(message)s"
_DATEFMT = "%H:%M:%S"


class _StderrHandler(logging.StreamHandler):
    """Resolve ``sys.stderr`` at emit time, not at construction.

    ``logging.StreamHandler`` captures the stream object it is given. Logging
    is configured once per process, so anything that replaces ``sys.stderr``
    afterwards -- ``contextlib.redirect_stderr``, a supervisor that rebinds it,
    pytest's per-test capture -- leaves the handler writing to a stream nobody
    reads, and the output simply disappears.
    """

    def __init__(self) -> None:
        super().__init__(sys.stderr)

    @property
    def stream(self):
        return sys.stderr

    @stream.setter
    def stream(self, value) -> None:
        # StreamHandler.__init__ and setStream() both assign here; the whole
        # point of this class is that the stream is not remembered.
        pass

_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def configure_logging(level: str | None = None, *, default: str = "WARNING") -> None:
    """Configure the ``edllm`` logger.

    Priority: explicit ``level`` > ``DIFFUSORRT_LOG_LEVEL`` env > ``default``.
    Idempotent: updates the level without duplicating handlers. An unknown
    level raises rather than silently degrading — a typo in the env var would
    otherwise hide output that the caller asked for.
    """
    source = "level"
    if level is None:
        level = os.environ.get("DIFFUSORRT_LOG_LEVEL")
        source = "DIFFUSORRT_LOG_LEVEL"
    if level is None:
        level, source = default, "default"

    upper = level.upper()
    if upper not in _LEVELS:
        raise ValueError(
            f"{source}={level!r} is not a valid level; "
            f"expected one of {', '.join(sorted(_LEVELS))}"
        )

    log = logging.getLogger("edllm")
    log.setLevel(_LEVELS[upper])
    if not log.handlers:
        handler = _StderrHandler()
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        log.addHandler(handler)
    log.propagate = False
