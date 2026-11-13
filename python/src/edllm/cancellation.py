"""Request-local cooperative cancellation; never interrupts a device kernel.

The caller signals from another thread. The executing thread observes the token
at a shared decode boundary and unwinds its existing resource owners normally.
Context is scoped/reset even on failure; a later request cannot inherit a token.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Event


class GenerationCancelled(RuntimeError):
    """Generation stopped at a cooperative boundary; no reply was committed."""


class CancellationToken:
    def __init__(self):
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


_current: ContextVar[CancellationToken | None] = ContextVar("generation_cancel", default=None)


def check_cancelled() -> None:
    token = _current.get()
    if token is not None and token.cancelled:
        raise GenerationCancelled("generation cancelled at a safe execution boundary")


@contextmanager
def cancellation_scope(token: CancellationToken | None):
    marker = _current.set(token) if token is not None else None
    try:
        check_cancelled()
        yield
        check_cancelled()
    finally:
        if marker is not None:
            _current.reset(marker)
