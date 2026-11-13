"""Request-local decode progress: the canvas as a reply forms.

The same shape as :mod:`edllm.cancellation`: a caller installs an observer for
one request; a decode loop reports its canvas to it at each step. With no
observer a report is one context-variable read. An observer runs on the
generating thread and must return quickly; if it raises, the failure is logged
and generation continues, because a display must never cost the reply.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

_log = logging.getLogger("edllm.decode_progress")


@dataclass(frozen=True, slots=True)
class CanvasProgress:
    """The generated region after ``step`` of ``steps`` decode steps."""

    step: int
    steps: int
    generated: tuple[int, ...]
    mask_token_id: int
    eos_token_id: int


@dataclass(frozen=True, slots=True)
class ReplyProgress:
    """A forming reply: decoded text runs, and counts of undecided positions.

    ``segments`` alternates ``str`` runs and ``int`` counts in reply order; the
    reply ends at the first decided EOS, as the finished reply does, so
    ``decided`` and ``positions`` can fall when an end is found. The canvas
    counts cover the region the decode will fill: use them, not the step
    budget (which can exceed the steps taken), to show how far along it is.
    ``canvas_decided`` only grows. ``canvas_positions`` is the whole cap,
    except that block-wise BD3 shortens it once an EOS is decided, to the end
    of the block it will stop after, so the share can only jump forward.
    """

    step: int
    steps: int
    segments: tuple[str | int, ...]
    decided: int
    positions: int
    canvas_decided: int = 0
    canvas_positions: int = 0


_observer: ContextVar[Callable[[CanvasProgress], None] | None] = ContextVar(
    "decode_progress", default=None
)


def report_canvas(
    *, step: int, steps: int, generated, mask_token_id: int, eos_token_id: int
) -> None:
    observer = _observer.get()
    if observer is None:
        return
    try:
        observer(CanvasProgress(
            step=step,
            steps=steps,
            generated=tuple(generated),
            mask_token_id=mask_token_id,
            eos_token_id=eos_token_id,
        ))
    except Exception:  # noqa: BLE001 - a display must never cost the reply
        # Said once: reports come once per decode step, and the scope that
        # installed this observer restores what was there before it.
        _log.exception(
            "a decode progress observer failed; generation continues without it"
        )
        _observer.set(None)


@contextmanager
def progress_scope(observer: Callable[[CanvasProgress], None] | None):
    marker = _observer.set(observer) if observer is not None else None
    try:
        yield
    finally:
        if marker is not None:
            _observer.reset(marker)


def reply_progress(
    canvas: CanvasProgress, decode: Callable[[list[int]], str]
) -> ReplyProgress:
    """Turn a canvas into what a reader sees; ``decode`` keeps whitespace."""
    generated = list(canvas.generated)
    canvas_decided = sum(token != canvas.mask_token_id for token in generated)
    canvas_positions = len(generated)
    if canvas.eos_token_id in generated:
        generated = generated[:generated.index(canvas.eos_token_id)]
    segments: list[str | int] = []
    run: list[int] = []
    undecided = 0
    for token in generated:
        if token == canvas.mask_token_id:
            if run:
                segments.append(decode(run))
                run = []
            undecided += 1
        else:
            if undecided:
                segments.append(undecided)
                undecided = 0
            run.append(token)
    if run:
        segments.append(decode(run))
    if undecided:
        segments.append(undecided)
    # The finished reply is stripped; so is the start of a forming one.
    if segments and isinstance(segments[0], str):
        segments[0] = segments[0].lstrip()
    decided = sum(token != canvas.mask_token_id for token in generated)
    return ReplyProgress(
        step=canvas.step,
        steps=canvas.steps,
        segments=tuple(segments),
        decided=decided,
        positions=len(generated),
        canvas_decided=canvas_decided,
        canvas_positions=canvas_positions,
    )
