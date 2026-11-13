"""One reply's live decode, as frames for a window.

Used by both Mac windows: the page reads frames through its server, the native
window through its engine host. The frames are a view of the decode loop's
canvas, never the reply and never history.

author:Opus
"""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from ...decode_observation import CanvasInfo, DecodeEnd, DecodeStep

FRAME_INTERVAL_SECONDS = 1 / 30
"""Frames leave at most this often; steps in between are merged, not dropped."""
# author:Opus -- the interval limits how often frames leave, not how many wait for a
# window that has stopped reading. A caller whose frames queue behind a pipe passes
# `ready`; while it says no, steps keep merging here, where they cost one entry per
# canvas position, and the window gets them as one frame when it reads again.


class DecodeFrames:
    """A `DecodeObserver` that turns steps into JSON frames.

    Called on the decode thread between forwards, so it only merges changes and
    hands at most one small frame per interval to `post`.

    One instance serves one reply. A reply without an output limit is decoded in
    chunks, each reported as its own decode; this joins them into one canvas for
    the window. When a further chunk begins, the window gets a canvas as long as
    all chunks so far, with the text already written filled in, and the chunk's
    positions, steps, blocks and forward counts continue from the earlier ones.
    The engine continues only from a chunk written to its last position, so the
    next chunk starts exactly where that canvas ended.
    """

    def __init__(self, post: Callable[[dict[str, Any]], None], *,
                 interval: float = FRAME_INTERVAL_SECONDS,
                 clock: Callable[[], float] = time.monotonic,
                 ready: Callable[[], bool] | None = None) -> None:
        self._post = post
        self._interval = interval
        self._clock = clock
        self._ready = ready
        self._started = 0.0
        self._last_frame = float("-inf")
        self._committed: dict[int, tuple[int, str | None, bool]] = {}
        self._remasked: set[int] = set()
        self._latest: DecodeStep | None = None
        self._first_step: int | None = None
        # The chunks of this reply decoded before the current one.
        self._began = False
        self._prompt_length = 0
        self._offset = 0
        self._length = 0
        self._blocks: list[list[int]] = []
        self._block_offset = 0
        self._step_limit = 0
        self._steps_before = 0
        self._forwards_before = 0
        self._chunk_steps = 0
        self._chunk_forwards = 0
        self._written: dict[int, tuple[int, str | None, bool]] = {}

    def on_begin(self, canvas: CanvasInfo) -> None:
        continuing = self._began
        if continuing:
            self._offset = self._length
            self._block_offset = len(self._blocks)
            self._steps_before += self._chunk_steps
            self._forwards_before += self._chunk_forwards
        else:
            self._started = self._clock()
            self._prompt_length = canvas.prompt_length
            self._began = True
        self._chunk_steps = 0
        self._chunk_forwards = 0
        self._length = self._offset + canvas.length
        self._blocks += [[start + self._offset, end + self._offset] for start, end in canvas.block_spans]
        self._step_limit += canvas.step_limit
        self._post({
            "type": "DecodeBegan",
            "length": self._length,
            "prompt_length": self._prompt_length,
            "strategy": canvas.decode_strategy,
            "step_limit": self._step_limit,
            "blocks": [list(span) for span in self._blocks],
        })
        if continuing and self._written:
            # The new canvas starts with what the earlier chunks wrote.
            self._post({
                "type": "DecodeFrame",
                "step": self._steps_before - 1,
                "first_step": self._steps_before - 1,
                "forwards": self._forwards_before,
                "remaining": canvas.length,
                "block": self._block_offset,
                "elapsed_ms": self._elapsed_ms(),
                "committed": [
                    [index, token_id, text, is_end]
                    for index, (token_id, text, is_end) in sorted(self._written.items())
                ],
                "remasked": [],
            })

    def on_step(self, step: DecodeStep) -> None:
        offset = self._offset
        for change in step.committed:
            index = change.index + offset
            self._remasked.discard(index)
            self._committed[index] = (change.token_id, change.text, change.is_end)
            self._written[index] = (change.token_id, change.text, change.is_end)
        for raw_index in step.remasked:
            index = raw_index + offset
            self._committed.pop(index, None)
            self._written.pop(index, None)
            self._remasked.add(index)
        if step.step is not None:
            self._chunk_steps = step.step + 1  # steps count from 0
        self._chunk_forwards = step.forward_count
        if self._latest is None:
            self._first_step = step.step
        self._latest = step
        if self._clock() - self._last_frame >= self._interval and (self._ready is None or self._ready()):
            self._flush()

    def on_end(self, end: DecodeEnd) -> None:
        self._flush()
        self._chunk_forwards = end.forward_count
        self._post({
            "type": "DecodeEnded",
            "forwards": self._forwards_before + end.forward_count,
            "remaining": end.remaining,
            "completed": end.completed,
            "elapsed_ms": self._elapsed_ms(),
        })

    def _flush(self) -> None:
        latest = self._latest
        if latest is None:
            return
        self._post({
            "type": "DecodeFrame",
            "step": None if latest.step is None else self._steps_before + latest.step,
            "first_step": None if self._first_step is None else self._steps_before + self._first_step,
            "forwards": self._forwards_before + latest.forward_count,
            "remaining": latest.remaining,
            "block": None if latest.block_index is None else self._block_offset + latest.block_index,
            "elapsed_ms": self._elapsed_ms(),
            "committed": [
                [index, token_id, text, is_end]
                for index, (token_id, text, is_end) in sorted(self._committed.items())
            ],
            "remasked": sorted(self._remasked),
        })
        self._committed.clear()
        self._remasked.clear()
        self._latest = None
        self._last_frame = self._clock()

    def _elapsed_ms(self) -> float:
        return round((self._clock() - self._started) * 1000.0, 1)
