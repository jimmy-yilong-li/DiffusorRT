"""Live observation of one decode request: the public names.

```python
from edllm.decode_observation import decode_observation

class Printer:
    def on_begin(self, canvas): print(canvas.length, "positions")
    def on_step(self, step): print([change.text for change in step.committed])
    def on_end(self, end): print("done" if end.completed else "stopped")

engine.generate("hello", observer=Printer())
```

An observer sees what the production decode loop did: the generated canvas
before the first forward, what changed after each commit, and the end of the
request. It cannot change the result, and what it sees is a preview; only the
returned text is the reply. With no observer, the loop does no extra work.

A step is a net change: the canvas at this report compared with the canvas at
the previous one. A position that was remasked and committed again between two
reports appears as one commit, and not at all when it got the same token back.
So a step's `remasked` is not a count of the loop's remask or recovery actions,
and a view built on steps must not claim to show every one of them.

The implementation is in `edllm.backends.mask_canvas`, beside the trace
recorder and inside the producer boundary, because its scope lookup runs on
every request.

author:Opus
"""
from __future__ import annotations

from .backends.mask_canvas import (
    CanvasInfo,
    DecodeEnd,
    DecodeObserver,
    DecodeStep,
    TokenChange,
    decode_observation,
)

__all__ = [
    "CanvasInfo",
    "DecodeEnd",
    "DecodeObserver",
    "DecodeStep",
    "TokenChange",
    "decode_observation",
]
