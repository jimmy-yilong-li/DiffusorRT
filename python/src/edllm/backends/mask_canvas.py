"""MaskCanvas v0 — a public, MLX-free, trace-first view of the dLLM canvas.

Passive observer (spec 2026-06-15-maskcanvas-v0-design.md): it DERIVES an
immutable snapshot + trace from the authoritative MaskState (tokens), the
TokenStateTable (per-position state), and the DecodeEventLog (provenance). It
never mutates decode state and is never a second source of truth.

v0 field-population limits (honest — None/empty until a later phase, NOT silently
populated): ``dependency`` is always None (no graph); ``block_id`` / window data is
None unless the real ``D2FBlockPipelineCommit`` is passed — it is built inside
``run_decode_loop`` and not yet retained on the strategy, so it is None for the
default and D2F engine paths in v0 (the active-window phase wires it through);
per-token ``conf/entropy/margin/stability/kv_version`` are None unless a strategy
emits them; ``cache_action`` is "unknown" until the cache manager exposes per-step
actions; per-step ``remask_set/recommit_set`` are empty — remask provenance is
per-token (``remask_history``). Engine tracing is wired for ``generate`` in v0;
``infill`` tracing is a follow-up (the snapshot/trace derivation is infill-correct
and tested).
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from edllm.backends.token_state import TokenStateTable, snapshot_d2f_blocks

_log = logging.getLogger("edllm.decode_observation")

_SUMMARY_LIMIT = 32


def _maybe_summarize(seq, full: bool):
    """Return seq as a list when full=True or len<=_SUMMARY_LIMIT, else a summary dict."""
    seq = list(seq)
    if full or len(seq) <= _SUMMARY_LIMIT:
        return seq
    return {"len": len(seq), "head": seq[:8], "tail": seq[-8:]}


@dataclass(frozen=True, slots=True)
class TokenDependency:
    """Schema-only in v0 (DEMASK/DAPD reserve). Production values are always None."""
    kind: str = "unknown"          # "unknown" | "local_window" | "attention_graph" | "learned"
    positions: tuple[int, ...] = ()
    source: str = ""
    score: float | None = None


@dataclass(frozen=True, slots=True)
class CanvasTokenView:
    pos: int
    token_id: int
    state: str                      # TokenState.value
    block_id: int | None
    active: bool
    conf: float | None
    entropy: float | None
    margin: float | None
    stability: float | None
    kv_version: int | None
    dependency: TokenDependency | None
    commit_provenance: dict | None  # {"token","score","step"} from DecodeEventLog, else None
    remask_history: tuple[dict, ...]  # derived in Task 3; () in Task 1


@dataclass(frozen=True, slots=True)
class MaskCanvasSnapshot:
    tokens: tuple[int, ...]
    mask_token_id: int
    eos_token_id: int
    prompt_len: int
    fillable_positions: tuple[int, ...]
    remaining_mask_positions: tuple[int, ...]
    tokens_meta: tuple[CanvasTokenView, ...]
    step_index: int
    forward_count: int
    decode_strategy: str
    finish_reason: str | None

    def to_dict(self, *, full: bool = False) -> dict:
        """A JSON-safe dict (tuples -> lists, dataclasses -> dicts, no enums).

        Large position/token arrays are summarized ({"len","head","tail"}) when
        ``full=False`` and the array length exceeds ``_SUMMARY_LIMIT``.
        ``tokens_meta`` is always a full list of per-token dicts (inspection detail).
        Pass ``full=True`` to receive all arrays as plain lists.
        """
        return {
            "tokens": _maybe_summarize(self.tokens, full),
            "mask_token_id": self.mask_token_id,
            "eos_token_id": self.eos_token_id,
            "prompt_len": self.prompt_len,
            "fillable_positions": _maybe_summarize(self.fillable_positions, full),
            "remaining_mask_positions": _maybe_summarize(self.remaining_mask_positions, full),
            "tokens_meta": [asdict(v) for v in self.tokens_meta],
            "step_index": self.step_index,
            "forward_count": self.forward_count,
            "decode_strategy": self.decode_strategy,
            "finish_reason": self.finish_reason,
        }


def _remask_history(event_log: Any) -> dict[int, tuple[dict, ...]]:
    """position -> ordered tuple of remask/recommit event dicts for that position.

    Derived from the event log: remasks() and recommits() are iterated and merged
    per-position, then sorted by (round, remask-before-recommit) so the history
    reads chronologically.  Returns {} when event_log is None.
    """
    if event_log is None:
        return {}
    out: dict[int, list[dict]] = {}
    for e in event_log.remasks():
        out.setdefault(e.position, []).append(
            {"event": "remask", "prev_token": e.prev_token, "round": e.round})
    for e in event_log.recommits():
        out.setdefault(e.position, []).append(
            {"event": "recommit", "new_token": e.new_token, "prev_token": e.prev_token,
             "round": e.round, "correction": e.new_token != e.prev_token})
    # stable order: by round, remask before recommit within a round
    return {p: tuple(sorted(evs, key=lambda d: (d["round"], d["event"] == "recommit")))
            for p, evs in out.items()}


def _commit_provenance(event_log: Any) -> dict[int, dict]:
    """position -> the CURRENT committed {token, score, step}.

    Uses commit_history() (folds Commit + Recommit) so a position re-committed by
    the recovery loop shows its CURRENT token, not the stale original — raw
    commits() would lag after a remask/recommit.
    """
    out: dict[int, dict] = {}
    if event_log is None:
        return out
    for pos, ev in event_log.commit_history().items():  # ev = current Commit(token, score, step)
        out[pos] = {"token": ev.token, "score": ev.score, "step": ev.step}
    return out


def build_snapshot(
    state: Any,
    *,
    event_log: Any = None,
    commit_policy: Any = None,
    decode_strategy: str = "",
    step_index: int = 0,
    forward_count: int = 0,
    finish_reason: str | None = None,
) -> MaskCanvasSnapshot:
    """Derive an immutable canvas snapshot. READ-ONLY: never mutates `state`."""
    table = TokenStateTable()
    blocks = snapshot_d2f_blocks(commit_policy) if commit_policy is not None else None
    remasked = [e.position for e in event_log.remasks()] if event_log is not None else None
    table.sync_from(state, blocks=blocks, remasked_positions=remasked)

    provenance = _commit_provenance(event_log)
    history = _remask_history(event_log)
    meta = tuple(
        CanvasTokenView(
            pos=m.pos, token_id=int(state.x[m.pos]), state=m.state.value,
            block_id=m.block_id, active=m.active, conf=m.conf, entropy=m.entropy,
            margin=m.margin, stability=m.stability, kv_version=m.kv_version,
            dependency=None,                       # honesty: no graph in v0
            commit_provenance=provenance.get(m.pos),
            remask_history=history.get(m.pos, ()),
        )
        for m in (table[i] for i in range(len(table)))
    )
    return MaskCanvasSnapshot(
        tokens=tuple(int(t) for t in state.x),
        mask_token_id=int(state.mask_token_id),
        eos_token_id=int(state.eos_token_id),
        prompt_len=int(state.prompt_len),
        fillable_positions=tuple(state.all_fillable_positions()),
        remaining_mask_positions=tuple(state.masked_positions()),
        tokens_meta=meta,
        step_index=int(step_index),
        forward_count=int(forward_count),
        decode_strategy=str(decode_strategy),
        finish_reason=finish_reason,
    )


# ---------------------------------------------------------------------------
# Trace: the per-forward control-plane capture (scope / frontier / commit set /
# forwarded-token count) recorded by a behavior-neutral observer hooked into the
# decode loop — NOT reverse-engineered from the event log.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CanvasStepTrace:
    step_index: int
    forward_scope: str              # "full" or "window:<end>" — the forwarded region
    forwarded_token_count: int      # rows actually forwarded this step (the FLOPs/token signal)
    frontier_positions: tuple[int, ...]   # the scored/candidate frontier this forward
    commit_set: tuple[int, ...]
    confidence_profile: dict        # {"count","min","mean","max"} over this step's commit scores
    remask_set: tuple[int, ...] = ()
    recommit_set: tuple[int, ...] = ()
    cache_action: str = "unknown"   # best-effort; "unknown" until the cache manager exposes it
    stop_reason: str | None = None

    def to_dict(self, *, full: bool = False) -> dict:
        """A JSON-safe dict for this step (tuples -> lists, large arrays summarized).

        Large position arrays (frontier_positions, commit_set, remask_set,
        recommit_set) are summarized when ``full=False`` and len > ``_SUMMARY_LIMIT``.
        Pass ``full=True`` to receive them as plain lists.
        """
        return {
            "step_index": self.step_index,
            "forward_scope": self.forward_scope,
            "forwarded_token_count": self.forwarded_token_count,
            "frontier_positions": _maybe_summarize(self.frontier_positions, full),
            "commit_set": _maybe_summarize(self.commit_set, full),
            "confidence_profile": dict(self.confidence_profile),
            "remask_set": _maybe_summarize(self.remask_set, full),
            "recommit_set": _maybe_summarize(self.recommit_set, full),
            "cache_action": self.cache_action,
            "stop_reason": self.stop_reason,
        }


@dataclass(frozen=True, slots=True)
class MaskCanvasTrace:
    steps: tuple[CanvasStepTrace, ...]
    final: MaskCanvasSnapshot

    def to_dict(self, *, full: bool = False) -> dict:
        """A JSON-safe dict: every step + the final snapshot, all JSON-native."""
        return {
            "steps": [s.to_dict(full=full) for s in self.steps],
            "final": self.final.to_dict(full=full),
        }


def _profile(scores) -> dict:
    s = [x for x in scores if x == x]  # drop NaN
    if not s:
        return {"count": 0, "min": None, "mean": None, "max": None}
    return {"count": len(s), "min": min(s), "mean": sum(s) / len(s), "max": max(s)}


class CanvasTraceObserver:
    """Behavior-neutral per-forward recorder. The decode loop calls on_forward /
    on_commit as PURE side-effects (the ctx no-ops when no observer is attached),
    so trace ON is byte-identical to trace OFF. Captures the control-plane data
    the event log does NOT have: forwarded scope + frontier + forwarded count."""

    def __init__(self) -> None:
        self._steps: list[dict] = []

    def on_forward(self, *, scope, forwarded_count: int) -> None:
        self._steps.append({
            "forward_scope": "full" if scope is None else f"window:{int(scope)}",
            "forwarded_token_count": int(forwarded_count),
            "frontier_positions": (), "commit_set": (), "scores": (),
        })

    def on_commit(self, *, positions, rows, token_ids, scores, step) -> None:
        if not self._steps:
            return
        cur = self._steps[-1]
        cur["frontier_positions"] = tuple(int(p) for p in positions)
        cur["commit_set"] = tuple(int(positions[int(r)]) for r in rows)
        # ``scores`` mirrors commit_rows' own argument: a per-frontier sequence
        # (often a numpy array) or None. Use the same ``is not None`` guard the
        # loop uses -- a truthiness test on a numpy array raises.
        cur["scores"] = (
            tuple(float(scores[int(r)]) for r in rows) if scores is not None else ()
        )

    def step_traces(self) -> tuple[CanvasStepTrace, ...]:
        return tuple(
            CanvasStepTrace(
                step_index=i, forward_scope=s["forward_scope"],
                forwarded_token_count=s["forwarded_token_count"],
                frontier_positions=s["frontier_positions"], commit_set=s["commit_set"],
                confidence_profile=_profile(s["scores"]),
            )
            for i, s in enumerate(self._steps)
        )


def build_trace(
    state: Any,
    *,
    observer: Any,
    event_log: Any = None,
    commit_policy: Any = None,
    decode_strategy: str = "",
    forward_count: int = 0,
    finish_reason: str | None = None,
) -> MaskCanvasTrace:
    """Per-step trace from the OBSERVER (forwarded scope / frontier / commit set /
    forwarded-token count) + the final snapshot.

    Per-step ``remask_set`` / ``recommit_set`` are intentionally left EMPTY:
    remask/recommit are recovery-ROUND events (they happen BETWEEN forwards, in the
    error-fixing loop), not per-forward, so attributing them to a forward
    ``step_index`` would be wrong (the round counter and the forward index are
    different numbering spaces). The authoritative remask/recommit provenance is
    PER-TOKEN — ``final.tokens_meta[i].remask_history`` — which the recovery loop
    fills correctly via ``DecodeEventLog``. A later phase may add a round-attributed
    recovery view; v0 does not fake a per-step one.
    """
    steps = tuple(observer.step_traces()) if observer is not None else ()
    final = build_snapshot(
        state, event_log=event_log, commit_policy=commit_policy,
        decode_strategy=decode_strategy, step_index=len(steps),
        forward_count=forward_count,
        finish_reason=finish_reason,
    )
    return MaskCanvasTrace(steps=steps, final=final)


class MaskCanvas:
    """Public handle over a finished decode's objects. Snapshots/traces are immutable."""

    def __init__(self, state, *, observer=None, event_log=None, commit_policy=None,
                 decode_strategy=""):
        self._state, self._observer, self._event_log = state, observer, event_log
        self._commit_policy, self._decode_strategy = commit_policy, decode_strategy

    def snapshot(self, **kw) -> MaskCanvasSnapshot:
        return build_snapshot(self._state, event_log=self._event_log,
                              commit_policy=self._commit_policy,
                              decode_strategy=self._decode_strategy, **kw)

    def trace(self, **kw) -> MaskCanvasTrace:
        return build_trace(self._state, observer=self._observer, event_log=self._event_log,
                           commit_policy=self._commit_policy,
                           decode_strategy=self._decode_strategy, **kw)

    def token(self, position: int) -> CanvasTokenView:
        return next(v for v in self.snapshot().tokens_meta if v.pos == position)

# ---------------------------------------------------------------------------
# Live observation of one decode request (author:Opus).
#
# A caller scopes a `DecodeObserver` around a request; `edllm.decode_observation`
# is the public name for that. The decode session picks it up and reports the
# generated canvas before the first forward, what changed after each commit, and
# the end of the request. Every value is read from the session's own MaskState
# after the manager has changed it, through the same trace hooks the recorder
# above uses, so the observer sees what the production loop did.
#
# No observer scoped means no work: the session holds no observation and the
# loop runs exactly as before. An observer can never change decoding. It is
# called only after state changes, and one that raises is dropped for the rest
# of the request while generation continues.
#
# This lives here, inside every producer pathset, because the scope lookup runs
# on every request, observed or not.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CanvasInfo:
    """The generated region of one request, before its first forward."""

    length: int
    """Generated positions. Every index below counts from the first of them."""
    prompt_length: int
    decode_strategy: str
    step_limit: int
    block_spans: tuple[tuple[int, int], ...]
    """Half-open index ranges of the attention blocks; empty without a block layout."""


@dataclass(frozen=True, slots=True)
class TokenChange:
    index: int
    token_id: int
    text: str | None
    """This one token as the tokenizer renders it; None when no tokenizer was given."""
    is_end: bool
    """The end-of-sequence token."""


@dataclass(frozen=True, slots=True)
class DecodeStep:
    """What changed on the canvas since the previous report."""

    step: int | None
    """The decode-loop step that committed; None for changes made outside a step."""
    forward_count: int
    """Real forwards so far in this request."""
    committed: tuple[TokenChange, ...]
    """Positions that now hold a token and did not hold this token before."""
    remasked: tuple[int, ...]
    """Positions that returned to the mask."""
    remaining: int
    """Masked positions left."""
    block_index: int | None
    """The first block that still has a mask; None without blocks or when none is left."""


@dataclass(frozen=True, slots=True)
class DecodeEnd:
    forward_count: int
    remaining: int
    completed: bool
    """False when the request raised: a failure or a cancellation."""


class DecodeObserver(Protocol):
    """Receives one request's canvas, its changes and its end, on the decode thread.

    Keep each call short: it runs between forwards.
    """

    def on_begin(self, canvas: CanvasInfo) -> None: ...

    def on_step(self, step: DecodeStep) -> None: ...

    def on_end(self, end: DecodeEnd) -> None: ...


@dataclass(frozen=True, slots=True)
class _Scope:
    observer: DecodeObserver
    token_text: Callable[[int], str] | None


_current: ContextVar[_Scope | None] = ContextVar("decode_observation", default=None)


@contextmanager
def decode_observation(
    observer: DecodeObserver | None,
    *,
    token_text: Callable[[int], str] | None = None,
) -> Iterator[None]:
    """Observe the decode requests made inside this block, on this thread.

    `token_text` renders one token ID for display. The scope is reset on exit,
    also on failure, so a later request cannot inherit an observer.
    """
    if observer is None:
        yield
        return
    marker = _current.set(_Scope(observer, token_text))
    try:
        yield
    finally:
        _current.reset(marker)


def session_observation(
    *,
    state: Any,
    start: int,
    length: int,
    decode_strategy: str,
    step_limit: int,
    block_layout: Any = None,
) -> "SessionObservation | None":
    """The observation for a new decode session, or None when nobody is observing."""
    scope = _current.get()
    if scope is None:
        return None
    spans: tuple[tuple[int, int], ...] = ()
    if block_layout is not None:
        spans = tuple(
            (first - start, last - start)
            for first, last in (block_layout.span(index) for index in range(block_layout.count))
        )
    return SessionObservation(
        scope,
        state=state,
        start=start,
        canvas=CanvasInfo(
            length=length,
            prompt_length=start,
            decode_strategy=decode_strategy,
            step_limit=step_limit,
            block_spans=spans,
        ),
    )


class SessionObservation:
    """Reports one session's canvas to its observer through the manager's trace hooks.

    `on_forward` and `on_commit` are the hooks the decode loop already calls on
    a trace observer, as pure side effects.
    """

    def __init__(self, scope: _Scope, *, state: Any, start: int, canvas: CanvasInfo) -> None:
        self._observer: DecodeObserver | None = scope.observer
        self._token_text = scope.token_text
        self._state = state
        self._start = start
        self._canvas = canvas
        self._tokens: list[int] = []
        self._forward_count = 0
        self._texts: dict[int, str | None] = {}

    def begin(self) -> None:
        self._safely(self._begin)

    def on_forward(self, *, scope: Any, forwarded_count: int) -> None:
        self._forward_count += 1

    def on_commit(self, *, positions: Sequence[int], rows: Sequence[int],
                  token_ids: Sequence[int], scores: Any, step: int) -> None:
        self._safely(self._report, int(step))

    def end(self, *, completed: bool) -> None:
        self._safely(self._end, completed)

    def _safely(self, work: Callable[..., None], *arguments: Any) -> None:
        """Run observation work; a failure stops the observation, never the request."""
        if self._observer is None:
            return
        try:
            work(*arguments)
        except Exception:
            self._observer = None
            _log.exception("decode observation failed; observation stopped")

    def _begin(self) -> None:
        self._tokens = self._read()
        self._observer.on_begin(self._canvas)

    def _end(self, completed: bool) -> None:
        # Changes made after the last commit (a recovery round) are still reported.
        self._report(None)
        mask = self._state.mask_token_id
        self._observer.on_end(DecodeEnd(
            forward_count=self._forward_count,
            remaining=sum(1 for token in self._tokens if token == mask),
            completed=completed,
        ))

    def _read(self) -> list[int]:
        return list(self._state.x[self._start:self._start + self._canvas.length])

    def _report(self, step: int | None) -> None:
        current = self._read()
        previous = self._tokens
        if current == previous:
            return
        mask = self._state.mask_token_id
        end_token = self._state.eos_token_id
        committed: list[TokenChange] = []
        remasked: list[int] = []
        for index, token in enumerate(current):
            if token == previous[index]:
                continue
            if token == mask:
                remasked.append(index)
            else:
                committed.append(TokenChange(index, token, self._text(token), token == end_token))
        self._tokens = current
        self._observer.on_step(DecodeStep(
            step=step,
            forward_count=self._forward_count,
            committed=tuple(committed),
            remasked=tuple(remasked),
            remaining=sum(1 for token in current if token == mask),
            block_index=self._block_index(current, mask),
        ))

    def _block_index(self, tokens: list[int], mask: int) -> int | None:
        for index, (first, last) in enumerate(self._canvas.block_spans):
            if mask in tokens[first:last]:
                return index
        return None

    def _text(self, token: int) -> str | None:
        if self._token_text is None:
            return None
        if token not in self._texts:
            self._texts[token] = self._token_text(token)
        return self._texts[token]


class TraceFanOut:
    """Calls a session's own trace observer first, then its live observation."""

    __slots__ = ("_first", "_second")

    def __init__(self, first: Any, second: Any) -> None:
        self._first, self._second = first, second

    def on_forward(self, **arguments: Any) -> None:
        self._first.on_forward(**arguments)
        self._second.on_forward(**arguments)

    def on_commit(self, **arguments: Any) -> None:
        self._first.on_commit(**arguments)
        self._second.on_commit(**arguments)
