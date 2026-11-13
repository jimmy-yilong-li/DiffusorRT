"""Token-State IR (Commit 2 of the active-window runtime plan).

This module makes the per-token decode state EXPLICIT. Today that state is
scattered: MASKED vs COMMITTED live implicitly in ``MaskState.x`` (a position is
masked iff ``x[pos] == mask_token_id``), while FROZEN (a block that left the D2F
pipeline's sliding window) and the active window live privately inside the
``d2f_block_pipeline`` scheduler (``D2FBlockPipelineCommit._blocks`` /
``_appended_count``). The :class:`TokenStateTable` is one indexed-by-position
table over the whole canvas that DERIVES all of it in one place — the data model
Commit 3's active-window executor will read to select the live frontier.

**Strictly additive + observational.** The table is built ON DEMAND from the
existing decode objects (:func:`TokenStateTable.from_mask_state`, plus
:func:`snapshot_d2f_blocks` for the scheduler's block/freeze view) and only ever
READS them: it never commits, never remasks, never forwards, and never alters a
scheduler decision. Nothing in the decode hot path calls into this module —
the byte-exact decode gate is the proof.

**Derivation precedence** (highest first, evaluated per position):

1. **Fixed context** — a position outside ``MaskState.all_fillable_positions()``
   (the prompt in trailing generation; prefix AND suffix in infill) is
   ``COMMITTED``, even when its token value happens to equal the mask id.
2. **Canvas-masked** (fillable and ``x[pos] == mask_token_id`` — the canvas is
   ground truth, so a masked position is NEVER ``FROZEN``, whatever block label
   it carries): ``REMASKED`` (known re-masked after having been filled) beats
   ``CANDIDATE`` (inside the active window) beats ``MASKED``.
3. **Filled** (fillable and ``x[pos] != mask_token_id``): ``FROZEN`` (inside a
   block the SCHEDULER has marked frozen — the table never infers freezing
   itself) beats ``COMMITTED``.

``DIRTY`` is part of the vocabulary but is NEVER derived here: Commit 3's
dirty-neighborhood invalidation is what sets it. :meth:`TokenStateTable.
dirty_neighborhood` ships now as the pure index math that invalidation will use.

Like the rest of the diffusion runtime layer this module is model-agnostic and
import-light: stdlib only — **no mlx, no numpy** (pinned by test).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique
from typing import Any, Iterable, Sequence


@unique
class TokenState(Enum):
    """The explicit per-token decode states (plan: the Token-State IR)."""

    MASKED = "masked"  # fillable, still mask on the canvas, outside the window
    CANDIDATE = "candidate"  # fillable + masked + inside the active window
    COMMITTED = "committed"  # filled (or fixed context: prompt / infill suffix)
    FROZEN = "frozen"  # filled AND its block left the D2F window (KV reusable)
    REMASKED = "remasked"  # re-opened after having been filled (recovery)
    DIRTY = "dirty"  # reserved: Commit 3 invalidation marks stale neighbors


@dataclass(slots=True)
class TokenMeta:
    """Per-position metadata row of the :class:`TokenStateTable`.

    Commit 2 populates ``pos`` / ``state`` / ``block_id`` / ``active`` only.
    The remaining fields are RESERVED for Commit 3 (the commit-policy scores and
    the versioned-KV bookkeeping) and stay ``None`` until that slice wires them.
    """

    pos: int
    state: TokenState
    block_id: int | None = None
    active: bool = False
    # -- reserved for Commit 3 (never set by sync_from today) ------------------
    conf: float | None = None
    entropy: float | None = None
    margin: float | None = None
    stability: float | None = None
    kv_version: int | None = None


@dataclass(frozen=True, slots=True)
class BlockInfo:
    """An immutable snapshot of one scheduler block (absolute positions).

    ``in_window`` mirrors the D2F scheduler's "appended" notion: the block has
    joined the sliding decode window. ``frozen`` mirrors the scheduler's freeze
    decision (the block fully decoded and left the window oldest-first). A block
    that is ``in_window`` and not ``frozen`` is ACTIVE — its span forms the
    active window the table derives when no explicit one is given.
    """

    block_id: int
    start: int
    end: int  # half-open
    frozen: bool = False
    in_window: bool = True


def snapshot_d2f_blocks(commit_policy: Any) -> list[BlockInfo]:
    """READ-ONLY adapter: the D2F pipeline scheduler's block/freeze state.

    Reads ``D2FBlockPipelineCommit``'s private snapshot (``_blocks`` — the
    absolute-position block partition with its sticky ``frozen`` flags — and
    ``_appended_count`` — how many blocks have joined the window) and returns
    plain :class:`BlockInfo` rows. Duck-typed on purpose so this module imports
    nothing from ``diffusion_runtime``; it NEVER mutates the policy and never
    calls ``plan``/``select`` (those advance the scheduler's lifecycle). Before
    the first ``plan`` the partition does not exist yet -> ``[]``. Commit 3's
    executor, which owns the commit policy directly, can replace this peek with
    a first-class accessor when the table starts driving the forward.
    """
    raw_blocks = getattr(commit_policy, "_blocks", None)
    if not raw_blocks:
        return []
    appended = int(getattr(commit_policy, "_appended_count", 0) or 0)
    return [
        BlockInfo(
            block_id=index,
            start=int(block.start),
            end=int(block.end),
            frozen=bool(block.frozen),
            in_window=index < appended,
        )
        for index, block in enumerate(raw_blocks)
    ]


class TokenStateTable:
    """The explicit per-token state table, indexed by position over the canvas.

    Built/refreshed via :meth:`sync_from` (or the :meth:`from_mask_state`
    convenience): every position's :class:`TokenMeta` is DERIVED from the given
    ``MaskState`` (duck-typed: ``x`` / ``mask_token_id`` /
    ``all_fillable_positions()``) plus the optional scheduler view (``blocks``
    from :func:`snapshot_d2f_blocks`, an explicit ``active_positions`` override,
    and ``remasked_positions`` evidence). Re-syncing the same table also tracks
    commit->remask transitions across syncs: a position seen filled by an
    earlier sync that is masked again now derives ``REMASKED`` (sticky until it
    is filled again). Sync never mutates its inputs.
    """

    __slots__ = ("_meta", "_remasked_memory")

    def __init__(self) -> None:
        self._meta: list[TokenMeta] = []
        self._remasked_memory: set[int] = set()

    @classmethod
    def from_mask_state(
        cls,
        state: Any,
        *,
        blocks: Sequence[BlockInfo] | None = None,
        active_positions: Iterable[int] | None = None,
        remasked_positions: Iterable[int] | None = None,
    ) -> "TokenStateTable":
        """Build a fresh table from a ``MaskState`` (the on-demand entry point)."""
        table = cls()
        table.sync_from(
            state,
            blocks=blocks,
            active_positions=active_positions,
            remasked_positions=remasked_positions,
        )
        return table

    # -- container surface ---------------------------------------------------- #

    def __len__(self) -> int:
        return len(self._meta)

    def __getitem__(self, pos: int) -> TokenMeta:
        return self._meta[pos]

    def reset(self) -> None:
        """Drop ALL sync history (the explicit cross-generation boundary).

        The table tracks ONE generation's remask memory and previous-state
        comparison; reusing it for a new generation must go through reset (or a
        fresh table), otherwise gen-N history leaks into gen-N+1 states whenever
        the two canvases share a length.
        """
        self._meta = []
        self._remasked_memory = set()

    # -- derivation ------------------------------------------------------------ #

    def sync_from(
        self,
        state: Any,
        *,
        blocks: Sequence[BlockInfo] | None = None,
        active_positions: Iterable[int] | None = None,
        remasked_positions: Iterable[int] | None = None,
    ) -> None:
        """Derive every position's state from the given decode objects (READ-ONLY).

        ``blocks`` (e.g. :func:`snapshot_d2f_blocks`) contributes ``block_id``,
        the frozen spans, and — when ``active_positions`` is not given — the
        derived active window (the spans of in-window, unfrozen blocks).
        ``active_positions`` overrides that derivation explicitly.
        ``remasked_positions`` is caller-supplied evidence (e.g. from
        ``DecodeEventLog.remasks()``) that a position was re-opened after a
        fill; the table also detects this on its own across successive syncs.
        Precedence is pinned in the module docstring. Inputs are validated
        fail-closed; nothing is mutated.
        """
        canvas_len = len(state.x)
        mask_id = state.mask_token_id
        fillable = set(state.all_fillable_positions())

        block_by_pos, frozen_spans = self._index_blocks(blocks, canvas_len=canvas_len)
        active = self._resolve_active(
            active_positions, blocks=blocks, canvas_len=canvas_len
        )
        explicit_remasked = self._validated_positions(
            remasked_positions, canvas_len=canvas_len, label="remasked_positions"
        )

        if len(self._meta) != canvas_len:
            # A different canvas LENGTH is provably a different generation: drop
            # sync history. A SAME-length new generation is indistinguishable
            # from a mass-remask by content alone — the table tracks ONE
            # generation and cross-generation reuse must call :meth:`reset`
            # (or build a fresh table, as the in-repo canvas surfaces do).
            self._remasked_memory = set()
        previous = self._meta if len(self._meta) == canvas_len else []

        filled_states = (TokenState.COMMITTED, TokenState.FROZEN)
        remasked = self._remasked_memory
        remasked |= explicit_remasked
        meta: list[TokenMeta] = []
        for pos in range(canvas_len):
            is_masked = pos in fillable and state.x[pos] == mask_id
            if is_masked and previous and previous[pos].state in filled_states:
                remasked.add(pos)  # seen filled by an earlier sync, masked now
            if not is_masked:
                remasked.discard(pos)  # (re)filled: the REMASKED chapter is over

            if pos not in fillable:
                token_state = TokenState.COMMITTED  # rule 1: fixed context
            elif is_masked:  # rule 2: the canvas is ground truth
                if pos in remasked:
                    token_state = TokenState.REMASKED
                elif pos in active:
                    token_state = TokenState.CANDIDATE
                else:
                    token_state = TokenState.MASKED
            else:  # rule 3: filled
                if pos in frozen_spans:
                    token_state = TokenState.FROZEN
                else:
                    token_state = TokenState.COMMITTED

            meta.append(
                TokenMeta(
                    pos=pos,
                    state=token_state,
                    block_id=block_by_pos.get(pos),
                    active=pos in active,
                )
            )
        self._meta = meta

    # -- queries (what Commit 3's executor reads) ------------------------------- #

    def masked_positions(self) -> list[int]:
        """Every position still to fill (MASKED | CANDIDATE | REMASKED).

        Mirrors ``MaskState.masked_positions()`` exactly: the canvas-truth
        complement of :meth:`committed_positions`.
        """
        still_masked = (TokenState.MASKED, TokenState.CANDIDATE, TokenState.REMASKED)
        return [meta.pos for meta in self._meta if meta.state in still_masked]

    def candidate_positions(self) -> list[int]:
        """Masked positions inside the active window — the live frontier."""
        return self._positions_in(TokenState.CANDIDATE)

    def committed_positions(self) -> list[int]:
        """Every filled position (COMMITTED | FROZEN; fixed context included)."""
        filled = (TokenState.COMMITTED, TokenState.FROZEN)
        return [meta.pos for meta in self._meta if meta.state in filled]

    def frozen_positions(self) -> list[int]:
        """Filled positions whose block left the window (their KV is reusable)."""
        return self._positions_in(TokenState.FROZEN)

    def remasked_positions(self) -> list[int]:
        """Positions re-opened after a fill, still masked now."""
        return self._positions_in(TokenState.REMASKED)

    def dirty_positions(self) -> list[int]:
        """Positions marked DIRTY. Always empty in Commit 2 (Commit 3 sets it)."""
        return self._positions_in(TokenState.DIRTY)

    def active_positions(self) -> list[int]:
        """The active-window positions (the ``active`` flag, any state)."""
        return [meta.pos for meta in self._meta if meta.active]

    def dirty_neighborhood(self, pos: int, *, radius: int) -> list[int]:
        """Positions within ``radius`` of ``pos``, clipped to the canvas.

        The local invalidation window Commit 3's dirty-neighborhood pass opens
        around a remask/revise — pure index math today (no dependency graph):
        ``[max(0, pos-radius), min(canvas, pos+radius+1))``, ``pos`` included.
        """
        canvas_len = len(self._meta)
        if not 0 <= int(pos) < canvas_len:
            raise ValueError(
                f"dirty_neighborhood pos must be within the canvas [0, {canvas_len}), "
                f"got {pos}"
            )
        if int(radius) < 0:
            raise ValueError(f"dirty_neighborhood radius must be >= 0, got {radius}")
        pos = int(pos)
        radius = int(radius)
        return list(range(max(0, pos - radius), min(canvas_len, pos + radius + 1)))

    # -- internals --------------------------------------------------------------- #

    def _positions_in(self, token_state: TokenState) -> list[int]:
        return [meta.pos for meta in self._meta if meta.state is token_state]

    @staticmethod
    def _index_blocks(
        blocks: Sequence[BlockInfo] | None, *, canvas_len: int
    ) -> tuple[dict[int, int], set[int]]:
        """Validate the block partition; map position -> block_id + frozen spans."""
        block_by_pos: dict[int, int] = {}
        frozen_spans: set[int] = set()
        for block in blocks or ():
            if not (0 <= int(block.start) < int(block.end) <= canvas_len):
                raise ValueError(
                    f"block {block.block_id} span [{block.start}, {block.end}) is "
                    f"not within the canvas [0, {canvas_len})"
                )
            for pos in range(int(block.start), int(block.end)):
                if pos in block_by_pos:
                    raise ValueError(
                        f"blocks overlap at position {pos} "
                        f"(blocks {block_by_pos[pos]} and {block.block_id})"
                    )
                block_by_pos[pos] = int(block.block_id)
                if block.frozen:
                    frozen_spans.add(pos)
        return block_by_pos, frozen_spans

    @staticmethod
    def _resolve_active(
        active_positions: Iterable[int] | None,
        *,
        blocks: Sequence[BlockInfo] | None,
        canvas_len: int,
    ) -> set[int]:
        """The effective active window: the explicit set, else derived from blocks."""
        if active_positions is not None:
            return TokenStateTable._validated_positions(
                active_positions, canvas_len=canvas_len, label="active_positions"
            )
        return {
            pos
            for block in blocks or ()
            if block.in_window and not block.frozen
            for pos in range(int(block.start), int(block.end))
        }

    @staticmethod
    def _validated_positions(
        positions: Iterable[int] | None, *, canvas_len: int, label: str
    ) -> set[int]:
        validated: set[int] = set()
        for pos in positions or ():
            if not 0 <= int(pos) < canvas_len:
                raise ValueError(
                    f"{label} entry {pos} is outside the canvas [0, {canvas_len})"
                )
            validated.add(int(pos))
        return validated
