"""dKV-Cache spike — pure one-step decoded-token delay model (no MLX).

The correct absorption of dKV-Cache (sota-kv-cache-management plan, Chunk 2 / Task
4): NOT a relaxation of ``DelayedKVCacheStrategy.is_valid()`` to accept an old
token version, but a write-side LIFECYCLE rule. A token that COMMITS (is decoded)
at step ``t`` may still have its K/V updated during step ``t`` itself, so it only
becomes **cache-eligible at step ``t+1``** — one full step later — and only while
its token id + geometry remain unchanged.

This module is PURE: it models eligibility (which decoded positions a later step
may reuse), with **no MLX import, no cache payloads, and no forward hook**.

**Status (2026-07-02 review finding #7): UNWIRED — the plan-level model, not the
live path.** Task 6 implemented the one-step delay directly inside
``DelayedKVCacheStrategy._serve_dkv_decode`` in the forward's coordinate
(``epoch_drift >= 1``, commit-epochs), NOT by consuming this policy (decode
steps). The two clocks are DIFFERENT; **the strategy's epoch-based gate is the
authoritative one** — this module exists for the deferred plan-level route
(``classify_window(dkv_reuse_positions=...)``, also unwired, mirroring the
layer0/upper_band precedent). If that route is ever wired, reconcile the clocks
first (backlog note in the dKV results doc); do not fix eligibility bugs here
expecting the live path to change.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DecodedTokenRecord:
    """The lifecycle record of one decoded (committed) position.

    ``cache_eligible_step`` = ``decoded_at_step + 1`` (the one-step delay).
    ``token_version`` counts how many times this position's committed token has
    CHANGED (each change restarts the delay), so a stale entry is never confused
    with a fresh one. ``geometry_signature`` is the geometry at decode time; a
    later step under a different geometry is not eligible (the K/V would differ).
    """

    position: int
    token_id: int
    decoded_at_step: int
    cache_eligible_step: int
    token_version: int
    geometry_signature: Any


class OneStepDecodedTokenPolicy:
    """Tracks decoded-token commits and reports which positions a later step may
    reuse under the one-step delay rule.

    Pure + deterministic. ``observe_commit`` is the only writer; ``eligible_positions``
    and ``invalidate_changed_tokens`` are read/prune operations. No cache payloads
    live here — only identity + lifecycle.
    """

    def __init__(self) -> None:
        self._records: dict[int, DecodedTokenRecord] = {}

    def observe_commit(
        self, *, position: int, token_id: int, step: int, geometry_signature: Any
    ) -> None:
        """Record that ``position`` committed ``token_id`` at ``step``.

        A re-commit with the SAME token id + geometry is idempotent (keeps the
        original delay + version — the token is unchanged, already progressing to
        eligibility). A DIFFERENT token id or geometry RESTARTS the one-step delay
        from ``step`` and bumps ``token_version`` (a change invalidates any prior
        eligibility).
        """
        prev = self._records.get(position)
        if (
            prev is not None
            and prev.token_id == token_id
            and prev.geometry_signature == geometry_signature
        ):
            return
        version = (prev.token_version + 1) if prev is not None else 0
        self._records[position] = DecodedTokenRecord(
            position=position,
            token_id=token_id,
            decoded_at_step=step,
            cache_eligible_step=step + 1,
            token_version=version,
            geometry_signature=geometry_signature,
        )

    def eligible_positions(
        self,
        *,
        current_step: int,
        geometry_signature: Any,
        window: tuple[int, int] | None = None,
    ) -> list[int]:
        """The decoded positions reusable at ``current_step`` under the one-step
        delay: ``cache_eligible_step <= current_step`` (the full step has elapsed)
        AND the recorded geometry matches ``geometry_signature`` (unchanged). An
        optional half-open ``window`` ``(lo, hi)`` further limits the candidates to
        decoded positions in ``[lo, hi)``. Sorted for determinism. An un-decoded
        position (no record) is never a candidate.
        """
        out: list[int] = []
        for pos, rec in self._records.items():
            if rec.cache_eligible_step > current_step:
                continue
            if rec.geometry_signature != geometry_signature:
                continue
            if window is not None and not (window[0] <= pos < window[1]):
                continue
            out.append(pos)
        return sorted(out)

    def invalidate_changed_tokens(self, *, current_tokens: dict[int, int]) -> None:
        """Drop any record whose recorded token id no longer matches the current
        token at that position (an explicit batch invalidation for changes the
        decode loop detects outside of ``observe_commit``). Positions absent from
        ``current_tokens`` are left untouched (unknown, not changed)."""
        for pos in list(self._records):
            current = current_tokens.get(pos)
            if current is not None and current != self._records[pos].token_id:
                del self._records[pos]

    def _record(self, position: int) -> DecodedTokenRecord | None:
        """The current :class:`DecodedTokenRecord` for ``position`` (or ``None``)."""
        return self._records.get(position)
