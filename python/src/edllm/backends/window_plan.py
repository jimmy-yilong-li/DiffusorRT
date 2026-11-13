"""WindowExecutionPlan — MLX-free execution plan for the active-window runtime.

Pure data model (Task A1, WS1, spec 2026-06-15-active-window-runtime-v1.md):
carries the per-step active/dirty/remask/frozen/dependency-risk classification
together with the ``forward_scope`` (positions to forward as query rows) and
per-position ``cache_action``.  No MLX, no torch, no numpy — import-light
(mirrors ``mask_canvas.py``).

Three-track cache semantics:
  RECOMPUTE     — query row is included; K/V not reused from a prior step.
  REUSE_EXACT   — K/V reused from a prior step under a byte-exact guarantee
                  (block-causal committed blocks only).
  REUSE_APPROX  — K/V reused under an approximation (aggressive window;
                  only emitted when an explicit policy enables it).

The exact path (default) emits only RECOMPUTE / REUSE_EXACT; REUSE_APPROX is
gated behind an explicit aggressive policy (WS3) that also enables error-fixing.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

_SUMMARY_LIMIT = 32


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TokenAction(Enum):
    """Per-position classification of a token on the canvas for this step."""
    ACTIVE = "active"               # masked; being decoded this step
    DIRTY = "dirty"                 # filled but invalidated; needs re-forward
    REMASK = "remask"               # previously committed; re-opened for correction
    FROZEN = "frozen"               # committed / fixed context; not a query row
    DEPENDENCY_RISK = "dependency_risk"  # active but flagged for attention-graph risk


class CacheAction(Enum):
    """K/V cache treatment for a position that appears in forward_scope."""
    RECOMPUTE = "recompute"         # recompute K/V this step (default safe path)
    REUSE_EXACT = "reuse_exact"     # reuse prior K/V (byte-exact; block-causal only)
    REUSE_APPROX = "reuse_approx"   # reuse prior K/V (approximate; aggressive policy)


# ---------------------------------------------------------------------------
# Summarization helper (mirrors mask_canvas._maybe_summarize)
# ---------------------------------------------------------------------------


def _maybe_summarize(seq, full: bool):
    """Return seq as a list when full=True or len<=_SUMMARY_LIMIT, else a summary dict."""
    seq = list(seq)
    if full or len(seq) <= _SUMMARY_LIMIT:
        return seq
    return {"len": len(seq), "head": seq[:8], "tail": seq[-8:]}


# ---------------------------------------------------------------------------
# WindowExecutionPlan
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WindowExecutionPlan:
    """Immutable per-step execution plan for the active-window runtime.

    Fields
    ------
    step : int
        Decode step index (0-based).
    active_positions : tuple[int, ...]
        Positions classified ACTIVE (currently masked; being decoded).
    dirty_positions : tuple[int, ...]
        Positions classified DIRTY (filled but invalidated).
    remask_positions : tuple[int, ...]
        Positions classified REMASK (re-opened committed positions).
    frozen_positions : tuple[int, ...]
        Positions classified FROZEN (committed / fixed context; not query rows).
    dependency_risk_positions : tuple[int, ...]
        Positions flagged DEPENDENCY_RISK (active but attention-graph risky).
    cache_action : dict[int, CacheAction]
        Per-position K/V cache treatment for every position in ``forward_scope``.
    forward_scope : tuple[int, ...]
        Positions forwarded as query rows this step.  Must be a subset of
        ``active ∪ dirty ∪ remask``.  The length of this tuple is the
        ``forwarded_token_count`` (the honest FLOP signal).
    reuse_source : dict[int, str]
        Provenance tag for each position whose ``cache_action == REUSE_EXACT``
        (C2b-2, WS3).  Valid tag values:

        * ``"frozen_block"`` — frozen/committed block reuse (the pre-C2b path).
        * ``"delayed_kv_layer0_exact"`` — layer-0 exact stale-KV reuse (C2b-v0).

        Positions with ``RECOMPUTE`` or absent from this dict have no reuse
        provenance.  Default ``{}`` (empty) — byte-neutral with all pre-C2b callers
        (frozen-block positions that are not in ``forward_scope``/``cache_action``
        do not need a reuse_source entry; this field covers only active positions
        in ``forward_scope`` whose layer-0 K/V is reused, distinguishing the two
        exact sources for metrics attribution).

    Invariants (enforced in ``__post_init__``)
    ------------------------------------------
    1. ``active``, ``dirty``, ``remask``, ``frozen`` are PAIRWISE DISJOINT.
       ``dependency_risk_positions`` is exempt — it is a flag on active positions
       (``dependency_risk ⊆ active`` is allowed and expected).
    2. ``forward_scope ⊆ active ∪ dirty ∪ remask``.
    3. Every position in ``forward_scope`` has an entry in ``cache_action``.
    """

    step: int
    active_positions: tuple[int, ...]
    dirty_positions: tuple[int, ...]
    remask_positions: tuple[int, ...]
    frozen_positions: tuple[int, ...]
    dependency_risk_positions: tuple[int, ...]
    cache_action: dict[int, CacheAction]
    forward_scope: tuple[int, ...]
    # Forward/context boundary (B2.5): the commit-policy ``StepPlan.window_end``.
    # The query rows (``forward_scope``) attend to K/V over positions
    # ``[0, window_end)``; the real windowed forward (B3) reads THIS explicit
    # boundary instead of guessing ``max(forward_scope) + 1``. ``None`` => the full
    # working sequence is the context (no window truncation this step).
    window_end: int | None = None
    # C2b-2: per-position reuse provenance for REUSE_EXACT positions in
    # forward_scope.  Empty dict by default — byte-neutral with all pre-C2b callers.
    # Uses None as a sentinel; __post_init__ replaces None with {}.
    reuse_source: dict[int, str] = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Derived property
    # ------------------------------------------------------------------

    @property
    def forwarded_token_count(self) -> int:
        """Number of query rows forwarded this step (the per-step FLOP signal)."""
        return len(self.forward_scope)

    # ------------------------------------------------------------------
    # Invariant enforcement + mutable-default sentinel fixup
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        # Sentinel → empty dict: frozen=True + slots=True prevents a mutable
        # field default; we use None as a sentinel and replace it here via
        # object.__setattr__ (the only mutation allowed on a frozen dataclass).
        if self.reuse_source is None:
            object.__setattr__(self, "reuse_source", {})

        frozen = set(self.frozen_positions)
        active = set(self.active_positions)
        dirty = set(self.dirty_positions)
        remask = set(self.remask_positions)

        # Invariant 1: active, dirty, remask, frozen are PAIRWISE DISJOINT.
        # dependency_risk is exempt — it is a flag on active positions (not a
        # separate partition), so dependency_risk ⊆ active is expected and allowed.
        pairs = [
            ("frozen", "active", frozen & active),
            ("frozen", "dirty", frozen & dirty),
            ("frozen", "remask", frozen & remask),
            ("active", "dirty", active & dirty),
            ("active", "remask", active & remask),
            ("dirty", "remask", dirty & remask),
        ]
        for name_a, name_b, overlap in pairs:
            if overlap:
                raise ValueError(
                    f"{name_a}_positions and {name_b}_positions must be disjoint; "
                    f"overlapping positions: {sorted(overlap)}"
                )

        # Invariant 2: forward_scope ⊆ active ∪ dirty ∪ remask
        non_frozen_set = active | dirty | remask
        scope_set = set(self.forward_scope)
        bad_scope = scope_set - non_frozen_set
        if bad_scope:
            raise ValueError(
                f"forward_scope must be a subset of (active ∪ dirty ∪ remask); "
                f"positions not in that union: {sorted(bad_scope)}"
            )

        # Invariant 3: every position in forward_scope has a cache_action entry
        missing_ca = scope_set - set(self.cache_action)
        if missing_ca:
            raise ValueError(
                f"Every position in forward_scope must have a cache_action entry; "
                f"missing: {sorted(missing_ca)}"
            )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self, *, full: bool = False) -> dict[str, Any]:
        """A JSON-safe dict (tuples → lists, enum values by name, arrays summarized).

        Large position sequences are summarized as ``{"len","head","tail"}`` when
        ``full=False`` and the length exceeds ``_SUMMARY_LIMIT``.
        Pass ``full=True`` to receive all sequences as plain lists.
        ``cache_action`` keys are always ints; values are enum ``.name`` strings.
        ``reuse_source`` keys are ints; values are provenance tag strings.
        """
        return {
            "step": self.step,
            "active_positions": _maybe_summarize(self.active_positions, full),
            "dirty_positions": _maybe_summarize(self.dirty_positions, full),
            "remask_positions": _maybe_summarize(self.remask_positions, full),
            "frozen_positions": _maybe_summarize(self.frozen_positions, full),
            "dependency_risk_positions": _maybe_summarize(
                self.dependency_risk_positions, full
            ),
            "cache_action": {pos: act.name for pos, act in self.cache_action.items()},
            "forward_scope": _maybe_summarize(self.forward_scope, full),
            "forwarded_token_count": self.forwarded_token_count,
            "window_end": self.window_end,
            "reuse_source": dict(self.reuse_source),  # copy; keys int, values str
        }


# ---------------------------------------------------------------------------
# WindowExecutionResult — the return type of BackendRunner.execute_window (B1.5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WindowExecutionResult:
    """The result returned by ``BackendRunner.execute_window`` (Task B1.5, WS2).

    Carries both the selected logits AND an explicit ``positions`` map so the
    caller can index the full logit space without relying on implicit offsets.

    Fields
    ------
    logits : Any
        Logits for exactly the rows in ``plan.forward_scope``, in
        ``forward_scope`` order, shape ``[len(positions), vocab]``.  The array
        type matches whatever the backend produces (numpy or mlx array).
    positions : tuple[int, ...]
        Absolute sequence positions corresponding to each row of ``logits``,
        in the same order as ``logits``.  Always equals
        ``tuple(plan.forward_scope)`` for the MLX parity wrapper (B1); a future
        windowed backend (B3) will emit the same contract.
    cache_mutations : Any
        Cache deltas produced this step.  Always an empty tuple ``()`` today: the
        cache mutates via ``StateCacheManager`` side-effect and there is no consumer
        for populated mutations, so Phase 3A deliberately keeps ``()`` (no fabricated
        data) — this field is reserved for a future consumer.  Callers must treat
        ``()`` as "no mutations" and must not assume any other container type.
    forwarded_rows : int | None
        The honest count of query rows actually computed this forward (the d2f
        compute window ``len(plan.forward_positions())`` when a cache path is
        reduced; ``seq_len`` otherwise); ``None`` when unknown. This is the
        per-forward delta of ``mlx_timing_active_tokens_total`` — the loop uses
        it for the trace metric so the recorded forwarded count is the real computed-row
        count rather than the full window prefix ``len(token_ids)``.  Kept last
        with a ``None`` default so existing constructors are unaffected.
    """

    logits: Any
    positions: tuple[int, ...]
    cache_mutations: Any
    forwarded_rows: int | None = None


# ---------------------------------------------------------------------------
# classify_window — pure function, MLX-free (Task A2, WS1)
# ---------------------------------------------------------------------------


def classify_window(
    state,
    event_log=None,
    *,
    frontier=None,
    window_end=None,
    step: int = 0,
    policy: str = "exact",
    corrections=None,
    layer0_reuse_positions=None,
    upper_band_reuse_positions=None,
    dkv_reuse_positions=None,
) -> WindowExecutionPlan:
    """Derive a ``WindowExecutionPlan`` from a duck-typed canvas state.

    Duck-typed ``state`` attributes (do NOT import MaskState here):
    - ``state.masked_positions`` — ``list[int]`` of still-masked positions
      (already excludes committed holes).  May be a plain list attribute or a
      zero-argument callable; both are supported.
    - ``state.prompt_len`` — ``int`` length of the fixed prompt prefix.
    - ``state.x`` — the working token sequence; ``len(state.x)`` is the total
      sequence length.

    Parameters
    ----------
    state:
        Duck-typed canvas state (see above).
    event_log:
        Optional decode-event log (``DecodeEventLog`` or compatible stub).
        Accepted for signature compatibility.  **Unused in exact mode** — remask
        history does not change the exact-mode plan.
    frontier:
        Optional sequence of absolute positions that the commit-policy's
        ``StepPlan.frontier`` selected for this step.  When supplied,
        ``forward_scope`` is set to ``tuple(frontier)`` (the commit-policy's
        actual scope) and must be a subset of ``active_positions``
        (``forward_scope ⊆ active_positions`` is enforced).  When ``None``
        (the default, preserving all existing callers), ``forward_scope``
        falls back to ``active_positions`` (all masked positions), which is
        byte-identical to the pre-B1.5 behaviour.
    window_end:
        Optional integer hint from the commit-policy's ``StepPlan.window_end``
        (the D2F pipelined scheduler's growing window end).  Accepted for
        API completeness; not used by the data model (the window boundary is
        not a field of ``WindowExecutionPlan`` in WS1/WS2).
    step:
        Decode step index (0-based); recorded in the returned plan.
    policy:
        Execution policy.  Only ``"exact"`` is implemented; ``"aggressive"`` and
        any other value raise ``NotImplementedError`` (reserved for WS3).
    corrections:
        Optional ``CorrectionPlan`` from ``correction_policy.CorrectionPolicy``.
        When provided, committed positions named in ``corrections.remask_positions``
        or ``corrections.dirty_positions`` are reclassified OUT of
        ``frozen_positions`` and INTO ``remask_positions`` / ``dirty_positions``
        respectively.  Only committed (frozen, non-active, in-range) positions can
        be reclassified; any position listed in ``corrections`` that is NOT frozen
        is silently ignored (it is already active / out of range).

        C1 constraint: ``active_positions``, ``forward_scope``, ``cache_action``,
        and ``window_end`` are UNCHANGED by corrections (no forward expansion, no
        state mutation).  Default ``None`` → dirty/remask both empty, behavior is
        byte-identical to the pre-C1 path.

    layer0_reuse_positions : frozenset[int] | None, optional
        C2b-2 (WS3): the set of active positions whose **layer-0** K/V is
        eligible for exact reuse this step.  Produced by
        ``DelayedKVPolicy.eligible_layer0_positions``; the classifier does not
        invoke the policy itself (separation of concerns).

        * ``None`` (default, OFF) — all positions in ``forward_scope`` get
          ``cache_action = RECOMPUTE`` and ``reuse_source`` is empty.
          **Byte-identical to the pre-C2b-2 path.**
        * A ``frozenset`` of positions — only positions BOTH in this set AND in
          ``forward_scope`` are retaged:
          ``cache_action[p] = REUSE_EXACT`` + ``reuse_source[p] = "delayed_kv_layer0_exact"``.
          Positions in the frozenset but NOT in ``forward_scope`` are silently
          ignored (they are not Q rows this step).
          Remaining ``forward_scope`` positions keep ``RECOMPUTE``.

        **``REUSE_APPROX`` is NEVER emitted by this function in v0.**
        ``active_positions``, ``forward_scope``, and ``window_end`` are UNCHANGED
        by this parameter (it only retags some ``cache_action`` entries).

    upper_band_reuse_positions : frozenset[int] | None, optional
        C2b-v1 (WS3): the set of active positions whose **upper-band** K/V is
        eligible for approximate reuse this step.  Per-layer truth is determined
        by ``DelayedKVCacheStrategy.serve()`` in the forward; this per-position
        tag is the headline plan action.

        * ``None`` (default, OFF) — no ``REUSE_APPROX`` emitted for band positions.
          **Byte-identical to the pre-C2b-v1 path.**
        * A ``frozenset`` — positions in both this set AND ``forward_scope`` get
          ``cache_action[p] = REUSE_APPROX`` + ``reuse_source[p] = "delayed_kv_upper_band"``.
          Positions in the frozenset but NOT in ``forward_scope`` are silently ignored.

        When BOTH ``layer0_reuse_positions`` and ``upper_band_reuse_positions`` are
        provided for the same position, ``layer0_reuse_positions`` wins (REUSE_EXACT).
        ``active_positions``, ``forward_scope``, and ``window_end`` are UNCHANGED.

    dkv_reuse_positions : frozenset[int] | None, optional
        dKV spike (sota plan Task 5): active positions whose decoded-token K/V is
        eligible for one-step-delayed approximate reuse this step.  Produced by
        ``OneStepDecodedTokenPolicy.eligible_positions`` (a *write-side lifecycle*
        rule — a token committed at step ``t`` is only eligible at ``t+1``); the
        classifier does not invoke the policy.

        * ``None`` (default, OFF) — no ``REUSE_APPROX`` emitted for dKV positions.
          **Byte-identical to the pre-dKV path.**
        * A ``frozenset`` — positions in both this set AND ``forward_scope`` get
          ``cache_action[p] = REUSE_APPROX`` + ``reuse_source[p] = "dkv_decode_one_step"``.
          The tag is DISTINCT from ``"delayed_kv_upper_band"`` (kept clean of the C2b
          negative result).  Positions not in ``forward_scope`` are silently ignored.

        Precedence: ``layer0`` (REUSE_EXACT) > ``upper_band`` > ``dkv`` for a position
        named in more than one set.  ``active_positions``, ``forward_scope``, and
        ``window_end`` are UNCHANGED (this only retags some ``cache_action`` entries).

    Returns
    -------
    WindowExecutionPlan
        Immutable, byte-exact plan for this decode step.

    Raises
    ------
    NotImplementedError
        If ``policy != "exact"``.
    ValueError
        If ``frontier`` is given and contains positions outside
        ``active_positions`` (``forward_scope ⊄ active_positions``).
    """
    if policy != "exact":
        raise NotImplementedError(
            f"classify_window: policy={policy!r} is not implemented. "
            "Only policy='exact' is available; aggressive windowing is reserved "
            "for WS3."
        )

    # ------------------------------------------------------------------
    # Resolve masked_positions — support both plain-list attribute and
    # zero-argument callable (matches real MaskState.masked_positions()).
    # ------------------------------------------------------------------
    raw_masked = state.masked_positions
    if callable(raw_masked):
        raw_masked = raw_masked()

    active_positions: tuple[int, ...] = tuple(sorted(raw_masked))
    active_set: set[int] = set(active_positions)

    seq_len: int = len(state.x)

    # All positions not in the active set are frozen (prompt + committed holes).
    # This is the full frozen set before any correction reclassification.
    base_frozen_set: set[int] = set(
        p for p in range(seq_len) if p not in active_set
    )

    # ------------------------------------------------------------------
    # C1: reclassify committed (frozen) positions into dirty / remask
    # when a CorrectionPlan is supplied.
    #
    # Only frozen positions can be reclassified (a flagged position must
    # already be committed — i.e. not currently masked / active).
    # active_positions / forward_scope / cache_action are UNCHANGED.
    # ------------------------------------------------------------------
    if corrections is not None:
        remask_positions: tuple[int, ...] = tuple(
            sorted(p for p in corrections.remask_positions if p in base_frozen_set)
        )
        dirty_positions: tuple[int, ...] = tuple(
            sorted(p for p in corrections.dirty_positions if p in base_frozen_set)
        )
        # Remove the reclassified positions from frozen
        reclassified: set[int] = set(remask_positions) | set(dirty_positions)
        frozen_positions: tuple[int, ...] = tuple(
            sorted(base_frozen_set - reclassified)
        )
    else:
        remask_positions = ()
        dirty_positions = ()
        frozen_positions = tuple(sorted(base_frozen_set))

    # Exact mode: forward_scope is determined by the commit-policy frontier
    # when one is supplied, or falls back to all active positions.
    # C1: forward_scope is determined solely from active_positions — corrections
    # do NOT expand the forward (no aggressive window in C1).
    if frontier is not None:
        forward_scope: tuple[int, ...] = tuple(frontier)
        # Invariant: forward_scope ⊆ active_positions (the commit policy may only
        # scope within still-masked positions; it may not include frozen/committed).
        bad = set(forward_scope) - active_set
        if bad:
            raise ValueError(
                f"classify_window: frontier positions {sorted(bad)} are not in "
                f"active_positions (they are frozen/committed); "
                "forward_scope must be a subset of active_positions"
            )
    else:
        # Degenerate / no commit policy: forward all active positions.
        # This preserves byte-identical behaviour for all existing callers that
        # do not pass a frontier.
        forward_scope = active_positions

    # C2b-2 + C2b-v1 + dKV: build cache_action + reuse_source.
    # Default (all None): all RECOMPUTE, reuse_source empty.
    # layer0_reuse_positions:     set ∩ forward_scope → REUSE_EXACT  ("delayed_kv_layer0_exact").
    # upper_band_reuse_positions: set ∩ forward_scope → REUSE_APPROX ("delayed_kv_upper_band").
    # dkv_reuse_positions:        set ∩ forward_scope → REUSE_APPROX ("dkv_decode_one_step").
    # Precedence when a position is in several: layer0 > upper_band > dkv.
    if (
        layer0_reuse_positions is None
        and upper_band_reuse_positions is None
        and dkv_reuse_positions is None
    ):
        cache_action: dict[int, CacheAction] = {
            p: CacheAction.RECOMPUTE for p in forward_scope
        }
        reuse_source: dict[int, str] = {}
    else:
        scope_set_for_reuse = set(forward_scope)
        l0_eligible = (
            frozenset(layer0_reuse_positions) & scope_set_for_reuse
            if layer0_reuse_positions is not None
            else frozenset()
        )
        band_eligible = (
            frozenset(upper_band_reuse_positions) & scope_set_for_reuse
            if upper_band_reuse_positions is not None
            else frozenset()
        )
        dkv_eligible = (
            frozenset(dkv_reuse_positions) & scope_set_for_reuse
            if dkv_reuse_positions is not None
            else frozenset()
        )
        cache_action = {}
        reuse_source = {}
        for p in forward_scope:
            if p in l0_eligible:
                cache_action[p] = CacheAction.REUSE_EXACT
                reuse_source[p] = "delayed_kv_layer0_exact"
            elif p in band_eligible:
                cache_action[p] = CacheAction.REUSE_APPROX
                reuse_source[p] = "delayed_kv_upper_band"
            elif p in dkv_eligible:
                cache_action[p] = CacheAction.REUSE_APPROX
                reuse_source[p] = "dkv_decode_one_step"
            else:
                cache_action[p] = CacheAction.RECOMPUTE

    return WindowExecutionPlan(
        step=step,
        active_positions=active_positions,
        dirty_positions=dirty_positions,
        remask_positions=remask_positions,
        frozen_positions=frozen_positions,
        dependency_risk_positions=(),
        cache_action=cache_action,
        forward_scope=forward_scope,
        window_end=window_end,
        reuse_source=reuse_source,
    )
