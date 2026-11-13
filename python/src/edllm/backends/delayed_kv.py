"""Delayed-KV cache identity + invalidation model (C2b-1, WS3).

Pure data + logic — MLX-free, no torch, no numpy.  Mirrors ``window_plan.py``
in style: import-light, dataclass-based, a single module-level pure function
(``is_valid``) that encodes the §2a/§4 drift invariants from the spec.

Spec: docs/superpowers/specs/2026-06-16-cache-ladder-delayed-kv.md §2a + §4.
Plan: docs/superpowers/plans/2026-06-16-c2b-delayed-kv.md Task C2b-1 + C2b-2.
C2b-v1 extension: docs/superpowers/specs/2026-06-16-c2b-v1-upper-layer-approx.md §1,§2,§7.

The drift invariant (§2a, binding):
  - layer-0 K/V of position ``p``: depends only on ``p``'s own token + RoPE.
    Invalidated ONLY when ``p``'s token changes (``token_version``).
    Stable across any number of context_epoch bumps (commits elsewhere).
  - layer-``l`` (``l > 0``) K/V of ``p``: depends on the full sequence at
    layer ``l-1``.  Drifts on ANY commit since it was computed.  Invalidated
    as soon as ``current_context_epoch > computed_at_context_epoch``.

``is_valid`` checks in this priority order (first failure wins):
  1. geometry_mismatch  — dtype/head/RoPE signature changed.
  2. commit-invalidation (HARD, not tunable):
       layer == 0 → token_changed (token at this position changed).
       layer > 0  → context_epoch_drift (any commit since computed).
  3. age_bound          — reuse_approx_age >= max_stale_steps.
  4. periodic_refresh   — step % refresh_interval == 0 (when refresh_interval
                         is an int > 0).

This module deliberately keeps the full per-layer abstraction.  The v0
restriction "serve only layer-0 stale K/V" is enforced by the forward/strategy
layer (C2b-4/C2b-6), NOT here.  Upper layers are modelled correctly (via
context_epoch_drift) so that measurement and future expansion are sound.

C2b-2 additions:
  ``DelayedKVPolicy`` — decides layer-0 eligibility from state + cache_state.
  Default OFF (``enable=False``): returns ``frozenset()``; ``classify_window``
  then emits only ``RECOMPUTE`` for all active positions.  When enabled with
  ``scope="layer0"``, calls ``is_valid`` on each active position's layer-0
  entry and returns those that pass.  ``scope="all_layers"`` is FORBIDDEN in
  v0 and raises immediately (fail-closed).  ``REUSE_APPROX`` is NEVER emitted
  in v0.

C2b-v1 additions:
  ``DelayedKVCacheStrategy`` gains ``scope="upper_band"`` support.  When
  ``scope="upper_band"``, layers in ``[lo, hi)`` are served as ``REUSE_APPROX``
  when within the ``stale_epoch_budget`` and ``max_stale_steps`` bounds.
  Layer-0 stays the v0 exact path.  Layers outside the band get ``RECOMPUTE``
  (``out_of_band``).  ``scope="all_layers"`` remains fail-closed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DelayedKVKey:
    """Immutable identity key for a single cached K/V entry.

    Fields
    ------
    layer_index : int
        Transformer layer index (0-based).  Layer-0 has a different
        invalidation rule from layers > 0 — see ``is_valid``.
    position : int
        Absolute sequence position whose K/V is stored.
    """

    layer_index: int
    position: int


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DelayedKVEntry:
    """A single delayed-KV cache entry carrying the K/V payload + provenance.

    Fields
    ------
    key : DelayedKVKey
        Identity key ``(layer_index, position)``.
    kv : Any
        OPAQUE K/V payload — this model never inspects its contents.
        Storing/passing it is the only operation performed.  The MLX-free
        contract is upheld: no mlx/torch/numpy import is required.
    computed_at_token_version : int
        The token identity at ``key.position`` when this K/V was computed.
        Used by the layer-0 invalidation check (``token_changed``).
    computed_at_context_epoch : int
        The global commit-epoch when this K/V was computed.  Bumped by any
        commit anywhere in the sequence.  Used by the upper-layer drift check
        (``context_epoch_drift``).
    geometry_signature : Any
        dtype / head geometry (``n_heads``, ``n_kv_heads``, ``head_dim``) /
        RoPE signature (``rope_theta`` / scaling / applied position).  Compared
        by ``==``.  A mismatch renders the entry permanently unusable.
    reuse_approx_age : int
        Steps elapsed since this K/V was computed (incremented each time the
        entry is reused without refresh).  Bounded by ``max_stale_steps``.
    refresh_count : int
        Provenance counter: how many times this entry was refreshed (K/V
        recomputed at a new step while the slot was retained).
    invalidated_count : int
        Provenance counter: how many times this entry was invalidated
        (commit-invalidation or age-bound triggered a ``RECOMPUTE``).
    """

    key: DelayedKVKey
    kv: Any
    computed_at_token_version: int
    computed_at_context_epoch: int
    geometry_signature: Any
    reuse_approx_age: int
    refresh_count: int
    invalidated_count: int


# ---------------------------------------------------------------------------
# Validation logic
# ---------------------------------------------------------------------------


def is_valid(
    entry: DelayedKVEntry,
    *,
    current_token_version: int,
    current_context_epoch: int,
    current_geometry_signature: Any,
    max_stale_steps: int,
    refresh_interval: int | None,
    step: int,
) -> tuple[bool, str]:
    """Return ``(valid, reason)`` for a cached K/V entry under the §2a/§4 rules.

    Check priority (first failure wins — geometry is always outermost):

    1. **geometry_mismatch** — ``entry.geometry_signature != current_geometry_signature``.
       A changed dtype, head shape, or RoPE config makes the stored K/V
       structurally incompatible.

    2. **commit-invalidation (HARD RULE, not tunable):**

       - ``entry.key.layer_index == 0``:
           Layer-0 K/V depends only on its own token + RoPE.
           Invalid iff *its* token changed:
           ``current_token_version != entry.computed_at_token_version``
           → ``(False, "token_changed")``.

       - ``entry.key.layer_index > 0``:
           Upper-layer K/V depends on the full sequence at the layer below.
           Drifts on ANY commit since it was computed.
           Invalid iff the context epoch advanced:
           ``current_context_epoch > entry.computed_at_context_epoch``
           → ``(False, "context_epoch_drift")``.

    3. **age_bound** — ``entry.reuse_approx_age >= max_stale_steps``
       → ``(False, "age_bound")``.

    4. **periodic_refresh** — ``refresh_interval`` is a positive int AND
       ``step % refresh_interval == 0``
       → ``(False, "periodic_refresh")``.

    Otherwise → ``(True, "valid")``.

    Parameters
    ----------
    entry:
        The cached entry to evaluate.
    current_token_version:
        The token identity currently at ``entry.key.position``.  Used for
        the layer-0 commit-invalidation check.
    current_context_epoch:
        The current global commit-epoch.  Used for the layer>0 drift check.
    current_geometry_signature:
        The current dtype/head/RoPE signature.  Compared by ``==``.
    max_stale_steps:
        Mandatory upper bound on ``reuse_approx_age``.  Must be non-None.
    refresh_interval:
        Optional periodic safety valve.  ``None`` or ``0`` → disabled.
        A positive int ``r`` triggers a refresh when ``step % r == 0``.
    step:
        Current decode step index (0-based).

    Returns
    -------
    tuple[bool, str]
        ``(True, "valid")`` or ``(False, <reason>)`` where ``<reason>`` is one
        of ``"geometry_mismatch"``, ``"token_changed"``, ``"context_epoch_drift"``,
        ``"age_bound"``, ``"periodic_refresh"``.
    """
    # 1. Geometry check — always first.
    if entry.geometry_signature != current_geometry_signature:
        return False, "geometry_mismatch"

    # 2. Commit-invalidation — HARD RULE, not a knob.
    if entry.key.layer_index == 0:
        # Layer-0: invalidated only when this position's own token changed.
        if current_token_version != entry.computed_at_token_version:
            return False, "token_changed"
    else:
        # Upper layers: invalidated on ANY commit (any context_epoch advance).
        if current_context_epoch > entry.computed_at_context_epoch:
            return False, "context_epoch_drift"

    # 3. Age bound — mandatory when enabled.
    if entry.reuse_approx_age >= max_stale_steps:
        return False, "age_bound"

    # 4. Periodic refresh — optional safety valve. step > 0: the live d2f forward
    # serves with step=0, and 0 % k == 0 for every k, so without the guard any
    # positive interval would permanently disable ALL reuse instead of refreshing
    # at real step multiples (2026-07-02 review finding #1).
    if refresh_interval is not None and refresh_interval > 0 and step > 0 and step % refresh_interval == 0:
        return False, "periodic_refresh"

    return True, "valid"


# ---------------------------------------------------------------------------
# DelayedKVPolicy — eligibility decision (C2b-2)
# ---------------------------------------------------------------------------

_SUPPORTED_SCOPES = frozenset({"layer0", "upper_band"})
_SENTINEL_GEO = object()  # used when no geometry check is needed in the policy


@dataclass
class DelayedKVPolicy:
    """Decides which active positions are eligible for layer-0 delayed-KV reuse.

    Default OFF (``enable=False``): ``eligible_layer0_positions`` always returns
    ``frozenset()`` — zero-cost, byte-neutral with the exact path.

    When ``enable=True`` and ``scope="layer0"``, consults ``is_valid`` on each
    active position's layer-0 cache entry (from ``cache_state``).  A position is
    eligible iff:

    * It has an entry in ``cache_state`` (mapping position → ``DelayedKVEntry``
      for ``layer_index == 0``).
    * ``is_valid(entry, ...)`` returns ``(True, "valid")``.

    The ``state`` object is duck-typed; the policy reads ``state.token_versions``
    (a mapping ``{position: token_version_int}``) to determine the current token
    identity at each position.  The real cache-backed implementation arrives in
    C2b-3; here a minimal stub (``SimpleNamespace(token_versions={...})``) is
    sufficient for standalone testing.

    Parameters
    ----------
    enable : bool
        Whether the policy is active.  Default ``False`` (OFF).
    scope : str
        Which layer range is eligible.  Only ``"layer0"`` is supported in v0.
        Any other value raises ``ValueError`` on construction (fail-closed).
    max_stale_steps : int
        Mandatory staleness bound for ``is_valid``.  Only consulted when
        ``enable=True``; ignored when ``enable=False``.  Default ``1`` (minimal
        non-zero; callers should always pass an explicit value when enabling).
    refresh_interval : int | None
        Optional periodic refresh valve passed to ``is_valid``.  ``None`` →
        no periodic refresh beyond commit-invalidation + age bound.

    Raises
    ------
    ValueError
        If ``scope`` is not ``"layer0"`` (any unsupported scope fails closed).

    Notes
    -----
    ``REUSE_APPROX`` is **NEVER** emitted by this policy or by
    ``classify_window`` in v0.  Upper-layer reuse (``scope="all_layers"``) is
    deferred to C2b-v1 and deliberately raises here to prevent silent misuse.
    """

    enable: bool = False
    scope: str = "layer0"
    max_stale_steps: int = 1
    refresh_interval: int | None = None

    def __post_init__(self) -> None:
        if self.scope not in _SUPPORTED_SCOPES:
            raise ValueError(
                f"DelayedKVPolicy: scope={self.scope!r} is not supported in v0. "
                f"Only scope='layer0' is valid; scope='all_layers' (upper-layer "
                f"approximate reuse, REUSE_APPROX) is deferred to C2b-v1. "
                f"Supported scopes: {sorted(_SUPPORTED_SCOPES)}"
            )

    def eligible_layer0_positions(
        self,
        *,
        state: Any,
        active_positions: frozenset[int],
        cache_state: dict[int, DelayedKVEntry],
        step: int,
    ) -> frozenset[int]:
        """Return the subset of ``active_positions`` eligible for layer-0 reuse.

        Parameters
        ----------
        state:
            Duck-typed state object.  Must expose ``state.token_versions``: a
            mapping ``{position: int}`` giving the current token version at each
            position.  Used for the layer-0 ``token_changed`` check.  May be
            ``None`` when ``enable=False`` (the result is always ``frozenset()``
            and ``state`` is never accessed).
        active_positions:
            The positions being decoded this step (must be a frozenset of ints).
        cache_state:
            Mapping ``{position: DelayedKVEntry}`` for ``layer_index == 0``
            entries only.  A position absent from this mapping has no cached
            layer-0 entry and is therefore ineligible.
        step:
            Current decode step index (0-based), forwarded to ``is_valid``.

        Returns
        -------
        frozenset[int]
            Positions in ``active_positions`` whose layer-0 K/V is valid for
            reuse.  Always a frozenset (possibly empty).
        """
        if not self.enable:
            return frozenset()

        # scope is guaranteed "layer0" by __post_init__; no branch needed here.
        eligible: set[int] = set()
        token_versions: dict[int, int] = state.token_versions if state is not None else {}

        for pos in active_positions:
            entry = cache_state.get(pos)
            if entry is None:
                # No cached layer-0 entry → cannot reuse.
                continue

            current_token_version = token_versions.get(pos, -1)
            # Use a sentinel geometry: since this policy does not hold the real
            # geometry, we pass the entry's own geometry as "current" so the
            # geometry check always passes.  The real geometry check is the
            # cache strategy's responsibility (C2b-3); here we verify only the
            # token-stability / staleness / refresh invariants.
            valid, _reason = is_valid(
                entry,
                current_token_version=current_token_version,
                current_context_epoch=0,  # layer-0 does not use context_epoch
                current_geometry_signature=entry.geometry_signature,
                max_stale_steps=self.max_stale_steps,
                refresh_interval=self.refresh_interval,
                step=step,
            )
            if valid:
                eligible.add(pos)

        return frozenset(eligible)


# ---------------------------------------------------------------------------
# DelayedKVCacheStrategy — per-(layer, position) cache (C2b-3)
# ---------------------------------------------------------------------------

# CacheAction string constants (mirrors CacheAction enum intent, string-typed
# here so this module stays MLX-free and import-free from window_plan).
_REUSE_EXACT = "REUSE_EXACT"
_RECOMPUTE = "RECOMPUTE"
_REUSE_SOURCE_LAYER0 = "delayed_kv_layer0_exact"



# v1 cap: the maximum allowed band width (hi - lo). Documented in spec §1.
_MAX_BAND_WIDTH = 8

# String constants for v1 actions.
_REUSE_APPROX = "REUSE_APPROX"
_REUSE_SOURCE_UPPER_BAND = "delayed_kv_upper_band"

# dKV one-step-delay spike (sota plan Task 6). A DISTINCT reuse source so its
# savings/quality are measured in isolation and never conflated with the C2b
# upper-band negative result ("delayed_kv_upper_band").
_REUSE_SOURCE_DKV_DECODE = "dkv_decode_one_step"
# The one-step delay, expressed in the forward's epoch coordinate: a decoded
# token's K/V becomes reusable only once at least this many commit-epochs have
# elapsed since it was computed (the dKV-Cache insight — a just-decoded token's
# K/V is not yet stable at its own decode step).
_DKV_MIN_EPOCH_DELAY = 1


def _validate_upper_band_config(
    *,
    upper_layer_range: tuple[int, int] | None,
    stale_epoch_budget: int | None,
    max_stale_steps: int,
    num_layers: int | None,
) -> None:
    """Validate upper_band config params; raise ValueError (fail-closed) on any violation."""
    if stale_epoch_budget is None:
        raise ValueError(
            "DelayedKVCacheStrategy: stale_epoch_budget is MANDATORY when "
            "scope='upper_band'. Pass stale_epoch_budget=<int> (0 <= budget <= "
            "max_stale_steps). Budget=0 means invalidate on the very next commit "
            "(v0-equivalent strictness for the band)."
        )
    if stale_epoch_budget < 0:
        raise ValueError(
            f"DelayedKVCacheStrategy: stale_epoch_budget must be >= 0, "
            f"got {stale_epoch_budget}."
        )
    if stale_epoch_budget > max_stale_steps:
        raise ValueError(
            f"DelayedKVCacheStrategy: stale_epoch_budget ({stale_epoch_budget}) "
            f"must be <= max_stale_steps ({max_stale_steps}). "
            "A budget exceeding max_stale_steps is rejected (fail-closed)."
        )
    if upper_layer_range is None:
        raise ValueError(
            "DelayedKVCacheStrategy: upper_layer_range is MANDATORY when "
            "scope='upper_band'. Pass upper_layer_range=(lo, hi) with "
            "1 <= lo < hi <= num_layers and hi - lo <= MAX_BAND_WIDTH."
        )
    lo, hi = upper_layer_range
    if lo >= hi:
        raise ValueError(
            f"DelayedKVCacheStrategy: invalid range upper_layer_range={upper_layer_range!r}. "
            "Require lo < hi (half-open interval [lo, hi))."
        )
    if lo < 1:
        raise ValueError(
            "DelayedKVCacheStrategy: lo=0 is invalid for upper_band scope. "
            "Layer-0 is the v0 EXACT path — it must NOT be included in the "
            "upper_band range. Set lo >= 1."
        )
    # Reject [1, num_layers) BEFORE the width check — "all upper layers" is a
    # named forbidden case in spec §1 and should surface its own message.
    if num_layers is not None and lo == 1 and hi == num_layers:
        raise ValueError(
            f"DelayedKVCacheStrategy: upper_layer_range=[1, {num_layers}) is the "
            "forbidden 'all upper layers' full upper range — it confounds variables "
            "and is rejected by the v1 spec. Use a smaller band."
        )
    if hi - lo > _MAX_BAND_WIDTH:
        raise ValueError(
            f"DelayedKVCacheStrategy: band width (hi - lo = {hi - lo}) exceeds "
            f"MAX_BAND_WIDTH={_MAX_BAND_WIDTH}. "
            "Widening beyond the v1 cap is a later iteration gated on quality. "
            "Reduce the band range."
        )


class DelayedKVCacheStrategy:
    """Per-``(layer_index, position)`` delayed-KV cache for the v0 layer-0 rung
    and the v1 upper-band approximate rung.

    Stores :class:`DelayedKVEntry` objects keyed by :class:`DelayedKVKey`.
    The KV payload is OPAQUE (any Python object); this class never inspects it.

    **v0 serve rule (scope="layer0", default):**

    * ``layer_index == 0`` and ``is_valid(...)`` → ``(kv, REUSE_EXACT,
      "delayed_kv_layer0_exact")``.  Increments ``reuse_count``.
    * ``layer_index == 0`` but invalid → ``(None, RECOMPUTE, reason)``.
    * ``layer_index > 0`` → ALWAYS ``(None, RECOMPUTE, "context_epoch_drift")``.

    **v1 upper-band serve rule (scope="upper_band"):**

    * ``layer_index == 0`` → v0 exact path unchanged (REUSE_EXACT / RECOMPUTE).
    * ``lo <= layer_index < hi`` and band-valid → ``(kv, REUSE_APPROX,
      "delayed_kv_upper_band")``.  Increments ``reuse_approx_count``.
    * ``lo <= layer_index < hi`` but invalid → ``(None, RECOMPUTE, reason)``.
    * ``layer_index`` outside band (and != 0) → ``(None, RECOMPUTE, "out_of_band")``.
    * ``scope="all_layers"`` → ALWAYS fail-closed (ValueError on construction).

    **Metrics counters** (read via :meth:`metrics`)::

        reuse_count               — REUSE_EXACT serves (layer-0 exact)
        reuse_approx_count        — REUSE_APPROX serves (upper-band approximate)
        recompute_count           — total RECOMPUTE returns
        recompute_reasons         — dict {reason: count} for all RECOMPUTE returns
        invalidated_count         — calls to invalidate_on_commit that cleared entries
        error_fix_recompute_count — recomputes forced by error-fixing (Task 6 slot)

    Parameters
    ----------
    max_stale_steps : int
        Mandatory staleness bound passed to ``is_valid``.
    refresh_interval : int | None
        Optional periodic refresh valve.  ``None`` → disabled.
    geometry_signature : Any
        Default geometry signature for new entries.
    scope : str
        ``"layer0"`` (v0, default) or ``"upper_band"`` (v1).
        ``"all_layers"`` raises ``ValueError`` (fail-closed).
    stale_epoch_budget : int | None
        MANDATORY when ``scope="upper_band"``.  Validated
        ``0 <= stale_epoch_budget <= max_stale_steps``.
    upper_layer_range : tuple[int, int] | None
        MANDATORY when ``scope="upper_band"``.  Half-open ``[lo, hi)``;
        validated: ``1 <= lo < hi``, ``hi - lo <= MAX_BAND_WIDTH``,
        ``[1, num_layers)`` forbidden.
    num_layers : int | None
        Total transformer layer count.  Required for the ``[1, num_layers)``
        guard; ``None`` skips that specific check.
    """

    __slots__ = (
        "_max_stale_steps",
        "_refresh_interval",
        "_geometry_signature",
        "_store",
        "_reuse_count",
        "_reuse_approx_count",
        "_dkv_decode_reuse_count",
        "_recompute_count",
        "_recompute_reasons",
        "_invalidated_count",
        "_error_fix_recompute_count",
        "_scope",
        "_stale_epoch_budget",
        "_band_lo",
        "_band_hi",
        "_context_epoch",
    )

    def __init__(
        self,
        *,
        max_stale_steps: int,
        refresh_interval: int | None = None,
        geometry_signature: Any = None,
        scope: str = "layer0",
        stale_epoch_budget: int | None = None,
        upper_layer_range: tuple[int, int] | None = None,
        num_layers: int | None = None,
    ) -> None:
        # Fail-closed guard: all_layers is never allowed.
        if scope == "all_layers":
            raise ValueError(
                "DelayedKVCacheStrategy: scope='all_layers' is forbidden. "
                "Only scope='layer0' (v0 exact) and scope='upper_band' (v1 "
                "approximate) are supported. 'all_layers' confounds variables "
                "and is explicitly rejected."
            )
        if scope not in ("layer0", "upper_band", "dkv_decode"):
            raise ValueError(
                f"DelayedKVCacheStrategy: scope={scope!r} is not supported. "
                "Valid scopes: 'layer0', 'upper_band', 'dkv_decode'."
            )

        self._max_stale_steps = int(max_stale_steps)
        self._refresh_interval = refresh_interval
        self._geometry_signature = geometry_signature
        self._scope = scope

        if scope in ("upper_band", "dkv_decode"):
            # dkv_decode reuses the upper-band band/budget config + validation; it
            # adds only the one-step delay gate at serve time (see _serve_dkv_decode).
            _validate_upper_band_config(
                upper_layer_range=upper_layer_range,
                stale_epoch_budget=stale_epoch_budget,
                max_stale_steps=self._max_stale_steps,
                num_layers=num_layers,
            )
            self._stale_epoch_budget: int = int(stale_epoch_budget)  # type: ignore[arg-type]
            # dkv_decode fail-closed: the one-step delay makes drift 0 delay-pending,
            # so the reuse window is [_DKV_MIN_EPOCH_DELAY, budget] — budget below the
            # delay is structurally EMPTY (no row would EVER be served; a silent no-op).
            # budget=0 stays valid for upper_band (its byte-exact zone, serves at drift 0).
            # 2026-07-02 review finding #2.
            if scope == "dkv_decode" and self._stale_epoch_budget < _DKV_MIN_EPOCH_DELAY:
                raise ValueError(
                    f"DelayedKVCacheStrategy: scope='dkv_decode' requires "
                    f"stale_epoch_budget >= {_DKV_MIN_EPOCH_DELAY} (the one-step delay); "
                    f"got {self._stale_epoch_budget}. Drift 0 is delay-pending, so a "
                    f"budget below the delay yields an EMPTY reuse window — no row "
                    "would ever be served. Use stale_epoch_budget >= 1."
                )
            lo, hi = upper_layer_range  # type: ignore[misc]
            self._band_lo: int = lo
            self._band_hi: int = hi
        else:
            # scope == "layer0": band fields unused.
            self._stale_epoch_budget = 0
            self._band_lo = 0
            self._band_hi = 0

        # The primary store: DelayedKVKey → DelayedKVEntry.
        self._store: dict[DelayedKVKey, DelayedKVEntry] = {}
        # Metrics.
        self._reuse_count: int = 0
        self._reuse_approx_count: int = 0
        self._dkv_decode_reuse_count: int = 0
        self._recompute_count: int = 0
        self._recompute_reasons: dict[str, int] = {}
        self._invalidated_count: int = 0
        self._error_fix_recompute_count: int = 0
        # Context epoch: incremented by invalidate_on_commit on each commit.
        # Used by upper-band serve() as current_context_epoch.
        self._context_epoch: int = 0

    # -- write-side ---------------------------------------------------------- #

    def store(
        self,
        key: DelayedKVKey,
        kv: Any,
        *,
        token_version: int,
        context_epoch: int,
        geometry_signature: Any = None,
        step: int,
    ) -> None:
        """Store (or overwrite) a cached K/V entry.

        Parameters
        ----------
        key:
            ``DelayedKVKey(layer_index, position)`` identifying the entry.
        kv:
            Opaque K/V payload — stored as-is, never inspected.
        token_version:
            The token identity at ``key.position`` when this K/V was computed.
        context_epoch:
            The global commit-epoch when this K/V was computed.
        geometry_signature:
            dtype/head/RoPE signature; defaults to the strategy's default if
            not supplied.
        step:
            Current decode step (informational; stored via refresh tracking).
        """
        geo = geometry_signature if geometry_signature is not None else self._geometry_signature
        existing = self._store.get(key)
        if existing is not None:
            # Refresh: update in-place, bump refresh_count, reset age.
            existing.kv = kv
            existing.computed_at_token_version = token_version
            existing.computed_at_context_epoch = context_epoch
            existing.geometry_signature = geo
            existing.reuse_approx_age = 0
            existing.refresh_count += 1
        else:
            self._store[key] = DelayedKVEntry(
                key=key,
                kv=kv,
                computed_at_token_version=token_version,
                computed_at_context_epoch=context_epoch,
                geometry_signature=geo,
                reuse_approx_age=0,
                refresh_count=0,
                invalidated_count=0,
            )

    # -- read-side ----------------------------------------------------------- #

    def serve(
        self,
        *,
        layer_index: int,
        position: int,
        current_token_version: int,
        current_context_epoch: int,
        current_geometry_signature: Any,
        max_stale_steps: int | None = None,
        refresh_interval: int | None = None,
        step: int,
    ) -> tuple[Any, str, str]:
        """Decide whether to serve a cached KV or request a recompute.

        Parameters
        ----------
        layer_index, position:
            Which entry to look up.
        current_token_version:
            The token identity currently at ``position``.
        current_context_epoch:
            The current global commit-epoch.
        current_geometry_signature:
            The current dtype/head/RoPE signature (compared by ``==``).
        max_stale_steps:
            Override for the staleness bound (defaults to the strategy's own).
        refresh_interval:
            Override for the refresh interval (defaults to the strategy's own).
        step:
            Current decode step.

        Returns
        -------
        tuple[kv | None, action, source_or_reason]
            * On REUSE_EXACT: ``(kv, "REUSE_EXACT", "delayed_kv_layer0_exact")``.
            * On REUSE_APPROX: ``(kv, "REUSE_APPROX", "delayed_kv_upper_band")``.
            * On RECOMPUTE: ``(None, "RECOMPUTE", reason)`` where reason is one
              of ``"not_found"``, ``"context_epoch_drift"``, ``"token_changed"``,
              ``"age_bound"``, ``"periodic_refresh"``, ``"geometry_mismatch"``,
              ``"out_of_band"``, ``"stale_epoch_budget_exceeded"``.
        """
        # Layer-0: always use the v0 exact path (unchanged).
        if layer_index == 0:
            return self._serve_layer0(
                position=position,
                current_token_version=current_token_version,
                current_context_epoch=current_context_epoch,
                current_geometry_signature=current_geometry_signature,
                max_stale_steps=max_stale_steps,
                refresh_interval=refresh_interval,
                step=step,
            )

        # Upper layers: branch on scope.
        if self._scope == "upper_band":
            return self._serve_upper_band(
                layer_index=layer_index,
                position=position,
                current_token_version=current_token_version,
                current_context_epoch=current_context_epoch,
                current_geometry_signature=current_geometry_signature,
                max_stale_steps=max_stale_steps,
                refresh_interval=refresh_interval,
                step=step,
            )

        if self._scope == "dkv_decode":
            return self._serve_dkv_decode(
                layer_index=layer_index,
                position=position,
                current_token_version=current_token_version,
                current_context_epoch=current_context_epoch,
                current_geometry_signature=current_geometry_signature,
                max_stale_steps=max_stale_steps,
                refresh_interval=refresh_interval,
                step=step,
            )

        # scope == "layer0": upper layers NEVER served (v0 rule).
        self._record_recompute("context_epoch_drift")
        return None, _RECOMPUTE, "context_epoch_drift"

    def _serve_layer0(
        self,
        *,
        position: int,
        current_token_version: int,
        current_context_epoch: int,
        current_geometry_signature: Any,
        max_stale_steps: int | None,
        refresh_interval: int | None,
        step: int,
    ) -> tuple[Any, str, str]:
        """v0 exact path for layer_index == 0. Unchanged by v1."""
        key = DelayedKVKey(layer_index=0, position=position)
        entry = self._store.get(key)

        if entry is None:
            self._record_recompute("not_found")
            return None, _RECOMPUTE, "not_found"

        ms = max_stale_steps if max_stale_steps is not None else self._max_stale_steps
        ri = refresh_interval if refresh_interval is not None else self._refresh_interval

        valid, reason = is_valid(
            entry,
            current_token_version=current_token_version,
            current_context_epoch=current_context_epoch,
            current_geometry_signature=current_geometry_signature,
            max_stale_steps=ms,
            refresh_interval=ri,
            step=step,
        )

        if valid:
            entry.reuse_approx_age += 1
            self._reuse_count += 1
            return entry.kv, _REUSE_EXACT, _REUSE_SOURCE_LAYER0

        entry.invalidated_count += 1
        self._record_recompute(reason)
        return None, _RECOMPUTE, reason

    def _serve_upper_band(
        self,
        *,
        layer_index: int,
        position: int,
        current_token_version: int,
        current_context_epoch: int,
        current_geometry_signature: Any,
        max_stale_steps: int | None,
        refresh_interval: int | None,
        step: int,
    ) -> tuple[Any, str, str]:
        """v1 approximate path for upper-band layers (lo <= layer_index < hi).

        Bands outside [lo, hi) return RECOMPUTE("out_of_band").
        Within the band, applies geometry check + staleness budget + age bound.
        """
        # Out-of-band: layers between layer-0 and lo, or >= hi.
        if not (self._band_lo <= layer_index < self._band_hi):
            self._record_recompute("out_of_band")
            return None, _RECOMPUTE, "out_of_band"

        key = DelayedKVKey(layer_index=layer_index, position=position)
        entry = self._store.get(key)

        if entry is None:
            self._record_recompute("not_found")
            return None, _RECOMPUTE, "not_found"

        ms = max_stale_steps if max_stale_steps is not None else self._max_stale_steps
        ri = refresh_interval if refresh_interval is not None else self._refresh_interval

        # 1. Geometry mismatch (always first).
        if entry.geometry_signature != current_geometry_signature:
            entry.invalidated_count += 1
            self._record_recompute("geometry_mismatch")
            return None, _RECOMPUTE, "geometry_mismatch"

        # 1b. Own-token change — UNCONDITIONAL invalidation (carried from v0
        # is_valid; spec §2). The staleness budget relaxes context_epoch drift,
        # NOT a change of the position's own token: if this position now holds a
        # different token, its K/V is wrong regardless of budget/age.
        if current_token_version != entry.computed_at_token_version:
            entry.invalidated_count += 1
            self._record_recompute("token_changed")
            return None, _RECOMPUTE, "token_changed"

        # 2. Age bound.
        if entry.reuse_approx_age >= ms:
            entry.invalidated_count += 1
            self._record_recompute("age_bound")
            return None, _RECOMPUTE, "age_bound"

        # 3. Periodic refresh.
        # step > 0: the live forward serves with step=0 (0 % k == 0 for every k) —
        # without the guard a positive interval permanently disables all reuse
        # instead of refreshing at real multiples (2026-07-02 review finding #1).
        if ri is not None and ri > 0 and step > 0 and step % ri == 0:
            entry.invalidated_count += 1
            self._record_recompute("periodic_refresh")
            return None, _RECOMPUTE, "periodic_refresh"

        # 4. Staleness budget (v1 relaxation): allow drift up to stale_epoch_budget.
        epoch_drift = current_context_epoch - entry.computed_at_context_epoch
        if epoch_drift > self._stale_epoch_budget:
            entry.invalidated_count += 1
            self._record_recompute("stale_epoch_budget_exceeded")
            return None, _RECOMPUTE, "stale_epoch_budget_exceeded"

        # Band-valid: serve as REUSE_APPROX.
        entry.reuse_approx_age += 1
        self._reuse_approx_count += 1
        return entry.kv, _REUSE_APPROX, _REUSE_SOURCE_UPPER_BAND

    def _serve_dkv_decode(
        self,
        *,
        layer_index: int,
        position: int,
        current_token_version: int,
        current_context_epoch: int,
        current_geometry_signature: Any,
        max_stale_steps: int | None,
        refresh_interval: int | None,
        step: int,
    ) -> tuple[Any, str, str]:
        """dKV one-step-delay approximate path for band layers (lo <= layer < hi).

        Identical gating to :meth:`_serve_upper_band` (geometry → own-token → age →
        refresh → stale-epoch budget) PLUS the dKV one-step delay: a decoded token's
        K/V is NOT reusable until at least ``_DKV_MIN_EPOCH_DELAY`` (=1) commit-epochs
        have elapsed since it was computed — the dKV-Cache insight that a just-decoded
        token's K/V is not yet stable at its own decode step. The delay is the ONLY
        addition over the upper band; everything else mirrors it so the spike is a
        clean A/B against the C2b structure. Served hits are tagged
        ``"dkv_decode_one_step"`` and counted on ``_dkv_decode_reuse_count`` — kept
        DISTINCT from the upper-band reuse so the spike is measured in isolation.
        """
        # Out-of-band: layers between layer-0 and lo, or >= hi.
        if not (self._band_lo <= layer_index < self._band_hi):
            self._record_recompute("out_of_band")
            return None, _RECOMPUTE, "out_of_band"

        key = DelayedKVKey(layer_index=layer_index, position=position)
        entry = self._store.get(key)
        if entry is None:
            self._record_recompute("not_found")
            return None, _RECOMPUTE, "not_found"

        ms = max_stale_steps if max_stale_steps is not None else self._max_stale_steps
        ri = refresh_interval if refresh_interval is not None else self._refresh_interval

        # 1. Geometry mismatch (always first).
        if entry.geometry_signature != current_geometry_signature:
            entry.invalidated_count += 1
            self._record_recompute("geometry_mismatch")
            return None, _RECOMPUTE, "geometry_mismatch"

        # 1b. Own-token change — UNCONDITIONAL invalidation (even past the delay):
        # a position now holding a different token has wrong K/V regardless of delay.
        if current_token_version != entry.computed_at_token_version:
            entry.invalidated_count += 1
            self._record_recompute("token_changed")
            return None, _RECOMPUTE, "token_changed"

        epoch_drift = current_context_epoch - entry.computed_at_context_epoch

        # 2. dKV one-step delay: a just-decoded token's K/V is not yet stable; it
        # becomes reusable only once at least one commit-epoch has elapsed. This is
        # the DEFINING dKV gate — NOT present in the upper band (which serves at
        # drift 0). A pending entry is a clean RECOMPUTE, not an invalidation.
        if epoch_drift < _DKV_MIN_EPOCH_DELAY:
            self._record_recompute("dkv_decode_delay_pending")
            return None, _RECOMPUTE, "dkv_decode_delay_pending"

        # 3. Age bound.
        if entry.reuse_approx_age >= ms:
            entry.invalidated_count += 1
            self._record_recompute("age_bound")
            return None, _RECOMPUTE, "age_bound"

        # 4. Periodic refresh.
        # step > 0: the live forward serves with step=0 (0 % k == 0 for every k) —
        # without the guard a positive interval permanently disables all reuse
        # instead of refreshing at real multiples (2026-07-02 review finding #1).
        if ri is not None and ri > 0 and step > 0 and step % ri == 0:
            entry.invalidated_count += 1
            self._record_recompute("periodic_refresh")
            return None, _RECOMPUTE, "periodic_refresh"

        # 5. Staleness budget upper bound (drift already >= the one-step delay).
        if epoch_drift > self._stale_epoch_budget:
            entry.invalidated_count += 1
            self._record_recompute("stale_epoch_budget_exceeded")
            return None, _RECOMPUTE, "stale_epoch_budget_exceeded"

        # Band-valid + past the one-step delay: serve as REUSE_APPROX (dKV source).
        entry.reuse_approx_age += 1
        self._dkv_decode_reuse_count += 1
        return entry.kv, _REUSE_APPROX, _REUSE_SOURCE_DKV_DECODE

    # -- commit hook --------------------------------------------------------- #

    def invalidate_on_commit(self, *, context_epoch: int) -> None:
        """Handle a context_epoch bump (a commit happened somewhere).

        For layer-0 entries, ``context_epoch`` alone does NOT invalidate —
        they are invalidated only by a token change (handled in ``serve``).
        For upper-layer entries (v1 upper_band scope), the epoch drift vs
        the stale_epoch_budget is checked lazily in ``serve()`` — no eager
        per-entry invalidation is performed here (the budget-based staleness
        model defers the decision to serve time).

        This hook exists to satisfy the commit-lifecycle contract: callers
        MUST call it after every commit.  The epoch value is bumped so
        future ``serve`` calls get the updated epoch they need.
        """
        # Bump the running context epoch so upper-band serve() can compute drift.
        self._context_epoch += 1
        # Count the invalidation event for metrics.
        self._invalidated_count += 1

    # -- private helpers ----------------------------------------------------- #

    def _record_recompute(self, reason: str) -> None:
        self._recompute_count += 1
        self._recompute_reasons[reason] = self._recompute_reasons.get(reason, 0) + 1

    # -- metrics ------------------------------------------------------------- #

    @property
    def context_epoch(self) -> int:
        """The current commit epoch (bumped once per ``invalidate_on_commit`` call)."""
        return self._context_epoch

    @property
    def scope(self) -> str:
        """The reuse scope: ``"layer0"`` (v0 exact) or ``"upper_band"`` (v1 approx)."""
        return self._scope

    @property
    def reuse_count(self) -> int:
        return self._reuse_count

    @property
    def reuse_approx_count(self) -> int:
        return self._reuse_approx_count

    @property
    def recompute_count(self) -> int:
        return self._recompute_count

    @property
    def invalidated_count(self) -> int:
        return self._invalidated_count

    @property
    def error_fix_recompute_count(self) -> int:
        return self._error_fix_recompute_count

    def increment_error_fix_recompute_count(self, count: int = 1) -> None:
        """Increment the error-fixing forced-recompute counter (called by the forward hook)."""
        self._error_fix_recompute_count += count

    def invalidate_active_window_band_on_error_fix(self, remasked_positions: Any) -> int:
        """Error-fixing forced invalidation of the active-window upper-band cache.

        Spec §3/§3a (the load-bearing C2b-v1 contract): after ANY active-window
        remask, an upper-band entry's K/V is stale because the remask perturbs the
        layer-``l-1`` context for the WHOLE active window — **not just the flagged
        position's own entries**.  So drop ALL upper-band entries (the active-window
        band cache; frozen positions never hold reusable band entries) so the
        subsequent re-decode recomputes FRESH K/V.  No ``REUSE_APPROX`` may resume
        for those (layer, position) keys until a ``store()`` refreshes them.

        Called with the list of remasked active-window positions; a non-empty list
        triggers the drop.  Returns the number of band entries dropped and bumps
        ``error_fix_recompute_count`` by the same amount (distinct from age/epoch
        refresh).  Returns 0 (no-op) when nothing was remasked.
        """
        if not list(remasked_positions):
            return 0
        to_drop = [
            key
            for key in self._store
            if self._band_lo <= key.layer_index < self._band_hi
        ]
        for key in to_drop:
            del self._store[key]
        self._error_fix_recompute_count += len(to_drop)
        return len(to_drop)

    @property
    def recompute_reasons(self) -> dict[str, int]:
        return dict(self._recompute_reasons)

    def metrics(self) -> dict[str, int]:
        """Flat metrics dict for logging / assertions.

        ``delayed_kv_reuse_approx_count`` is included only when > 0 (or when
        scope='upper_band') to preserve the C2b-v0 invariant: in v0 (scope='layer0')
        REUSE_APPROX is never emitted and the key must be absent from the dict.
        Callers that assert 'delayed_kv_reuse_approx_count' not in metrics must
        continue to pass on v0 runs.
        """
        m: dict[str, int] = {
            "delayed_kv_reuse_count": self._reuse_count,
            "delayed_kv_recompute_count": self._recompute_count,
            "delayed_kv_invalidated_count": self._invalidated_count,
            "delayed_kv_error_fix_recompute_count": self._error_fix_recompute_count,
            "delayed_kv_entries": len(self._store),
        }
        # Include reuse_approx_count only when the upper_band scope is active
        # (v0/layer0 never emits REUSE_APPROX; the key's absence is an invariant).
        if self._scope == "upper_band" or self._reuse_approx_count > 0:
            m["delayed_kv_reuse_approx_count"] = self._reuse_approx_count
        # dKV one-step-delay spike counter — surfaced separately so its reuse is
        # never conflated with the C2b upper-band approx count.
        if self._scope == "dkv_decode" or self._dkv_decode_reuse_count > 0:
            m["delayed_kv_dkv_decode_reuse_count"] = self._dkv_decode_reuse_count
        for reason, count in self._recompute_reasons.items():
            m[f"delayed_kv_recompute_{reason}"] = count
        return m
