"""StateCacheManager — the cache-strategy boundary for the dLLM forward.

This module owns the *decision* side of the engine's caching: which cache
strategies exist and, per forward, whether the prefix can be reused (block-only
forward) or must be recomputed (full forward that repopulates the cache). It is
intentionally independent from ``mlx.core`` — it stores opaque tensor handles the
forward hands it and never does tensor math — so importing it (or constructing a
manager) pulls in no MLX/Metal/torch. The MLX forward (``mlx_dream``) consults a
manager and supplies/reads the per-layer K/V tensors.

The first LOAD-BEARING strategy is the **approximate prefix KV cache**
(:class:`PrefixKVCacheStrategy`), Fast-dLLM-style. The generation sequence is
``x = [prefix...(committed), block...(masks being filled)]``; today every
diffusion step re-forwards the whole ``[prefix + block]``. With the prefix cache:

* the FIRST forward for a given prefix runs full and caches the per-layer prefix
  K/V (positions ``[0:prompt_len)``);
* subsequent forwards run BLOCK-ONLY — the forward computes Q/K/V only for the
  block (positions ``[prompt_len:)``, plus the single boundary row
  ``prompt_len-1`` the Dream shift reads), attends block-Q against
  ``concat(cached_prefix_K, block_K)``, and never recomputes the prefix hidden
  states.

This is an APPROXIMATION (bidirectional attention lets the prefix K/V drift
slightly as the block fills); Fast-dLLM showed it is ~lossless under confident
decoding, and it is quality- + latency-gated, NOT byte-exact. A changed prefix
(new turn) invalidates the cache and triggers a fresh full forward; an optional
``refresh_interval`` periodically upgrades a block-only step to a full forward
that repopulates the prefix K/V (the quality safety valve).

The existing layer-0 KV cache (``mlx_cache``) stays as the prior, measured-neutral
strategy; this manager is the new boundary the plan calls for, with the prefix
cache as its first non-neutral member.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class _PrefixLayerKV:
    """Opaque per-layer cached prefix K/V (whatever the forward stored)."""

    key: Any
    value: Any



class PrefixKVCacheStrategy:
    """Approximate within-turn prefix KV cache (the load-bearing strategy).

    Owns the per-layer prefix K/V tensors and the reuse decision. The forward
    drives it with three calls per forward:

    * :meth:`begin_forward` — tells the strategy the current ``token_ids`` and
      gets back a plan (full vs block-only, and whether to store).
    * :meth:`store_layer` — during a full forward, hands the strategy the prefix
      K/V for a layer (positions ``[0:prompt_len)``).
    * :meth:`layer_prefix_kv` — during a block-only forward, returns the cached
      prefix K/V for a layer to concatenate with the freshly-projected block K/V.

    All tensors are opaque to this module (no mlx import); it only stores/returns
    the handles the forward gives it.
    """

    __slots__ = (
        "_prompt_len",
        "_refresh_interval",
        "_layers",
        "_prefix_signature",
        "_full_forward_count",
        "_block_only_count",
        "_steps_since_refresh",
    )

    def __init__(self, *, prompt_len: int, refresh_interval: int = 0) -> None:
        self._prompt_len = int(prompt_len)
        # 0 (or negative) => never periodically refresh after the initial populate.
        self._refresh_interval = max(0, int(refresh_interval))
        self._layers: dict[int, _PrefixLayerKV] = {}
        self._prefix_signature: tuple[int, ...] | None = None
        self._full_forward_count = 0
        self._block_only_count = 0
        self._steps_since_refresh = 0

    # -- read-side decision -------------------------------------------------- #

    @property
    def prompt_len(self) -> int:
        return self._prompt_len

    def is_populated(self) -> bool:
        return self._prefix_signature is not None and bool(self._layers)

    def should_full_forward(self, *, token_ids: list[int]) -> bool:
        """Whether THIS forward must be a full (populate) forward.

        True when: nothing cached yet, the prefix changed (new turn), or a
        periodic refresh is due. False => the cache serves this forward block-only.
        """
        return self._plan(token_ids)[0]

    def _prefix_of(self, token_ids: list[int]) -> tuple[int, ...]:
        return tuple(int(t) for t in token_ids[: self._prompt_len])

    def _plan(self, token_ids: list[int]) -> tuple[bool, tuple[int, ...]]:
        signature = self._prefix_of(token_ids)
        if not self.is_populated():
            return True, signature
        if signature != self._prefix_signature:
            return True, signature
        if self._refresh_interval > 0 and self._steps_since_refresh >= self._refresh_interval:
            return True, signature
        return False, signature

    # -- forward-driven lifecycle ------------------------------------------- #

    def begin_forward(self, *, token_ids: list[int]) -> "PrefixForwardPlan":
        """Open a forward: decide full vs block-only and update counters.

        On a full forward the cache is cleared so :meth:`store_layer` repopulates
        it fresh for this prefix. The boundary row index (``prompt_len-1``, the
        position the Dream shift reads) and the block start are returned so the
        forward can carve the block-only work.
        """
        full, signature = self._plan(token_ids)
        seq_len = len(token_ids)
        prefix_len = min(self._prompt_len, seq_len)
        # Boundary row the shift reads (last prefix position). With an empty
        # prefix there is no boundary; the block then starts at 0.
        boundary = max(0, prefix_len - 1)
        if full:
            self._layers = {}
            self._prefix_signature = signature
            self._full_forward_count += 1
            self._steps_since_refresh = 0
        else:
            self._block_only_count += 1
            self._steps_since_refresh += 1
        return PrefixForwardPlan(
            strategy=self,
            is_full_forward=full,
            prefix_len=prefix_len,
            boundary=boundary,
            block_start=prefix_len,
            seq_len=seq_len,
        )

    def store_layer(self, *, layer_index: int, key: Any, value: Any) -> None:
        """Cache a layer's prefix K/V (positions ``[0:prompt_len)``)."""
        self._layers[int(layer_index)] = _PrefixLayerKV(key=key, value=value)

    def layer_prefix_kv(self, *, layer_index: int) -> tuple[Any, Any]:
        entry = self._layers.get(int(layer_index))
        if entry is None:
            raise KeyError(
                f"prefix KV cache has no layer {layer_index}; "
                "a block-only forward ran before the cache was populated"
            )
        return entry.key, entry.value

    def metrics(self) -> dict[str, int | bool]:
        return {
            "mlx_prefix_kv_cache_enabled": True,
            "mlx_prefix_kv_populated": self.is_populated(),
            "mlx_prefix_kv_prompt_len": self._prompt_len,
            "mlx_prefix_kv_refresh_interval": self._refresh_interval,
            "mlx_prefix_kv_full_forward_count": self._full_forward_count,
            "mlx_prefix_kv_block_only_count": self._block_only_count,
            "mlx_prefix_kv_cached_layers": len(self._layers),
        }


@dataclass(slots=True)
class PrefixForwardPlan:
    """Per-forward plan handed back by :meth:`PrefixKVCacheStrategy.begin_forward`.

    ``is_full_forward`` selects the path: a full forward computes every position
    and (via :meth:`store_layer`) repopulates the prefix K/V; a block-only forward
    computes only the rows ``[boundary : seq_len)`` and reuses the cached prefix
    K/V. ``block_start`` is the first FRESHLY-projected block position
    (``prompt_len``); ``boundary`` (``prompt_len-1``) is included in the computed
    rows so the Dream logit shift has a real boundary logit, but its K/V comes
    from the cache (it is NOT re-projected into the block K).
    """

    strategy: PrefixKVCacheStrategy
    is_full_forward: bool
    prefix_len: int
    boundary: int
    block_start: int
    seq_len: int

    def forward_positions(self) -> list[int]:
        """The absolute rows a block-only forward computes hidden states for."""
        return list(range(self.boundary, self.seq_len))

    def block_positions(self) -> list[int]:
        """The absolute positions whose K/V are freshly projected (appended)."""
        return list(range(self.block_start, self.seq_len))


@dataclass(slots=True)
class _BlockLayerKV:
    """Opaque per-layer committed-block K/V keyed by absolute position.

    Holds, for one layer, the cached K/V of every COMMITTED block position (the
    Fast-dLLM approximation: a position's K/V frozen at the step it committed).
    Position -> opaque (key, value) row handle the forward stored.
    """

    keys: dict[int, Any] = field(default_factory=dict)
    values: dict[int, Any] = field(default_factory=dict)


class BlockKVCacheStrategy:
    """Approximate ``prefix + committed-block`` KV cache (Phase D throughput).

    EXTENDS :class:`PrefixKVCacheStrategy` from prefix-only to ``prefix +
    committed-block``. The generation sequence is ``x = [prefix...(committed),
    block...(committing/masks)]``; the profile showed 51–55% of every diffusion
    step is wasted re-forwarding already-committed positions. This strategy caches
    BOTH the fixed prefix K/V (positions ``[0:prompt_len)``, like the prefix cache)
    AND the K/V of each committed BLOCK position, so a step forwards only the
    still-masked frontier.

    Per block-only forward the cached region is ``[0 : first_masked)`` and the
    forwarded rows are ``[first_masked-1 : seq_len)`` (the masked frontier plus the
    single shift-feeder boundary row the Dream logit shift reads). K/V for the
    forwarded rows are freshly projected EXCEPT for any committed positions inside
    that span, whose cached K/V are spliced in — so the result is byte-identical to
    the full forward when the committed tokens are held frozen.

    Lifecycle (driven by the forward + the decode loop, all tensors opaque):

    * :meth:`begin_forward` — decide full vs block-only; returns a
      :class:`BlockForwardPlan` carving the forward rows / cached vs fresh splits.
    * :meth:`store_prefix_layer` — during a full forward, cache the prefix K/V.
    * :meth:`store_frontier_layer` — each block-only forward hands the per-layer
      K/V SNAPSHOT for its forwarded rows (so a position that commits this step has
      its commit-time K/V available to promote).
    * :meth:`committed_block_kv` — return cached committed-block K/V for splicing.
    * :meth:`observe_commit` + :meth:`promote_committed` — after a step commits,
      promote the just-committed positions' K/V from the snapshot into the
      persistent committed-block cache for subsequent steps.

    A changed prefix (new turn) invalidates EVERYTHING (prefix + committed block)
    and forces a fresh full forward. An optional ``refresh_interval`` upgrades
    every K-th block-only step to a full forward that repopulates the prefix K/V
    AND clears the committed-block cache (the quality safety valve); because that
    full forward re-projects every row, all already-committed block positions are
    re-registered as pending and re-captured from the refresh forward's own
    snapshot, so post-refresh block-only plans keep full K/V coverage.
    """

    __slots__ = (
        "_prompt_len",
        "_refresh_interval",
        "_prefix_layers",
        "_block_layers",
        "_prefix_signature",
        "_full_forward_count",
        "_block_only_count",
        "_steps_since_refresh",
        "_committed_positions",
        "_frontier_snapshots",
        "_pending_commits",
    )

    def __init__(self, *, prompt_len: int, refresh_interval: int = 0) -> None:
        self._prompt_len = int(prompt_len)
        self._refresh_interval = max(0, int(refresh_interval))
        self._prefix_layers: dict[int, _PrefixLayerKV] = {}
        self._block_layers: dict[int, _BlockLayerKV] = {}
        self._prefix_signature: tuple[int, ...] | None = None
        self._full_forward_count = 0
        self._block_only_count = 0
        self._steps_since_refresh = 0
        # Absolute positions whose committed-block K/V is persisted (across layers).
        self._committed_positions: set[int] = set()
        # Per-layer snapshot of the CURRENT step's forwarded-row K/V (position ->
        # (key, value)); promoted into _block_layers when a position commits.
        self._frontier_snapshots: dict[int, dict[int, tuple[Any, Any]]] = {}
        # Positions that committed since the last promote_committed().
        self._pending_commits: list[int] = []

    # -- read-side ----------------------------------------------------------- #

    @property
    def prompt_len(self) -> int:
        return self._prompt_len

    def is_populated(self) -> bool:
        return self._prefix_signature is not None and bool(self._prefix_layers)

    def has_committed_position(self, position: int) -> bool:
        return int(position) in self._committed_positions

    def _prefix_of(self, token_ids: list[int]) -> tuple[int, ...]:
        return tuple(int(t) for t in token_ids[: self._prompt_len])

    def _plan_full(self, token_ids: list[int]) -> tuple[bool, tuple[int, ...]]:
        signature = self._prefix_of(token_ids)
        if not self.is_populated():
            return True, signature
        if signature != self._prefix_signature:
            return True, signature
        if self._refresh_interval > 0 and self._steps_since_refresh >= self._refresh_interval:
            return True, signature
        return False, signature

    def should_full_forward(self, *, token_ids: list[int], masked_positions: list[int]) -> bool:
        del masked_positions  # signature parity with the forward driver
        return self._plan_full(token_ids)[0]

    # -- forward-driven lifecycle ------------------------------------------- #

    def begin_forward(
        self, *, token_ids: list[int], masked_positions: list[int]
    ) -> "BlockForwardPlan":
        """Open a forward: full (populate) vs block-only over the masked frontier.

        A committed position's K/V is cached (reused) ONLY after it has been
        freshly projected in a forward where it held its committed token — captured
        by :meth:`promote_pending` at THAT forward's end. So the block-only forward
        window always reaches back far enough to cover (a) the masked frontier's
        shift-feeder boundary AND (b) any committed-but-not-yet-captured (pending)
        position, projecting them fresh from their committed tokens this step. Only
        already-CAPTURED committed positions strictly before that window are served
        purely from cache. This is the exactness invariant: every cached committed
        K/V was produced from the committed token while the position sat in the
        window; everything in the window is recomputed exactly; so a frozen
        committed sequence reproduces the full forward — while the bulk of committed
        positions (captured, behind the frontier) skip the attention+MLP compute,
        erasing the 51–55% per-step waste.
        """
        full, signature = self._plan_full(token_ids)
        seq_len = len(token_ids)
        prefix_len = min(self._prompt_len, seq_len)
        if full:
            # A full forward rebuilds the world: clear prefix + committed-block.
            self._prefix_layers = {}
            self._block_layers = {}
            self._committed_positions = set()
            self._prefix_signature = signature
            self._full_forward_count += 1
            self._steps_since_refresh = 0
            self._frontier_snapshots = {}
            # Every committed block position is freshly re-projected by THIS full
            # forward from its committed token, so register all of them as
            # pending: promote_pending() at this forward's end captures their
            # exact frozen K/V from the full forward's snapshot. (Clearing them
            # here instead — the old behavior — made a refresh full forward drop
            # every already-committed position from subsequent block-only K/V
            # coverage: neither cached nor fresh, silently wrong attention.)
            masked_set = {int(p) for p in masked_positions}
            self._pending_commits = [
                p for p in range(prefix_len, seq_len) if p not in masked_set
            ]
            boundary = max(0, prefix_len - 1)
            return BlockForwardPlan(
                is_full_forward=True,
                prefix_len=prefix_len,
                boundary=boundary,
                block_start=prefix_len,
                seq_len=seq_len,
                cached_block=[],
                fresh_block=list(range(prefix_len, seq_len)),
                boundary_is_cached=False,
            )

        self._block_only_count += 1
        self._steps_since_refresh += 1
        self._frontier_snapshots = {}
        masked = sorted(int(p) for p in masked_positions if int(p) >= prefix_len)
        if masked:
            first_masked = masked[0]
        else:
            # Degenerate: nothing masked in the block -> forward just the boundary
            # so the shift still has a real feeder row (rare; keeps callers sane).
            first_masked = seq_len
        # The window must include the shift-feeder boundary AND every pending
        # (committed-but-uncaptured) block position, so they are projected fresh.
        window_start = first_masked - 1
        if self._pending_commits:
            window_start = min(window_start, min(self._pending_commits))
        boundary = max(0, window_start)
        forward_positions = list(range(boundary, seq_len))
        # Already-captured committed positions strictly before the window: cached.
        cached_block = [
            p for p in range(prefix_len, boundary) if p in self._committed_positions
        ]
        # Freshly-projected K/V: every BLOCK position inside the window (masked,
        # just-committed/pending). A prefix boundary row (< prefix_len) is NOT
        # re-projected (its K/V is in the prefix cache).
        fresh_block = [p for p in forward_positions if p >= prefix_len]
        boundary_is_cached = False
        return BlockForwardPlan(
            is_full_forward=False,
            prefix_len=prefix_len,
            boundary=boundary,
            block_start=prefix_len,
            seq_len=seq_len,
            cached_block=cached_block,
            fresh_block=fresh_block,
            boundary_is_cached=boundary_is_cached,
        )

    def store_prefix_layer(self, *, layer_index: int, key: Any, value: Any) -> None:
        """Cache a layer's prefix K/V (positions ``[0:prompt_len)``)."""
        self._prefix_layers[int(layer_index)] = _PrefixLayerKV(key=key, value=value)

    def layer_prefix_kv(self, *, layer_index: int) -> tuple[Any, Any]:
        entry = self._prefix_layers.get(int(layer_index))
        if entry is None:
            raise KeyError(
                f"block KV cache has no prefix layer {layer_index}; "
                "a block-only forward ran before the cache was populated"
            )
        return entry.key, entry.value

    def store_frontier_layer(
        self, *, layer_index: int, positions: list[int], key: list[Any], value: list[Any]
    ) -> None:
        """Record THIS forward's per-row K/V snapshot for the forwarded ``positions``.

        ``key`` / ``value`` are per-position row handles (already RoPE'd / GQA-
        repeated — exactly what attention consumed), produced from each position's
        CURRENT token. They are held until :meth:`promote_pending` (called at the
        forward's end) copies the rows of any pending-committed positions present
        here into the persistent committed-block cache.
        """
        layer = int(layer_index)
        snap = self._frontier_snapshots.setdefault(layer, {})
        # strict=True, not a bare zip. positions/key/value are three parallel
        # lists; a length mismatch silently stored only the shorter prefix, so
        # a forward that produced fewer K/V rows than positions turned into a
        # cache missing rows -- with no error, and the damage surfacing later
        # as wrong output rather than a failure here. Measured: 2 positions and
        # 1 key stored one row and returned normally.
        for position, key_row, value_row in zip(positions, key, value, strict=True):
            snap[int(position)] = (key_row, value_row)

    def committed_block_kv(
        self, *, layer_index: int, positions: list[int]
    ) -> tuple[list[Any], list[Any]]:
        """Cached committed-block K/V rows for ``positions`` (in the given order)."""
        layer = self._block_layers.get(int(layer_index))
        keys: list[Any] = []
        values: list[Any] = []
        for position in positions:
            pos = int(position)
            if layer is None or pos not in layer.keys:
                raise KeyError(
                    f"block KV cache has no committed K/V for layer {layer_index} "
                    f"position {pos}"
                )
            keys.append(layer.keys[pos])
            values.append(layer.values[pos])
        return keys, values

    def observe_commit(self, *, position: int, token_id: int) -> None:
        """Record that ``position`` committed (awaiting capture). token_id incidental."""
        del token_id
        pos = int(position)
        if pos not in self._pending_commits:
            self._pending_commits.append(pos)

    def promote_pending(self) -> None:
        """Capture pending-committed positions present in the CURRENT snapshot.

        Called at the END of each block-only / populate forward. A pending position
        present in this forward's snapshot was just projected from its committed
        token, so its K/V is the correct frozen value to cache; it is moved into the
        persistent committed-block cache and marked captured (so a later step serves
        it from cache). A pending position NOT in the snapshot (none, since the
        window covers all pending) stays pending. Subsequent block-only forwards
        then splice these instead of re-projecting them.
        """
        if not self._pending_commits:
            return
        still_pending: list[int] = []
        for position in self._pending_commits:
            captured_any = False
            for layer_index, snap in self._frontier_snapshots.items():
                row = snap.get(position)
                if row is None:
                    continue
                layer = self._block_layers.setdefault(layer_index, _BlockLayerKV())
                layer.keys[position] = row[0]
                layer.values[position] = row[1]
                captured_any = True
            if captured_any:
                self._committed_positions.add(position)
            else:
                still_pending.append(position)
        self._pending_commits = still_pending

    # Back-compat alias: the early bookkeeping tests + any direct driver may call
    # promote_committed(); it now means "capture pending from the current snapshot".
    promote_committed = promote_pending

    def metrics(self) -> dict[str, int | bool]:
        return {
            "mlx_block_kv_cache_enabled": True,
            "mlx_block_kv_populated": self.is_populated(),
            "mlx_block_kv_prompt_len": self._prompt_len,
            "mlx_block_kv_refresh_interval": self._refresh_interval,
            "mlx_block_kv_full_forward_count": self._full_forward_count,
            "mlx_block_kv_block_only_count": self._block_only_count,
            "mlx_block_kv_cached_prefix_layers": len(self._prefix_layers),
            "mlx_block_kv_committed_positions": len(self._committed_positions),
        }


@dataclass(slots=True)
class BlockForwardPlan:
    """Per-forward plan handed back by :meth:`BlockKVCacheStrategy.begin_forward`.

    ``is_full_forward`` selects the path. On a block-only forward the engine
    computes hidden states for :meth:`forward_positions` (``[boundary : seq_len)``,
    the masked frontier plus the shift-feeder boundary row), freshly projects K/V
    only for :meth:`fresh_block_positions`, and splices the cached committed-block
    K/V for :meth:`cached_block_positions` — the union, in absolute order, is the
    full key/value set each frontier query attends. ``boundary_is_cached`` is True
    when the boundary row is itself a committed position: its hidden state is still
    computed (the shift reads its logit) but its K/V comes from the committed cache,
    not a fresh projection.
    """

    is_full_forward: bool
    prefix_len: int
    boundary: int
    block_start: int
    seq_len: int
    cached_block: list[int]
    fresh_block: list[int]
    boundary_is_cached: bool

    def forward_positions(self) -> list[int]:
        """Absolute rows a block-only forward computes hidden states for."""
        return list(range(self.boundary, self.seq_len))

    def cached_block_positions(self) -> list[int]:
        """Committed-block positions whose K/V come from the cache (after prefix)."""
        return list(self.cached_block)

    def fresh_block_positions(self) -> list[int]:
        """Positions whose K/V are freshly projected this step (still-masked etc.)."""
        return list(self.fresh_block)



@dataclass(slots=True)
class D2FForwardPlan:
    """Provider-neutral exact-D2F forward geometry.

    On an active-block-only forward the engine computes hidden states for
    :meth:`forward_positions` (``[boundary : active_end)``), freshly projects
    K/V only for :meth:`fresh_positions` (``[cached_len : active_end)`` — the
    pending blocks being captured plus the active block), and attends them
    after the cached parts, so the kv axis is exactly the absolute positions
    ``[0 : kv_len)`` with ``kv_len == active_end``. A boundary row before
    ``cached_len`` (the Dream shift feeder) gets its hidden state recomputed but
    its K/V from the cache. Rows at/after ``active_end`` are dead this forward
    (their logits stay zero): the d2f mask forbids the active block from seeing
    them, so they cannot affect exactness.
    """

    is_full_forward: bool
    prompt_len: int
    block_size: int
    seq_len: int
    cached_len: int
    boundary: int
    active_block: int
    active_start: int
    active_end: int
    pending_blocks: list[int]
    kv_len: int

    def forward_positions(self) -> list[int]:
        """Absolute rows this forward computes hidden states for."""
        if self.is_full_forward:
            return list(range(self.seq_len))
        return list(range(self.boundary, self.active_end))

    def fresh_positions(self) -> list[int]:
        """Absolute positions whose K/V are freshly projected this forward."""
        if self.is_full_forward:
            return list(range(self.seq_len))
        return list(range(self.cached_len, self.active_end))

    def pending_block_rows(self) -> list[tuple[int, int, int]]:
        """``(block_index, start, end)`` for each pending block, in order."""
        rows: list[tuple[int, int, int]] = []
        for block_index in self.pending_blocks:
            start = self.prompt_len + block_index * self.block_size
            end = min(start + self.block_size, self.seq_len)
            rows.append((block_index, start, end))
        return rows



@dataclass(slots=True)
class StateCacheManager:
    """Owns the engine's cache strategies (the boundary the plan calls for).

    Carries the approximate prefix/block strategies and, when exact-D2F is
    selected, the injected native scheduling facade. It owns no exact-D2F
    tensor storage: the C++ manager behind that facade is the sole owner.
    """

    prefix_kv: PrefixKVCacheStrategy | None = None
    block_kv: BlockKVCacheStrategy | None = None
    d2f_exact_kv: Any = None

    def __post_init__(self) -> None:
        if self.d2f_exact_kv is not None and (
            self.prefix_kv is not None or self.block_kv is not None
        ):
            raise ValueError(
                "the exact d2f KV cache cannot share a StateCacheManager with "
                "the approximate prefix/block KV caches (bidirectional-mask "
                "strategies); enable exactly one"
            )

    @classmethod
    def for_prefix_kv(cls, *, prompt_len: int, refresh_interval: int = 0) -> "StateCacheManager":
        return cls(
            prefix_kv=PrefixKVCacheStrategy(
                prompt_len=prompt_len, refresh_interval=refresh_interval
            )
        )

    @classmethod
    def for_block_kv(cls, *, prompt_len: int, refresh_interval: int = 0) -> "StateCacheManager":
        return cls(
            block_kv=BlockKVCacheStrategy(
                prompt_len=prompt_len, refresh_interval=refresh_interval
            )
        )

    def prefix_metrics(self) -> dict[str, int | bool]:
        if self.prefix_kv is None:
            return empty_prefix_metrics()
        return self.prefix_kv.metrics()

    def block_metrics(self) -> dict[str, int | bool]:
        if self.block_kv is None:
            return empty_block_metrics()
        return self.block_kv.metrics()

    def d2f_metrics(self) -> dict[str, int | bool | str]:
        if self.d2f_exact_kv is None:
            return empty_d2f_metrics()
        return self.d2f_exact_kv.metrics()



def empty_prefix_metrics() -> dict[str, int | bool]:
    """Neutral prefix-cache metrics for the cache-OFF path (byte-identical)."""
    return {
        "mlx_prefix_kv_cache_enabled": False,
        "mlx_prefix_kv_populated": False,
        "mlx_prefix_kv_prompt_len": 0,
        "mlx_prefix_kv_refresh_interval": 0,
        "mlx_prefix_kv_full_forward_count": 0,
        "mlx_prefix_kv_block_only_count": 0,
        "mlx_prefix_kv_cached_layers": 0,
    }


def empty_block_metrics() -> dict[str, int | bool]:
    """Neutral block-cache metrics for the cache-OFF path (byte-identical)."""
    return {
        "mlx_block_kv_cache_enabled": False,
        "mlx_block_kv_populated": False,
        "mlx_block_kv_prompt_len": 0,
        "mlx_block_kv_refresh_interval": 0,
        "mlx_block_kv_full_forward_count": 0,
        "mlx_block_kv_block_only_count": 0,
        "mlx_block_kv_cached_prefix_layers": 0,
        "mlx_block_kv_committed_positions": 0,
    }


def empty_d2f_metrics() -> dict[str, int | bool]:
    """Neutral exact-d2f-cache metrics for the cache-OFF path."""
    return {
        "mlx_d2f_exact_kv_cache_enabled": False,
        "mlx_d2f_exact_kv_populated": False,
        "mlx_d2f_exact_kv_prompt_len": 0,
        "mlx_d2f_exact_kv_block_size": 0,
        "mlx_d2f_exact_kv_full_forward_count": 0,
        "mlx_d2f_exact_kv_active_only_count": 0,
        "mlx_d2f_exact_kv_frozen_blocks": 0,
        "mlx_d2f_exact_kv_pending_blocks": 0,
    }


__all__ = [
    "BlockForwardPlan",
    "BlockKVCacheStrategy",
    "D2FForwardPlan",
    "PrefixForwardPlan",
    "PrefixKVCacheStrategy",
    "StateCacheManager",
    "empty_block_metrics",
    "empty_d2f_metrics",
    "empty_prefix_metrics",
]
