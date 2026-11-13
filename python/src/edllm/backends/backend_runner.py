"""Backend execution boundary for full, active-window and diagnostic forwards.

The engine owns request/session assembly; DiffusionIterationManager owns policy,
commit and stop. Runners execute tensors from the supplied GraphIR-derived plan,
kernel table and cache owner. They never call back into engine orchestration.

MLXBackendRunner delegates to mlx_forward and its exact-D2F implementation.
MetalBackendRunner exposes the existing primitive seam but rejects unsupported
model execution. CUDA device sessions retain their separate typed provider ABI;
this protocol does not claim that every backend implements every decode policy.
MLX imports remain lazy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..graph import DenseTransformerPlan, GraphIR
    from .window_plan import WindowExecutionResult


@runtime_checkable
class BackendRunner(Protocol):
    """The forward surface the dLLM engine delegates to.

    Any execution backend (MLX today; Metal/ANE next) implements
    ``full_forward``, ``execute_window``, and ``layer0_trace``. The engine routes
    its supported forward and diagnostic surfaces through this interface, so
    swapping the backend never touches the engine / decode / adapter.
    """

    def full_forward(
        self,
        *,
        weights: Any,
        model_config: dict[str, Any],
        token_ids: list[int],
        attention_mask_kind: str,
        kv_cache_state: Any | None = None,
        timing_metrics: dict[str, float | int | bool] | None = None,
        graph: "GraphIR | None" = None,
        backend_plan: "DenseTransformerPlan | None" = None,
        state_cache_manager: Any | None = None,
        masked_positions: list[int] | None = None,
        mask_geometry: Any | None = None,
        kernels: Any | None = None,
    ) -> Any:
        """Return full-vocab logits (a numpy/array of shape ``[seq_len, vocab]``).

        ``kv_cache_state`` (the layer-0 KV cache) and ``timing_metrics`` are
        threaded through unchanged so the cache reuse/invalidate flow and the
        forward-stage timing stay intact. ``backend_plan`` is the validated
        GraphIR lowering used by optimized backends; Product callers provide it.
        ``graph``/``model_config`` remain a compatibility input for standalone
        low-level parity callers and are compiled at this boundary when no plan
        is supplied.
        ``state_cache_manager`` (the StateCacheManager owning the approximate
        prefix / block KV cache) is ``None`` for the byte-exact path; when supplied
        the forward reuses cached K/V (block-only) after the first full forward.
        ``masked_positions`` (the still-masked frontier) is consumed only by the
        block KV cache; ``None`` => derived/unused (prefix path stays identical).
        ``mask_geometry`` (``prompt_len`` / ``block_size``) is the per-request
        runtime geometry consumed ONLY by the ``d2f_hybrid`` attention mask
        kind; it flows as forward-call data (the mask FAMILY is Graph IR data),
        and ``None`` keeps every other kind byte-identical. ``kernels`` (M2-1)
        is the per-op KernelTable the forward dispatches its swappable ops
        through; ``None`` => the forward resolves the default (all-MLX) table
        from ``model_config`` — byte-identical.
        """
        ...

    def execute_window(
        self,
        *,
        plan: Any,
        weights: Any,
        model_config: dict[str, Any],
        token_ids: list[int],
        attention_mask_kind: str,
        kv_cache_state: Any | None = None,
        timing_metrics: dict[str, float | int | bool] | None = None,
        graph: "GraphIR | None" = None,
        backend_plan: "DenseTransformerPlan | None" = None,
        state_cache_manager: Any | None = None,
        masked_positions: list[int] | None = None,
        mask_geometry: Any | None = None,
        kernels: Any | None = None,
    ) -> "WindowExecutionResult":
        """Execute the forward for ``plan.forward_scope`` positions only.

        Returns a :class:`~edllm.backends.window_plan.WindowExecutionResult`
        with three fields:

        ``logits`` — **decode-selection-ready** logits (logits convention
        already applied) for exactly the rows in ``plan.forward_scope``, in
        ``forward_scope`` order, shape ``[len(plan.forward_scope), vocab]``.
        The convention (``dream_ar_shift`` | ``raw``) is sourced from the
        :class:`~edllm.graph.GraphIR` (built from ``model_config`` when
        ``graph`` is ``None``). This is mathematically equivalent to
        ``_logits_for_selection(full_forward(...), logits_convention)[forward_scope]``.

        ``positions`` — absolute sequence positions corresponding to each row
        of ``logits``, always ``tuple(plan.forward_scope)``.  The explicit
        position map lets callers index the full logit space without implicit
        offsets.

        ``cache_mutations`` — cache deltas produced this step.  Always an empty
        tuple ``()`` today: the cache mutates via ``StateCacheManager`` side-effect
        and there is no consumer for populated mutations, so Phase 3A keeps ``()``
        (no fabricated data) — reserved for a future consumer.  Callers must treat
        ``()`` as "no mutations" and must not assume a dict or any other container type.

        All keyword arguments mirror ``full_forward`` exactly so the parity
        wrapper can delegate to ``full_forward`` transparently, and a future
        real windowed forward can consume the same arguments.
        """
        ...

    def layer0_trace(
        self,
        *,
        weights: Any,
        prompt_token_ids: list[int],
        block_token_ids: list[int],
        attention_mask_kind: str,
        mask_geometry: Any | None = None,
        backend_plan: "DenseTransformerPlan",
        kernels: Any,
    ) -> dict[str, dict]:
        """Trace layer 0 with the same resident weights and plan as execution."""
        ...


class MLXBackendRunner:
    """MLX execution through backend-owned math; no dependency on MLXDreamEngine."""

    __slots__ = ()

    def full_forward(
        self,
        *,
        weights: Any,
        model_config: dict[str, Any],
        token_ids: list[int],
        attention_mask_kind: str,
        kv_cache_state: Any | None = None,
        timing_metrics: dict[str, float | int | bool] | None = None,
        graph: "GraphIR | None" = None,
        backend_plan: "DenseTransformerPlan | None" = None,
        state_cache_manager: Any | None = None,
        masked_positions: list[int] | None = None,
        mask_geometry: Any | None = None,
        kernels: Any | None = None,
    ) -> Any:
        # Local import keeps this module import-light and avoids any import cycle;
        # the attribute is dereferenced at CALL time so the monkeypatch seam on
        # ``mlx_forward._forward_full_logits_from_weights`` is honored through here.
        from . import mlx_forward

        return mlx_forward._forward_full_logits_from_weights(
            weights=weights,
            model_config=model_config,
            token_ids=token_ids,
            attention_mask_kind=attention_mask_kind,
            kv_cache_state=kv_cache_state,
            timing_metrics=timing_metrics,
            graph=graph,
            backend_plan=backend_plan,
            state_cache_manager=state_cache_manager,
            masked_positions=masked_positions,
            mask_geometry=mask_geometry,
            kernels=kernels,
        )

    def windowed_forward(
        self,
        *,
        weights: Any,
        model_config: dict[str, Any],
        token_ids: list[int],
        attention_mask_kind: str,
        kv_cache_state: Any | None = None,
        timing_metrics: dict[str, float | int | bool] | None = None,
        graph: "GraphIR | None" = None,
        backend_plan: "DenseTransformerPlan | None" = None,
        state_cache_manager: Any | None = None,
        masked_positions: list[int] | None = None,
        mask_geometry: Any | None = None,
        kernels: Any | None = None,
        return_sparse: bool = False,
        output_rows: tuple[int, ...] | None = None,
    ) -> Any:
        """Execute the shared forward dispatcher, allowing a sparse exact-D2F result.

        All cache branches share preparation and accounting in mlx_forward.
        return_sparse is a hint: cache branches may still return dense logits.
        The full-context path can select output_rows before projection;
        execute_window handles either container and owns the convention map.
        """
        from . import mlx_forward

        return mlx_forward._forward_full_logits_from_weights(
            weights=weights,
            model_config=model_config,
            token_ids=token_ids,
            attention_mask_kind=attention_mask_kind,
            kv_cache_state=kv_cache_state,
            timing_metrics=timing_metrics,
            graph=graph,
            backend_plan=backend_plan,
            state_cache_manager=state_cache_manager,
            masked_positions=masked_positions,
            mask_geometry=mask_geometry,
            kernels=kernels,
            return_sparse=return_sparse,
            output_rows=output_rows,
        )

    def execute_window(
        self,
        *,
        plan: Any,
        weights: Any,
        model_config: dict[str, Any],
        token_ids: list[int],
        attention_mask_kind: str,
        kv_cache_state: Any | None = None,
        timing_metrics: dict[str, float | int | bool] | None = None,
        graph: "GraphIR | None" = None,
        backend_plan: "DenseTransformerPlan | None" = None,
        state_cache_manager: Any | None = None,
        masked_positions: list[int] | None = None,
        mask_geometry: Any | None = None,
        kernels: Any | None = None,
    ) -> "WindowExecutionResult":
        """Resolves the logits convention, delegates to ``windowed_forward``,
        then selects ``forward_scope`` rows under that convention.

        It performs no independent forward math: it resolves the
        :class:`~edllm.graph.GraphIR` (building it from ``model_config`` via
        ``_graph_for_forward`` when ``graph`` is ``None``, using the identical seam
        that ``_forward_full_logits_from_weights`` uses internally), then calls
        ``windowed_forward`` (the active-window forward entry — NOT the
        ``full_forward`` oracle) exactly once with the resolved graph, and returns a
        :class:`~edllm.backends.window_plan.WindowExecutionResult` whose
        ``logits`` are mathematically equivalent to
        ``_logits_for_selection(full_forward(...), logits_convention)[list(plan.forward_scope)]``.

        Projection on fewer rows can change floating-point rounding; compare
        numerical tolerances and decoded tokens against the dense oracle.

        **Two selection routes, one result.** ``windowed_forward`` is
        asked for the sparse shape, and this method branches on WHAT CAME BACK,
        never on what was requested:

        * :class:`~edllm.backends.mlx_d2f_forward.SparseForwardLogits` (the
          exact-D2F branch or full-context projection) — ``select_sparse_logits`` folds the
          convention into the row map and gathers straight from the ``[R, V]``
          rows. No ``[seq_len, vocab]`` array is built anywhere on this route.
        * anything else — a dense cache result or a test double — takes the original
          route below unchanged: apply the convention to the whole ``[S, V]``
          tensor, then slice. Returning dense is always valid, so a backend or
          stub that ignores the hint keeps working untouched.

        Both routes produce the same values when their dataflow is valid
        (spec §2/§3: each shifted row depends on exactly one raw row). The sparse
        production route additionally requires that every requested decision row
        was computed; historical zero-fill remains available only to the
        component parity/debug helper, not to token commit.

        ``cache_mutations`` remains an empty tuple ``()`` because exact K/V
        ownership and mutation are applied synchronously by
        ``state_cache_manager`` inside the delegated forward. Returning a second
        mutation stream here would duplicate that ownership contract.

        mlx.core stays lazy: this method imports ``mlx_forward`` and
        ``mlx_d2f_forward`` only at call time (the same pattern as
        ``full_forward`` above), so constructing ``MLXBackendRunner`` or importing
        this module does not pull in MLX.
        """
        from . import mlx_forward
        from .mlx_d2f_forward import SparseForwardLogits, _required_raw_row, select_sparse_logits
        from .window_plan import WindowExecutionResult

        # Resolve the GraphIR ONCE here.  When the caller supplies a graph we use
        # it directly (engine hot-path: graph already built and cached per
        # generation).  When graph is None we build it from model_config via the
        # IDENTICAL seam that _forward_full_logits_from_weights uses internally
        # (mlx_forward._graph_for_forward).  We then pass the resolved graph into
        # full_forward so both this wrapper and the forward math see the same IR —
        # no double-build, no inconsistency.
        backend_plan = mlx_forward._backend_plan_for_forward(
            weights=weights,
            model_config=model_config,
            graph=graph,
            backend_plan=backend_plan,
        )
        graph = backend_plan.graph
        logits_convention = backend_plan.logits_convention

        # Delegate the entire forward to windowed_forward (the active-window entry,
        # NOT the full_forward oracle) — no math duplicated here. windowed_forward
        # shares the dispatcher and the same layer math. Pass the resolved graph
        # so it need not rebuild it.
        #
        # Honest forwarded-row count: the per-forward DELTA of
        # ``mlx_timing_active_tokens_total`` IS the number of query rows actually
        # computed this forward — ``seq_len`` for a full forward, and the reduced
        # compute window ``len(plan.forward_positions())`` when a cache routes to
        # an active/block-only path (mlx_forward corrects the initial full-window
        # count before each reduced return). We read the metric before and after
        # so the loop's
        # trace metric can report the real computed-row count instead of
        # ``len(token_ids)`` (the full window prefix). When no timing_metrics is
        # supplied, a dense result can only report its materialized row count;
        # reduced production paths always provide the corrected counter.
        # SHARED REQUEST BOUNDARY -- validated BEFORE the backend runs.
        #
        # Range checking used to live only inside the sparse selector, which runs
        # AFTER windowed_forward has already executed: an out-of-range scope
        # raised, but only once the forward had computed layers, mutated the
        # exact-KV cache and promoted pending blocks. Raising after side effects
        # is not fail-closed. Worse, the dense fallback below never checked at
        # all -- it indexed the selection tensor with the raw scope list, so a
        # negative position silently wrapped and returned a real logits row, and
        # an out-of-range one read past the array (MLX does not bounds-check
        # gathers).
        #
        # One check, before either route, covering both. `seq_len` is the
        # backend's own view of the canvas: `len(token_ids)`.
        _seq_len = len(token_ids)

        # window_end first: it narrows the canvas, so scope must be checked
        # against IT, not merely against len(token_ids). Checking only the token
        # count let `window_end=5, forward_scope=(7,), token_count=9` through --
        # a position outside the declared window reached the backend -- and a
        # negative window_end was not rejected at all.
        _window_end = plan.window_end
        if _window_end is not None and not (0 <= _window_end <= _seq_len):
            raise ValueError(
                f"window_end={_window_end} is outside [0, {_seq_len}]; the plan "
                "and the token buffer disagree about the canvas."
            )
        _limit = _seq_len if _window_end is None else _window_end
        _bad = [p for p in plan.forward_scope if p < 0 or p >= _limit]
        if _bad:
            raise ValueError(
                f"forward_scope contains out-of-range positions {_bad}; positions "
                f"are absolute canvas indices in [0, {_limit})"
                + (f" (window_end={_window_end})" if _window_end is not None
                   else f" (len(token_ids)={_seq_len})")
                + ". Rejected before the forward runs, so no cache mutation or "
                "block promotion has occurred."
            )

        # None when the counter is ABSENT, not 0. A missing key means the
        # backend did not report row accounting; substituting 0 turned "unknown"
        # into "measured zero rows", which reads as a perfect sparse forward.
        _before = (
            timing_metrics.get("mlx_timing_active_tokens_total")
            if timing_metrics is not None
            else None
        )
        _before = None if _before is None else int(_before)
        full_logits = self.windowed_forward(
            weights=weights,
            model_config=model_config,
            token_ids=token_ids,
            attention_mask_kind=attention_mask_kind,
            kv_cache_state=kv_cache_state,
            timing_metrics=timing_metrics,
            graph=graph,
            backend_plan=backend_plan,
            state_cache_manager=state_cache_manager,
            masked_positions=masked_positions,
            mask_geometry=mask_geometry,
            kernels=kernels,
            # Gate 1: ask for the sparse shape. This is a HINT — the branch below
            # reads the returned type, so a dispatcher branch or test double that
            # ignores it is served by the unchanged dense route. The row
            # accounting above is untouched by it (the flag selects a container,
            # not which rows are computed).
            return_sparse=True,
            # Choose RAW rows before output projection. Dream's convention maps
            # a decision row to its predecessor (with row zero repeated).
            output_rows=tuple(sorted({
                _required_raw_row(position, logits_convention=logits_convention)
                for position in plan.forward_scope
            })),
        )
        _after = (
            timing_metrics.get("mlx_timing_active_tokens_total")
            if timing_metrics is not None
            else None
        )
        if _after is not None:
            # The backend accounted for rows. An absent counter BEFORE means the
            # accumulator started empty, which is a real zero, not a guess.
            _forwarded_rows = int(_after) - (_before or 0)
        elif isinstance(full_logits, SparseForwardLogits):
            # Output projection may be sparse even when all transformer query
            # rows ran. Never report output rows as saved attention work.
            _forwarded_rows = (
                full_logits.forwarded_rows if full_logits.forwarded_rows is not None
                else len(full_logits.rows)
            )
        else:
            # A dense container alone does not prove every row was computed:
            # approximate caches scatter reduced rows into a dense result. Those
            # production paths publish a corrected counter above; this fallback
            # exists only for full-row backends/test doubles without accounting.
            _forwarded_rows = len(token_ids)

        if isinstance(full_logits, SparseForwardLogits):
            # Gate 1 route: the producer kept its own [R, V] rows. Selection folds
            # the convention into the row map (spec §2) and gathers directly, so
            # NEITHER dense [seq_len, vocab] intermediate is built — not the
            # producer's scatter target, not this method's shifted copy. A
            # requested row that the producer did not compute is a scheduler/cache
            # lifecycle defect, not a valid zero-logit token decision, so the
            # production boundary rejects it explicitly.
            selected_logits = select_sparse_logits(
                full_logits,
                plan.forward_scope,
                logits_convention=logits_convention,
                require_computed=True,
            )
        else:
            # Dense fallback for cache branches and test doubles that do not
            # return sparse projections.
            #
            # Apply the convention to the FULL [seq_len, vocab] logits first.
            # Dream uses "dream_ar_shift" (position-dependent shift that couples
            # adjacent positions — cannot compress to forward_scope first then
            # shift). "raw" is identity, so this is byte-identical to the old raw
            # path for unshifted models.
            selection_full = mlx_forward._logits_for_selection(
                full_logits, logits_convention=logits_convention
            )

            # Select exactly the forward_scope rows in forward_scope order.
            # Using a list index preserves order and is compatible with both
            # numpy arrays and mlx arrays (both support list indexing).
            scope = list(plan.forward_scope)
            if scope:  # noqa: SIM108 - keep the empty-slice contract beside branch
                selected_logits = selection_full[scope]
            else:
                # Empty forward_scope: backend-preserving zero-row slice.
                # full_logits[:0] gives a zero-row array without converting to
                # numpy, preserving dtype and device for both numpy and mlx
                # arrays.
                selected_logits = selection_full[:0]

        # cache_mutations stays () : the cache mutates via StateCacheManager
        # side-effect and there is no consumer for populated mutations, so 3A keeps
        # () (no fabricated data) — reserved for a future consumer.
        # forwarded_rows carries the honest computed-row count (see above); both
        # the scope and empty-scope cases share this single return path.
        return WindowExecutionResult(
            logits=selected_logits,
            positions=tuple(plan.forward_scope),
            cache_mutations=(),
            forwarded_rows=_forwarded_rows,
        )

    def layer0_trace(
        self,
        *,
        weights: Any,
        prompt_token_ids: list[int],
        block_token_ids: list[int],
        attention_mask_kind: str,
        mask_geometry: Any | None = None,
        backend_plan: "DenseTransformerPlan",
        kernels: Any,
    ) -> dict[str, dict]:
        from . import mlx_forward

        return mlx_forward._layer0_trace_from_weights(
            weights=weights,
            prompt_token_ids=prompt_token_ids,
            block_token_ids=block_token_ids,
            attention_mask_kind=attention_mask_kind,
            mask_geometry=mask_geometry,
            backend_plan=backend_plan,
            kernels=kernels,
        )


class MetalBackendRunner:
    """``BackendRunner`` shell for the first Metal primitive slice.

    Stage 4 proves the second-backend substrate one op at a time. Full model
    forward and diagnostic traces are intentionally fail-closed until enough ops
    are parity-gated to make a real Metal forward honest.
    """

    __slots__ = ()

    def full_forward(
        self,
        *,
        weights: Any,
        model_config: dict[str, Any],
        token_ids: list[int],
        attention_mask_kind: str,
        kv_cache_state: Any | None = None,
        timing_metrics: dict[str, float | int | bool] | None = None,
        graph: "GraphIR | None" = None,
        backend_plan: "DenseTransformerPlan | None" = None,
        state_cache_manager: Any | None = None,
        masked_positions: list[int] | None = None,
        mask_geometry: Any | None = None,
        kernels: Any | None = None,
    ) -> Any:
        raise NotImplementedError(
            "MetalBackendRunner full_forward is not implemented; Stage 4 only "
            "parity-gates individual Metal ops"
        )

    def windowed_forward(
        self,
        *,
        weights: Any,
        model_config: dict[str, Any],
        token_ids: list[int],
        attention_mask_kind: str,
        kv_cache_state: Any | None = None,
        timing_metrics: dict[str, float | int | bool] | None = None,
        graph: "GraphIR | None" = None,
        backend_plan: "DenseTransformerPlan | None" = None,
        state_cache_manager: Any | None = None,
        masked_positions: list[int] | None = None,
        mask_geometry: Any | None = None,
        kernels: Any | None = None,
        return_sparse: bool = False,
    ) -> Any:
        raise NotImplementedError(
            "MetalBackendRunner windowed_forward is not implemented; Stage 4 only "
            "parity-gates individual Metal ops"
        )

    def execute_window(
        self,
        *,
        plan: Any,
        weights: Any,
        model_config: dict[str, Any],
        token_ids: list[int],
        attention_mask_kind: str,
        kv_cache_state: Any | None = None,
        timing_metrics: dict[str, float | int | bool] | None = None,
        graph: "GraphIR | None" = None,
        backend_plan: "DenseTransformerPlan | None" = None,
        state_cache_manager: Any | None = None,
        masked_positions: list[int] | None = None,
        mask_geometry: Any | None = None,
        kernels: Any | None = None,
    ) -> "WindowExecutionResult":
        raise NotImplementedError(
            "MetalBackendRunner execute_window is not implemented; Stage 4 only "
            "parity-gates individual Metal ops"
        )

    def layer0_trace(
        self,
        *,
        weights: Any,
        prompt_token_ids: list[int],
        block_token_ids: list[int],
        attention_mask_kind: str,
        mask_geometry: Any | None = None,
        backend_plan: "DenseTransformerPlan",
        kernels: Any,
    ) -> dict[str, dict]:
        raise NotImplementedError(
            "MetalBackendRunner layer0_trace is not implemented; Stage 4 only "
            "parity-gates individual Metal ops"
        )

    def rms_norm(self, x: Any, weight: Any, eps: float) -> Any:
        from . import metal_ops

        return metal_ops.rms_norm(x, weight, eps)
