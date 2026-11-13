"""MLX tensor execution owned by BackendRunner, independent of decode orchestration.

Product callers supply the compiled GraphIR plan. Config/weight inference is
kept only for standalone low-level callers. Cache owners and kernel tables are
injected; this module neither creates sessions nor chooses decode policies.
MLX is imported lazily by the operations that need it.
"""
from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

from ..graph import (
    DenseTransformerPlan, GraphIR, build_graph_ir,
    compile_dense_transformer_plan, same_mask_kind_family,
)
from ..model_definitions import runtime_architecture_for_config
from . import mlx_ops
from .kernel_table import (
    KernelTable,
    canonical_compute_dtype_name as _canonical_compute_dtype_name,
    resolve_kernel_table,
)

# Geometry defaults apply only to legacy standalone forward callers.
_GRAPH_GEOMETRY_DEFAULTS: dict[str, Any] = {
    "layer_count": 1,
    "attention_heads": 1,
    "num_kv_heads": 1,
    "head_dim": 1,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
}


def _graph_for_forward(model_config: dict[str, Any]) -> GraphIR:
    """Build the forward's :class:`GraphIR` from ``model_config``.

    This is the Engine Foundation Stage-2 chokepoint: the fast MLX forward reads
    its family-conditional flags (``apply_qk_norm`` / ``logits_convention``) from
    the IR rather than from an inline ``architecture``-string ``if``. The emitter
    resolves those flags via the unified architecture registry keyed by exact
    model identity. Runtime callers do not pass a second copy beside the IR.

    Resilience for forward-only callers: the standalone forward
    (``forward_full_logits`` / the parity + KV-cache paths) is invoked with a
    minimal ``model_config`` that may omit identity and geometry. This builder
    supplies benign geometry defaults and uses conservative structural defaults
    only when identity is absent. A present but unknown identity fails closed
    whenever a structural value must be inferred; fully explicit low-level test
    configs remain supported without pretending to be Product model identities.
    """
    cfg = dict(model_config)
    for key, default in _GRAPH_GEOMETRY_DEFAULTS.items():
        cfg.setdefault(key, default)
    # Resolve network structure independently of decode-policy validation. BD3,
    # for example, has QK norm whether or not this low-level call carries its
    # fixed-quota scheduler settings.
    if not {"uses_qk_norm", "uses_shifted_logits", "bias_present"} <= cfg.keys():
        has_identity = bool(cfg.get("architecture") or cfg.get("architecture_family"))
        architecture = runtime_architecture_for_config(cfg) if has_identity else None
        cfg.setdefault(
            "uses_qk_norm", architecture.uses_qk_norm if architecture else False
        )
        cfg.setdefault(
            "uses_shifted_logits",
            architecture.uses_shifted_logits if architecture else False,
        )
        cfg.setdefault(
            "bias_present",
            architecture.qkv_bias_present if architecture else False,
        )
    return build_graph_ir(cfg)


def _ir_apply_qk_norm(graph: GraphIR) -> bool:
    """Read the per-head QK-RMSNorm flag from the GraphIR (the ``qk_norm`` op).

    Mirrors how ``NumpyRefBackendRunner`` reads the IR: the flag is op DATA on the
    layer ``qk_norm`` op, identical across layers. Qwen3 (A2DQwen3 / qwen3_mdlm)
    normalizes Q/K per head over ``head_dim`` before RoPE; Dream / LLaDA do not, so
    the IR carries ``apply_qk_norm=False`` and the forward stays byte-identical.

    Fail-closed (F3): the MLX forward is layer-UNIFORM by design — it reads this one
    flag and applies it to every layer. The GraphIR / NumpyRef are per-layer, so a
    NON-uniform ``apply_qk_norm`` graph would make the MLX forward (norms all layers
    off the layer-0 flag) and NumpyRef (per-layer) disagree SILENTLY. So validate
    uniformity across all layers and raise ``ValueError`` instead — the layer-uniform
    contract is enforced, not silently assumed.
    """

    def _layer_flag(layer) -> bool:
        qk_op = next(op for op in layer.ops if op.type == "qk_norm")
        return bool(qk_op.params.get("apply_qk_norm", False))

    flags = [_layer_flag(layer) for layer in graph.layers]
    if len(set(flags)) > 1:
        raise ValueError(
            "non-uniform apply_qk_norm across layers is unsupported by the "
            "layer-uniform MLX forward (it reads the layer-0 flag for every "
            f"layer); per-layer flags={flags}"
        )
    return flags[0] if flags else False


def _logits_for_selection(logits: Any, *, logits_convention: str):
    """Return the logits the decode selection should read.

    The convention is package metadata carried by the :class:`GraphIR`
    (``dream_ar_shift`` | ``raw``). Dream (Qwen2-based) uses the shifted AR-style
    convention: position ``i`` predicts token ``i`` from the hidden state at
    ``i-1``, so the selection reads ``shift_logits_for_dream(logits)``. Unshifted
    masked-diffusion models predict position ``i`` directly, so the selection reads
    the RAW logits. Stage 2 sources this from the IR (``logits_convention``) rather
    than an arch-derived ``uses_shifted_logits`` bool.
    """

    # Fail-closed (F4): only the two known conventions are accepted. An unknown
    # value (a typo, or a future convention this build does not implement) must NOT
    # silently fall through to the raw view — it raises.
    if logits_convention == "dream_ar_shift":
        return mlx_ops.shift_logits_for_dream(logits)
    if logits_convention == "raw":
        return logits
    raise ValueError(f"unknown logits_convention: {logits_convention!r}")


def _layer0_trace_from_weights(
    *,
    weights: dict[str, Any],
    prompt_token_ids: list[int],
    block_token_ids: list[int],
    attention_mask_kind: str,
    mask_geometry: Any | None,
    backend_plan: DenseTransformerPlan,
    kernels: KernelTable,
) -> dict[str, dict]:
    """Observe layer 0 through the production MLX layer implementation."""
    import mlx.core as mx

    token_ids = [*prompt_token_ids, *block_token_ids]
    if not same_mask_kind_family(
        attention_mask_kind, backend_plan.attention_mask_kind
    ):
        raise ValueError(
            f"attention_mask_kind {attention_mask_kind!r} disagrees with "
            f"GraphIR {backend_plan.attention_mask_kind!r}"
        )

    hidden = _embedding_rows(
        weights,
        token_ids,
        weight_key=backend_plan.embedding_weight,
    )
    trace = {"embed": _trace_stage(hidden)}
    seq_len = len(token_ids)
    cos, sin = _rope_tables(
        seq_len=seq_len,
        head_dim=backend_plan.head_dim,
        rope_theta=backend_plan.rope_theta,
        rope_scaling=backend_plan.rope_scaling,
    )

    def observer(stage: str, value: Any) -> None:
        trace[stage] = _trace_stage(value)

    hidden = _transformer_layer_forward(
        weights=weights,
        hidden=hidden,
        layer_index=0,
        n_heads=backend_plan.n_heads,
        num_kv_heads=backend_plan.num_kv_heads,
        head_dim=backend_plan.head_dim,
        eps=backend_plan.eps,
        cos=mx.array(cos),
        sin=mx.array(sin),
        attention_mask_kind=attention_mask_kind,
        mask_geometry=mask_geometry,
        apply_qk_norm=backend_plan.apply_qk_norm,
        kernels=kernels,
        observer=observer,
    )
    mx.eval(hidden)
    return trace


def _layer_count(weights: Any, config: dict[str, Any]) -> int:
    layer_count = int(config.get("layer_count", 0) or 0)
    if layer_count > 0:
        return layer_count
    layer_indices = []
    for key in _weight_keys(weights):
        prefix = "model.layers."
        suffix = ".self_attn.q_proj.weight"
        if key.startswith(prefix) and key.endswith(suffix):
            layer_indices.append(int(key[len(prefix) : -len(suffix)]))
    if not layer_indices:
        raise ValueError("unable to infer layer_count from weights")
    return max(layer_indices) + 1


def _backend_plan_for_forward(
    *,
    weights: Any,
    model_config: dict[str, Any],
    graph: GraphIR | None,
    backend_plan: DenseTransformerPlan | None,
) -> DenseTransformerPlan:
    """Resolve one dense plan; Product callers always provide the compiled one.

    The config/weight fallback is retained only for direct low-level parity and
    test callers. Product loading compiles GraphIR before constructing the MLX
    engine, so the optimized path never infers structure from resident weights.
    """

    if backend_plan is not None:
        if graph is not None and backend_plan.graph is not graph:
            raise ValueError("compiled backend plan does not belong to GraphIR")
        return backend_plan
    if graph is None:
        compatibility_config = dict(model_config)
        if int(compatibility_config.get("layer_count", 0) or 0) <= 0:
            # Monkeypatched/direct callers may provide no real archive.
            # _graph_for_forward retains its one-layer compatibility default.
            with contextlib.suppress(ValueError):
                compatibility_config["layer_count"] = _layer_count(
                    weights,
                    compatibility_config,
                )
        graph = _graph_for_forward(compatibility_config)
    return compile_dense_transformer_plan(graph)


def _forward_full_logits_from_weights(
    *,
    weights: Any,
    model_config: dict[str, Any],
    token_ids: list[int],
    attention_mask_kind: str,
    kv_cache_state: Any | None = None,
    timing_metrics: dict[str, float | int | bool] | None = None,
    graph: GraphIR | None = None,
    backend_plan: DenseTransformerPlan | None = None,
    state_cache_manager: Any | None = None,
    masked_positions: list[int] | None = None,
    mask_geometry: Any | None = None,
    kernels: KernelTable | None = None,
    return_sparse: bool = False,
    output_rows: tuple[int, ...] | None = None,
):
    """Route one forward to the branch its cache configuration selects.

    Full forward keeps its dense oracle result. Window execution may request
    sparse logits plus raw ``output_rows``; the full-context branch selects
    hidden rows before final norm/lm_head. Attention still sees every required
    context row. Existing cache branches retain their own row execution and
    may return dense or sparse results; callers inspect the returned type.
    """
    import mlx.core as mx

    forward_started_at = _timer_now()
    backend_plan = _backend_plan_for_forward(
        weights=weights,
        model_config=model_config,
        graph=graph,
        backend_plan=backend_plan,
    )
    graph = backend_plan.graph
    # M2-1 op-dispatch seam: every swappable op below goes through ``kernels``
    # (rms_norm/linear/rope/swiglu/attention). The engine resolves the
    # table once per generation and threads it here (like ``graph``); a
    # standalone caller (parity/debug) gets the model_config-resolved default —
    # the current ``mlx_ops`` functions, so the math is byte-identical.
    if kernels is None:
        kernels = resolve_kernel_table(model_config)
    n_layers = len(backend_plan.layers)
    if timing_metrics is not None:
        timing_metrics["mlx_timing_layer_count"] = n_layers
    n_heads = backend_plan.n_heads
    head_dim = backend_plan.head_dim
    num_kv_heads = backend_plan.num_kv_heads
    eps = backend_plan.eps
    apply_qk_norm = backend_plan.apply_qk_norm
    # The mask FAMILY is Graph IR data; the kwarg (what the caller asked to run)
    # must agree with it at the family level — running either interpretation
    # silently would diverge from the IR-driven NumpyRef backend. The mask
    # GEOMETRY (d2f_hybrid prompt_len/block_size) is forward-call data, validated
    # up front: required for d2f_hybrid, rejected (no silently-dead argument)
    # for the no-mask kinds. Pure validation — the bidirectional path is
    # byte-identical.
    ir_mask_kind = backend_plan.attention_mask_kind
    if not same_mask_kind_family(attention_mask_kind, ir_mask_kind):
        raise ValueError(
            f"attention_mask_kind kwarg {attention_mask_kind!r} disagrees with "
            f"the Graph IR attention mask_kind {ir_mask_kind!r}; refusing to run "
            "either interpretation silently"
        )
    # Exact d2f cross-block KV cache (StateCacheManager.d2f_exact_kv, D2F-2):
    # valid ONLY under the d2f_hybrid mask (the exactness thesis lives in the
    # block-causal structure; any other kind fails closed). The strategy owns
    # the mask geometry — a caller-supplied ``mask_geometry`` must match it,
    # and when omitted (the engine path) it is derived from the strategy.
    d2f_strategy = getattr(state_cache_manager, "d2f_exact_kv", None)
    if d2f_strategy is not None:
        if attention_mask_kind != "d2f_hybrid":
            raise ValueError(
                "the exact d2f cross-block KV cache is valid only with "
                "attention_mask_kind='d2f_hybrid'; got "
                f"{attention_mask_kind!r}"
            )
        strategy_geometry = {
            "prompt_len": d2f_strategy.prompt_len,
            "block_size": d2f_strategy.block_size,
        }
        if mask_geometry is None:
            mask_geometry = strategy_geometry
        elif mlx_ops._validated_mask_geometry(mask_geometry) != (
            d2f_strategy.prompt_len,
            d2f_strategy.block_size,
        ):
            raise ValueError(
                f"mask geometry {dict(mask_geometry)!r} disagrees with the d2f "
                f"exact KV cache strategy geometry {strategy_geometry!r}"
            )
    mlx_ops.validate_attention_kind(
        kind=attention_mask_kind, mask_geometry=mask_geometry
    )

    seq_len = len(token_ids)
    # Active-window accounting (Commit 1): every decode forward funnels through here
    # with its forwarded token_ids (the active window for d2f / window_end; the full
    # canvas for the no-cache strategies). Reduced cache paths initially add the
    # full window here, then replace it with len(plan.forward_positions()) before
    # their early return. Keeping the correction next to each branch prevents a
    # dense scatter result from being mistaken for dense computation.
    if timing_metrics is not None:
        timing_metrics["mlx_timing_active_tokens_total"] = (
            int(timing_metrics.get("mlx_timing_active_tokens_total", 0)) + seq_len
        )
    cos, sin = _rope_tables(
        seq_len=seq_len,
        head_dim=head_dim,
        rope_theta=backend_plan.rope_theta,
        rope_scaling=backend_plan.rope_scaling,
    )
    cos_mx = mx.array(cos)
    sin_mx = mx.array(sin)

    # Approximate prefix KV cache (StateCacheManager): when a prefix-KV strategy is
    # supplied, the first forward for a prefix runs full (and caches per-layer
    # prefix K/V) while subsequent forwards run BLOCK-ONLY (reuse the cached prefix
    # K/V, project only the block). With no manager (or its prefix_kv is None) this
    # whole branch is skipped and the forward is byte-identical to before.
    prefix_strategy = getattr(state_cache_manager, "prefix_kv", None)
    plan = None
    if prefix_strategy is not None:
        plan = prefix_strategy.begin_forward(token_ids=token_ids)
        if not plan.is_full_forward:
            if timing_metrics is not None:
                timing_metrics["mlx_timing_active_tokens_total"] = (
                    int(timing_metrics["mlx_timing_active_tokens_total"])
                    - seq_len
                    + len(plan.forward_positions())
                )
            # Gate 1: ``return_sparse`` is deliberately NOT threaded here. This
            # approximate, default-OFF branch keeps returning dense [S, V]
            # (spec §1 "OUT"); the consumer's returned-type branch handles it.
            return _block_only_forward_from_weights(
                weights=weights,
                token_ids=token_ids,
                plan=plan,
                prefix_strategy=prefix_strategy,
                n_layers=n_layers,
                n_heads=n_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                eps=eps,
                cos=cos_mx,
                sin=sin_mx,
                attention_mask_kind=attention_mask_kind,
                apply_qk_norm=apply_qk_norm,
                timing_metrics=timing_metrics,
                forward_started_at=forward_started_at,
                kernels=kernels,
            )

    # Approximate block KV cache (StateCacheManager.block_kv): extends the prefix
    # cache from prefix-only to prefix+committed-block. Each step forwards ONLY the
    # still-masked frontier (plus the shift-feeder boundary row), splicing cached
    # K/V for the prefix AND for committed block positions. With no block strategy
    # this whole branch is skipped — byte-identical. The masked frontier is derived
    # from token_ids vs the mask id supplied by the engine (config-driven).
    block_strategy = getattr(state_cache_manager, "block_kv", None)
    block_plan = None
    if block_strategy is not None:
        if masked_positions is None:
            mask_id = _mask_token_id_for(model_config)
            masked_positions = (
                [i for i, t in enumerate(token_ids) if int(t) == mask_id]
                if mask_id is not None
                else []
            )
        block_plan = block_strategy.begin_forward(
            token_ids=token_ids, masked_positions=list(masked_positions)
        )
        if not block_plan.is_full_forward:
            if timing_metrics is not None:
                timing_metrics["mlx_timing_active_tokens_total"] = (
                    int(timing_metrics["mlx_timing_active_tokens_total"])
                    - seq_len
                    + len(block_plan.forward_positions())
                )
            # Gate 1: ``return_sparse`` is deliberately NOT threaded here either
            # — same reason as the prefix branch above (approximate, default-OFF,
            # spec §1 "OUT"). It keeps returning dense [S, V].
            return _block_kv_forward_from_weights(
                weights=weights,
                token_ids=token_ids,
                plan=block_plan,
                block_strategy=block_strategy,
                n_layers=n_layers,
                n_heads=n_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                eps=eps,
                cos=cos_mx,
                sin=sin_mx,
                attention_mask_kind=attention_mask_kind,
                apply_qk_norm=apply_qk_norm,
                timing_metrics=timing_metrics,
                forward_started_at=forward_started_at,
                kernels=kernels,
            )

    # Exact d2f cross-block KV cache (D2F-2): the first forward for a prompt
    # runs full (populating the prompt K/V and capturing any pre-frozen blocks);
    # subsequent forwards run ACTIVE-BLOCK-ONLY — recompute only the active
    # block (+ pending frozen blocks being captured + the shift-feeder row),
    # attending the exact cached prompt + frozen-block K/V via the non-square
    # d2f mask. With no d2f strategy this branch is skipped — byte-identical.
    d2f_plan = None
    if d2f_strategy is not None:
        d2f_plan = d2f_strategy.begin_forward(token_ids=token_ids)
        if not d2f_plan.is_full_forward:
            if timing_metrics is not None:
                # Honest active-window accounting: the shared accumulator above
                # added the full window seq_len, but the active-block-only forward
                # computes hidden states for only len(forward_positions()) Q
                # rows (feeder + pending blocks + active block). Correct the
                # over-count so the metric reflects the real Q-row reduction.
                fwd_rows = len(d2f_plan.forward_positions())
                timing_metrics["mlx_timing_active_tokens_total"] = (
                    int(timing_metrics["mlx_timing_active_tokens_total"])
                    - seq_len
                    + fwd_rows
                )
            return _d2f_exact_forward_from_weights(
                weights=weights,
                token_ids=token_ids,
                plan=d2f_plan,
                d2f_strategy=d2f_strategy,
                n_layers=n_layers,
                n_heads=n_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                eps=eps,
                cos=cos_mx,
                sin=sin_mx,
                mask_geometry=mask_geometry,
                apply_qk_norm=apply_qk_norm,
                timing_metrics=timing_metrics,
                forward_started_at=forward_started_at,
                kernels=kernels,
                # Gate 1: the ONLY branch that honours the shape hint. False
                # (every pre-Gate-1 caller) keeps the dense [S, V] scatter
                # byte-identical; True returns SparseForwardLogits instead. No
                # forward math is selected by this flag.
                return_sparse=return_sparse,
            )
    hidden = _embedding_rows(
        weights,
        token_ids,
        weight_key=backend_plan.embedding_weight,
    )
    # On a full forward WITH a prefix strategy, collect each layer's prefix K/V
    # (positions [0:prompt_len)) for reuse by later block-only forwards.
    prefix_store = None
    if plan is not None and plan.is_full_forward:
        prefix_len = plan.prefix_len

        def prefix_store(layer_index, key_full, value_full):
            prefix_strategy.store_layer(
                layer_index=layer_index,
                key=key_full[:prefix_len],
                value=value_full[:prefix_len],
            )
    elif block_plan is not None and block_plan.is_full_forward:
        # Block-KV full forward (populate): cache the prefix K/V AND snapshot the
        # whole block's per-row K/V keyed by position, so any block position that
        # commits THIS step has its commit-time K/V ready to promote. The hook gets
        # the full compact, RoPE'd K/V consumed by grouped attention.
        block_prefix_len = block_plan.prefix_len
        block_positions = list(range(block_prefix_len, seq_len))

        def prefix_store(layer_index, key_full, value_full):
            block_strategy.store_prefix_layer(
                layer_index=layer_index,
                key=key_full[:block_prefix_len],
                value=value_full[:block_prefix_len],
            )
            block_strategy.store_frontier_layer(
                layer_index=layer_index,
                positions=block_positions,
                key=[key_full[p : p + 1] for p in block_positions],
                value=[value_full[p : p + 1] for p in block_positions],
            )
    elif d2f_plan is not None and d2f_plan.is_full_forward:
        # d2f populate forward: cache the prompt K/V (exact — prompt rows attend
        # only to the prompt) and capture any block frozen BEFORE this forward
        # directly from the full computation (its tokens are final in token_ids,
        # so its K/V here IS the exact frozen value).
        d2f_prompt_len = d2f_plan.prompt_len
        d2f_pending_rows = d2f_plan.pending_block_rows()

        def prefix_store(layer_index, key_full, value_full):
            if d2f_prompt_len > 0:
                d2f_strategy.capture_prompt_layer(
                    layer_index=layer_index,
                    key=key_full[:d2f_prompt_len],
                    value=value_full[:d2f_prompt_len],
                )
            for block_index, row_start, row_end in d2f_pending_rows:
                d2f_strategy.capture_block_layer(
                    block_index=block_index,
                    layer_index=layer_index,
                    key=key_full[row_start:row_end],
                    value=value_full[row_start:row_end],
                )

    attention_mask = mlx_ops.prepare_attention_mask(
        kind=attention_mask_kind, q_len=seq_len, kv_len=seq_len,
        mask_geometry=mask_geometry,
    )
    for layer_index in range(n_layers):
        layer_started_at = _timer_now()
        hidden = _transformer_layer_forward(
            weights=weights,
            hidden=hidden,
            layer_index=layer_index,
            n_heads=n_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            eps=eps,
            cos=cos_mx,
            sin=sin_mx,
            attention_mask_kind=attention_mask_kind,
            token_ids=token_ids,
            kv_cache_state=kv_cache_state,
            apply_qk_norm=apply_qk_norm,
            prefix_store=prefix_store,
            mask_geometry=mask_geometry,
            kernels=kernels,
            attention_mask=attention_mask,
        )
        mx.eval(hidden)
        if timing_metrics is not None:
            _accumulate_timing_ms(
                timing_metrics,
                "mlx_timing_layer0_forward_ms"
                if layer_index == 0
                else "mlx_timing_upper_layers_forward_ms",
                layer_started_at,
            )

    del attention_mask
    final_started_at = _timer_now()
    projected_rows = output_rows if return_sparse else None
    if projected_rows is not None:
        # A shrinking frontier otherwise allocates a different vocabulary-sized
        # buffer every round, growing the allocator cache despite a lower live
        # peak. Bucket projection shapes, never beyond the full context size.
        # Padding repeats an independent row; it cannot affect requested logits.
        count = len(projected_rows)
        capacity = min(seq_len, 1 << (count - 1).bit_length()) if count else 0
        gather_rows = projected_rows + projected_rows[-1:] * (capacity - count)
        hidden = hidden[list(gather_rows)] if gather_rows else hidden[:0]
    hidden = kernels.rms_norm(
        hidden,
        mx.array(_weight(weights, backend_plan.final_norm_weight)),
        eps,
    )
    # Final logits in fp32 regardless of compute dtype: the lm_head matmul already
    # accumulates in fp32 internally, and the selection / argmax / parity
    # comparison read fp32 (numpy cannot interpret a bf16 buffer). ``mx.array`` is a
    # no-op on an mlx array and lifts a numpy stub (test seam); on the fp32 path the
    # astype is a no-op -> byte-identical.
    logits = mx.array(
        kernels.linear(hidden, _linear_weight(weights, backend_plan.lm_head_weight))
    ).astype(mx.float32)
    mx.eval(logits)
    # A block-KV populate/refresh full forward snapshots the whole block; capture
    # any pending-committed positions from it (a refresh mid-decode re-freezes their
    # committed-token K/V). On the populate forward there is nothing pending yet.
    if (
        block_plan is not None
        and block_plan.is_full_forward
        and block_strategy is not None
    ):
        block_strategy.promote_pending()
    # A d2f populate forward captures any pre-frozen blocks (stored by the hook
    # above); promote them so the next forward serves them from cache.
    if d2f_plan is not None and d2f_plan.is_full_forward:
        d2f_strategy.promote_pending()
    if timing_metrics is not None:
        timing_metrics["mlx_timing_full_context_output_rows_total"] = int(
            timing_metrics.get("mlx_timing_full_context_output_rows_total", 0)
        ) + int(hidden.shape[0])
        _accumulate_timing_ms(
            timing_metrics,
            "mlx_timing_final_norm_lm_head_ms",
            final_started_at,
        )
        _accumulate_timing_ms(
            timing_metrics,
            "mlx_timing_forward_total_ms",
            forward_started_at,
        )
    if projected_rows is not None:
        from .mlx_d2f_forward import SparseForwardLogits

        return SparseForwardLogits(
            rows=projected_rows, logits=logits[:len(projected_rows)],
            seq_len=seq_len, forwarded_rows=seq_len,
        )
    return logits


def _block_only_forward_from_weights(
    *,
    weights: Any,
    token_ids: list[int],
    plan: Any,
    prefix_strategy: Any,
    n_layers: int,
    n_heads: int,
    num_kv_heads: int,
    head_dim: int,
    eps: float,
    cos: Any,
    sin: Any,
    attention_mask_kind: str,
    apply_qk_norm: bool,
    timing_metrics: dict[str, float | int | bool] | None,
    forward_started_at: float,
    kernels: KernelTable,
):
    """Block-only forward over the cached prefix K/V (the approximate fast path).

    Computes hidden states ONLY for the rows ``[boundary : seq_len)`` (the block
    plus the single boundary position ``prompt_len-1`` the Dream shift reads),
    reusing the cached per-layer prefix K/V for everything before the block. The
    returned array is full ``[seq_len, vocab]`` with the un-computed prefix rows
    ``[0 : boundary)`` left zero — those rows are never read by the decode
    selection (masked positions are ``>= prompt_len``; after the shift they read
    ``>= boundary``), so they are dead. Per-step cost is ~O(block) instead of
    O(prefix+block).
    """
    import mlx.core as mx

    forward_positions = plan.forward_positions()  # [boundary : seq_len)
    block_positions = plan.block_positions()  # [prompt_len : seq_len)
    seq_len = plan.seq_len
    # Rows we actually carry hidden states for. The first row is the boundary
    # (a prefix token whose K/V lives in the cache); the rest are the block.
    fwd_token_ids = [int(token_ids[p]) for p in forward_positions]
    n_block = len(block_positions)

    hidden = _embedding_rows(weights, fwd_token_ids)
    cos_fwd = cos[forward_positions]
    sin_fwd = sin[forward_positions]
    # The block rows are exactly the forward rows AFTER the boundary, so their RoPE
    # tables are a tail slice (avoids a second fancy-index).
    boundary_in_fwd = len(forward_positions) - n_block
    cos_block = cos_fwd[boundary_in_fwd:]
    sin_block = sin_fwd[boundary_in_fwd:]

    for layer_index in range(n_layers):
        layer_started_at = _timer_now()
        hidden = _transformer_layer_forward_block_only(
            weights=weights,
            hidden=hidden,
            layer_index=layer_index,
            prefix_strategy=prefix_strategy,
            n_heads=n_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            eps=eps,
            cos_fwd=cos_fwd,
            sin_fwd=sin_fwd,
            cos_block=cos_block,
            sin_block=sin_block,
            boundary_in_fwd=boundary_in_fwd,
            attention_mask_kind=attention_mask_kind,
            apply_qk_norm=apply_qk_norm,
            kernels=kernels,
        )
        mx.eval(hidden)
        if timing_metrics is not None:
            _accumulate_timing_ms(
                timing_metrics,
                "mlx_timing_layer0_forward_ms"
                if layer_index == 0
                else "mlx_timing_upper_layers_forward_ms",
                layer_started_at,
            )

    final_started_at = _timer_now()
    hidden = kernels.rms_norm(
        hidden,
        mx.array(_weight(weights, "model.norm.weight")),
        eps,
    )
    # fp32 logits (numpy-readable; the matmul accumulated in fp32 anyway). mx.array
    # lifts a numpy stub; no-op on the fp32 path.
    fwd_logits = mx.array(
        kernels.linear(hidden, _linear_weight(weights, "lm_head.weight"))
    ).astype(mx.float32)
    mx.eval(fwd_logits)
    vocab = fwd_logits.shape[-1]
    # Scatter the computed forward-row logits back into a full-length array; the
    # prefix rows [0:boundary) stay zero (dead under both shift conventions).
    full = mx.zeros((seq_len, vocab), dtype=fwd_logits.dtype)
    start = forward_positions[0]
    full[start : start + len(forward_positions)] = fwd_logits
    mx.eval(full)
    if timing_metrics is not None:
        _accumulate_timing_ms(
            timing_metrics, "mlx_timing_final_norm_lm_head_ms", final_started_at
        )
        _accumulate_timing_ms(
            timing_metrics, "mlx_timing_forward_total_ms", forward_started_at
        )
    return full


def _transformer_layer_forward_block_only(
    *,
    weights: Any,
    hidden: Any,
    layer_index: int,
    prefix_strategy: Any,
    n_heads: int,
    num_kv_heads: int,
    head_dim: int,
    eps: float,
    cos_fwd: Any,
    sin_fwd: Any,
    cos_block: Any,
    sin_block: Any,
    boundary_in_fwd: int,
    attention_mask_kind: str,
    apply_qk_norm: bool,
    kernels: KernelTable,
):
    """One transformer layer over the block (+boundary) rows, reusing prefix K/V.

    ``hidden`` carries only the forward rows ``[boundary : seq_len)``. Q is
    projected for all forward rows; K/V are freshly projected ONLY for the block
    rows (``hidden`` after ``boundary_in_fwd``) and concatenated AFTER the cached
    prefix K/V (positions ``[0:prompt_len)``) so block-Q attends against the full
    sequence. The boundary row's own K/V is the cached one (it is a prefix
    position), so it is NOT re-projected — no double counting.
    """
    import mlx.core as mx

    residual = hidden
    normalized = kernels.rms_norm(
        hidden,
        mx.array(
            _weight(weights, f"model.layers.{layer_index}.input_layernorm.weight")
        ),
        eps,
    )

    n_fwd = hidden.shape[0]
    q = kernels.linear(
        normalized,
        _linear_weight(weights, f"model.layers.{layer_index}.self_attn.q_proj.weight"),
        _optional_bias(weights, f"model.layers.{layer_index}.self_attn.q_proj.bias"),
    )
    q = mx.reshape(q, (n_fwd, n_heads, head_dim))
    if apply_qk_norm:
        q = kernels.rms_norm(
            q,
            mx.array(
                _weight(weights, f"model.layers.{layer_index}.self_attn.q_norm.weight")
            ),
            eps,
        )
    q = kernels.rope(q, cos_fwd, sin_fwd)

    # Fresh K/V for the BLOCK rows only (the rows of ``normalized`` at/after the
    # boundary index). RoPE uses the block's absolute positions.
    block_normalized = normalized[boundary_in_fwd:]
    block_count = block_normalized.shape[0]
    k_block = kernels.linear(
        block_normalized,
        _linear_weight(weights, f"model.layers.{layer_index}.self_attn.k_proj.weight"),
        _optional_bias(weights, f"model.layers.{layer_index}.self_attn.k_proj.bias"),
    )
    v_block = kernels.linear(
        block_normalized,
        _linear_weight(weights, f"model.layers.{layer_index}.self_attn.v_proj.weight"),
        _optional_bias(weights, f"model.layers.{layer_index}.self_attn.v_proj.bias"),
    )
    k_block = mx.reshape(k_block, (block_count, num_kv_heads, head_dim))
    v_block = mx.reshape(v_block, (block_count, num_kv_heads, head_dim))
    if apply_qk_norm:
        k_block = kernels.rms_norm(
            k_block,
            mx.array(
                _weight(weights, f"model.layers.{layer_index}.self_attn.k_norm.weight")
            ),
            eps,
        )
    k_block = kernels.rope(k_block, cos_block, sin_block)
    prefix_key, prefix_value = prefix_strategy.layer_prefix_kv(layer_index=layer_index)
    k_full = mx.concatenate([prefix_key, k_block], axis=0)
    v_full = mx.concatenate([prefix_value, v_block], axis=0)

    attention_output = kernels.attention(
        q,
        k_full,
        v_full,
        head_dim=head_dim,
        kind=attention_mask_kind,
    )
    attention_projected = kernels.linear(
        attention_output,
        _linear_weight(weights, f"model.layers.{layer_index}.self_attn.o_proj.weight"),
        _optional_bias(weights, f"model.layers.{layer_index}.self_attn.o_proj.bias"),
    )
    hidden = residual + attention_projected

    residual = hidden
    hidden = kernels.rms_norm(
        hidden,
        mx.array(
            _weight(
                weights, f"model.layers.{layer_index}.post_attention_layernorm.weight"
            )
        ),
        eps,
    )
    gate = kernels.linear(
        hidden,
        _linear_weight(weights, f"model.layers.{layer_index}.mlp.gate_proj.weight"),
        _optional_bias(weights, f"model.layers.{layer_index}.mlp.gate_proj.bias"),
    )
    up = kernels.linear(
        hidden,
        _linear_weight(weights, f"model.layers.{layer_index}.mlp.up_proj.weight"),
        _optional_bias(weights, f"model.layers.{layer_index}.mlp.up_proj.bias"),
    )
    mlp = kernels.linear(
        kernels.swiglu(gate, up),
        _linear_weight(weights, f"model.layers.{layer_index}.mlp.down_proj.weight"),
        _optional_bias(weights, f"model.layers.{layer_index}.mlp.down_proj.bias"),
    )
    return residual + mlp


def _block_kv_forward_from_weights(
    *,
    weights: Any,
    token_ids: list[int],
    plan: Any,
    block_strategy: Any,
    n_layers: int,
    n_heads: int,
    num_kv_heads: int,
    head_dim: int,
    eps: float,
    cos: Any,
    sin: Any,
    attention_mask_kind: str,
    apply_qk_norm: bool,
    timing_metrics: dict[str, float | int | bool] | None,
    forward_started_at: float,
    kernels: KernelTable,
):
    """Block KV cache forward: forward ONLY the still-masked frontier.

    Computes hidden states for the rows ``[boundary : seq_len)`` (the masked
    frontier plus the single shift-feeder boundary row the Dream shift reads).
    Each frontier query attends ``concat(cached_prefix_K, cached_committed_block_K,
    fresh_K)`` — the full sequence — where the prefix AND committed-block K/V are
    spliced from the cache and only the still-masked (fresh) positions are
    re-projected. Bidirectional SDPA is permutation-invariant over keys, so the
    concat order is irrelevant as long as K and V match (each key carries its own
    RoPE phase). The returned array is full ``[seq_len, vocab]`` with the
    un-computed rows ``[0 : boundary)`` left zero (dead under both shift
    conventions). This skips the attention+MLP compute for committed positions —
    the 51–55% per-step waste the profile found.
    """
    import mlx.core as mx

    forward_positions = plan.forward_positions()  # [boundary : seq_len)
    fresh_positions = plan.fresh_block_positions()  # masked frontier rows
    cached_positions = plan.cached_block_positions()  # committed-block rows (cached)
    seq_len = plan.seq_len
    fwd_token_ids = [int(token_ids[p]) for p in forward_positions]

    hidden = _embedding_rows(weights, fwd_token_ids)
    cos_fwd = cos[forward_positions]
    sin_fwd = sin[forward_positions]
    # Index of each fresh/cached position WITHIN the forward-row hidden states, and
    # the fresh positions' RoPE tables (fresh rows are re-projected; their K is
    # RoPE'd by absolute position).
    fwd_index = {p: i for i, p in enumerate(forward_positions)}
    fresh_in_fwd = [fwd_index[p] for p in fresh_positions]
    cos_fresh = cos[fresh_positions] if fresh_positions else cos[0:0]
    sin_fresh = sin[fresh_positions] if fresh_positions else sin[0:0]

    for layer_index in range(n_layers):
        layer_started_at = _timer_now()
        hidden = _transformer_layer_forward_block_kv(
            weights=weights,
            hidden=hidden,
            layer_index=layer_index,
            block_strategy=block_strategy,
            fresh_positions=fresh_positions,
            cached_positions=cached_positions,
            fresh_in_fwd=fresh_in_fwd,
            n_heads=n_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            eps=eps,
            cos_fwd=cos_fwd,
            sin_fwd=sin_fwd,
            cos_fresh=cos_fresh,
            sin_fresh=sin_fresh,
            attention_mask_kind=attention_mask_kind,
            apply_qk_norm=apply_qk_norm,
            kernels=kernels,
        )
        mx.eval(hidden)
        if timing_metrics is not None:
            _accumulate_timing_ms(
                timing_metrics,
                "mlx_timing_layer0_forward_ms"
                if layer_index == 0
                else "mlx_timing_upper_layers_forward_ms",
                layer_started_at,
            )

    final_started_at = _timer_now()
    hidden = kernels.rms_norm(
        hidden,
        mx.array(_weight(weights, "model.norm.weight")),
        eps,
    )
    fwd_logits = mx.array(
        kernels.linear(hidden, _linear_weight(weights, "lm_head.weight"))
    ).astype(mx.float32)
    mx.eval(fwd_logits)
    vocab = fwd_logits.shape[-1]
    full = mx.zeros((seq_len, vocab), dtype=fwd_logits.dtype)
    start = forward_positions[0]
    full[start : start + len(forward_positions)] = fwd_logits
    mx.eval(full)
    # Capture any pending-committed positions from THIS step's fresh-K/V snapshot
    # (they were just projected from their committed tokens) into the persistent
    # committed-block cache, so the next step serves them from cache.
    block_strategy.promote_pending()
    if timing_metrics is not None:
        _accumulate_timing_ms(
            timing_metrics, "mlx_timing_final_norm_lm_head_ms", final_started_at
        )
        _accumulate_timing_ms(
            timing_metrics, "mlx_timing_forward_total_ms", forward_started_at
        )
    return full


def _transformer_layer_forward_block_kv(
    *,
    weights: Any,
    hidden: Any,
    layer_index: int,
    block_strategy: Any,
    fresh_positions: list[int],
    cached_positions: list[int],
    fresh_in_fwd: list[int],
    n_heads: int,
    num_kv_heads: int,
    head_dim: int,
    eps: float,
    cos_fwd: Any,
    sin_fwd: Any,
    cos_fresh: Any,
    sin_fresh: Any,
    attention_mask_kind: str,
    apply_qk_norm: bool,
    kernels: KernelTable,
):
    """One transformer layer over the masked frontier, reusing prefix+block K/V.

    ``hidden`` carries the forward rows ``[boundary : seq_len)``. Q is projected for
    all of them. K/V are freshly projected ONLY for the still-masked (fresh) rows
    and concatenated AFTER ``concat(cached_prefix_K, cached_committed_block_K)``.
    The freshly-projected fresh K/V are also stored into the strategy's frontier
    snapshot so a fresh position that commits this step has its commit-time K/V
    captured for promotion. Committed positions inside the window (incl. the
    boundary, if committed) contribute NO fresh K/V — their cached K/V is spliced —
    so they are never double-counted.
    """
    import mlx.core as mx

    residual = hidden
    normalized = kernels.rms_norm(
        hidden,
        mx.array(
            _weight(weights, f"model.layers.{layer_index}.input_layernorm.weight")
        ),
        eps,
    )

    n_fwd = hidden.shape[0]
    q = kernels.linear(
        normalized,
        _linear_weight(weights, f"model.layers.{layer_index}.self_attn.q_proj.weight"),
        _optional_bias(weights, f"model.layers.{layer_index}.self_attn.q_proj.bias"),
    )
    q = mx.reshape(q, (n_fwd, n_heads, head_dim))
    if apply_qk_norm:
        q = kernels.rms_norm(
            q,
            mx.array(
                _weight(weights, f"model.layers.{layer_index}.self_attn.q_norm.weight")
            ),
            eps,
        )
    q = kernels.rope(q, cos_fwd, sin_fwd)

    # Cached side: prefix K/V + committed-block K/V (frozen at their commit step).
    prefix_key, prefix_value = block_strategy.layer_prefix_kv(layer_index=layer_index)
    cached_parts_k = [prefix_key]
    cached_parts_v = [prefix_value]
    if cached_positions:
        ck, cv = block_strategy.committed_block_kv(
            layer_index=layer_index, positions=cached_positions
        )
        cached_parts_k.append(mx.concatenate(ck, axis=0))
        cached_parts_v.append(mx.concatenate(cv, axis=0))

    # Fresh side: project K/V only for the still-masked rows of ``normalized``.
    if fresh_in_fwd:
        fresh_normalized = normalized[fresh_in_fwd]
        fresh_count = fresh_normalized.shape[0]
        k_fresh = kernels.linear(
            fresh_normalized,
            _linear_weight(
                weights, f"model.layers.{layer_index}.self_attn.k_proj.weight"
            ),
            _optional_bias(
                weights, f"model.layers.{layer_index}.self_attn.k_proj.bias"
            ),
        )
        v_fresh = kernels.linear(
            fresh_normalized,
            _linear_weight(
                weights, f"model.layers.{layer_index}.self_attn.v_proj.weight"
            ),
            _optional_bias(
                weights, f"model.layers.{layer_index}.self_attn.v_proj.bias"
            ),
        )
        k_fresh = mx.reshape(k_fresh, (fresh_count, num_kv_heads, head_dim))
        v_fresh = mx.reshape(v_fresh, (fresh_count, num_kv_heads, head_dim))
        if apply_qk_norm:
            k_fresh = kernels.rms_norm(
                k_fresh,
                mx.array(
                    _weight(
                        weights, f"model.layers.{layer_index}.self_attn.k_norm.weight"
                    )
                ),
                eps,
            )
        k_fresh = kernels.rope(k_fresh, cos_fresh, sin_fresh)
        # Snapshot the fresh rows' (RoPE'd, compact-GQA) K/V for promotion on
        # commit — one row per fresh position, in fresh_positions order.
        block_strategy.store_frontier_layer(
            layer_index=layer_index,
            positions=fresh_positions,
            key=[k_fresh[i : i + 1] for i in range(fresh_count)],
            value=[v_fresh[i : i + 1] for i in range(fresh_count)],
        )
        cached_parts_k.append(k_fresh)
        cached_parts_v.append(v_fresh)

    k_full = (
        mx.concatenate(cached_parts_k, axis=0)
        if len(cached_parts_k) > 1
        else cached_parts_k[0]
    )
    v_full = (
        mx.concatenate(cached_parts_v, axis=0)
        if len(cached_parts_v) > 1
        else cached_parts_v[0]
    )

    attention_output = kernels.attention(
        q,
        k_full,
        v_full,
        head_dim=head_dim,
        kind=attention_mask_kind,
    )
    attention_projected = kernels.linear(
        attention_output,
        _linear_weight(weights, f"model.layers.{layer_index}.self_attn.o_proj.weight"),
        _optional_bias(weights, f"model.layers.{layer_index}.self_attn.o_proj.bias"),
    )
    hidden = residual + attention_projected

    residual = hidden
    hidden = kernels.rms_norm(
        hidden,
        mx.array(
            _weight(
                weights, f"model.layers.{layer_index}.post_attention_layernorm.weight"
            )
        ),
        eps,
    )
    gate = kernels.linear(
        hidden,
        _linear_weight(weights, f"model.layers.{layer_index}.mlp.gate_proj.weight"),
        _optional_bias(weights, f"model.layers.{layer_index}.mlp.gate_proj.bias"),
    )
    up = kernels.linear(
        hidden,
        _linear_weight(weights, f"model.layers.{layer_index}.mlp.up_proj.weight"),
        _optional_bias(weights, f"model.layers.{layer_index}.mlp.up_proj.bias"),
    )
    mlp = kernels.linear(
        kernels.swiglu(gate, up),
        _linear_weight(weights, f"model.layers.{layer_index}.mlp.down_proj.weight"),
        _optional_bias(weights, f"model.layers.{layer_index}.mlp.down_proj.bias"),
    )
    return residual + mlp


def _d2f_exact_forward_from_weights(*args: Any, **kwargs: Any):
    """Load exact-D2F execution lazily; both sides remain backend-owned."""
    from .mlx_d2f_forward import _d2f_exact_forward_from_weights as _impl

    return _impl(*args, **kwargs)


def _transformer_layer_forward(
    *,
    weights: Any,
    hidden: Any,
    layer_index: int,
    n_heads: int,
    num_kv_heads: int,
    head_dim: int,
    eps: float,
    cos: Any,
    sin: Any,
    attention_mask_kind: str,
    token_ids: list[int] | None = None,
    kv_cache_state: Any | None = None,
    apply_qk_norm: bool = False,
    prefix_store: Any | None = None,
    mask_geometry: Any | None = None,
    kernels: KernelTable,
    observer: Callable[[str, Any], None] | None = None,
    attention_mask: mlx_ops.PreparedAttentionMask | None = None,
):
    import mlx.core as mx

    residual = hidden
    normalized = kernels.rms_norm(
        hidden,
        mx.array(
            _weight(weights, f"model.layers.{layer_index}.input_layernorm.weight")
        ),
        eps,
    )
    if observer is not None:
        observer("input_layernorm", normalized)

    q = kernels.linear(
        normalized,
        _linear_weight(weights, f"model.layers.{layer_index}.self_attn.q_proj.weight"),
        _optional_bias(weights, f"model.layers.{layer_index}.self_attn.q_proj.bias"),
    )

    seq_len = hidden.shape[0]
    q = mx.reshape(q, (seq_len, n_heads, head_dim))
    if observer is not None:
        observer("q_linear", q)
    # Qwen3 QK-norm: per-head RMSNorm over the head_dim axis, applied AFTER the
    # projection+reshape and BEFORE RoPE (Dream leaves apply_qk_norm=False, so this
    # is skipped and the path is byte-identical).
    if apply_qk_norm:
        q = kernels.rms_norm(
            q,
            mx.array(
                _weight(weights, f"model.layers.{layer_index}.self_attn.q_norm.weight")
            ),
            eps,
        )
    q = kernels.rope(q, cos, sin)
    if observer is not None:
        observer("q_after_rope", q)
    if layer_index == 0 and kv_cache_state is not None and token_ids is not None:
        k, v = _layer0_kv_with_cache(
            weights=weights,
            normalized=normalized,
            layer_index=layer_index,
            kv_cache_state=kv_cache_state,
            token_ids=token_ids,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            cos=cos,
            sin=sin,
            eps=eps,
            apply_qk_norm=apply_qk_norm,
            kernels=kernels,
            observer=observer,
        )
    else:
        k, v = _project_kv_rows(
            weights=weights,
            normalized=normalized,
            layer_index=layer_index,
            row_positions=None,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            cos=cos,
            sin=sin,
            eps=eps,
            apply_qk_norm=apply_qk_norm,
            kernels=kernels,
            observer=observer,
        )

    # Prefix KV cache populate hook: on a full forward driven by a prefix-KV
    # strategy, stash this layer's compact RoPE'd K/V so a later block-only
    # forward can reuse the prefix slice. ``None`` (every other caller) leaves the
    # path byte-identical.
    if prefix_store is not None:
        prefix_store(layer_index, k, v)

    attention_output = kernels.attention(
        q,
        k,
        v,
        head_dim=head_dim,
        kind=attention_mask_kind,
        mask_geometry=mask_geometry if attention_mask is None else None,
        prepared_mask=attention_mask,
    )
    if observer is not None:
        observer("attention_output", attention_output)
    attention_projected = kernels.linear(
        attention_output,
        _linear_weight(weights, f"model.layers.{layer_index}.self_attn.o_proj.weight"),
        _optional_bias(weights, f"model.layers.{layer_index}.self_attn.o_proj.bias"),
    )
    if observer is not None:
        observer("attention_projected", attention_projected)
    hidden = residual + attention_projected

    residual = hidden
    hidden = kernels.rms_norm(
        hidden,
        mx.array(
            _weight(
                weights, f"model.layers.{layer_index}.post_attention_layernorm.weight"
            )
        ),
        eps,
    )
    gate = kernels.linear(
        hidden,
        _linear_weight(weights, f"model.layers.{layer_index}.mlp.gate_proj.weight"),
        _optional_bias(weights, f"model.layers.{layer_index}.mlp.gate_proj.bias"),
    )
    up = kernels.linear(
        hidden,
        _linear_weight(weights, f"model.layers.{layer_index}.mlp.up_proj.weight"),
        _optional_bias(weights, f"model.layers.{layer_index}.mlp.up_proj.bias"),
    )
    mlp = kernels.linear(
        kernels.swiglu(gate, up),
        _linear_weight(weights, f"model.layers.{layer_index}.mlp.down_proj.weight"),
        _optional_bias(weights, f"model.layers.{layer_index}.mlp.down_proj.bias"),
    )
    output = residual + mlp
    if observer is not None:
        observer("layer0_output", output)
    return output


def _project_kv_rows(
    *,
    weights: Any,
    normalized: Any,
    layer_index: int,
    row_positions: list[int] | None,
    num_kv_heads: int,
    head_dim: int,
    cos: Any,
    sin: Any,
    eps: float = 1e-6,
    apply_qk_norm: bool = False,
    kernels: KernelTable,
    observer: Callable[[str, Any], None] | None = None,
):
    import mlx.core as mx

    source = normalized if row_positions is None else normalized[row_positions]
    row_count = source.shape[0]
    k = kernels.linear(
        source,
        _linear_weight(weights, f"model.layers.{layer_index}.self_attn.k_proj.weight"),
        _optional_bias(weights, f"model.layers.{layer_index}.self_attn.k_proj.bias"),
    )
    v = kernels.linear(
        source,
        _linear_weight(weights, f"model.layers.{layer_index}.self_attn.v_proj.weight"),
        _optional_bias(weights, f"model.layers.{layer_index}.self_attn.v_proj.bias"),
    )
    k = mx.reshape(k, (row_count, num_kv_heads, head_dim))
    v = mx.reshape(v, (row_count, num_kv_heads, head_dim))
    if observer is not None:
        observer("k_linear", k)
        observer("v_linear", v)
    # Qwen3 QK-norm: per-head RMSNorm over head_dim on K, after reshape and BEFORE
    # RoPE (V is NOT normed). Dream leaves apply_qk_norm=False -> byte-identical.
    if apply_qk_norm:
        k = kernels.rms_norm(
            k,
            mx.array(
                _weight(weights, f"model.layers.{layer_index}.self_attn.k_norm.weight")
            ),
            eps,
        )
    row_cos = cos if row_positions is None else cos[row_positions]
    row_sin = sin if row_positions is None else sin[row_positions]
    k = kernels.rope(k, row_cos, row_sin)
    if observer is not None:
        observer("k_after_rope", k)
    return k, v


def _layer0_kv_with_cache(
    *,
    weights: Any,
    normalized: Any,
    layer_index: int,
    kv_cache_state: Any,
    token_ids: list[int],
    num_kv_heads: int,
    head_dim: int,
    cos: Any,
    sin: Any,
    eps: float = 1e-6,
    apply_qk_norm: bool = False,
    kernels: KernelTable,
    observer: Callable[[str, Any], None] | None = None,
):
    import mlx.core as mx

    key_rows: list[Any | None] = []
    value_rows: list[Any | None] = []
    missing_positions: list[int] = []
    skipped_count = 0
    for position, token_id in enumerate(token_ids):
        cached = kv_cache_state.lookup(
            layer_index=0,
            position=position,
            token_id=int(token_id),
        )
        if cached is None:
            missing_positions.append(position)
            key_rows.append(None)
            value_rows.append(None)
        else:
            key_row, value_row = cached
            key_rows.append(key_row)
            value_rows.append(value_row)
            skipped_count += 1

    if missing_positions:
        projected_keys, projected_values = _project_kv_rows(
            weights=weights,
            normalized=normalized,
            layer_index=layer_index,
            row_positions=missing_positions,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            cos=cos,
            sin=sin,
            eps=eps,
            apply_qk_norm=apply_qk_norm,
            kernels=kernels,
            observer=observer,
        )
        for projected_index, position in enumerate(missing_positions):
            key_row = projected_keys[projected_index : projected_index + 1]
            value_row = projected_values[projected_index : projected_index + 1]
            kv_cache_state.store_entry(
                layer_index=0,
                position=position,
                token_id=int(token_ids[position]),
                key_tensor=key_row,
                value_tensor=value_row,
            )
            key_rows[position] = key_row
            value_rows[position] = value_row
    kv_cache_state.record_layer0_projection_rows(
        projected_count=len(missing_positions),
        skipped_count=skipped_count,
    )
    return mx.concatenate(key_rows, axis=0), mx.concatenate(value_rows, axis=0)


def _mask_token_id_for(model_config: dict[str, Any]) -> int | None:
    """Best-effort mask id from config (fallback when masked_positions is omitted).

    The engine threads the explicit ``masked_positions`` for the block cache, so
    this is only a fallback for a direct forward caller that enabled the block
    cache without supplying the frontier. ``None`` => unknown (treat as no masks).
    """
    value = (model_config or {}).get("mask_token_id")
    return int(value) if value is not None else None


def _timer_now() -> float:
    import time

    return time.perf_counter()


def _accumulate_timing_ms(
    metrics: dict[str, float | int | bool],
    key: str,
    started_at: float,
) -> None:
    elapsed_ms = max(0.0, (_timer_now() - started_at) * 1000.0)
    metrics[key] = float(metrics.get(key, 0.0)) + elapsed_ms


def _resolve_compute_dtype(name: Any):
    """Map an ``mlx_compute_dtype`` config string to an ``mx`` dtype (fail-closed).

    Accepts ``"bf16"``/``"bfloat16"`` (the new throughput default) and
    ``"fp32"``/``"float32"`` (A/B + per-family fallback). Anything else raises —
    this is a precision/throughput knob, NOT quantization, so int4/int8 are
    rejected here. An already-resolved mx dtype passes through unchanged so the
    forward can be called with either a string (config) or a dtype (internal).
    """
    import mlx.core as mx

    if name in (mx.bfloat16, mx.float32):
        return name
    return getattr(mx, _canonical_compute_dtype_name(name))


def _weight(weights: Any, key: str, *, dtype: Any = None):
    """Materialize weight ``key`` for the forward.

    The compute dtype is carried by the RESIDENT weights, loaded once at the
    engine's compute dtype by :func:`_load_mlx_weight_state` (fp32 by default,
    bf16 for the throughput path). So:

    * ``dtype=None`` (the hot path): an mlx-array weight is returned at its OWN
      dtype — i.e. the resident compute dtype (bf16 or fp32), with NO per-call
      cast. A numpy weight (the standalone trace path, which loads the npz fp16
      directly) is upcast to fp32 — the byte-identical legacy behavior.
    * ``dtype=<bf16|fp32>`` (explicit): force that dtype. numpy has no bf16, so
      the fp16-npy -> bf16 cast MUST route through MLX; the mlx-array case casts
      directly.
    """
    if key not in weights:
        raise KeyError(f"missing weight key: {key}")
    value = weights[key]
    import mlx.core as mx
    import numpy as np

    if dtype is None:
        if isinstance(value, np.ndarray):
            # Byte-identical legacy default: numpy upcast to fp32, no MLX needed.
            return value.astype("float32")
        # mlx-array weight: keep its resident dtype (the compute dtype).
        return value
    target = _resolve_compute_dtype(dtype)
    if isinstance(value, np.ndarray):
        if target is mx.float32:
            return value.astype("float32")
        # numpy has no bf16 -> route the cast through MLX.
        return mx.array(value).astype(target)
    return value.astype(target)


def _linear_weight(weights: Any, key: str):
    """The weight operand for a ``kernels.linear`` call site (Quant-1).

    A resident :class:`mlx_ops.QuantizedWeight` bundle passes through UNTOUCHED
    to the polymorphic ``linear`` kernel; anything else takes the exact legacy
    lift (``mx.array(_weight(...))``), so the fp path — and the default
    quantization-OFF forward — stays byte-identical.
    """
    value = _weight(weights, key)
    if isinstance(value, mlx_ops.QuantizedWeight):
        return value
    import mlx.core as mx

    return mx.array(value)


def _optional_bias(weights: Any, key: str, *, dtype: Any = None):
    if key not in weights:
        return None
    import mlx.core as mx

    return mx.array(_weight(weights, key, dtype=dtype))


def _embedding_rows(
    weights: Any,
    token_ids: list[int],
    *,
    weight_key: str = "model.embed_tokens.weight",
):
    """Initial activations: embedding rows, ALWAYS in fp32.

    The activation / residual stream is fp32 for accumulation parity (the bf16
    lever is on the WEIGHTS, applied at each ``mlx_ops.linear``). The embed table
    may be a bf16 resident weight or a numpy fp16 array (trace path); either way
    the looked-up rows are cast to fp32 for the residual stream.
    """
    import mlx.core as mx
    import numpy as np

    value = weights[weight_key]
    if isinstance(value, np.ndarray):
        return mx.array(value[token_ids].astype(np.float32))
    return value[token_ids].astype(mx.float32)


def _weight_keys(weights: Any):
    if isinstance(weights, dict):
        return weights.keys()
    return getattr(weights, "files", [])


def _rope_tables(
    *,
    seq_len: int,
    head_dim: int,
    rope_theta: float,
    rope_scaling: Any,
):
    import numpy as np

    half_dim = head_dim // 2
    if half_dim <= 0:
        cos = np.ones((seq_len, 1, head_dim), dtype=np.float32)
        sin = np.zeros((seq_len, 1, head_dim), dtype=np.float32)
        return cos, sin
    # I1: fail closed on unsupported rope_scaling types (yarn/dynamic/su would
    # be silently misread as linear here AND on the numpyref path).
    from edllm.graph.numpyref import _validated_linear_rope_factor

    factor = _validated_linear_rope_factor(rope_scaling)
    inv_freq = 1.0 / (
        rope_theta ** (np.arange(0, half_dim, dtype=np.float32) / half_dim)
    )
    inv_freq = inv_freq / max(1e-6, factor)
    freqs = np.einsum("i,j->ij", np.arange(seq_len, dtype=np.float32), inv_freq)
    cos_half = np.cos(freqs)[:, None, :]
    sin_half = np.sin(freqs)[:, None, :]
    return (
        np.concatenate([cos_half, cos_half], axis=-1).astype(np.float32),
        np.concatenate([sin_half, sin_half], axis=-1).astype(np.float32),
    )


def _trace_stage(value: Any) -> dict:
    import numpy as np

    rows = np.array(value, dtype=np.float32)
    return {"shape": list(rows.shape), "rows": rows.tolist()}
