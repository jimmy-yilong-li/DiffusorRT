"""MLX exact-D2F tensor execution and sparse row selection.

The backend dispatcher in mlx_forward calls this module lazily. Shared weight
and timing helpers also live in mlx_forward, so neither module imports the
decode engine. Exact-KV lifecycle remains with the injected native cache owner;
this module performs only the requested forward and its cache reads/writes.
MLX is imported lazily inside the operations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import mlx_ops
from .kernel_table import KernelTable
from .mlx_forward import (
    _accumulate_timing_ms,
    _embedding_rows,
    _linear_weight,
    _optional_bias,
    _timer_now,
    _weight,
)


@dataclass(frozen=True, slots=True)
class SparseForwardLogits:
    """The rows a windowed forward actually computed, and their RAW logits.

    ``logits`` is ``[len(rows), vocab]`` — the shape the forward already produces
    internally. The pre-sparse path scattered this into a zeroed
    ``[seq_len, vocab]`` array purely so the decode loop could stay unchanged;
    ``seq_len`` is carried here so that densification remains available for the
    oracle/debug path without the hot path paying for it.

    ``rows`` are absolute row indices in the canvas.
    """

    rows: tuple[int, ...]
    logits: Any
    seq_len: int
    # Query rows executed by the transformer, if different from projected rows.
    forwarded_rows: int | None = None


def densify_forward_logits(sparse: SparseForwardLogits):
    """Re-expand to the full ``[seq_len, vocab]`` tensor. Debug and tests only.

    Byte-identical to what the pre-sparse forward returned, INCLUDING the zero
    rows at every uncomputed position — verified against both the old
    contiguous slice-assign and the old holey scatter, in float32, float16 and
    bfloat16.

    NO PRODUCTION CALLER TODAY. An earlier version of this docstring said it
    existed "so the full-forward parity oracle keeps its shape"; that was wrong.
    ``full_forward`` never passes ``return_sparse``, so the oracle receives a
    dense tensor directly and never densifies anything. Grep over
    ``python/src/`` and ``scripts/`` finds only this definition. It is kept
    because it is the inverse of the sparse form and tests compare against it —
    not because anything in the runtime depends on it.

    It also does NOT reproduce the producer's empty-rows guard: given
    ``rows=()`` it returns an all-zero tensor rather than raising.
    """
    import mlx.core as mx

    vocab = sparse.logits.shape[-1]
    dense = mx.zeros((sparse.seq_len, vocab), dtype=sparse.logits.dtype)
    if sparse.rows:
        dense[mx.array(list(sparse.rows), dtype=mx.int32)] = sparse.logits
    return dense


def _required_raw_row(position: int, *, logits_convention: str) -> int:
    """Which RAW row a logical position reads under the given convention.

    ``shift_logits_for_dream`` is ``concatenate([logits[:1], logits[:-1]])``, so
    ``shifted[p] == raw[p - 1]`` for ``p >= 1`` and ``shifted[0] == raw[0]``.
    Each output row therefore depends on EXACTLY ONE raw row, which is what
    makes sparse selection possible at all.

    This is not in tension with the warning at ``backend_runner`` that the shift
    "couples adjacent positions — cannot compress to forward_scope first then
    shift". That warning is about shifting AFTER selection, which genuinely
    differs. Here the remap is applied BEFORE selection, which does not.
    """
    if logits_convention == "dream_ar_shift":
        return position - 1 if position >= 1 else 0
    if logits_convention == "raw":
        return position
    # Mirrors _logits_for_selection's fail-closed rule: an unknown convention
    # must not silently fall through to the raw view.
    raise ValueError(f"unknown logits_convention: {logits_convention!r}")


def select_sparse_logits(
    sparse: SparseForwardLogits,
    forward_scope,
    *,
    logits_convention: str,
    require_computed: bool = False,
):
    """Select ``forward_scope`` rows straight from the sparse form.

    Byte-identical to ``_logits_for_selection(densify(sparse))[forward_scope]``
    while allocating no ``[seq_len, vocab]`` array.

    By default, rows outside the computed window are emitted as EXACT ZERO rows,
    because that is what the historical dense/debug path returned for them.
    Production decode passes ``require_computed=True``: every requested token
    decision must have a real source row, so a planner/producer mismatch raises
    instead of silently committing token 0 from an all-zero distribution.
    """
    import mlx.core as mx

    scope = list(forward_scope)
    vocab = sparse.logits.shape[-1]
    # Validate the convention even on the empty-scope path, so an unknown value
    # cannot pass silently just because there was nothing to select.
    _required_raw_row(0, logits_convention=logits_convention)

    # Reject negative positions. The dense route this replaces indexed an array
    # with the raw list, so Python's negative indexing silently applied:
    # `dense[-2]` read the SECOND-TO-LAST row and returned real logits, while
    # this function maps -2 through `_required_raw_row` to raw row 0 and emits
    # zeros. Measured on rows=(6,7,8): dense gave [5,6,7,8], sparse gave
    # [0,0,0,0] -- a silent value divergence in BOTH conventions.
    #
    # Fail closed rather than reproduce the old behaviour. A negative canvas
    # position is meaningless, no planner emits one, and `WindowExecutionPlan`
    # does not reject it either -- so an ABI that silently reinterpreted it as
    # "count from the end" would freeze a footgun into the contract. An error is
    # the honest answer; a different-but-plausible logits row is the worst one.
    negative = [p for p in scope if p < 0]
    if negative:
        raise ValueError(
            f"forward_scope contains negative positions {negative}; positions are "
            "absolute canvas indices, and negative values would silently select "
            "different rows than the dense reference path"
        )

    # Reject positions at or past the canvas end, for the same reason and a
    # sharper one. The dense route indexed past the array, which MLX does NOT
    # bounds-check -- it read adjacent device memory and returned garbage. This
    # route maps p == seq_len to raw row seq_len - 1, which is usually COMPUTED,
    # so it returns a real, entirely plausible logits row instead. Measured:
    # dense gave neighbouring-memory values, sparse gave the genuine last
    # computed row.
    #
    # Both are wrong; the new one is more dangerous, because garbage is
    # eventually noticed and a plausible row never is. No planner emits such a
    # position today (`frontier < window_end` in the D2F pipeline), but neither
    # `classify_window` nor `WindowExecutionPlan` bounds forward_scope by
    # window_end, so a future commit policy could reach it silently.
    out_of_range = [p for p in scope if p >= sparse.seq_len]
    if out_of_range:
        raise ValueError(
            f"forward_scope contains positions {out_of_range} at or beyond "
            f"seq_len={sparse.seq_len}; the dense reference path read out of "
            "bounds here (MLX does not bounds-check gathers), so there is no "
            "correct value to reproduce"
        )

    if not scope:
        return sparse.logits[:0]

    row_position = {row: index for index, row in enumerate(sparse.rows)}
    gather: list[int] = []
    zero_slots: list[int] = []
    missing: list[tuple[int, int]] = []
    for slot, position in enumerate(scope):
        raw_row = _required_raw_row(position, logits_convention=logits_convention)
        index = row_position.get(raw_row)
        if index is None:
            zero_slots.append(slot)
            missing.append((position, raw_row))
            gather.append(0)  # placeholder; overwritten with zeros below
        else:
            gather.append(index)

    if require_computed and missing:
        raise ValueError(
            "the sparse forward did not compute required raw rows for "
            f"forward_scope: {missing}; refusing to select synthetic zero "
            "logits for a production token decision"
        )

    if not sparse.rows:
        # A producer that computed NO rows. The placeholder index 0 above cannot
        # be gathered from a [0, V] array -- mx.take raises "cannot do a
        # non-empty take from an empty axis" -- so the all-zeros answer this
        # function promises has to be built directly. The dense path reached the
        # same result trivially, because its zeros array had full height
        # regardless of how many rows were computed.
        return mx.zeros((len(scope), vocab), dtype=sparse.logits.dtype)

    selected = sparse.logits[mx.array(gather, dtype=mx.int32)]
    if zero_slots:
        selected[mx.array(zero_slots, dtype=mx.int32)] = mx.zeros(
            (len(zero_slots), vocab), dtype=sparse.logits.dtype
        )
    return selected


def _d2f_exact_forward_from_weights(
    *,
    weights: Any,
    token_ids: list[int],
    plan: Any,
    d2f_strategy: Any,
    n_layers: int,
    n_heads: int,
    num_kv_heads: int,
    head_dim: int,
    eps: float,
    cos: Any,
    sin: Any,
    mask_geometry: Any,
    apply_qk_norm: bool,
    timing_metrics: dict[str, float | int | bool] | None,
    forward_started_at: float,
    kernels: KernelTable,
    return_sparse: bool = False,
):
    """Active-block-only d2f forward over the EXACT cached prompt+frozen K/V.

    Computes hidden states for the rows ``[boundary : active_end)`` — the active
    block, any pending frozen block being captured (recomputed fresh from its
    final tokens), and the single Dream shift-feeder row when no pending block
    precedes the active one. Each query attends ``concat(cached prompt K/V,
    cached frozen-block K/V, fresh K/V)`` — exactly the absolute positions
    ``[0 : kv_len)`` — under the NON-SQUARE d2f mask (the q rows carry their
    absolute positions, the same ``eff`` closed form as the full mask).

    Exactness: prompt rows attend only to the prompt and frozen blocks attend
    only to prompt + earlier blocks + themselves, so every cached K/V equals
    what a full forward would recompute; the active rows therefore see
    bit-equivalent context and their logits match the full no-cache d2f forward
    to fp-reassociation tolerance. Rows at/after ``active_end`` are DEAD this
    forward (zero logits): the d2f mask forbids the computed rows from seeing
    them. The returned array is full ``[seq_len, vocab]`` so the decode loop is
    unchanged.

    K2-3 removes the former row-skip, delayed-K/V and paged-slice branches.
    This function now has one exact path, backed by the native owner.
    """
    import mlx.core as mx

    forward_positions = plan.forward_positions()  # [boundary : active_end)
    fresh_positions = plan.fresh_positions()  # [cached_len : active_end)
    pending_rows = plan.pending_block_rows()
    seq_len = plan.seq_len
    expected_kv_len = plan.kv_len  # == active_end
    cached_len = int(plan.cached_len)

    forward_rows = forward_positions
    fresh_offset = len(forward_positions) - len(fresh_positions)

    if not forward_rows:
        # Guarded HERE -- immediately after forward_rows is known, before any
        # layer runs, before lm_head, before mx.eval. The previous placement was
        # after all of that: it prevented the bad RESULT but not the wasted
        # compute or the cache mutation along the way.
        #
        # The dense branch further down opens with `forward_rows[0]`, so an empty
        # row set used to raise IndexError -- a de-facto guard. Once
        # execute_window always requests sparse that branch stops running, an
        # empty SparseForwardLogits flows out, select_sparse_logits returns
        # all-zero logits, and the decode loop argmaxes a uniform-zero
        # distribution and commits token 0. Turning a loud failure into silent
        # corruption is the one regression a shape change must never introduce.
        raise ValueError(
            "the d2f active-window forward has no rows to compute; refusing to "
            "run a forward whose logits would be entirely zero"
        )

    fwd_token_ids = [int(token_ids[p]) for p in forward_rows]

    hidden = _embedding_rows(weights, fwd_token_ids)
    cos_fwd = cos[forward_rows]
    sin_fwd = sin[forward_rows]
    # Fresh rows are exactly the forward rows after the (optional) cached
    # boundary/feeder row, so their RoPE tables are a tail slice.
    cos_fresh = cos_fwd[fresh_offset:]
    sin_fresh = sin_fwd[fresh_offset:]

    attention_mask = mlx_ops.prepare_attention_mask(
        kind="d2f_hybrid", q_len=len(forward_rows), kv_len=expected_kv_len,
        mask_geometry=mask_geometry, q_positions=forward_rows,
    )
    for layer_index in range(n_layers):
        layer_started_at = _timer_now()
        hidden = _transformer_layer_forward_d2f_exact(
            weights=weights,
            hidden=hidden,
            layer_index=layer_index,
            d2f_strategy=d2f_strategy,
            fresh_offset=fresh_offset,
            cached_len=cached_len,
            pending_rows=pending_rows,
            expected_kv_len=expected_kv_len,
            q_positions=forward_rows,
            n_heads=n_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            eps=eps,
            cos_fwd=cos_fwd,
            sin_fwd=sin_fwd,
            cos_fresh=cos_fresh,
            sin_fresh=sin_fresh,
            mask_geometry=mask_geometry,
            apply_qk_norm=apply_qk_norm,
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

    # Every layer has evaluated its consumer; do not retain the mask alongside
    # the (potentially much larger) logits allocation.
    del attention_mask
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
    # SHAPE-OWNERSHIP SPLIT (Gate 1, spec 2026-07-31-gate1-sparse-logits-exact-d2f).
    # ``fwd_logits`` is ALREADY [len(forward_rows), vocab] -- the sparse result
    # exists here and the dense tensor below was only ever built "so the decode
    # loop is unchanged". Callers that can consume the sparse form take it and
    # skip a [seq_len, vocab] allocation entirely; everyone else gets the
    # byte-identical dense tensor as before. NO forward math is touched by this:
    # every value in ``fwd_logits`` is computed above and is unmodified here.
    sparse = SparseForwardLogits(
        rows=tuple(forward_rows), logits=fwd_logits, seq_len=seq_len
    )
    if return_sparse:
        # The [seq_len, vocab] allocation and its mx.eval are skipped entirely.
        full = sparse
    else:
        full = mx.zeros((seq_len, vocab), dtype=fwd_logits.dtype)
        start = forward_rows[0]
        full[start : start + len(forward_rows)] = fwd_logits
        mx.eval(full)
    # Promote any pending frozen blocks captured this forward (their rows were
    # just recomputed from their final tokens) into the served-from-cache set.
    d2f_strategy.promote_pending()
    if timing_metrics is not None:
        _accumulate_timing_ms(
            timing_metrics, "mlx_timing_final_norm_lm_head_ms", final_started_at
        )
        _accumulate_timing_ms(
            timing_metrics, "mlx_timing_forward_total_ms", forward_started_at
        )
    return full


def _transformer_layer_forward_d2f_exact(
    *,
    weights: Any,
    hidden: Any,
    layer_index: int,
    d2f_strategy: Any,
    fresh_offset: int,
    cached_len: int,
    pending_rows: list[tuple[int, int, int]],
    expected_kv_len: int,
    q_positions: list[int],
    n_heads: int,
    num_kv_heads: int,
    head_dim: int,
    eps: float,
    cos_fwd: Any,
    sin_fwd: Any,
    cos_fresh: Any,
    sin_fresh: Any,
    mask_geometry: Any,
    apply_qk_norm: bool,
    kernels: KernelTable,
    attention_mask: mlx_ops.PreparedAttentionMask | None = None,
):
    """One transformer layer over the active rows, attending exact cached K/V.

    ``hidden`` carries the forward rows. Q is projected for all of them; K/V are
    freshly projected ONLY for the fresh rows (``hidden`` at/after
    ``fresh_offset`` — the pending blocks plus the active block) and
    concatenated AFTER the cached parts (prompt, then each frozen block, in
    absolute order), so the kv axis is exactly the absolute positions
    ``[0 : expected_kv_len)``. A cached boundary/feeder row contributes only its
    Q (its K/V lives in the cache — never double-counted). Pending blocks' fresh
    K/V slices are staged on the strategy for promotion at the forward's end.

    The retired row-skip and delayed-K/V branches are absent: every fresh row
    is projected once and its payload is handed to the native exact owner.
    """
    import mlx.core as mx

    residual = hidden
    normalized = kernels.rms_norm(
        hidden,
        mx.array(_weight(weights, f"model.layers.{layer_index}.input_layernorm.weight")),
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
            mx.array(_weight(weights, f"model.layers.{layer_index}.self_attn.q_norm.weight")),
            eps,
        )
    q = kernels.rope(q, cos_fwd, sin_fwd)

    # Fresh K/V for the rows at/after the boundary offset (pending + active).
    fresh_normalized = normalized[fresh_offset:]
    fresh_count = fresh_normalized.shape[0]

    # Project every fresh row. K2-3 deliberately has no delayed-K/V product hook.
    k_fresh = kernels.linear(
        fresh_normalized,
        _linear_weight(weights, f"model.layers.{layer_index}.self_attn.k_proj.weight"),
        _optional_bias(weights, f"model.layers.{layer_index}.self_attn.k_proj.bias"),
    )
    v_fresh = kernels.linear(
        fresh_normalized,
        _linear_weight(weights, f"model.layers.{layer_index}.self_attn.v_proj.weight"),
        _optional_bias(weights, f"model.layers.{layer_index}.self_attn.v_proj.bias"),
    )
    k_fresh = mx.reshape(k_fresh, (fresh_count, num_kv_heads, head_dim))
    v_fresh = mx.reshape(v_fresh, (fresh_count, num_kv_heads, head_dim))
    if apply_qk_norm:
        k_fresh = kernels.rms_norm(
            k_fresh,
            mx.array(_weight(weights, f"model.layers.{layer_index}.self_attn.k_norm.weight")),
            eps,
        )
    k_fresh = kernels.rope(k_fresh, cos_fresh, sin_fresh)
    # Keep K/V compact at num_kv_heads. The shared attention primitive consumes
    # grouped K/V directly, so persistent storage never pays the query-head
    # expansion factor.

    # Stage each pending frozen block's (RoPE'd, compact-GQA) K/V slice — its
    # rows sit inside the fresh span at offset (start - cached_len).
    for block_index, row_start, row_end in pending_rows:
        lo = row_start - cached_len
        hi = row_end - cached_len
        d2f_strategy.capture_block_layer(
            block_index=block_index,
            layer_index=layer_index,
            key=k_fresh[lo:hi],
            value=v_fresh[lo:hi],
        )

    k_window = k_fresh
    v_window = v_fresh

    cached_keys, cached_values = d2f_strategy.cached_kv_parts(layer_index=layer_index)
    # Borrowed parts remain pinned by this forward's manager lease. Assemble
    # cached and fresh rows in one pass, without a separate cached-prefix copy.
    k_full = mx.concatenate([*cached_keys, k_window], axis=0) if cached_keys else k_window
    v_full = mx.concatenate([*cached_values, v_window], axis=0) if cached_values else v_window
    if int(k_full.shape[0]) != int(expected_kv_len):
        raise ValueError(
            "d2f exact KV cache assembled a wrong-length kv axis: got "
            f"{int(k_full.shape[0])} rows, expected {int(expected_kv_len)} — "
            "the cached parts do not cover [0, cached_len)"
        )

    attention_output = kernels.attention(
        q,
        k_full,
        v_full,
        head_dim=head_dim,
        kind="d2f_hybrid",
        mask_geometry=mask_geometry if attention_mask is None else None,
        q_positions=q_positions if attention_mask is None else None,
        prepared_mask=attention_mask,
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
        mx.array(_weight(weights, f"model.layers.{layer_index}.post_attention_layernorm.weight")),
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
