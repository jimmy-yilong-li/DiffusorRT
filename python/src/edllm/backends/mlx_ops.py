from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

# Attention mask kinds: the no-mask family and the D2F hybrid block-causal kind
# (within-block bidirectional + between-block causal; prompt = one fully-visible
# segment). Kept as module data so the validation below stays table-driven.
_NO_MASK_KINDS = frozenset({"full", "bidirectional"})
_D2F_HYBRID = "d2f_hybrid"


def _mx():
    import mlx.core as mx

    return mx


def rms_norm(x: Any, weight: Any, eps: float):
    """RMSNorm with fp32-accumulated variance (mixed precision).

    The variance reduction (``mean(x*x)``) and the normalize are computed in
    fp32 regardless of the input dtype, then cast back to the input dtype — bf16
    inputs get an fp32-accurate norm (the sensitive op), while an fp32 input is
    untouched (``astype(fp32)`` is a no-op, so the fp32 path stays byte-identical).
    The weight multiply happens in the input dtype after the renormalize.
    """
    mx = _mx()
    in_dtype = x.dtype
    xf = x.astype(mx.float32)
    variance = mx.mean(xf * xf, axis=-1, keepdims=True)
    normalized = (xf * mx.rsqrt(variance + eps)).astype(in_dtype)
    return normalized * weight


# Quantized-linear contract (Quant-1): only int8 / int4 ship. Anything else is
# out of contract and fails closed at quantize time AND at config resolution.
_LINEAR_QUANT_BITS = frozenset({8, 4})


@dataclass(frozen=True, slots=True)
class QuantizedWeight:
    """An immutable quantized linear-weight bundle for the ``linear`` kernel.

    Produced ONCE at weight-prep time by :func:`quantize_linear_weight` from the
    fp ``(out, in)`` projection matrix via ``mx.quantize``; consumed by the
    polymorphic :func:`linear`, which routes it through ``mx.quantized_matmul``
    with ``transpose=True`` (the linear convention is ``x @ W.T``). ``shape`` is
    the ORIGINAL fp ``(out, in)`` shape (``w_q`` is bit-packed, so its own shape
    is not the matmul geometry).
    """

    w_q: Any
    scales: Any
    biases: Any
    group_size: int
    bits: int
    shape: tuple[int, int]


def quantize_linear_weight(weight: Any, *, group_size: int, bits: int) -> QuantizedWeight:
    """Quantize one 2D fp linear weight into a :class:`QuantizedWeight`.

    Fail-closed: only 2D weights, only int8/int4, and the in-dim (the grouped
    reduction axis) must divide by ``group_size`` — the weight-prep hook checks
    divisibility FIRST and keeps fp for the rare indivisible weight; calling
    this directly with one is a bug, not a fallback.
    """
    mx = _mx()
    if bits not in _LINEAR_QUANT_BITS:
        raise ValueError(
            f"unsupported linear quantization bits: {bits!r} (expected 8 or 4)"
        )
    shape = tuple(int(dim) for dim in getattr(weight, "shape", ()))
    if len(shape) != 2:
        raise ValueError(
            f"quantize_linear_weight requires a 2D (out, in) weight; got shape {shape}"
        )
    if shape[1] % int(group_size) != 0:
        raise ValueError(
            f"linear weight in-dim {shape[1]} is not divisible by "
            f"group_size {group_size}"
        )
    w_q, scales, biases = mx.quantize(weight, group_size=int(group_size), bits=int(bits))
    return QuantizedWeight(
        w_q=w_q,
        scales=scales,
        biases=biases,
        group_size=int(group_size),
        bits=int(bits),
        shape=(shape[0], shape[1]),
    )


def linear(x: Any, weight: Any, bias: Any | None = None):
    """Affine projection ``x @ weight.T (+ bias)`` — the matmul boundary.

    Polymorphic over the WEIGHT operand (Quant-1):

    * a :class:`QuantizedWeight` bundle routes through ``mx.quantized_matmul``
      (``transpose=True`` == the same ``x @ W.T`` convention); the output dtype
      follows ``x``, so the fp32 residual stream is preserved;
    * a plain array takes the pre-existing fp path BYTE-FOR-BYTE: the
      activation stream is fp32 (accumulation parity) and the weight carries
      the compute dtype — a bf16 weight casts the activation to bf16 so the
      matmul runs bf16xbf16 (the bandwidth lever) and returns fp32; an fp32
      weight does the plain fp32 matmul with NO cast.
    """
    mx = _mx()
    if isinstance(weight, QuantizedWeight):
        y = mx.quantized_matmul(
            x,
            weight.w_q,
            weight.scales,
            weight.biases,
            transpose=True,
            group_size=weight.group_size,
            bits=weight.bits,
        )
    elif weight.dtype != x.dtype:
        # bf16 weight + fp32 activation: do the matmul in the weight dtype
        # (bf16xbf16, fp32 accumulate), then return fp32 for the residual stream.
        y = (x.astype(weight.dtype) @ weight.T).astype(x.dtype)
    else:
        y = x @ weight.T
    if bias is not None:
        y = y + bias.astype(y.dtype)
    return y


def apply_split_half_rope(x: Any, cos: Any, sin: Any):
    # RoPE angle tables (cos/sin) are built in fp32 (numpy) and stay fp32; the
    # rotate+blend runs in fp32 for accuracy then casts back to x's dtype. On an
    # fp32 x with fp32 tables every cast is a no-op -> byte-identical.
    mx = _mx()
    in_dtype = x.dtype
    xf = x.astype(mx.float32)
    half = xf.shape[-1] // 2
    first = xf[..., :half]
    second = xf[..., half:]
    rotated = mx.concatenate([-second, first], axis=-1)
    return (xf * cos.astype(mx.float32) + rotated * sin.astype(mx.float32)).astype(in_dtype)


def swiglu(gate: Any, up: Any):
    # SwiGLU activation in fp32 (sigmoid is the sensitive op), result back to the
    # input dtype. fp32 inputs -> no-op casts -> byte-identical.
    mx = _mx()
    in_dtype = gate.dtype
    gf = gate.astype(mx.float32)
    uf = up.astype(mx.float32)
    return (gf * mx.sigmoid(gf) * uf).astype(in_dtype)


def shift_logits_for_dream(logits: Any):
    mx = _mx()
    return mx.concatenate([logits[:1], logits[:-1]], axis=0)


def _validated_mask_geometry(mask_geometry: Any) -> tuple[int, int]:
    """Validate the d2f_hybrid mask GEOMETRY (per-request runtime state).

    Mirrors ``edllm.graph.numpyref._validated_mask_geometry`` (kept separate so
    this module stays import-light): a mapping with exactly ``prompt_len``
    (>= 0) and ``block_size`` (>= 1), both int-like. Fail-closed otherwise.
    """
    if not isinstance(mask_geometry, Mapping):
        raise ValueError(
            "d2f_hybrid requires mask geometry as a mapping with exactly "
            f"prompt_len and block_size; got {mask_geometry!r}"
        )
    if set(mask_geometry.keys()) != {"prompt_len", "block_size"}:
        raise ValueError(
            "mask geometry must carry exactly the keys prompt_len and "
            f"block_size; got {sorted(map(str, mask_geometry.keys()))}"
        )
    try:
        prompt_len = int(mask_geometry["prompt_len"])
        block_size = int(mask_geometry["block_size"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"mask geometry values must be ints: {mask_geometry!r}") from exc
    if prompt_len < 0:
        raise ValueError(f"mask geometry prompt_len must be >= 0, got {prompt_len}")
    if block_size < 1:
        raise ValueError(f"mask geometry block_size must be >= 1, got {block_size}")
    return prompt_len, block_size


def d2f_hybrid_attention_bias(seq_len: int, *, prompt_len: int, block_size: int):
    """Additive fp32 attention bias [seq_len, seq_len] for the d2f_hybrid mask.

    Exactly the D2F reference convention (``create_full_block_attention_mask``
    in ``D2F-eval/eval_dream.py`` of
    github.com/SJTU-DENG-Lab/Discrete-Diffusion-Forcing; paper arXiv
    2508.09192): 0 where row i may attend to column j, -inf elsewhere.

    * the PROMPT ``[0, prompt_len)`` is ONE fully-visible irregular segment —
      prompt rows attend to exactly the prompt columns;
    * the generated region is partitioned into half-open blocks of
      ``block_size`` starting AT the prompt boundary (last block clipped);
    * a generated row in block b attends to the full prompt, all earlier
      blocks, and its own block — nothing later.

    Closed form: ``eff(p) = -1 if p < prompt_len else (p - prompt_len) //
    block_size``; visible iff ``eff(col) <= eff(row)``. The table is built in
    numpy (the ``_rope_tables`` precedent) and lifted to an mx array.
    """
    import numpy as np

    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    if not 0 <= prompt_len <= seq_len:
        raise ValueError(
            f"prompt_len must be in [0, seq_len={seq_len}], got {prompt_len}"
        )
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    idx = np.arange(seq_len, dtype=np.int64)
    eff = np.where(idx < prompt_len, np.int64(-1), (idx - prompt_len) // block_size)
    visible = eff[None, :] <= eff[:, None]
    bias = np.where(visible, 0.0, -np.inf).astype(np.float32)
    return _mx().array(bias)


def _validated_q_positions(q_positions: Any, *, q_len: int, kv_len: int) -> list[int]:
    """Validate the explicit absolute q-row positions for a non-square d2f call.

    ``q_positions[r]`` is the ABSOLUTE sequence position of q row ``r``; the kv
    axis is the absolute positions ``[0, kv_len)``. Fail-closed: must be
    int-like, one per q row, each inside the kv range (a query must be able to
    see at least its own block's keys).
    """
    try:
        positions = [int(p) for p in q_positions]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"q_positions must be int-like: {q_positions!r}") from exc
    if len(positions) != q_len:
        raise ValueError(
            f"q_positions must carry one absolute position per q row; got "
            f"{len(positions)} positions for q_len={q_len}"
        )
    for position in positions:
        if not 0 <= position < kv_len:
            raise ValueError(
                f"q_positions entries must be in [0, kv_len={kv_len}); got {position}"
            )
    return positions


def d2f_hybrid_attention_bias_rows(
    kv_len: int, *, q_positions: Any, prompt_len: int, block_size: int
):
    """Non-square additive fp32 d2f bias ``[len(q_positions), kv_len]`` (D2F-2).

    Row ``r`` is the SAME row the square :func:`d2f_hybrid_attention_bias`
    produces at absolute position ``q_positions[r]``: the closed form
    ``eff(p) = -1 if p < prompt_len else (p - prompt_len) // block_size``,
    visible iff ``eff(col) <= eff(q_positions[r])``. This is what lets the
    exact cross-block KV cache forward only the active rows while attending the
    full cached sequence: the q rows carry their absolute positions explicitly.
    """
    import numpy as np

    if kv_len < 1:
        raise ValueError(f"kv_len must be >= 1, got {kv_len}")
    if not 0 <= prompt_len <= kv_len:
        raise ValueError(
            f"prompt_len must be in [0, kv_len={kv_len}], got {prompt_len}"
        )
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    materialized = list(q_positions)
    positions = _validated_q_positions(
        materialized, q_len=len(materialized), kv_len=kv_len
    )

    def _eff(idx: "np.ndarray") -> "np.ndarray":
        return np.where(
            idx < prompt_len, np.int64(-1), (idx - prompt_len) // block_size
        )

    cols = _eff(np.arange(kv_len, dtype=np.int64))
    rows = _eff(np.asarray(positions, dtype=np.int64))
    visible = cols[None, :] <= rows[:, None]
    bias = np.where(visible, 0.0, -np.inf).astype(np.float32)
    return _mx().array(bias)


@dataclass(frozen=True, slots=True)
class PreparedAttentionMask:
    """Validated mask for one forward's fixed query and K/V row geometry.

    Construct with ``prepare_attention_mask``. It owns one device bias, not a
    session cache; callers drop it after evaluating the last transformer layer.
    """

    kind: str
    q_len: int
    kv_len: int
    bias: Any


def prepare_attention_mask(
    *,
    kind: str,
    q_len: int,
    kv_len: int,
    mask_geometry: Any | None = None,
    q_positions: Any | None = None,
) -> PreparedAttentionMask | None:
    """Build mask/absolute-row metadata once for all layers of one forward."""
    validate_attention_kind(kind=kind, mask_geometry=mask_geometry)
    if q_positions is not None and kind != _D2F_HYBRID:
        raise ValueError(
            "q_positions is only consumed by the d2f_hybrid attention kind; "
            f"got q_positions with kind={kind!r}"
        )
    if kind != _D2F_HYBRID:
        return None
    prompt_len, block_size = _validated_mask_geometry(mask_geometry)
    if q_positions is not None:
        positions = list(q_positions)
        if len(positions) != q_len:
            raise ValueError(
                f"q_positions must have one entry per query row: "
                f"got {len(positions)} for q_len={q_len}"
            )
        bias = d2f_hybrid_attention_bias_rows(
            kv_len, q_positions=positions,
            prompt_len=prompt_len, block_size=block_size,
        )
    elif q_len != kv_len:
        raise ValueError(
            "d2f_hybrid attention requires square q/kv (q rows are the "
            f"absolute sequence positions); got q_len={q_len} "
            f"kv_len={kv_len} and no q_positions"
        )
    else:
        bias = d2f_hybrid_attention_bias(
            q_len, prompt_len=prompt_len, block_size=block_size,
        )
    return PreparedAttentionMask(kind, q_len, kv_len, bias)


def scaled_dot_product_attention(
    q: Any,
    k: Any,
    v: Any,
    *,
    head_dim: int,
    kind: str,
    mask_geometry: Any | None = None,
    q_positions: Any | None = None,
    prepared_mask: PreparedAttentionMask | None = None,
):
    mx = _mx()
    q_shape = tuple(int(dimension) for dimension in q.shape)
    k_shape = tuple(int(dimension) for dimension in k.shape)
    v_shape = tuple(int(dimension) for dimension in v.shape)
    if len(q_shape) != 3 or len(k_shape) != 3 or len(v_shape) != 3:
        raise ValueError("attention expects q, k and v shaped [rows, heads, head_dim]")
    if k_shape != v_shape:
        raise ValueError(f"attention K/V shapes disagree: {k_shape} != {v_shape}")
    if q_shape[0] < 1 or k_shape[0] < 1:
        raise ValueError("attention requires at least one query and K/V row")
    if type(head_dim) is not int or head_dim < 1:
        raise ValueError("attention head_dim must be a positive integer")
    if q_shape[2] != head_dim or k_shape[2] != head_dim:
        raise ValueError(
            "attention tensors disagree with head_dim: "
            f"q={q_shape[2]} k/v={k_shape[2]} requested={head_dim}"
        )
    if k_shape[1] < 1 or q_shape[1] % k_shape[1] != 0:
        raise ValueError(
            "attention query heads must be divisible by compact K/V heads: "
            f"q_heads={q_shape[1]} kv_heads={k_shape[1]}"
        )

    if prepared_mask is None:
        prepared_mask = prepare_attention_mask(
            kind=kind, q_len=q_shape[0], kv_len=k_shape[0],
            mask_geometry=mask_geometry, q_positions=q_positions,
        )
    else:
        if mask_geometry is not None or q_positions is not None:
            raise ValueError("prepared attention mask cannot override raw mask metadata")
        if kind != _D2F_HYBRID or (
            prepared_mask.kind, prepared_mask.q_len, prepared_mask.kv_len
        ) != (kind, q_shape[0], k_shape[0]):
            raise ValueError("prepared attention mask disagrees with attention geometry")
    attention_bias = None if prepared_mask is None else prepared_mask.bias

    out_dtype = q.dtype
    # MLX fast SDPA consumes compact GQA/MQA K/V directly. The boundary stays
    # [rows, heads, head_dim]; only the call layout changes to
    # [batch, heads, rows, head_dim]. Inputs are promoted to fp32 so the
    # numerically-sensitive reduction retains the previous precision contract.
    qf = mx.transpose(q.astype(mx.float32), (1, 0, 2))[None]
    kf = mx.transpose(k.astype(mx.float32), (1, 0, 2))[None]
    vf = mx.transpose(v.astype(mx.float32), (1, 0, 2))[None]
    scale = 1.0 / math.sqrt(float(head_dim))
    # MLX 0.31's fused vector kernel accepts at most 32 query-head groups per
    # launch. Chunk only that narrow product shape; full-sequence attention is
    # already tiled by MLX. Unsupported fixture head dimensions are allowed to
    # use MLX's reference fallback and are never Product admission evidence.
    gqa_factor = q_shape[1] // k_shape[1]
    max_vector_rows = max(1, 32 // gqa_factor)
    if (
        head_dim in {64, 128}
        and gqa_factor <= 32
        and q_shape[0] <= 8
        and q_shape[0] > max_vector_rows
    ):
        pieces = []
        for start in range(0, q_shape[0], max_vector_rows):
            stop = min(start + max_vector_rows, q_shape[0])
            chunk_mask = (
                None
                if attention_bias is None
                else attention_bias[..., start:stop, :]
            )
            pieces.append(
                mx.fast.scaled_dot_product_attention(
                    qf[:, :, start:stop, :],
                    kf,
                    vf,
                    scale=scale,
                    mask=chunk_mask,
                )
            )
        output = mx.concatenate(pieces, axis=2)
    else:
        output = mx.fast.scaled_dot_product_attention(
            qf,
            kf,
            vf,
            scale=scale,
            mask=attention_bias,
        )
    output = mx.transpose(output[0], (1, 0, 2))
    return mx.reshape(output, (output.shape[0], -1)).astype(out_dtype)


def validate_attention_kind(*, kind: str, mask_geometry: Any | None = None):
    """Fail-closed mask-kind + geometry validation.

    The no-mask kinds reject a supplied geometry (no silently-dead arguments);
    ``d2f_hybrid`` requires a well-formed geometry; anything else is closed.
    """
    if kind in _NO_MASK_KINDS:
        if mask_geometry is not None:
            raise ValueError(
                "mask geometry is only consumed by the d2f_hybrid attention "
                f"kind; got geometry with kind={kind!r}"
            )
        return None
    if kind == _D2F_HYBRID:
        if mask_geometry is None:
            raise ValueError(
                "d2f_hybrid requires mask geometry "
                "(prompt_len/block_size) supplied as forward-call data"
            )
        _validated_mask_geometry(mask_geometry)
        return None
    raise ValueError(f"unsupported MLX attention mask kind: {kind}")
