"""``NumpyRefBackendRunner`` — the literal, op-by-op NumPy interpreter of the
Graph IR. This is the EXECUTABLE SPEC: it consumes a :class:`GraphIR` + a weights
mapping + token ids and produces full-vocab logits, walking each op in order and
reading that op's ``params`` / ``weight_keys`` straight from the IR.

It is numpy-only (no mlx, no torch) and computes in fp64 — the high-precision
reference the cross-backend parity gate measures the MLX fp32 forward against
(top-1 / top-5 at the decode positions). If this interpreter cannot run a family,
the IR is incomplete by construction.

The per-op math mirrors ``edllm.backends.mlx_ops`` exactly (same RoPE split-half
convention, same RMSNorm, same bidirectional SDPA, same SwiGLU). The op handlers
are module-level functions so each is independently unit-testable. The interpreter
threads a tiny set of named slots (residual / hidden / q / k / v / attn) through
the fixed v0 op sequence; each op reads/writes slots per the v0 contract while its
parameters come from the IR data, not from any ``if family ==``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from ..runtime_config import normalize_rope_scaling
from .ir import (
    CANONICAL_LAYER_OP_ORDER,
    NO_MASK_ATTENTION_KINDS,
    GraphIR,
    Op,
    same_mask_kind_family,
    uniform_attention_mask_kind,
)

# Logits conventions (kept in sync with ir.LOGITS_CONVENTIONS); applied by the
# runtime's selection step, exposed here as a helper for callers that want the
# selection-ready view.
_DREAM_AR_SHIFT = "dream_ar_shift"

# The D2F hybrid block-causal mask kind (within-block bidirectional + between-
# block causal; the prompt is one fully-visible irregular segment).
_D2F_HYBRID = "d2f_hybrid"


def _validated_mask_geometry(mask_geometry: Any) -> tuple[int, int]:
    """Validate the d2f_hybrid mask GEOMETRY (per-request runtime state).

    The geometry flows as forward-call data (like ``masked_positions``), never
    into the IR. It must be a mapping with exactly ``prompt_len`` (>= 0) and
    ``block_size`` (>= 1), both int-like. Fail-closed on anything else.
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


# --------------------------------------------------------------------------- #
# Per-op math (numpy fp64) — independently unit-tested. Each mirrors mlx_ops.
# --------------------------------------------------------------------------- #

def op_rms_norm(x: np.ndarray, weight: np.ndarray, *, eps: float) -> np.ndarray:
    """RMSNorm over the last axis: ``x * rsqrt(mean(x^2)+eps) * weight``."""
    xf = np.asarray(x, dtype=np.float64)
    variance = np.mean(xf * xf, axis=-1, keepdims=True)
    normalized = xf / np.sqrt(variance + eps)
    return normalized * np.asarray(weight, dtype=np.float64)


def _validated_linear_rope_factor(rope_scaling: Any) -> float:
    """Return the linear-scaling factor, failing closed on unsupported types.

    Only ``linear``/``default`` (and absent) rope_scaling are implemented here
    and on the MLX fast path; both read only the ``factor`` key. yarn/dynamic/su
    would be silently misread as linear, so reject them explicitly.
    """
    return float(normalize_rope_scaling(rope_scaling)["factor"])


def op_qk_norm_per_head(x: np.ndarray, weight: np.ndarray, *, eps: float) -> np.ndarray:
    """Per-head RMSNorm over the head_dim (last) axis of a [seq, heads, head_dim] x.

    Identical math to :func:`op_rms_norm` (the reduction is over the last axis),
    but named for the Qwen3 q_norm/k_norm op so the IR walk is self-documenting.
    """
    return op_rms_norm(x, weight, eps=eps)


def _rope_tables(seq_len: int, head_dim: int, rope_theta: float, rope_scaling: Any):
    half = head_dim // 2
    positions = np.arange(seq_len, dtype=np.float64)
    # I1: fail closed on unsupported rope_scaling types. Only linear/default
    # (which read the `factor` key) are implemented; yarn/dynamic/su would be
    # silently misread as linear, producing wrong angle tables on BOTH the
    # reference and fast paths, so parity cannot catch it.
    factor = _validated_linear_rope_factor(rope_scaling)
    inv_freq = 1.0 / (rope_theta ** (np.arange(0, half, dtype=np.float64) / half))
    inv_freq = inv_freq / max(1e-6, factor)
    freqs = np.einsum("i,j->ij", positions, inv_freq)
    cos = np.cos(freqs)[:, None, :]
    sin = np.sin(freqs)[:, None, :]
    return cos, sin


def op_rope_split_half(
    x: np.ndarray, *, head_dim: int, rope_theta: float, rope_scaling: Any
) -> np.ndarray:
    """Split-half RoPE on a [seq, heads, head_dim] tensor.

    ``out[:half] = x[:half]*cos - x[half:]*sin`` ;
    ``out[half:] = x[:half]*sin + x[half:]*cos`` (the executor / HF convention,
    equal to ``mlx_ops.apply_split_half_rope`` whose cos/sin are the half tables
    duplicated). At position 0 every angle is 0 -> identity.
    """
    xf = np.asarray(x, dtype=np.float64)
    half = head_dim // 2
    if half <= 0:
        return xf
    seq_len = xf.shape[0]
    cos, sin = _rope_tables(seq_len, head_dim, rope_theta, rope_scaling)
    out = np.empty_like(xf)
    out[..., :half] = xf[..., :half] * cos - xf[..., half:] * sin
    out[..., half:] = xf[..., :half] * sin + xf[..., half:] * cos
    return out


def op_gqa_repeat(kv: np.ndarray, *, kv_repeat: int) -> np.ndarray:
    """Repeat each KV head ``kv_repeat`` times along the head axis (no-op if 1)."""
    if kv_repeat == 1:
        return np.asarray(kv, dtype=np.float64)
    return np.repeat(np.asarray(kv, dtype=np.float64), kv_repeat, axis=1)


def op_attention_bidirectional(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, *, head_dim: int
) -> np.ndarray:
    """Bidirectional scaled-dot-product attention -> [seq, n_heads*head_dim]."""
    qf = np.asarray(q, dtype=np.float64)
    kf = np.asarray(k, dtype=np.float64)
    vf = np.asarray(v, dtype=np.float64)
    scores = np.einsum("thd,shd->hts", qf, kf) / math.sqrt(float(head_dim))
    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    output = np.einsum("hts,shd->thd", probs, vf)
    return output.reshape(output.shape[0], -1)


def d2f_hybrid_visibility(
    seq_len: int, *, prompt_len: int, block_size: int
) -> np.ndarray:
    """Boolean visibility matrix for the D2F hybrid block-causal mask.

    Pinned to the D2F reference (``create_full_block_attention_mask`` /
    ``build_custom_float_attention_mask`` in ``D2F-eval/eval_dream.py`` of
    github.com/SJTU-DENG-Lab/Discrete-Diffusion-Forcing; paper arXiv 2508.09192):

    * the PROMPT ``[0, prompt_len)`` is ONE fully-visible irregular segment —
      prompt rows attend to exactly the prompt columns;
    * the generated region is partitioned into half-open blocks of
      ``block_size`` starting AT the prompt boundary, the last block clipped at
      ``seq_len``;
    * a generated row in block b attends to the full prompt, every earlier
      block, and its own block (bidirectional within the block) — nothing later.

    Closed form: ``eff(p) = -1 if p < prompt_len else (p - prompt_len) //
    block_size``; ``visible[i, j] = eff(j) <= eff(i)``.
    """
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
    return eff[None, :] <= eff[:, None]


def op_attention_d2f_hybrid(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    head_dim: int,
    prompt_len: int,
    block_size: int,
) -> np.ndarray:
    """D2F hybrid block-causal SDPA -> [seq, n_heads*head_dim].

    Identical math to :func:`op_attention_bidirectional` plus the reference's
    ADDITIVE float mask (0 where visible, -inf where masked) applied to the
    scores before the softmax. Every row keeps at least its own block visible
    (the diagonal), so no softmax row is fully masked.
    """
    qf = np.asarray(q, dtype=np.float64)
    kf = np.asarray(k, dtype=np.float64)
    vf = np.asarray(v, dtype=np.float64)
    if qf.shape[0] != kf.shape[0]:
        raise ValueError(
            "d2f_hybrid attention requires square q/kv (q rows are the absolute "
            f"sequence positions); got q_len={qf.shape[0]} kv_len={kf.shape[0]}"
        )
    visible = d2f_hybrid_visibility(
        qf.shape[0], prompt_len=prompt_len, block_size=block_size
    )
    bias = np.where(visible, 0.0, -np.inf)
    scores = np.einsum("thd,shd->hts", qf, kf) / math.sqrt(float(head_dim)) + bias
    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    output = np.einsum("hts,shd->thd", probs, vf)
    return output.reshape(output.shape[0], -1)


def op_linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None) -> np.ndarray:
    """Affine ``x @ weight.T (+ bias)``."""
    y = np.asarray(x, dtype=np.float64) @ np.asarray(weight, dtype=np.float64).T
    if bias is not None:
        y = y + np.asarray(bias, dtype=np.float64)
    return y


def op_mlp_swiglu(
    x: np.ndarray, gate_w: np.ndarray, up_w: np.ndarray, down_w: np.ndarray
) -> np.ndarray:
    """SwiGLU MLP: ``(silu(x@gate.T) * (x@up.T)) @ down.T``."""
    xf = np.asarray(x, dtype=np.float64)
    gate = xf @ np.asarray(gate_w, dtype=np.float64).T
    up = xf @ np.asarray(up_w, dtype=np.float64).T
    silu = gate / (1.0 + np.exp(-gate))
    return (silu * up) @ np.asarray(down_w, dtype=np.float64).T


def op_embedding(table: np.ndarray, token_ids: list[int]) -> np.ndarray:
    return np.asarray(table, dtype=np.float64)[list(token_ids)]


def shift_logits_for_dream(logits: np.ndarray) -> np.ndarray:
    """Dream AR shift: row ``i`` reads the prediction from row ``i-1`` (row 0 kept)."""
    arr = np.asarray(logits, dtype=np.float64)
    return np.concatenate([arr[:1], arr[:-1]], axis=0)


def apply_logits_convention(logits: np.ndarray, convention: str) -> np.ndarray:
    """Apply the package logits convention (the runtime's decode-time selection)."""
    if convention == _DREAM_AR_SHIFT:
        return shift_logits_for_dream(logits)
    return np.asarray(logits, dtype=np.float64)


# --------------------------------------------------------------------------- #
# The interpreter.
# --------------------------------------------------------------------------- #

def _weight(weights: Any, key: str) -> np.ndarray:
    if key not in weights:
        raise KeyError(f"missing weight key: {key}")
    return np.asarray(weights[key], dtype=np.float64)


def _optional_weight(weights: Any, key: str | None) -> np.ndarray | None:
    if key is None:
        return None
    if key not in weights:
        # I4: a declared weight (e.g. bias_present=True => a bias key) that is
        # absent from the archive must raise, not silently return None. Dream
        # q/k/v bias is load-bearing; a dropped bias would make the oracle
        # silently wrong, and both paths agreeing means parity cannot catch it.
        raise ValueError(
            f"declared weight {key!r} is missing from the weights archive"
        )
    return np.asarray(weights[key], dtype=np.float64)


class _LayerState:
    """Mutable named slots threaded through one layer's op walk (the v0 contract).

    The attention-side ops (linear / qk_norm / rope) each carry a ``role``
    ("q"|"k"|"v"|"o") as op DATA, so the interpreter dispatches each to its slot by
    IDENTITY, not by position — a reordered-but-same-type graph routes correctly
    (or raises on an unrecognized role) instead of being silently mis-assigned.
    """

    __slots__ = ("residual", "hidden", "q", "k", "v", "attn")

    def __init__(self, hidden: np.ndarray) -> None:
        self.residual = hidden
        self.hidden = hidden
        self.q: np.ndarray | None = None
        self.k: np.ndarray | None = None
        self.v: np.ndarray | None = None
        self.attn: np.ndarray | None = None


# The attention-projection roles, by the substring each carries in its weight key
# (the fallback when an op predates the explicit ``role`` param).
_ATTN_ROLES: tuple[str, ...] = ("q", "k", "v", "o")


def _proj_role(op: Op) -> str:
    """The attention-projection role ("q"|"k"|"v"|"o") of a ``linear`` op.

    Prefer the explicit ``role`` op param (current emitter); otherwise infer it
    from the projection name in the weight key (``q_proj`` / ... / ``o_proj``) so
    graphs that predate the param still route by identity. Raise on an
    unrecognizable projection rather than guess — silent mis-routing is the very
    bug this closes.
    """
    role = op.params.get("role")
    if role in _ATTN_ROLES:
        return str(role)
    weight_key = op.weight_keys.get("weight", "")
    for candidate in _ATTN_ROLES:
        if f".{candidate}_proj." in weight_key:
            return candidate
    raise ValueError(
        f"linear op carries no recognizable attention-projection role "
        f"(params role={role!r}, weight_key={weight_key!r})"
    )


def _qk_side(op: Op) -> str:
    """The q-or-k side ("q"|"k") a qk_norm/rope op targets, by op DATA.

    Prefer the explicit ``role`` param; otherwise (qk_norm only, when applied)
    infer from the ``q_norm`` / ``k_norm`` weight key. Raise if neither is present.
    """
    role = op.params.get("role")
    if role in ("q", "k"):
        return str(role)
    weight_key = op.weight_keys.get("weight", "")
    if ".q_norm." in weight_key:
        return "q"
    if ".k_norm." in weight_key:
        return "k"
    raise ValueError(
        f"{op.type} op carries no recognizable q/k side "
        f"(params role={role!r}, weight_key={weight_key!r})"
    )


def _run_layer(
    layer_ops: tuple[Op, ...],
    state: _LayerState,
    *,
    weights: Any,
    n_heads: int,
    num_kv_heads: int,
    head_dim: int,
    d2f_geometry: tuple[int, int] | None = None,
) -> np.ndarray:
    """Interpret one layer's op list against ``state``; return the layer output."""
    # B1 — op-order assertion: a type-level corruption (a reordered or wrong-typed
    # op) must fail loudly, not be mis-run. The canonical order is the single
    # source of truth in ir.py (the emitter test pins the same constant).
    actual_order = tuple(op.type for op in layer_ops)
    if actual_order != CANONICAL_LAYER_OP_ORDER:
        raise ValueError(
            "layer op order does not match the v0 canonical contract: "
            f"got {actual_order} expected {CANONICAL_LAYER_OP_ORDER}"
        )

    seq_len = state.hidden.shape[0]

    for op in layer_ops:
        t = op.type
        if t == "rms_norm":
            weight = _weight(weights, op.weight_keys["weight"])
            state.hidden = op_rms_norm(state.hidden, weight, eps=float(op.params["eps"]))

        elif t == "linear":
            weight = _weight(weights, op.weight_keys["weight"])
            bias = _optional_weight(weights, op.weight_keys.get("bias"))
            role = _proj_role(op)
            # q/k/v project from the normalized hidden; o_proj projects from the
            # attention output. Dispatch by the op's ROLE (its identity), so a
            # same-type reorder (e.g. k_proj <-> v_proj) routes each projection to
            # its correct slot instead of being mis-assigned positionally. Selecting
            # the input before the matmul is also required when hidden_size !=
            # n_heads*head_dim (e.g. Qwen3: 1024 vs 2048).
            if role == "q":
                state.q = op_linear(state.hidden, weight, bias).reshape(
                    seq_len, n_heads, head_dim
                )
            elif role == "k":
                state.k = op_linear(state.hidden, weight, bias).reshape(
                    seq_len, num_kv_heads, head_dim
                )
            elif role == "v":
                state.v = op_linear(state.hidden, weight, bias).reshape(
                    seq_len, num_kv_heads, head_dim
                )
            else:  # role == "o": read the attention output, write back to hidden
                state.hidden = op_linear(state.attn, weight, bias)

        elif t == "qk_norm":
            if op.params.get("apply_qk_norm"):
                weight = _weight(weights, op.weight_keys["weight"])
                side = _qk_side(op)
                target = state.q if side == "q" else state.k
                if target is None:
                    raise ValueError(
                        f"qk_norm targets the {side!r} slot before it was projected "
                        "(out-of-contract op order)"
                    )
                normed = op_qk_norm_per_head(target, weight, eps=float(op.params["eps"]))
                if side == "q":
                    state.q = normed
                else:
                    state.k = normed
            # absent / not applied -> no-op (Dream has no q_norm/k_norm tensors)

        elif t == "rope":
            params = op.params
            side = _qk_side(op)
            roped_target = state.q if side == "q" else state.k
            if roped_target is None:
                raise ValueError(
                    f"rope targets the {side!r} slot before it was projected "
                    "(out-of-contract op order)"
                )
            roped = op_rope_split_half(
                roped_target,
                head_dim=int(params["head_dim"]),
                rope_theta=float(params["rope_theta"]),
                rope_scaling=params.get("rope_scaling"),
            )
            if side == "q":
                state.q = roped
            else:
                state.k = roped

        elif t == "gqa_repeat":
            kv_repeat = int(op.params["kv_repeat"])
            state.k = op_gqa_repeat(state.k, kv_repeat=kv_repeat)
            state.v = op_gqa_repeat(state.v, kv_repeat=kv_repeat)

        elif t == "attention":
            # The mask FAMILY is op DATA (Graph IR ``mask_kind``); the mask
            # GEOMETRY is forward-call data. Dispatch by the IR, fail-closed on
            # anything the interpreter cannot honor.
            mask_kind = str(op.params.get("mask_kind", "bidirectional"))
            if mask_kind in NO_MASK_ATTENTION_KINDS:
                state.attn = op_attention_bidirectional(
                    state.q, state.k, state.v, head_dim=int(op.params["head_dim"])
                )
            elif mask_kind == _D2F_HYBRID:
                if d2f_geometry is None:
                    raise ValueError(
                        "d2f_hybrid attention requires mask geometry "
                        "(prompt_len/block_size) supplied as forward-call data"
                    )
                prompt_len, block_size = d2f_geometry
                state.attn = op_attention_d2f_hybrid(
                    state.q,
                    state.k,
                    state.v,
                    head_dim=int(op.params["head_dim"]),
                    prompt_len=prompt_len,
                    block_size=block_size,
                )
            else:
                raise ValueError(
                    f"NumpyRef cannot interpret attention mask_kind: {mask_kind!r}"
                )

        elif t == "residual_add":
            # Both residual_add ops are `residual + current-hidden`; after the add
            # the sum becomes the residual for the next block (the post-attn add
            # feeds the MLP block; the post-mlp add yields the layer output).
            state.hidden = state.residual + state.hidden
            state.residual = state.hidden

        elif t == "mlp_swiglu":
            gate_w = _weight(weights, op.weight_keys["gate"])
            up_w = _weight(weights, op.weight_keys["up"])
            down_w = _weight(weights, op.weight_keys["down"])
            state.hidden = op_mlp_swiglu(state.hidden, gate_w, up_w, down_w)

        else:  # pragma: no cover - guarded by Op.__post_init__
            raise ValueError(f"NumpyRef cannot interpret op type: {t!r}")

    return state.hidden


def interpret_graph(
    graph: GraphIR,
    *,
    weights: Any,
    token_ids: list[int],
    mask_geometry: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Walk the whole graph op-by-op in fp64 -> raw logits [seq_len, vocab].

    The returned logits are the RAW forward output; the package
    ``logits_convention`` (the decode-time selection) is NOT applied here — use
    :func:`apply_logits_convention` for the selection-ready view.

    ``mask_geometry`` (``prompt_len`` / ``block_size``) is the per-request
    runtime state consumed by a ``d2f_hybrid`` attention op; it is forward-call
    data, never IR data.
    """
    d2f_geometry = (
        _validated_mask_geometry(mask_geometry) if mask_geometry is not None else None
    )
    # embedding
    table = _weight(weights, graph.embedding.weight_keys["embedding"])
    hidden = op_embedding(table, token_ids)

    # geometry read from the first layer's ops (IR data).
    first = graph.layers[0]
    attn_op = next(op for op in first.ops if op.type == "attention")
    gqa_op = next(op for op in first.ops if op.type == "gqa_repeat")
    n_heads = int(attn_op.params["n_heads"])
    head_dim = int(attn_op.params["head_dim"])
    num_kv_heads = int(gqa_op.params["num_kv_heads"])

    for layer in graph.layers:
        state = _LayerState(hidden)
        hidden = _run_layer(
            layer.ops,
            state,
            weights=weights,
            n_heads=n_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            d2f_geometry=d2f_geometry,
        )

    # final_norm
    hidden = op_rms_norm(
        hidden,
        _weight(weights, graph.final_norm.weight_keys["weight"]),
        eps=float(graph.final_norm.params["eps"]),
    )
    # lm_head
    logits = op_linear(hidden, _weight(weights, graph.lm_head.weight_keys["weight"]), None)
    return logits


class NumpyRefBackendRunner:
    """A ``BackendRunner`` that interprets the Graph IR in NumPy (the spec runner).

    Signature-compatible with the MLX runner's ``full_forward`` on the args the
    interpreter needs; the diffusion-layer-only kwargs (``kv_cache_state`` /
    ``state_cache_manager`` / ``masked_positions`` / ``timing_metrics``) are
    accepted and IGNORED — the IR describes forward math only, no cache logic.
    ``model_config`` is likewise accepted-and-ignored (the ``graph`` is the
    authoritative description of the forward math), so this runner is a DROP-IN
    for ``MLXBackendRunner`` behind the engine seam: callable with the exact same
    kwargs the engine threads to the MLX runner (which includes ``model_config``).
    """

    __slots__ = ()

    def full_forward(
        self,
        *,
        graph: GraphIR,
        weights: Any,
        token_ids: list[int],
        model_config: dict[str, Any] | None = None,
        attention_mask_kind: str = "bidirectional",
        kv_cache_state: Any | None = None,
        timing_metrics: dict[str, Any] | None = None,
        state_cache_manager: Any | None = None,
        masked_positions: list[int] | None = None,
        mask_geometry: Mapping[str, Any] | None = None,
        kernels: Any | None = None,
    ) -> np.ndarray:
        # ``model_config`` is accepted for ABI-compatibility with MLXBackendRunner
        # and ignored: when ``graph`` is supplied it is authoritative. ``kernels``
        # (the M2-1 per-op MLX KernelTable) likewise does not apply here — this
        # runner IS the numpy fp64 reference the kernel tables are gated against.
        del model_config, kv_cache_state, timing_metrics
        del state_cache_manager, masked_positions, kernels
        # The IR's attention ops carry the mask FAMILY; the runner supports the
        # no-mask kinds and d2f_hybrid, and fails closed on anything else.
        if attention_mask_kind not in NO_MASK_ATTENTION_KINDS | {_D2F_HYBRID}:
            raise ValueError(
                "NumpyRef supports bidirectional/full/d2f_hybrid attention only, "
                f"got {attention_mask_kind!r}"
            )
        # The kwarg (what the caller asked to run) and the Graph IR mask_kind
        # (what the graph describes) must agree at the family level — running
        # either interpretation silently would diverge from the other backend.
        ir_kind = uniform_attention_mask_kind(graph)
        if not same_mask_kind_family(attention_mask_kind, ir_kind):
            raise ValueError(
                f"attention_mask_kind kwarg {attention_mask_kind!r} disagrees with "
                f"the Graph IR attention mask_kind {ir_kind!r}; refusing to run "
                "either interpretation silently"
            )
        # Geometry is consumed ONLY by d2f_hybrid: required there, rejected
        # elsewhere (no silently-dead arguments).
        if attention_mask_kind == _D2F_HYBRID:
            if mask_geometry is None:
                raise ValueError(
                    "d2f_hybrid requires mask geometry "
                    "(prompt_len/block_size) supplied as forward-call data"
                )
        elif mask_geometry is not None:
            raise ValueError(
                "mask geometry is only consumed by the d2f_hybrid attention kind; "
                f"got geometry with attention_mask_kind={attention_mask_kind!r}"
            )
        return interpret_graph(
            graph, weights=weights, token_ids=token_ids, mask_geometry=mask_geometry
        )
