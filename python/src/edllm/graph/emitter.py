"""Emit a :class:`GraphIR` from an enriched ``model_config``.

The emitter is the bridge from today's config-driven forward to the described,
executable Graph IR. It derives the op-sequence from architecture-agnostic DATA:

* the geometry (``layer_count`` / ``hidden_size`` / ``attention_heads`` /
  ``num_kv_heads`` / ``head_dim`` / ``rope_theta`` / ``rope_scaling`` /
  ``rms_norm_eps``), read straight from the config;
* the family-conditional bits, sourced from the resolved
  :class:`RuntimeArchitectureSpec` (Q/K norm, logits convention and q/k/v bias).
  These become op DATA, never ``if family ==`` inside the emitter.

Product conversion and loading pass the already selected runtime spec. Repeated
config flags may only agree with it. A config-only inference path remains for
pre-U0 fixtures. ``mlx`` / ``numpy`` are never imported here — the emitter is
pure definition + config -> dataclasses.
"""

from __future__ import annotations

from typing import Any

from ..model_definitions import (
    RuntimeArchitectureSpec,
    runtime_architecture_for_config,
)
from ..runtime_config import normalize_rope_scaling, validate_core_model_config
from .ir import GraphIR, LayerSubgraph, Op


def _resolve_flags(
    model_config: dict[str, Any],
    runtime: RuntimeArchitectureSpec | None,
) -> tuple[bool, bool, bool]:
    """Resolve (apply_qk_norm, uses_shifted_logits, qkv_bias_present) from config.

    Product callers pass a resolved RuntimeArchitectureSpec; its values are
    authoritative and any repeated config flag may only agree. The config-only
    fallback exists for pre-U0 fixtures and low-level callers.

    The architecture is consulted LAZILY — only when at least one flag is not already
    pinned in the config. So a config that pins all three flags as DATA builds a
    graph WITHOUT needing a recognizable architecture (the forward-only callers
    that hand the MLX backend a minimal config rely on this); a config missing any
    flag selects the architecture exactly as before (and raises the same
    ``unsupported_architecture_family`` if the arch is genuinely unknown).
    """
    have_qk = "uses_qk_norm" in model_config
    have_shift = "uses_shifted_logits" in model_config
    have_bias = "bias_present" in model_config

    if runtime is not None:
        expected = {
            "uses_qk_norm": runtime.uses_qk_norm,
            "uses_shifted_logits": runtime.uses_shifted_logits,
            "bias_present": runtime.qkv_bias_present,
        }
        for key, value in expected.items():
            if key in model_config and model_config[key] is not value:
                raise ValueError(f"model config {key} contradicts ModelDefinition")
        return (
            runtime.uses_qk_norm,
            runtime.uses_shifted_logits,
            runtime.qkv_bias_present,
        )

    architecture = None
    if not (have_qk and have_shift and have_bias):
        # Compatibility for direct pre-U0 callers. Product conversion/loading
        # passes the already resolved ModelDefinition runtime explicitly.
        architecture = architecture or runtime_architecture_for_config(model_config)

    apply_qk_norm = (
        model_config["uses_qk_norm"] if have_qk else architecture.uses_qk_norm
    )
    uses_shifted_logits = (
        model_config["uses_shifted_logits"]
        if have_shift
        else architecture.uses_shifted_logits
    )
    if have_bias:
        qkv_bias_present = model_config["bias_present"]
    else:
        qkv_bias_present = architecture.qkv_bias_present
    return apply_qk_norm, uses_shifted_logits, qkv_bias_present


def _layer_subgraph(
    *,
    layer_index: int,
    n_heads: int,
    num_kv_heads: int,
    head_dim: int,
    kv_repeat: int,
    eps: float,
    rope_theta: float,
    rope_scaling: Any,
    apply_qk_norm: bool,
    qkv_bias_present: bool,
    mask_kind: str,
) -> LayerSubgraph:
    """One layer's ordered ops, with concrete per-index weight keys + DATA params."""
    p = f"model.layers.{layer_index}"

    def linear_op(role: str, *, bias_present: bool) -> Op:
        # The attention-projection ``role`` ("q"|"k"|"v"|"o") is carried as op DATA
        # so the interpreter dispatches each linear to its slot by identity, not by
        # position (role-robust: a reordered-but-same-type graph routes correctly or
        # raises, never silently mis-assigns). It also matches the weight key.
        weight_keys = {"weight": f"{p}.self_attn.{role}_proj.weight"}
        if bias_present:
            weight_keys["bias"] = f"{p}.self_attn.{role}_proj.bias"
        return Op(
            type="linear",
            params={"role": role, "bias_present": bias_present},
            weight_keys=weight_keys,
        )

    def qk_norm_op(role: str) -> Op:
        # Op is ALWAYS present in the order; ``apply_qk_norm`` data decides whether
        # the interpreter applies it. The weight key is carried only when applied
        # (Dream has no q_norm/k_norm tensors). ``role`` ("q"|"k") routes by DATA.
        weight_keys = (
            {"weight": f"{p}.self_attn.{role}_norm.weight"} if apply_qk_norm else {}
        )
        return Op(
            type="qk_norm",
            params={
                "role": role,
                "apply_qk_norm": apply_qk_norm,
                "eps": eps,
                "head_dim": head_dim,
            },
            weight_keys=weight_keys,
        )

    def rope_op(role: str) -> Op:
        # The ``role`` ("q"|"k") makes the interpreter route this rope to the right
        # slot by op DATA, not by its position in the layer (role-robust dispatch,
        # so a reordered-but-same-type graph cannot be silently mis-run).
        return Op(
            type="rope",
            params={
                "role": role,
                "head_dim": head_dim,
                "rope_theta": rope_theta,
                "rope_scaling": rope_scaling,
            },
        )

    ops = (
        # rms_norm(input)
        Op(
            type="rms_norm",
            params={"eps": eps},
            weight_keys={"weight": f"{p}.input_layernorm.weight"},
        ),
        # linear q (+bias?)
        linear_op("q", bias_present=qkv_bias_present),
        # qk_norm? (q)
        qk_norm_op("q"),
        # rope q
        rope_op("q"),
        # linear k (+bias?)
        linear_op("k", bias_present=qkv_bias_present),
        # qk_norm? (k)
        qk_norm_op("k"),
        # rope k
        rope_op("k"),
        # linear v (+bias?)
        linear_op("v", bias_present=qkv_bias_present),
        # gqa_repeat (no-op when kv_repeat == 1)
        Op(
            type="gqa_repeat",
            params={
                "kv_repeat": kv_repeat,
                "num_kv_heads": num_kv_heads,
                "n_heads": n_heads,
            },
        ),
        # attention(mask_kind)
        Op(
            type="attention",
            params={"mask_kind": mask_kind, "head_dim": head_dim, "n_heads": n_heads},
        ),
        # linear o (NO bias for the four shipped families)
        Op(
            type="linear",
            params={"role": "o", "bias_present": False},
            weight_keys={"weight": f"{p}.self_attn.o_proj.weight"},
        ),
        # residual_add (+ attn)
        Op(type="residual_add"),
        # rms_norm(post_attn)
        Op(
            type="rms_norm",
            params={"eps": eps},
            weight_keys={"weight": f"{p}.post_attention_layernorm.weight"},
        ),
        # mlp_swiglu(gate, up, down)
        Op(
            type="mlp_swiglu",
            params={},
            weight_keys={
                "gate": f"{p}.mlp.gate_proj.weight",
                "up": f"{p}.mlp.up_proj.weight",
                "down": f"{p}.mlp.down_proj.weight",
            },
        ),
        # residual_add (+ mlp)
        Op(type="residual_add"),
    )
    return LayerSubgraph(ops=ops)


def build_graph_ir(
    model_config: dict[str, Any],
    *,
    runtime: RuntimeArchitectureSpec | None = None,
) -> GraphIR:
    """Emit a :class:`GraphIR` for an enriched ``model_config``.

    The config must already be enriched (``head_dim`` / ``num_kv_heads`` /
    ``rope_theta`` / ``rms_norm_eps`` present) — i.e. what
    ``_enriched_model_config`` produces at load. The emitter does no defaulting of
    geometry; it only translates config DATA into the op-sequence.
    """
    validate_core_model_config(model_config)
    layer_count = int(model_config["layer_count"])
    n_heads = int(model_config["attention_heads"])
    num_kv_heads = int(model_config["num_kv_heads"])
    head_dim = int(model_config["head_dim"])
    if n_heads % num_kv_heads != 0:
        raise ValueError("attention_heads must be divisible by num_kv_heads")
    kv_repeat = max(1, n_heads // max(1, num_kv_heads))
    eps = float(model_config["rms_norm_eps"])
    rope_theta = float(model_config["rope_theta"])
    rope_scaling = normalize_rope_scaling(model_config.get("rope_scaling"))
    mask_kind = str(model_config.get("attention_mask_kind", "bidirectional"))

    apply_qk_norm, uses_shifted_logits, qkv_bias_present = _resolve_flags(
        model_config,
        runtime,
    )
    logits_convention = "dream_ar_shift" if uses_shifted_logits else "raw"

    embedding = Op(
        type="embedding",
        params={},
        weight_keys={"embedding": "model.embed_tokens.weight"},
    )
    layers = tuple(
        _layer_subgraph(
            layer_index=i,
            n_heads=n_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            kv_repeat=kv_repeat,
            eps=eps,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            apply_qk_norm=apply_qk_norm,
            qkv_bias_present=qkv_bias_present,
            mask_kind=mask_kind,
        )
        for i in range(layer_count)
    )
    final_norm = Op(
        type="rms_norm",
        params={"eps": eps},
        weight_keys={"weight": "model.norm.weight"},
    )
    lm_head = Op(
        type="lm_head",
        params={},
        weight_keys={"weight": "lm_head.weight"},
    )
    return GraphIR(
        embedding=embedding,
        layers=layers,
        final_norm=final_norm,
        lm_head=lm_head,
        logits_convention=logits_convention,
    )
