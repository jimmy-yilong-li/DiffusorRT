"""Validated lowering from GraphIR v0 to the shared dense backend contract.

The optimized backends are intentionally not graph interpreters. They implement
one dense transformer topology, so this module validates that topology once and
materializes every structural value the implementation consumes. A backend may
then use its fixed control flow without re-inferring model semantics from a
model name, a package config, or the weight archive.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from ..runtime_config import normalize_rope_scaling
from .ir import CANONICAL_LAYER_OP_ORDER, IR_VERSION, GraphIR, Op


@dataclass(frozen=True, slots=True)
class DenseLayerPlan:
    index: int
    input_norm_weight: str
    q_weight: str
    q_bias: str | None
    q_norm_weight: str | None
    k_weight: str
    k_bias: str | None
    k_norm_weight: str | None
    v_weight: str
    v_bias: str | None
    o_weight: str
    post_attention_norm_weight: str
    gate_weight: str
    up_weight: str
    down_weight: str


@dataclass(frozen=True, slots=True)
class DenseTransformerPlan:
    """Immutable backend view compiled from one exact GraphIR."""

    graph: GraphIR
    layers: tuple[DenseLayerPlan, ...]
    embedding_weight: str
    final_norm_weight: str
    lm_head_weight: str
    n_heads: int
    num_kv_heads: int
    head_dim: int
    kv_repeat: int
    eps: float
    rope_theta: float
    rope_scaling: Any
    apply_qk_norm: bool
    attention_mask_kind: str
    logits_convention: str


def _require_type(op: Op, expected: str, *, location: str) -> None:
    if op.type != expected:
        raise ValueError(
            f"backend_graph_topology_mismatch:{location}:"
            f"expected={expected}:actual={op.type}"
        )


def _require_role(op: Op, expected: str, *, location: str) -> None:
    if op.params.get("role") != expected:
        raise ValueError(f"backend_graph_role_mismatch:{location}:expected={expected}")


def _required_weight(op: Op, role: str, *, location: str) -> str:
    value = op.weight_keys.get(role)
    if type(value) is not str or not value:
        raise ValueError(f"backend_graph_weight_missing:{location}:{role}")
    return value


def _require_weight_roles(op: Op, expected: set[str], *, location: str) -> None:
    actual = set(op.weight_keys)
    if actual != expected:
        raise ValueError(
            f"backend_graph_weight_roles_unsupported:{location}:"
            f"expected={sorted(expected)!r}:actual={sorted(actual)!r}"
        )


def _optional_bias(op: Op, *, location: str) -> str | None:
    present = op.params.get("bias_present")
    if type(present) is not bool:
        raise ValueError(f"backend_graph_bias_flag_invalid:{location}")
    value = op.weight_keys.get("bias")
    if present:
        if type(value) is not str or not value:
            raise ValueError(f"backend_graph_bias_missing:{location}")
        return value
    if value is not None:
        raise ValueError(f"backend_graph_bias_unexpected:{location}")
    return None


def _positive_int(op: Op, name: str, *, location: str) -> int:
    value = op.params.get(name)
    if type(value) is not int or value <= 0:
        raise ValueError(f"backend_graph_parameter_invalid:{location}:{name}")
    return value


def _positive_float(op: Op, name: str, *, location: str) -> float:
    value = op.params.get(name)
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"backend_graph_parameter_invalid:{location}:{name}")
    return float(value)


def _expected_layer_weights(
    index: int, *, qkv_bias: bool, qk_norm: bool
) -> dict[str, str | None]:
    prefix = f"model.layers.{index}"
    return {
        "input_norm": f"{prefix}.input_layernorm.weight",
        "q": f"{prefix}.self_attn.q_proj.weight",
        "q_bias": f"{prefix}.self_attn.q_proj.bias" if qkv_bias else None,
        "q_norm": f"{prefix}.self_attn.q_norm.weight" if qk_norm else None,
        "k": f"{prefix}.self_attn.k_proj.weight",
        "k_bias": f"{prefix}.self_attn.k_proj.bias" if qkv_bias else None,
        "k_norm": f"{prefix}.self_attn.k_norm.weight" if qk_norm else None,
        "v": f"{prefix}.self_attn.v_proj.weight",
        "v_bias": f"{prefix}.self_attn.v_proj.bias" if qkv_bias else None,
        "o": f"{prefix}.self_attn.o_proj.weight",
        "post_norm": f"{prefix}.post_attention_layernorm.weight",
        "gate": f"{prefix}.mlp.gate_proj.weight",
        "up": f"{prefix}.mlp.up_proj.weight",
        "down": f"{prefix}.mlp.down_proj.weight",
    }


def compile_dense_transformer_plan(graph: GraphIR) -> DenseTransformerPlan:
    """Validate and lower the GraphIR topology implemented by dense backends."""

    ir_major = graph.ir_version.get("major")
    if type(ir_major) is not int or ir_major != IR_VERSION["major"]:
        raise ValueError("backend_graph_ir_major_unsupported")
    _require_type(graph.embedding, "embedding", location="embedding")
    _require_type(graph.final_norm, "rms_norm", location="final_norm")
    _require_type(graph.lm_head, "lm_head", location="lm_head")
    _require_weight_roles(graph.embedding, {"embedding"}, location="embedding")
    _require_weight_roles(graph.final_norm, {"weight"}, location="final_norm")
    _require_weight_roles(graph.lm_head, {"weight"}, location="lm_head")
    if not graph.layers:
        raise ValueError("backend_graph_has_no_layers")

    embedding_weight = _required_weight(
        graph.embedding, "embedding", location="embedding"
    )
    final_norm_weight = _required_weight(
        graph.final_norm, "weight", location="final_norm"
    )
    lm_head_weight = _required_weight(graph.lm_head, "weight", location="lm_head")
    if (
        embedding_weight != "model.embed_tokens.weight"
        or final_norm_weight != "model.norm.weight"
        or lm_head_weight != "lm_head.weight"
    ):
        raise ValueError("backend_graph_top_level_weight_layout_unsupported")

    compiled_layers: list[DenseLayerPlan] = []
    signature: tuple[object, ...] | None = None
    for index, layer in enumerate(graph.layers):
        actual_order = tuple(op.type for op in layer.ops)
        if actual_order != CANONICAL_LAYER_OP_ORDER:
            raise ValueError(
                "backend_graph_layer_order_mismatch:"
                f"layer={index}:actual={actual_order!r}"
            )
        (
            input_norm,
            q_linear,
            q_norm,
            q_rope,
            k_linear,
            k_norm,
            k_rope,
            v_linear,
            gqa,
            attention,
            o_linear,
            _attention_residual,
            post_norm,
            mlp,
            _mlp_residual,
        ) = layer.ops
        _require_weight_roles(
            input_norm,
            {"weight"},
            location=f"layer={index}:input_norm",
        )
        _require_weight_roles(
            post_norm,
            {"weight"},
            location=f"layer={index}:post_norm",
        )
        _require_weight_roles(
            mlp,
            {"gate", "up", "down"},
            location=f"layer={index}:mlp",
        )
        for op, location in (
            (q_rope, "q_rope"),
            (k_rope, "k_rope"),
            (gqa, "gqa"),
            (attention, "attention"),
            (_attention_residual, "attention_residual"),
            (_mlp_residual, "mlp_residual"),
        ):
            _require_weight_roles(op, set(), location=f"layer={index}:{location}")
        for op, role, location in (
            (q_linear, "q", "q_linear"),
            (q_norm, "q", "q_norm"),
            (q_rope, "q", "q_rope"),
            (k_linear, "k", "k_linear"),
            (k_norm, "k", "k_norm"),
            (k_rope, "k", "k_rope"),
            (v_linear, "v", "v_linear"),
            (o_linear, "o", "o_linear"),
        ):
            _require_role(op, role, location=f"layer={index}:{location}")

        eps = _positive_float(input_norm, "eps", location=f"layer={index}:input_norm")
        if (
            _positive_float(post_norm, "eps", location=f"layer={index}:post_norm")
            != eps
        ):
            raise ValueError(f"backend_graph_norm_eps_nonuniform:layer={index}")
        apply_qk_norm = q_norm.params.get("apply_qk_norm")
        if (
            type(apply_qk_norm) is not bool
            or k_norm.params.get("apply_qk_norm") is not apply_qk_norm
        ):
            raise ValueError(f"backend_graph_qk_norm_nonuniform:layer={index}")
        q_norm_weight = (
            _required_weight(q_norm, "weight", location=f"layer={index}:q_norm")
            if apply_qk_norm
            else None
        )
        k_norm_weight = (
            _required_weight(k_norm, "weight", location=f"layer={index}:k_norm")
            if apply_qk_norm
            else None
        )
        if not apply_qk_norm and (q_norm.weight_keys or k_norm.weight_keys):
            raise ValueError(f"backend_graph_qk_norm_weight_unexpected:layer={index}")

        n_heads = _positive_int(
            attention, "n_heads", location=f"layer={index}:attention"
        )
        head_dim = _positive_int(
            attention, "head_dim", location=f"layer={index}:attention"
        )
        num_kv_heads = _positive_int(gqa, "num_kv_heads", location=f"layer={index}:gqa")
        kv_repeat = _positive_int(gqa, "kv_repeat", location=f"layer={index}:gqa")
        if _positive_int(gqa, "n_heads", location=f"layer={index}:gqa") != n_heads:
            raise ValueError(f"backend_graph_head_count_nonuniform:layer={index}")
        if n_heads % num_kv_heads or kv_repeat != n_heads // num_kv_heads:
            raise ValueError(f"backend_graph_gqa_invalid:layer={index}")
        rope_theta = _positive_float(
            q_rope, "rope_theta", location=f"layer={index}:q_rope"
        )
        rope_scaling = normalize_rope_scaling(q_rope.params.get("rope_scaling"))
        k_rope_scaling = normalize_rope_scaling(k_rope.params.get("rope_scaling"))
        if (
            _positive_int(q_rope, "head_dim", location=f"layer={index}:q_rope")
            != head_dim
            or k_rope.params.get("head_dim") != head_dim
            or k_rope.params.get("rope_theta") != q_rope.params.get("rope_theta")
            or k_rope_scaling != rope_scaling
            or q_norm.params.get("head_dim") != head_dim
            or k_norm.params.get("head_dim") != head_dim
            or q_norm.params.get("eps") != input_norm.params.get("eps")
            or k_norm.params.get("eps") != input_norm.params.get("eps")
        ):
            raise ValueError(
                f"backend_graph_attention_geometry_nonuniform:layer={index}"
            )
        mask_kind = attention.params.get("mask_kind")
        if type(mask_kind) is not str or not mask_kind:
            raise ValueError(f"backend_graph_mask_kind_invalid:layer={index}")

        q_bias = _optional_bias(q_linear, location=f"layer={index}:q_linear")
        k_bias = _optional_bias(k_linear, location=f"layer={index}:k_linear")
        v_bias = _optional_bias(v_linear, location=f"layer={index}:v_linear")
        _require_weight_roles(
            q_linear,
            {"weight", "bias"} if q_bias is not None else {"weight"},
            location=f"layer={index}:q_linear",
        )
        _require_weight_roles(
            k_linear,
            {"weight", "bias"} if k_bias is not None else {"weight"},
            location=f"layer={index}:k_linear",
        )
        _require_weight_roles(
            v_linear,
            {"weight", "bias"} if v_bias is not None else {"weight"},
            location=f"layer={index}:v_linear",
        )
        qkv_bias = q_bias is not None
        if (k_bias is not None) is not qkv_bias or (v_bias is not None) is not qkv_bias:
            raise ValueError(f"backend_graph_qkv_bias_nonuniform:layer={index}")
        if _optional_bias(o_linear, location=f"layer={index}:o_linear") is not None:
            raise ValueError(f"backend_graph_o_bias_unsupported:layer={index}")
        _require_weight_roles(
            o_linear,
            {"weight"},
            location=f"layer={index}:o_linear",
        )
        _require_weight_roles(
            q_norm,
            {"weight"} if apply_qk_norm else set(),
            location=f"layer={index}:q_norm",
        )
        _require_weight_roles(
            k_norm,
            {"weight"} if apply_qk_norm else set(),
            location=f"layer={index}:k_norm",
        )

        layer_plan = DenseLayerPlan(
            index=index,
            input_norm_weight=_required_weight(
                input_norm, "weight", location=f"layer={index}:input_norm"
            ),
            q_weight=_required_weight(
                q_linear, "weight", location=f"layer={index}:q_linear"
            ),
            q_bias=q_bias,
            q_norm_weight=q_norm_weight,
            k_weight=_required_weight(
                k_linear, "weight", location=f"layer={index}:k_linear"
            ),
            k_bias=k_bias,
            k_norm_weight=k_norm_weight,
            v_weight=_required_weight(
                v_linear, "weight", location=f"layer={index}:v_linear"
            ),
            v_bias=v_bias,
            o_weight=_required_weight(
                o_linear, "weight", location=f"layer={index}:o_linear"
            ),
            post_attention_norm_weight=_required_weight(
                post_norm, "weight", location=f"layer={index}:post_norm"
            ),
            gate_weight=_required_weight(mlp, "gate", location=f"layer={index}:mlp"),
            up_weight=_required_weight(mlp, "up", location=f"layer={index}:mlp"),
            down_weight=_required_weight(mlp, "down", location=f"layer={index}:mlp"),
        )
        expected_weights = _expected_layer_weights(
            index,
            qkv_bias=qkv_bias,
            qk_norm=apply_qk_norm,
        )
        actual_weights = {
            "input_norm": layer_plan.input_norm_weight,
            "q": layer_plan.q_weight,
            "q_bias": layer_plan.q_bias,
            "q_norm": layer_plan.q_norm_weight,
            "k": layer_plan.k_weight,
            "k_bias": layer_plan.k_bias,
            "k_norm": layer_plan.k_norm_weight,
            "v": layer_plan.v_weight,
            "v_bias": layer_plan.v_bias,
            "o": layer_plan.o_weight,
            "post_norm": layer_plan.post_attention_norm_weight,
            "gate": layer_plan.gate_weight,
            "up": layer_plan.up_weight,
            "down": layer_plan.down_weight,
        }
        if actual_weights != expected_weights:
            raise ValueError(
                f"backend_graph_layer_weight_layout_unsupported:layer={index}"
            )

        layer_signature = (
            n_heads,
            num_kv_heads,
            head_dim,
            kv_repeat,
            eps,
            rope_theta,
            rope_scaling,
            apply_qk_norm,
            mask_kind,
            qkv_bias,
        )
        if signature is None:
            signature = layer_signature
        elif layer_signature != signature:
            raise ValueError(f"backend_graph_layer_contract_nonuniform:layer={index}")
        compiled_layers.append(layer_plan)

    if signature is None:
        raise ValueError("backend_graph_has_no_layers")
    (
        n_heads,
        num_kv_heads,
        head_dim,
        kv_repeat,
        eps,
        rope_theta,
        rope_scaling,
        apply_qk_norm,
        mask_kind,
        _qkv_bias,
    ) = signature
    if _positive_float(graph.final_norm, "eps", location="final_norm") != eps:
        raise ValueError("backend_graph_final_norm_eps_mismatch")
    return DenseTransformerPlan(
        graph=graph,
        layers=tuple(compiled_layers),
        embedding_weight=embedding_weight,
        final_norm_weight=final_norm_weight,
        lm_head_weight=lm_head_weight,
        n_heads=int(n_heads),
        num_kv_heads=int(num_kv_heads),
        head_dim=int(head_dim),
        kv_repeat=int(kv_repeat),
        eps=float(eps),
        rope_theta=float(rope_theta),
        rope_scaling=rope_scaling,
        apply_qk_norm=bool(apply_qk_norm),
        attention_mask_kind=str(mask_kind),
        logits_convention=graph.logits_convention,
    )


__all__ = [
    "DenseLayerPlan",
    "DenseTransformerPlan",
    "compile_dense_transformer_plan",
]
