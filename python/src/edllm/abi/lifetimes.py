"""Configuration split by lifetime.

EngineConfig changes once per process, ModelGeometry once per model,
SessionDecodeConfig once per request. Per-step data lives in ForwardRequest and
carries no policy knobs at all. Research-tier tuning parameters (gamma, the d2f
thresholds, the recovery knobs) deliberately do NOT cross the C ABI: the engine
receives a policy HANDLE, so tuning never changes the contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .enums import (
    ATTENTION_MASK_KIND_BY_NAME,
    COMMIT_POLICY_BY_NAME,
    COMPUTE_DTYPE_BY_NAME,
    DECODE_STRATEGY_BY_NAME,
    WindowExec,
    encode,
)

UNSET = -1

_GEOMETRY_KEYS = (
    "layer_count", "hidden_size", "num_heads", "num_kv_heads",
    "vocab_size", "max_position_embeddings",
)


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """Process lifetime. Numerics and cache topology."""
    compute_dtype: int
    linear_quant_bits: int
    linear_quant_group_size: int
    layer0_kv_cache_enabled: int
    prefix_kv_cache: int
    block_kv_cache: int
    d2f_exact_kv_cache: int
    cross_request_cache: int
    canvas_cache: int
    canvas_cache_budget_bytes: int
    window_exec: int


@dataclass(frozen=True, slots=True)
class ModelGeometry:
    """Model lifetime. From the package/GraphIR, never defaulted."""
    layer_count: int
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    vocab_size: int
    max_position_embeddings: int


@dataclass(frozen=True, slots=True)
class SessionDecodeConfig:
    """Request lifetime. Structural decode shape only — not tuning."""
    decode_strategy: int
    commit_policy: int
    attention_mask_kind: int
    block_size: int
    steps: int
    max_tokens: int
    trace_canvas: int
    error_fixing: int
    reserved: tuple[int, ...] = field(default=(0,) * 8)


def split_resolved(resolved, *, geometry: dict, max_tokens: int = 0):
    """Split a ResolvedMLXRuntimeConfig into the three ABI lifetimes.

    ``geometry`` must supply every key in ``_GEOMETRY_KEYS``; a defaulted
    geometry field is a silently wrong model, so a missing key raises.
    """
    missing = [key for key in _GEOMETRY_KEYS if key not in geometry]
    if missing:
        raise KeyError(
            f"geometry is missing {missing}; model geometry must come from the "
            "package/GraphIR and is never defaulted"
        )
    engine = EngineConfig(
        compute_dtype=encode(
            COMPUTE_DTYPE_BY_NAME, resolved.compute_dtype, "compute_dtype"),
        linear_quant_bits=(
            UNSET if resolved.linear_quant_bits is None
            else int(resolved.linear_quant_bits)),
        linear_quant_group_size=int(resolved.linear_quant_group_size),
        layer0_kv_cache_enabled=int(resolved.layer0_kv_cache_enabled),
        prefix_kv_cache=int(resolved.prefix_kv_cache),
        block_kv_cache=int(resolved.block_kv_cache),
        d2f_exact_kv_cache=int(resolved.d2f_exact_kv_cache),
        cross_request_cache=int(resolved.cross_request_cache),
        canvas_cache=int(resolved.canvas_cache),
        canvas_cache_budget_bytes=(
            UNSET if resolved.canvas_cache_budget_bytes is None
            else int(resolved.canvas_cache_budget_bytes)),
        window_exec=int(
            WindowExec.AUTO if resolved.window_exec is None
            else (WindowExec.ON if resolved.window_exec else WindowExec.OFF)),
    )
    model = ModelGeometry(**{key: int(geometry[key]) for key in _GEOMETRY_KEYS})
    session = SessionDecodeConfig(
        decode_strategy=encode(
            DECODE_STRATEGY_BY_NAME, resolved.decode_strategy, "decode_strategy"),
        commit_policy=encode(
            COMMIT_POLICY_BY_NAME, resolved.d2f_commit_policy, "d2f_commit_policy"),
        attention_mask_kind=encode(
            ATTENTION_MASK_KIND_BY_NAME, resolved.attention_mask_kind,
            "attention_mask_kind"),
        block_size=int(resolved.block_size),
        steps=int(resolved.steps),
        max_tokens=int(max_tokens),
        trace_canvas=int(resolved.trace_canvas),
        error_fixing=int(resolved.error_fixing),
    )
    return engine, model, session
