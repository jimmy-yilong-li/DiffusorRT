"""Runtime adapter compatibility surface over unified model definitions.

The authoritative architecture data lives in :mod:`edllm.model_definitions` so
the converter, GraphIR emitter, loader, and runtime cannot grow separate model
registries. Named factory functions remain here for source compatibility; all
of them build from the same declarative definitions.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Callable

from ..model_definitions import (
    DECODE_POLICIES,
    ModelAdapter,
    RUNTIME_ARCHITECTURES,
    build_runtime_adapter,
)


def dream_adapter(model_config: dict[str, Any], mask_token_id: int) -> ModelAdapter:
    return RUNTIME_ARCHITECTURES["qwen2_dream"].build(
        model_config,
        mask_token_id,
        decode_policy=DECODE_POLICIES["block_threshold_v1"],
    )


def qwen3_mdlm_adapter(
    model_config: dict[str, Any], mask_token_id: int
) -> ModelAdapter:
    return RUNTIME_ARCHITECTURES["qwen3_mdlm"].build(
        model_config,
        mask_token_id,
        decode_policy=DECODE_POLICIES["block_threshold_v1"],
    )


def qwen3_bd3lm_adapter(
    model_config: dict[str, Any], mask_token_id: int
) -> ModelAdapter:
    return RUNTIME_ARCHITECTURES["qwen3_bd3lm"].build(
        model_config,
        mask_token_id,
        decode_policy=DECODE_POLICIES["bd3_fixed_quota_v1"],
    )


def llada_adapter(model_config: dict[str, Any], mask_token_id: int) -> ModelAdapter:
    return RUNTIME_ARCHITECTURES["llada"].build(
        model_config,
        mask_token_id,
        decode_policy=DECODE_POLICIES["block_threshold_v1"],
    )


AdapterFactory = Callable[[dict[str, Any], int], ModelAdapter]
_FACTORIES: Mapping[str, AdapterFactory] = MappingProxyType({
    "qwen2_dream": dream_adapter,
    "qwen3_mdlm": qwen3_mdlm_adapter,
    "qwen3_bd3lm": qwen3_bd3lm_adapter,
    "llada": llada_adapter,
})

# Compatibility registry generated from the authoritative alias sets. No
# architecture name is repeated here by hand.
ARCHITECTURE_ADAPTERS: Mapping[str, AdapterFactory] = MappingProxyType({
    alias: _FACTORIES[key]
    for key, spec in RUNTIME_ARCHITECTURES.items()
    for alias in spec.aliases
})


def select_adapter(model_config: dict[str, Any], mask_token_id: int) -> ModelAdapter:
    """Resolve exact architecture/family identity and build its runtime policy."""

    return build_runtime_adapter(model_config, mask_token_id)


def select_numpy_executor_adapter(
    model_config: dict[str, Any],
    mask_token_id: int,
) -> ModelAdapter:
    """Select an adapter only when the legacy NumPy executor implements it.

    ``DreamTransformerExecutor`` implements one precise contract: the
    ``qwen2_dream`` forward without per-head Q/K norm and with shifted logits.
    Explicit graph flags override descriptor defaults and must be real booleans.
    """

    adapter = select_adapter(model_config, mask_token_id)

    def effective_flag(name: str, default: bool) -> bool:
        if name not in model_config:
            return default
        value = model_config[name]
        if type(value) is not bool:
            raise ValueError(f"{name} must be a boolean")
        return value

    uses_qk_norm = effective_flag("uses_qk_norm", adapter.uses_qk_norm)
    uses_shifted_logits = effective_flag(
        "uses_shifted_logits", adapter.uses_shifted_logits
    )
    if (
        adapter.architecture_family != "qwen2_dream"
        or uses_qk_norm
        or not uses_shifted_logits
    ):
        architecture = str(
            model_config.get("architecture")
            or model_config.get("architecture_family")
            or "<missing>"
        )
        raise ValueError(
            f"numpy_local does not support real architecture {architecture}"
        )
    return adapter


__all__ = [
    "ARCHITECTURE_ADAPTERS",
    "ModelAdapter",
    "dream_adapter",
    "llada_adapter",
    "qwen3_bd3lm_adapter",
    "qwen3_mdlm_adapter",
    "select_adapter",
    "select_numpy_executor_adapter",
]
