"""Allocation-free sizing for the MLX exact-D2F compact-GQA contract.

These are native-manager admission limits, not a physical peak or a device-fit
test. The forward stores unquantized FP32 K/V at the model's declared K/V-head
count, even with BF16 compute or quantized weights. Capture checks the same spec.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

from .kv_manager import KVTensorSpec

if TYPE_CHECKING:
    from ..graph import DenseTransformerPlan


def _count(name: str, value: object, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def validate_request_bounds(prompt_tokens: int | None, max_tokens: int | None) -> None:
    """Diagnostics may omit both bounds, but must not guess a missing half."""
    if prompt_tokens is None and max_tokens is None:
        return
    if prompt_tokens is None or max_tokens is None:
        raise ValueError("prompt_tokens and max_tokens must be supplied together")
    _count("prompt_tokens", prompt_tokens)
    _count("max_tokens", max_tokens)


@dataclass(frozen=True, slots=True)
class ExactD2FKVCapacity:
    layer_count: int
    tensor_spec: KVTensorSpec
    prompt_tokens: int
    max_tokens: int

    @property
    def shared_cache_budget_bytes(self) -> int:
        return self.layer_count * 2 * self.tensor_spec.nbytes(
            self.prompt_tokens + self.max_tokens
        )

    @property
    def staging_headroom_bytes(self) -> int:
        # A rebuild may capture prompt + several frozen blocks in one forward.
        # max(prompt, block_size) is insufficient; bound the whole canvas.
        return self.shared_cache_budget_bytes

    def to_dict(self) -> dict[str, object]:
        return {
            "storage_layout": "compact_gqa_rows_kv_heads_dim",
            "storage_dtype": self.tensor_spec.dtype,
            "kv_quantization": self.tensor_spec.quantization_identity,
            "layer_count": self.layer_count,
            "stored_heads": self.tensor_spec.kv_heads,
            "head_dim": self.tensor_spec.head_dim,
            "prompt_tokens": self.prompt_tokens,
            "max_tokens": self.max_tokens,
            "total_rows": self.prompt_tokens + self.max_tokens,
            "session_reserved_bytes": 0,
            "shared_cache_budget_bytes": self.shared_cache_budget_bytes,
            "staging_headroom_bytes": self.staging_headroom_bytes,
            "physical_peak_bytes": None,
            "device_memory_fit": None,
            "excluded_from_budget": [
                "weights", "forward_workspaces", "materialization_and_concatenation",
                "allocator_cache_and_cleanup_debt",
            ],
        }


def exact_d2f_kv_capacity(
    model_config: Mapping[str, object] | None = None,
    *,
    prompt_tokens: int,
    max_tokens: int,
    backend_plan: "DenseTransformerPlan | None" = None,
) -> ExactD2FKVCapacity:
    """Size one generation canvas, with the same int64 limit as admission.

    Product execution supplies ``backend_plan`` so allocation uses the exact
    GraphIR lowering already consumed by the forward. ``model_config`` remains
    the SDK-free diagnostics/legacy input and is ignored when a plan is given.
    """
    if backend_plan is not None:
        layer_count = len(backend_plan.layers)
        stored_heads = backend_plan.num_kv_heads
        head_dim = backend_plan.head_dim
    else:
        if model_config is None:
            raise ValueError("model_config or backend_plan is required")
        layer_count = _count(
            "layer_count", model_config.get("layer_count"), minimum=1
        )
        attention_heads = _count(
            "attention_heads", model_config.get("attention_heads"), minimum=1
        )
        stored_heads = _count(
            "num_kv_heads", model_config.get("num_kv_heads"), minimum=1
        )
        if attention_heads % stored_heads != 0:
            raise ValueError("attention_heads must be divisible by num_kv_heads")
        head_dim = _count("head_dim", model_config.get("head_dim"), minimum=1)
    capacity = ExactD2FKVCapacity(
        layer_count=layer_count,
        tensor_spec=KVTensorSpec(
            dtype="float32",
            quantization_identity="none",
            kv_heads=stored_heads,
            head_dim=head_dim,
        ),
        prompt_tokens=_count("prompt_tokens", prompt_tokens),
        max_tokens=_count("max_tokens", max_tokens),
    )
    if capacity.shared_cache_budget_bytes > (1 << 63) - 1:
        raise ValueError(
            "the exact-D2F K/V generation requires "
            f"{capacity.shared_cache_budget_bytes} bytes, which the native int64 "
            "budget cannot represent"
        )
    return capacity
