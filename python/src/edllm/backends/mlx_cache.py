"""Pure policy/state helpers for MLX KV cache evidence.

This module is intentionally independent from `mlx.core`. It defines the
layer-0 cache contract and metrics; callers decide where tensor storage lives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_METRIC_KEYS = (
    "mlx_kv_cache_enabled",
    "mlx_kv_cache_strategy",
    "mlx_kv_layer0_entries",
    "mlx_kv_layer0_projected_rows",
    "mlx_kv_layer0_skipped_projection_rows",
    "mlx_kv_reuse_count",
    "mlx_kv_refresh_count",
    "mlx_kv_invalidated_count",
    "mlx_kv_upper_layer_reuse_count",
)


@dataclass(frozen=True, slots=True)
class MLXKVCacheEntry:
    strategy: str
    layer_index: int
    position: int
    token_id: int
    attention_mask_kind: str
    rope_signature: Any
    head_dim: int
    num_kv_heads: int
    dtype: str
    input_fingerprint: str | None = None


@dataclass(frozen=True, slots=True)
class MLXKVCachePolicy:
    """Stateless cache reuse policy.

    `can_reuse(...)` checks whether a stored entry is mathematically equivalent
    for the requested token/position. Use `new_state()` as the canonical
    constructor for mutable cache state that records reuse/invalidation counts.
    """

    strategy: str
    attention_mask_kind: str
    rope_signature: Any
    head_dim: int
    num_kv_heads: int
    dtype: str

    def make_entry(
        self,
        *,
        layer_index: int,
        position: int,
        token_id: int,
        input_fingerprint: str | None = None,
    ) -> MLXKVCacheEntry:
        return MLXKVCacheEntry(
            strategy=self.strategy,
            layer_index=int(layer_index),
            position=int(position),
            token_id=int(token_id),
            attention_mask_kind=self.attention_mask_kind,
            rope_signature=self.rope_signature,
            head_dim=int(self.head_dim),
            num_kv_heads=int(self.num_kv_heads),
            dtype=str(self.dtype),
            input_fingerprint=input_fingerprint,
        )

    def can_reuse(
        self,
        entry: MLXKVCacheEntry,
        *,
        layer_index: int,
        position: int,
        token_id: int,
        input_fingerprint: str | None = None,
    ) -> bool:
        if self.strategy != "bidirectional_layer0_kv":
            return False
        if int(layer_index) != 0 or entry.layer_index != 0:
            return False
        if entry.position != int(position) or entry.token_id != int(token_id):
            return False
        if entry.attention_mask_kind != self.attention_mask_kind:
            return False
        if entry.rope_signature != self.rope_signature:
            return False
        if entry.head_dim != int(self.head_dim):
            return False
        if entry.num_kv_heads != int(self.num_kv_heads):
            return False
        if entry.dtype != str(self.dtype):
            return False
        return entry.input_fingerprint == input_fingerprint

    def new_state(self) -> MLXKVCacheState:
        return MLXKVCacheState(policy=self)


class MLXKVCacheState:
    """Mutable per-cache state owned by a caller-created policy.

    `can_reuse(...)` is the stateful read-side check and increments reuse
    evidence on a hit. `observe_commit(...)` invalidates mismatched token entries
    once; a second observation is a no-op until a new entry is stored.
    """

    def __init__(self, *, policy: MLXKVCachePolicy) -> None:
        self._policy = policy
        self._entries: dict[tuple[int, int], tuple[MLXKVCacheEntry, Any, Any]] = {}
        self._layer0_projected_rows = 0
        self._layer0_skipped_projection_rows = 0
        self._reuse_count = 0
        self._refresh_count = 0
        self._invalidated_count = 0
        self._upper_layer_reuse_count = 0

    def store(self, entry: MLXKVCacheEntry, *, key_tensor: Any, value_tensor: Any) -> None:
        self._entries[(entry.layer_index, entry.position)] = (entry, key_tensor, value_tensor)

    def store_entry(
        self,
        *,
        layer_index: int,
        position: int,
        token_id: int,
        key_tensor: Any,
        value_tensor: Any,
        input_fingerprint: str | None = None,
    ) -> None:
        self.store(
            self._policy.make_entry(
                layer_index=layer_index,
                position=position,
                token_id=token_id,
                input_fingerprint=input_fingerprint,
            ),
            key_tensor=key_tensor,
            value_tensor=value_tensor,
        )

    def lookup(
        self,
        *,
        layer_index: int,
        position: int,
        token_id: int,
        input_fingerprint: str | None = None,
    ) -> tuple[Any, Any] | None:
        stored = self._entries.get((int(layer_index), int(position)))
        if stored is None:
            return None
        entry, key_tensor, value_tensor = stored
        reusable = self._policy.can_reuse(
            entry,
            layer_index=layer_index,
            position=position,
            token_id=token_id,
            input_fingerprint=input_fingerprint,
        )
        if not reusable:
            return None
        self._reuse_count += 1
        return key_tensor, value_tensor

    def can_reuse(
        self,
        *,
        layer_index: int,
        position: int,
        token_id: int,
        input_fingerprint: str | None = None,
    ) -> bool:
        return self.lookup(
            layer_index=layer_index,
            position=position,
            token_id=token_id,
            input_fingerprint=input_fingerprint,
        ) is not None

    def observe_commit(self, *, position: int, token_id: int) -> None:
        for cache_key, (entry, _key_tensor, _value_tensor) in list(self._entries.items()):
            if entry.position == int(position) and entry.token_id != int(token_id):
                del self._entries[cache_key]
                self._invalidated_count += 1

    def record_layer0_projection_rows(self, *, projected_count: int, skipped_count: int) -> None:
        self._layer0_projected_rows += int(projected_count)
        self._layer0_skipped_projection_rows += int(skipped_count)

    def metrics(self) -> dict[str, int | bool | str]:
        metrics = empty_metrics(strategy=self._policy.strategy)
        metrics["mlx_kv_layer0_entries"] = sum(
            1 for entry, _key, _value in self._entries.values() if entry.layer_index == 0
        )
        metrics["mlx_kv_layer0_projected_rows"] = self._layer0_projected_rows
        metrics["mlx_kv_layer0_skipped_projection_rows"] = self._layer0_skipped_projection_rows
        metrics["mlx_kv_reuse_count"] = self._reuse_count
        metrics["mlx_kv_refresh_count"] = self._refresh_count
        metrics["mlx_kv_invalidated_count"] = self._invalidated_count
        metrics["mlx_kv_upper_layer_reuse_count"] = self._upper_layer_reuse_count
        return metrics


def empty_metrics(*, strategy: str = "none") -> dict[str, int | bool | str]:
    enabled = strategy != "none"
    return {
        "mlx_kv_cache_enabled": enabled,
        "mlx_kv_cache_strategy": strategy,
        "mlx_kv_layer0_entries": 0,
        "mlx_kv_layer0_projected_rows": 0,
        "mlx_kv_layer0_skipped_projection_rows": 0,
        "mlx_kv_reuse_count": 0,
        "mlx_kv_refresh_count": 0,
        "mlx_kv_invalidated_count": 0,
        "mlx_kv_upper_layer_reuse_count": 0,
    }


def derive_refresh_real(
    *,
    forward_count: int,
    quality_passed: bool,
    reuse_count: int,
    refresh_count: int,
    invalidated_count: int,
) -> bool:
    return (
        int(forward_count) > 0
        and bool(quality_passed)
        and int(reuse_count) > 0
        and (int(refresh_count) > 0 or int(invalidated_count) > 0)
    )


__all__ = [
    "MLXKVCacheEntry",
    "MLXKVCachePolicy",
    "MLXKVCacheState",
    "derive_refresh_real",
    "empty_metrics",
]
