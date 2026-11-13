from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class RefreshSchedule:
    interval_blocks: int = 2

    def should_refresh(self, block_index: int) -> bool:
        if self.interval_blocks <= 0:
            return False
        return block_index > 0 and block_index % self.interval_blocks == 0


@dataclass(slots=True)
class KVCacheEvent:
    reused_positions: int = 0


@dataclass(slots=True)
class _LayerKVState:
    token_ids: tuple[int, ...]
    key: np.ndarray
    value: np.ndarray
    reusable_positions: tuple[bool, ...]


class DelayedKVCache:
    """Layer-local K/V cache for unchanged decoded positions.

    This is deliberately narrower than an autoregressive KV cache. It only
    stores K/V entries that the caller marks reusable and only reuses them when
    the absolute position and token id still match.
    """

    def __init__(self) -> None:
        self._layers: dict[int, _LayerKVState] = {}

    def clear(self) -> None:
        self._layers.clear()

    def has_entries(self) -> bool:
        return bool(self._layers)

    def apply_layer(
        self,
        *,
        layer_index: int,
        token_ids: list[int],
        key: np.ndarray,
        value: np.ndarray,
        reusable_positions: list[bool],
    ) -> tuple[np.ndarray, np.ndarray, KVCacheEvent]:
        state = self._layers.get(layer_index)
        if state is None or len(state.token_ids) != len(token_ids):
            return key, value, KVCacheEvent()

        merged_key = np.array(key, copy=True)
        merged_value = np.array(value, copy=True)
        reused_positions = 0
        for position, is_reusable in enumerate(reusable_positions):
            if (
                is_reusable
                and state.reusable_positions[position]
                and state.token_ids[position] == int(token_ids[position])
            ):
                merged_key[position] = state.key[position]
                merged_value[position] = state.value[position]
                reused_positions += 1
        return merged_key, merged_value, KVCacheEvent(reused_positions=reused_positions)

    def update_layer(
        self,
        *,
        layer_index: int,
        token_ids: list[int],
        key: np.ndarray,
        value: np.ndarray,
        reusable_positions: list[bool],
    ) -> None:
        self._layers[layer_index] = _LayerKVState(
            token_ids=tuple(int(token_id) for token_id in token_ids),
            key=np.array(key, copy=True),
            value=np.array(value, copy=True),
            reusable_positions=tuple(bool(position) for position in reusable_positions),
        )
