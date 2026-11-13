"""Pure-Python contracts for exact K/V slice data-plane handoff."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import operator
from typing import Any, Iterable


class ExactKVSliceMode(str, Enum):
    """Exact K/V storage mode values, including reserved historical values."""

    OFF = "off"
    PAGED_CONCAT = "paged_concat"
    PAGED_SLICED = "paged_sliced"
    AUTO = "auto"


def resolve_exact_kv_slice_mode(value: Any) -> ExactKVSliceMode:
    """Accept only the exact ``off`` compatibility tombstone."""
    try:
        mode = value if isinstance(value, ExactKVSliceMode) else ExactKVSliceMode(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"exact_kv_slice_mode={value!r} is no longer supported in "
            "DiffusorRT 0.2; use 'off'"
        ) from exc
    if mode is not ExactKVSliceMode.OFF:
        raise ValueError(
            f"exact_kv_slice_mode={mode.value!r} is no longer supported in "
            "DiffusorRT 0.2; use 'off'"
        )
    return ExactKVSliceMode.OFF


def _shape_of(value: Any, name: str) -> tuple[int, ...]:
    try:
        raw_shape = tuple(value.shape)
        if not raw_shape:
            raise ValueError("empty shape")
        if any(isinstance(dimension, bool) for dimension in raw_shape):
            raise ValueError("boolean dimension")
        shape = tuple(operator.index(dimension) for dimension in raw_shape)
    except (AttributeError, IndexError, TypeError, ValueError, OverflowError, RuntimeError) as exc:
        raise ValueError(f"{name} must expose a non-empty shape") from exc
    if any(dimension < 0 for dimension in shape):
        raise ValueError(f"{name} shape dimensions must be non-negative integers")
    return shape


@dataclass(frozen=True, slots=True)
class KVSlice:
    """An immutable exact row span and its K/V payload references."""

    start: int
    end: int
    key: Any
    value: Any

    def __post_init__(self) -> None:
        if type(self.start) is not int or self.start < 0:
            raise ValueError("start must be non-negative")
        if type(self.end) is not int or self.start >= self.end:
            raise ValueError("start must be less than end")

        key_shape = _shape_of(self.key, "key")
        value_shape = _shape_of(self.value, "value")
        span = self.end - self.start
        if key_shape[0] != span:
            raise ValueError("key row count must match span")
        if key_shape[0] != value_shape[0]:
            raise ValueError("key/value row count mismatch")
        if value_shape[0] != span:
            raise ValueError("value row count must match span")


@dataclass(frozen=True, slots=True)
class KVSliceBundle:
    """An immutable, gap-free partition of ``[0, expected_rows)``."""

    slices: tuple[KVSlice, ...]
    expected_rows: int

    def __init__(self, slices: Iterable[KVSlice], expected_rows: int) -> None:
        try:
            normalized_slices = tuple(slices)
        except TypeError as exc:
            raise ValueError("slices must be non-empty") from exc
        object.__setattr__(self, "slices", normalized_slices)
        object.__setattr__(self, "expected_rows", expected_rows)
        self._validate()

    def _validate(self) -> None:
        if not self.slices:
            raise ValueError("slices must be non-empty")
        if type(self.expected_rows) is not int or self.expected_rows <= 0:
            raise ValueError("expected_rows must be positive")

        cursor = 0
        for item in self.slices:
            if not isinstance(item, KVSlice):
                raise ValueError("slices must contain KVSlice instances")
            if item.start != cursor:
                if item.start < cursor:
                    raise ValueError("slices must be strictly ordered with no overlap")
                raise ValueError("slices must have no gap and be strictly ordered")
            cursor = item.end
        if cursor != self.expected_rows:
            raise ValueError("slices must cover exactly [0:expected_rows)")
