"""Pure accounting helpers for the synchronized exact K/V concat probe."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import floor, isfinite, prod
from statistics import median
import time
from typing import Any


@dataclass(frozen=True, slots=True)
class KVConcatGeometry:
    """Logical geometry for one axis-0 rank-3 K/V concatenation."""

    part_shapes: tuple[tuple[int, int, int], ...]
    output_shape: tuple[int, int, int]
    dtype: str
    itemsize: int
    logical_input_bytes: int
    logical_output_bytes: int


@dataclass(frozen=True, slots=True)
class KVConcatSample:
    """One measured real-generation sample."""

    repeat: int
    wall_ms: float
    forward_ms: float
    concat_call_ms: tuple[float, ...]
    concat_calls: int
    logical_input_bytes: int
    logical_output_bytes: int
    output_tokens: int
    forward_count: int

    def __post_init__(self) -> None:
        if self.concat_calls != len(self.concat_call_ms):
            raise ValueError("concat_calls must match concat_call_ms length")
        if self.repeat < 1:
            raise ValueError("repeat must be >= 1")
        scalar_times = (self.wall_ms, self.forward_ms, *self.concat_call_ms)
        if any(not isfinite(value) or value < 0.0 for value in scalar_times):
            raise ValueError("timings must be finite and nonnegative")
        counts = (
            self.concat_calls,
            self.logical_input_bytes,
            self.logical_output_bytes,
            self.output_tokens,
            self.forward_count,
        )
        if any(value < 0 for value in counts):
            raise ValueError("counts and logical bytes must be nonnegative")

    @property
    def concat_ms(self) -> float:
        return sum(self.concat_call_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "repeat": self.repeat,
            "wall_ms": self.wall_ms,
            "forward_ms": self.forward_ms,
            "concat_ms": self.concat_ms,
            "concat_calls": self.concat_calls,
            "logical_input_bytes": self.logical_input_bytes,
            "logical_output_bytes": self.logical_output_bytes,
            "output_tokens": self.output_tokens,
            "forward_count": self.forward_count,
        }


def _shape(value: Any) -> tuple[int, ...] | None:
    raw_shape = getattr(value, "shape", None)
    if raw_shape is None:
        return None
    try:
        shape = tuple(int(dimension) for dimension in raw_shape)
    except (TypeError, ValueError, OverflowError):
        return None
    if any(dimension < 0 for dimension in shape):
        return None
    return shape


def _dtype_geometry(value: Any) -> tuple[str, int] | None:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return None
    raw_itemsize = getattr(dtype, "size", None)
    if raw_itemsize is None:
        raw_itemsize = getattr(dtype, "itemsize", None)
    try:
        itemsize = int(raw_itemsize)
    except (TypeError, ValueError, OverflowError):
        return None
    if itemsize < 1:
        return None
    return str(dtype), itemsize


def classify_kv_concat(
    parts: Sequence[Any],
    *,
    axis: int,
) -> KVConcatGeometry | None:
    """Classify the physical cached-plus-active K/V concat shape.

    The target has at least one cached part and one active part, all shaped
    ``[rows, heads, head_dim]`` with the same dtype, concatenated on rows.
    Everything else is deliberately left unclassified.
    """

    try:
        values = tuple(parts)
    except TypeError:
        return None
    if len(values) < 2:
        return None

    try:
        normalized_axis = int(axis)
    except (TypeError, ValueError, OverflowError):
        return None
    if normalized_axis < 0:
        normalized_axis += 3
    if normalized_axis != 0:
        return None

    shapes = tuple(_shape(value) for value in values)
    if any(shape is None or len(shape) != 3 for shape in shapes):
        return None
    part_shapes = tuple(shape for shape in shapes if shape is not None)
    if any(shape[1:] != part_shapes[0][1:] for shape in part_shapes[1:]):
        return None

    dtype_geometries = tuple(_dtype_geometry(value) for value in values)
    if any(geometry is None for geometry in dtype_geometries):
        return None
    known_dtypes = tuple(
        geometry for geometry in dtype_geometries if geometry is not None
    )
    if any(geometry != known_dtypes[0] for geometry in known_dtypes[1:]):
        return None

    dtype_name, itemsize = known_dtypes[0]
    output_shape = (
        sum(shape[0] for shape in part_shapes),
        part_shapes[0][1],
        part_shapes[0][2],
    )
    logical_input_bytes = sum(prod(shape) * itemsize for shape in part_shapes)
    logical_output_bytes = prod(output_shape) * itemsize
    return KVConcatGeometry(
        part_shapes=part_shapes,
        output_shape=output_shape,
        dtype=dtype_name,
        itemsize=itemsize,
        logical_input_bytes=logical_input_bytes,
        logical_output_bytes=logical_output_bytes,
    )


class KVConcatRecorder:
    """Wrap a concatenate callable and record synchronized matching calls."""

    def __init__(self, *, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self.reset()

    def reset(self) -> None:
        self._call_durations_ms: list[float] = []
        self._logical_input_bytes = 0
        self._logical_output_bytes = 0

    @property
    def calls(self) -> int:
        return len(self._call_durations_ms)

    @property
    def call_durations_ms(self) -> tuple[float, ...]:
        return tuple(self._call_durations_ms)

    @property
    def logical_input_bytes(self) -> int:
        return self._logical_input_bytes

    @property
    def logical_output_bytes(self) -> int:
        return self._logical_output_bytes

    def wrap(
        self,
        concatenate: Callable[..., Any],
        *,
        evaluate: Callable[[Any], Any],
    ) -> Callable[..., Any]:
        """Return a drop-in wrapper without importing or mutating MLX."""

        def measured(parts: Sequence[Any], axis: int = 0, *args: Any, **kwargs: Any) -> Any:
            materialized_parts = tuple(parts)
            geometry = classify_kv_concat(materialized_parts, axis=axis)
            if geometry is None:
                return concatenate(materialized_parts, axis, *args, **kwargs)

            started_at = self._clock()
            output = concatenate(materialized_parts, axis, *args, **kwargs)
            evaluate(output)
            duration_ms = round((self._clock() - started_at) * 1000.0, 9)
            self._call_durations_ms.append(duration_ms)
            self._logical_input_bytes += geometry.logical_input_bytes
            self._logical_output_bytes += geometry.logical_output_bytes
            return output

        return measured

    def sample(
        self,
        *,
        repeat: int,
        wall_ms: float,
        forward_ms: float,
        output_tokens: int,
        forward_count: int,
    ) -> KVConcatSample:
        return KVConcatSample(
            repeat=repeat,
            wall_ms=wall_ms,
            forward_ms=forward_ms,
            concat_call_ms=self.call_durations_ms,
            concat_calls=self.calls,
            logical_input_bytes=self.logical_input_bytes,
            logical_output_bytes=self.logical_output_bytes,
            output_tokens=output_tokens,
            forward_count=forward_count,
        )


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * weight, 9)


def _mad(values: Sequence[float]) -> float:
    center = median(values)
    return round(float(median(abs(value - center) for value in values)), 9)


def build_kv_concat_report(
    *,
    package_identity: Mapping[str, Any],
    config_identity: Mapping[str, Any],
    warmup: int,
    repeats: int,
    max_tokens: int,
    samples: Sequence[KVConcatSample],
) -> dict[str, Any]:
    """Build the auditable Task 0 JSON report and 3% gate verdict."""

    measured = tuple(samples)
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    if repeats < 1 or len(measured) != repeats:
        raise ValueError("samples length must equal positive repeats")

    concat_totals = [sample.concat_ms for sample in measured]
    forward_times = [sample.forward_ms for sample in measured]
    wall_times = [sample.wall_ms for sample in measured]
    concat_p50 = _percentile(concat_totals, 0.50)
    forward_p50 = _percentile(forward_times, 0.50)
    total_calls = sum(sample.concat_calls for sample in measured)

    # A missing call in any repeat means the instrumentation or runtime path
    # changed. Do not average that failure into an apparently valid result.
    measurement_valid = (
        all(sample.concat_calls > 0 for sample in measured)
        and all(sample.forward_ms > 0.0 for sample in measured)
        and forward_p50 > 0.0
    )
    concat_share = (
        round(concat_p50 / forward_p50 * 100.0, 9)
        if measurement_valid
        else None
    )
    should_continue = bool(concat_share is not None and concat_share >= 3.0)
    if not measurement_valid:
        verdict = "BLOCKED"
        reason = "every repeat must contain matching concat calls and positive forward timing"
    elif should_continue:
        verdict = "CONTINUE"
        reason = "synchronized concat is at least 3% of p50 forward time"
    else:
        verdict = "CONCLUDE_NEGATIVE"
        reason = "synchronized concat is below 3% of p50 forward time"

    return {
        "schema_version": 1,
        "status": "DONE" if measurement_valid else "BLOCKED",
        "package_identity": dict(package_identity),
        "config_identity": dict(config_identity),
        "benchmark": {
            "warmup": int(warmup),
            "repeats": int(repeats),
            "max_tokens": int(max_tokens),
        },
        "raw_samples": [sample.to_dict() for sample in measured],
        "summary": {
            "calls": {"total": total_calls},
            "bytes": {
                "logical_input_total": sum(
                    sample.logical_input_bytes for sample in measured
                ),
                "logical_output_total": sum(
                    sample.logical_output_bytes for sample in measured
                ),
            },
            "concat_ms": {
                "p50": concat_p50,
                "p95": _percentile(concat_totals, 0.95),
            },
            "forward_ms": {"p50": forward_p50},
            "wall_ms": {"p50": _percentile(wall_times, 0.50)},
            "mad_ms": {
                "concat": _mad(concat_totals),
                "forward": _mad(forward_times),
                "wall": _mad(wall_times),
            },
        },
        "continuation_verdict": {
            "threshold_percent": 3.0,
            "concat_share_of_p50_forward_percent": concat_share,
            "shape_matched_no_concat_measured": False,
            "continue": should_continue,
            "verdict": verdict,
            "reason": reason,
        },
    }


def build_blocked_kv_concat_report(
    *,
    package_path: str,
    warmup: int,
    repeats: int,
    max_tokens: int,
    error: BaseException,
) -> dict[str, Any]:
    """Build fail-closed evidence when the real-package gate cannot run."""

    return {
        "schema_version": 1,
        "status": "BLOCKED",
        "package_identity": {"path": package_path},
        "config_identity": {},
        "benchmark": {
            "warmup": int(warmup),
            "repeats": int(repeats),
            "max_tokens": int(max_tokens),
        },
        "raw_samples": [],
        "summary": None,
        "continuation_verdict": {
            "threshold_percent": 3.0,
            "concat_share_of_p50_forward_percent": None,
            "shape_matched_no_concat_measured": False,
            "continue": False,
            "verdict": "BLOCKED",
            "reason": f"{type(error).__name__}: {error}",
        },
    }


__all__ = [
    "KVConcatGeometry",
    "KVConcatRecorder",
    "KVConcatSample",
    "build_blocked_kv_concat_report",
    "build_kv_concat_report",
    "classify_kv_concat",
]
