from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

_METAL_DTYPES = frozenset({"float32", "float16", "bfloat16"})


class TargetFeatureHandoffError(ValueError):
    def __init__(self, code: str, message: str | None = None) -> None:
        super().__init__(message or code)
        self.code = code


@dataclass(frozen=True, slots=True)
class ProjectionGeometry:
    batch_size: int
    sequence_length: int
    hidden_size: int
    output_size: int
    layer_count: int
    start: int
    length: int
    target_layer_ids: tuple[int, ...]
    dtype: str

    def to_dict(self) -> dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "layer_count": self.layer_count,
            "start": self.start,
            "length": self.length,
            "target_layer_ids": list(self.target_layer_ids),
            "dtype": self.dtype,
        }


@dataclass(frozen=True, slots=True)
class ProjectionEvidence:
    verdict: str
    promotion_verdict: str
    geometry: ProjectionGeometry
    oracle_digest: str
    native_digest: str
    max_abs_error: float | None
    max_relative_error: float | None
    comparison: dict[str, object]
    timings_ms: dict[str, float]
    memory_bytes: dict[str, int]
    environment: dict[str, str]
    round_summary: dict[str, object]
    reasons: tuple[str, ...]
    promotion_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "verdict": self.verdict,
            "promotion_verdict": self.promotion_verdict,
            "geometry": self.geometry.to_dict(),
            "oracle_digest": self.oracle_digest,
            "native_digest": self.native_digest,
            "max_abs_error": self.max_abs_error,
            "max_relative_error": self.max_relative_error,
            "comparison": dict(sorted(self.comparison.items())),
            "timings_ms": dict(sorted(self.timings_ms.items())),
            "memory_bytes": dict(sorted(self.memory_bytes.items())),
            "environment": dict(sorted(self.environment.items())),
            "round_summary": dict(sorted(self.round_summary.items())),
            "reasons": list(self.reasons),
            "promotion_reasons": list(self.promotion_reasons),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ) + "\n"


@dataclass(frozen=True, slots=True)
class PreparedProjectionWeights:
    chunks: tuple[Any, ...]
    layer_count: int
    hidden_size: int
    output_size: int
    dtype: str


@dataclass(frozen=True, slots=True)
class ProjectedFeatureHandoff:
    values: Any
    geometry: ProjectionGeometry


def _shape(value: Any, *, code: str) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value.shape)
    except (AttributeError, TypeError, ValueError) as exc:
        raise TargetFeatureHandoffError(code) from exc


def _dtype_name(value: Any, *, code: str) -> str:
    try:
        dtype = value.dtype
    except AttributeError as exc:
        raise TargetFeatureHandoffError(code) from exc
    return str(dtype).removeprefix("mlx.core.")


def _strict_nonnegative_int(value: object, *, code: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise TargetFeatureHandoffError(code)
    return value


def validate_projection_inputs(
    hidden_layers: Sequence[Any],
    weight: Any,
    *,
    start: int,
    length: int,
    target_layer_ids: Sequence[int],
    expected_target_layer_ids: Sequence[int] | None = None,
) -> ProjectionGeometry:
    if not hidden_layers:
        raise TargetFeatureHandoffError("hidden_layers_empty")

    first_shape = _shape(hidden_layers[0], code="hidden_shape_invalid")
    if len(first_shape) != 3 or any(dim <= 0 for dim in first_shape):
        raise TargetFeatureHandoffError("hidden_shape_invalid")
    batch_size, sequence_length, hidden_size = first_shape
    first_dtype = _dtype_name(hidden_layers[0], code="hidden_dtype_invalid")

    for hidden in hidden_layers[1:]:
        if _shape(hidden, code="hidden_shape_invalid") != first_shape:
            raise TargetFeatureHandoffError("hidden_shape_mismatch")
        if _dtype_name(hidden, code="hidden_dtype_invalid") != first_dtype:
            raise TargetFeatureHandoffError("hidden_dtype_mismatch")

    weight_shape = _shape(weight, code="weight_shape_invalid")
    if len(weight_shape) != 2 or any(dim <= 0 for dim in weight_shape):
        raise TargetFeatureHandoffError("weight_shape_invalid")
    output_size, weight_input_size = weight_shape
    if weight_input_size != len(hidden_layers) * hidden_size:
        raise TargetFeatureHandoffError("weight_geometry_mismatch")
    if _dtype_name(weight, code="weight_dtype_invalid") != first_dtype:
        raise TargetFeatureHandoffError("weight_dtype_mismatch")

    range_start = _strict_nonnegative_int(start, code="range_start_invalid")
    range_length = _strict_nonnegative_int(length, code="range_length_invalid")
    if range_length == 0:
        raise TargetFeatureHandoffError("range_length_invalid")
    if range_start + range_length > sequence_length:
        raise TargetFeatureHandoffError("range_out_of_bounds")

    layer_ids = tuple(target_layer_ids)
    if len(layer_ids) != len(hidden_layers):
        raise TargetFeatureHandoffError("target_layer_ids_length_mismatch")
    if any(
        not isinstance(layer_id, int)
        or isinstance(layer_id, bool)
        or layer_id < 0
        for layer_id in layer_ids
    ):
        raise TargetFeatureHandoffError("target_layer_ids_invalid")
    if len(set(layer_ids)) != len(layer_ids):
        raise TargetFeatureHandoffError("target_layer_ids_duplicate")
    if (
        expected_target_layer_ids is not None
        and layer_ids != tuple(expected_target_layer_ids)
    ):
        raise TargetFeatureHandoffError("target_layer_order_mismatch")

    return ProjectionGeometry(
        batch_size=batch_size,
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        output_size=output_size,
        layer_count=len(hidden_layers),
        start=range_start,
        length=range_length,
        target_layer_ids=layer_ids,
        dtype=first_dtype,
    )


def reference_split_range_projection(
    hidden_layers: Sequence[Any],
    weight: Any,
    *,
    start: int,
    length: int,
    target_layer_ids: Sequence[int],
    expected_target_layer_ids: Sequence[int] | None = None,
) -> np.ndarray:
    geometry = validate_projection_inputs(
        hidden_layers,
        weight,
        start=start,
        length=length,
        target_layer_ids=target_layer_ids,
        expected_target_layer_ids=expected_target_layer_ids,
    )
    if geometry.dtype not in _METAL_DTYPES:
        raise TargetFeatureHandoffError("dtype_unsupported")
    concatenated = np.concatenate(
        [np.asarray(hidden) for hidden in hidden_layers],
        axis=-1,
    )
    selected = concatenated[
        :,
        geometry.start : geometry.start + geometry.length,
        :,
    ]
    return selected @ np.asarray(weight).T


def prepare_projection_weights(
    weight: Any,
    *,
    layer_count: int,
    hidden_size: int,
) -> PreparedProjectionWeights:
    if (
        not isinstance(layer_count, int)
        or isinstance(layer_count, bool)
        or layer_count <= 0
    ):
        raise TargetFeatureHandoffError("layer_count_invalid")
    if (
        not isinstance(hidden_size, int)
        or isinstance(hidden_size, bool)
        or hidden_size <= 0
    ):
        raise TargetFeatureHandoffError("hidden_size_invalid")
    shape = _shape(weight, code="weight_shape_invalid")
    if len(shape) != 2 or shape[0] <= 0:
        raise TargetFeatureHandoffError("weight_shape_invalid")
    output_size, input_size = shape
    if input_size != layer_count * hidden_size:
        raise TargetFeatureHandoffError("weight_geometry_mismatch")
    dtype = _dtype_name(weight, code="weight_dtype_invalid")
    if dtype not in _METAL_DTYPES:
        raise TargetFeatureHandoffError("dtype_unsupported")

    try:
        import mlx.core as mx
    except Exception as exc:
        raise RuntimeError("MLX Metal projection unavailable") from exc

    chunks = tuple(
        mx.contiguous(
            weight[:, layer_index * hidden_size : (layer_index + 1) * hidden_size]
        )
        for layer_index in range(layer_count)
    )
    mx.eval(*chunks)
    return PreparedProjectionWeights(
        chunks=chunks,
        layer_count=layer_count,
        hidden_size=hidden_size,
        output_size=output_size,
        dtype=dtype,
    )


def native_split_projection(
    hidden_layers: Sequence[Any],
    prepared_weights: PreparedProjectionWeights,
    *,
    start: int,
    length: int,
    target_layer_ids: Sequence[int],
    expected_target_layer_ids: Sequence[int],
) -> ProjectedFeatureHandoff:
    layers = tuple(hidden_layers)
    if not layers:
        raise TargetFeatureHandoffError("hidden_layers_empty")
    first_layer = next(iter(layers))
    validation_weight = _PreparedWeightShape(
        shape=(
            prepared_weights.output_size,
            prepared_weights.layer_count * prepared_weights.hidden_size,
        ),
        dtype=first_layer.dtype,
    )
    geometry = validate_projection_inputs(
        layers,
        validation_weight,
        start=start,
        length=length,
        target_layer_ids=target_layer_ids,
        expected_target_layer_ids=expected_target_layer_ids,
    )
    if geometry.dtype not in _METAL_DTYPES:
        raise TargetFeatureHandoffError("dtype_unsupported")
    if (
        prepared_weights.layer_count != geometry.layer_count
        or prepared_weights.hidden_size != geometry.hidden_size
        or prepared_weights.output_size != geometry.output_size
    ):
        raise TargetFeatureHandoffError("prepared_weight_geometry_mismatch")
    if prepared_weights.dtype != geometry.dtype:
        raise TargetFeatureHandoffError("prepared_weight_dtype_mismatch")
    if len(prepared_weights.chunks) != geometry.layer_count:
        raise TargetFeatureHandoffError("prepared_weight_count_mismatch")
    expected_chunk_shape = (geometry.output_size, geometry.hidden_size)
    for chunk in prepared_weights.chunks:
        if _shape(chunk, code="prepared_weight_shape_invalid") != expected_chunk_shape:
            raise TargetFeatureHandoffError("prepared_weight_shape_mismatch")
        if (
            _dtype_name(chunk, code="prepared_weight_dtype_invalid")
            != geometry.dtype
        ):
            raise TargetFeatureHandoffError("prepared_weight_dtype_mismatch")

    try:
        import mlx.core as mx
    except Exception as exc:
        raise RuntimeError("MLX Metal projection unavailable") from exc

    if geometry.dtype == "float32":
        projected = mx.matmul(first_layer, prepared_weights.chunks[0].T)
    else:
        projected = mx.matmul(
            first_layer.astype(mx.float32),
            prepared_weights.chunks[0].T,
        )
    for hidden, chunk in zip(
        layers[1:],
        prepared_weights.chunks[1:],
        strict=True,
    ):
        projected = mx.addmm(
            projected,
            hidden if geometry.dtype == "float32" else hidden.astype(mx.float32),
            chunk.T,
        )
    if geometry.dtype != "float32":
        projected = projected.astype(first_layer.dtype)
    return ProjectedFeatureHandoff(values=projected, geometry=geometry)


@dataclass(frozen=True, slots=True)
class _PreparedWeightShape:
    shape: tuple[int, int]
    dtype: Any


def array_digest(value: Any) -> str:
    array = np.asarray(value)
    hasher = hashlib.sha256()
    hasher.update(
        json.dumps(
            {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    hasher.update(array.tobytes(order="C"))
    return hasher.hexdigest()
