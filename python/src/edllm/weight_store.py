"""Versioned weight artifacts shared by the compiler and runtime.

An execution package may carry either the established single-file NPZ payload or
a dtype-preserving safetensors store.  Model definitions never choose between
these encodings: the compiler selects a storage implementation, records it in
the package, and the immutable build plan carries that decision to the backend.

The sharded format uses one bounded JSON index as its root.  The index names and
hashes every shard and records the exact tensor set.  Runtime readers open every
file relative to the verified package directory and never follow symlinks.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from .package_artifacts import (
    artifact_source_stat,
    open_package_artifact,
    read_open_artifact,
    safe_relative_artifact_ref,
    sha256_stream,
)

NPZ_V1 = "npz_v1"
SAFETENSORS_SHARDED_V1 = "safetensors_sharded_v1"
PRESERVED_DTYPE = "preserved"
WEIGHT_STORE_INDEX_VERSION = 1
WEIGHT_STORE_INDEX_REF = "weights/weights.index.json"
DEFAULT_MAX_SHARD_BYTES = 512 * 1024 * 1024
MAX_WEIGHT_INDEX_BYTES = 16 * 1024 * 1024

_INDEX_FIELDS = frozenset(
    {
        "schema_version",
        "format",
        "storage_dtype",
        "tensor_count",
        "total_tensor_bytes",
        "shards",
    }
)
_SHARD_FIELDS = frozenset({"ref", "sha256", "byte_length", "tensors"})
_TENSOR_FIELDS = frozenset({"name", "dtype", "shape", "byte_length"})


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_json_loads(payload: bytes) -> object:
    def object_from_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON number: {value}")

    try:
        return json.loads(
            payload,
            object_pairs_hook=object_from_pairs,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, RecursionError, UnicodeDecodeError) as exc:
        raise ValueError("weight_store_index_malformed") from exc


@dataclass(frozen=True, slots=True)
class WeightArtifactSpec:
    """The package-level storage decision carried by a resolved build plan."""

    format: str
    primary_ref: str
    storage_dtype: str
    digest: str = ""

    def __post_init__(self) -> None:
        if type(self.format) is not str or type(self.storage_dtype) is not str:
            raise ValueError("weight_artifact_identity_invalid")
        if self.format not in {NPZ_V1, SAFETENSORS_SHARDED_V1}:
            raise ValueError(f"weight_artifact_format_unknown: {self.format!r}")
        if type(self.primary_ref) is not str or not self.primary_ref:
            raise ValueError("weight_artifact_primary_ref_invalid")
        safe_relative_artifact_ref(
            self.primary_ref,
            error_prefix="weight_store",
            artifact_name="primary",
        )
        if self.format == NPZ_V1 and self.storage_dtype != "fp32":
            raise ValueError("npz_v1 requires weight_storage_dtype='fp32'")
        if (
            self.format == SAFETENSORS_SHARDED_V1
            and self.storage_dtype != PRESERVED_DTYPE
        ):
            raise ValueError(
                "safetensors_sharded_v1 requires weight_storage_dtype='preserved'"
            )
        if self.digest and not _is_sha256(self.digest):
            raise ValueError("weight_artifact_digest_invalid")

    @classmethod
    def legacy_npz(cls, primary_ref: str, *, digest: str = "") -> WeightArtifactSpec:
        return cls(
            format=NPZ_V1,
            primary_ref=primary_ref,
            storage_dtype="fp32",
            digest=digest,
        )

    def supports_backend(self, backend: str) -> bool:
        if self.format == NPZ_V1:
            return True
        return backend in {"mlx_local", "unbound"}


def weight_artifact_spec_from_manifest(manifest: Mapping[str, object]) -> WeightArtifactSpec:
    """Resolve one explicit storage contract, preserving legacy Product packages."""

    primary_ref = manifest.get("merged_weights_ref")
    if type(primary_ref) is not str or not primary_ref:
        raise ValueError("package_weights_ref_missing")
    digest = manifest.get("output_hash", "")
    if digest is None:
        digest = ""
    if type(digest) is not str:
        raise ValueError("weight_artifact_digest_invalid")

    package_format = manifest.get("package_format")
    storage_dtype = manifest.get("weight_storage_dtype")
    if package_format is None and storage_dtype is None:
        return WeightArtifactSpec.legacy_npz(primary_ref, digest=digest)
    if type(package_format) is not str or type(storage_dtype) is not str:
        raise ValueError("package_weight_storage_identity_incomplete")
    return WeightArtifactSpec(
        format=package_format,
        primary_ref=primary_ref,
        storage_dtype=storage_dtype,
        digest=digest,
    )


@dataclass(frozen=True, slots=True)
class WeightTensorRecord:
    name: str
    dtype: str
    shape: tuple[int, ...]
    byte_length: int


@dataclass(frozen=True, slots=True)
class WeightShardRecord:
    ref: str
    sha256: str
    byte_length: int
    tensors: tuple[WeightTensorRecord, ...]


@dataclass(frozen=True, slots=True)
class WeightStoreIndex:
    shards: tuple[WeightShardRecord, ...]
    tensor_count: int
    total_tensor_bytes: int
    source_path: Path
    source_stat: tuple[int, int, int, int, int]
    source_digest: str

    @property
    def tensor_names(self) -> frozenset[str]:
        return frozenset(
            tensor.name for shard in self.shards for tensor in shard.tensors
        )


@dataclass(frozen=True, slots=True)
class WrittenWeightArtifact:
    spec: WeightArtifactSpec
    tensor_names: frozenset[str]
    tensor_count: int
    total_tensor_bytes: int


def _parse_tensor_record(payload: object) -> WeightTensorRecord:
    if type(payload) is not dict or set(payload) != _TENSOR_FIELDS:
        raise ValueError("weight_store_tensor_record_invalid")
    name = payload["name"]
    dtype = payload["dtype"]
    shape = payload["shape"]
    byte_length = payload["byte_length"]
    if type(name) is not str or not name:
        raise ValueError("weight_store_tensor_name_invalid")
    if type(dtype) is not str or not dtype:
        raise ValueError("weight_store_tensor_dtype_invalid")
    if type(shape) is not list or any(
        type(dimension) is not int or dimension < 0 for dimension in shape
    ):
        raise ValueError("weight_store_tensor_shape_invalid")
    if type(byte_length) is not int or byte_length < 0:
        raise ValueError("weight_store_tensor_bytes_invalid")
    return WeightTensorRecord(
        name=name,
        dtype=dtype,
        shape=tuple(shape),
        byte_length=byte_length,
    )


def _parse_weight_store_index(
    payload: bytes,
    *,
    source_path: Path,
    source_stat: os.stat_result,
) -> WeightStoreIndex:
    parsed = _strict_json_loads(payload)
    if type(parsed) is not dict or set(parsed) != _INDEX_FIELDS:
        raise ValueError("weight_store_index_contract_invalid")
    if (
        type(parsed["schema_version"]) is not int
        or parsed["schema_version"] != WEIGHT_STORE_INDEX_VERSION
    ):
        raise ValueError("weight_store_index_version_unsupported")
    if parsed["format"] != "safetensors":
        raise ValueError("weight_store_index_format_unsupported")
    if parsed["storage_dtype"] != PRESERVED_DTYPE:
        raise ValueError("weight_store_index_storage_dtype_invalid")
    tensor_count = parsed["tensor_count"]
    total_tensor_bytes = parsed["total_tensor_bytes"]
    shards_payload = parsed["shards"]
    if type(tensor_count) is not int or tensor_count <= 0:
        raise ValueError("weight_store_index_tensor_count_invalid")
    if type(total_tensor_bytes) is not int or total_tensor_bytes <= 0:
        raise ValueError("weight_store_index_tensor_bytes_invalid")
    if type(shards_payload) is not list or not shards_payload:
        raise ValueError("weight_store_index_shards_invalid")

    shards: list[WeightShardRecord] = []
    seen_refs: set[str] = set()
    seen_tensors: set[str] = set()
    calculated_tensor_bytes = 0
    for shard_payload in shards_payload:
        if type(shard_payload) is not dict or set(shard_payload) != _SHARD_FIELDS:
            raise ValueError("weight_store_shard_record_invalid")
        ref = shard_payload["ref"]
        digest = shard_payload["sha256"]
        byte_length = shard_payload["byte_length"]
        tensors_payload = shard_payload["tensors"]
        if type(ref) is not str or not ref:
            raise ValueError("weight_store_shard_ref_invalid")
        safe_relative_artifact_ref(
            ref,
            error_prefix="weight_store",
            artifact_name="shard",
        )
        if ref in seen_refs:
            raise ValueError("weight_store_shard_ref_duplicated")
        seen_refs.add(ref)
        if not _is_sha256(digest):
            raise ValueError("weight_store_shard_digest_invalid")
        if type(byte_length) is not int or byte_length <= 0:
            raise ValueError("weight_store_shard_bytes_invalid")
        if type(tensors_payload) is not list or not tensors_payload:
            raise ValueError("weight_store_shard_tensors_invalid")
        tensors = tuple(_parse_tensor_record(item) for item in tensors_payload)
        for tensor in tensors:
            if tensor.name in seen_tensors:
                raise ValueError("weight_store_tensor_name_duplicated")
            seen_tensors.add(tensor.name)
            calculated_tensor_bytes += tensor.byte_length
        shards.append(
            WeightShardRecord(
                ref=ref,
                sha256=digest,
                byte_length=byte_length,
                tensors=tensors,
            )
        )
    if len(seen_tensors) != tensor_count:
        raise ValueError("weight_store_index_tensor_count_mismatch")
    if calculated_tensor_bytes != total_tensor_bytes:
        raise ValueError("weight_store_index_tensor_bytes_mismatch")
    return WeightStoreIndex(
        shards=tuple(shards),
        tensor_count=tensor_count,
        total_tensor_bytes=total_tensor_bytes,
        source_path=source_path,
        source_stat=artifact_source_stat(source_stat),
        source_digest=hashlib.sha256(payload).hexdigest(),
    )


def read_weight_store_index(
    package_path: str | os.PathLike,
    spec: WeightArtifactSpec,
) -> WeightStoreIndex:
    """Read and validate the bounded index through the package artifact boundary."""

    if spec.format != SAFETENSORS_SHARDED_V1:
        raise ValueError("weight_store_index_requires_sharded_safetensors")
    with open_package_artifact(
        package_path=package_path,
        artifact_ref=spec.primary_ref,
        error_prefix="weight_store",
        artifact_name="index",
        max_bytes=MAX_WEIGHT_INDEX_BYTES,
    ) as (handle, source_path, source_stat):
        payload = read_open_artifact(
            handle,
            max_bytes=MAX_WEIGHT_INDEX_BYTES,
            error_prefix="weight_store",
            artifact_name="index",
        )
        final_stat = os.fstat(handle.fileno())
        if artifact_source_stat(final_stat) != artifact_source_stat(source_stat):
            raise ValueError("weight_store_index_changed")
    index = _parse_weight_store_index(
        payload,
        source_path=source_path,
        source_stat=source_stat,
    )
    if spec.digest and index.source_digest != spec.digest:
        raise ValueError("weight_store_index_digest_mismatch")
    return index


@contextmanager
def open_verified_weight_shard(
    package_path: str | os.PathLike,
    shard: WeightShardRecord,
) -> Iterator[BinaryIO]:
    """Yield one safe shard handle, then verify its recorded bytes before return."""

    with open_package_artifact(
        package_path=package_path,
        artifact_ref=shard.ref,
        error_prefix="weight_store",
        artifact_name="shard",
    ) as (handle, _source_path, source_stat):
        if source_stat.st_size != shard.byte_length:
            raise ValueError("weight_store_shard_size_mismatch")
        yield handle
        final_stat = os.fstat(handle.fileno())
        if artifact_source_stat(final_stat) != artifact_source_stat(source_stat):
            raise ValueError("weight_store_shard_changed")
        handle.seek(0)
        if sha256_stream(handle) != shard.sha256:
            raise ValueError("weight_store_shard_digest_mismatch")
        if artifact_source_stat(os.fstat(handle.fileno())) != artifact_source_stat(
            source_stat
        ):
            raise ValueError("weight_store_shard_changed")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _torch_tensor_signature(tensor: Any) -> tuple[tuple[int, ...], str, str]:
    value = tensor.detach().to("cpu").contiguous()
    # NumPy cannot represent bfloat16, but a uint8 view preserves every source
    # bit and exposes a buffer without converting the tensor's numerical dtype.
    import torch

    raw_bytes = value.view(torch.uint8).numpy()
    return (
        tuple(int(dimension) for dimension in value.shape),
        str(value.dtype).removeprefix("torch."),
        hashlib.sha256(memoryview(raw_bytes).cast("B")).hexdigest(),
    )


def write_safetensors_weight_store(
    package_dir: str | os.PathLike,
    items: Iterable[tuple[str, Any]],
    *,
    tied_weight_keys: tuple[str, ...] = (),
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
) -> WrittenWeightArtifact:
    """Write mapped CPU tensors into deterministic safetensors shards.

    ``max_shard_bytes`` is a target, not permission to split a tensor: an
    individual tensor larger than the target occupies one larger shard. This
    matches common checkpoint sharding and keeps canonical tensor names intact.
    """

    if type(max_shard_bytes) is not int or max_shard_bytes <= 0:
        raise ValueError("max_shard_bytes must be a positive integer")
    required_tied = set(tied_weight_keys)
    if tied_weight_keys and len(required_tied) != 2:
        raise ValueError("declared_tied_embeddings_keys_invalid")

    from safetensors.torch import save_file

    package_path = Path(package_dir)
    target_dir = package_path / "weights"
    if target_dir.exists():
        raise FileExistsError("weight store target already exists")
    partial_dir = package_path / f".weights.partial-{uuid.uuid4().hex}"
    partial_dir.mkdir(parents=False)
    iterator = iter(items)
    seen: set[str] = set()
    tied_signatures: dict[str, tuple[tuple[int, ...], str, str]] = {}
    shard_records: list[dict[str, object]] = []
    batch: dict[str, Any] = {}
    batch_records: list[dict[str, object]] = []
    batch_bytes = 0
    total_tensor_bytes = 0

    def flush() -> None:
        nonlocal batch, batch_records, batch_bytes
        if not batch:
            return
        shard_number = len(shard_records) + 1
        filename = f"shard-{shard_number:05d}.safetensors"
        shard_path = partial_dir / filename
        save_file(batch, str(shard_path), metadata={"format": "diffusorrt"})
        shard_path.chmod(0o644)
        shard_records.append(
            {
                "ref": f"weights/{filename}",
                "sha256": _sha256_file(shard_path),
                "byte_length": shard_path.stat().st_size,
                "tensors": batch_records,
            }
        )
        batch = {}
        batch_records = []
        batch_bytes = 0

    try:
        try:
            for name, source_tensor in iterator:
                if type(name) is not str or not name:
                    raise ValueError("weight_store_tensor_name_invalid")
                if name in seen:
                    raise ValueError(f"duplicate tensor name across shards: {name!r}")
                seen.add(name)
                tensor = source_tensor.detach().to("cpu").contiguous()
                tensor_bytes = int(tensor.numel()) * int(tensor.element_size())
                if batch and batch_bytes + tensor_bytes > max_shard_bytes:
                    flush()
                dtype_name = str(tensor.dtype).removeprefix("torch.")
                record = {
                    "name": name,
                    "dtype": dtype_name,
                    "shape": [int(dimension) for dimension in tensor.shape],
                    "byte_length": tensor_bytes,
                }
                if name in required_tied:
                    signature = _torch_tensor_signature(tensor)
                    if tied_signatures and signature != next(
                        iter(tied_signatures.values())
                    ):
                        raise ValueError("declared_tied_embeddings_mismatch")
                    tied_signatures[name] = signature
                batch[name] = tensor
                batch_records.append(record)
                batch_bytes += tensor_bytes
                total_tensor_bytes += tensor_bytes
            flush()
        except BaseException:
            with contextlib.suppress(Exception):
                close = getattr(iterator, "close", None)
                if close is not None:
                    close()
            raise
        else:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()

        missing_tied = sorted(required_tied - tied_signatures.keys())
        if missing_tied:
            raise ValueError(
                "declared_tied_embeddings_missing: " + ", ".join(missing_tied)
            )
        if not shard_records:
            raise ValueError("weight_store_requires_at_least_one_tensor")
        index_payload = {
            "schema_version": WEIGHT_STORE_INDEX_VERSION,
            "format": "safetensors",
            "storage_dtype": PRESERVED_DTYPE,
            "tensor_count": len(seen),
            "total_tensor_bytes": total_tensor_bytes,
            "shards": shard_records,
        }
        index_path = partial_dir / "weights.index.json"
        index_path.write_text(
            json.dumps(index_payload, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        index_path.chmod(0o644)
        index_digest = _sha256_file(index_path)
        partial_dir.replace(target_dir)
    except BaseException:
        shutil.rmtree(partial_dir, ignore_errors=True)
        raise

    spec = WeightArtifactSpec(
        format=SAFETENSORS_SHARDED_V1,
        primary_ref=WEIGHT_STORE_INDEX_REF,
        storage_dtype=PRESERVED_DTYPE,
        digest=index_digest,
    )
    return WrittenWeightArtifact(
        spec=spec,
        tensor_names=frozenset(seen),
        tensor_count=len(seen),
        total_tensor_bytes=total_tensor_bytes,
    )
