"""Snapshot save/restore v0 — pure durable MaskCanvas artifact.

The snapshot store persists a JSON-native bundle containing a full MaskCanvas
trace/snapshot plus schema-v1 compatibility fields for the retired Python K/V
stores. New saves keep those fields neutral. It is deliberately pure: no MLX /
torch / numpy, no state-cache imports, and no model execution on load.
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..package_artifacts import (
    artifact_source_stat,
    open_package_artifact,
    read_open_artifact,
    safe_relative_artifact_ref,
    sha256_bounded_stream,
    sha256_stream,
)
from .cache_identity import PolicyVersion

SNAPSHOT_SCHEMA_VERSION = 1
SNAPSHOT_KIND = "edllm.snapshot"
DEFAULT_MAX_SNAPSHOT_MANIFEST_BYTES = 64 * 1024 * 1024
_MAX_PACKAGE_MANIFEST_BYTES = 4 * 1024 * 1024
_MAX_TOKENIZER_BYTES = 256 * 1024 * 1024
_MAX_GRAPH_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class SnapshotIdentity:
    """Identity that gates whether a snapshot belongs to the current runtime."""

    package_hash: str
    tokenizer_hash: str
    graph_hash: str
    policy_version: str
    cache_salt: str = ""
    privacy_scope: str = "session"


def _identity_hash(domain: str, value: Any) -> str:
    return hashlib.sha256(
        (domain + json.dumps(value, sort_keys=True, default=str)).encode("utf-8")
    ).hexdigest()


def _verified_weights_digest(
    *, package_path: Path, weights_ref: str, expected_digest: str
) -> str:
    with open_package_artifact(
        package_path=package_path,
        artifact_ref=weights_ref,
        error_prefix="snapshot",
        artifact_name="weights",
    ) as (handle, _display_path, opened_stat):
        actual_digest = sha256_stream(handle)
        final_stat = os.fstat(handle.fileno())
        if artifact_source_stat(opened_stat) != artifact_source_stat(final_stat):
            raise ValueError("snapshot_weights_artifact_changed")
    if actual_digest != expected_digest:
        raise ValueError("snapshot_weights_hash_mismatch")
    return actual_digest


def build_snapshot_identity(
    *,
    package_dir: str | os.PathLike,
    merged_weights_ref: str,
    model_config: dict[str, Any] | None,
    policy_version: PolicyVersion,
    cache_salt: str = "",
    privacy_scope: str = "session",
    resident_source_path: str | os.PathLike | None = None,
    resident_source_digest: str | None = None,
    resident_source_stat: tuple[int, int, int, int, int] | None = None,
) -> SnapshotIdentity:
    """Build the durable identity used by schema-v1 snapshots.

    This is the surviving, snapshot-specific part of the retired cross-session
    store. It binds package output, tokenizer, graph, runtime weight numerics,
    policy and privacy without using the package's filesystem location as
    identity.
    """
    if package_dir is None:
        raise ValueError("snapshot_package_dir_required")
    package_path = Path(package_dir)
    try:
        with open_package_artifact(
            package_path=package_path,
            artifact_ref="model.json",
            error_prefix="snapshot",
            artifact_name="manifest",
            max_bytes=_MAX_PACKAGE_MANIFEST_BYTES,
        ) as (handle, _display_path, _source_stat):
            manifest_payload = read_open_artifact(
                handle,
                max_bytes=_MAX_PACKAGE_MANIFEST_BYTES,
                error_prefix="snapshot",
                artifact_name="manifest",
            )
        manifest = json.loads(manifest_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("snapshot_manifest_unavailable") from exc

    output_hash = manifest.get("output_hash")
    manifest_weights_ref = manifest.get("merged_weights_ref")
    if not output_hash:
        raise ValueError("snapshot_output_hash_required")
    if not manifest_weights_ref or not merged_weights_ref:
        raise ValueError("snapshot_weights_ref_required")
    selected_ref = safe_relative_artifact_ref(
        manifest_weights_ref,
        error_prefix="snapshot",
        artifact_name="weights",
    )
    engine_ref = safe_relative_artifact_ref(
        merged_weights_ref,
        error_prefix="snapshot",
        artifact_name="weights",
    )
    if selected_ref != engine_ref:
        raise ValueError("snapshot_engine_manifest_weights_ref_mismatch")

    has_resident_identity = any(
        value is not None
        for value in (
            resident_source_path,
            resident_source_digest,
            resident_source_stat,
        )
    )
    if has_resident_identity:
        if resident_source_path is None:
            raise ValueError("snapshot_resident_weights_source_required")
        if not resident_source_digest:
            raise ValueError("snapshot_resident_weights_digest_required")
        if resident_source_stat is None:
            raise ValueError("snapshot_resident_weights_stat_required")
        try:
            resident_path = Path(resident_source_path).absolute()
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError(
                "snapshot_resident_weights_source_unavailable"
            ) from exc
        with open_package_artifact(
            package_path=package_path,
            artifact_ref=selected_ref,
            error_prefix="snapshot",
            artifact_name="weights",
        ) as (_handle, current_path, current_stat):
            if resident_path != current_path:
                raise ValueError("snapshot_resident_weights_source_mismatch")
            if str(resident_source_digest) != str(output_hash):
                raise ValueError("snapshot_weights_hash_mismatch")
            if tuple(resident_source_stat) != artifact_source_stat(current_stat):
                raise ValueError("snapshot_resident_weights_file_changed")
        actual_weights_digest = str(resident_source_digest)
    else:
        actual_weights_digest = _verified_weights_digest(
            package_path=package_path,
            weights_ref=os.fspath(selected_ref),
            expected_digest=str(output_hash),
        )

    package_hash = _identity_hash(
        "package-output:", {"output_hash": actual_weights_digest}
    )
    try:
        with open_package_artifact(
            package_path=package_path,
            artifact_ref="tokenizer.json",
            error_prefix="snapshot",
            artifact_name="tokenizer",
            max_bytes=_MAX_TOKENIZER_BYTES,
        ) as (handle, _display_path, _source_stat):
            tokenizer_hash = sha256_bounded_stream(
                handle,
                max_bytes=_MAX_TOKENIZER_BYTES,
                error_prefix="snapshot",
                artifact_name="tokenizer",
            )
    except ValueError as exc:
        if str(exc) == "snapshot_tokenizer_artifact_missing":
            raise ValueError("snapshot_tokenizer_artifact_required") from exc
        raise

    graph_ref = manifest.get("graph_ref")
    if graph_ref:
        with open_package_artifact(
            package_path=package_path,
            artifact_ref=graph_ref,
            error_prefix="snapshot",
            artifact_name="graph",
            max_bytes=_MAX_GRAPH_BYTES,
        ) as (handle, _display_path, _source_stat):
            graph_hash = sha256_bounded_stream(
                handle,
                max_bytes=_MAX_GRAPH_BYTES,
                error_prefix="snapshot",
                artifact_name="graph",
            )
    else:
        graph_hash = _identity_hash("graph-config:", dict(model_config or {}))
    numerics = {
        key: (model_config or {}).get(key)
        for key in ("mlx_linear_quant_bits", "mlx_linear_quant_group_size")
    }
    graph_hash = _identity_hash(
        "graph+numerics:", {"graph": graph_hash, "numerics": numerics}
    )
    return SnapshotIdentity(
        package_hash=package_hash,
        tokenizer_hash=tokenizer_hash,
        graph_hash=graph_hash,
        policy_version=policy_version.digest,
        cache_salt=cache_salt,
        privacy_scope=privacy_scope,
    )


@dataclass(frozen=True)
class SnapshotCacheState:
    """Exact-cache state recorded alongside the canvas snapshot."""

    exact_only: bool = True
    canvas_cache_enabled: bool = False
    cross_session_cache_enabled: bool = False
    cross_session_cache_root: str | None = None
    immutable_entry_count: int = 0
    page_pool_stats: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SnapshotBundle:
    """Versioned, JSON-native snapshot bundle."""

    identity: SnapshotIdentity
    canvas: dict[str, Any]
    resume: dict[str, Any]
    cache: SnapshotCacheState
    schema_version: int = SNAPSHOT_SCHEMA_VERSION
    kind: str = SNAPSHOT_KIND

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "kind": str(self.kind),
            "identity": asdict(self.identity),
            "canvas": dict(self.canvas),
            "resume": dict(self.resume),
            "cache": asdict(self.cache),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SnapshotBundle | None":
        try:
            schema_version = int(value["schema_version"])
            kind = str(value["kind"])
            cache = SnapshotCacheState(**dict(value["cache"]))
            if schema_version != SNAPSHOT_SCHEMA_VERSION:
                return None
            if kind != SNAPSHOT_KIND:
                return None
            if not cache.exact_only:
                return None
            return cls(
                schema_version=schema_version,
                kind=kind,
                identity=SnapshotIdentity(**dict(value["identity"])),
                canvas=dict(value["canvas"]),
                resume=dict(value["resume"]),
                cache=cache,
            )
        except (KeyError, TypeError, ValueError):
            return None


class SnapshotStore:
    """Manifest-last disk store for one local snapshot directory."""

    def __init__(
        self,
        *,
        root: str | os.PathLike,
        max_manifest_bytes: int = DEFAULT_MAX_SNAPSHOT_MANIFEST_BYTES,
    ) -> None:
        if type(max_manifest_bytes) is not int:
            raise TypeError("max_manifest_bytes must be a positive integer")
        if max_manifest_bytes <= 0:
            raise ValueError("max_manifest_bytes must be a positive integer")
        self.root = Path(root)
        self.manifest_path = self.root / "manifest.json"
        self.max_manifest_bytes = max_manifest_bytes

    def save(self, bundle: SnapshotBundle) -> None:
        payload = json.dumps(bundle.to_dict(), sort_keys=True, indent=2).encode("utf-8")
        if len(payload) > self.max_manifest_bytes:
            raise ValueError("snapshot_manifest_too_large")
        self.root.mkdir(parents=True, exist_ok=True)
        temp_id = uuid.uuid4().hex
        tmp = self.root / f".manifest.{temp_id}.json"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        fd = -1
        try:
            fd = os.open(tmp, flags, 0o600)
            with os.fdopen(fd, "wb", closefd=True) as handle:
                fd = -1
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.manifest_path)
        finally:
            if fd >= 0:
                os.close(fd)
            tmp.unlink(missing_ok=True)

    def load(
        self, *, expected_identity: SnapshotIdentity | None = None
    ) -> "SnapshotBundle | None":
        try:
            with open_package_artifact(
                package_path=self.root,
                artifact_ref="manifest.json",
                error_prefix="snapshot",
                artifact_name="manifest",
                max_bytes=self.max_manifest_bytes,
            ) as (handle, _manifest_path, _source_stat):
                payload = read_open_artifact(
                    handle,
                    max_bytes=self.max_manifest_bytes,
                    error_prefix="snapshot",
                    artifact_name="manifest",
                )
            raw = json.loads(payload.decode("utf-8"))
        except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
            return None
        if not isinstance(raw, dict):
            return None
        bundle = SnapshotBundle.from_dict(raw)
        if bundle is None:
            return None
        if expected_identity is not None and bundle.identity != expected_identity:
            return None
        return bundle
