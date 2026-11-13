"""Strict, wheel-local evidence receipts for product model admission."""

from __future__ import annotations

import hashlib
from importlib import resources
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
from typing import Any

from .producer_pathsets import RECEIPT_PATHSETS, ProducerPathset


_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_OID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_COMMIT_HEADER_KEY = re.compile(rb"^[a-z][a-z0-9-]*$")
_DECIMAL_SIZE = re.compile(rb"^(?:0|[1-9][0-9]*)$")
_RECORD_KIND = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_V2_BOUNDARY_ALGORITHM = "sha256-canonical-git-blobs-v1"
_V2_RUN_KINDS = (
    "candidate-four-gate",
    "postpromotion-public-product-path",
)
_PUBLIC_GATE = "release-quality-v1 W1-W6"
_PUBLIC_ENVIRONMENT_TOOLING_PACKAGES = frozenset({"pip", "setuptools"})
_PUBLIC_WORKFLOW = (
    "doctor_empty_cache",
    "pull_recommended",
    "doctor_after_pull",
    "run_prompt_json",
    "serve_start",
    "service_health",
    "service_generate",
    "serve_sigint",
)
_V2_REPORT_HEADINGS = {
    "postpromotion-public-product-path": (
        "# Installed-wheel W6 — generated from the gate run"
    ),
}
_MAX_BATCH_HEADER_BYTES = 128
_MAX_COMMIT_OBJECT_BYTES = 8 * 1024 * 1024
_MAX_COMMIT_PARENTS = 4096
_MAX_COMMIT_GRAPH_OBJECTS = 100_000
_MAX_MACHINE_RESULT_BYTES = 4 * 1024 * 1024
_MAX_RENDERED_RESULT_BYTES = 4 * 1024 * 1024
_PROCESS_WAIT_SECONDS = 1.0
_SAFE_GIT_ENVIRONMENT = {
    "GIT_NO_LAZY_FETCH": "1",
    "GIT_NO_REPLACE_OBJECTS": "1",
    "GIT_TERMINAL_PROMPT": "0",
}
_V1_GATES = {
    "package_graph_ir": "pass",
    "numerical_parity": "pass",
    "declared_capability": "pass",
    "installed_product_path": "pass",
}
_LEGACY_DREAM_DESCRIPTOR = (
    "dream_v0_instruct_7b",
    "Dream-org/Dream-v0-Instruct-7B",
    "05334cb9faaf763692dcf9d8737c642be2b2a6ae",
    "0cb9c169da557bc050f90da9f3107cbecb94d9e1b5747b2e735a5a5ff4dce780",
)
_LEGACY_DREAM_RECORDS = frozenset(
    {
        (
            "numerical-parity",
            "results/2026-06-05-bf16-precision.md",
            "7dc57ccd77e0a3ebd98ce72e2619f2aa80b83fdf6cde6feeb46f710868cb42f7",
        ),
        (
            "installed-product-path",
            "results/data/2026-08-11-w6-evidence.md",
            "6b8c17dd8f87e82f362e4ede9e7fae277b80baa7b14417e0e31edc76233c8e4f",
        ),
        (
            "cold-network-product-path",
            "results/2026-08-13-cold-network-acquisition.md",
            "dd205a1cd4fea81bf478e3185d4acdbc318444a22277d891b7414f06930251e4",
        ),
    }
)
_LEGACY_DREAM_LIMITATIONS = frozenset(
    {
        "not_model_coverage_v1_four_gate_replay",
        "evidence_spans_multiple_fixed_code_shas",
        "numerical_record_uses_d2f_merged_dream_weights",
    }
)


class _DuplicateKeyError(ValueError):
    pass


class _InvalidConstantError(ValueError):
    pass


def _is_utf8_encodable(value: str) -> bool:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _package_resource_bytes(relative: str) -> bytes:
    return resources.files(__package__).joinpath(relative).read_bytes()


def _resource_bytes(relative: str) -> bytes:
    return _package_resource_bytes(relative)


def _validate_resource_path(relative: object) -> None:
    if type(relative) is not str or not relative:
        raise ValueError("admission_evidence_resource_invalid")
    if (
        "\x00" in relative
        or "\\" in relative
        or ":" in relative
        or not _is_utf8_encodable(relative)
    ):
        raise ValueError("admission_evidence_resource_invalid")
    path = PurePosixPath(relative)
    if (
        path.is_absolute()
        or path.parts[:2] != ("data", "admissions")
        or len(path.parts) != 3
        or path.name in {"", ".", ".."}
        or path.suffix != ".json"
    ):
        raise ValueError("admission_evidence_resource_invalid")


def validate_admission_pointer(metadata: Any) -> None:
    resource = metadata.evidence_resource
    digest = metadata.evidence_digest
    if (resource is None) != (digest is None):
        raise ValueError(
            f"admission_evidence_pointer_incomplete: {metadata.key}"
        )
    if resource is None:
        return
    _validate_resource_path(resource)
    if type(digest) is not str or not _SHA256.fullmatch(digest):
        raise ValueError(f"admission_evidence_digest_invalid: {metadata.key}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(key)
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise _InvalidConstantError(value)


def _parse_json(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except _DuplicateKeyError as exc:
        raise ValueError(
            f"admission_evidence_json_duplicate_key: {exc.args[0]}"
        ) from None
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        _InvalidConstantError,
        RecursionError,
    ):
        raise ValueError("admission_evidence_json_invalid") from None
    if type(value) is not dict:
        raise ValueError("admission_evidence_schema_invalid")
    return value


def _require_exact_keys(
    value: object, expected: set[str], *, error: str
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        raise ValueError(error)
    return value


def _safe_result_path(value: object) -> bool:
    if type(value) is not str or not value:
        return False
    if (
        "\x00" in value
        or "\\" in value
        or ":" in value
        or not _is_utf8_encodable(value)
    ):
        return False
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and len(path.parts) >= 2
        and path.parts[0] == "results"
        and ".." not in path.parts
        and path.suffix == ".md"
    )


def _safe_machine_result_path(value: object) -> bool:
    if type(value) is not str or not value:
        return False
    if (
        "\x00" in value
        or "\\" in value
        or ":" in value
        or not _is_utf8_encodable(value)
    ):
        return False
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and len(path.parts) >= 3
        and path.parts[:2] == ("results", "data")
        and ".." not in path.parts
        and path.suffix == ".json"
    )


def _expected_profiles(descriptor: Any) -> list[dict[str, str]]:
    return [
        {
            "backend": profile.backend,
            "precision": profile.precision,
            "minimum_device": profile.minimum_device,
        }
        for profile in descriptor.admitted_backends
    ]


def _validate_common(
    receipt: dict[str, Any], descriptor: Any, *, expected_version: int
) -> None:
    schema_version = receipt.get("schema_version")
    if type(schema_version) is not int:
        raise ValueError("admission_evidence_schema_invalid")
    if schema_version != expected_version:
        raise ValueError("admission_evidence_schema_unsupported")
    if (
        receipt.get("candidate_key") != descriptor.key
        or receipt.get("model_id") != descriptor.model_id
        or receipt.get("revision") != descriptor.revision
    ):
        raise ValueError("admission_evidence_identity_mismatch")
    if receipt.get("descriptor_digest") != descriptor.descriptor_digest:
        raise ValueError("admission_evidence_descriptor_digest_mismatch")
    if receipt.get("profiles") != _expected_profiles(descriptor):
        raise ValueError("admission_evidence_profiles_mismatch")


def _validate_v1(receipt: dict[str, Any], descriptor: Any) -> None:
    _require_exact_keys(
        receipt,
        {
            "schema_version",
            "evidence_kind",
            "candidate_key",
            "model_id",
            "revision",
            "descriptor_digest",
            "fixed_code_sha",
            "verdict",
            "gates",
            "profiles",
            "result_record",
        },
        error="admission_evidence_schema_invalid",
    )
    _validate_common(receipt, descriptor, expected_version=1)
    fixed_code_sha = receipt["fixed_code_sha"]
    if type(fixed_code_sha) is not str or not _GIT_OID.fullmatch(fixed_code_sha):
        raise ValueError("admission_evidence_fixed_code_sha_invalid")
    if receipt["verdict"] != "pass":
        raise ValueError("admission_evidence_verdict_not_pass")
    if receipt["gates"] != _V1_GATES:
        raise ValueError("admission_evidence_gates_mismatch")
    record = _require_exact_keys(
        receipt["result_record"],
        {"path", "sha256"},
        error="admission_evidence_result_record_invalid",
    )
    if (
        not _safe_result_path(record["path"])
        or type(record["sha256"]) is not str
        or not _SHA256.fullmatch(record["sha256"])
    ):
        raise ValueError("admission_evidence_result_record_invalid")


def _canonical_profiles(
    profiles: list[dict[str, str]],
) -> tuple[tuple[tuple[str, str], ...], ...]:
    return tuple(tuple(sorted(profile.items())) for profile in profiles)


def _validate_v2_run(
    raw_run: object,
    *,
    descriptor: Any,
    profiles: list[dict[str, str]],
    expected_kind: str,
    expected_bootstrap: str | None,
) -> None:
    if (
        type(raw_run) is dict
        and "result_record" in raw_run
        and "result_bundle" not in raw_run
    ):
        raise ValueError("admission_evidence_result_bundle_invalid")
    run = _require_exact_keys(
        raw_run,
        {
            "kind",
            "producer_commit",
            "producer_boundary",
            "result_bundle",
            "executed_wheel_sha256",
            "provisional_bootstrap_candidate",
        },
        error="admission_evidence_v2_runs_invalid",
    )
    if run["kind"] != expected_kind:
        raise ValueError("admission_evidence_v2_runs_invalid")
    producer_commit = run["producer_commit"]
    if type(producer_commit) is not str or not _GIT_OID.fullmatch(producer_commit):
        raise ValueError("admission_evidence_producer_commit_invalid")
    wheel_digest = run["executed_wheel_sha256"]
    if type(wheel_digest) is not str or not _SHA256.fullmatch(wheel_digest):
        raise ValueError("admission_evidence_wheel_digest_invalid")
    if run["provisional_bootstrap_candidate"] != expected_bootstrap:
        raise ValueError("admission_evidence_v2_runs_invalid")

    boundary = _require_exact_keys(
        run["producer_boundary"],
        {"algorithm", "pathset_id", "entry_count", "sha256"},
        error="admission_evidence_producer_boundary_invalid",
    )
    if boundary["algorithm"] != _V2_BOUNDARY_ALGORITHM:
        raise ValueError("admission_evidence_boundary_algorithm_unsupported")
    pathset_id = boundary["pathset_id"]
    if type(pathset_id) is not str or pathset_id not in RECEIPT_PATHSETS:
        raise ValueError("admission_evidence_pathset_unknown")
    pathset = RECEIPT_PATHSETS[pathset_id]
    if (
        pathset.pathset_id != pathset_id
        or pathset.candidate_key != descriptor.key
        or pathset.run_kind != expected_kind
        or pathset.profiles != _canonical_profiles(profiles)
        or type(boundary["entry_count"]) is not int
        or boundary["entry_count"] != len(pathset.paths)
    ):
        raise ValueError("admission_evidence_pathset_mismatch")
    boundary_digest = boundary["sha256"]
    if type(boundary_digest) is not str or not _SHA256.fullmatch(boundary_digest):
        raise ValueError("admission_evidence_producer_boundary_invalid")

    bundle = _require_exact_keys(
        run["result_bundle"],
        {"machine", "report"},
        error="admission_evidence_result_bundle_invalid",
    )
    for name, path_validator in (
        ("machine", _safe_machine_result_path),
        ("report", _safe_result_path),
    ):
        record = _require_exact_keys(
            bundle[name],
            {"path", "sha256"},
            error="admission_evidence_result_bundle_invalid",
        )
        if (
            not path_validator(record["path"])
            or type(record["sha256"]) is not str
            or not _SHA256.fullmatch(record["sha256"])
        ):
            raise ValueError("admission_evidence_result_bundle_invalid")


def _validate_v2(receipt: dict[str, Any], descriptor: Any) -> None:
    _require_exact_keys(
        receipt,
        {
            "schema_version",
            "evidence_kind",
            "candidate_key",
            "model_id",
            "revision",
            "descriptor_digest",
            "profiles",
            "stage",
            "verdict",
            "runs",
        },
        error="admission_evidence_schema_invalid",
    )
    _validate_common(receipt, descriptor, expected_version=2)
    if receipt["evidence_kind"] != "model-coverage-v2":
        raise ValueError("admission_evidence_kind_unsupported")
    if receipt["verdict"] != "pass":
        raise ValueError("admission_evidence_verdict_not_pass")
    stage = receipt["stage"]
    if type(stage) is not str or stage not in {
        "provisional",
        "transition",
        "final",
    }:
        raise ValueError("admission_evidence_stage_invalid")
    expected_kinds = (
        _V2_RUN_KINDS[:1] if stage == "provisional" else _V2_RUN_KINDS
    )
    runs = receipt["runs"]
    if type(runs) is not list or len(runs) != len(expected_kinds):
        raise ValueError("admission_evidence_v2_runs_invalid")
    seen_paths: set[str] = set()
    seen_digests: set[str] = set()
    seen_producer_commits: set[str] = set()
    for run, expected_kind in zip(runs, expected_kinds, strict=True):
        expected_bootstrap = (
            descriptor.key
            if stage == "transition"
            and expected_kind == "postpromotion-public-product-path"
            else None
        )
        _validate_v2_run(
            run,
            descriptor=descriptor,
            profiles=receipt["profiles"],
            expected_kind=expected_kind,
            expected_bootstrap=expected_bootstrap,
        )
        for record in run["result_bundle"].values():
            result_path = record["path"]
            result_digest = record["sha256"]
            if result_path in seen_paths or result_digest in seen_digests:
                raise ValueError("admission_evidence_v2_runs_invalid")
            seen_paths.add(result_path)
            seen_digests.add(result_digest)
        producer_commit = run["producer_commit"]
        if producer_commit in seen_producer_commits:
            raise ValueError("admission_evidence_v2_runs_invalid")
        seen_producer_commits.add(producer_commit)


def _validate_legacy(receipt: dict[str, Any], descriptor: Any) -> None:
    descriptor_identity = (
        descriptor.key,
        descriptor.model_id,
        descriptor.revision,
        descriptor.descriptor_digest,
    )
    if descriptor_identity != _LEGACY_DREAM_DESCRIPTOR:
        raise ValueError("admission_evidence_legacy_descriptor_not_allowed")
    _require_exact_keys(
        receipt,
        {
            "schema_version",
            "evidence_kind",
            "candidate_key",
            "model_id",
            "revision",
            "descriptor_digest",
            "profiles",
            "evidence_records",
            "limitations",
        },
        error="admission_evidence_schema_invalid",
    )
    _validate_common(receipt, descriptor, expected_version=1)

    records = receipt["evidence_records"]
    if type(records) is not list or len(records) != len(_LEGACY_DREAM_RECORDS):
        raise ValueError("admission_evidence_legacy_records_invalid")
    kinds: set[str] = set()
    paths: set[str] = set()
    bindings: set[tuple[str, str, str]] = set()
    for raw_record in records:
        record = _require_exact_keys(
            raw_record,
            {"kind", "path", "sha256"},
            error="admission_evidence_legacy_records_invalid",
        )
        kind = record["kind"]
        path = record["path"]
        digest = record["sha256"]
        if (
            type(kind) is not str
            or not _RECORD_KIND.fullmatch(kind)
            or not _safe_result_path(path)
            or type(digest) is not str
            or not _SHA256.fullmatch(digest)
            or kind in kinds
            or path in paths
        ):
            raise ValueError("admission_evidence_legacy_records_invalid")
        kinds.add(kind)
        paths.add(path)
        bindings.add((kind, path, digest))
    if bindings != _LEGACY_DREAM_RECORDS:
        raise ValueError("admission_evidence_legacy_records_invalid")

    limitations = receipt["limitations"]
    if (
        type(limitations) is not list
        or any(type(item) is not str or not item for item in limitations)
        or len(limitations) != len(set(limitations))
        or set(limitations) != _LEGACY_DREAM_LIMITATIONS
    ):
        raise ValueError("admission_evidence_legacy_limitations_invalid")


def _resolve_admission_metadata(descriptor: Any, metadata: Any | None) -> Any:
    if metadata is None:
        metadata = getattr(descriptor, "admission_metadata", None)
    if metadata is None:
        # Lazy import keeps the producer descriptor module independent from the
        # admission control plane while preserving the public one-argument API.
        from .admission_catalog import admission_metadata_for_key

        metadata = admission_metadata_for_key(descriptor.key)
    if metadata.key != descriptor.key:
        raise ValueError("admission_evidence_identity_mismatch")
    return metadata


def load_admission_evidence(
    descriptor: Any, *, metadata: Any | None = None
) -> dict[str, Any]:
    """Load and validate the descriptor-bound, wheel-local admission receipt."""
    metadata = _resolve_admission_metadata(descriptor, metadata)
    validate_admission_pointer(metadata)
    resource = metadata.evidence_resource
    expected_digest = metadata.evidence_digest
    if resource is None or expected_digest is None:
        raise ValueError(f"admission_evidence_missing: {descriptor.key}")
    try:
        raw = _resource_bytes(resource)
    except (OSError, TypeError, ValueError, AttributeError):
        raise ValueError(f"admission_evidence_resource_unreadable: {resource}") from None
    actual_digest = hashlib.sha256(raw).hexdigest()
    if actual_digest != expected_digest:
        raise ValueError(
            "admission_evidence_digest_mismatch: "
            f"{resource}: expected={expected_digest} actual={actual_digest}"
        )
    receipt = _parse_json(raw)
    kind = receipt.get("evidence_kind")
    if kind == "model-coverage-v1":
        _validate_v1(receipt, descriptor)
    elif kind == "model-coverage-v2":
        _validate_v2(receipt, descriptor)
    elif kind == "legacy-product-migration-v1":
        _validate_legacy(receipt, descriptor)
    else:
        raise ValueError("admission_evidence_kind_unsupported")
    return receipt


def _repository_git_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("GIT_")
    }
    environment.update(_SAFE_GIT_ENVIRONMENT)
    return environment


def _repository_git_executable(environment: dict[str, str]) -> str:
    try:
        git = shutil.which("git", path=environment.get("PATH"))
    except (OSError, ValueError, UnicodeError):
        raise ValueError("admission_repository_invalid") from None
    if git is None:
        raise ValueError("admission_repository_invalid")
    return git


def _repository_git_bytes(
    repository: Path, *args: str
) -> subprocess.CompletedProcess[bytes]:
    environment = _repository_git_environment()
    git = _repository_git_executable(environment)
    try:
        return subprocess.run(  # noqa: S603 - argv only; paths are validated.
            [git, "--no-replace-objects", "-C", str(repository), *args],
            check=False,
            capture_output=True,
            env=environment,
        )
    except (OSError, ValueError, UnicodeError, subprocess.SubprocessError):
        raise ValueError("admission_repository_invalid") from None


def _start_repository_git_process(
    repository: Path, *args: str
) -> subprocess.Popen[bytes]:
    environment = _repository_git_environment()
    git = _repository_git_executable(environment)
    try:
        return subprocess.Popen(  # noqa: S603 - argv only; paths are validated.
            [git, "--no-replace-objects", "-C", str(repository), *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=environment,
        )
    except (OSError, ValueError, UnicodeError, subprocess.SubprocessError):
        raise ValueError("admission_repository_invalid") from None


def _repository_git(repository: Path, *args: str) -> subprocess.CompletedProcess[str]:
    raw = _repository_git_bytes(repository, *args)
    try:
        stdout = raw.stdout.decode("utf-8")
        stderr = raw.stderr.decode("utf-8")
    except UnicodeError:
        raise ValueError("admission_repository_invalid") from None
    return subprocess.CompletedProcess(raw.args, raw.returncode, stdout, stderr)


def _resolve_repository_root(repository_root: str | Path) -> Path:
    try:
        raw = os.fspath(repository_root)
    except (TypeError, ValueError, OSError):
        raise ValueError("admission_repository_invalid") from None
    if (
        type(raw) is not str
        or not raw
        or any(ord(character) < 32 for character in raw)
        or not _is_utf8_encodable(raw)
    ):
        raise ValueError("admission_repository_invalid")
    try:
        return Path(raw).resolve()
    except (OSError, ValueError, RuntimeError, UnicodeError):
        raise ValueError("admission_repository_invalid") from None


def _repository_head_oid(repository: Path) -> str:
    head = _repository_git(repository, "rev-parse", "--verify", "HEAD^{commit}")
    oid = head.stdout.strip()
    if head.returncode != 0 or not _GIT_OID.fullmatch(oid):
        raise ValueError("admission_repository_invalid")
    return oid


def _is_git_oid_bytes(value: bytes, oid_length: int) -> bool:
    return (
        oid_length in {40, 64}
        and len(value) == oid_length
        and all(character in b"0123456789abcdef" for character in value)
    )


def _parse_commit_parent_oids(raw: bytes, oid_length: int) -> tuple[str, ...]:
    """Parse parent object IDs from canonical raw Git commit headers."""
    if type(raw) is not bytes or type(oid_length) is not int:
        raise ValueError("admission_repository_commit_invalid")
    if len(raw) > _MAX_COMMIT_OBJECT_BYTES:
        raise ValueError("admission_repository_commit_too_large")
    header, separator, _message = raw.partition(b"\n\n")
    if not separator or not header or b"\x00" in header or b"\r" in header:
        raise ValueError("admission_repository_commit_invalid")

    lines = header.split(b"\n")
    tree_prefix = b"tree "
    if not lines or not lines[0].startswith(tree_prefix):
        raise ValueError("admission_repository_commit_invalid")
    if not _is_git_oid_bytes(lines[0][len(tree_prefix) :], oid_length):
        raise ValueError("admission_repository_commit_invalid")

    index = 1
    parents: list[str] = []
    seen_parents: set[str] = set()
    parent_prefix = b"parent "
    while index < len(lines) and lines[index].startswith(parent_prefix):
        if len(parents) >= _MAX_COMMIT_PARENTS:
            raise ValueError("admission_repository_commit_invalid")
        parent = lines[index][len(parent_prefix) :]
        if not _is_git_oid_bytes(parent, oid_length):
            raise ValueError("admission_repository_commit_invalid")
        decoded_parent = parent.decode("ascii")
        if decoded_parent in seen_parents:
            raise ValueError("admission_repository_commit_invalid")
        parents.append(decoded_parent)
        seen_parents.add(decoded_parent)
        index += 1

    for required_key in (b"author", b"committer"):
        if index >= len(lines):
            raise ValueError("admission_repository_commit_invalid")
        key, space, value = lines[index].partition(b" ")
        if key != required_key or not space or not value:
            raise ValueError("admission_repository_commit_invalid")
        index += 1

    can_continue = False
    for line in lines[index:]:
        if line.startswith(b" "):
            if not can_continue:
                raise ValueError("admission_repository_commit_invalid")
            continue
        key, space, value = line.partition(b" ")
        if (
            not space
            or not value
            or not _COMMIT_HEADER_KEY.fullmatch(key)
            or key in {b"tree", b"parent", b"author", b"committer"}
        ):
            raise ValueError("admission_repository_commit_invalid")
        can_continue = True

    return tuple(parents)


def _finish_repository_git_process(process: subprocess.Popen[bytes]) -> bool:
    clean = True
    for pipe_name in ("stdin", "stdout"):
        pipe = getattr(process, pipe_name, None)
        if pipe is None:
            clean = False
            continue
        try:
            pipe.close()
        except (OSError, ValueError):
            clean = False

    try:
        returncode = process.wait(timeout=_PROCESS_WAIT_SECONDS)
    except subprocess.TimeoutExpired:
        returncode = None
    except (OSError, ValueError, subprocess.SubprocessError):
        clean = False
        returncode = None
    if returncode is not None:
        return clean and returncode == 0

    try:
        process.terminate()
    except (OSError, ValueError, subprocess.SubprocessError):
        clean = False
    try:
        returncode = process.wait(timeout=_PROCESS_WAIT_SECONDS)
    except subprocess.TimeoutExpired:
        returncode = None
    except (OSError, ValueError, subprocess.SubprocessError):
        clean = False
        returncode = None
    if returncode is not None:
        return clean and returncode == 0

    try:
        process.kill()
    except (OSError, ValueError, subprocess.SubprocessError):
        clean = False
    try:
        returncode = process.wait(timeout=_PROCESS_WAIT_SECONDS)
    except (OSError, ValueError, subprocess.SubprocessError):
        return False
    return clean and returncode == 0


class _GitCommitBatch:
    def __init__(self, repository: Path, oid_length: int) -> None:
        if type(oid_length) is not int or oid_length not in {40, 64}:
            raise ValueError("admission_repository_invalid")
        self._oid_length = oid_length
        self._process = _start_repository_git_process(
            repository, "cat-file", "--batch"
        )
        if any(
            getattr(self._process, pipe_name, None) is None
            for pipe_name in ("stdin", "stdout")
        ):
            _finish_repository_git_process(self._process)
            raise ValueError("admission_repository_invalid")
        self._closed = False

    def __enter__(self) -> _GitCommitBatch:
        return self

    def __exit__(self, exc_type, _exc, _traceback) -> bool:
        clean = self.close()
        if exc_type is None and not clean:
            raise ValueError("admission_repository_invalid")
        return False

    def close(self) -> bool:
        if self._closed:
            return True
        self._closed = True
        return _finish_repository_git_process(self._process)

    def _readline(self) -> bytes:
        try:
            line = self._process.stdout.readline(_MAX_BATCH_HEADER_BYTES + 1)
        except (OSError, ValueError):
            raise ValueError("admission_repository_batch_invalid") from None
        if (
            type(line) is not bytes
            or len(line) > _MAX_BATCH_HEADER_BYTES
            or not line.endswith(b"\n")
        ):
            raise ValueError("admission_repository_batch_invalid")
        return line

    def _read_exact(self, size: int) -> bytes:
        remaining = size
        chunks: list[bytes] = []
        while remaining:
            try:
                chunk = self._process.stdout.read(remaining)
            except (OSError, ValueError):
                raise ValueError("admission_repository_batch_invalid") from None
            if type(chunk) is not bytes or not chunk or len(chunk) > remaining:
                raise ValueError("admission_repository_batch_invalid")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def read_commit_parents(
        self, oid: str, *, missing_error: str
    ) -> tuple[str, ...]:
        if (
            self._closed
            or type(oid) is not str
            or len(oid) != self._oid_length
            or not _GIT_OID.fullmatch(oid)
        ):
            raise ValueError(missing_error)
        requested = oid.encode("ascii")
        try:
            self._process.stdin.write(requested + b"\n")
            self._process.stdin.flush()
        except (OSError, ValueError):
            raise ValueError("admission_repository_batch_invalid") from None

        response = self._readline()
        if response == requested + b" missing\n":
            raise ValueError(missing_error)
        fields = response[:-1].split(b" ")
        if len(fields) != 3:
            raise ValueError("admission_repository_batch_invalid")
        canonical_oid, object_type, raw_size = fields
        if (
            canonical_oid != requested
            or not _is_git_oid_bytes(canonical_oid, self._oid_length)
        ):
            raise ValueError("admission_repository_batch_invalid")
        if object_type != b"commit":
            if object_type in {b"blob", b"tree", b"tag"}:
                raise ValueError(missing_error)
            raise ValueError("admission_repository_batch_invalid")
        if not _DECIMAL_SIZE.fullmatch(raw_size):
            raise ValueError("admission_repository_batch_invalid")
        size = int(raw_size)
        if size > _MAX_COMMIT_OBJECT_BYTES:
            raise ValueError("admission_repository_commit_too_large")
        raw = self._read_exact(size)
        try:
            separator = self._process.stdout.read(1)
        except (OSError, ValueError):
            raise ValueError("admission_repository_batch_invalid") from None
        if separator != b"\n":
            raise ValueError("admission_repository_batch_invalid")
        return _parse_commit_parent_oids(raw, self._oid_length)


def _is_raw_commit_ancestor(
    repository: Path, fixed_code_sha: str, head_oid: str
) -> bool:
    oid_length = len(head_oid)
    with _GitCommitBatch(repository, oid_length) as batch:
        batch.read_commit_parents(
            fixed_code_sha,
            missing_error="admission_fixed_code_commit_missing",
        )
        if fixed_code_sha == head_oid:
            return True

        scheduled = {fixed_code_sha}
        visited = {fixed_code_sha}
        if len(scheduled) >= _MAX_COMMIT_GRAPH_OBJECTS:
            raise ValueError("admission_repository_graph_limit_exceeded")
        scheduled.add(head_oid)
        pending = [head_oid]
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            if len(visited) >= _MAX_COMMIT_GRAPH_OBJECTS:
                raise ValueError("admission_repository_graph_limit_exceeded")
            visited.add(current)
            parents = batch.read_commit_parents(
                current,
                missing_error="admission_repository_commit_invalid",
            )
            for parent in parents:
                if parent == fixed_code_sha:
                    return True
                if parent in scheduled:
                    continue
                if len(scheduled) >= _MAX_COMMIT_GRAPH_OBJECTS:
                    raise ValueError("admission_repository_graph_limit_exceeded")
                scheduled.add(parent)
                pending.append(parent)
        return False


def _reject_repository_grafts(repository: Path) -> None:
    resolved = _repository_git(
        repository, "rev-parse", "--git-path", "info/grafts"
    )
    raw = resolved.stdout.strip()
    if (
        resolved.returncode != 0
        or not raw
        or any(ord(character) < 32 for character in raw)
        or not _is_utf8_encodable(raw)
    ):
        raise ValueError("admission_repository_invalid")
    try:
        grafts = Path(raw)
        if not grafts.is_absolute():
            grafts = repository / grafts
        grafts = grafts.resolve()
        forbidden = grafts.exists() and (
            not grafts.is_file() or grafts.stat().st_size > 0
        )
    except (OSError, ValueError, RuntimeError, UnicodeError):
        raise ValueError("admission_repository_invalid") from None
    if forbidden:
        raise ValueError("admission_repository_grafts_forbidden")


def _repository_pathset_digest(
    repository: Path, commit_oid: str, pathset: ProducerPathset
) -> str:
    """Hash exact Git blobs and modes for one verifier-owned pathset."""
    entries: list[dict[str, str]] = []
    for relative in sorted(pathset.paths):
        tree = _repository_git_bytes(
            repository, "ls-tree", "-z", commit_oid, "--", relative
        )
        if tree.returncode != 0:
            raise ValueError("admission_repository_invalid")
        raw_entries = [entry for entry in tree.stdout.split(b"\x00") if entry]
        try:
            expected_path = relative.encode("utf-8")
        except UnicodeError:
            raise ValueError("admission_producer_path_invalid") from None
        if len(raw_entries) != 1 or b"\t" not in raw_entries[0]:
            raise ValueError("admission_producer_path_not_blob")
        metadata, actual_path = raw_entries[0].split(b"\t", 1)
        parts = metadata.split(b" ")
        if (
            len(parts) != 3
            or parts[0] not in {b"100644", b"100755"}
            or parts[1] != b"blob"
            or actual_path != expected_path
            or not _is_git_oid_bytes(parts[2], len(parts[2]))
        ):
            raise ValueError("admission_producer_path_not_blob")
        blob = _repository_git_bytes(
            repository, "cat-file", "blob", parts[2].decode("ascii")
        )
        if blob.returncode != 0:
            raise ValueError("admission_producer_path_not_blob")
        entries.append(
            {
                "mode": parts[0].decode("ascii"),
                "path": relative,
                "sha256": hashlib.sha256(blob.stdout).hexdigest(),
            }
        )
    canonical = json.dumps(
        entries, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _repository_result_blob_bytes(
    repository: Path,
    head_oid: str,
    relative: str,
    *,
    path_validator,
    max_bytes: int,
    too_large_error: str,
) -> bytes:
    if not path_validator(relative):
        raise ValueError("admission_result_record_missing_or_unsafe")
    tree = _repository_git_bytes(
        repository, "ls-tree", "-z", head_oid, "--", relative
    )
    if tree.returncode != 0:
        raise ValueError("admission_repository_invalid")
    entries = [entry for entry in tree.stdout.split(b"\x00") if entry]
    try:
        expected_path = relative.encode("utf-8")
    except UnicodeError:
        raise ValueError("admission_result_record_missing_or_unsafe") from None
    if len(entries) != 1 or b"\t" not in entries[0]:
        raise ValueError("admission_result_record_not_head_blob")
    metadata, head_path = entries[0].split(b"\t", 1)
    metadata_parts = metadata.split(b" ")
    if (
        len(metadata_parts) != 3
        or metadata_parts[0] not in {b"100644", b"100755"}
        or metadata_parts[1] != b"blob"
        or head_path != expected_path
        or not _is_git_oid_bytes(metadata_parts[2], len(metadata_parts[2]))
    ):
        raise ValueError("admission_result_record_not_head_blob")

    object_id = metadata_parts[2].decode("ascii")
    size = _repository_git_bytes(repository, "cat-file", "-s", object_id)
    try:
        raw_size = size.stdout.strip()
    except AttributeError:
        raise ValueError("admission_result_record_not_head_blob") from None
    if (
        size.returncode != 0
        or not _DECIMAL_SIZE.fullmatch(raw_size)
        or int(raw_size) > max_bytes
    ):
        if size.returncode == 0 and _DECIMAL_SIZE.fullmatch(raw_size):
            raise ValueError(too_large_error)
        raise ValueError("admission_result_record_not_head_blob")

    head = _repository_git_bytes(
        repository, "cat-file", "blob", object_id
    )
    if head.returncode != 0 or len(head.stdout) != int(raw_size):
        raise ValueError("admission_result_record_not_head_blob")
    return head.stdout


def _repository_result_record_bytes(
    repository: Path,
    head_oid: str,
    record: dict[str, Any],
    *,
    path_validator,
    max_bytes: int,
    too_large_error: str,
) -> bytes:
    raw = _repository_result_blob_bytes(
        repository,
        head_oid,
        record["path"],
        path_validator=path_validator,
        max_bytes=max_bytes,
        too_large_error=too_large_error,
    )
    if hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError("admission_result_record_head_digest_mismatch")
    return raw


def _validate_repository_result_record(
    repository: Path, head_oid: str, record: dict[str, Any]
) -> None:
    _repository_result_record_bytes(
        repository,
        head_oid,
        record,
        path_validator=_safe_result_path,
        max_bytes=_MAX_RENDERED_RESULT_BYTES,
        too_large_error="admission_result_record_too_large",
    )


def _canonical_machine_result_bytes(machine: dict[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                machine,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        raise ValueError("admission_machine_result_invalid") from None


def render_admission_machine_report(
    machine: dict[str, Any], run_kind: str
) -> str:
    """Render the public W6 report bound to canonical machine JSON bytes."""
    try:
        heading = _V2_REPORT_HEADINGS[run_kind]
    except KeyError:
        raise ValueError("admission_rendered_result_kind_mismatch") from None
    machine_raw = _canonical_machine_result_bytes(machine)
    digest = hashlib.sha256(machine_raw).hexdigest()
    return (
        f"{heading}\n\n"
        f"Machine SHA-256: `{digest}`\n\n"
        "```json\n"
        f"{machine_raw.decode('utf-8')}"
        "```\n"
    )


def _validate_rendered_result(
    raw: bytes, machine: dict[str, Any], run_kind: str
) -> None:
    if run_kind == "candidate-four-gate":
        from . import candidate_evidence

        try:
            expected = candidate_evidence.render_candidate_report(machine)
        except candidate_evidence.EvidenceError:
            raise ValueError("admission_rendered_result_invalid") from None
    else:
        try:
            expected = render_admission_machine_report(machine, run_kind).encode(
                "utf-8"
            )
        except UnicodeError:
            raise ValueError("admission_rendered_result_invalid") from None
    if raw != expected:
        raise ValueError("admission_rendered_result_mismatch")


def _machine_dict(value: object, keys: set[str] | None = None) -> dict[str, Any]:
    if type(value) is not dict or (keys is not None and set(value) != keys):
        raise ValueError("admission_machine_result_mismatch")
    return value


def _load_product_runtime_lock(sdk_version: str = "0.2.0") -> dict[str, Any]:
    from .candidate_evidence import EvidenceError, load_candidate_product_lock

    try:
        return load_candidate_product_lock(sdk_version)
    except EvidenceError:
        raise ValueError("admission_product_runtime_lock_invalid") from None


def _machine_sha(value: object) -> bool:
    return type(value) is str and _SHA256.fullmatch(value) is not None


def _machine_nonnegative_number(value: object) -> bool:
    return (
        type(value) in {int, float}
        and math.isfinite(value)
        and value >= 0
    )


def _validate_candidate_machine_result(
    machine: dict[str, Any],
    receipt: dict[str, Any],
    run: dict[str, Any],
    descriptor: Any,
) -> None:
    from . import candidate_evidence

    try:
        candidate_evidence.validate_candidate_machine(machine)
    except candidate_evidence.EvidenceError:
        raise ValueError("admission_machine_result_mismatch") from None
    if (
        machine["commit"] != run["producer_commit"]
        or machine["wheel_sha256"] != run["executed_wheel_sha256"]
        or machine["profile"] != receipt["profiles"][0]
        or machine["identity"]
        != {
            "candidate_key": receipt["candidate_key"],
            "model_id": receipt["model_id"],
            "revision": receipt["revision"],
            "descriptor_digest": receipt["descriptor_digest"],
            "case_sha256": descriptor.parity_case_digest,
            "oracle_lock_sha256": descriptor.oracle_lock_digest,
        }
    ):
        raise ValueError("admission_machine_result_mismatch")


def validate_installed_reply(value: object) -> None:
    """Validate the complete reply exposed by installed MLX generation."""
    result = _machine_dict(value, {"text", "debug_metrics"})
    metrics = _machine_dict(result["debug_metrics"])
    if (
        type(result["text"]) is not str or not result["text"].strip()
        or type(metrics.get("mlx_remaining_mask_count")) is not int
        or metrics["mlx_remaining_mask_count"] != 0
        or metrics.get("mlx_finish_reason") not in {"stop", "length"}
    ):
        raise ValueError("ordinary_reply_incomplete")


def validate_installed_user_paths(
    value: object, *, backend: str, model_id: str, package_path: str
) -> None:
    """Require resolved ordinary replies and a real two-process chat restart."""
    acceptance = _machine_dict(value, {"generation", "chat"})
    samples = acceptance["generation"]
    if type(samples) is not list or len(samples) < 2:
        raise ValueError("ordinary_prompt_suite_missing")
    prompts = []
    for item in samples:
        sample = _machine_dict(item, {"prompt", "command", "wall_seconds", "result"})
        prompt = sample["prompt"]
        if type(prompt) is not str or not prompt.strip():
            raise ValueError("ordinary_prompt_invalid")
        prompts.append(prompt)
        command = sample["command"]
        if (
            type(command) is not list
            or not all(type(part) is str and part for part in command)
            or not _machine_nonnegative_number(sample["wall_seconds"])
        ):
            raise ValueError("ordinary_prompt_command_invalid")
        argv = command[:-1] if command[-1:] == ["--trust-remote-code"] else command
        if (
            len(argv) != 11
            or argv[1:] != [
                "generate", "--package", package_path, "--backend", backend,
                "--prompt", prompt, "--max-tokens", argv[9], "--json",
            ]
            or not argv[9].isdigit() or int(argv[9]) <= 0
        ):
            raise ValueError("ordinary_prompt_command_invalid")
        validate_installed_reply(sample["result"])
    if len(set(prompts)) != len(prompts):
        raise ValueError("ordinary_prompts_not_distinct")
    chat = _machine_dict(acceptance["chat"], {"turns", "session"})
    turns = chat["turns"]
    session = _machine_dict(chat["session"])
    if (
        type(turns) is not list or len(turns) != 2
        or type(session.get("revision")) is not int or session["revision"] != 2
        or type(session.get("session_id")) is not str or not session["session_id"]
    ):
        raise ValueError("chat_restart_missing")
    for turn, prompt in zip(turns, ("hello", "Repeat my previous message verbatim."), strict=True):
        if (
            type(turn) is not dict or type(turn.get("text")) is not str
            or not turn["text"].strip()
            or not _machine_nonnegative_number(turn.get("wall_seconds"))
            or type(turn.get("command")) is not list
            or not all(type(part) is str and part for part in turn["command"])
        ):
            raise ValueError("chat_turn_invalid")
        command = turn["command"]
        argv = command[:-1] if command[-1:] == ["--trust-remote-code"] else command
        if (
            len(argv) != 12
            or argv[1:] != [
                "chat", model_id, "--offline", "--backend", backend,
                "--session", session["session_id"], "--once", prompt,
                "--max-tokens", argv[11],
            ]
            or not argv[11].isdigit() or int(argv[11]) <= 0
        ):
            raise ValueError("chat_turn_invalid")
    if "hello" not in turns[1]["text"].lower():
        raise ValueError("chat_history_not_observed")


def _public_step(value: object, expected_name: str) -> dict[str, Any]:
    step = _machine_dict(value)
    if (
        step.get("step") != expected_name
        or type(step.get("command")) is not list
        or not all(type(part) is str and part for part in step["command"])
        or not _machine_nonnegative_number(step.get("wall_seconds"))
    ):
        raise ValueError("admission_machine_result_mismatch")
    return step


def _validate_private_runtime_machine(
    value: object,
    *,
    commit: str,
    python_version: str,
    system: str,
    machine_name: str,
    mlx_version: str,
) -> str:
    artifact = _machine_dict(
        value,
        {"wheel", "wheel_sha256", "receipt_sha256", "receipt", "probe"},
    )
    receipt = _machine_dict(
        artifact["receipt"],
        {"schema_version", "kind", "source", "build", "wheel"},
    )
    source = _machine_dict(
        receipt["source"], {"commit", "tree", "paths", "archive_sha256"}
    )
    build = _machine_dict(
        receipt["build"],
        {
            "python",
            "platform",
            "mlx_version",
            "command",
            "dependency_resolution",
            "source_date_epoch",
        },
    )
    build_python = _machine_dict(
        build["python"], {"implementation", "version"}
    )
    build_platform = _machine_dict(
        build["platform"], {"system", "machine"}
    )
    wheel = _machine_dict(
        receipt["wheel"],
        {
            "filename",
            "distribution",
            "version",
            "sha256",
            "size_bytes",
            "member_count",
            "logical_bytes",
            "tree_sha256",
            "requires_dist",
        },
    )
    probe = _machine_dict(
        artifact["probe"],
        {
            "distribution",
            "version",
            "runtime_module",
            "device_module",
            "manager",
            "pending_commit_receipts",
        },
    )
    try:
        canonical_receipt = (
            json.dumps(
                receipt,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
            + b"\n"
        )
    except (TypeError, ValueError, UnicodeEncodeError):
        raise ValueError("admission_machine_result_mismatch") from None
    if (
        receipt["schema_version"] != "1.0"
        or receipt["kind"] != "diffusorrt-private-runtime-wheel"
        or source["commit"] != commit
        or type(source["tree"]) is not str
        or re.fullmatch(r"[0-9a-f]{40}", source["tree"]) is None
        or source["paths"]
        != [
            "include",
            "native/binding",
            "native/core",
            "native/cuda",
            "python/private",
        ]
        or not _machine_sha(source["archive_sha256"])
        or build_python
        != {"implementation": "CPython", "version": python_version}
        or build_platform != {"system": system, "machine": machine_name}
        or build["mlx_version"] != mlx_version
        or build["dependency_resolution"] != "none"
        or build["source_date_epoch"] != "315532800"
        or build["command"]
        != [
            "python",
            "-m",
            "pip",
            "wheel",
            "<COMMITTED_PRIVATE_SOURCE>",
            "--no-deps",
            "--no-build-isolation",
            "--no-cache-dir",
        ]
        or wheel["filename"] != artifact["wheel"]
        or wheel["distribution"] != "diffusorrt-runtime"
        or wheel["version"] not in ("0.1.0", "0.1.1")
        or wheel["sha256"] != artifact["wheel_sha256"]
        or not _machine_sha(wheel["sha256"])
        or not _machine_sha(wheel["tree_sha256"])
        or type(wheel["size_bytes"]) is not int
        or wheel["size_bytes"] <= 0
        or type(wheel["member_count"]) is not int
        or wheel["member_count"] <= 0
        or type(wheel["logical_bytes"]) is not int
        or wheel["logical_bytes"] <= 0
        or type(wheel["requires_dist"]) is not list
        or f"mlx=={mlx_version}" not in wheel["requires_dist"]
        or artifact["receipt_sha256"]
        != hashlib.sha256(canonical_receipt).hexdigest()
        or probe["distribution"] != "diffusorrt-runtime"
        or probe["version"] != wheel["version"]
        or probe["manager"] != "native.core"
        or probe["pending_commit_receipts"] != 0
        or type(probe["runtime_module"]) is not str
        or "site-packages" not in probe["runtime_module"]
        or type(probe["device_module"]) is not str
        or "site-packages" not in probe["device_module"]
    ):
        raise ValueError("admission_machine_result_mismatch")
    return wheel["version"]


def _validate_public_steps(
    steps: object,
    *,
    model_id: str,
    revision: str,
    identity: dict[str, Any],
    backend: str,
    profile: str,
) -> None:
    if type(steps) is not list or len(steps) != len(_PUBLIC_WORKFLOW):
        raise ValueError("admission_machine_result_mismatch")
    bound = [
        _public_step(step, name)
        for step, name in zip(steps, _PUBLIC_WORKFLOW, strict=True)
    ]
    doctor_empty, pull, doctor_ready, run, serve, health, generate, sigint = bound
    if doctor_empty["command"] != [
        "diffusorrt",
        "doctor",
        model_id,
        "--offline",
        "--json",
    ]:
        raise ValueError("admission_machine_result_mismatch")
    if (
        type(doctor_empty.get("exit_code")) is not int
        or doctor_empty["exit_code"] == 0
        or doctor_empty.get("package_cache_status") != "missing"
        or doctor_empty.get("model_id") != model_id
        or doctor_empty.get("revision") != revision
        or doctor_empty.get("trust_required") is not False
        or doctor_empty.get("trust_satisfied") is not True
    ):
        raise ValueError("admission_machine_result_mismatch")

    pull_command = ["diffusorrt", "pull", model_id, "--offline", "--json"]
    if (
        pull["command"] != pull_command
        or pull.get("recommended_argv") != pull_command[:-1]
        or pull.get("appended_for_checking") != ["--json"]
        or pull.get("appended_to_recommendation") != ["--json"]
        or pull.get("reused_cache") is not False
        or pull.get("stages")
        != [
            "planning",
            "cache_verification",
            "acquisition",
            "conversion",
            "validation",
            "publication",
            "ready",
        ]
        or pull.get("package_identity") != identity
        or type(pull.get("package_dir")) is not str
        or not pull["package_dir"].startswith("/")
        or "\x00" in pull["package_dir"]
    ):
        raise ValueError("admission_machine_result_mismatch")

    if doctor_ready["command"] != [
        "diffusorrt",
        "doctor",
        model_id,
        "--json",
    ]:
        raise ValueError("admission_machine_result_mismatch")
    if (
        doctor_ready.get("exit_code") != 0
        or doctor_ready.get("ready") is not True
        or doctor_ready.get("package_cache_status") != "valid"
        or doctor_ready.get("package_compatibility") != "compatible"
        or doctor_ready.get("selected_backend") != backend
        or doctor_ready.get("model_id") != model_id
        or doctor_ready.get("revision") != revision
        or doctor_ready.get("trust_required") is not False
        or doctor_ready.get("trust_satisfied") is not True
    ):
        raise ValueError("admission_machine_result_mismatch")

    run_command = run["command"]
    run_result = _machine_dict(
        run.get("result"),
        {"text", "model", "backend", "max_tokens", "system"},
    )
    verified_identity = _machine_dict(
        run.get("verified_package_identity"), {"source", *identity}
    )
    execution = _machine_dict(
        run.get("execution_binding"),
        {
            "source",
            "package_path",
            "package_st_dev",
            "package_st_ino",
            "backend",
            "profile",
        },
    )
    if (
        len(run_command) != 11
        or run_command[:6]
        != [
            "diffusorrt",
            "run",
            model_id,
            "--offline",
            "--backend",
            backend,
        ]
        or run_command[6] != "--prompt"
        or not run_command[7]
        or run_command[8] != "--max-tokens"
        or not run_command[9].isdigit()
        or int(run_command[9]) <= 0
        or run_command[10] != "--json"
        or type(run.get("text")) is not str
        or not run["text"].strip()
        or run_result.get("text") != run["text"]
        or run.get("backend") != backend
        or run_result.get("backend") != backend
        or run_result.get("max_tokens") != int(run_command[9])
        or run_result.get("system") is not None
        or run_result.get("model") != pull["package_dir"]
        or verified_identity != {"source": "installed_compile_snapshot", **identity}
        or execution.get("source")
        != "run_json+verified_package_stat+service_profile"
        or execution.get("package_path") != pull["package_dir"]
        or execution.get("package_path") != run_result["model"]
        or type(execution.get("package_st_dev")) is not int
        or execution["package_st_dev"] <= 0
        or type(execution.get("package_st_ino")) is not int
        or execution["package_st_ino"] <= 0
        or execution.get("backend") != backend
        or execution.get("profile") != profile
    ):
        raise ValueError("admission_machine_result_mismatch")

    serve_command = serve["command"]
    if (
        len(serve_command) != 8
        or serve_command[:5]
        != ["diffusorrt", "serve", model_id, "--offline", "--port"]
        or not serve_command[5].isdigit()
        or not 0 < int(serve_command[5]) <= 65535
        or serve_command[6:] != ["--backend", backend]
    ):
        raise ValueError("admission_machine_result_mismatch")
    port = serve_command[5]
    if health["command"] != ["GET", f"http://127.0.0.1:{port}/health"]:
        raise ValueError("admission_machine_result_mismatch")
    if generate["command"] != [
        "POST",
        f"http://127.0.0.1:{port}/v1/generate",
    ]:
        raise ValueError("admission_machine_result_mismatch")
    if sigint["command"] != ["SIGINT", "diffusorrt", "serve"] or sigint.get(
        "exit_code"
    ) != 0:
        raise ValueError("admission_machine_result_mismatch")


def _validate_public_machine_result(
    machine: dict[str, Any],
    receipt: dict[str, Any],
    run: dict[str, Any],
    descriptor: Any,
) -> None:
    expected_profile = receipt["profiles"][0]
    try:
        admission = descriptor.resolve_backend(
            expected_profile["backend"], expected_profile["precision"]
        )
        public_profile = (
            None if admission.deployment is None else admission.deployment.profile
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        raise ValueError("admission_public_profile_missing") from None
    if public_profile is None:
        raise ValueError("admission_public_profile_missing")
    schema_version = machine.get("schema_version")
    machine_keys = {
        "schema_version",
        "gate",
        "verdict",
        "commit",
        "tree_clean",
        "allow_dirty",
        "tree_dirty_paths",
        "wheel",
        "wheel_sha256",
        "build",
        "extras",
        "environment",
        "model",
        "identity",
        "expected_identity",
        "trust_remote_code",
        "backend",
        "profile",
        "expected_execution",
        "provisional_bootstrap_candidate",
        "public_workflow",
        "cache",
        "service",
        "diffusorrt_cache_root",
        "hf_hub_offline",
        "pythonpath_unset",
        "steps",
    }
    if schema_version == "1.3":
        machine_keys.add("private_runtime")
    _machine_dict(
        machine,
        machine_keys,
    )
    backend = expected_profile["backend"]
    if (
        schema_version not in {"1.2", "1.3"}
        or machine["gate"] != _PUBLIC_GATE
        or machine["verdict"] != "pass"
        or machine["commit"] != run["producer_commit"]
        or machine["tree_clean"] is not True
        or machine["allow_dirty"] is not False
        or machine["tree_dirty_paths"] != []
        or type(machine["wheel"]) is not str
        or not machine["wheel"].endswith(".whl")
        or machine["wheel_sha256"] != run["executed_wheel_sha256"]
        or machine["extras"] != "compiler,mlx-local,serve"
        or machine["model"] != receipt["model_id"]
        or machine["trust_remote_code"] is not False
        or machine["backend"] != backend
        or machine["profile"] != public_profile
        or machine["provisional_bootstrap_candidate"]
        != run["provisional_bootstrap_candidate"]
        or machine["public_workflow"] != list(_PUBLIC_WORKFLOW)
        or machine["diffusorrt_cache_root"]
        != "<FRESH, emptied before the run>"
        or machine["hf_hub_offline"] is not True
        or machine["pythonpath_unset"] is not True
    ):
        raise ValueError("admission_machine_result_mismatch")

    identity = _machine_dict(
        machine["identity"],
        {
            "model_id",
            "revision",
            "descriptor_key",
            "descriptor_digest",
            "source_license",
            "manifest_sha256",
            "receipt_sha256",
            "model_precision",
            "admitted_backend",
        },
    )
    if (
        identity["model_id"] != receipt["model_id"]
        or identity["revision"] != receipt["revision"]
        or identity["descriptor_key"] != receipt["candidate_key"]
        or identity["descriptor_digest"] != receipt["descriptor_digest"]
        or identity["source_license"] != descriptor.source_license
        or not _machine_sha(identity["manifest_sha256"])
        or not _machine_sha(identity["receipt_sha256"])
        or identity["manifest_sha256"] == identity["receipt_sha256"]
        or identity["model_precision"] != expected_profile["precision"]
        or identity["admitted_backend"] != backend
    ):
        raise ValueError("admission_machine_result_mismatch")
    if machine["expected_identity"] != {
        "revision": receipt["revision"],
        "descriptor_digest": receipt["descriptor_digest"],
    }:
        raise ValueError("admission_machine_result_mismatch")
    if machine["expected_execution"] != {
        "device": expected_profile["minimum_device"],
        "precision": expected_profile["precision"],
        "profile": public_profile,
    }:
        raise ValueError("admission_machine_result_mismatch")

    environment = _machine_dict(
        machine["environment"],
        {
            "python",
            "system",
            "platform",
            "machine",
            "processor",
            "apple_chip",
            "hardware_model",
            "memory_bytes",
            "packages",
            "device",
        },
    )
    packages = _machine_dict(environment["packages"])
    sdk_version = packages.get("diffusor-rt")
    product_lock = _load_product_runtime_lock(sdk_version)
    locked_packages = product_lock["packages"]
    private_runtime_version = None
    if schema_version == "1.3":
        private_runtime_version = _validate_private_runtime_machine(
            machine["private_runtime"],
            commit=machine["commit"],
            python_version=environment["python"],
            system=environment["system"],
            machine_name=environment["machine"],
            mlx_version=packages.get("mlx"),
        )
    expected_packages = dict(locked_packages)
    if private_runtime_version is not None:
        expected_packages["diffusorrt-runtime"] = private_runtime_version
    extra_packages = set(packages) - set(expected_packages)
    memory_bytes = environment["memory_bytes"]
    derived_device = "_".join(
        (
            environment["system"].lower(),
            environment["machine"].lower(),
            "_".join(environment["apple_chip"].lower().split()),
            f"{round(memory_bytes / 2**30)}gb",
        )
    ) if (
        type(environment["system"]) is str
        and type(environment["machine"]) is str
        and type(environment["apple_chip"]) is str
        and type(memory_bytes) is int
    ) else None
    if (
        environment["python"] != product_lock["python"]["version"]
        or environment["system"] != product_lock["platform"]["system"]
        or type(environment["platform"]) is not str
        or not environment["platform"].strip()
        or environment["machine"] != product_lock["platform"]["machine"]
        or (
            environment["processor"] is not None
            and (
                type(environment["processor"]) is not str
                or not environment["processor"].strip()
            )
        )
        or environment["apple_chip"] != "Apple M4 Max"
        or type(environment["hardware_model"]) is not str
        or not environment["hardware_model"].strip()
        or type(memory_bytes) is not int
        or memory_bytes < 64 * 1024**3
        or derived_device != expected_profile["minimum_device"]
        or environment["device"] != expected_profile["minimum_device"]
        or any(packages.get(name) != version for name, version in expected_packages.items())
        or not extra_packages.issubset(_PUBLIC_ENVIRONMENT_TOOLING_PACKAGES)
        or any(
            type(name) is not str
            or name != re.sub(r"[-_.]+", "-", name).lower()
            or type(version) is not str
            or not version
            for name, version in packages.items()
        )
    ):
        raise ValueError("admission_machine_result_mismatch")

    from . import candidate_evidence

    try:
        build_source_sha256 = machine["build"]["receipt"]["source"][
            "archive_sha256"
        ]
        if not _machine_sha(build_source_sha256):
            raise candidate_evidence.EvidenceError("build_source_invalid")
        candidate_evidence.validate_build_receipt(
            machine["build"],
            source_sha256=build_source_sha256,
            wheel_sha256=machine["wheel_sha256"],
            python_version=environment["python"],
            expected_wheel_filename=f"diffusor_rt-{sdk_version}-py3-none-any.whl",
        )
    except (candidate_evidence.EvidenceError, KeyError, TypeError):
        raise ValueError("admission_machine_result_mismatch") from None

    cache = _machine_dict(
        machine["cache"], {"root", "fresh_before_run", "states"}
    )
    if (
        type(cache["root"]) is not str
        or not cache["root"]
        or cache["fresh_before_run"] is not True
        or cache["states"]
        != {
            "doctor_before_pull": "missing",
            "pull_reused_cache": False,
            "doctor_after_pull": "valid",
        }
    ):
        raise ValueError("admission_machine_result_mismatch")

    service = _machine_dict(
        machine["service"], {"health", "generate", "sigint"}
    )
    health = _machine_dict(service["health"])
    health_service = _machine_dict(health.get("service"))
    generated = _machine_dict(service["generate"])
    if (
        health.get("status") != "ok"
        or health_service.get("state") != "ready"
        or health_service.get("backend") != backend
        or health_service.get("profile") != public_profile
        or generated.get("status") != "ok"
        or type(generated.get("text")) is not str
        or not generated["text"].strip()
        or generated.get("backend") != backend
        or generated.get("profile") != public_profile
        or service["sigint"] != {"signal": "SIGINT", "exit_code": 0}
    ):
        raise ValueError("admission_machine_result_mismatch")
    for payload in (health, generated):
        for field, expected in (
            ("model_id", receipt["model_id"]),
            ("revision", receipt["revision"]),
            ("descriptor_digest", receipt["descriptor_digest"]),
        ):
            if field in payload and payload[field] != expected:
                raise ValueError("admission_machine_result_mismatch")

    _validate_public_steps(
        machine["steps"],
        model_id=receipt["model_id"],
        revision=receipt["revision"],
        identity=identity,
        backend=backend,
        profile=public_profile,
    )
    steps = machine["steps"]
    if sdk_version != "0.2.0":
        validate_installed_user_paths(
            steps[3].get("user_acceptance"), backend=backend,
            model_id=receipt["model_id"], package_path=steps[1]["package_dir"],
        )
    if (
        steps[5].get("response") != health
        or steps[6].get("response") != generated
    ):
        raise ValueError("admission_machine_result_mismatch")


def _validate_machine_result(
    raw: bytes,
    receipt: dict[str, Any],
    run: dict[str, Any],
    descriptor: Any,
) -> dict[str, Any]:
    try:
        if run["kind"] == "candidate-four-gate":
            from . import candidate_evidence

            machine = candidate_evidence.strict_json_loads(raw)
            _validate_candidate_machine_result(machine, receipt, run, descriptor)
            canonical = candidate_evidence.canonical_json_bytes(machine)
        elif run["kind"] == "postpromotion-public-product-path":
            machine = _parse_json(raw)
            _validate_public_machine_result(machine, receipt, run, descriptor)
            canonical = _canonical_machine_result_bytes(machine)
        else:
            raise ValueError("admission_machine_result_mismatch")
        if raw != canonical:
            raise ValueError("admission_machine_result_not_canonical")
        return machine
    except ValueError as error:
        if str(error) == "admission_machine_result_mismatch":
            raise
        raise ValueError("admission_machine_result_invalid") from None
    except (AttributeError, KeyError, OverflowError, TypeError):
        raise ValueError("admission_machine_result_invalid") from None


def _validate_repository_v2_receipt(
    descriptor: Any,
    repository: Path,
    receipt: dict[str, Any],
    evidence_commit: str,
    *,
    require_final: bool,
) -> None:
    """Validate v2 blobs and boundaries at one immutable evidence commit."""
    _validate_v2(receipt, descriptor)
    if require_final and receipt["stage"] != "final":
        raise ValueError("admission_evidence_nonfinal_not_releasable")
    for run in receipt["runs"]:
        bundle = run["result_bundle"]
        machine = _repository_result_record_bytes(
            repository,
            evidence_commit,
            bundle["machine"],
            path_validator=_safe_machine_result_path,
            max_bytes=_MAX_MACHINE_RESULT_BYTES,
            too_large_error="admission_machine_result_too_large",
        )
        report = _repository_result_record_bytes(
            repository,
            evidence_commit,
            bundle["report"],
            path_validator=_safe_result_path,
            max_bytes=_MAX_RENDERED_RESULT_BYTES,
            too_large_error="admission_result_record_too_large",
        )
        machine_document = _validate_machine_result(
            machine, receipt, run, descriptor
        )
        _validate_rendered_result(report, machine_document, run["kind"])
        producer_commit = run["producer_commit"]
        if not _is_raw_commit_ancestor(
            repository, producer_commit, evidence_commit
        ):
            raise ValueError("admission_producer_commit_not_ancestor")
        boundary = run["producer_boundary"]
        pathset = RECEIPT_PATHSETS[boundary["pathset_id"]]
        producer_digest = _repository_pathset_digest(
            repository, producer_commit, pathset
        )
        if producer_digest != boundary["sha256"]:
            raise ValueError("admission_producer_commit_digest_mismatch")
        evidence_digest = _repository_pathset_digest(
            repository, evidence_commit, pathset
        )
        if evidence_digest != boundary["sha256"]:
            raise ValueError("admission_producer_head_digest_mismatch")


def validate_repository_admission_evidence(
    descriptor: Any,
    repository_root: str | Path,
    *,
    metadata: Any | None = None,
) -> dict[str, Any]:
    """Verify receipt claims that intentionally remain outside an installed wheel."""
    repository = _resolve_repository_root(repository_root)
    top_level = _repository_git(repository, "rev-parse", "--show-toplevel")
    try:
        raw_top_level = top_level.stdout.strip()
        resolved_top_level = Path(raw_top_level).resolve()
    except (OSError, ValueError, RuntimeError, UnicodeError):
        raise ValueError("admission_repository_invalid") from None
    if (
        top_level.returncode != 0
        or not raw_top_level
        or resolved_top_level != repository
    ):
        raise ValueError("admission_repository_invalid")
    _reject_repository_grafts(repository)
    head_oid = _repository_head_oid(repository)

    metadata = _resolve_admission_metadata(descriptor, metadata)
    receipt = load_admission_evidence(descriptor, metadata=metadata)
    kind = receipt["evidence_kind"]
    if kind == "model-coverage-v1":
        _validate_repository_result_record(
            repository, head_oid, receipt["result_record"]
        )
        fixed_code_sha = receipt["fixed_code_sha"]
        if not _is_raw_commit_ancestor(repository, fixed_code_sha, head_oid):
            raise ValueError("admission_fixed_code_not_ancestor")
    elif kind == "model-coverage-v2":
        # A candidate-only provisional receipt is the terminal evidence for an
        # Experimentally-gated catalog entry.  Only Product-supported entries
        # must bind the subsequent public product-path run in a final receipt.
        require_final = str(metadata.state) == "product-supported"
        _validate_repository_v2_receipt(
            descriptor,
            repository,
            receipt,
            head_oid,
            require_final=require_final,
        )
    else:
        for record in receipt["evidence_records"]:
            _validate_repository_result_record(repository, head_oid, record)
    return receipt


__all__ = [
    "load_admission_evidence",
    "render_admission_machine_report",
    "validate_admission_pointer",
    "validate_repository_admission_evidence",
]
