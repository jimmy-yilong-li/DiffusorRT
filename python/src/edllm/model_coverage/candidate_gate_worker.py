"""Installed-wheel workers for the model-coverage Candidate gate.

Every subcommand runs inside a fresh virtual environment.  Heavy model work is
kept out of the repository-side orchestrator, and every successful invocation
emits exactly one strict JSON object on stdout.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys
import sysconfig
from typing import Any, Callable, Mapping, NamedTuple, TypeVar

try:
    import resource
except ModuleNotFoundError:  # pragma: no cover - exercised only off POSIX
    # Importable everywhere, measurable only on POSIX. A top-level
    # `import resource` made tests/model_coverage/test_candidate_gate.py fail at
    # COLLECTION on Windows; the peak-RSS read refuses at its point of use.
    resource = None  # type: ignore[assignment]

from .candidate_evidence import (
    CANDIDATE_CAPABILITY_CHECKS,
    candidate_product_lock_resource,
    canonical_json_bytes,
    expected_commands,
    load_candidate_product_lock,
)
from .producer_descriptors import (
    candidate_runtime_policy,
    descriptor_for_key,
    load_oracle_lock,
    validate_execution_package,
    validate_snapshot,
)


PROVISIONAL_BOOTSTRAP_ENV = (
    "DIFFUSORRT_INTERNAL_PROVISIONAL_BOOTSTRAP_CANDIDATE"
)
_SHA256_LENGTH = 64
_MAX_INVENTORY_ENTRIES = 100_000
_MAX_INVENTORY_FILES = 100_000
_MAX_INVENTORY_BYTES = 32 * 1024**3
_PEP503_SEPARATORS = re.compile(r"[-_.]+")
_ENVIRONMENT_TOOLING = frozenset({"pip", "setuptools"})
_T = TypeVar("_T")


class WorkerError(RuntimeError):
    """One Candidate worker could not establish its gate."""


def clean_subprocess_environment(
    additions: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return an installed-wheel environment with source/bootstrap paths absent."""
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", PROVISIONAL_BOOTSTRAP_ENV}
        and not key.startswith("PIP_")
    }
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    if additions:
        for key, value in additions.items():
            if type(key) is not str or not key or type(value) is not str:
                raise WorkerError("subprocess_environment_invalid")
            if key in {"PYTHONPATH", PROVISIONAL_BOOTSTRAP_ENV} or key.startswith(
                "PIP_"
            ):
                raise WorkerError("subprocess_environment_forbidden")
            environment[key] = value
    return environment


def _identity(descriptor) -> dict[str, str]:
    return {
        "candidate_key": descriptor.key,
        "model_id": descriptor.model_id,
        "revision": descriptor.revision,
        "descriptor_digest": descriptor.descriptor_digest,
        "case_sha256": descriptor.parity_case_digest,
        "oracle_lock_sha256": descriptor.oracle_lock_digest,
    }


def _profile(descriptor):
    if len(descriptor.admitted_backends) != 1:
        raise WorkerError("candidate_profile_ambiguous")
    return descriptor.admitted_backends[0]


def _require_sha256(value: str) -> str:
    if (
        type(value) is not str
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise WorkerError("wheel_sha256_invalid")
    return value


def _peak_rss_bytes() -> int:
    if resource is None:
        # A gate reports a measured peak or nothing; 0 would be a fabricated one.
        raise WorkerError("peak_rss_unavailable_on_this_platform")
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024
    return value


def _resource_record(descriptor) -> dict[str, int | bool]:
    peak = _peak_rss_bytes()
    if peak <= 0 or peak > descriptor.peak_rss_bytes:
        raise WorkerError(
            "resource_envelope_exceeded: "
            f"peak={peak} limit={descriptor.peak_rss_bytes}"
        )
    return {
        "peak_rss_bytes": peak,
        "peak_rss_limit_bytes": descriptor.peak_rss_bytes,
        "within_resource_envelope": True,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    descriptor: int | None = None
    nofollow = getattr(os, "O_NOFOLLOW", None)
    try:
        target = path.resolve(strict=True) if path.is_symlink() else path
        flags = os.O_RDONLY | (nofollow or 0)
        descriptor = os.open(target, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise WorkerError(f"artifact_not_regular: {path}")
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise WorkerError(f"artifact_changed: {path}")
        return digest.hexdigest()
    except WorkerError:
        raise
    except OSError as error:
        raise WorkerError(f"artifact_read_failed: {path}: {error}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _source_weights_sha256(snapshot: Path) -> str:
    """Bind the exact converter-selected safetensors layout and contents.

    Preserve the historical digest for the canonical single-file layout. For
    indexed checkpoints, hash every lexical shard name and content digest in
    the same order used by the converter.
    """
    from edllm.converter.convert_hf_masked_diffusion import (
        _safetensors_shard_paths,
    )

    try:
        planned = _safetensors_shard_paths(snapshot)
        actual = sorted(
            path
            for path in snapshot.rglob("*.safetensors")
            if path.is_file()
        )
        planned_names = [path.relative_to(snapshot).as_posix() for path in planned]
        actual_names = [path.relative_to(snapshot).as_posix() for path in actual]
    except (OSError, ValueError) as error:
        raise WorkerError(f"source_weights_layout_invalid: {error}") from None
    if planned_names != actual_names:
        raise WorkerError(
            "source_weights_layout_invalid: "
            f"planned={planned_names!r} actual={actual_names!r}"
        )
    if planned_names == ["model.safetensors"]:
        return _sha256_file(planned[0])

    digest = hashlib.sha256()
    for name, path in zip(planned_names, planned, strict=True):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _inventory(
    root: Path,
    *,
    suffix: str | None = None,
    exclude_bytecode: bool = False,
) -> tuple[int, int, str]:
    """Return count, logical bytes and a content-bound deterministic tree hash."""
    resolved_root = root.resolve(strict=True)
    snapshot_blob_root = (
        resolved_root.parent.parent / "blobs"
        if resolved_root.parent.name == "snapshots"
        else None
    )
    entries: list[Path] = []
    for path in root.rglob("*"):
        if len(entries) >= _MAX_INVENTORY_ENTRIES:
            raise WorkerError("artifact_inventory_entry_limit")
        entries.append(path)
    records: list[tuple[str, int, str]] = []
    logical_bytes = 0
    for path in sorted(entries):
        if not (path.is_file() or path.is_symlink()):
            continue
        relative_path = path.relative_to(root)
        if exclude_bytecode and (
            "__pycache__" in relative_path.parts
            or path.suffix in {".pyc", ".pyo"}
        ):
            continue
        if suffix is not None and path.suffix != suffix:
            continue
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(resolved_root)
        except ValueError:
            # Hugging Face snapshots intentionally link into sibling blobs/.
            try:
                if snapshot_blob_root is None:
                    raise ValueError
                resolved.relative_to(snapshot_blob_root)
            except ValueError:
                raise WorkerError(f"artifact_escapes_root: {path}") from None
        size = resolved.stat().st_size
        if len(records) >= _MAX_INVENTORY_FILES:
            raise WorkerError("artifact_inventory_file_limit")
        if size > _MAX_INVENTORY_BYTES - logical_bytes:
            raise WorkerError("artifact_inventory_byte_limit")
        logical_bytes += size
        records.append((relative_path.as_posix(), size, _sha256_file(path)))
    if not records:
        raise WorkerError(f"artifact_inventory_empty: {root}")
    digest = hashlib.sha256()
    for relative, size, file_digest in records:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\n")
    return len(records), logical_bytes, digest.hexdigest()


def _normalize_distribution_name(value: str) -> str:
    normalized = _PEP503_SEPARATORS.sub("-", value).lower()
    if not normalized or normalized != value:
        raise WorkerError(f"environment_distribution_name_noncanonical: {value!r}")
    return normalized


def _environment_packages(lock: Mapping[str, Any]) -> dict[str, Any]:
    """Enumerate every installed distribution and bind the complete set.

    Runtime packages must equal the selected lock.  A venv's own ``pip`` is
    required and ``setuptools`` is the only optional tooling distribution;
    neither is silently folded into the runtime lock.
    """
    locked = lock.get("packages")
    if type(locked) is not dict or not locked:
        raise WorkerError("environment_lock_packages_invalid")
    expected: dict[str, str] = {}
    for raw_name, version in locked.items():
        if type(raw_name) is not str or type(version) is not str or not version:
            raise WorkerError("environment_lock_packages_invalid")
        name = _normalize_distribution_name(raw_name)
        if name in expected:
            raise WorkerError("environment_lock_distribution_name_duplicate")
        expected[name] = version

    actual: dict[str, str] = {}
    try:
        distributions = importlib.metadata.distributions()
        for distribution in distributions:
            raw_name = distribution.metadata.get("Name")
            version = distribution.version
            if (
                type(raw_name) is not str
                or not raw_name.strip()
                or type(version) is not str
                or not version.strip()
            ):
                raise WorkerError("environment_distribution_metadata_invalid")
            canonical = _PEP503_SEPARATORS.sub("-", raw_name).lower()
            if not canonical:
                raise WorkerError("environment_distribution_metadata_invalid")
            if canonical in actual:
                raise WorkerError(
                    f"environment_distribution_name_duplicate: {canonical}"
                )
            actual[canonical] = version
    except WorkerError:
        raise
    except (AttributeError, TypeError, ValueError) as error:
        raise WorkerError("environment_distribution_metadata_invalid") from error

    tooling = {
        name: actual[name]
        for name in sorted(actual.keys() & _ENVIRONMENT_TOOLING)
    }
    runtime = {
        name: actual[name]
        for name in sorted(actual.keys() - _ENVIRONMENT_TOOLING)
    }
    if set(runtime) != set(expected):
        missing = sorted(set(expected) - set(runtime))
        extra = sorted(set(runtime) - set(expected))
        raise WorkerError(
            "environment_distribution_set_mismatch: "
            f"missing={missing!r} extra={extra!r}"
        )
    if "pip" not in tooling:
        raise WorkerError("environment_tooling_pip_missing")
    mismatches = [
        f"{name}={runtime[name]}(expected {expected[name]})"
        for name in sorted(expected)
        if runtime[name] != expected[name]
    ]
    if mismatches:
        raise WorkerError("environment_package_mismatch: " + "; ".join(mismatches))
    return {
        "packages": runtime,
        "tooling_packages": tooling,
        "distribution_count": len(actual),
    }


def _directory_identity(root: Path, *, reason: str) -> tuple[int, int]:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise WorkerError(f"{reason}_nofollow_unavailable")
    descriptor: int | None = None
    try:
        descriptor = os.open(root, os.O_RDONLY | nofollow | directory)
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise WorkerError(f"{reason}_not_directory")
        return metadata.st_dev, metadata.st_ino
    except WorkerError:
        raise
    except OSError as error:
        raise WorkerError(f"{reason}_identity_failed: {error}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _tree_snapshot(
    root: Path, *, reason: str, exclude_bytecode: bool = False
) -> dict[str, Any]:
    before = _directory_identity(root, reason=reason)
    count, logical_bytes, tree_sha256 = _inventory(
        root, exclude_bytecode=exclude_bytecode
    )
    after = _directory_identity(root, reason=reason)
    if before != after:
        raise WorkerError(f"{reason}_root_changed")
    return {
        "root_dev": before[0],
        "root_ino": before[1],
        "file_count": count,
        "logical_bytes": logical_bytes,
        "tree_sha256": tree_sha256,
    }


def _package_snapshot(package: Path) -> dict[str, Any]:
    return _tree_snapshot(package, reason="package")


def _installed_code_snapshot() -> dict[str, Any]:
    purelib = Path(sysconfig.get_path("purelib")).resolve(strict=True)
    platlib = Path(sysconfig.get_path("platlib")).resolve(strict=True)
    if purelib != platlib:
        raise WorkerError("installed_site_packages_roots_differ")
    try:
        purelib.relative_to(Path(sys.prefix).resolve(strict=True))
        Path(__file__).resolve(strict=True).relative_to(purelib)
    except ValueError:
        raise WorkerError("installed_code_outside_active_environment") from None
    return _tree_snapshot(
        purelib,
        reason="installed_code",
        exclude_bytecode=True,
    )


class _StableOperation(NamedTuple):
    value: Any
    package_snapshot: dict[str, Any]
    code_snapshot: dict[str, Any]


def _run_stable_operation(
    package: Path, operation: Callable[[], _T]
) -> _StableOperation:
    package_before = _package_snapshot(package)
    code_before = _installed_code_snapshot()
    value = operation()
    package_after = _package_snapshot(package)
    code_after = _installed_code_snapshot()
    if package_after != package_before:
        raise WorkerError("package_changed_during_operation")
    if code_after != code_before:
        raise WorkerError("installed_code_changed_during_operation")
    return _StableOperation(
        value,
        package_before,
        code_before,
    )


def _sysctl(name: str) -> str | None:
    try:
        completed = subprocess.run(  # noqa: S603
            ["/usr/sbin/sysctl", "-n", name],
            check=False,
            capture_output=True,
            text=True,
            env=clean_subprocess_environment(),
        )
    except OSError:
        return None
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None


def _device_key(memory_bytes: int, *, apple_chip: str) -> str:
    parts = [platform.system().lower(), platform.machine().lower()]
    parts.append("_".join(apple_chip.lower().split()))
    parts.append(f"{round(memory_bytes / 2**30)}gb")
    return "_".join(parts)


def environment_record(
    *, candidate_key: str, wheel_sha256: str, role: str
) -> dict[str, Any]:
    descriptor = descriptor_for_key(candidate_key)
    descriptor_record(candidate_key)
    wheel_sha256 = _require_sha256(wheel_sha256)
    if role == "product":
        sdk_version = importlib.metadata.version("diffusor-rt")
        lock = load_candidate_product_lock(sdk_version)
        lock_resource = candidate_product_lock_resource(sdk_version)
        lock_sha256 = lock["lock_sha256"]
    elif role == "oracle":
        lock = load_oracle_lock(descriptor)
        lock_resource = descriptor.oracle_lock_resource
        lock_sha256 = descriptor.oracle_lock_digest
    else:
        raise WorkerError("environment_role_invalid")

    actual_python = {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
    }
    actual_platform = {
        "system": platform.system(),
        "machine": platform.machine(),
    }
    if actual_python != lock["python"] or actual_platform != lock["platform"]:
        raise WorkerError("environment_platform_or_python_mismatch")
    distributions = _environment_packages(lock)
    result: dict[str, Any] = {
        "python": actual_python,
        "platform": actual_platform,
        "packages": distributions["packages"],
        "tooling_packages": distributions["tooling_packages"],
        "distribution_count": distributions["distribution_count"],
        "wheel_sha256": wheel_sha256,
        "code_origin": (
            "installed-distribution" if role == "product" else "wheel-code-extract"
        ),
        "pythonpath_unset": "PYTHONPATH" not in os.environ,
        "provisional_bootstrap_absent": PROVISIONAL_BOOTSTRAP_ENV not in os.environ,
        "lock_profile": lock["profile"],
        "lock_resource": lock_resource,
        "lock_sha256": lock_sha256,
        "code_snapshot": _installed_code_snapshot(),
    }
    if (
        result["pythonpath_unset"] is not True
        or result["provisional_bootstrap_absent"] is not True
    ):
        raise WorkerError("installed_environment_not_clean")
    if role == "product":
        memory = _sysctl("hw.memsize")
        apple_chip = _sysctl("machdep.cpu.brand_string")
        hardware_model = _sysctl("hw.model")
        if (
            memory is None
            or not memory.isdigit()
            or int(memory) <= 0
            or apple_chip != "Apple M4 Max"
            or hardware_model is None
        ):
            raise WorkerError("product_memory_identity_unavailable")
        memory_bytes = int(memory)
        result["platform"] = {
            **actual_platform,
            "device": _device_key(memory_bytes, apple_chip=apple_chip),
            "apple_chip": apple_chip,
            "hardware_model": hardware_model,
            "memory_bytes": memory_bytes,
        }
        expected_device = _profile(descriptor).minimum_device
        if result["platform"]["device"] != expected_device:
            raise WorkerError(
                "product_device_mismatch: "
                f"expected={expected_device} actual={result['platform']['device']}"
            )
    return result


def descriptor_record(candidate_key: str) -> dict[str, Any]:
    """Read identity/profile from this installed wheel, never the source tree."""
    descriptor = descriptor_for_key(candidate_key)
    profile = _profile(descriptor)
    try:
        Path(__file__).resolve(strict=True).relative_to(Path(sys.prefix).resolve())
    except ValueError:
        raise WorkerError("worker_not_loaded_from_active_environment") from None
    return {
        "identity": _identity(descriptor),
        "profile": {
            "backend": profile.backend,
            "precision": profile.precision,
            "minimum_device": profile.minimum_device,
        },
        "numeric_tier": {
            "key": descriptor.numeric_tier.key,
            "atol": descriptor.numeric_tier.atol,
            "rtol": descriptor.numeric_tier.rtol,
            "exact_argmax": descriptor.numeric_tier.exact_argmax,
            "exact_ordered_top_k": descriptor.numeric_tier.exact_ordered_top_k,
        },
    }


def lock_record(*, candidate_key: str, role: str) -> dict[str, Any]:
    """Expose the lock bundled in this wheel so environment creation is wheel-led."""
    descriptor = descriptor_for_key(candidate_key)
    descriptor_record(candidate_key)
    if role == "product":
        sdk_version = importlib.metadata.version("diffusor-rt")
        lock = load_candidate_product_lock(sdk_version)
        return {
            "resource": candidate_product_lock_resource(sdk_version),
            "sha256": lock["lock_sha256"],
            "lock": {
                key: value for key, value in lock.items() if key != "lock_sha256"
            },
        }
    if role == "oracle":
        return {
            "resource": descriptor.oracle_lock_resource,
            "sha256": descriptor.oracle_lock_digest,
            "lock": load_oracle_lock(descriptor),
        }
    raise WorkerError("environment_role_invalid")


def run_package(
    *, candidate_key: str, hf_cache: Path, package: Path, wheel_sha256: str
) -> dict[str, Any]:
    descriptor = descriptor_for_key(candidate_key)
    profile = _profile(descriptor)
    wheel_sha256 = _require_sha256(wheel_sha256)
    if package.exists() or package.is_symlink():
        raise WorkerError("package_output_preexisting")
    if hf_cache.is_symlink():
        raise WorkerError("hf_cache_symlink")
    hf_cache.mkdir(parents=True, exist_ok=True)
    cache_preexisting = any(hf_cache.iterdir())
    if cache_preexisting:
        raise WorkerError("hf_cache_not_fresh")
    code_before = _installed_code_snapshot()

    from huggingface_hub import snapshot_download

    first = Path(
        snapshot_download(
            repo_id=descriptor.model_id,
            revision=descriptor.revision,
            cache_dir=str(hf_cache / "hub"),
            local_files_only=False,
        )
    ).resolve()
    validate_snapshot(descriptor, first)
    old_offline = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        replay = Path(
            snapshot_download(
                repo_id=descriptor.model_id,
                revision=descriptor.revision,
                cache_dir=str(hf_cache / "hub"),
                local_files_only=True,
            )
        ).resolve()
    finally:
        if old_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = old_offline
    if replay != first:
        raise WorkerError("offline_snapshot_replay_mismatch")

    from edllm.converter import build_package_from_hf_masked_diffusion

    build_package_from_hf_masked_diffusion(
        first,
        package,
        candidate_key=descriptor.key,
        backend=profile.backend,
        precision=profile.precision,
    )
    manifest, weights_sha256 = validate_execution_package(
        descriptor, package, verify_weights_hash=True
    )
    try:
        vocab_size = manifest["model_config"]["vocab_size"]
    except (KeyError, TypeError):
        raise WorkerError("package_vocab_size_missing") from None
    if type(vocab_size) is not int or vocab_size <= descriptor.mask_token_id:
        raise WorkerError("package_vocab_size_invalid")
    snapshot_count, snapshot_bytes, snapshot_tree_sha256 = _inventory(first)
    source_weights_sha256 = _source_weights_sha256(first)
    package_snapshot = _package_snapshot(package)
    package_count = package_snapshot["file_count"]
    package_bytes = package_snapshot["logical_bytes"]
    package_tree_sha256 = package_snapshot["tree_sha256"]
    if snapshot_bytes != descriptor.snapshot_bytes:
        raise WorkerError(
            f"snapshot_bytes_mismatch: {snapshot_bytes} != {descriptor.snapshot_bytes}"
        )
    if package_bytes != descriptor.package_bytes:
        raise WorkerError(
            f"package_bytes_mismatch: {package_bytes} != {descriptor.package_bytes}"
        )
    code_after = _installed_code_snapshot()
    if code_after != code_before:
        raise WorkerError("installed_code_changed_during_operation")
    return {
        "schema_version": "2.0",
        "gate": "package_graph_ir",
        "verdict": "pass",
        "identity": _identity(descriptor),
        "command": expected_commands(descriptor.key)["package_graph_ir"],
        "wheel_sha256": wheel_sha256,
        "trust_remote_code": False,
        "stable_package_snapshot": package_snapshot,
        "stable_code_snapshot": code_before,
        "source_snapshot": {
            "resolved_revision": descriptor.revision,
            "file_count": snapshot_count,
            "logical_bytes": snapshot_bytes,
            "source_weights_sha256": source_weights_sha256,
            "tree_sha256": snapshot_tree_sha256,
            "cache_preexisting": False,
            "first_acquisition_reused": False,
            "offline_replay_reused": True,
        },
        "package_conversion": {
            "output_preexisting": False,
            "reused": False,
            "file_count": package_count,
            "logical_bytes": package_bytes,
            "manifest_sha256": _sha256_file(package / "model.json"),
            "weights_sha256": weights_sha256,
            "graph_ir_sha256": _sha256_file(package / "graph.json"),
            "tree_sha256": package_tree_sha256,
            "vocab_size": vocab_size,
            "backend": profile.backend,
            "precision": profile.precision,
        },
        "resource": _resource_record(descriptor),
    }


def run_conformance(
    *,
    candidate_key: str,
    package: Path,
    wheel_sha256: str,
) -> dict[str, Any]:
    descriptor = descriptor_for_key(candidate_key)
    wheel_sha256 = _require_sha256(wheel_sha256)

    def operation():
        validate_execution_package(
            descriptor, package, verify_weights_hash=True
        )
        from edllm.conformance import run_dllm_conformance

        report = run_dllm_conformance(
            package,
            run_generation=True,
            trust_remote_code=False,
        )
        return report

    stable = _run_stable_operation(package, operation)
    report = stable.value
    checks = [
        {"name": check.name, "passed": check.passed, "detail": check.detail}
        for check in report.checks
    ]
    if tuple(check["name"] for check in checks) != CANDIDATE_CAPABILITY_CHECKS:
        raise WorkerError("conformance_check_set_mismatch")
    if not report.passed:
        failed = [check["name"] for check in checks if not check["passed"]]
        raise WorkerError(f"conformance_failed: {failed}")
    return {
        "schema_version": "2.0",
        "gate": "declared_capability",
        "verdict": "pass",
        "identity": _identity(descriptor),
        "command": expected_commands(descriptor.key)["declared_capability"],
        "wheel_sha256": wheel_sha256,
        "trust_remote_code": False,
        "stable_package_snapshot": stable.package_snapshot,
        "stable_code_snapshot": stable.code_snapshot,
        "checks": checks,
        "checks_passed": len(checks),
        "checks_total": len(checks),
        "resource": _resource_record(descriptor),
        "execution_policy": candidate_runtime_policy(descriptor),
    }


def run_parity(
    *,
    candidate_key: str,
    package: Path,
    reference: Path,
    reference_replay: Path,
    wheel_sha256: str,
) -> dict[str, Any]:
    descriptor = descriptor_for_key(candidate_key)
    profile = _profile(descriptor)
    wheel_sha256 = _require_sha256(wheel_sha256)

    def operation():
        reference_sha256 = _sha256_file(reference)
        reference_replay_sha256 = _sha256_file(reference_replay)
        reference_sidecar = reference.with_name(reference.name + ".json")
        replay_sidecar = reference_replay.with_name(
            reference_replay.name + ".json"
        )
        reference_sidecar_sha256 = _sha256_file(reference_sidecar)
        replay_sidecar_sha256 = _sha256_file(replay_sidecar)
        if (
            reference_replay_sha256 != reference_sha256
            or replay_sidecar_sha256 != reference_sidecar_sha256
        ):
            raise WorkerError("reference_replay_not_deterministic")
        from .parity import run_parity as execute_parity

        raw = execute_parity(
            candidate_key=candidate_key,
            package=package,
            reference=reference,
        )
        return (
            raw,
            reference_sha256,
            reference_sidecar_sha256,
            reference_replay_sha256,
            replay_sidecar_sha256,
        )

    stable = _run_stable_operation(package, operation)
    (
        raw,
        reference_sha256,
        reference_sidecar_sha256,
        replay_sha256,
        replay_sidecar_sha256,
    ) = stable.value
    expected_raw = {
        "candidate_key": descriptor.key,
        "model_id": descriptor.model_id,
        "revision": descriptor.revision,
        "descriptor_digest": descriptor.descriptor_digest,
        "case_digest": descriptor.parity_case_digest,
        "lock_digest": descriptor.oracle_lock_digest,
        "backend": profile.backend,
        "precision": profile.precision,
        "numeric_tier": descriptor.numeric_tier.key,
    }
    if type(raw) is not dict or any(raw.get(key) != value for key, value in expected_raw.items()):
        raise WorkerError("parity_raw_identity_mismatch")
    if raw.get("verdict") != "pass":
        raise WorkerError("parity_failed")
    package_weights_sha256 = raw.get("package_weights_sha256")
    if type(package_weights_sha256) is not str:
        raise WorkerError("parity_raw_package_digest_invalid")
    if raw.get("reference_artifact_sha256") != reference_sha256:
        raise WorkerError("parity_raw_reference_digest_mismatch")
    cases = raw.get("cases")
    if type(cases) is not list or not cases:
        raise WorkerError("parity_cases_missing")
    maxima = [case.get("max_abs_diff") for case in cases if type(case) is dict]
    if len(maxima) != len(cases) or not all(
        type(value) in {int, float} for value in maxima
    ):
        raise WorkerError("parity_case_max_invalid")
    commands = expected_commands(candidate_key)
    return {
        "schema_version": "2.0",
        "gate": "numerical_parity",
        "verdict": "pass",
        "identity": _identity(descriptor),
        "commands": {
            "reference_first": commands["reference_first"],
            "reference_replay": commands["reference_replay"],
            "parity": commands["parity"],
        },
        "wheel_sha256": wheel_sha256,
        "stable_package_snapshot": stable.package_snapshot,
        "stable_code_snapshot": stable.code_snapshot,
        "artifacts": {
            "package_weights_sha256": package_weights_sha256,
            "reference_sha256": reference_sha256,
            "reference_sidecar_sha256": reference_sidecar_sha256,
            "reference_replay_sha256": replay_sha256,
            "reference_replay_sidecar_sha256": replay_sidecar_sha256,
        },
        "reference_generation_count": 2,
        "vocab_size": cases[0]["shape"][1],
        "max_abs_difference": max(maxima),
        "numeric_tier": {
            "key": descriptor.numeric_tier.key,
            "atol": descriptor.numeric_tier.atol,
            "rtol": descriptor.numeric_tier.rtol,
            "exact_argmax": descriptor.numeric_tier.exact_argmax,
            "exact_ordered_top_k": descriptor.numeric_tier.exact_ordered_top_k,
        },
        "cases": cases,
        "raw_parity": raw,
    }


def snapshot_record(
    *, candidate_key: str, package: Path, wheel_sha256: str
) -> dict[str, Any]:
    descriptor = descriptor_for_key(candidate_key)
    stable = _run_stable_operation(package, lambda: None)
    return {
        "schema_version": "1.0",
        "identity": _identity(descriptor),
        "wheel_sha256": _require_sha256(wheel_sha256),
        "package_snapshot": stable.package_snapshot,
        "code_snapshot": stable.code_snapshot,
    }


def source_snapshot_record(
    *, candidate_key: str, snapshot: Path, wheel_sha256: str
) -> dict[str, Any]:
    descriptor = descriptor_for_key(candidate_key)
    validate_snapshot(descriptor, snapshot)
    file_count, logical_bytes, tree_sha256 = _inventory(snapshot)
    return {
        "schema_version": "1.0",
        "identity": _identity(descriptor),
        "wheel_sha256": _require_sha256(wheel_sha256),
        "source_snapshot": {
            "resolved_revision": descriptor.revision,
            "file_count": file_count,
            "logical_bytes": logical_bytes,
            "source_weights_sha256": _source_weights_sha256(snapshot),
            "tree_sha256": tree_sha256,
        },
    }


def run_installed(
    *,
    candidate_key: str,
    package: Path,
    wheel_sha256: str,
    prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    descriptor = descriptor_for_key(candidate_key)
    profile = _profile(descriptor)
    wheel_sha256 = _require_sha256(wheel_sha256)
    if prompt != descriptor.capability_case.prompt or max_tokens != 4:
        raise WorkerError("installed_probe_input_mismatch")
    if "PYTHONPATH" in os.environ or PROVISIONAL_BOOTSTRAP_ENV in os.environ:
        raise WorkerError("installed_environment_not_clean")
    def operation():
        validate_execution_package(
            descriptor, package, verify_weights_hash=True
        )
        from edllm.compiler.registry import REGISTRY

        registered = REGISTRY.get(descriptor.model_id)
        if registered is not None and (
            registered.key != descriptor.key
            or registered.revision != descriptor.revision
            or registered.descriptor_digest != descriptor.descriptor_digest
        ):
            raise WorkerError("candidate_registry_identity_mismatch")
        from edllm import Engine

        engine = None
        body_error: BaseException | None = None
        try:
            engine = Engine.from_package(
                package,
                backend=profile.backend,
                deployment_profile="baseline",
                trust_remote_code=False,
            )
            result = engine.generate(
                prompt,
                max_tokens=max_tokens,
                return_debug=True,
            )
            metrics = result.debug.metrics if result.debug is not None else None
            if type(metrics) is not dict:
                raise WorkerError("installed_generation_debug_missing")
            generation = {
                "text": result.text,
                "text_sha256": hashlib.sha256(
                    result.text.encode("utf-8")
                ).hexdigest(),
                "finish_reason": result.finish_reason,
                "debug_metrics": metrics,
            }
            return {
                "generation": generation,
                "registry_contains_model": registered is not None,
            }
        except BaseException as error:
            body_error = error
            raise
        finally:
            if engine is not None:
                try:
                    engine.close()
                except BaseException as cleanup_error:
                    if body_error is None:
                        raise
                    body_error.add_note(f"engine close failed: {cleanup_error}")

    stable = _run_stable_operation(package, operation)
    operation_result = stable.value
    generation = operation_result["generation"]
    return {
        "schema_version": "2.0",
        "gate": "installed_product_path",
        "verdict": "pass",
        "identity": _identity(descriptor),
        "command": expected_commands(descriptor.key)["installed_product_path"],
        "wheel_sha256": wheel_sha256,
        "trust_remote_code": False,
        "stable_package_snapshot": stable.package_snapshot,
        "stable_code_snapshot": stable.code_snapshot,
        "pythonpath_unset": True,
        "provisional_bootstrap_candidate": None,
        "admission_state": "candidate",
        # Candidate is the evidence stage, not a second runtime registry.
        # Existing Product entries remain installed while their producer is
        # revalidated; the exact membership is evidence, not a refusal.
        "registry_contains_model": operation_result["registry_contains_model"],
        "candidate_descriptor_available": True,
        "package_path_load": True,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "max_tokens": max_tokens,
        "execution_policy": candidate_runtime_policy(descriptor),
        "generation": generation,
        "resource": _resource_record(descriptor),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    environment = subparsers.add_parser("environment", allow_abbrev=False)
    environment.add_argument("--candidate", required=True)
    environment.add_argument("--role", choices=("product", "oracle"), required=True)
    environment.add_argument("--wheel-sha256", required=True)
    identity = subparsers.add_parser("identity", allow_abbrev=False)
    identity.add_argument("--candidate", required=True)
    lock = subparsers.add_parser("lock", allow_abbrev=False)
    lock.add_argument("--candidate", required=True)
    lock.add_argument("--role", choices=("product", "oracle"), required=True)
    package = subparsers.add_parser("package", allow_abbrev=False)
    package.add_argument("--candidate", required=True)
    package.add_argument("--hf-cache", type=Path, required=True)
    package.add_argument("--package", type=Path, required=True)
    package.add_argument("--wheel-sha256", required=True)
    conformance = subparsers.add_parser("conformance", allow_abbrev=False)
    conformance.add_argument("--candidate", required=True)
    conformance.add_argument("--package", type=Path, required=True)
    conformance.add_argument("--wheel-sha256", required=True)
    parity = subparsers.add_parser("parity", allow_abbrev=False)
    parity.add_argument("--candidate", required=True)
    parity.add_argument("--package", type=Path, required=True)
    parity.add_argument("--reference", type=Path, required=True)
    parity.add_argument("--reference-replay", type=Path, required=True)
    parity.add_argument("--wheel-sha256", required=True)
    snapshot = subparsers.add_parser("snapshot", allow_abbrev=False)
    snapshot.add_argument("--candidate", required=True)
    snapshot.add_argument("--package", type=Path, required=True)
    snapshot.add_argument("--wheel-sha256", required=True)
    source_snapshot = subparsers.add_parser(
        "source-snapshot", allow_abbrev=False
    )
    source_snapshot.add_argument("--candidate", required=True)
    source_snapshot.add_argument("--snapshot", type=Path, required=True)
    source_snapshot.add_argument("--wheel-sha256", required=True)
    installed = subparsers.add_parser("installed", allow_abbrev=False)
    installed.add_argument("--candidate", required=True)
    installed.add_argument("--package", type=Path, required=True)
    installed.add_argument("--wheel-sha256", required=True)
    installed.add_argument("--prompt", required=True)
    installed.add_argument("--max-tokens", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.operation == "lock":
            result = lock_record(candidate_key=args.candidate, role=args.role)
        elif args.operation == "identity":
            result = descriptor_record(args.candidate)
        elif args.operation == "environment":
            result = environment_record(
                candidate_key=args.candidate,
                wheel_sha256=args.wheel_sha256,
                role=args.role,
            )
        elif args.operation == "package":
            result = run_package(
                candidate_key=args.candidate,
                hf_cache=args.hf_cache,
                package=args.package,
                wheel_sha256=args.wheel_sha256,
            )
        elif args.operation == "conformance":
            result = run_conformance(
                candidate_key=args.candidate,
                package=args.package,
                wheel_sha256=args.wheel_sha256,
            )
        elif args.operation == "parity":
            result = run_parity(
                candidate_key=args.candidate,
                package=args.package,
                reference=args.reference,
                reference_replay=args.reference_replay,
                wheel_sha256=args.wheel_sha256,
            )
        elif args.operation == "snapshot":
            result = snapshot_record(
                candidate_key=args.candidate,
                package=args.package,
                wheel_sha256=args.wheel_sha256,
            )
        elif args.operation == "source-snapshot":
            result = source_snapshot_record(
                candidate_key=args.candidate,
                snapshot=args.snapshot,
                wheel_sha256=args.wheel_sha256,
            )
        else:
            result = run_installed(
                candidate_key=args.candidate,
                package=args.package,
                wheel_sha256=args.wheel_sha256,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
            )
    except (MemoryError, RecursionError):
        raise
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"candidate worker failed: {error}", file=sys.stderr)
        return 1
    sys.stdout.buffer.write(canonical_json_bytes(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PROVISIONAL_BOOTSTRAP_ENV",
    "WorkerError",
    "clean_subprocess_environment",
    "descriptor_record",
    "environment_record",
    "lock_record",
    "main",
    "run_conformance",
    "run_installed",
    "run_package",
    "run_parity",
    "snapshot_record",
    "source_snapshot_record",
]
