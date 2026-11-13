"""Canonical Candidate four-gate evidence shared by runner and verifier.

The authoritative object embeds all four receipts.  A receipt digest is always
recomputed from its canonical bytes; it is never accepted as an unbound string.
"""
from __future__ import annotations

import copy
import hashlib
from importlib import resources
import json
import math
import re
from typing import Any, Mapping


CANDIDATE_GATE = "model-coverage-candidate-four-gate-v3"
CANDIDATE_RUN_KIND = "candidate-four-gate"
CANDIDATE_GATE_NAMES = (
    "package_graph_ir",
    "numerical_parity",
    "declared_capability",
    "installed_product_path",
)
CANDIDATE_REPORT_HEADING = (
    "# Model Coverage Candidate Four-Gate — generated from machine evidence"
)
CANDIDATE_PRODUCT_LOCK_RESOURCE = (
    "data/locks/qwen3_product_311_mlx031.json"
)
_PRODUCT_LOCK_RESOURCES = {
    "0.2.0": CANDIDATE_PRODUCT_LOCK_RESOURCE,
    "0.2.1": "data/locks/qwen3_product_311_mlx031_sdk021.json",
    "0.2.2": "data/locks/qwen3_product_311_mlx031_sdk022.json",
    "0.2.3": "data/locks/qwen3_product_311_mlx031_sdk023.json",
    "0.2.4": "data/locks/qwen3_product_311_mlx031_sdk024.json",
    "0.2.5": "data/locks/qwen3_product_311_mlx031_sdk025.json",
    "0.2.6a3": "data/locks/qwen3_product_311_mlx031_sdk026a3.json",
    "0.2.6": "data/locks/qwen3_product_311_mlx031_sdk026.json",
}
CANDIDATE_CAPABILITY_CHECKS = (
    "load_runtime",
    "tokenizer_round_trip",
    "mask_token_id_resolvable",
    "model_config_geometry_keys",
    "model_descriptor_identity",
    "generation_runs",
    "generation_text_quality",
    "generation_debug_metrics",
    "generation_provenance_metrics",
    "generation_baseline_status_honest",
    "generation_capability_case",
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_OID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_PEP503_SEPARATORS = re.compile(r"[-_.]+")
_ENVIRONMENT_TOOLING = frozenset({"pip", "setuptools"})
_BUILD_LOCK_SHA256 = "afd456f890afd473b0c8c2de00f1142a243d57f0c64a366059160c0f97a56189"
_BUILD_ARTIFACTS_SHA256 = "329902cad03c17fc406ddfd4865db555270781d35c603970a832c94699981576"


class EvidenceError(ValueError):
    """The candidate evidence is ambiguous, incomplete, or inconsistent."""


class _DuplicateKeyError(ValueError):
    pass


class _NonfiniteNumberError(ValueError):
    pass


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(key)
        result[key] = value
    return result


def _reject_constant(value: str):
    raise _NonfiniteNumberError(value)


def strict_json_loads(raw: bytes | str) -> dict[str, Any]:
    """Parse one UTF-8 JSON object, rejecting duplicate keys and NaN/Infinity."""
    try:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except _DuplicateKeyError as error:
        raise EvidenceError(f"json_duplicate_key: {error.args[0]}") from None
    except _NonfiniteNumberError as error:
        raise EvidenceError(f"json_nonfinite_number: {error.args[0]}") from None
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, TypeError):
        raise EvidenceError("json_invalid") from None
    if type(value) is not dict:
        raise EvidenceError("json_top_level_not_object")
    return value


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Return the one canonical byte representation used by hashes and files."""
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, RecursionError) as error:
        raise EvidenceError(f"canonical_json_invalid: {error}") from error
    return encoded + b"\n"


def receipt_digest(receipt: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(receipt)).hexdigest()


def machine_digest(machine: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(machine)).hexdigest()


def _resource_bytes(relative: str) -> bytes:
    try:
        return resources.files(__package__).joinpath(relative).read_bytes()
    except (OSError, TypeError, ValueError, AttributeError):
        raise EvidenceError(f"evidence_resource_unreadable: {relative}") from None


def _strict_lock(relative: str, *, expected_digest: str | None = None) -> dict[str, Any]:
    raw = _resource_bytes(relative)
    actual_digest = hashlib.sha256(raw).hexdigest()
    if expected_digest is not None and actual_digest != expected_digest:
        raise EvidenceError(f"environment_lock_digest_mismatch: {relative}")
    lock = strict_json_loads(raw)
    allowed = {"schema_version", "profile", "python", "platform", "packages"}
    if "scope" in lock:
        allowed.add("scope")
    _exact_dict(lock, allowed, "environment_lock_schema")
    if type(lock["schema_version"]) is not int or lock["schema_version"] != 1:
        raise EvidenceError("environment_lock_schema")
    if "scope" in lock and lock["scope"] != "runtime-transitive":
        raise EvidenceError("environment_lock_scope")
    _exact_dict(lock["python"], {"implementation", "version"}, "environment_lock_python")
    _exact_dict(lock["platform"], {"system", "machine"}, "environment_lock_platform")
    if (
        type(lock["profile"]) is not str
        or not lock["profile"]
        or not all(type(value) is str and value for value in lock["python"].values())
        or not all(type(value) is str and value for value in lock["platform"].values())
        or type(lock["packages"]) is not dict
        or not lock["packages"]
        or any(
            type(name) is not str
            or not name
            or type(version) is not str
            or not version
            for name, version in lock["packages"].items()
        )
    ):
        raise EvidenceError("environment_lock_invalid")
    normalized_names = [
        _PEP503_SEPARATORS.sub("-", name).lower()
        for name in lock["packages"]
    ]
    if (
        normalized_names != list(lock["packages"])
        or len(normalized_names) != len(set(normalized_names))
    ):
        raise EvidenceError("environment_lock_package_name_invalid")
    return {**lock, "lock_sha256": actual_digest}


def candidate_product_lock_resource(sdk_version: str = "0.2.0") -> str:
    """Select an explicit SDK environment contract; never rewrite a past lock."""
    try:
        return _PRODUCT_LOCK_RESOURCES[sdk_version]
    except (KeyError, TypeError):
        raise EvidenceError(f"product_environment_sdk_unsupported: {sdk_version!r}") from None


def load_candidate_product_lock(sdk_version: str = "0.2.0") -> dict[str, Any]:
    """Load the versioned product lock shared by runner and verifier.

    The default preserves historical preview receipts. Installed workers pass
    the actual distribution version rather than inferring it from model identity.
    """
    resource = candidate_product_lock_resource(sdk_version)
    lock = _strict_lock(resource)
    if lock.get("scope") != "runtime-transitive":
        raise EvidenceError("product_environment_lock_scope")
    if (
        lock["profile"] != resource.rsplit("/", 1)[-1].removesuffix(".json")
        or lock["python"] != {"implementation": "CPython", "version": "3.11.15"}
        or lock["platform"] != {"system": "Darwin", "machine": "arm64"}
    ):
        raise EvidenceError("product_environment_lock_invalid")
    if lock["packages"].get("diffusor-rt") != sdk_version:
        raise EvidenceError("product_environment_lock_sdk_mismatch")
    return lock


def expected_commands(candidate_key: str) -> dict[str, Any]:
    """Canonical redacted argv for every subprocess in the Candidate gate."""
    package = [
        "product-python",
        "-m",
        "edllm.model_coverage.candidate_gate_worker",
        "package",
        "--candidate",
        candidate_key,
        "--hf-cache",
        "<FRESH_HF_CACHE>",
        "--package",
        "<PACKAGE>",
        "--wheel-sha256",
        "<WHEEL_SHA256>",
    ]
    reference_prefix = [
        "oracle-python",
        "-m",
        "edllm.model_coverage.reference",
        "--candidate",
        candidate_key,
        "--snapshot",
        "<SNAPSHOT>",
        "--package",
        "<PACKAGE>",
        "--output",
    ]
    return {
        "package_graph_ir": package,
        "reference_first": [*reference_prefix, "<REFERENCE>"],
        "reference_replay": [*reference_prefix, "<REFERENCE_REPLAY>"],
        "parity": [
            "product-python",
            "-m",
            "edllm.model_coverage.candidate_gate_worker",
            "parity",
            "--candidate",
            candidate_key,
            "--package",
            "<PACKAGE>",
            "--reference",
            "<REFERENCE>",
            "--reference-replay",
            "<REFERENCE_REPLAY>",
            "--wheel-sha256",
            "<WHEEL_SHA256>",
        ],
        "declared_capability": [
            "product-python",
            "-m",
            "edllm.model_coverage.candidate_gate_worker",
            "conformance",
            "--candidate",
            candidate_key,
            "--package",
            "<PACKAGE>",
            "--wheel-sha256",
            "<WHEEL_SHA256>",
        ],
        "installed_product_path": [
            "product-python",
            "-m",
            "edllm.model_coverage.candidate_gate_worker",
            "installed",
            "--candidate",
            candidate_key,
            "--package",
            "<PACKAGE>",
            "--wheel-sha256",
            "<WHEEL_SHA256>",
            "--prompt",
            "<PROMPT>",
            "--max-tokens",
            "4",
        ],
    }


def _exact_dict(value: object, keys: set[str], reason: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise EvidenceError(reason)
    return value


def _is_sha(value: object) -> bool:
    return type(value) is str and _SHA256.fullmatch(value) is not None


def _is_oid(value: object) -> bool:
    return type(value) is str and _GIT_OID.fullmatch(value) is not None


def _positive_int(value: object) -> bool:
    return type(value) is int and value > 0


def _nonnegative_number(value: object) -> bool:
    return type(value) in {int, float} and math.isfinite(value) and value >= 0


def _command(value: object, reason: str) -> list[str]:
    if (
        type(value) is not list
        or not value
        or not all(type(part) is str and part for part in value)
    ):
        raise EvidenceError(reason)
    return value


def _validate_identity(value: object) -> dict[str, Any]:
    identity = _exact_dict(
        value,
        {
            "candidate_key",
            "model_id",
            "revision",
            "descriptor_digest",
            "case_sha256",
            "oracle_lock_sha256",
        },
        "identity_schema",
    )
    if (
        not all(
            type(identity[field]) is str and identity[field]
            for field in ("candidate_key", "model_id")
        )
        or not _is_oid(identity["revision"])
        or not all(
            _is_sha(identity[field])
            for field in (
                "descriptor_digest",
                "case_sha256",
                "oracle_lock_sha256",
            )
        )
    ):
        raise EvidenceError("identity_invalid")
    return identity


def _validate_profile(value: object) -> dict[str, Any]:
    profile = _exact_dict(
        value,
        {"backend", "precision", "minimum_device"},
        "profile_schema",
    )
    if not all(type(item) is str and item for item in profile.values()):
        raise EvidenceError("profile_invalid")
    return profile


def _validate_resource(value: object) -> dict[str, Any]:
    resource = _exact_dict(
        value,
        {
            "peak_rss_bytes",
            "peak_rss_limit_bytes",
            "within_resource_envelope",
        },
        "resource_schema",
    )
    if (
        not _positive_int(resource["peak_rss_bytes"])
        or not _positive_int(resource["peak_rss_limit_bytes"])
        or resource["peak_rss_bytes"] > resource["peak_rss_limit_bytes"]
        or resource["within_resource_envelope"] is not True
    ):
        raise EvidenceError("resource_envelope_invalid")
    return resource


def _validate_tree_snapshot(value: object, reason: str) -> dict[str, Any]:
    snapshot = _exact_dict(
        value,
        {
            "root_dev",
            "root_ino",
            "file_count",
            "logical_bytes",
            "tree_sha256",
        },
        f"{reason}_schema",
    )
    if (
        type(snapshot["root_dev"]) is not int
        or snapshot["root_dev"] < 0
        or not _positive_int(snapshot["root_ino"])
        or not _positive_int(snapshot["file_count"])
        or not _positive_int(snapshot["logical_bytes"])
        or not _is_sha(snapshot["tree_sha256"])
    ):
        raise EvidenceError(f"{reason}_invalid")
    return snapshot


def validate_build_receipt(
    value: object,
    *,
    source_sha256: str,
    wheel_sha256: str,
    python_version: str,
    lock_bytes: bytes | None = None,
    expected_wheel_filename: str = "diffusor_rt-0.2.0-py3-none-any.whl",
) -> dict:
    """Validate a build against lock truth and the caller's source-bound identity.

    Historical model records retain the preview filename by default. New
    artifact gates supply the identity from the committed project metadata.
    """
    build = _exact_dict(value, {"receipt_sha256", "receipt"}, "build_schema")
    receipt = _exact_dict(
        build["receipt"],
        {"schema_version", "kind", "source", "lock", "backend", "execution", "wheel"},
        "build_receipt_schema",
    )
    if build["receipt_sha256"] != receipt_digest(receipt):
        raise EvidenceError("build_receipt_digest_mismatch")
    source = _exact_dict(receipt["source"], {
        "archive_sha256", "payload_file_count", "payload_logical_bytes",
        "payload_tree_sha256",
    }, "build_source_schema")
    lock = _exact_dict(receipt["lock"], {"resource", "sha256", "profile"}, "build_lock_schema")
    backend = _exact_dict(receipt["backend"], {
        "entry_point", "hook", "version", "python", "artifacts",
    }, "build_backend_schema")
    execution = _exact_dict(receipt["execution"], {
        "command", "dependency_resolution", "artifact_access", "network_isolation",
    }, "build_execution_schema")
    wheel = _exact_dict(receipt["wheel"], {
        "filename", "sha256", "member_count", "logical_bytes", "tree_sha256",
        "payload_file_count", "payload_logical_bytes", "payload_tree_sha256",
        "record_sha256",
    }, "build_wheel_schema")
    python = _exact_dict(backend["python"], {"implementation", "version"}, "build_python_schema")
    artifacts = backend["artifacts"]
    if type(artifacts) is not list:
        raise EvidenceError("build_artifacts_schema")
    artifact_fields = {
        "name", "version", "filename", "url", "size_bytes", "sha256",
        "requires_python",
    }
    for artifact in artifacts:
        artifact = _exact_dict(artifact, artifact_fields, "build_artifact_schema")
        if (
            not all(type(artifact[name]) is str and artifact[name]
                    for name in ("name", "version", "filename", "url"))
            or type(artifact["size_bytes"]) is not int
            or artifact["size_bytes"] <= 0
            or not _is_sha(artifact["sha256"])
            or (artifact["requires_python"] is not None
                and type(artifact["requires_python"]) is not str)
        ):
            raise EvidenceError("build_artifact_invalid")
    expected_artifacts = None
    if lock_bytes is not None:
        expected_lock = strict_json_loads(lock_bytes)
        if hashlib.sha256(lock_bytes).hexdigest() != _BUILD_LOCK_SHA256:
            raise EvidenceError("build_lock_invalid")
        expected_artifacts = expected_lock.get("artifacts")
    if (
        receipt["schema_version"] != "1.0"
        or receipt["kind"] != "hash-locked-wheel-build"
        or source["archive_sha256"] != source_sha256
        or not all(_positive_int(source[name]) for name in ("payload_file_count", "payload_logical_bytes"))
        or not _is_sha(source["payload_tree_sha256"])
        or lock != {"resource": "build-backend.lock.json", "sha256": _BUILD_LOCK_SHA256,
                    "profile": "hatchling-1.32.0-py311"}
        or backend["entry_point"] != "hatchling.build" or backend["hook"] != "build_wheel"
        or backend["version"] != "1.32.0" or python["implementation"] != "CPython"
        or type(python["version"]) is not str
        or re.fullmatch(r"3\.11\.\d+", python["version"]) is None
        or python["version"] != python_version
        or receipt_digest({"artifacts": artifacts}) != _BUILD_ARTIFACTS_SHA256
        or (expected_artifacts is not None and artifacts != expected_artifacts)
        or execution != {"command": ["build-python", "-I", "-S", "-c", "<LOCKED_BUILD_HOOK>"],
                         "dependency_resolution": "none", "artifact_access": "local-hash-verified",
                         "network_isolation": "none"}
        or wheel["filename"] != expected_wheel_filename
        or wheel["sha256"] != wheel_sha256
        or not all(_positive_int(wheel[name]) for name in (
            "member_count", "logical_bytes", "payload_file_count", "payload_logical_bytes"
        ))
        or not all(_is_sha(wheel[name]) for name in ("tree_sha256", "payload_tree_sha256", "record_sha256"))
        or wheel["payload_file_count"] != source["payload_file_count"]
        or wheel["payload_logical_bytes"] != source["payload_logical_bytes"]
        or wheel["payload_tree_sha256"] != source["payload_tree_sha256"]
    ):
        raise EvidenceError("build_provenance_invalid")
    return build


def _validate_environment(
    value: object,
    *,
    wheel_sha256: str,
    product: bool,
    profile: Mapping[str, Any],
    descriptor: object,
) -> dict[str, Any]:
    environment = _exact_dict(
        value,
        {
            "python",
            "platform",
            "packages",
            "tooling_packages",
            "distribution_count",
            "wheel_sha256",
            "code_origin",
            "pythonpath_unset",
            "provisional_bootstrap_absent",
            "lock_profile",
            "lock_resource",
            "lock_sha256",
            "code_snapshot",
        },
        "environment_schema",
    )
    python = _exact_dict(
        environment["python"],
        {"implementation", "version"},
        "environment_python_schema",
    )
    platform_keys = (
        {
            "system",
            "machine",
            "device",
            "apple_chip",
            "hardware_model",
            "memory_bytes",
        }
        if product
        else {"system", "machine"}
    )
    platform = _exact_dict(
        environment["platform"], platform_keys, "environment_platform_schema"
    )
    packages = environment["packages"]
    tooling_packages = environment["tooling_packages"]
    distribution_count = environment["distribution_count"]
    if (
        not all(type(item) is str and item for item in python.values())
        or not all(
            type(platform[field]) is str and platform[field]
            for field in ("system", "machine")
        )
        or type(packages) is not dict
        or not packages
        or any(
            type(name) is not str
            or not name
            or type(version) is not str
            or not version
            for name, version in packages.items()
        )
        or type(tooling_packages) is not dict
        or set(tooling_packages) - _ENVIRONMENT_TOOLING
        or "pip" not in tooling_packages
        or any(
            type(name) is not str
            or _PEP503_SEPARATORS.sub("-", name).lower() != name
            or type(version) is not str
            or not version
            for name, version in tooling_packages.items()
        )
        or type(distribution_count) is not int
        or distribution_count != len(packages) + len(tooling_packages)
        or environment["wheel_sha256"] != wheel_sha256
        or environment["code_origin"]
        != ("installed-distribution" if product else "wheel-code-extract")
        or environment["pythonpath_unset"] is not True
        or environment["provisional_bootstrap_absent"] is not True
        or type(environment["lock_profile"]) is not str
        or not environment["lock_profile"]
        or type(environment["lock_resource"]) is not str
        or not environment["lock_resource"]
        or not _is_sha(environment["lock_sha256"])
    ):
        raise EvidenceError("environment_invalid")
    _validate_tree_snapshot(environment["code_snapshot"], "environment_code_snapshot")
    if product:
        apple_chip = platform["apple_chip"]
        hardware_model = platform["hardware_model"]
        memory_bytes = platform["memory_bytes"]
        if (
            apple_chip != "Apple M4 Max"
            or type(hardware_model) is not str
            or not hardware_model.strip()
            or type(memory_bytes) is not int
            or memory_bytes < 64 * 1024**3
        ):
            raise EvidenceError("product_device_invalid")
        derived_device = "_".join(
            (
                platform["system"].lower(),
                platform["machine"].lower(),
                "_".join(apple_chip.lower().split()),
                f"{round(memory_bytes / 2**30)}gb",
            )
        )
        if (
            derived_device != profile["minimum_device"]
            or platform["device"] != derived_device
        ):
            raise EvidenceError("product_device_invalid")
    if product:
        sdk_version = packages.get("diffusor-rt")
        expected_resource = candidate_product_lock_resource(sdk_version)
        lock = load_candidate_product_lock(sdk_version)
        if (
            environment["lock_resource"] != expected_resource
            or environment["lock_sha256"] != lock["lock_sha256"]
            or environment["lock_profile"] != lock["profile"]
            or python != lock["python"]
            or {field: platform[field] for field in ("system", "machine")}
            != lock["platform"]
            or packages != lock["packages"]
        ):
            raise EvidenceError("product_environment_lock_mismatch")
    else:
        oracle_resource = getattr(descriptor, "oracle_lock_resource", None)
        oracle_digest = getattr(descriptor, "oracle_lock_digest", None)
        lock = _strict_lock(oracle_resource, expected_digest=oracle_digest)
        if (
            environment["lock_resource"] != oracle_resource
            or environment["lock_sha256"] != oracle_digest
            or environment["lock_profile"] != lock["profile"]
            or python != lock["python"]
            or platform != lock["platform"]
            or packages != lock["packages"]
        ):
            raise EvidenceError("oracle_environment_lock_mismatch")
    return environment


def _validate_package_receipt(
    receipt: object,
    *,
    identity: Mapping[str, Any],
    profile: Mapping[str, Any],
    wheel_sha256: str,
    descriptor: object,
    commands: Mapping[str, Any],
) -> dict[str, Any]:
    package_receipt = _exact_dict(
        receipt,
        {
            "schema_version",
            "gate",
            "verdict",
            "identity",
            "command",
            "wheel_sha256",
            "trust_remote_code",
            "stable_package_snapshot",
            "stable_code_snapshot",
            "source_snapshot",
            "package_conversion",
            "resource",
        },
        "package_receipt_schema",
    )
    if (
        package_receipt["schema_version"] != "2.0"
        or package_receipt["gate"] != "package_graph_ir"
        or package_receipt["verdict"] != "pass"
        or package_receipt["identity"] != identity
        or package_receipt["wheel_sha256"] != wheel_sha256
        or package_receipt["trust_remote_code"] is not False
    ):
        raise EvidenceError("package_receipt_identity")
    if package_receipt["command"] != commands["package_graph_ir"]:
        raise EvidenceError("package_command_invalid")
    snapshot = _exact_dict(
        package_receipt["source_snapshot"],
        {
            "resolved_revision",
            "file_count",
            "logical_bytes",
            "source_weights_sha256",
            "tree_sha256",
            "cache_preexisting",
            "first_acquisition_reused",
            "offline_replay_reused",
        },
        "snapshot_schema",
    )
    if (
        snapshot["resolved_revision"] != identity["revision"]
        or not _positive_int(snapshot["file_count"])
        or snapshot["logical_bytes"] != getattr(descriptor, "snapshot_bytes", None)
        or type(snapshot["logical_bytes"]) is not int
        or not _is_sha(snapshot["source_weights_sha256"])
        or not _is_sha(snapshot["tree_sha256"])
        or snapshot["cache_preexisting"] is not False
        or snapshot["first_acquisition_reused"] is not False
        or snapshot["offline_replay_reused"] is not True
    ):
        raise EvidenceError("snapshot_cache_semantics")
    package = _exact_dict(
        package_receipt["package_conversion"],
        {
            "output_preexisting",
            "reused",
            "file_count",
            "logical_bytes",
            "manifest_sha256",
            "weights_sha256",
            "graph_ir_sha256",
            "tree_sha256",
            "vocab_size",
            "backend",
            "precision",
        },
        "package_artifact_schema",
    )
    if (
        package["output_preexisting"] is not False
        or package["reused"] is not False
        or not _positive_int(package["file_count"])
        or package["logical_bytes"] != getattr(descriptor, "package_bytes", None)
        or type(package["logical_bytes"]) is not int
        or not all(
            _is_sha(package[field])
            for field in (
                "manifest_sha256",
                "weights_sha256",
                "graph_ir_sha256",
                "tree_sha256",
            )
        )
        or type(package["vocab_size"]) is not int
        or package["vocab_size"] <= getattr(descriptor, "mask_token_id", -1)
        or package["backend"] != profile["backend"]
        or package["precision"] != profile["precision"]
    ):
        raise EvidenceError("package_artifact_invalid")
    package_snapshot = _validate_tree_snapshot(
        package_receipt["stable_package_snapshot"], "package_snapshot"
    )
    if package_snapshot != {
        "root_dev": package_snapshot["root_dev"],
        "root_ino": package_snapshot["root_ino"],
        "file_count": package["file_count"],
        "logical_bytes": package["logical_bytes"],
        "tree_sha256": package["tree_sha256"],
    }:
        raise EvidenceError("package_snapshot_mismatch")
    _validate_tree_snapshot(
        package_receipt["stable_code_snapshot"], "package_code_snapshot"
    )
    resource = _validate_resource(package_receipt["resource"])
    if resource["peak_rss_limit_bytes"] != getattr(descriptor, "peak_rss_bytes", None):
        raise EvidenceError("resource_envelope_invalid")
    return package_receipt


def _validate_parity_receipt(
    receipt: object,
    *,
    identity: Mapping[str, Any],
    profile: Mapping[str, Any],
    wheel_sha256: str,
    package_snapshot: Mapping[str, Any],
    code_snapshot: Mapping[str, Any],
    package_weights_sha256: str,
    package_vocab_size: int,
    descriptor: object,
    commands: Mapping[str, Any],
) -> dict[str, Any]:
    parity = _exact_dict(
        receipt,
        {
            "schema_version",
            "gate",
            "verdict",
            "identity",
            "commands",
            "wheel_sha256",
            "stable_package_snapshot",
            "stable_code_snapshot",
            "artifacts",
            "reference_generation_count",
            "vocab_size",
            "max_abs_difference",
            "numeric_tier",
            "cases",
            "raw_parity",
        },
        "parity_receipt_schema",
    )
    if (
        parity["schema_version"] != "2.0"
        or parity["gate"] != "numerical_parity"
        or parity["verdict"] != "pass"
        or parity["identity"] != identity
        or parity["wheel_sha256"] != wheel_sha256
        or parity["reference_generation_count"] != 2
        or type(parity["reference_generation_count"]) is not int
        or parity["vocab_size"] != package_vocab_size
        or type(parity["vocab_size"]) is not int
        or not _nonnegative_number(parity["max_abs_difference"])
    ):
        raise EvidenceError("parity_receipt_identity")
    parity_package = _validate_tree_snapshot(
        parity["stable_package_snapshot"], "parity_package_snapshot"
    )
    parity_code = _validate_tree_snapshot(
        parity["stable_code_snapshot"], "parity_code_snapshot"
    )
    if (
        parity_package != package_snapshot
        or parity_code != code_snapshot
    ):
        raise EvidenceError("parity_snapshot_mismatch")
    parity_commands = _exact_dict(
        parity["commands"],
        {"reference_first", "reference_replay", "parity"},
        "parity_commands_schema",
    )
    if parity_commands != {
        "reference_first": commands["reference_first"],
        "reference_replay": commands["reference_replay"],
        "parity": commands["parity"],
    }:
        raise EvidenceError("parity_command_invalid")
    artifacts = _exact_dict(
        parity["artifacts"],
        {
            "package_weights_sha256",
            "reference_sha256",
            "reference_sidecar_sha256",
            "reference_replay_sha256",
            "reference_replay_sidecar_sha256",
        },
        "parity_artifact_schema",
    )
    if (
        not all(_is_sha(value) for value in artifacts.values())
        or artifacts["package_weights_sha256"] != package_weights_sha256
        or artifacts["reference_replay_sha256"] != artifacts["reference_sha256"]
        or artifacts["reference_replay_sidecar_sha256"]
        != artifacts["reference_sidecar_sha256"]
    ):
        raise EvidenceError("parity_artifact_invalid")
    numeric = _exact_dict(
        parity["numeric_tier"],
        {"key", "atol", "rtol", "exact_argmax", "exact_ordered_top_k"},
        "numeric_tier_schema",
    )
    if (
        numeric
        != {
            "key": descriptor.numeric_tier.key,
            "atol": descriptor.numeric_tier.atol,
            "rtol": descriptor.numeric_tier.rtol,
            "exact_argmax": descriptor.numeric_tier.exact_argmax,
            "exact_ordered_top_k": descriptor.numeric_tier.exact_ordered_top_k,
        }
    ):
        raise EvidenceError("numeric_tier_invalid")
    cases = parity["cases"]
    try:
        from .producer_descriptors import load_case_corpus

        corpus_cases = load_case_corpus(descriptor)["cases"]
    except (KeyError, TypeError, ValueError) as error:
        raise EvidenceError(f"parity_case_corpus_invalid: {error}") from None
    if type(cases) is not list or len(cases) != len(corpus_cases):
        raise EvidenceError("parity_cases_invalid")
    maxima: list[float | int] = []
    for case, corpus_case in zip(cases, corpus_cases, strict=True):
        case = _exact_dict(
            case,
            {"name", "shape", "max_abs_diff", "verdict", "reasons"},
            "parity_case_schema",
        )
        if (
            case["name"] != corpus_case["name"]
            or type(case["shape"]) is not list
            or case["shape"]
            != [len(corpus_case["selected_rows"]), package_vocab_size]
            or not _nonnegative_number(case["max_abs_diff"])
            or case["verdict"] != "pass"
            or case["reasons"] != []
        ):
            raise EvidenceError("parity_case_invalid")
        maxima.append(case["max_abs_diff"])
    if parity["max_abs_difference"] != max(maxima):
        raise EvidenceError("parity_max_abs_difference_mismatch")
    raw = _exact_dict(
        parity["raw_parity"],
        {
            "schema_version",
            "candidate_key",
            "model_id",
            "revision",
            "descriptor_digest",
            "case_digest",
            "lock_digest",
            "package_weights_sha256",
            "reference_artifact_sha256",
            "backend",
            "precision",
            "numeric_tier",
            "verdict",
            "cases",
        },
        "raw_parity_schema",
    )
    if raw != {
        "schema_version": 1,
        "candidate_key": identity["candidate_key"],
        "model_id": identity["model_id"],
        "revision": identity["revision"],
        "descriptor_digest": identity["descriptor_digest"],
        "case_digest": identity["case_sha256"],
        "lock_digest": identity["oracle_lock_sha256"],
        "package_weights_sha256": package_weights_sha256,
        "reference_artifact_sha256": artifacts["reference_sha256"],
        "backend": profile["backend"],
        "precision": profile["precision"],
        "numeric_tier": numeric["key"],
        "verdict": "pass",
        "cases": cases,
    }:
        raise EvidenceError("raw_parity_mismatch")
    return parity


def _validate_capability_receipt(
    receipt: object,
    *,
    identity: Mapping[str, Any],
    wheel_sha256: str,
    package_snapshot: Mapping[str, Any],
    code_snapshot: Mapping[str, Any],
    descriptor: object,
    commands: Mapping[str, Any],
) -> dict[str, Any]:
    capability = _exact_dict(
        receipt,
        {
            "schema_version",
            "gate",
            "verdict",
            "identity",
            "command",
            "wheel_sha256",
            "trust_remote_code",
            "stable_package_snapshot",
            "stable_code_snapshot",
            "checks",
            "checks_passed",
            "checks_total",
            "resource",
            "execution_policy",
        },
        "capability_receipt_schema",
    )
    if (
        capability["schema_version"] != "2.0"
        or capability["gate"] != "declared_capability"
        or capability["verdict"] != "pass"
        or capability["identity"] != identity
        or capability["wheel_sha256"] != wheel_sha256
        or capability["trust_remote_code"] is not False
    ):
        raise EvidenceError("capability_receipt_identity")
    if (
        _validate_tree_snapshot(
            capability["stable_package_snapshot"], "capability_package_snapshot"
        )
        != package_snapshot
        or _validate_tree_snapshot(
            capability["stable_code_snapshot"], "capability_code_snapshot"
        )
        != code_snapshot
    ):
        raise EvidenceError("capability_snapshot_mismatch")
    if capability["command"] != commands["declared_capability"]:
        raise EvidenceError("capability_command_invalid")
    checks = capability["checks"]
    if type(checks) is not list or not checks:
        raise EvidenceError("capability_checks_invalid")
    for check in checks:
        check = _exact_dict(
            check, {"name", "passed", "detail"}, "capability_check_schema"
        )
        if (
            type(check["name"]) is not str
            or not check["name"]
            or check["passed"] is not True
            or type(check["detail"]) is not str
        ):
            raise EvidenceError("capability_check_invalid")
    if tuple(check["name"] for check in checks) != CANDIDATE_CAPABILITY_CHECKS:
        raise EvidenceError("capability_check_set_invalid")
    if (
        type(capability["checks_passed"]) is not int
        or type(capability["checks_total"]) is not int
        or capability["checks_total"] != len(checks)
        or capability["checks_passed"] != len(checks)
    ):
        raise EvidenceError("capability_check_count_invalid")
    resource = _validate_resource(capability["resource"])
    if resource["peak_rss_limit_bytes"] != getattr(descriptor, "peak_rss_bytes", None):
        raise EvidenceError("resource_envelope_invalid")
    from .producer_descriptors import candidate_runtime_policy

    expected_policy = candidate_runtime_policy(descriptor)
    policy = _exact_dict(
        capability["execution_policy"],
        set(expected_policy),
        "capability_policy_schema",
    )
    if canonical_json_bytes(policy) != canonical_json_bytes(expected_policy):
        raise EvidenceError("capability_policy_invalid")
    return capability


def _validate_installed_generation_metrics(
    metrics: object,
    *,
    profile: Mapping[str, Any],
    descriptor: object,
) -> None:
    """Bind installed generation telemetry to the descriptor runtime policy."""
    if type(metrics) is not dict:
        raise EvidenceError("product_gate_invalid")
    from .producer_descriptors import candidate_runtime_policy

    policy = candidate_runtime_policy(descriptor)
    common_invalid = (
        metrics.get("resolved_backend") != profile["backend"]
        or metrics.get("requested_backend") != profile["backend"]
        or metrics.get("generation_backend") != profile["backend"]
        or metrics.get("mlx_diffusion_policy_profile") != "baseline"
        or metrics.get("mlx_diffusion_decode_strategy")
        != policy["decode_strategy"]
        or metrics.get("mlx_d2f_exact_kv_cache_enabled")
        is not policy["d2f_exact_kv_cache"]
        or metrics.get("mlx_block_kv_cache_enabled") is not False
        or metrics.get("mlx_prefix_kv_cache_enabled") is not False
        or type(metrics.get("mlx_remaining_mask_count")) is not int
        or metrics.get("mlx_remaining_mask_count") != 0
    )
    if common_invalid:
        raise EvidenceError("product_gate_invalid")

    if policy["decode_strategy"] == "block_threshold":
        invalid = (
            metrics.get("mlx_kv_cache_enabled") is not True
            or metrics.get("mlx_kv_cache_strategy")
            != "bidirectional_layer0_kv"
            or type(metrics.get("mlx_kv_layer0_entries")) is not int
            or metrics.get("mlx_kv_layer0_entries") <= 0
            or type(metrics.get("mlx_kv_reuse_count")) is not int
            or metrics.get("mlx_kv_reuse_count") <= 0
            or type(metrics.get("mlx_kv_invalidated_count")) is not int
            or metrics.get("mlx_kv_invalidated_count") <= 0
            or metrics.get("mlx_kv_upper_layer_reuse_count") != 0
        )
    elif policy["decode_strategy"] in {"bd3_fixed_quota", "fixed_quota"}:
        contract = getattr(descriptor, "decode_contract", None)
        if policy["decode_strategy"] == "bd3_fixed_quota":
            if contract is None:
                raise EvidenceError("product_gate_invalid")
            loop_kind = "mlx_qwen3_bd3_fixed_quota"
            block_size = contract.block_size
        else:
            loop_kind = "mlx_bidirectional_fixed_quota"
            block_size = policy["block_size"]
        invalid = (
            metrics.get("decode_loop_kind") != loop_kind
            or metrics.get("mlx_diffusion_block_size") != block_size
            or (
                policy["decode_strategy"] == "fixed_quota"
                and metrics.get("mlx_diffusion_configured_steps") != policy["steps"]
            )
            or type(metrics.get("mlx_diffusion_temperature")) not in (int, float)
            or metrics.get("mlx_diffusion_temperature") != policy["temperature"]
            or type(metrics.get("mlx_diffusion_top_p")) not in (int, float)
            or metrics.get("mlx_diffusion_top_p") != policy["top_p"]
            or metrics.get("mlx_diffusion_remasking") != policy["remasking"]
            or metrics.get("mlx_kv_cache_enabled") is not False
            or metrics.get("mlx_kv_cache_strategy") != "none"
            or type(metrics.get("mlx_kv_layer0_entries")) is not int
            or metrics.get("mlx_kv_layer0_entries") != 0
            or type(metrics.get("mlx_kv_reuse_count")) is not int
            or metrics.get("mlx_kv_reuse_count") != 0
            or type(metrics.get("mlx_kv_invalidated_count")) is not int
            or metrics.get("mlx_kv_invalidated_count") != 0
            or type(metrics.get("mlx_kv_upper_layer_reuse_count")) is not int
            or metrics.get("mlx_kv_upper_layer_reuse_count") != 0
        )
    else:
        invalid = True
    if invalid:
        raise EvidenceError("product_gate_invalid")


def _validate_installed_receipt(
    receipt: object,
    *,
    identity: Mapping[str, Any],
    profile: Mapping[str, Any],
    wheel_sha256: str,
    package_snapshot: Mapping[str, Any],
    code_snapshot: Mapping[str, Any],
    descriptor: object,
    commands: Mapping[str, Any],
) -> dict[str, Any]:
    installed = _exact_dict(
        receipt,
        {
            "schema_version",
            "gate",
            "verdict",
            "identity",
            "command",
            "wheel_sha256",
            "trust_remote_code",
            "stable_package_snapshot",
            "stable_code_snapshot",
            "pythonpath_unset",
            "provisional_bootstrap_candidate",
            "admission_state",
            "registry_contains_model",
            "candidate_descriptor_available",
            "package_path_load",
            "prompt_sha256",
            "max_tokens",
            "execution_policy",
            "generation",
            "resource",
        },
        "installed_receipt_schema",
    )
    if (
        installed["schema_version"] != "2.0"
        or installed["gate"] != "installed_product_path"
        or installed["verdict"] != "pass"
        or installed["identity"] != identity
        or installed["wheel_sha256"] != wheel_sha256
        or installed["trust_remote_code"] is not False
        or installed["pythonpath_unset"] is not True
        or installed["provisional_bootstrap_candidate"] is not None
        or installed["admission_state"] != "candidate"
        or type(installed["registry_contains_model"]) is not bool
        or installed["candidate_descriptor_available"] is not True
        or installed["package_path_load"] is not True
        or not _is_sha(installed["prompt_sha256"])
        or installed["prompt_sha256"]
        != hashlib.sha256(
            descriptor.capability_case.prompt.encode("utf-8")
        ).hexdigest()
        or type(installed["max_tokens"]) is not int
        or installed["max_tokens"] != 4
    ):
        raise EvidenceError("installed_receipt_identity")
    if (
        _validate_tree_snapshot(
            installed["stable_package_snapshot"], "installed_package_snapshot"
        )
        != package_snapshot
        or _validate_tree_snapshot(
            installed["stable_code_snapshot"], "installed_code_snapshot"
        )
        != code_snapshot
    ):
        raise EvidenceError("installed_snapshot_mismatch")
    if installed["command"] != commands["installed_product_path"]:
        raise EvidenceError("installed_commands_invalid")
    from .producer_descriptors import candidate_runtime_policy

    expected_policy = candidate_runtime_policy(descriptor)
    policy = _exact_dict(
        installed["execution_policy"],
        set(expected_policy),
        "installed_execution_policy_schema",
    )
    if canonical_json_bytes(policy) != canonical_json_bytes(expected_policy):
        raise EvidenceError("product_gate_invalid")
    generation = _exact_dict(
        installed["generation"],
        {"text", "text_sha256", "finish_reason", "debug_metrics"},
        "installed_generation_schema",
    )
    metrics = generation["debug_metrics"]
    if (
        type(generation["text"]) is not str
        or not generation["text"].strip()
        or generation["text_sha256"]
        != hashlib.sha256(generation["text"].encode("utf-8")).hexdigest()
        or type(generation["finish_reason"]) is not str
        or not generation["finish_reason"]
    ):
        raise EvidenceError("product_gate_invalid")
    _validate_installed_generation_metrics(
        metrics,
        profile=profile,
        descriptor=descriptor,
    )
    resource = _validate_resource(installed["resource"])
    if resource["peak_rss_limit_bytes"] != getattr(descriptor, "peak_rss_bytes", None):
        raise EvidenceError("resource_envelope_invalid")
    return installed


def build_candidate_machine(
    *,
    source: Mapping[str, Any],
    identity: Mapping[str, Any],
    profile: Mapping[str, Any],
    wheel_sha256: str,
    build: Mapping[str, Any],
    environments: Mapping[str, Any],
    execution_policy: Mapping[str, Any],
    receipts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build and validate one authoritative Candidate four-gate object."""
    receipts_copy = copy.deepcopy(dict(receipts))
    if set(receipts_copy) != set(CANDIDATE_GATE_NAMES):
        raise EvidenceError("candidate_receipts_schema")
    package = receipts_copy["package_graph_ir"]
    parity = receipts_copy["numerical_parity"]
    capability = receipts_copy["declared_capability"]
    installed = receipts_copy["installed_product_path"]
    machine = {
        "schema_version": "3.0",
        "gate": CANDIDATE_GATE,
        "run_kind": CANDIDATE_RUN_KIND,
        "verdict": "pass",
        "commit": source.get("commit"),
        "source": copy.deepcopy(dict(source)),
        "admission_state": "candidate",
        "identity": copy.deepcopy(dict(identity)),
        "profile": copy.deepcopy(dict(profile)),
        "wheel_sha256": wheel_sha256,
        "build": copy.deepcopy(dict(build)),
        "environments": copy.deepcopy(dict(environments)),
        "execution_policy": copy.deepcopy(dict(execution_policy)),
        "commands": {
            "package_graph_ir": copy.deepcopy(package.get("command")),
            "reference_first": copy.deepcopy(
                parity.get("commands", {}).get("reference_first")
            ),
            "reference_replay": copy.deepcopy(
                parity.get("commands", {}).get("reference_replay")
            ),
            "parity": copy.deepcopy(parity.get("commands", {}).get("parity")),
            "declared_capability": copy.deepcopy(capability.get("command")),
            "installed_product_path": copy.deepcopy(installed.get("command")),
        },
        "artifacts": {
            "source_snapshot_tree_sha256": package.get("source_snapshot", {}).get(
                "tree_sha256"
            ),
            "package_tree_sha256": package.get("package_conversion", {}).get(
                "tree_sha256"
            ),
            "package_manifest_sha256": package.get("package_conversion", {}).get(
                "manifest_sha256"
            ),
            "package_weights_sha256": package.get("package_conversion", {}).get(
                "weights_sha256"
            ),
            "graph_ir_sha256": package.get("package_conversion", {}).get(
                "graph_ir_sha256"
            ),
            "reference_sha256": parity.get("artifacts", {}).get(
                "reference_sha256"
            ),
            "reference_sidecar_sha256": parity.get("artifacts", {}).get(
                "reference_sidecar_sha256"
            ),
            "reference_replay_sha256": parity.get("artifacts", {}).get(
                "reference_replay_sha256"
            ),
            "reference_replay_sidecar_sha256": parity.get("artifacts", {}).get(
                "reference_replay_sidecar_sha256"
            ),
        },
        "gates": {
            name: {
                "receipt_sha256": receipt_digest(receipts_copy[name]),
                "receipt": receipts_copy[name],
            }
            for name in CANDIDATE_GATE_NAMES
        },
    }
    validate_candidate_machine(machine)
    return machine


def validate_candidate_machine(machine: object) -> dict[str, Any]:
    """Strictly validate a canonical Candidate four-gate machine object."""
    machine = _exact_dict(
        machine,
        {
            "schema_version",
            "gate",
            "run_kind",
            "verdict",
            "commit",
            "source",
            "admission_state",
            "identity",
            "profile",
            "wheel_sha256",
            "build",
            "environments",
            "execution_policy",
            "commands",
            "artifacts",
            "gates",
        },
        "candidate_machine_schema",
    )
    source = _exact_dict(
        machine["source"],
        {"commit", "tree", "archive_sha256", "tree_clean"},
        "source_schema",
    )
    if (
        machine["schema_version"] != "3.0"
        or machine["gate"] != CANDIDATE_GATE
        or machine["run_kind"] != CANDIDATE_RUN_KIND
        or machine["verdict"] != "pass"
        or machine["admission_state"] != "candidate"
        or not _is_oid(machine["commit"])
        or source["commit"] != machine["commit"]
        or not _is_oid(source["tree"])
        or not _is_sha(source["archive_sha256"])
        or source["tree_clean"] is not True
        or not _is_sha(machine["wheel_sha256"])
    ):
        raise EvidenceError("candidate_machine_identity")
    identity = _validate_identity(machine["identity"])
    profile = _validate_profile(machine["profile"])
    try:
        from .producer_descriptors import descriptor_for_key

        descriptor = descriptor_for_key(identity["candidate_key"])
        admission = descriptor.resolve_backend(
            profile["backend"], profile["precision"]
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise EvidenceError(f"candidate_descriptor_invalid: {error}") from None
    if identity != {
        "candidate_key": descriptor.key,
        "model_id": descriptor.model_id,
        "revision": descriptor.revision,
        "descriptor_digest": descriptor.descriptor_digest,
        "case_sha256": descriptor.parity_case_digest,
        "oracle_lock_sha256": descriptor.oracle_lock_digest,
    }:
        raise EvidenceError("candidate_descriptor_identity_mismatch")
    if profile != {
        "backend": admission.backend,
        "precision": admission.precision,
        "minimum_device": admission.minimum_device,
    }:
        raise EvidenceError("candidate_profile_mismatch")
    canonical_commands = expected_commands(descriptor.key)
    environments = _exact_dict(
        machine["environments"], {"product", "oracle"}, "environments_schema"
    )
    product_environment = _validate_environment(
        environments["product"],
        wheel_sha256=machine["wheel_sha256"],
        product=True,
        profile=profile,
        descriptor=descriptor,
    )
    validate_build_receipt(
        machine["build"],
        source_sha256=source["archive_sha256"],
        wheel_sha256=machine["wheel_sha256"],
        python_version=product_environment["python"]["version"],
        expected_wheel_filename=(
            f"diffusor_rt-{product_environment['packages']['diffusor-rt']}-py3-none-any.whl"
        ),
    )
    _validate_environment(
        environments["oracle"],
        wheel_sha256=machine["wheel_sha256"],
        product=False,
        profile=profile,
        descriptor=descriptor,
    )
    policy = _exact_dict(
        machine["execution_policy"],
        {
            "product_trust_remote_code",
            "reference_model_code_execution",
            "reference_code_consent",
            "provisional_bootstrap_candidate",
        },
        "execution_policy_schema",
    )
    if policy != {
        "product_trust_remote_code": False,
        "reference_model_code_execution": True,
        "reference_code_consent": True,
        "provisional_bootstrap_candidate": None,
    }:
        raise EvidenceError("execution_policy_invalid")

    gates = _exact_dict(
        machine["gates"], set(CANDIDATE_GATE_NAMES), "candidate_gates_schema"
    )
    embedded: dict[str, dict[str, Any]] = {}
    for name in CANDIDATE_GATE_NAMES:
        gate = _exact_dict(
            gates[name], {"receipt_sha256", "receipt"}, "embedded_receipt_schema"
        )
        if gate["receipt_sha256"] != receipt_digest(gate["receipt"]):
            raise EvidenceError(f"receipt_digest_mismatch: {name}")
        embedded[name] = gate["receipt"]

    package = _validate_package_receipt(
        embedded["package_graph_ir"],
        identity=identity,
        profile=profile,
        wheel_sha256=machine["wheel_sha256"],
        descriptor=descriptor,
        commands=canonical_commands,
    )
    package_snapshot = package["stable_package_snapshot"]
    product_code_snapshot = product_environment["code_snapshot"]
    if package["stable_code_snapshot"] != product_code_snapshot:
        raise EvidenceError("package_code_environment_mismatch")
    parity = _validate_parity_receipt(
        embedded["numerical_parity"],
        identity=identity,
        profile=profile,
        wheel_sha256=machine["wheel_sha256"],
        package_snapshot=package_snapshot,
        code_snapshot=product_code_snapshot,
        package_weights_sha256=package["package_conversion"]["weights_sha256"],
        package_vocab_size=package["package_conversion"]["vocab_size"],
        descriptor=descriptor,
        commands=canonical_commands,
    )
    capability = _validate_capability_receipt(
        embedded["declared_capability"],
        identity=identity,
        wheel_sha256=machine["wheel_sha256"],
        package_snapshot=package_snapshot,
        code_snapshot=product_code_snapshot,
        descriptor=descriptor,
        commands=canonical_commands,
    )
    installed = _validate_installed_receipt(
        embedded["installed_product_path"],
        identity=identity,
        profile=profile,
        wheel_sha256=machine["wheel_sha256"],
        package_snapshot=package_snapshot,
        code_snapshot=product_code_snapshot,
        descriptor=descriptor,
        commands=canonical_commands,
    )

    embedded_commands = {
        "package_graph_ir": package["command"],
        "reference_first": parity["commands"]["reference_first"],
        "reference_replay": parity["commands"]["reference_replay"],
        "parity": parity["commands"]["parity"],
        "declared_capability": capability["command"],
        "installed_product_path": installed["command"],
    }
    if embedded_commands != canonical_commands or machine["commands"] != canonical_commands:
        raise EvidenceError("candidate_commands_mismatch")
    expected_artifacts = {
        "source_snapshot_tree_sha256": package["source_snapshot"]["tree_sha256"],
        "package_tree_sha256": package["package_conversion"]["tree_sha256"],
        "package_manifest_sha256": package["package_conversion"]["manifest_sha256"],
        "package_weights_sha256": package["package_conversion"]["weights_sha256"],
        "graph_ir_sha256": package["package_conversion"]["graph_ir_sha256"],
        "reference_sha256": parity["artifacts"]["reference_sha256"],
        "reference_sidecar_sha256": parity["artifacts"][
            "reference_sidecar_sha256"
        ],
        "reference_replay_sha256": parity["artifacts"][
            "reference_replay_sha256"
        ],
        "reference_replay_sidecar_sha256": parity["artifacts"][
            "reference_replay_sidecar_sha256"
        ],
    }
    if machine["artifacts"] != expected_artifacts:
        raise EvidenceError("candidate_artifacts_mismatch")
    return machine


def render_candidate_report(machine: Mapping[str, Any]) -> bytes:
    """Render the report deterministically; no data exists only in Markdown."""
    validate_candidate_machine(machine)
    identity = machine["identity"]
    profile = machine["profile"]
    lines = [
        CANDIDATE_REPORT_HEADING,
        "",
        f"- Verdict: `{machine['verdict']}`",
        f"- Producer commit: `{machine['commit']}`",
        f"- Machine evidence SHA-256: `{machine_digest(machine)}`",
        f"- Candidate: `{identity['candidate_key']}`",
        f"- Model: `{identity['model_id']}`",
        f"- Revision: `{identity['revision']}`",
        f"- Descriptor digest: `{identity['descriptor_digest']}`",
        f"- Wheel SHA-256: `{machine['wheel_sha256']}`",
        "- Profile: "
        f"`{profile['backend']}` / `{profile['precision']}` / "
        f"`{profile['minimum_device']}`",
        "",
        "## Four gates",
        "",
    ]
    for name in CANDIDATE_GATE_NAMES:
        gate = machine["gates"][name]
        lines.append(
            f"- `{name}`: `pass`; receipt SHA-256 "
            f"`{gate['receipt_sha256']}`"
        )
    lines.extend(
        [
            "",
            "## Canonical machine evidence",
            "",
            "```json",
            canonical_json_bytes(machine).decode("ascii").rstrip("\n"),
            "```",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


__all__ = [
    "CANDIDATE_CAPABILITY_CHECKS",
    "CANDIDATE_GATE",
    "CANDIDATE_GATE_NAMES",
    "CANDIDATE_PRODUCT_LOCK_RESOURCE",
    "CANDIDATE_REPORT_HEADING",
    "CANDIDATE_RUN_KIND",
    "EvidenceError",
    "build_candidate_machine",
    "candidate_product_lock_resource",
    "canonical_json_bytes",
    "expected_commands",
    "load_candidate_product_lock",
    "machine_digest",
    "receipt_digest",
    "render_candidate_report",
    "strict_json_loads",
    "validate_candidate_machine",
]
