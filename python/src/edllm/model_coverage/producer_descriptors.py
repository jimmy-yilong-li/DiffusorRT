"""Coverage evidence and the legacy flat descriptor projection.

Executable identity and behavior live in :mod:`edllm.model_definitions`.
``CoverageRecord`` owns resource, reference, parity, capability and backend
evidence only. ``ModelDescriptor`` remains a digest-compatible projection for
existing admission tools; no product loader reads runtime policy from it.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator, Mapping
import copy
from dataclasses import asdict, dataclass
from enum import StrEnum
from functools import cache
import hashlib
from importlib import resources
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any

from ..model_definitions import (
    DecodeContract,
    DeploymentContract,
    ModelDefinition,
    model_definition_for_catalog_key,
    model_definition_for_descriptor,
)

_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_KNOWN_BACKENDS = frozenset({"mlx_local"})
_KNOWN_PRECISIONS = frozenset({"fp32", "bf16", "int8_weight_only", "fp8"})
_KNOWN_UPSTREAM_TYING_FIELDS = frozenset(
    {"tie_word_embeddings", "weight_tying"}
)
_CAPABILITY_REQUIRED_CHECKS: dict[str, frozenset[str]] = {
    "chat": frozenset({"nonempty", "no_mask_token"}),
    "infill": frozenset(
        {
            "nonempty",
            "no_mask_token",
            "prefix_preserved",
            "suffix_preserved",
            "holes_filled",
        }
    ),
    "code-infill": frozenset(
        {
            "nonempty",
            "no_mask_token",
            "prefix_preserved",
            "suffix_preserved",
            "holes_filled",
            "fixed_signature",
            "python_parse",
        }
    ),
}


def _canonical_required_signature_function(
    required_signature: object,
) -> ast.FunctionDef | None:
    if type(required_signature) is not str or not required_signature:
        return None
    try:
        module = ast.parse(f"{required_signature}\n    pass")
    except (SyntaxError, ValueError, TypeError):
        return None
    if len(module.body) != 1 or type(module.body[0]) is not ast.FunctionDef:
        return None
    function = module.body[0]
    if function.decorator_list or getattr(function, "type_params", ()):
        return None
    if len(function.body) != 1 or type(function.body[0]) is not ast.Pass:
        return None
    return function


class LogitsConvention(StrEnum):
    SHIFTED = "shifted"
    UNSHIFTED = "unshifted"


@dataclass(frozen=True, slots=True)
class NumericTier:
    key: str
    atol: float
    rtol: float
    exact_argmax: bool
    exact_ordered_top_k: int


@dataclass(frozen=True, slots=True)
class CapabilityCase:
    kind: str
    prompt: str
    seed: int
    max_tokens: int
    required_checks: tuple[str, ...]
    prefix: str | None = None
    suffix: str | None = None
    required_signature: str | None = None
    mask_token_text: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "required_checks", tuple(self.required_checks))

    def validate(self) -> None:
        if type(self.kind) is not str or self.kind not in _CAPABILITY_REQUIRED_CHECKS:
            raise ValueError(f"capability_kind_unknown: {self.kind!r}")
        if type(self.prompt) is not str or not self.prompt.strip():
            raise ValueError("capability_prompt_invalid")
        if type(self.seed) is not int or self.seed != 0:
            raise ValueError("capability_seed_invalid")
        if type(self.max_tokens) is not int or self.max_tokens <= 0:
            raise ValueError("capability_max_tokens_invalid")
        if type(self.required_checks) is not tuple or not all(
            type(check) is str and check for check in self.required_checks
        ):
            raise ValueError("capability_required_checks_invalid")
        if len(set(self.required_checks)) != len(self.required_checks):
            raise ValueError("capability_required_check_duplicate")

        required = _CAPABILITY_REQUIRED_CHECKS[self.kind]
        unknown = set(self.required_checks) - required
        if unknown:
            raise ValueError(f"capability_check_unknown: {sorted(unknown)!r}")
        missing = required - set(self.required_checks)
        if missing:
            raise ValueError(f"capability_required_check_missing: {sorted(missing)!r}")

        for name, value in (
            ("prefix", self.prefix),
            ("suffix", self.suffix),
            ("required_signature", self.required_signature),
            ("mask_token_text", self.mask_token_text),
        ):
            if value is not None and (type(value) is not str or not value):
                raise ValueError(f"capability_{name}_invalid")
        if self.mask_token_text is not None and not self.mask_token_text.strip():
            raise ValueError("capability_mask_token_text_invalid")

        if self.kind == "chat":
            if any(
                value is not None
                for value in (self.prefix, self.suffix, self.required_signature)
            ):
                raise ValueError("capability_chat_field_forbidden")
            return

        if self.prefix is None or self.suffix is None:
            raise ValueError("capability_infill_context_missing")
        if self.mask_token_text is None:
            raise ValueError("capability_mask_token_text_missing")
        if self.kind == "infill":
            if self.required_signature is not None:
                raise ValueError("capability_infill_signature_forbidden")
            return
        if self.required_signature is None:
            raise ValueError("capability_required_signature_missing")
        if _canonical_required_signature_function(self.required_signature) is None:
            raise ValueError("capability_required_signature_invalid")


@dataclass(frozen=True, slots=True)
class BackendAdmission:
    backend: str
    precision: str
    minimum_device: str
    evidence: str
    deployment: DeploymentContract | None = None


@dataclass(frozen=True, slots=True)
class BackendEvidence:
    """Evidence attached to a deployment target owned by ModelDefinition."""

    backend: str
    precision: str
    evidence: str


@dataclass(frozen=True, slots=True)
class CoverageRecord:
    """Reference, parity, resource, and admission evidence for one model.

    This record cannot alter conversion, tokenizer, GraphIR, decode, cache, or
    deployment behavior. Those decisions belong to the linked ModelDefinition.
    """

    key: str
    model_definition_key: str
    product_license_decision: str
    trust_remote_code: bool
    snapshot_bytes: int
    package_bytes: int
    peak_rss_bytes: int
    resource_envelope_evidence: str
    reference_adapter: str
    parity_case_resource: str
    parity_case_digest: str
    oracle_lock_resource: str
    oracle_lock_digest: str
    logits_convention: LogitsConvention
    comparison_dtype: str
    numeric_tier: NumericTier
    capability_case: CapabilityCase
    backend_evidence: tuple[BackendEvidence, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend_evidence", tuple(self.backend_evidence))

    def validate(self, definition: ModelDefinition) -> None:
        if self.key != definition.key:
            raise ValueError(f"coverage_definition_identity_mismatch: {self.key}")
        if self.model_definition_key != definition.key:
            raise ValueError(f"coverage_definition_key_mismatch: {self.key}")
        if type(self.trust_remote_code) is not bool:
            raise ValueError(f"coverage_trust_remote_code_invalid: {self.key}")
        if min(self.snapshot_bytes, self.package_bytes, self.peak_rss_bytes) <= 0:
            raise ValueError(f"coverage_resource_envelope_invalid: {self.key}")
        targets = {
            (target.backend, target.precision): target
            for target in definition.deployments
        }
        seen: set[tuple[str, str]] = set()
        for evidence in self.backend_evidence:
            identity = (evidence.backend, evidence.precision)
            if identity not in targets:
                raise ValueError(f"coverage_deployment_target_unknown: {identity!r}")
            if identity in seen:
                raise ValueError(f"coverage_deployment_evidence_duplicate: {identity!r}")
            if not evidence.evidence:
                raise ValueError(f"coverage_deployment_evidence_missing: {identity!r}")
            seen.add(identity)


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    """Digest-compatible projection of ModelDefinition plus CoverageRecord."""
    key: str
    model_id: str
    revision: str
    source_license: str
    product_license_decision: str
    expected_architectures: tuple[str, ...]
    expected_model_types: tuple[str, ...]
    runtime_architecture: str
    model_family: str
    trust_remote_code: bool
    custom_tokenizer_module: str | None
    mask_token_id: int
    tie_word_embeddings: bool
    snapshot_bytes: int
    package_bytes: int
    peak_rss_bytes: int
    resource_envelope_evidence: str
    converter_adapter: str
    reference_adapter: str
    runtime_adapter: str
    graph_ir_family: str
    architecture_features: tuple[str, ...]
    parity_case_resource: str
    parity_case_digest: str
    oracle_lock_resource: str
    oracle_lock_digest: str
    logits_convention: LogitsConvention
    comparison_dtype: str
    numeric_tier: NumericTier
    capability_case: CapabilityCase
    admitted_backends: tuple[BackendAdmission, ...]
    upstream_tying_field: str | None = None
    decode_contract: DecodeContract | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "expected_architectures",
            tuple(self.expected_architectures),
        )
        object.__setattr__(self, "expected_model_types", tuple(self.expected_model_types))
        object.__setattr__(
            self,
            "architecture_features",
            tuple(self.architecture_features),
        )
        object.__setattr__(self, "admitted_backends", tuple(self.admitted_backends))

    @property
    def effective_upstream_tying_field(self) -> str:
        return self.upstream_tying_field or "tie_word_embeddings"

    @property
    def descriptor_digest(self) -> str:
        """Digest every immutable producer and gate input."""
        data = asdict(self)
        if data["upstream_tying_field"] is None:
            data.pop("upstream_tying_field")
        if data["decode_contract"] is None:
            data.pop("decode_contract")
        for admission in data["admitted_backends"]:
            if admission["deployment"] is None:
                admission.pop("deployment")
            else:
                # Product packages and existing admission receipts bind the
                # v2 flat descriptor format. ``attention_mask_kind`` moved an
                # already-existing profile semantic into ModelDefinition; it
                # is runtime truth, not a new certification input. Keep this
                # compatibility digest byte-stable while the execution package
                # continues to use ``model_descriptor_digest``.
                admission["deployment"].pop("attention_mask_kind", None)
        capability = data["capability_case"]
        for field in (
            "prefix",
            "suffix",
            "required_signature",
            "mask_token_text",
        ):
            if capability[field] is None:
                capability.pop(field)
        payload = json.dumps(
            data, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def resolve_backend(self, backend: str, precision: str) -> BackendAdmission:
        if backend not in _KNOWN_BACKENDS:
            raise ValueError(f"unknown_backend: {backend}")
        if precision not in _KNOWN_PRECISIONS:
            raise ValueError(f"unknown_precision: {precision}")
        for admission in self.admitted_backends:
            if admission.backend == backend and admission.precision == precision:
                return admission
        raise ValueError(
            f"unsupported_backend_precision: {self.key}:{backend}:{precision}"
        )

    def validate(self) -> None:
        if not self.key or not self.model_id:
            raise ValueError("descriptor_identity_missing")
        if not _SHA40.fullmatch(self.revision):
            raise ValueError(f"descriptor_revision_invalid: {self.key}")
        if not self.expected_architectures or not self.expected_model_types:
            raise ValueError(f"descriptor_architecture_missing: {self.key}")
        if not self.converter_adapter or not self.reference_adapter:
            raise ValueError(f"descriptor_adapter_missing: {self.key}")
        if type(self.mask_token_id) is not int or self.mask_token_id < 0:
            raise ValueError(f"descriptor_mask_token_id_invalid: {self.key}")
        if type(self.trust_remote_code) is not bool:
            raise ValueError(f"descriptor_trust_remote_code_invalid: {self.key}")
        if type(self.tie_word_embeddings) is not bool:
            raise ValueError(f"descriptor_tie_word_embeddings_invalid: {self.key}")
        if self.upstream_tying_field is not None:
            if type(self.upstream_tying_field) is not str:
                raise ValueError(
                    f"descriptor_upstream_tying_field_invalid: {self.key}"
                )
            if self.upstream_tying_field not in _KNOWN_UPSTREAM_TYING_FIELDS:
                raise ValueError(
                    f"descriptor_upstream_tying_field_unknown: "
                    f"{self.upstream_tying_field!r}"
                )
        # Conversion and runtime semantics are two halves of one model
        # definition. Validate their join after the descriptor's own fields so
        # malformed local values keep their precise public diagnostics.
        from ..model_definitions import validate_descriptor_binding

        validate_descriptor_binding(self)
        if type(self.capability_case) is not CapabilityCase:
            raise ValueError(f"descriptor_capability_case_invalid: {self.key}")
        self.capability_case.validate()
        if self.decode_contract is not None:
            if type(self.decode_contract) is not DecodeContract:
                raise ValueError(f"descriptor_decode_contract_invalid: {self.key}")
            self.decode_contract.validate()
        tied_marker = "tied_embeddings_declared"
        untied_marker = "untied_lm_head"
        expected_marker = tied_marker if self.tie_word_embeddings else untied_marker
        conflicting_marker = untied_marker if self.tie_word_embeddings else tied_marker
        if (
            expected_marker not in self.architecture_features
            or conflicting_marker in self.architecture_features
        ):
            raise ValueError(
                f"descriptor_tied_embedding_feature_mismatch: {self.key}"
            )
        if min(self.snapshot_bytes, self.package_bytes, self.peak_rss_bytes) <= 0:
            raise ValueError(f"descriptor_resource_envelope_invalid: {self.key}")
        if not _SHA256.fullmatch(self.parity_case_digest):
            raise ValueError(f"descriptor_case_digest_invalid: {self.key}")
        if not _SHA256.fullmatch(self.oracle_lock_digest):
            raise ValueError(f"descriptor_lock_digest_invalid: {self.key}")
        _verify_resource_digest(
            self.parity_case_resource,
            self.parity_case_digest,
            mismatch="case_digest_mismatch",
        )
        _verify_resource_digest(
            self.oracle_lock_resource,
            self.oracle_lock_digest,
            mismatch="lock_digest_mismatch",
        )
        seen: set[tuple[str, str]] = set()
        for admission in self.admitted_backends:
            if admission.backend not in _KNOWN_BACKENDS:
                raise ValueError(f"unknown_backend: {admission.backend}")
            if admission.precision not in _KNOWN_PRECISIONS:
                raise ValueError(f"unknown_precision: {admission.precision}")
            combination = (admission.backend, admission.precision)
            if combination in seen:
                raise ValueError(
                    f"duplicate_backend_precision: {self.key}:{combination}"
                )
            seen.add(combination)
            if admission.deployment is not None:
                if type(admission.deployment) is not DeploymentContract:
                    raise ValueError(
                        f"deployment_contract_invalid: {self.key}:{combination}"
                    )
                admission.deployment.validate()
                if (
                    admission.deployment.config_source == "package"
                    and self.decode_contract is None
                ):
                    raise ValueError(
                        f"package_deployment_contract_missing: {self.key}:{combination}"
                    )


def _compose_descriptor(
    definition: ModelDefinition,
    coverage: CoverageRecord,
) -> ModelDescriptor:
    """Build the legacy flat view without creating another source of truth."""

    coverage.validate(definition)
    targets = {
        (target.backend, target.precision): target
        for target in definition.deployments
    }
    admissions = tuple(
        BackendAdmission(
            backend=evidence.backend,
            precision=evidence.precision,
            minimum_device=targets[(evidence.backend, evidence.precision)].minimum_device,
            evidence=evidence.evidence,
            deployment=targets[(evidence.backend, evidence.precision)].deployment,
        )
        for evidence in coverage.backend_evidence
    )
    expected_logits = (
        LogitsConvention.SHIFTED
        if definition.runtime.uses_shifted_logits
        else LogitsConvention.UNSHIFTED
    )
    if coverage.logits_convention is not expected_logits:
        raise ValueError(f"coverage_logits_convention_mismatch: {coverage.key}")
    return ModelDescriptor(
        key=definition.key,
        model_id=definition.model_id,
        revision=definition.revision,
        source_license=definition.source_license,
        product_license_decision=coverage.product_license_decision,
        expected_architectures=definition.expected_architectures,
        expected_model_types=definition.expected_model_types,
        runtime_architecture=definition.runtime_architecture,
        model_family=definition.model_family,
        trust_remote_code=coverage.trust_remote_code,
        custom_tokenizer_module=definition.custom_tokenizer_module,
        mask_token_id=definition.mask_token_id,
        tie_word_embeddings=definition.tie_word_embeddings,
        snapshot_bytes=coverage.snapshot_bytes,
        package_bytes=coverage.package_bytes,
        peak_rss_bytes=coverage.peak_rss_bytes,
        resource_envelope_evidence=coverage.resource_envelope_evidence,
        converter_adapter=definition.converter.key,
        reference_adapter=coverage.reference_adapter,
        runtime_adapter=definition.runtime.key,
        graph_ir_family=definition.graph_ir_family,
        architecture_features=definition.architecture_features,
        parity_case_resource=coverage.parity_case_resource,
        parity_case_digest=coverage.parity_case_digest,
        oracle_lock_resource=coverage.oracle_lock_resource,
        oracle_lock_digest=coverage.oracle_lock_digest,
        logits_convention=coverage.logits_convention,
        comparison_dtype=coverage.comparison_dtype,
        numeric_tier=coverage.numeric_tier,
        capability_case=coverage.capability_case,
        admitted_backends=admissions,
        upstream_tying_field=(
            None
            if definition.upstream_tying_field == "tie_word_embeddings"
            else definition.upstream_tying_field
        ),
        decode_contract=definition.decode_contract,
    )


def candidate_runtime_policy(descriptor: ModelDescriptor) -> dict[str, object]:
    """Return the exact runtime policy embedded in Candidate gate receipts."""
    definition = model_definition_for_catalog_key(descriptor.key)
    if definition.decode_contract is not None:
        return definition.decode_contract.evidence_policy()
    defaults = definition.decode_policy.resolve({})
    policy = {
        "name": "baseline-no-d2f",
        "decode_strategy": defaults["decode_strategy"],
        "attention_mask_kind": "bidirectional",
        "d2f_exact_kv_cache": False,
    }
    if policy["decode_strategy"] == "fixed_quota":
        policy["name"] = definition.decode_policy.key
        policy.update({
            key: defaults[key]
            for key in ("block_size", "steps", "temperature", "top_p", "remasking")
        })
    return policy


def _resource_bytes(relative: str) -> bytes:
    return resources.files(__package__).joinpath(relative).read_bytes()


def _verify_resource_digest(relative: str, expected: str, *, mismatch: str) -> None:
    actual = hashlib.sha256(_resource_bytes(relative)).hexdigest()
    if actual != expected:
        raise ValueError(f"{mismatch}: {relative}: expected={expected} actual={actual}")


def _exact_json_equal(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _exact_json_equal(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _exact_json_equal(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return actual == expected


def load_case_corpus(descriptor: ModelDescriptor) -> dict[str, Any]:
    _verify_resource_digest(
        descriptor.parity_case_resource,
        descriptor.parity_case_digest,
        mismatch="case_digest_mismatch",
    )
    corpus = json.loads(_resource_bytes(descriptor.parity_case_resource))
    if corpus.get("schema_version") != 1:
        raise ValueError("case_schema_unsupported")
    if corpus.get("candidate_key") != descriptor.key:
        raise ValueError("case_candidate_mismatch")
    cases = corpus.get("cases")
    if not isinstance(cases, list) or len(cases) < 3:
        raise ValueError("case_corpus_too_small")
    names: set[str] = set()
    for case in cases:
        _validate_case(case, descriptor=descriptor)
        name = case["name"]
        if name in names:
            raise ValueError(f"duplicate_case_name: {name}")
        names.add(name)
    return corpus


def _validate_case(case: object, *, descriptor: ModelDescriptor) -> None:
    if not isinstance(case, dict):
        raise ValueError("case_not_object")
    required = ("name", "input_ids", "masked_positions", "selected_rows")
    if any(name not in case for name in required):
        raise ValueError("case_field_missing")
    input_ids = case["input_ids"]
    masked = case["masked_positions"]
    selected = case["selected_rows"]
    if not isinstance(input_ids, list) or not input_ids:
        raise ValueError("case_input_ids_invalid")
    if not all(type(token) is int and token >= 0 for token in input_ids):
        raise ValueError("case_input_ids_invalid")
    if not isinstance(masked, list) or len(masked) < 4:
        raise ValueError("case_masked_positions_invalid")
    if not isinstance(selected, list) or not selected:
        raise ValueError("case_selected_rows_invalid")
    limit = len(input_ids)
    if any(
        type(row) is not int or not 0 <= row < limit for row in (*masked, *selected)
    ):
        raise ValueError("case_row_out_of_range")
    if len(set(masked)) != len(masked):
        raise ValueError("case_masked_positions_duplicate")
    if len(set(selected)) != len(selected):
        raise ValueError("case_selected_rows_duplicate")
    if any(input_ids[row] != descriptor.mask_token_id for row in masked):
        raise ValueError("case_mask_token_mismatch")
    if descriptor.logits_convention == LogitsConvention.UNSHIFTED and not set(
        selected
    ).issubset(masked):
        raise ValueError("case_unshifted_rows_must_be_masked")


def load_oracle_lock(descriptor: ModelDescriptor) -> dict[str, Any]:
    _verify_resource_digest(
        descriptor.oracle_lock_resource,
        descriptor.oracle_lock_digest,
        mismatch="lock_digest_mismatch",
    )
    lock = json.loads(_resource_bytes(descriptor.oracle_lock_resource))
    if lock.get("schema_version") != 1:
        raise ValueError("lock_schema_unsupported")
    packages = lock.get("packages")
    if not isinstance(packages, dict) or not packages:
        raise ValueError("lock_packages_missing")
    if any(
        not isinstance(name, str) or not isinstance(version, str)
        for name, version in packages.items()
    ):
        raise ValueError("lock_package_invalid")
    return lock


def validate_snapshot(
    descriptor: ModelDescriptor | ModelDefinition,
    snapshot: str | Path,
    *,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prove exact HF cache identity and architecture before opening weights."""
    from ..model_definitions import validate_model_snapshot

    # Accepting the historical flat descriptor keeps evidence-tool mutation
    # tests source compatible; normal conversion passes ModelDefinition.
    return validate_model_snapshot(descriptor, snapshot, config=config)


def validate_execution_package(
    descriptor: ModelDescriptor,
    package: str | Path,
    *,
    verify_weights_hash: bool = True,
    manifest_snapshot: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], str | None]:
    """Validate descriptor, identity, adapter, backend, precision, and digest.

    ``manifest_snapshot`` is the already-opened package manifest.  Callers that
    supply it bind every subsequent check to those bytes instead of reopening
    ``model.json`` and creating a manifest TOCTOU window.
    """
    root = Path(package).resolve()
    definition = model_definition_for_catalog_key(descriptor.key)
    if manifest_snapshot is None:
        manifest_path = root / "model.json"
        if not manifest_path.is_file():
            raise ValueError("package_manifest_missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        if not isinstance(manifest_snapshot, Mapping):
            raise ValueError("package_manifest_invalid")
        manifest = copy.deepcopy(dict(manifest_snapshot))
    expected = {
        "model_descriptor_key": descriptor.key,
        "model_descriptor_digest": descriptor.descriptor_digest,
        "source_checkpoint_reference": definition.model_id,
        "source_base_model_reference": definition.model_id,
        "source_checkpoint_sha": definition.revision,
        "source_base_model_sha": definition.revision,
        "source_license": definition.source_license,
        "graph_ir_family": definition.graph_ir_family,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(
                f"package_{field}_mismatch: expected={value!r} "
                f"actual={manifest.get(field)!r}"
            )
    definition_key = manifest.get("model_definition_key")
    if definition_key is not None and definition_key != definition.key:
        raise ValueError(
            "package_model_definition_key_mismatch: "
            f"expected={definition.key!r} actual={definition_key!r}"
        )
    model_config = manifest.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("package_model_config_missing")
    if model_config.get("architecture") != definition.runtime_architecture:
        raise ValueError("package_runtime_architecture_mismatch")
    if model_config.get("architecture_family") != definition.runtime_adapter:
        raise ValueError("package_runtime_adapter_mismatch")
    actual_mask_token_id = model_config.get("mask_token_id")
    if (
        type(actual_mask_token_id) is not int
        or actual_mask_token_id != definition.mask_token_id
    ):
        raise ValueError(
            "package_mask_token_id_mismatch: "
            f"expected={definition.mask_token_id!r} actual={actual_mask_token_id!r}"
        )
    actual_tie_word_embeddings = model_config.get("tie_word_embeddings")
    if (
        type(actual_tie_word_embeddings) is not bool
        or actual_tie_word_embeddings is not definition.tie_word_embeddings
    ):
        raise ValueError(
            "package_tie_word_embeddings_mismatch: "
            f"expected={definition.tie_word_embeddings!r} "
            f"actual={actual_tie_word_embeddings!r}"
        )
    if definition.decode_contract is not None:
        for field, expected_value in (
            definition.decode_contract.package_model_config().items()
        ):
            if not _exact_json_equal(model_config.get(field), expected_value):
                raise ValueError(
                    "package_decode_contract_mismatch: "
                    f"field={field!r} expected={expected_value!r} "
                    f"actual={model_config.get(field)!r}"
                )
    backend = manifest.get("admitted_backend")
    precision = manifest.get("model_precision")
    if not isinstance(backend, str):
        raise ValueError("package_backend_missing")
    if not isinstance(precision, str):
        raise ValueError("package_precision_missing")
    descriptor.resolve_backend(backend, precision)
    deployment = definition.resolve_deployment(backend, precision).deployment
    # Package-owned masks, including the identity "baseline" preset, must
    # already satisfy the deployment contract. A non-baseline preset is checked
    # after its overlay is applied by the runtime plan resolver.
    if deployment is not None and (
        deployment.config_source == "package" or deployment.profile == "baseline"
    ):
        deployment.validate_attention_mask_kind(
            model_config.get("attention_mask_kind", "bidirectional")
        )

    graph_ref = manifest.get("graph_ref")
    if not isinstance(graph_ref, str) or not graph_ref:
        raise ValueError("package_graph_ref_missing")
    graph_path = (root / graph_ref).resolve()
    if root not in graph_path.parents or not graph_path.is_file():
        raise ValueError("package_graph_missing_or_unsafe")
    weights_ref = manifest.get("merged_weights_ref")
    if not isinstance(weights_ref, str) or not weights_ref:
        raise ValueError("package_weights_ref_missing")
    weights_path = (root / weights_ref).resolve()
    if root not in weights_path.parents or not weights_path.is_file():
        raise ValueError("package_weights_missing_or_unsafe")
    actual_hash = None
    if verify_weights_hash:
        digest = hashlib.sha256()
        with weights_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        actual_hash = digest.hexdigest()
        if manifest.get("output_hash") != actual_hash:
            raise ValueError("package_weights_digest_mismatch")
    return manifest, actual_hash


_DENSE_FP32 = NumericTier(
    key="dense_fp32_v1",
    atol=2e-4,
    rtol=2e-5,
    exact_argmax=True,
    exact_ordered_top_k=5,
)

_DREAM_SDPA_FP32 = NumericTier(
    key="dream_sdpa_fp32_v1",
    atol=5e-4,
    rtol=2e-5,
    exact_argmax=True,
    exact_ordered_top_k=5,
)


DREAM_MODEL_ID = model_definition_for_catalog_key("dream_v0_instruct_7b").model_id
DREAM_CODER_MODEL_ID = model_definition_for_catalog_key(
    "dream_coder_v0_instruct_7b"
).model_id
LLADA_MODEL_ID = model_definition_for_catalog_key("llada_8b_instruct").model_id
QWEN3_MDLM_MODEL_ID = model_definition_for_catalog_key(
    "qwen3_06b_mdlm_v0_1"
).model_id
QWEN3_BD3LM_MODEL_ID = model_definition_for_catalog_key(
    "qwen3_06b_bd3lm_v0_1"
).model_id


_COVERAGE_RECORD_DATA: dict[str, CoverageRecord] = {
    "dream_v0_instruct_7b": CoverageRecord(
        key="dream_v0_instruct_7b",
        model_definition_key="dream_v0_instruct_7b",
        product_license_decision="approved",
        trust_remote_code=True,
        snapshot_bytes=15_236_443_179,
        package_bytes=30_462_466_048,
        peak_rss_bytes=3_455_000_000,
        resource_envelope_evidence="compiler_registry_measurement_2026_07_30",
        reference_adapter="dream_hf_446_v1",
        parity_case_resource="data/cases/dream_v0_instruct_7b.json",
        parity_case_digest="924b4b0e7ca42bec000b267ecd1a0952eecf6e02e11f06b840ebcd0573e4735a",
        oracle_lock_resource="data/locks/dream_hf_446.json",
        oracle_lock_digest="7b9e34545659f27eecdc67d72932eef6a32da5d8106c9fdfdf744edb6eb4fb6a",
        logits_convention=LogitsConvention.SHIFTED,
        comparison_dtype="float32",
        numeric_tier=_DENSE_FP32,
        capability_case=CapabilityCase(
            kind="chat",
            prompt="The capital of France is",
            seed=0,
            max_tokens=8,
            required_checks=("nonempty", "no_mask_token"),
        ),
        backend_evidence=(
            BackendEvidence(
                backend="mlx_local",
                precision="fp32",
                evidence="client_p0_r3_2026_08_13",
            ),
        ),
    ),
    "qwen3_06b_mdlm_v0_1": CoverageRecord(
        key="qwen3_06b_mdlm_v0_1",
        model_definition_key="qwen3_06b_mdlm_v0_1",
        product_license_decision="approved",
        trust_remote_code=True,
        snapshot_bytes=1_519_203_744,
        package_bytes=3_022_613_227,
        peak_rss_bytes=4_000_000_000,
        resource_envelope_evidence=(
            "fixed-SHA dab73847ca368bca62549472d51ebe3f74a1d0c1 installed "
            "wheel on Apple M4 Max 64GB: 3022613227-byte package, "
            "1788854272-byte conversion peak, 3584950272-byte generation peak; "
            "4000000000-byte admission planning envelope"
        ),
        reference_adapter="qwen3_mdlm_hf_457_v1",
        parity_case_resource="data/cases/qwen3_06b_mdlm_v0_1.json",
        parity_case_digest="f12c50fc3c048e4469d80faf0c0837930936daddda105ae30d793d920f180585",
        oracle_lock_resource="data/locks/qwen3_hf_457.json",
        oracle_lock_digest="efac7bec12f20dccafa288cccd40bdeee946f03b760a1c61fe9a13f01ee7fcf3",
        logits_convention=LogitsConvention.UNSHIFTED,
        comparison_dtype="float32",
        numeric_tier=_DENSE_FP32,
        capability_case=CapabilityCase(
            kind="chat",
            prompt="Hello, who are you?",
            seed=0,
            max_tokens=40,
            required_checks=("nonempty", "no_mask_token"),
        ),
        backend_evidence=(
            BackendEvidence(
                backend="mlx_local",
                precision="fp32",
                evidence="i2_installed_wheel_2026_08_17_apple_m4_max_64gb",
            ),
        ),
    ),
    "qwen3_06b_bd3lm_v0_1": CoverageRecord(
        key="qwen3_06b_bd3lm_v0_1",
        model_definition_key="qwen3_06b_bd3lm_v0_1",
        product_license_decision="approved",
        trust_remote_code=True,
        snapshot_bytes=1_519_205_834,
        package_bytes=3_022_613_581,
        peak_rss_bytes=4_000_000_000,
        resource_envelope_evidence=(
            "2026-09-06 Candidate v1 measured a 3734929408-byte peak RSS; "
            "snapshot and package sizes are exact for the fixed revision"
        ),
        reference_adapter="qwen3_bd3lm_hf_457_v1",
        parity_case_resource="data/cases/qwen3_06b_bd3lm_v0_1.json",
        parity_case_digest=(
            "73a342e8999f99c8644ccbd847760c1f4d6e68d57c4b6ab6c800f4707474d21b"
        ),
        oracle_lock_resource="data/locks/qwen3_hf_457.json",
        oracle_lock_digest=(
            "efac7bec12f20dccafa288cccd40bdeee946f03b760a1c61fe9a13f01ee7fcf3"
        ),
        logits_convention=LogitsConvention.UNSHIFTED,
        comparison_dtype="float32",
        numeric_tier=_DENSE_FP32,
        capability_case=CapabilityCase(
            kind="chat",
            prompt="Hello, who are you?",
            seed=0,
            max_tokens=16,
            required_checks=("nonempty", "no_mask_token"),
        ),
        backend_evidence=(
            BackendEvidence(
                backend="mlx_local",
                precision="fp32",
                evidence="model_coverage_v1_receipt_required_before_promotion",
            ),
        ),
    ),
    "llada_8b_instruct": CoverageRecord(
        key="llada_8b_instruct",
        model_definition_key="llada_8b_instruct",
        product_license_decision="approved",
        trust_remote_code=True,
        snapshot_bytes=16_037_450_386,
        package_bytes=32_068_682_047,
        peak_rss_bytes=48_000_000_000,
        resource_envelope_evidence=(
            "2026-09-06 Candidate v2 measured a 27870871552-byte peak RSS; "
            "snapshot and fp32 package sizes are exact for the fixed revision"
        ),
        reference_adapter="llada_hf_446_v1",
        parity_case_resource="data/cases/llada_8b_instruct.json",
        parity_case_digest=(
            "218963e7f32583d784fc4f8ec117afb3177fafcd0dc015ef7f2ce413850a6fd5"
        ),
        oracle_lock_resource="data/locks/dream_hf_446.json",
        oracle_lock_digest=(
            "7b9e34545659f27eecdc67d72932eef6a32da5d8106c9fdfdf744edb6eb4fb6a"
        ),
        logits_convention=LogitsConvention.UNSHIFTED,
        comparison_dtype="float32",
        numeric_tier=_DENSE_FP32,
        capability_case=CapabilityCase(
            kind="infill",
            prompt="Explain bidirectional text infilling in one sentence.",
            seed=0,
            max_tokens=8,
            required_checks=(
                "nonempty",
                "no_mask_token",
                "prefix_preserved",
                "suffix_preserved",
                "holes_filled",
            ),
            prefix="The capital of ",
            suffix=" is Paris.",
            mask_token_text="<|mdm_mask|>",
        ),
        backend_evidence=(
            BackendEvidence(
                backend="mlx_local",
                precision="fp32",
                evidence="model_coverage_v1_receipt_required_before_promotion",
            ),
        ),
    ),
    "dream_coder_v0_instruct_7b": CoverageRecord(
        key="dream_coder_v0_instruct_7b",
        model_definition_key="dream_coder_v0_instruct_7b",
        product_license_decision="pending-owner-approval",
        trust_remote_code=True,
        snapshot_bytes=15_236_445_642,
        package_bytes=30_467_744_795,
        peak_rss_bytes=46_000_000_000,
        resource_envelope_evidence=(
            "2026-09-06 Candidate v2 measured a 28930703360-byte peak RSS; "
            "snapshot and fp32 package sizes are exact for the fixed revision"
        ),
        reference_adapter="dream_hf_446_v1",
        parity_case_resource="data/cases/dream_coder_v0_instruct_7b.json",
        parity_case_digest=(
            "b4f547f90e0f0f464e1e6ad6bc19521e28e4b035d8b002a928f90b6144af19c0"
        ),
        oracle_lock_resource="data/locks/dream_hf_446.json",
        oracle_lock_digest=(
            "7b9e34545659f27eecdc67d72932eef6a32da5d8106c9fdfdf744edb6eb4fb6a"
        ),
        logits_convention=LogitsConvention.SHIFTED,
        comparison_dtype="float32",
        numeric_tier=_DREAM_SDPA_FP32,
        capability_case=CapabilityCase(
            kind="code-infill",
            prompt="Write a short Python function that adds one to an integer.",
            seed=0,
            max_tokens=16,
            required_checks=(
                "nonempty",
                "no_mask_token",
                "prefix_preserved",
                "suffix_preserved",
                "holes_filled",
                "fixed_signature",
                "python_parse",
            ),
            prefix=(
                "# Write exactly one Python function named add_one with one "
                "parameter named value:\n"
            ),
            suffix="\nassert add_one(2) == 3",
            required_signature="def add_one(value):",
            mask_token_text="<|mask|>",
        ),
        backend_evidence=(
            BackendEvidence(
                backend="mlx_local",
                precision="fp32",
                evidence="model_coverage_v1_receipt_required_before_promotion",
            ),
        ),
    ),
}

COVERAGE_RECORDS: Mapping[str, CoverageRecord] = MappingProxyType(
    _COVERAGE_RECORD_DATA
)

_PRODUCER_CATALOG_DATA: dict[str, ModelDescriptor] = {
    key: _compose_descriptor(model_definition_for_catalog_key(key), coverage)
    for key, coverage in COVERAGE_RECORDS.items()
}


def coverage_record_for_key(key: str) -> CoverageRecord:
    try:
        return COVERAGE_RECORDS[key]
    except KeyError:
        raise ValueError(f"unknown_candidate_key: {key}") from None


def _validate_catalog_structure(catalog: Mapping[str, ModelDescriptor]) -> None:
    model_ids: set[str] = set()
    for key, descriptor in catalog.items():
        if key != descriptor.key:
            raise ValueError(f"descriptor_key_mismatch: {key}")
        if descriptor.model_id in model_ids:
            raise ValueError(f"duplicate_model_id: {descriptor.model_id}")
        model_ids.add(descriptor.model_id)


_validate_catalog_structure(_PRODUCER_CATALOG_DATA)


@cache
def runtime_descriptor_for_key(key: str) -> ModelDescriptor:
    """Return the Product package-identity projection without reading evidence.

    Historical Product packages bind ``ModelDescriptor.descriptor_digest``, so
    the runtime must retain that projection until the package format migrates.
    Loading a model must not, however, require parity cases, oracle locks, or an
    admission receipt.  Those artifacts certify a release; they are not inputs
    to model resolution, conversion, or execution.
    """

    try:
        descriptor = _PRODUCER_CATALOG_DATA[key]
    except KeyError:
        raise ValueError(f"unknown_candidate_key: {key}") from None
    definition = model_definition_for_catalog_key(key)
    model_definition_for_descriptor(descriptor)
    for name in (
        "model_id",
        "revision",
        "source_license",
        "custom_tokenizer_module",
        "mask_token_id",
        "tie_word_embeddings",
        "graph_ir_family",
        "decode_contract",
    ):
        if getattr(descriptor, name) != getattr(definition, name):
            raise ValueError(f"runtime_descriptor_{name}_mismatch: {key}")
    expected_targets = {
        (target.backend, target.precision): target
        for target in definition.deployments
    }
    actual_targets = {
        (target.backend, target.precision): target
        for target in descriptor.admitted_backends
    }
    if set(actual_targets) != set(expected_targets):
        raise ValueError(f"runtime_descriptor_deployment_set_mismatch: {key}")
    for identity, target in actual_targets.items():
        expected = expected_targets[identity]
        if (
            target.minimum_device != expected.minimum_device
            or target.deployment != expected.deployment
        ):
            raise ValueError(
                f"runtime_descriptor_deployment_mismatch: {key}:{identity!r}"
            )
    return descriptor


@cache
def _validated_descriptor_for_key(key: str) -> ModelDescriptor:
    descriptor = runtime_descriptor_for_key(key)
    descriptor.validate()
    load_case_corpus(descriptor)
    load_oracle_lock(descriptor)
    return descriptor


class _ProducerCatalog(Mapping[str, ModelDescriptor]):
    """Read-only catalog that validates only the descriptor being consumed."""

    def __getitem__(self, key: str) -> ModelDescriptor:
        return _validated_descriptor_for_key(key)

    def __iter__(self) -> Iterator[str]:
        return iter(_PRODUCER_CATALOG_DATA)

    def __len__(self) -> int:
        return len(_PRODUCER_CATALOG_DATA)


PRODUCER_CATALOG: Mapping[str, ModelDescriptor] = _ProducerCatalog()

# Compatibility alias: callers historically imported ``MODEL_CATALOG``. Both
# names point at the same producer-only mapping; admission metadata is separate.
MODEL_CATALOG = PRODUCER_CATALOG
ProducerDescriptor = ModelDescriptor


def descriptor_for_key(key: str) -> ModelDescriptor:
    return _validated_descriptor_for_key(key)


def descriptor_for_identity(model_id: str, revision: str) -> ModelDescriptor:
    for descriptor in _PRODUCER_CATALOG_DATA.values():
        if descriptor.model_id == model_id:
            if descriptor.revision != revision:
                raise ValueError(
                    "unknown_revision: "
                    f"{model_id}: expected={descriptor.revision} actual={revision}"
                )
            return _validated_descriptor_for_key(descriptor.key)
    raise ValueError(f"unknown_model_id: {model_id}")


__all__ = [
    "COVERAGE_RECORDS",
    "DREAM_CODER_MODEL_ID",
    "DREAM_MODEL_ID",
    "LLADA_MODEL_ID",
    "MODEL_CATALOG",
    "PRODUCER_CATALOG",
    "QWEN3_BD3LM_MODEL_ID",
    "QWEN3_MDLM_MODEL_ID",
    "BackendAdmission",
    "BackendEvidence",
    "CapabilityCase",
    "CoverageRecord",
    "DecodeContract",
    "DeploymentContract",
    "LogitsConvention",
    "ModelDescriptor",
    "NumericTier",
    "ProducerDescriptor",
    "candidate_runtime_policy",
    "coverage_record_for_key",
    "descriptor_for_identity",
    "descriptor_for_key",
    "load_case_corpus",
    "load_oracle_lock",
    "runtime_descriptor_for_key",
    "validate_execution_package",
    "validate_snapshot",
]
