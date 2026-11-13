"""Deterministic source-tree tooling for constructing admission-v2 receipts.

This module is intentionally not imported by the installed product surface and
is not part of a producer pathset.  It never writes a receipt or admission
pointer.  Callers commit machine JSON and its rendered report first, then pass
the exact evidence commit and producer commit IDs here and write the returned
canonical bytes in a separate control-plane commit.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from . import admission as _verifier
from . import candidate_evidence
from .producer_pathsets import PRODUCER_PATHSETS, ProducerPathset


def canonical_receipt_bytes(receipt: Mapping[str, Any]) -> bytes:
    """Return the unique on-disk byte representation emitted by this tool."""
    try:
        return (
            json.dumps(
                receipt,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
            + b"\n"
        )
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise ValueError("admission_finalize_receipt_not_canonicalizable") from None


def _repository_at_commit(
    repository_root: str | Path, evidence_commit: str
) -> tuple[Path, str]:
    repository = _verifier._resolve_repository_root(repository_root)
    top_level = _verifier._repository_git(
        repository, "rev-parse", "--show-toplevel"
    )
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
    if (
        type(evidence_commit) is not str
        or _verifier._GIT_OID.fullmatch(evidence_commit) is None
    ):
        raise ValueError("admission_finalize_evidence_commit_invalid")
    _verifier._reject_repository_grafts(repository)
    if not _verifier._is_raw_commit_ancestor(
        repository, evidence_commit, evidence_commit
    ):
        raise ValueError("admission_finalize_evidence_commit_invalid")
    return repository, evidence_commit


def _pathset_for(descriptor: Any, run_kind: str) -> ProducerPathset:
    profiles = _verifier._canonical_profiles(
        _verifier._expected_profiles(descriptor)
    )
    matches = [
        pathset
        for pathset in PRODUCER_PATHSETS.values()
        if pathset.candidate_key == descriptor.key
        and pathset.run_kind == run_kind
        and pathset.profiles == profiles
    ]
    if len(matches) != 1:
        raise ValueError("admission_finalize_pathset_not_unique")
    return matches[0]


def _machine_document(run_kind: str, raw: bytes) -> dict[str, Any]:
    if run_kind == "candidate-four-gate":
        machine = candidate_evidence.strict_json_loads(raw)
        if raw != candidate_evidence.canonical_json_bytes(machine):
            raise ValueError("admission_machine_result_not_canonical")
        return machine
    if run_kind == "postpromotion-public-product-path":
        machine = _verifier._parse_json(raw)
        if raw != _verifier._canonical_machine_result_bytes(machine):
            raise ValueError("admission_machine_result_not_canonical")
        return machine
    raise ValueError("admission_finalize_run_kind_invalid")


def _run_from_evidence(
    *,
    repository: Path,
    evidence_commit: str,
    descriptor: Any,
    run_kind: str,
    producer_commit: str,
    machine_path: str,
    report_path: str,
    provisional_bootstrap_candidate: str | None,
) -> dict[str, Any]:
    if (
        type(producer_commit) is not str
        or _verifier._GIT_OID.fullmatch(producer_commit) is None
    ):
        raise ValueError("admission_evidence_producer_commit_invalid")
    machine_raw = _verifier._repository_result_blob_bytes(
        repository,
        evidence_commit,
        machine_path,
        path_validator=_verifier._safe_machine_result_path,
        max_bytes=_verifier._MAX_MACHINE_RESULT_BYTES,
        too_large_error="admission_machine_result_too_large",
    )
    report_raw = _verifier._repository_result_blob_bytes(
        repository,
        evidence_commit,
        report_path,
        path_validator=_verifier._safe_result_path,
        max_bytes=_verifier._MAX_RENDERED_RESULT_BYTES,
        too_large_error="admission_result_record_too_large",
    )
    machine = _machine_document(run_kind, machine_raw)
    wheel_digest = machine.get("wheel_sha256")
    if (
        machine.get("commit") != producer_commit
        or type(wheel_digest) is not str
        or _verifier._SHA256.fullmatch(wheel_digest) is None
    ):
        raise ValueError("admission_finalize_machine_binding_invalid")

    pathset = _pathset_for(descriptor, run_kind)
    boundary_digest = _verifier._repository_pathset_digest(
        repository, producer_commit, pathset
    )
    evidence_digest = _verifier._repository_pathset_digest(
        repository, evidence_commit, pathset
    )
    if evidence_digest != boundary_digest:
        raise ValueError("admission_producer_head_digest_mismatch")
    return {
        "kind": run_kind,
        "producer_commit": producer_commit,
        "producer_boundary": {
            "algorithm": _verifier._V2_BOUNDARY_ALGORITHM,
            "pathset_id": pathset.pathset_id,
            "entry_count": len(pathset.paths),
            "sha256": boundary_digest,
        },
        "result_bundle": {
            "machine": {
                "path": machine_path,
                "sha256": hashlib.sha256(machine_raw).hexdigest(),
            },
            "report": {
                "path": report_path,
                "sha256": hashlib.sha256(report_raw).hexdigest(),
            },
        },
        "executed_wheel_sha256": wheel_digest,
        "provisional_bootstrap_candidate": provisional_bootstrap_candidate,
    }


def _receipt(
    descriptor: Any, *, stage: str, runs: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "evidence_kind": "model-coverage-v2",
        "candidate_key": descriptor.key,
        "model_id": descriptor.model_id,
        "revision": descriptor.revision,
        "descriptor_digest": descriptor.descriptor_digest,
        "profiles": _verifier._expected_profiles(descriptor),
        "stage": stage,
        "verdict": "pass",
        "runs": runs,
    }


def _validate_previous_final(receipt: dict[str, Any], descriptor: Any) -> None:
    """Validate a prior final receipt after its pathset version is retired.

    Revalidation exists specifically because the current producer boundary has
    changed.  Requiring the prior boundary identifier to remain registered
    would make that workflow impossible.  Preserve every other v2 invariant by
    validating a copy against today's exact pathsets; only the two historical
    pathset identifiers and their entry counts are projected forward.

    The fresh Candidate and default W6 are still checked against the current
    pathsets below.  This receipt establishes only that the exact descriptor
    previously held a canonical final record, not that old source satisfies a
    new producer boundary.
    """
    projected = copy.deepcopy(receipt)
    runs = projected.get("runs") if type(projected) is dict else None
    if type(runs) is not list or len(runs) != 2:
        raise ValueError("admission_finalize_previous_final_invalid")
    for run in runs:
        if type(run) is not dict or type(run.get("kind")) is not str:
            raise ValueError("admission_finalize_previous_final_invalid")
        boundary = run.get("producer_boundary")
        if type(boundary) is not dict:
            raise ValueError("admission_finalize_previous_final_invalid")
        historical_id = boundary.get("pathset_id")
        historical_count = boundary.get("entry_count")
        if (
            type(historical_id) is not str
            or _verifier._RECORD_KIND.fullmatch(historical_id) is None
            or type(historical_count) is not int
            or historical_count <= 0
        ):
            raise ValueError("admission_finalize_previous_final_invalid")
        current = _pathset_for(descriptor, run["kind"])
        boundary["pathset_id"] = current.pathset_id
        boundary["entry_count"] = len(current.paths)
    _verifier._validate_v2(projected, descriptor)
    if receipt["stage"] != "final":
        raise ValueError("admission_finalize_previous_final_required")


def build_provisional_receipt(
    *,
    repository_root: str | Path,
    descriptor: Any,
    evidence_commit: str,
    producer_commit: str,
    machine_path: str,
    report_path: str,
) -> bytes:
    """Build candidate-only canonical receipt bytes from committed evidence.

    ``evidence_commit`` must contain the machine/report blobs, and
    ``producer_commit`` must be its raw Git ancestor.  The producer pathset must
    have the same canonical blob digest at both commits.
    """
    repository, evidence_commit = _repository_at_commit(
        repository_root, evidence_commit
    )
    run = _run_from_evidence(
        repository=repository,
        evidence_commit=evidence_commit,
        descriptor=descriptor,
        run_kind="candidate-four-gate",
        producer_commit=producer_commit,
        machine_path=machine_path,
        report_path=report_path,
        provisional_bootstrap_candidate=None,
    )
    receipt = _receipt(descriptor, stage="provisional", runs=[run])
    _verifier._validate_repository_v2_receipt(
        descriptor,
        repository,
        receipt,
        evidence_commit,
        require_final=False,
    )
    return canonical_receipt_bytes(receipt)


def build_transition_receipt(
    *,
    repository_root: str | Path,
    descriptor: Any,
    evidence_commit: str,
    provisional_receipt: bytes,
    producer_commit: str,
    machine_path: str,
    report_path: str,
) -> bytes:
    """Bind the bootstrap W6 and expose an explicitly unreleaseable transition.

    The exact candidate run is preserved. ``evidence_commit`` must contain both
    the candidate and bootstrap-W6 bundles. Repository/release validation will
    reject the returned transition receipt even though the installed registry
    may use it to run the subsequent default-path W6 without an environment
    switch.
    """
    repository, evidence_commit = _repository_at_commit(
        repository_root, evidence_commit
    )
    provisional = _verifier._parse_json(provisional_receipt)
    if provisional_receipt != canonical_receipt_bytes(provisional):
        raise ValueError("admission_finalize_provisional_not_canonical")
    _verifier._validate_v2(provisional, descriptor)
    if provisional["stage"] != "provisional":
        raise ValueError("admission_finalize_provisional_required")

    public_run = _run_from_evidence(
        repository=repository,
        evidence_commit=evidence_commit,
        descriptor=descriptor,
        run_kind="postpromotion-public-product-path",
        producer_commit=producer_commit,
        machine_path=machine_path,
        report_path=report_path,
        provisional_bootstrap_candidate=descriptor.key,
    )
    receipt = _receipt(
        descriptor,
        stage="transition",
        runs=[copy.deepcopy(provisional["runs"][0]), public_run],
    )
    _verifier._validate_repository_v2_receipt(
        descriptor,
        repository,
        receipt,
        evidence_commit,
        require_final=False,
    )
    return canonical_receipt_bytes(receipt)


def build_final_receipt(
    *,
    repository_root: str | Path,
    descriptor: Any,
    evidence_commit: str,
    transition_receipt: bytes,
    producer_commit: str,
    machine_path: str,
    report_path: str,
) -> bytes:
    """Replace bootstrap W6 with the default-registry W6 in a final receipt.

    ``transition_receipt`` must be canonical and valid at ``evidence_commit``.
    The final receipt preserves only its candidate run and binds a new public
    machine/report whose bootstrap field is null. The public producer commit is
    the transition control commit (or its producer-equivalent descendant).
    """
    repository, evidence_commit = _repository_at_commit(
        repository_root, evidence_commit
    )
    transition = _verifier._parse_json(transition_receipt)
    if transition_receipt != canonical_receipt_bytes(transition):
        raise ValueError("admission_finalize_transition_not_canonical")
    _verifier._validate_v2(transition, descriptor)
    if transition["stage"] != "transition":
        raise ValueError("admission_finalize_transition_required")
    _verifier._validate_repository_v2_receipt(
        descriptor,
        repository,
        transition,
        evidence_commit,
        require_final=False,
    )

    public_run = _run_from_evidence(
        repository=repository,
        evidence_commit=evidence_commit,
        descriptor=descriptor,
        run_kind="postpromotion-public-product-path",
        producer_commit=producer_commit,
        machine_path=machine_path,
        report_path=report_path,
        provisional_bootstrap_candidate=None,
    )
    receipt = _receipt(
        descriptor,
        stage="final",
        runs=[copy.deepcopy(transition["runs"][0]), public_run],
    )
    _verifier._validate_repository_v2_receipt(
        descriptor,
        repository,
        receipt,
        evidence_commit,
        require_final=True,
    )
    return canonical_receipt_bytes(receipt)


def build_revalidated_final_receipt(
    *,
    repository_root: str | Path,
    descriptor: Any,
    evidence_commit: str,
    previous_final_receipt: bytes,
    provisional_receipt: bytes,
    producer_commit: str,
    machine_path: str,
    report_path: str,
) -> bytes:
    """Bind a fresh Candidate and default W6 for an existing Product model.

    An already-supported model remains reachable through its previous final
    receipt while shared producer code is revalidated.  This path therefore
    needs no provisional bootstrap run: the new Candidate plus an ordinary,
    bootstrap-free W6 are the complete replacement evidence.
    """
    repository, evidence_commit = _repository_at_commit(
        repository_root, evidence_commit
    )
    previous = _verifier._parse_json(previous_final_receipt)
    if previous_final_receipt != canonical_receipt_bytes(previous):
        raise ValueError("admission_finalize_previous_not_canonical")
    _validate_previous_final(previous, descriptor)

    provisional = _verifier._parse_json(provisional_receipt)
    if provisional_receipt != canonical_receipt_bytes(provisional):
        raise ValueError("admission_finalize_provisional_not_canonical")
    _verifier._validate_v2(provisional, descriptor)
    if provisional["stage"] != "provisional":
        raise ValueError("admission_finalize_provisional_required")
    _verifier._validate_repository_v2_receipt(
        descriptor,
        repository,
        provisional,
        evidence_commit,
        require_final=False,
    )

    public_run = _run_from_evidence(
        repository=repository,
        evidence_commit=evidence_commit,
        descriptor=descriptor,
        run_kind="postpromotion-public-product-path",
        producer_commit=producer_commit,
        machine_path=machine_path,
        report_path=report_path,
        provisional_bootstrap_candidate=None,
    )
    receipt = _receipt(
        descriptor,
        stage="final",
        runs=[copy.deepcopy(provisional["runs"][0]), public_run],
    )
    _verifier._validate_repository_v2_receipt(
        descriptor,
        repository,
        receipt,
        evidence_commit,
        require_final=True,
    )
    return canonical_receipt_bytes(receipt)


__all__ = [
    "build_final_receipt",
    "build_provisional_receipt",
    "build_revalidated_final_receipt",
    "build_transition_receipt",
    "canonical_receipt_bytes",
]
