"""Model-spec resolution and cache listing for the local client.

Deliberately thin. Every fact here comes from somewhere that already owns it:

* which model IDs exist, and their pinned revision -> ``compiler.REGISTRY``
* where a compiled package lives, and whether it is there -> ``plan_pretrained``
* whether a package is intact -> ``compiler.verify_compile_receipt``

Nothing in this module recomputes a cache path, a digest, or a supported-model
list. A second copy of the cache-path formula would silently diverge from the
compiler's the first time either changed, and `list` would then report on
directories the compiler never writes.

Importing this module pulls in NO heavy dependency: `diffusorrt list` and
`--help` must work on a machine with neither MLX nor Torch installed, which is
most machines before the first `pull`.
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, replace
import shutil
from pathlib import Path
from typing import Callable

from .application_state import (
    ApplicationState,
    ConfigurationError,
    RunProfile,
    resolve_run_config,
)
from .conversation import (
    ConversationInfo,
    _ConversationStore,
    _validate_session_id,
)
from .compiler import (
    CompilerDependencyError,
    compile_receipt_digest,
    plan_pretrained,
    verify_compile_receipt_snapshot,
    verify_package_receipt_snapshot,
)
from .compiler.planner import _resolve_cache_root
from .compiler.build import COMPILER_VERSION
from .backend_ids import (
    validate_backend,
    validate_deployment_profile,
    validate_device_options,
)
from .model_catalog import ModelCatalog
from .model_definitions import (
    model_definition_digest,
    model_definition_for_descriptor,
)
from .package_lifecycle import (
    PackageBusyError,
    acquire_package_exclusive_guard,
    acquire_package_load_lease,
)
from .trust_identity import UNSPECIFIED
from .backends.kv_capacity import validate_request_bounds


def _product_registry():
    """Load Product admission only when resolving a Product model id."""

    from .compiler.registry import REGISTRY

    return REGISTRY

# `list` states only what it checked.
#
# MISSING and CACHED are answerable from the filesystem alone. VALID and
# CORRUPT require reading the receipt and re-hashing every artifact, so they
# appear ONLY under --verify. Reporting "valid" for a directory that merely
# exists would be the more useful-sounding answer and the wrong one: presence
# is not integrity, and a half-written package is present.
CACHE_MISSING = "missing"
CACHE_CACHED = "cached"
CACHE_VALID = "valid"
CACHE_CORRUPT = "corrupt"
CACHE_NEEDS_REBUILD = "needs_rebuild"


class ModelSpecError(ValueError):
    """A model spec that is well-formed but not usable for this command."""


class ModelRemovalError(RuntimeError):
    """A package deletion is outside repository ownership or unsafe."""


class ModelPackageBusyError(ModelRemovalError):
    """The package has a live loader or another repository operation."""


class ModelVerificationError(RuntimeError):
    """A repository-owned package could not be verified."""


# The three things a MODEL argument can successfully be. There is deliberately
# no fourth "invalid" kind: a directory that is neither of the first two is a
# refusal, not a successfully-resolved unusable model.
KIND_PACKAGE = "package"
KIND_HF_SNAPSHOT = "hf_snapshot"
KIND_REGISTRY = "registry"
# Not a resolution outcome: what `doctor` with no MODEL reports on. Kept in the
# same schema so "what can this machine run" and "can this model run" are one
# document with one exit-code rule, rather than the two doctors this replaced.
KIND_MACHINE = "machine"

# A backend reason saying a CUDA budget was given but no package plan judged it.
BUDGET_NOT_CHECKED = "cuda_execution_budget_not_checked"


def classify_local_dir(path: Path) -> str | None:
    """"package", "hf_snapshot", or None for a directory that is neither.

    Marker-based and deliberately shallow. ``model.json`` is what makes a
    directory an execution package -- the production loader and the compile
    receipt decide whether it is a GOOD one, and duplicating that here would be
    a second integrity opinion to keep in sync.

    A raw Hugging Face snapshot is ``config.json`` plus weights; it is NOT
    runnable and must never be mistaken for a package. Architecture is not
    consulted here because classification is not admission. The converter later
    requires an exact registered definition and fails closed on identity or
    layout disagreement.
    """
    if (path / "model.json").is_file():
        return KIND_PACKAGE
    # is_file(), not glob truthiness: glob yields DIRECTORIES too, so a
    # directory named `weights.safetensors` next to a config.json classified as
    # a snapshot. is_file() also follows symlinks, so a dangling one -- common
    # in a partially-fetched HF cache -- no longer counts as weights.
    has_weights = (path / "model.safetensors.index.json").is_file() or any(
        candidate.is_file() for candidate in path.glob("*.safetensors")
    )
    if (path / "config.json").is_file() and has_weights:
        return KIND_HF_SNAPSHOT
    return None


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    """What a MODEL argument turned out to be.

    ``kind`` is one of the three KIND_* constants. ``model_id`` / ``revision``
    are None for both directory kinds: a directory handed to us is not claimed
    to be any registry entry, and pretending otherwise would let a later step
    verify a receipt against a model the package never came from.

    ``local_path`` is neutral on purpose. It was ``package_dir``, which was a
    lie the moment a raw snapshot could be resolved -- a snapshot is not a
    package directory, and a field name that says it is invites a caller to
    hand it to load_runtime.
    """

    spec: str
    kind: str
    local_path: Path
    """Always present. Every successful branch supplies one -- the resolved
    directory for the two local kinds, and where the package WOULD live for a
    registry hit. It was typed Optional for a state that cannot occur, which
    only pushes a pointless None check onto every caller."""

    model_id: str | None = None
    revision: str | None = None


@dataclass(frozen=True, slots=True)
class CachedModel:
    """One registry entry's cache state.

    Carries the six fields plan.md B1 requires in JSON mode: model ID, pinned
    revision, source license, package bytes, cache state, and location. The
    licence in particular is not decoration -- it is what a user needs before
    deciding to acquire 30 GiB of someone else's weights. The remaining fields
    are the richer in-process model-manager view and deliberately do not change
    the established ``list --json`` schema.
    """

    model_id: str
    revision: str
    source_license: str
    package_bytes: int
    """ESTIMATED package size: the registry's fp32 payload figure, carried
    through CompilePlan.estimated_package_bytes.

    NOT a measurement of the cache directory, even when status is `cached`.
    Naming it plainly here because the JSON key is a contract other tools will
    read, and a field that silently switches between estimate and measurement
    depending on cache state is worse than one that is always an estimate."""

    package_dir: Path
    status: str
    source_bytes: int = 0
    package_available_bytes: int = 0
    source_available_bytes: int = 0
    usable_backends: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "source_license": self.source_license,
            "package_bytes": self.package_bytes,
            "package_dir": str(self.package_dir),
            "status": self.status,
        }


# Snapshot states. There is no "valid": Hugging Face does not verify that a
# cached snapshot is complete, so presence is the only thing that can honestly
# be reported. UNKNOWN is not a hedge -- it is the answer when huggingface_hub
# is absent and the question cannot be asked at all; calling that "missing"
# would send a user downloading bytes they may already have.
# The seven stages a preparation passes through. Named for what actually
# happens: offline never emits ACQUISITION as a download, and a cache hit skips
# straight from CACHE_VERIFICATION to READY.
STAGE_PLANNING = "planning"
STAGE_CACHE_VERIFICATION = "cache_verification"
STAGE_ACQUISITION = "acquisition"
STAGE_CONVERSION = "conversion"
STAGE_VALIDATION = "validation"
STAGE_PUBLICATION = "publication"
STAGE_READY = "ready"
PREPARE_STAGES = (
    STAGE_PLANNING,
    STAGE_CACHE_VERIFICATION,
    STAGE_ACQUISITION,
    STAGE_CONVERSION,
    STAGE_VALIDATION,
    STAGE_PUBLICATION,
    STAGE_READY,
)

SNAPSHOT_PRESENT = "present"
SNAPSHOT_MISSING = "missing"
SNAPSHOT_UNKNOWN = "unknown"


# 2.1: online package reports name the deployment profile they validated and
# recommend that same deployment path instead of a raw run with other semantics.
# 2.0: one report for all three MODEL kinds, with backend readiness folded in.
# Bumped, not extended in place -- `kind`, `backend_status` and the nullable
# model_id/revision/source_license are not additive for a consumer that assumed
# a registry entry, and `ready` now answers a strictly stronger question.
# 2.2: optional request-bounded K/V admission budgets, never a device-fit claim.
SCHEMA_VERSION = "2.2"


@dataclass(frozen=True, slots=True)
class OfflineModelReport:
    """Can this registered model be run right now, and if not, what is missing?

    Establishes that WITHOUT downloading, converting, or executing model code.
    It does verify the package receipt when one is present, because that is the
    only honest way to answer the question the user is actually asking -- and
    re-hashing is cheap next to the 14.19 GiB download the answer avoids.

    There is no ``runnable`` field: even a valid receipt does not prove the
    production loader will succeed on this machine. ``package_cache_status ==
    "valid"`` says the bytes are the ones this compiler published, nothing more.
    """

    schema_version: str
    kind: str
    """Which of the three MODEL kinds this is. A local package has no registry
    ID, revision, licence or download budget, and reporting invented ones was
    why a second doctor existed at all."""

    model_id: str | None
    revision: str | None
    supported: bool
    source_license: str | None
    snapshot_cache_status: str
    package_cache_status: str
    package_dir: Path
    custom_tokenizer_module: str | None
    trust_required: bool
    estimated_package_bytes: int
    required_free_bytes: int
    available_free_bytes: int

    snapshot_cache_root: str | None
    """Where a download would land. None when this report is offline -- there
    is no second budget because nothing is downloaded."""

    snapshot_required_bytes: int
    snapshot_available_bytes: int
    """The SECOND budget. The report used to carry only the package one, so a
    machine short on the Hugging Face cache disk was told to go ahead."""

    fits: bool
    """Whether the disk budget for PREPARING this model fits.

    Irrelevant once the package is valid -- nothing further is written. Kept as
    the raw fact rather than zeroed, but `ready` deliberately does not depend on
    it: reporting ready=yes beside fits=false looked self-contradictory, and the
    resolution is that they answer different questions, which the human output
    now says.
    """

    backend_status: str
    """The requested backend's availability: "usable", "unavailable", or
    "not_checked" when the package is not there to check against."""

    selected_backend: str | None
    """The backend that WILL run this, when the user named one. None under
    `auto`, which resolves against the package's own backend list at load."""

    deployment_profile: str | None
    """The named profile applied before package validation, or None when no
    package was opened under deployment semantics."""

    usable_backends: tuple[str, ...]
    """Every backend this machine can run, by import and platform. Not a claim
    that any of them accepts this particular package."""

    backend_reasons: tuple[str, ...]

    @property
    def package_fits(self) -> bool:
        return self.available_free_bytes >= self.required_free_bytes

    @property
    def snapshot_fits(self) -> bool:
        return self.snapshot_available_bytes >= self.snapshot_required_bytes

    ready: bool
    """Can the requested run succeed, as asked? Package intact, a backend that
    can run it, and any required remote-code authorisation actually granted.

    Each was added because the weaker version lied. Receipt-only reported
    Ready: yes on a machine with no MLX. Backend-without-trust reported
    Ready: yes for a package whose next step demanded --trust-remote-code, on
    the same screen."""

    package_compatibility: str
    """Whether THIS package can actually be opened and run: "compatible",
    "incompatible", or "not_checked".

    Separate from `backend_status`, which is a fact about the MACHINE. Using
    the machine answer as a proxy for this one is what made `--offline` report
    `ready: true` for a receipt-valid package that fails to load -- measured:
    offline ready=True, online ready=False, same directory."""

    trust_satisfied: bool
    """Whether this report was given the authorisation the package needs. False
    with `trust_required: true` means: ready APART from consent."""

    next_action: str
    kv_capacity: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "backend_status": self.backend_status,
            "selected_backend": self.selected_backend,
            "deployment_profile": self.deployment_profile,
            "kv_capacity": self.kv_capacity,
            "usable_backends": list(self.usable_backends),
            "backend_reasons": list(self.backend_reasons),
            "package_compatibility": self.package_compatibility,
            "trust_satisfied": self.trust_satisfied,
            "model_id": self.model_id,
            "revision": self.revision,
            "supported": self.supported,
            "source_license": self.source_license,
            "snapshot_cache_status": self.snapshot_cache_status,
            "package_cache_status": self.package_cache_status,
            "package_dir": str(self.package_dir),
            "custom_tokenizer_module": self.custom_tokenizer_module,
            "trust_required": self.trust_required,
            "estimated_package_bytes": self.estimated_package_bytes,
            "required_free_bytes": self.required_free_bytes,
            "available_free_bytes": self.available_free_bytes,
            "snapshot_cache_root": self.snapshot_cache_root,
            "snapshot_required_bytes": self.snapshot_required_bytes,
            "snapshot_available_bytes": self.snapshot_available_bytes,
            "fits": self.fits,
            "ready": self.ready,
            "next_action": self.next_action,
        }


def _snapshot_cache_status(model_id: str, revision: str) -> str:
    """present / missing / unknown. Never downloads, never raises.

    Uses the same public lookup the compiler uses, so the pre-flight and the
    compile cannot disagree about whether the snapshot is there.
    """
    from .compiler import SnapshotNotCachedError, cached_snapshot_path

    try:
        cached_snapshot_path(model_id, revision=revision)
    except SnapshotNotCachedError:
        return SNAPSHOT_MISSING
    except CompilerDependencyError:
        # huggingface_hub absent: the question cannot be asked. Reporting
        # "missing" would be a guess that costs a 14.19 GiB download.
        return SNAPSHOT_UNKNOWN
    return SNAPSHOT_PRESENT


def model_report(
    spec: str,
    *,
    cache_root: str | Path | None = None,
    requested_backend: str = "auto",
    check_backend: bool = True,
    trust_remote_code: bool = False,
    open_package: bool = True,
    offline: bool = False,
    prompt_tokens: int | None = None,
    max_tokens: int | None = None,
    cuda_precision: str | None = None,
    cuda_device_memory_budget_bytes: int = 0,
    catalog: ModelCatalog | None = None,
) -> OfflineModelReport:
    """Can this MODEL be run right now, and if not, what is missing?

    ONE report, for all three MODEL kinds. There were two: a `--offline` one
    that took a registry ID and answered about the cache, and a default one
    that took a PACKAGE PATH and answered about backends. `doctor <model-id>`
    without --offline therefore treated the ID as a directory and reported on a
    path that never existed.

    Downloads nothing and performs no inference. With ``open_package=True``, a
    prepared package is loaded to establish compatibility; custom tokenizer
    code can therefore execute only when ``trust_remote_code=True``. With
    ``open_package=False``, ``check_backend`` probes machine capabilities by
    import and never opens the package.
    """
    # `offline` is the MODE the report is about, not a convenience. Hardcoding
    # True gave every report the offline budget -- snapshot cost zero, snapshot
    # disk None -- so a machine that could not fit the download was told
    # `fits: true` and handed an online `pull` command. Measured: real online
    # plan fits=False, doctor fits=True.
    validate_request_bounds(prompt_tokens, max_tokens)
    plan = plan_client_model(
        spec,
        cache_root=cache_root,
        offline=offline,
        catalog=catalog,
    )

    if plan.kind == KIND_REGISTRY:
        snapshot_status = _snapshot_cache_status(plan.model_id, plan.revision)
    else:
        # There is no upstream snapshot for a directory the user handed us, and
        # "missing" would read as something to download.
        snapshot_status = SNAPSHOT_UNKNOWN

    package_status = _package_cache_status(plan, catalog=catalog)
    # Open the package only when the caller allows it, it is there, and
    # opening it needs no authorisation they have not given.
    #
    # `open_package=False` is what --offline means at this layer: no network,
    # and no model code. The machine probe still runs -- it is an import and a
    # platform check -- but the package is not opened, so backend
    # COMPATIBILITY with this package stays unanswered.
    probe_package = (
        plan.package_dir
        if open_package
        and plan.package_present
        and package_status != CACHE_NEEDS_REBUILD
        and (trust_remote_code or not plan.trust_required)
        else None
    )
    (
        backend_status,
        selected,
        backend_reasons,
        usable_backends,
        deployment_profile,
        kv_capacity,
    ) = _backend_readiness(
        requested_backend,
        enabled=check_backend,
        package_dir=probe_package,
        trust_remote_code=trust_remote_code,
        expected_module=(
            plan.trust_module if trust_remote_code else UNSPECIFIED
        ),
        prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
        cuda_precision=cuda_precision,
        cuda_device_memory_budget_bytes=cuda_device_memory_budget_bytes,
        model_catalog=catalog,
    )
    backend_reasons = _note_unchecked_budget(
        backend_reasons, cuda_device_memory_budget_bytes, plan_resolved=probe_package is not None
    )

    # CACHE_CACHED is accepted, and only a directory the caller handed us can
    # reach it: _package_cache_status returns valid/corrupt/missing for a
    # registry package, where a receipt exists to check (an intact stale
    # compiler package is needs_rebuild). A user-supplied
    # package has no receipt this compiler wrote, so `cached` is the strongest
    # honest verdict available -- and demanding `valid` reported Ready: no for
    # a package the installed wheel then ran successfully.
    package_ok = package_status in (CACHE_VALID, CACHE_CACHED)
    # STRICT equality, not "anything but unavailable". `not_checked` is the
    # answer when the question could not be asked, and counting it as success
    # is how a readiness claim comes to cover something nobody looked at.
    backend_ok = backend_status == "usable"
    # A THIRD, independent fact: does this package open on this machine? The
    # machine having MLX says nothing about whether these weights and this
    # architecture suit it. Only answered when the package was opened.
    if probe_package is None:
        compatibility = "not_checked"
    elif backend_status == "usable":
        compatibility = "compatible"
    else:
        compatibility = "incompatible"
    # D3: trust is part of readiness. A package that imports its own tokenizer
    # cannot be run by a caller who has not authorised that, so `ready` is
    # relative to the authorisation THIS report was given -- which is exactly
    # what `next_action` then prints. Reporting `ready: true` beside
    # `trust_required: true` while the next step demands --trust-remote-code
    # is a contradiction the report itself displayed.
    trust_ok = trust_remote_code or not plan.trust_required
    # `not_checked` is NOT success. --offline therefore cannot report ready:
    # it declines to open the package, so the question of whether this package
    # runs was never asked. That is the honest cost of the cheap pre-flight,
    # and next_action says how to get the full answer.
    ready = package_ok and backend_ok and trust_ok and compatibility == "compatible"
    capacity_invalid = kv_capacity is not None and kv_capacity["status"] == "invalid"
    if capacity_invalid:
        ready = False

    return OfflineModelReport(
        schema_version=SCHEMA_VERSION,
        kind=plan.kind,
        model_id=plan.model_id,
        revision=plan.revision,
        supported=True,
        source_license=plan.source_license,
        snapshot_cache_status=snapshot_status,
        package_cache_status=package_status,
        package_dir=plan.package_dir,
        custom_tokenizer_module=plan.trust_module,
        trust_required=plan.trust_required,
        estimated_package_bytes=plan.estimated_package_bytes,
        required_free_bytes=plan.required_free_bytes,
        available_free_bytes=plan.available_free_bytes,
        snapshot_cache_root=(
            str(plan.snapshot_cache_root) if plan.snapshot_cache_root else None
        ),
        snapshot_required_bytes=plan.snapshot_required_bytes,
        snapshot_available_bytes=plan.snapshot_available_bytes,
        fits=plan.fits,
        backend_status=backend_status,
        selected_backend=selected,
        deployment_profile=deployment_profile,
        usable_backends=usable_backends,
        backend_reasons=backend_reasons,
        kv_capacity=kv_capacity,
        package_compatibility=compatibility,
        trust_satisfied=trust_ok,
        ready=ready,
        next_action=(
            "reduce --prompt-tokens/--max-tokens or correct the K/V geometry: "
            f"{kv_capacity['reason']}"
        ) if capacity_invalid else _render_action(_next_action_argv(
            plan,
            package_status=package_status,
            snapshot_status=snapshot_status,
            backend_status=backend_status,
            backend_reasons=backend_reasons,
            compatibility=compatibility,
            selected_backend=selected,
            deployment_profile=deployment_profile,
            cuda_precision=cuda_precision,
        )),
    )


def machine_report(
    *,
    requested_backend: str = "auto",
    check_backend: bool = True,
    cuda_precision: str | None = None,
    cuda_device_memory_budget_bytes: int = 0,
) -> OfflineModelReport:
    """What this machine can run, with no model named.

    `doctor` with no argument used to answer this and nothing else. Keeping it
    in the same schema means one exit-code rule: zero when something here can
    run a model, nonzero when nothing can.
    """
    (
        backend_status,
        selected,
        backend_reasons,
        usable_backends,
        deployment_profile,
        kv_capacity,
    ) = _backend_readiness(
        requested_backend,
        enabled=check_backend,
        cuda_precision=cuda_precision,
    )
    backend_reasons = _note_unchecked_budget(
        backend_reasons, cuda_device_memory_budget_bytes, plan_resolved=False
    )
    usable = backend_status == "usable"
    return OfflineModelReport(
        schema_version=SCHEMA_VERSION,
        kind=KIND_MACHINE,
        model_id=None,
        revision=None,
        supported=True,
        source_license=None,
        snapshot_cache_status=SNAPSHOT_UNKNOWN,
        package_cache_status=CACHE_MISSING,
        package_dir=Path("."),
        custom_tokenizer_module=None,
        trust_required=False,
        estimated_package_bytes=0,
        required_free_bytes=0,
        available_free_bytes=0,
        snapshot_cache_root=None,
        snapshot_required_bytes=0,
        snapshot_available_bytes=0,
        fits=True,
        backend_status=backend_status,
        selected_backend=selected,
        deployment_profile=deployment_profile,
        usable_backends=usable_backends,
        backend_reasons=backend_reasons,
        kv_capacity=kv_capacity,
        # No package was named, so there is no package to be compatible with.
        # `ready` here answers "can this machine run anything", which is a
        # different and clearly-labelled question (kind == "machine").
        package_compatibility="not_checked",
        trust_satisfied=True,
        ready=usable,
        next_action=(
            "diffusorrt doctor MODEL"
            if usable
            else _unavailable_action(
                backend_reasons[0] if backend_reasons else "none available"
            )
        ),
    )


def _package_cache_status(
    plan: ClientModelPlan,
    *,
    catalog: ModelCatalog | None = None,
) -> str:
    """missing / cached / valid / needs_rebuild / corrupt.

    A directory with a model.json is NOT evidence it can run: an empty `{}`
    manifest satisfies presence and fails the receipt. Presence and integrity
    are different questions and doctor must answer the second.

    A local package the user handed us has no receipt to verify -- it was not
    published by this compiler -- so it stops at `cached`. Calling it `valid`
    would be a claim nothing checked.
    """
    if not plan.package_present:
        return CACHE_MISSING
    if plan.kind != KIND_REGISTRY:
        return CACHE_CACHED
    expected = None if catalog is None else catalog[plan.model_id]
    verified = verify_compile_receipt_snapshot(
        plan.package_dir,
        model_id=plan.model_id,
        revision=plan.revision,
        expected_definition_key=(None if expected is None else expected.key),
        expected_definition_digest=(
            None if expected is None else expected.definition_digest
        ),
        _require_current_compiler=False,
    )
    if verified is not None:
        descriptor_current = (
            expected is not None
            or verified.descriptor_digest == _product_registry()[plan.model_id].descriptor_digest
        )
        return (
            CACHE_VALID if verified.compiler_version == COMPILER_VERSION and descriptor_current
            else CACHE_NEEDS_REBUILD
        )
    return CACHE_CORRUPT


def _unavailable_action(reason: str) -> str:
    """What to do about the first reason nothing can run."""
    if reason == "dependency_unloadable:diffusorrt_runtime":
        return (
            f"{reason} -- the runtime is installed but a library it needs cannot be "
            "loaded; make the CUDA runtime libraries loadable before starting, as "
            "native/binding/README.md describes"
        )
    if reason.startswith("dependency_unloadable:"):
        return f"{reason} -- installed but cannot be imported; reinstall it"
    return f"install a usable backend -- {reason}"


def _note_unchecked_budget(
    reasons: tuple[str, ...], budget: int, *, plan_resolved: bool
) -> tuple[str, ...]:
    """Whether a budget suits a package is a property of its resolved plan.

    Without one, say the check was not made rather than let silence read as
    approval of the budget.
    """
    if budget == 0 or plan_resolved:
        return reasons
    return (*reasons, f"{BUDGET_NOT_CHECKED}: no package plan was resolved")


def _backend_readiness(
    requested_backend: str,
    *,
    enabled: bool,
    package_dir: Path | None = None,
    trust_remote_code: bool = False,
    expected_module: object = UNSPECIFIED,
    prompt_tokens: int | None = None,
    max_tokens: int | None = None,
    cuda_precision: str | None = None,
    cuda_device_memory_budget_bytes: int = 0,
    model_catalog: ModelCatalog | None = None,
) -> tuple[str, str | None, tuple[str, ...], tuple[str, ...], str | None, dict | None]:
    """Report backend readiness, optionally under deployment package semantics.

    With no package, ``diagnose_runtime(package=None)`` probes imports and the
    machine only. With a package, the shared deployment loader applies the
    named profile before canonical validation, so doctor answers the same
    readiness question as deploy. Auto-selection has no answer in machine-only
    mode because it reads the package's own backend declaration.

    The first version of this read `selected_backend is None` as "unavailable"
    and reported `install a usable backend -- not_implemented` on a machine
    where `run` demonstrably works, because auto never selects without a
    package and a planned provider is not runnable. Both are facts about the
    probe, not about the machine.
    """
    if not enabled:
        return "not_checked", None, (), (), None, None
    try:
        from .capabilities import diagnose_deployment_runtime, diagnose_runtime
    except ImportError:
        return "not_checked", None, ("capabilities_unavailable",), (), None, None

    # ValueError is deliberately NOT caught: `--backend nonsense` is a bad
    # argument, and turning it into "not_checked" made the report answer
    # `Ready: yes` for a backend that does not exist. Argument errors belong to
    # the caller; only an unexpected probe failure degrades to not_checked.
    # With the package, the probe also answers whether THIS package's
    # architecture and weights suit each backend -- 0.7s, and the difference
    # between "MLX is installed" and "MLX can run this". Without it, only the
    # machine is checked.
    #
    # package_dir is passed only when loading it is already authorised: the
    # load imports the package's own tokenizer module, and doctor asking no
    # question means doctor may not execute code on the strength of silence.
    try:
        if package_dir is None:
            report = diagnose_runtime(
                requested_backend=requested_backend,
                prompt_tokens=prompt_tokens, max_tokens=max_tokens,
                cuda_precision=cuda_precision,
            )
        else:
            from .deployment import DEFAULT_MLX_DEPLOYMENT_PROFILE

            catalog_options = (
                {} if model_catalog is None else {"model_catalog": model_catalog}
            )
            report = diagnose_deployment_runtime(
                package_dir,
                requested_backend=requested_backend,
                trust_remote_code=trust_remote_code,
                expected_module=expected_module,
                mlx_profile_name=DEFAULT_MLX_DEPLOYMENT_PROFILE,
                prompt_tokens=prompt_tokens,
                max_tokens=max_tokens,
                cuda_precision=cuda_precision,
                cuda_device_memory_budget_bytes=cuda_device_memory_budget_bytes,
                **catalog_options,
            )
    except ValueError:
        raise
    except Exception as error:
        return "not_checked", None, (f"probe_failed: {error}",), (), None, None

    usable = tuple(b.backend_id for b in report.backends if b.usable)
    capacity = report.kv_capacity

    if package_dir is not None and report.package_status != "valid":
        # The package failed to open. That is a package fact, not a backend
        # fact, but it is the one that decides whether a run can start.
        return (
            "unavailable",
            None,
            (report.package_error or "package_invalid",),
            usable,
            report.deployment_profile,
            capacity,
        )

    if requested_backend != "auto":
        for backend in report.backends:
            if backend.backend_id != requested_backend:
                continue
            if backend.usable:
                return (
                    "usable",
                    backend.backend_id,
                    (),
                    usable,
                    report.deployment_profile,
                    capacity,
                )
            # Only THIS backend's reasons. Aggregating every backend's reasons
            # meant a machine with one working provider inherit unrelated
            # refusal reasons from every other provider.
            return (
                "unavailable",
                None,
                backend.reasons,
                usable,
                report.deployment_profile,
                capacity,
            )
        return "not_checked", None, ("unknown_backend",), usable, None, capacity

    if usable:
        # auto resolves against the package's own backend list, so it only has
        # an answer when the package was opened.
        return (
            "usable",
            report.selected_backend,
            (),
            usable,
            report.deployment_profile,
            capacity,
        )
    reasons = tuple(
        dict.fromkeys(r for b in report.backends for r in b.reasons)
    )
    return "unavailable", None, reasons, (), report.deployment_profile, capacity


def _render_action(action: "list[str] | str") -> str:
    """argv -> a copy-pasteable command line; prose passes through.

    Quoted, not " ".join: a package path with a space is legal and produced a
    Next: line that could not be run or parsed. Quoted for the shell the user
    has: cmd.exe does not read a single quote as a quote, and POSIX rules wrap
    every Windows path in them because of its backslashes."""
    if isinstance(action, str):
        return action
    if os.name == "nt":
        return " ".join(_windows_word(argument) for argument in action)
    return shlex.join(action)


# What cmd.exe and PowerShell both read as one bare word.
_WINDOWS_BARE_WORD = re.compile(r"[\w\-./\\:@+]+")


def _windows_word(argument: str) -> str:
    """One argument, in double quotes where either Windows shell needs them."""
    word = subprocess.list2cmdline([argument])  # escapes quotes, quotes for spaces
    if word.startswith('"') or _WINDOWS_BARE_WORD.fullmatch(argument):
        return word
    # "&", "(", "^" and the like, which list2cmdline leaves bare. Closing the
    # quote after a backslash would escape it, so trailing ones are doubled.
    trailing = len(word) - len(word.rstrip("\\"))
    return '"' + word + "\\" * trailing + '"'


def _next_action_argv(
    plan: ClientModelPlan,
    *,
    package_status: str,
    snapshot_status: str,
    backend_status: str,
    backend_reasons: tuple[str, ...],
    compatibility: str = "not_checked",
    selected_backend: str | None = None,
    deployment_profile: str | None = None,
    cuda_precision: str | None = None,
) -> list[str] | str:
    """The one command to run next, as ARGV. Must be executable AS PRINTED.

    argv rather than a string, rendered with shlex.join at the edge: a package
    path containing a space produced a `Next:` line nobody could copy, and the
    W6 gate parsed that line with .split() -- so a legal path silently produced
    an unrunnable recommendation and an unparseable gate step. A plain-prose
    answer (a refusal, not a command) is returned as a str and rendered as-is.

    Every command this report prints carries the flags the SAME report says are
    needed. `run` on a trust-requiring package without the flag dies at
    tokenizer load 13 seconds in -- a failure the report predicted one line
    above and then walked the user into.

    The flag follows `plan.trust_required`, NOT whether trust is already
    granted. This took a `trust_ok` argument that the caller threaded in and
    the body never read: a model that needs the flag needs it in the printed
    command whether or not this process already has consent, so the parameter
    could not have changed an answer. Accepted and ignored is the failure mode
    this file has produced before; the fix is to stop accepting it.
    """
    trust_flag = ["--trust-remote-code"] if plan.trust_required else []
    cuda_flags = (
        ["--precision", cuda_precision]
        if selected_backend == "cuda_native" and cuda_precision is not None
        else []
    )
    target = plan.model_id or plan.spec

    if backend_status == "unavailable":
        # No point recommending a run that cannot start. Naming the reason
        # beats naming a command -- but the reason must be about the MACHINE.
        # A package that fails to open produced "install a usable backend --
        # string indices must be integers", which is an internal error leaking
        # into user-facing advice.
        if compatibility == "incompatible":
            detail = backend_reasons[0] if backend_reasons else "it failed to open"
            return f"{plan.package_dir} cannot be loaded: {detail}"
        reason = backend_reasons[0] if backend_reasons else "no usable backend"
        return _unavailable_action(reason)

    if package_status in (CACHE_VALID, CACHE_CACHED):
        if compatibility == "not_checked":
            # The package was not opened, so "it runs" is unestablished. Say
            # how to establish it rather than recommending a run that might
            # fail 13 seconds in.
            return ["diffusorrt", "doctor", target, *trust_flag]
        if deployment_profile is not None:
            if selected_backend is None:
                return "deployment backend resolution produced no backend"
            return [
                "diffusorrt",
                "deploy",
                str(plan.package_dir),
                "--target",
                selected_backend,
                *cuda_flags,
                *trust_flag,
            ]
        if selected_backend == "cuda_native":
            return [
                "diffusorrt",
                "run",
                target,
                "--backend",
                selected_backend,
                *cuda_flags,
                *trust_flag,
            ]
        return ["diffusorrt", "run", target, *trust_flag]

    if plan.kind != KIND_REGISTRY:
        return f"{plan.package_dir} is not a prepared execution package"

    # DISK FIRST, for every path that still has to prepare something.
    #
    # The order used to be snapshot-status first, so a cold cache with an
    # undersized Hugging Face disk was handed `diffusorrt pull` -- a command
    # this same report already knew could not succeed, since `fits` was False
    # two fields above. Recommending a command you know will fail is the same
    # defect as `doctor` printing a `run` without the trust flag it required.
    if not plan.fits:
        if not plan.package_fits:
            short = plan.required_free_bytes - plan.available_free_bytes
            return f"free {short / 2**30:.1f} GiB on {plan.cache_root}"
        short = plan.snapshot_required_bytes - plan.snapshot_available_bytes
        return (
            f"free {short / 2**30:.1f} GiB on {plan.snapshot_cache_root} "
            "(the Hugging Face cache), or set HF_HUB_CACHE"
        )

    if package_status in (CACHE_CORRUPT, CACHE_NEEDS_REBUILD):
        # The recommendation is executable as printed. The package-status line
        # already explains why preparation is needed.
        return ["diffusorrt", "pull", target, *trust_flag]

    if snapshot_status == SNAPSHOT_UNKNOWN:
        return "pip install 'diffusor-rt[compiler]'"
    if snapshot_status == SNAPSHOT_MISSING:
        return ["diffusorrt", "pull", target, *trust_flag]
    return ["diffusorrt", "pull", target, "--offline", *trust_flag]


def offline_model_report(
    model_id: str,
    *,
    cache_root: str | Path | None = None,
    catalog: ModelCatalog | None = None,
) -> OfflineModelReport:
    """The cache-only half of model_report, kept for callers that must not
    probe backends. Same schema, same code path."""
    return model_report(
        model_id,
        cache_root=cache_root,
        check_backend=False,
        offline=True,
        catalog=catalog,
    )



@dataclass(frozen=True, slots=True)
class ClientModelPlan:
    """Everything the client knows about a MODEL argument, resolved ONCE.

    Before this existed, a single `diffusorrt run` resolved the same string
    four times: once to decide the trust question, once to plan the compile,
    once inside prepare_model, and once more for the report. Each resolution
    read the filesystem independently, so a directory that appeared or changed
    between them produced a run whose consent, plan and package disagreed --
    the structural source of the consent TOCTOU, not an incidental one.

    Frozen because a plan that can be edited is a plan that can be edited
    AFTER the user consented to what it said.

    Nothing here executes model code, hashes a package, or touches the network.
    Those are questions for `model_report` and `prepare_planned`, which take
    this as input.
    """

    spec: str
    kind: str
    """One of the KIND_* constants."""

    package_dir: Path
    """Where the package IS for a local package, or WOULD BE for a registry
    ID. Not a claim that it exists -- see ``package_present``."""

    package_present: bool
    """A model.json is there. NOT that it is intact: presence is not
    integrity, and a half-written package is present."""

    trust_module: str | None
    """The module this model imports at load, or None. From the REGISTRY for a
    registry ID (a cached snapshot cannot be asked -- its files are symlinks
    into blobs/ and the artifact opener refuses symlinks), and from the
    package's own tokenizer_config for a directory."""

    model_id: str | None = None
    revision: str | None = None
    source_license: str | None = None
    estimated_package_bytes: int = 0
    required_free_bytes: int = 0
    available_free_bytes: int = 0
    cache_root: Path | None = None

    snapshot_cache_root: Path | None = None
    """Where a download would land -- the Hugging Face cache, which is a
    different directory and often a different disk."""

    snapshot_required_bytes: int = 0
    snapshot_available_bytes: int = 0
    """The SECOND budget, carried through from CompilePlan.

    Dropping it here was the whole bug: the compiler correctly refused a
    compile whose snapshot could not fit, and `doctor` -- which reads this
    plan, not the compile plan -- reported `fits: true` and told the user to
    go ahead. Measured on two filesystems with 1 byte free on the snapshot
    disk: compiler fits=False, client fits=True."""

    @property
    def trust_required(self) -> bool:
        return self.trust_module is not None

    @property
    def package_fits(self) -> bool:
        return self.available_free_bytes >= self.required_free_bytes

    @property
    def snapshot_fits(self) -> bool:
        return self.snapshot_available_bytes >= self.snapshot_required_bytes

    @property
    def fits(self) -> bool:
        """BOTH budgets, on their own filesystems."""
        return self.package_fits and self.snapshot_fits

    @property
    def shortfall_bytes(self) -> int:
        return max(0, self.required_free_bytes - self.available_free_bytes) + max(
            0, self.snapshot_required_bytes - self.snapshot_available_bytes
        )


def plan_client_model(
    spec: str,
    *,
    cache_root: str | Path | None = None,
    allow_local_dir: bool = True,
    offline: bool = True,
    catalog: ModelCatalog | None = None,
) -> ClientModelPlan:
    """Resolve a MODEL argument into the one plan every command shares."""
    resolved = resolve_model(
        spec,
        allow_local_dir=allow_local_dir,
        cache_root=cache_root,
        catalog=catalog,
    )

    if resolved.kind != KIND_REGISTRY:
        # Imported HERE, not at the top of the function: edllm.tokenizer pulls
        # in `tokenizers`, which is part of the compiler extra. A registry ID
        # takes its trust answer from the REGISTRY and never needs it, and
        # importing it unconditionally made `doctor <model-id>` on a bare
        # install die with ModuleNotFoundError instead of answering.
        from .tokenizer import package_declares_custom_tokenizer_code

        discovery_lease = acquire_package_load_lease(resolved.local_path)
        try:
            trust_module = package_declares_custom_tokenizer_code(
                discovery_lease.package_dir
            )
        finally:
            discovery_lease.close()

        return ClientModelPlan(
            spec=spec,
            kind=resolved.kind,
            package_dir=resolved.local_path,
            package_present=resolved.kind == KIND_PACKAGE,
            trust_module=trust_module,
        )

    plan = plan_pretrained(
        resolved.model_id,
        cache_root=cache_root,
        offline=offline,
        catalog=catalog,
    )
    entries = _product_registry() if catalog is None else catalog
    return ClientModelPlan(
        spec=spec,
        kind=KIND_REGISTRY,
        package_dir=plan.target_package_dir,
        package_present=plan.already_compiled,
        trust_module=entries[plan.model_id].custom_tokenizer_module,
        model_id=plan.model_id,
        revision=plan.revision,
        source_license=plan.source_license,
        estimated_package_bytes=plan.estimated_package_bytes,
        required_free_bytes=plan.required_free_bytes,
        available_free_bytes=plan.available_free_bytes,
        cache_root=plan.cache_root,
        # Carried, not recomputed. Every field the compiler decided about disk
        # reaches the client unaltered, so the two cannot disagree about
        # whether a compile fits.
        snapshot_cache_root=plan.snapshot_cache_root,
        snapshot_required_bytes=plan.snapshot_required_bytes,
        snapshot_available_bytes=plan.snapshot_available_bytes,
    )


def required_tokenizer_module(
    spec: str,
    *,
    cache_root: str | Path | None = None,
    catalog: ModelCatalog | None = None,
) -> str | None:
    """The module this MODEL will import when it loads, or None -- WITHOUT
    reading a snapshot or preparing anything.

    Answerable before preparation, which is the whole point: a cold `run`
    spends minutes and 28 GiB, so the consent question has to be settled first
    or the answer arrives too late to affect what was prepared.
    """
    return plan_client_model(
        spec,
        cache_root=cache_root,
        catalog=catalog,
    ).trust_module


@dataclass(frozen=True, slots=True)
class PreparedModel:
    """A package that is ready to open, and how it got that way."""

    package_dir: Path
    model_id: str | None
    reused_cache: bool
    """False when this call compiled it. None-model (a directory the caller
    already had) is always True -- nothing was prepared."""


class ModelRepository:
    """Application-scoped resolver, compiler cache, and runtime entry point.

    All user-facing model inputs pass through this object.  The Product registry
    is the default scope; supplying a :class:`ModelCatalog` creates an explicit
    Local scope without mutating process-global state.  Backend code never sees
    a model id or catalog -- only the immutable plan produced by the package
    loader.
    """

    __slots__ = ("_application_state", "_cache_root", "_catalog", "_catalog_path")

    def __init__(
        self,
        *,
        catalog: ModelCatalog | None = None,
        cache_root: str | Path | None = None,
        data_root: str | Path | None = None,
        catalog_path: str | Path | None = None,
    ) -> None:
        if catalog is not None and not isinstance(catalog, ModelCatalog):
            raise TypeError("catalog must be a ModelCatalog")
        resolved_catalog_path = None
        if catalog_path is not None:
            if not isinstance(catalog_path, (str, Path)) or isinstance(
                catalog_path, bool
            ):
                raise TypeError("catalog_path must be a path")
            resolved_catalog_path = Path(catalog_path).expanduser().resolve()
            if catalog is None:
                catalog = ModelCatalog.from_toml(resolved_catalog_path)
        self._catalog = catalog
        self._cache_root = _resolve_cache_root(cache_root)
        self._catalog_path = resolved_catalog_path
        self._application_state = ApplicationState(data_root)

    @property
    def catalog(self) -> ModelCatalog | None:
        return self._catalog

    @property
    def catalog_path(self) -> Path | None:
        return self._catalog_path

    @property
    def data_root(self) -> Path:
        return self._application_state.root

    @property
    def cache_root(self) -> Path:
        """Effective package-cache root, fixed for this repository instance."""
        return self._cache_root

    @property
    def application_state(self) -> ApplicationState:
        return self._application_state

    def save_run_profile(self, profile: RunProfile) -> None:
        """Validate, normalize, and atomically persist an exact run profile.

        Most applications should use :meth:`configure_run_profile`, which binds
        identity fields on their behalf.  This lower-level entry point remains
        useful when importing a previously exported ``RunProfile``.
        """

        if not isinstance(profile, RunProfile):
            raise TypeError("profile must be a RunProfile")
        if profile.package_digest is not None:
            # Persist a stable package identity, not a path whose meaning
            # changes with the process working directory.
            profile = replace(
                profile,
                model=str(Path(profile.model).expanduser().resolve(strict=False)),
            )
            if self.verified_package_digest(profile.model) != profile.package_digest:
                raise ConfigurationError("run_profile_package_stale")
            definition = None
        else:
            definition = self._definition_for_run_profile(profile)
            if profile.catalog_path is not None:
                if self._catalog_path is None:  # validated above; keep narrowing local
                    raise ConfigurationError("run_profile_catalog_path_missing")
                profile = replace(profile, catalog_path=str(self._catalog_path))
        self._resolve_run_profile_value(profile, definition=definition)
        self._application_state.save_run_profile(profile)

    def configure_run_profile(
        self,
        name: str,
        model: str | Path,
        *,
        backend: str | None = None,
        precision: str | None = None,
        device_ordinal: int | None = None,
        device_memory_budget_bytes: int | None = None,
        offline: bool | None = None,
        max_tokens: int | None = None,
        context_policy: str | None = None,
        system_prompt: str | None = None,
        deployment_profile_id: str | None = None,
    ) -> RunProfile:
        """Create or replace one profile from a model ID or prepared package.

        The caller supplies choices, not identity bookkeeping.  This repository
        binds the exact catalog definition or a fully verified package receipt,
        checks the selected deployment, and only then publishes the profile.
        Raw Hugging Face snapshots are intentionally refused: they must first be
        prepared through this repository so the resulting bytes have a receipt.
        """

        if not isinstance(model, (str, Path)) or isinstance(model, bool):
            raise TypeError("model must be a model id or package path")
        spec = str(model)
        plan = self.plan(spec, offline=True)
        choices = {
            "backend": backend,
            "precision": precision,
            "device_ordinal": device_ordinal,
            "device_memory_budget_bytes": device_memory_budget_bytes,
            "offline": offline,
            "max_tokens": max_tokens,
            "context_policy": context_policy,
            "system_prompt": system_prompt,
            "deployment_profile_id": deployment_profile_id,
        }
        selected = {key: value for key, value in choices.items() if value is not None}

        if plan.kind == KIND_PACKAGE:
            snapshot = verify_package_receipt_snapshot(plan.package_dir)
            if snapshot is None:
                raise ConfigurationError("run_profile_package_unverifiable")
            profile = RunProfile(
                name=name,
                model=str(plan.package_dir.resolve()),
                package_digest=snapshot.receipt_sha256,
                **selected,
            )
            definition = None
        else:
            if plan.model_id is None or plan.revision is None:
                raise AssertionError("registry plan is missing exact identity")
            if self._catalog is not None and self._catalog_path is None:
                raise ConfigurationError(
                    "run_profile_catalog_path_missing: durable Local profiles "
                    "require ModelRepository(catalog_path=...)"
                )
            entries = _product_registry() if self._catalog is None else self._catalog
            entry = entries[plan.model_id]
            definition = (
                model_definition_for_descriptor(entry)
                if self._catalog is None
                else entry.definition
            )
            profile = RunProfile(
                name=name,
                model=plan.model_id,
                revision=plan.revision,
                model_definition_digest=model_definition_digest(definition),
                catalog_path=(
                    str(self._catalog_path) if self._catalog_path is not None else None
                ),
                **selected,
            )

        # Validate against the same resolved deployment that open_run_profile()
        # will use.  Persisting first would leave a broken profile behind after
        # an otherwise useful editor typo.
        self._resolve_run_profile_value(profile, definition=definition)
        self._application_state.save_run_profile(profile)
        return profile

    @staticmethod
    def verified_package_digest(package: str | Path) -> str:
        """Return a digest only after the compiler receipt verifies the package."""

        snapshot = verify_package_receipt_snapshot(package)
        if snapshot is None:
            raise ConfigurationError("run_profile_package_unverifiable")
        return snapshot.receipt_sha256

    def load_run_profile(self, name: str) -> RunProfile:
        return self._application_state.load_run_profile(name)

    def list_run_profiles(self) -> tuple[RunProfile, ...]:
        return self._application_state.list_run_profiles()

    def delete_run_profile(self, name: str) -> None:
        self._application_state.delete_run_profile(name)

    def list_conversations(self) -> tuple[ConversationInfo, ...]:
        """List local conversations without loading a model package."""

        store = _ConversationStore.persistent(self.data_root)
        try:
            return store.list_sessions()
        finally:
            store.close()

    def delete_conversation(
        self,
        session_id: str,
        *,
        expected_revision: int,
    ) -> None:
        """Delete one exact session using the caller's observed revision."""

        session_id = _validate_session_id(session_id)
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValueError("expected_revision must be a non-negative integer")
        store = _ConversationStore.persistent(self.data_root)
        try:
            store.delete_session_at_revision(
                session_id,
                expected_revision=expected_revision,
            )
        finally:
            store.close()

    def resolve_run_profile(
        self,
        name: str,
        *,
        explicit: "Mapping[str, object] | None" = None,
        framework_defaults: "Mapping[str, object] | None" = None,
    ) -> RunProfile:
        """Load and validate one saved profile before model acquisition."""

        _, resolved = self._resolve_saved_run_profile(
            name, explicit=explicit, framework_defaults=framework_defaults
        )
        return resolved

    def _resolve_saved_run_profile(
        self,
        name: str,
        *,
        explicit: "Mapping[str, object] | None",
        framework_defaults: "Mapping[str, object] | None",
    ) -> tuple["ModelRepository", RunProfile]:
        saved = self.load_run_profile(name)
        repository = self
        if saved.catalog_path is not None and self._catalog is None:
            # A saved Local profile explicitly names its scope. Reopen that
            # immutable catalog for this request, without changing the default
            # repository or allowing the ID to fall through to Product.
            repository = ModelRepository(
                catalog_path=saved.catalog_path,
                cache_root=self._cache_root,
                data_root=self.data_root,
            )
        definition = repository._definition_for_run_profile(saved)
        resolved = repository._resolve_run_profile_value(
            saved,
            definition=definition,
            explicit=explicit,
            framework_defaults=framework_defaults,
        )
        return repository, resolved

    def _resolve_run_profile_value(
        self,
        saved: RunProfile,
        *,
        definition,
        explicit: "Mapping[str, object] | None" = None,
        framework_defaults: "Mapping[str, object] | None" = None,
    ) -> RunProfile:
        """Resolve one already identity-checked record without reading state."""

        deployment_defaults: dict[str, object] = {}
        if definition is not None:
            target = definition.deployments[0]
            deployment_defaults["backend"] = target.backend
            # The current public ``precision`` option is a CUDA selection;
            # MLX precision remains part of its immutable deployment target.
            if target.backend == "cuda_native":
                deployment_defaults["precision"] = target.precision
            if target.deployment is not None:
                deployment_defaults["deployment_profile_id"] = (
                    target.deployment.profile
                )
        resolved = resolve_run_config(
            saved,
            explicit=explicit,
            deployment_defaults=deployment_defaults,
            framework_defaults=framework_defaults,
        )
        if definition is not None:
            target = self._resolve_run_profile_deployment(resolved, definition)
            device_ordinal = resolved.device_ordinal or 0
            device_memory_budget_bytes = (
                resolved.device_memory_budget_bytes or 0
            )
            if target.backend == "cuda_native":
                validate_device_options(
                    target.backend,
                    target.precision,
                    device_ordinal,
                    device_memory_budget_bytes,
                )
            elif device_ordinal != 0 or device_memory_budget_bytes != 0:
                raise ConfigurationError(
                    "device_ordinal and device_memory_budget_bytes are accepted "
                    "only with backend='cuda_native'"
                )
            resolved = replace(
                resolved,
                backend=target.backend,
                precision=target.precision,
                deployment_profile_id=(
                    target.deployment.profile
                    if target.deployment is not None
                    else None
                ),
            )
        else:
            # A bare package has no model definition that can translate semantic
            # precision.  Apply the public backend request contract directly so
            # an invalid profile is rejected before it is saved or acquired.
            backend = resolved.backend
            if backend is None:  # resolve_run_config guarantees this invariant
                raise ConfigurationError("run_configuration_backend_missing")
            try:
                validate_device_options(
                    backend,
                    resolved.precision,
                    resolved.device_ordinal or 0,
                    resolved.device_memory_budget_bytes or 0,
                )
            except ValueError as exc:
                raise ConfigurationError(str(exc)) from exc
        return resolved

    def plan_run_profile(
        self,
        name: str,
        *,
        explicit: "Mapping[str, object] | None" = None,
        framework_defaults: "Mapping[str, object] | None" = None,
    ) -> ClientModelPlan:
        """Plan in the saved profile's scope without acquiring model weights."""

        repository, resolved = self._resolve_saved_run_profile(
            name, explicit=explicit, framework_defaults=framework_defaults
        )
        return repository.plan(resolved.model, offline=bool(resolved.offline))

    def open_run_profile(
        self,
        name: str,
        *,
        engine_cls=None,
        explicit: "Mapping[str, object] | None" = None,
        framework_defaults: "Mapping[str, object] | None" = None,
        trust_remote_code: bool = False,
        expected_module: object = UNSPECIFIED,
    ):
        """Resolve and open the model selected by one saved run profile.

        A profile's precision is semantic deployment identity. The existing
        public Engine ``precision`` argument is narrower: it is a CUDA device
        request. This repository is the single translation boundary, so later
        Conversation, CLI, and TUI callers never need backend-specific rules.
        Generation and context fields remain inputs to the U1-B conversation
        layer; this method owns model acquisition only.
        """

        repository, resolved = self._resolve_saved_run_profile(
            name,
            explicit=explicit,
            framework_defaults=framework_defaults,
        )
        backend = resolved.backend
        if backend is None:  # resolve_run_profile() guarantees this invariant
            raise ConfigurationError("run_configuration_backend_missing")
        request_precision = (
            resolved.precision
            if backend == "cuda_native" or resolved.package_digest is not None
            else None
        )
        return repository.open(
            resolved.model,
            engine_cls=engine_cls,
            revision=(
                None if resolved.package_digest is not None else resolved.revision
            ),
            offline=bool(resolved.offline),
            trust_remote_code=trust_remote_code,
            expected_module=expected_module,
            backend=backend,
            deployment_profile=resolved.deployment_profile_id,
            # An exact model definition has already bound MLX precision to its
            # deployment. A bare package has no such authority, so preserving
            # its value lets open() reject unsupported non-CUDA precision
            # instead of silently discarding it.
            precision=request_precision,
            device_ordinal=resolved.device_ordinal or 0,
            device_memory_budget_bytes=(
                resolved.device_memory_budget_bytes or 0
            ),
        )

    def _definition_for_run_profile(self, profile: RunProfile):
        if profile.package_digest is not None:
            if profile.catalog_path is not None:
                raise ConfigurationError("run_profile_package_catalog_forbidden")
            lease = acquire_package_load_lease(profile.model)
            try:
                actual_digest = compile_receipt_digest(
                    profile.model,
                    _package_guard=lease,
                )
            finally:
                lease.close()
            if actual_digest is None:
                raise ConfigurationError("run_profile_package_unverifiable")
            if actual_digest != profile.package_digest:
                raise ConfigurationError("run_profile_package_stale")
            return None

        entries = _product_registry() if self._catalog is None else self._catalog
        try:
            entry = entries[profile.model]
        except KeyError:
            raise ConfigurationError("run_profile_model_stale") from None
        if self._catalog is None:
            if profile.catalog_path is not None:
                raise ConfigurationError("run_profile_product_catalog_forbidden")
            definition = model_definition_for_descriptor(entry)
        else:
            if profile.catalog_path is None or self._catalog_path is None:
                raise ConfigurationError("run_profile_catalog_path_missing")
            if Path(profile.catalog_path).expanduser().resolve() != self._catalog_path:
                raise ConfigurationError("run_profile_catalog_path_stale")
            definition = entry.definition
        if (
            profile.revision != entry.revision
            or profile.model_definition_digest
            != model_definition_digest(definition)
        ):
            raise ConfigurationError("run_profile_model_stale")
        return definition

    @staticmethod
    def _resolve_run_profile_deployment(profile: RunProfile, definition):
        targets = definition.deployments
        candidates = tuple(
            target
            for target in targets
            if (
                profile.backend == "auto"
                or target.backend == profile.backend
            )
            and (
                profile.precision is None
                or target.precision == profile.precision
            )
        )
        if profile.deployment_profile_id is not None:
            candidates = tuple(
                target
                for target in candidates
                if target.deployment is not None
                and target.deployment.profile == profile.deployment_profile_id
            )
        if not candidates:
            raise ConfigurationError(
                "run_profile_deployment_not_admitted: "
                f"{definition.key}:{profile.backend}:{profile.precision}:"
                f"{profile.deployment_profile_id}"
            )
        return candidates[0]

    def plan(
        self,
        spec: str,
        *,
        allow_local_dir: bool = True,
        offline: bool = True,
    ) -> ClientModelPlan:
        return plan_client_model(
            spec,
            cache_root=self._cache_root,
            allow_local_dir=allow_local_dir,
            offline=offline,
            catalog=self._catalog,
        )

    def prepare(
        self,
        spec: str,
        *,
        trust_remote_code: bool = False,
        offline: bool = True,
        on_stage: "Callable[[str], None] | None" = None,
    ) -> PreparedModel:
        emit = on_stage or (lambda _stage: None)
        emit(STAGE_PLANNING)
        plan = self.plan(spec, offline=offline)
        return self.prepare_plan(
            plan,
            trust_remote_code=trust_remote_code,
            offline=offline,
            on_stage=emit,
            _planning_emitted=True,
        )

    def prepare_plan(
        self,
        plan: ClientModelPlan,
        *,
        trust_remote_code: bool = False,
        offline: bool = True,
        on_stage: "Callable[[str], None] | None" = None,
        _planning_emitted: bool = False,
    ) -> PreparedModel:
        """Prepare one already-resolved plan through the guarded compiler."""

        if not isinstance(plan, ClientModelPlan):
            raise TypeError("plan must be a ClientModelPlan")
        return _prepare_planned_impl(
            plan,
            trust_remote_code=trust_remote_code,
            offline=offline,
            on_stage=on_stage,
            _planning_emitted=_planning_emitted,
            catalog=self._catalog,
        )

    def report(self, spec: str, **options) -> OfflineModelReport:
        """Diagnose a model through this repository's exact catalog scope."""

        return model_report(
            spec,
            cache_root=self._cache_root,
            catalog=self._catalog,
            **options,
        )

    def cached_models(self, *, verify: bool = False) -> list[CachedModel]:
        return list_cached_models(
            cache_root=self._cache_root,
            verify=verify,
            catalog=self._catalog,
        )

    def describe(self, model_id: str) -> dict[str, object]:
        """Describe a catalog model without downloading or loading its weights.

        Operations describe the default deployment's contract, not a device
        probe or certification. Use plan()/doctor for resource readiness.
        """
        from .presets import apply_preset, public_operation_support

        entries = _product_registry() if self._catalog is None else self._catalog
        try:
            entry = entries[model_id]
        except KeyError:
            raise ModelSpecError(
                f"Unknown model {model_id!r}. Use diffusorrt models --list for exact IDs."
            ) from None
        definition = (
            model_definition_for_descriptor(entry)
            if self._catalog is None else entry.definition
        )
        contract = definition.decode_policy.contract
        defaults = definition.decode_policy.defaults
        strategy = contract.decode_strategy if contract else defaults.get("decode_strategy", "block_threshold")
        deployments = []
        for target in definition.deployments:
            deployment = target.deployment
            profile = deployment.profile if deployment else "baseline"
            preset = (
                apply_preset({}, profile)
                if deployment and deployment.config_source == "preset" and profile != "baseline"
                else {}
            )
            operations = public_operation_support(
                backend=target.backend,
                decode_strategy=str(preset.get("mlx_diffusion_decode_strategy", strategy)),
                policy_profile=str(preset.get("mlx_diffusion_policy_profile") or "baseline"),
                attention_mask_kind=str(
                    preset.get("attention_mask_kind")
                    or (deployment.attention_mask_kind if deployment else None)
                    or "bidirectional"
                ),
            )
            deployments.append({
                "backend": target.backend, "precision": target.precision,
                "minimum_device": target.minimum_device, "profile": profile,
                "operations": operations,
            })
        return {
            "model_id": definition.model_id, "revision": definition.revision,
            "source_license": definition.source_license,
            "operations": deployments[0]["operations"],
            "deployments": deployments,
            "hardware_checked": False,
        }

    def managed_models(self) -> list[CachedModel]:
        """Return the lightweight model-manager view with machine readiness."""

        usable_backends = machine_report().usable_backends
        return [
            replace(model, usable_backends=usable_backends)
            for model in self.cached_models(verify=False)
        ]

    def verify(self, model_or_package: str | Path) -> Path:
        """Verify one repository-owned compiled package under a shared lease."""

        package_dir, model_id, revision, expected_entry = self._owned_package(
            model_or_package
        )
        if package_dir.is_symlink():
            raise ModelVerificationError(f"package_symlink_forbidden: {package_dir}")
        if not package_dir.is_dir():
            raise ModelVerificationError(f"package_not_found: {package_dir}")
        verified = verify_compile_receipt_snapshot(
            package_dir,
            model_id=model_id,
            revision=revision,
            expected_definition_key=(
                expected_entry.key if self._catalog is not None else None
            ),
            expected_definition_digest=(
                expected_entry.definition_digest
                if self._catalog is not None
                else None
            ),
        )
        if verified is None:
            raise ModelVerificationError(f"package_not_verified: {package_dir}")
        return package_dir

    def remove(self, model_or_package: str | Path) -> Path:
        """Remove one verified package owned by this repository's cache.

        External directories, symlinked targets, corrupt/unverified packages,
        and live packages are refused.  Source snapshots are never candidates.
        """

        package_dir, model_id, revision, expected_entry = self._owned_package(
            model_or_package
        )
        if package_dir.is_symlink():
            raise ModelRemovalError(f"package_symlink_forbidden: {package_dir}")
        if not package_dir.is_dir():
            raise ModelRemovalError(f"package_not_found: {package_dir}")
        try:
            guard = acquire_package_exclusive_guard(package_dir)
        except PackageBusyError as exc:
            raise ModelPackageBusyError(f"package_busy: {package_dir}") from exc
        try:
            verified = verify_compile_receipt_snapshot(
                package_dir,
                model_id=model_id,
                revision=revision,
                expected_definition_key=(
                    expected_entry.key if self._catalog is not None else None
                ),
                expected_definition_digest=(
                    expected_entry.definition_digest
                    if self._catalog is not None
                    else None
                ),
                _package_guard=guard,
            )
            if verified is None:
                raise ModelRemovalError(f"package_not_verified: {package_dir}")
            shutil.rmtree(package_dir)
        finally:
            guard.close()
        return package_dir

    def _owned_package(self, model_or_package: str | Path):
        if not isinstance(model_or_package, (str, Path)) or isinstance(
            model_or_package, bool
        ):
            raise TypeError("model_or_package must be a model id or path")
        text = str(model_or_package)
        entries = _product_registry() if self._catalog is None else self._catalog
        if text in entries:
            candidate_ids = (text,)
            requested_path = None
        else:
            candidate_ids = tuple(entries)
            requested_lexical = Path(model_or_package).expanduser().absolute()
            if requested_lexical.is_symlink():
                raise ModelRemovalError(
                    f"package_symlink_forbidden: {requested_lexical}"
                )
            requested_path = requested_lexical.resolve(strict=False)
        for model_id in candidate_ids:
            plan = plan_pretrained(
                model_id,
                cache_root=self._cache_root,
                offline=True,
                catalog=self._catalog,
            )
            target_lexical = plan.target_package_dir.expanduser().absolute()
            if target_lexical.is_symlink():
                raise ModelRemovalError(
                    f"package_symlink_forbidden: {target_lexical}"
                )
            target = target_lexical.resolve(strict=False)
            if requested_path is None or requested_path == target:
                root = plan.cache_root.expanduser().resolve(strict=False)
                if target.parent != root:
                    raise ModelRemovalError(
                        f"repository_package_outside_cache: {target}"
                    )
                return target_lexical, plan.model_id, plan.revision, entries[model_id]
        raise ModelRemovalError(f"package_not_repository_owned: {model_or_package}")

    def open_package(
        self,
        package: str | Path,
        *,
        engine_cls=None,
        backend: str = "auto",
        deployment_profile: str | None = None,
        precision: str | None = None,
        device_ordinal: int = 0,
        device_memory_budget_bytes: int = 0,
        trust_remote_code: bool = False,
        expected_module: object = UNSPECIFIED,
    ):
        """Load one prepared package through the shared immutable-plan seam."""

        if engine_cls is None:
            from .engine import Engine

            engine_cls = Engine
        return engine_cls._from_package_impl(
            package,
            backend=backend,
            deployment_profile=deployment_profile,
            precision=precision,
            device_ordinal=device_ordinal,
            device_memory_budget_bytes=device_memory_budget_bytes,
            trust_remote_code=trust_remote_code,
            expected_module=expected_module,
            catalog=self._catalog,
            data_root=self.data_root,
        )

    def open_deployment(
        self,
        package: str | Path,
        *,
        target: str = "auto",
        trust_remote_code: bool = False,
        expected_module: object = UNSPECIFIED,
        precision: str | None = None,
        device_ordinal: int = 0,
        device_memory_budget_bytes: int = 0,
    ):
        """Open a checked service session in this repository's catalog scope."""

        from .deployment import _prepare_local_deployment_impl

        return _prepare_local_deployment_impl(
            package,
            target=target,
            trust_remote_code=trust_remote_code,
            expected_module=expected_module,
            precision=precision,
            device_ordinal=device_ordinal,
            device_memory_budget_bytes=device_memory_budget_bytes,
            model_catalog=self._catalog,
        )

    def open(
        self,
        spec: str,
        *,
        engine_cls=None,
        revision: str | None = None,
        offline: bool = False,
        trust_remote_code: bool = False,
        expected_module: object = UNSPECIFIED,
        backend: str = "auto",
        deployment_profile: str | None = None,
        precision: str | None = None,
        device_ordinal: int = 0,
        device_memory_budget_bytes: int = 0,
        on_stage: Callable[[str], None] | None = None,
    ):
        """Resolve, prepare, and load an exact id or prepared package path."""

        from .engine import Engine

        if engine_cls is None:
            engine_cls = Engine
        backend = validate_backend(backend)
        validate_device_options(
            backend,
            precision,
            device_ordinal,
            device_memory_budget_bytes,
        )
        validate_deployment_profile(deployment_profile)
        plan = self.plan(spec, offline=offline)
        if revision is not None and plan.revision is None:
            raise ValueError(
                "revision is accepted only for a catalog or Product model ID; "
                "a prepared package already carries its source identity"
            )
        if revision is not None and plan.revision is not None and revision != plan.revision:
            from .compiler import RevisionMismatchError

            raise RevisionMismatchError(
                f"revision {revision!r} disagrees with the pinned revision "
                f"{plan.revision!r} for {spec}"
            )
        if self._catalog is not None and plan.model_id is not None:
            entry = self._catalog[plan.model_id]
            effective_backend = "mlx_local" if backend == "auto" else backend
            effective_precision = precision or "fp32"
            if effective_backend == "cuda_native":
                raise ValueError(
                    "cuda_native for a Local catalog requires a registered "
                    "provider package; the repository builds only the canonical "
                    "base package"
                )
            entry.definition.resolve_deployment(
                effective_backend, effective_precision
            )
        prepared = self.prepare_plan(
            plan,
            trust_remote_code=trust_remote_code,
            offline=offline,
            on_stage=on_stage,
        )
        load_expected_module = expected_module
        if load_expected_module is UNSPECIFIED and trust_remote_code:
            # The immutable pre-acquisition plan already bound the exact module
            # for both model IDs and package paths.  Re-deriving it from a
            # registry would leave package-bound profiles with a boolean grant.
            load_expected_module = plan.trust_module
        load_options = {
            "backend": backend,
            "deployment_profile": deployment_profile,
            "precision": precision,
            "device_ordinal": device_ordinal,
            "device_memory_budget_bytes": device_memory_budget_bytes,
            "trust_remote_code": trust_remote_code,
            "expected_module": load_expected_module,
        }
        if engine_cls is Engine:
            return self.open_package(
                prepared.package_dir,
                engine_cls=engine_cls,
                **load_options,
            )
        # A caller-supplied Engine subclass may intentionally customise the
        # public construction seam.  Keep that supported while the default
        # product path stays on this repository instance.
        return engine_cls.from_package(
            prepared.package_dir,
            catalog=self._catalog,
            **load_options,
        )


def prepare_model(
    spec: str,
    *,
    trust_remote_code: bool = False,
    cache_root: str | Path | None = None,
    offline: bool = True,
    on_stage: "Callable[[str], None] | None" = None,
    catalog: ModelCatalog | None = None,
) -> PreparedModel:
    """The ONE way pull/run/serve turn a MODEL argument into a usable package.

    Compile-or-reuse, per plan.md B3: a registered model that is not prepared
    is prepared here rather than refused with an instruction. A directory the
    caller already has is returned untouched.

    Exactly ONE receipt verification per call. The earlier client asked
    offline_model_report() first -- which hashes the package -- and then
    compile_pretrained(), which hashes it again: 28 GiB read twice, turning a
    "10 second" cache hit into 20 seconds of user wall clock.

    Goes through the public compiler surface only. The client has no business
    importing compiler.build internals, and did.
    """
    return ModelRepository(catalog=catalog, cache_root=cache_root).prepare(
        spec,
        trust_remote_code=trust_remote_code,
        offline=offline,
        on_stage=on_stage,
    )


def prepare_planned(
    plan: ClientModelPlan,
    *,
    trust_remote_code: bool = False,
    offline: bool = True,
    on_stage: "Callable[[str], None] | None" = None,
    _planning_emitted: bool = False,
    catalog: ModelCatalog | None = None,
) -> PreparedModel:
    """Compatibility adapter for a caller that already has the plan."""

    return ModelRepository(catalog=catalog, cache_root=plan.cache_root).prepare_plan(
        plan,
        trust_remote_code=trust_remote_code,
        offline=offline,
        on_stage=on_stage,
        _planning_emitted=_planning_emitted,
    )


def _prepare_planned_impl(
    plan: ClientModelPlan,
    *,
    trust_remote_code: bool = False,
    offline: bool = True,
    on_stage: "Callable[[str], None] | None" = None,
    _planning_emitted: bool = False,
    catalog: ModelCatalog | None = None,
) -> PreparedModel:
    """Repository-owned preparation for a caller that already has the plan.

    Every CLI command resolves once and then prepares, so re-resolving inside
    preparation would mean the package that gets built is chosen by a second,
    later reading of the filesystem than the one the user consented to.
    """
    from .compiler import compile_pretrained

    emit = on_stage or (lambda _stage: None)
    if not _planning_emitted:
        emit(STAGE_PLANNING)

    if plan.kind != KIND_REGISTRY:
        # A directory the caller already has: nothing is acquired, converted or
        # published, so those stages are not emitted. Reporting stages that did
        # not happen is the same defect class as a progress bar that finishes
        # before the work does.
        emit(STAGE_READY)
        return PreparedModel(
            package_dir=plan.package_dir,
            model_id=None,
            reused_cache=True,
        )

    result = compile_pretrained(
        plan.model_id,
        cache_root=plan.cache_root,
        offline=offline,
        trust_remote_code=trust_remote_code,
        on_stage=emit,
        catalog=catalog,
    )
    return PreparedModel(
        package_dir=result.package_dir,
        model_id=plan.model_id,
        reused_cache=result.reused_cache,
    )


def resolve_model(
    spec: str,
    *,
    allow_local_dir: bool = True,
    allow_hf_snapshot: bool = False,
    cache_root: str | Path | None = None,
    catalog: ModelCatalog | None = None,
) -> ResolvedModel:
    """Resolve a MODEL argument, in the order plan.md section 6 fixes.

    1. an existing directory is classified by its markers -- an execution
       package, a raw Hugging Face snapshot, or neither;
    2. an exact ID in the compiler registry is a supported pretrained model;
    3. anything else fails and lists the supported IDs.

    Step 1 used to accept ANY existing directory as a prepared package, so a
    raw snapshot -- or an empty directory -- was handed on as something
    load_runtime could open. It cannot.

    ``allow_local_dir=False`` is `pull`: a directory is already local and has
    nothing to acquire.

    ``allow_hf_snapshot`` defaults to False because no P0 command can run a raw
    snapshot; it must be compiled first. Detecting it is still worthwhile: the
    refusal can say "this is a snapshot, compile it" instead of "not a
    supported model", which is the difference between an actionable message and
    a dead end.

    Step 2 delegates to ``plan_pretrained``, which raises
    ``UnsupportedModelError`` naming the supported set. That refusal is not
    re-implemented here -- one wording, one place.
    """
    entries = _product_registry() if catalog is None else catalog
    # Before anything else: Path("") is PosixPath("."), and "." is a directory,
    # so an empty MODEL argument would resolve as "the current directory is a
    # prepared package" and fail much later with a confusing error. "." itself
    # stays legal -- running a package from the working directory is reasonable.
    if not spec.strip():
        raise ModelSpecError(
            "no model given. Pass a supported model ID "
            f"({', '.join(sorted(entries))}) or a path to a prepared "
            "execution package."
        )

    supported = ", ".join(sorted(entries))
    candidate = Path(spec).expanduser()
    if candidate.is_dir():
        # Classify FIRST, then apply the command's rule. Refusing on
        # allow_local_dir before knowing what the directory is gave an empty
        # directory and a raw snapshot the same "nothing to pull" message --
        # true of neither. An empty directory is not a local model at all, and
        # a snapshot's problem is identity, not redundancy.
        kind = classify_local_dir(candidate)

        if kind is None:
            raise ModelSpecError(
                f"{spec!r} is a directory but neither an execution package "
                "(no model.json) nor a Hugging Face snapshot (no config.json "
                "alongside .safetensors weights). Pass a prepared package, or "
                f"a supported model ID ({supported})."
            )

        if kind == KIND_PACKAGE and not allow_local_dir:
            raise ModelSpecError(
                f"{spec!r} is already a prepared execution package, so there "
                f"is nothing to pull. Pass a supported model ID ({supported}), "
                "or use the package directly."
            )

        if kind == KIND_HF_SNAPSHOT and not (allow_local_dir and allow_hf_snapshot):
            # Deliberately does NOT tell the user to `pull` this directory:
            # pull resolves a registered ID against the standard HF cache and
            # would not compile the path they passed. Handing a Qwen or LLaDA
            # snapshot and being told to pull Dream is worse than no advice.
            raise ModelSpecError(
                f"{spec!r} is a raw Hugging Face snapshot, not an execution "
                "package. P0 accepts only registered model IDs resolved "
                "against the standard Hugging Face cache; an arbitrary "
                "directory cannot prove its model ID or revision. "
                f"Supported IDs: {supported}."
            )

        return ResolvedModel(
            spec=spec,
            kind=kind,
            local_path=candidate.resolve(),
        )

    plan = plan_pretrained(spec, cache_root=cache_root, catalog=catalog)
    return ResolvedModel(
        spec=spec,
        kind=KIND_REGISTRY,
        # Where the package WOULD live. It need not exist yet -- the caller
        # checks, which is what `list` already does through the same plan.
        local_path=plan.target_package_dir,
        model_id=plan.model_id,
        revision=plan.revision,
    )


def list_cached_models(
    *,
    cache_root: str | Path | None = None,
    verify: bool = False,
    catalog: ModelCatalog | None = None,
) -> list[CachedModel]:
    """Report every REGISTRY entry's cache state, and only those.

    It does not walk the cache root looking for directories: a stray or
    hand-copied directory is not something the compiler manages, and listing it
    would imply `run` could use it. The registry is the enumeration.
    """
    models: list[CachedModel] = []
    registry = _product_registry() if catalog is None else catalog
    for model_id in sorted(registry):
        plan = plan_pretrained(model_id, cache_root=cache_root, catalog=catalog)
        if not plan.already_compiled:
            status = CACHE_MISSING
        elif not verify:
            status = CACHE_CACHED
        else:
            verified = verify_compile_receipt_snapshot(
                plan.target_package_dir,
                model_id=plan.model_id,
                revision=plan.revision,
                expected_definition_key=(
                    registry[model_id].key if catalog is not None else None
                ),
                expected_definition_digest=(
                    registry[model_id].definition_digest
                    if catalog is not None
                    else None
                ),
                _require_current_compiler=False,
            )
            status = (
                CACHE_CORRUPT if verified is None else
                CACHE_VALID if (
                    verified.compiler_version == COMPILER_VERSION
                    and (catalog is not None or verified.descriptor_digest == registry[model_id].descriptor_digest)
                ) else
                CACHE_NEEDS_REBUILD
            )
        models.append(
            CachedModel(
                model_id=plan.model_id,
                revision=plan.revision,
                # From the plan, not re-read from the registry here: one source
                # for the licence and the size estimate.
                source_license=plan.source_license,
                package_bytes=plan.estimated_package_bytes,
                package_dir=plan.target_package_dir,
                status=status,
                source_bytes=plan.estimated_snapshot_bytes,
                package_available_bytes=plan.available_free_bytes,
                source_available_bytes=plan.snapshot_available_bytes,
            )
        )
    return models
