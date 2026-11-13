"""Resolve, authorize, prepare and open one model, without a terminal.

`diffusorrt chat` and the chat window open models through this one function.
It never prompts and never logs a failure: trust consent is a callback the
caller supplies, and every refusal is a `ModelOpenError` carrying the exact
text the CLI has always logged, plus a next action where one exists.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .backend_ids import AUTO_BACKEND, CUDA_NATIVE_BACKEND

TrustConsent = Callable[[str, str], bool]
"""(spec, module) -> whether the user grants importing that module."""

PackageOpener = Callable[..., Any]
"""(package_dir, *, backend, trust_remote_code, expected_module, device_options) -> Engine."""


@dataclass(frozen=True)
class OpenRequest:
    spec: str | None
    backend: str = AUTO_BACKEND
    precision: str | None = None
    device_ordinal: int = 0
    device_memory_budget_bytes: int = 0
    offline: bool = False
    allow_prepare: bool = True
    run_profile: str | None = None
    run_profile_explicit: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class OpenedModel:
    engine: Any
    package_dir: Path
    backend: str
    precision: str | None


class ModelOpenError(Exception):
    """A model that could not be opened, with a stable code and a next action."""

    def __init__(self, code: str, message: str, next_action: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.next_action = next_action


def open_model(
    repository: Any,
    request: OpenRequest,
    *,
    consent: TrustConsent,
    on_stage: Callable[[str], None] | None = None,
    open_package: PackageOpener | None = None,
) -> OpenedModel:
    """Open the model `request` names through `repository`, or raise `ModelOpenError`."""
    if request.run_profile is not None:
        if not request.allow_prepare:
            # Opening a run profile prepares its model, which can download or compile.
            raise ModelOpenError(
                "run_profile_unsupported",
                f"run profile {request.run_profile!r} cannot be opened without preparing its model",
                "open the model by name or package folder instead",
            )
        return _open_run_profile(repository, request, consent=consent)

    plan = _plan(repository, request)
    if not request.allow_prepare and not plan.package_present:
        # Refused before the trust question: there is no code to allow yet.
        raise ModelOpenError(
            "not_prepared",
            f"{request.spec} is not prepared on this machine",
            f"diffusorrt pull {request.spec}",
        )
    trust_permitted, expected_module = _settle_trust(plan, consent)
    if request.allow_prepare:
        package_dir = _prepare(repository, plan, request, trust=trust_permitted, on_stage=on_stage)
    else:
        # Preparing can compile for minutes; a caller that refuses that opens
        # the package as it is, and a stale one fails to load below.
        package_dir = plan.package_dir

    opener = open_package or _repository_opener(repository)
    try:
        engine = opener(
            package_dir,
            backend=request.backend,
            trust_remote_code=trust_permitted,
            expected_module=expected_module,
            device_options=_device_options(request),
        )
    except Exception as error:
        # No next action: a load can fail for reasons a pull does not fix.
        raise ModelOpenError("load_failed", f"could not load {package_dir}: {error}") from error
    return OpenedModel(engine, Path(package_dir), request.backend, _precision(request))


def _plan(repository: Any, request: OpenRequest) -> Any:
    from .compiler import CompilerError
    from .local_models import ModelSpecError

    try:
        return repository.plan(request.spec, allow_local_dir=True, offline=request.offline)
    except (CompilerError, ModelSpecError) as error:
        raise ModelOpenError("model_spec", f"{error}") from error
    except (OSError, ValueError) as error:
        raise ModelOpenError(
            "repository_config", f"model repository configuration failed: {error}"
        ) from error


def _settle_trust(plan: Any, consent: TrustConsent) -> tuple[bool, object]:
    """Grant exactly what was asked for, and nothing when nothing needs it.

    A package that declares no module is loaded with trust_remote_code=False
    and an UNSPECIFIED expectation, so the loader refuses custom code outright.
    """
    from .trust_identity import UNSPECIFIED

    module = plan.trust_module
    if module is None:
        return False, UNSPECIFIED
    if not consent(plan.spec, module):
        raise ModelOpenError("trust_denied", f"remote code in {module} was not trusted")
    return True, module


def _prepare(repository: Any, plan: Any, request: OpenRequest, *, trust: bool,
             on_stage: Callable[[str], None] | None) -> Path:
    from .compiler import CompilerError
    from .local_models import ModelSpecError

    try:
        prepared = repository.prepare_plan(
            plan, trust_remote_code=trust, offline=request.offline, on_stage=on_stage
        )
    except (CompilerError, ModelSpecError) as error:
        raise ModelOpenError("prepare_failed", f"{error}") from error
    return prepared.package_dir


def _open_run_profile(repository: Any, request: OpenRequest, *, consent: TrustConsent) -> OpenedModel:
    name = request.run_profile
    explicit = dict(request.run_profile_explicit)
    try:
        plan = repository.plan_run_profile(name, explicit=explicit)
    except Exception as error:
        raise ModelOpenError(
            "run_profile_plan", f"could not plan run profile {name!r}: {error}"
        ) from error
    trust_permitted, expected_module = _settle_trust(plan, consent)
    try:
        engine = repository.open_run_profile(
            name,
            explicit=explicit,
            trust_remote_code=trust_permitted,
            expected_module=expected_module,
        )
    except Exception as error:
        raise ModelOpenError(
            "run_profile_load", f"could not load run profile {name!r}: {error}"
        ) from error
    return OpenedModel(engine, Path(plan.package_dir), request.backend, _precision(request))


def _repository_opener(repository: Any) -> PackageOpener:
    def opener(package_dir, *, backend, trust_remote_code, expected_module, device_options):
        return repository.open_package(
            package_dir,
            backend=backend,
            trust_remote_code=trust_remote_code,
            expected_module=expected_module,
            **device_options,
        )

    return opener


def _device_options(request: OpenRequest) -> dict[str, object]:
    if request.backend != CUDA_NATIVE_BACKEND:
        return {}
    return {
        "precision": request.precision,
        "device_ordinal": request.device_ordinal,
        "device_memory_budget_bytes": request.device_memory_budget_bytes,
    }


def _precision(request: OpenRequest) -> str | None:
    return request.precision if request.backend == CUDA_NATIVE_BACKEND else None
