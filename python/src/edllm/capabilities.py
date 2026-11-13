"""Pure capability diagnostics for DiffusorRT backends and execution packages.

Importing this module does not import NumPy, MLX, Torch, Transformers, or the
runtime loader. ``probe_environment`` imports optional dependencies only when a
diagnostic is executed, so broken installations are not reported as importable.
Package loading and the existing backend-selection function are imported only
when a caller supplies a package path.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from importlib import import_module, metadata
import platform
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any

from .backend_ids import (
    AUTO_BACKEND,
    CUDA_NATIVE_BACKEND,
    CUDA_PRECISION_IDS,
    EXPLICIT_BACKEND_IDS,
    HF_DREAM_REFERENCE_BACKEND,
    MLX_LOCAL_BACKEND,
    NUMPY_LOCAL_BACKEND,
    PLANNED_BACKEND_IDS,
    SELECTABLE_BACKEND_IDS,
)
from .trust_identity import UNSPECIFIED
from .backends.kv_capacity import exact_d2f_kv_capacity, validate_request_bounds

if TYPE_CHECKING:
    from .model_catalog import ModelCatalog


CAPABILITY_SCHEMA_VERSION = "1.2"


@dataclass(frozen=True, slots=True)
class CapabilityEnvironment:
    python_version: str
    python_supported: bool
    system: str
    machine: str
    available_modules: frozenset[str]
    distribution_versions: tuple[tuple[str, str], ...]
    # Modules that are installed but failed to import, as opposed to absent ones.
    module_failures: tuple[tuple[str, str], ...] = ()

    def has_module(self, name: str) -> bool:
        return name in self.available_modules

    def module_failure(self, name: str) -> str | None:
        return dict(self.module_failures).get(name)

    def distribution_version(self, name: str) -> str | None:
        return dict(self.distribution_versions).get(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "python_version": self.python_version,
            "python_supported": self.python_supported,
            "system": self.system,
            "machine": self.machine,
            "available_modules": sorted(self.available_modules),
            "distribution_versions": dict(self.distribution_versions),
        }


@dataclass(frozen=True, slots=True)
class BackendCapability:
    backend_id: str
    selectable: bool
    accelerated: bool
    platform_class: str
    importable: bool
    machine_usable: bool
    package_compatible: bool | None
    usable: bool
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "selectable": self.selectable,
            "accelerated": self.accelerated,
            "platform_class": self.platform_class,
            "importable": self.importable,
            "machine_usable": self.machine_usable,
            "package_compatible": self.package_compatible,
            "usable": self.usable,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class RuntimeCapabilityReport:
    environment: CapabilityEnvironment
    package_path: str | None
    package_status: str
    package_error: str | None
    requested_backend: str
    auto_backend: str | None
    selected_backend: str | None
    deployment_profile: str | None
    healthy: bool
    backends: tuple[BackendCapability, ...]
    reasons: tuple[str, ...] = ()
    schema_version: str = CAPABILITY_SCHEMA_VERSION
    kv_capacity: dict[str, Any] | None = None

    def backend(self, backend_id: str) -> BackendCapability:
        for capability in self.backends:
            if capability.backend_id == backend_id:
                return capability
        raise KeyError(backend_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "environment": self.environment.to_dict(),
            "package_path": self.package_path,
            "package_status": self.package_status,
            "package_error": self.package_error,
            "requested_backend": self.requested_backend,
            "auto_backend": self.auto_backend,
            "selected_backend": self.selected_backend,
            "deployment_profile": self.deployment_profile,
            "healthy": self.healthy,
            "backends": [backend.to_dict() for backend in self.backends],
            "reasons": list(self.reasons),
            "kv_capacity": self.kv_capacity,
        }


def probe_environment() -> CapabilityEnvironment:
    module_names = (
        "diffusorrt_runtime",
        "mlx",
        "numpy",
        "peft",
        "torch",
        "transformers",
    )
    statuses = {name: _module_import_status(name) for name in module_names}
    available_modules = frozenset(
        name for name, status in statuses.items() if status is None
    )
    module_failures = tuple(
        sorted(
            (name, status)
            for name, status in statuses.items()
            if status is not None and status != "absent"
        )
    )
    versions = tuple(
        (name, version)
        for name in ("peft", "transformers")
        if (version := _distribution_version(name)) is not None
    )
    return CapabilityEnvironment(
        python_version=platform.python_version(),
        python_supported=sys.version_info >= (3, 11),
        system=platform.system(),
        machine=platform.machine(),
        available_modules=available_modules,
        distribution_versions=versions,
        module_failures=module_failures,
    )


def diagnose_runtime(
    package: str | Path | None = None,
    *,
    requested_backend: str = AUTO_BACKEND,
    environment: CapabilityEnvironment | None = None,
    trust_remote_code: bool = False,
    prompt_tokens: int | None = None,
    max_tokens: int | None = None,
    cuda_precision: str | None = None,
) -> RuntimeCapabilityReport:
    """Diagnose backends, optionally against a package.

    ``trust_remote_code`` is threaded into the load. Without it a package whose
    tokenizer lives in its own repository fails to open, and the report called
    a working package `invalid` -- measured on the pinned Dream package, which
    `run` executes successfully: `real_tokenizer_load_error: AutoTokenizer
    failed for non-fixture package`. The caller must have obtained consent;
    this function does not ask.
    """
    return _diagnose_runtime(
        package,
        requested_backend=requested_backend,
        environment=environment,
        trust_remote_code=trust_remote_code,
        deployment_profile_name=None,
        expected_module=UNSPECIFIED,
        prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
        cuda_precision=cuda_precision,
    )


def diagnose_deployment_runtime(
    package: str | Path,
    *,
    requested_backend: str = AUTO_BACKEND,
    environment: CapabilityEnvironment | None = None,
    trust_remote_code: bool = False,
    expected_module: object = UNSPECIFIED,
    mlx_profile_name: str,
    prompt_tokens: int | None = None,
    max_tokens: int | None = None,
    cuda_precision: str | None = None,
    cuda_device_memory_budget_bytes: int = 0,
    model_catalog: ModelCatalog | None = None,
) -> RuntimeCapabilityReport:
    """Diagnose the same profiled package load that local deployment uses."""
    return _diagnose_runtime(
        package,
        requested_backend=requested_backend,
        environment=environment,
        trust_remote_code=trust_remote_code,
        deployment_profile_name=mlx_profile_name,
        expected_module=expected_module,
        prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
        cuda_precision=cuda_precision,
        cuda_device_memory_budget_bytes=cuda_device_memory_budget_bytes,
        model_catalog=model_catalog,
    )


def _diagnose_runtime(
    package: str | Path | None,
    *,
    requested_backend: str,
    environment: CapabilityEnvironment | None,
    trust_remote_code: bool,
    deployment_profile_name: str | None,
    expected_module: object,
    prompt_tokens: int | None,
    max_tokens: int | None,
    cuda_precision: str | None,
    model_catalog: ModelCatalog | None = None,
    cuda_device_memory_budget_bytes: int = 0,
) -> RuntimeCapabilityReport:
    validate_request_bounds(prompt_tokens, max_tokens)
    _validate_requested_backend(requested_backend)
    _validate_cuda_precision(requested_backend, cuda_precision)
    environment = environment or probe_environment()
    package_path = None if package is None else str(Path(package).expanduser().resolve())
    runtime = None
    package_status = "not_checked"
    package_error = None
    deployment_profile = None
    report_reasons: list[str] = []

    if package_path is not None:
        resolved_package = Path(package_path)
        if not resolved_package.is_dir():
            package_status = "invalid"
            package_error = "package_not_found"
            report_reasons.append("package_not_found")
        else:
            try:
                if deployment_profile_name is None:
                    if requested_backend == CUDA_NATIVE_BACKEND:
                        runtime = _load_cuda_runtime_package(
                            resolved_package,
                            trust_remote_code=trust_remote_code,
                        )
                    else:
                        runtime = _load_runtime_package(
                            resolved_package,
                            trust_remote_code=trust_remote_code,
                        )
                else:
                    deployment_options = {}
                    if requested_backend == CUDA_NATIVE_BACKEND:
                        deployment_options["precision"] = cuda_precision
                        deployment_options["device_memory_budget_bytes"] = (
                            cuda_device_memory_budget_bytes
                        )
                    if model_catalog is not None:
                        deployment_options["model_catalog"] = model_catalog
                    loaded = _load_deployment_runtime_package(
                        resolved_package,
                        requested_backend=requested_backend,
                        mlx_profile_name=deployment_profile_name,
                        trust_remote_code=trust_remote_code,
                        expected_module=expected_module,
                        **deployment_options,
                    )
                    runtime = loaded.runtime
                    deployment_profile = loaded.profile
                    resolved_backend = _resolve_runtime_backend(
                        runtime,
                        requested_backend,
                    )
                    if resolved_backend != loaded.backend:
                        raise ValueError(
                            "deployment_backend_resolution_mismatch: "
                            f"preload={loaded.backend!r} "
                            f"runtime={resolved_backend!r}"
                        )
            except Exception as exc:
                package_status = "invalid"
                package_error = str(exc) or type(exc).__name__
                report_reasons.append("package_invalid")
            else:
                package_status = "valid"

    try:
        report = _build_report(
            runtime=runtime,
            environment=environment,
            package_path=package_path,
            package_status=package_status,
            package_error=package_error,
            requested_backend=requested_backend,
            report_reasons=report_reasons,
            deployment_profile=deployment_profile,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            cuda_precision=cuda_precision,
        )
    except BaseException as primary:
        if runtime is not None:
            try:
                runtime.close()
            except BaseException as cleanup:
                if hasattr(primary, "add_note"):
                    primary.add_note(
                        "diagnostic runtime close also failed: "
                        f"{type(cleanup).__name__}: {cleanup}"
                    )
        raise
    if runtime is None:
        return report
    try:
        runtime.close()
    except Exception as exc:
        detail = str(exc) or type(exc).__name__
        return replace(
            report,
            package_status="invalid",
            package_error=f"runtime_close_failed:{detail}",
            selected_backend=None,
            healthy=False,
            reasons=(*report.reasons, "runtime_close_failed"),
        )
    return report


def diagnose_loaded_runtime(
    runtime: object,
    *,
    package_path: str | Path,
    requested_backend: str = AUTO_BACKEND,
    environment: CapabilityEnvironment | None = None,
    deployment_profile: str | None = None,
    prompt_tokens: int | None = None,
    max_tokens: int | None = None,
    cuda_precision: str | None = None,
) -> RuntimeCapabilityReport:
    """Diagnose an already-loaded package without loading it a second time."""
    _validate_requested_backend(requested_backend)
    _validate_cuda_precision(requested_backend, cuda_precision)
    validate_request_bounds(prompt_tokens, max_tokens)
    return _build_report(
        runtime=runtime,
        environment=environment or probe_environment(),
        package_path=str(Path(package_path).expanduser().resolve()),
        package_status="valid",
        package_error=None,
        requested_backend=requested_backend,
        report_reasons=[],
        deployment_profile=deployment_profile,
        prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
        cuda_precision=cuda_precision,
    )


def _build_report(
    *,
    runtime: object | None,
    environment: CapabilityEnvironment,
    package_path: str | None,
    package_status: str,
    package_error: str | None,
    requested_backend: str,
    report_reasons: list[str],
    deployment_profile: str | None,
    prompt_tokens: int | None,
    max_tokens: int | None,
    cuda_precision: str | None,
) -> RuntimeCapabilityReport:
    backends = tuple(
        _backend_capability(
            backend_id,
            environment=environment,
            runtime=runtime,
            cuda_precision=cuda_precision,
        )
        for backend_id in (*SELECTABLE_BACKEND_IDS, *PLANNED_BACKEND_IDS)
    )

    auto_backend = None
    selected_backend = None
    candidate = requested_backend
    if runtime is not None:
        auto_backend = _resolve_runtime_backend(runtime, AUTO_BACKEND)
        candidate = (
            auto_backend
            if requested_backend == AUTO_BACKEND
            else requested_backend
        )
        candidate_capability = next(
            backend for backend in backends if backend.backend_id == candidate
        )
        if candidate_capability.usable:
            selected_backend = candidate
        else:
            prefix = (
                "auto_backend_unavailable"
                if requested_backend == AUTO_BACKEND
                else "requested_backend_unavailable"
            )
            report_reasons.append(f"{prefix}:{candidate}")
    elif package_path is None and requested_backend != AUTO_BACKEND:
        candidate_capability = next(
            backend
            for backend in backends
            if backend.backend_id == requested_backend
        )
        if candidate_capability.machine_usable:
            selected_backend = requested_backend
        else:
            report_reasons.append(
                f"requested_backend_unavailable:{requested_backend}"
            )

    if not environment.python_supported:
        report_reasons.insert(0, "python_unsupported:requires_3_11_or_newer")

    if package_path is None and requested_backend == AUTO_BACKEND:
        healthy = environment.python_supported and any(
            backend.machine_usable
            for backend in backends
            if backend.selectable
        )
    else:
        healthy = (
            environment.python_supported
            and package_status != "invalid"
            and selected_backend is not None
        )

    kv_capacity = _kv_capacity_report(runtime, candidate, prompt_tokens, max_tokens)
    if kv_capacity["status"] == "invalid":
        # A supplied request that cannot be admitted must not get Ready: yes.
        healthy = False
        report_reasons.append(f"kv_capacity_invalid:{kv_capacity['reason']}")

    return RuntimeCapabilityReport(
        environment=environment,
        package_path=package_path,
        package_status=package_status,
        package_error=package_error,
        requested_backend=requested_backend,
        auto_backend=auto_backend,
        selected_backend=selected_backend,
        deployment_profile=deployment_profile,
        healthy=healthy,
        backends=backends,
        reasons=tuple(report_reasons),
        kv_capacity=kv_capacity,
    )


def _kv_capacity_report(
    runtime: object | None, backend: str,
    prompt_tokens: int | None, max_tokens: int | None,
) -> dict[str, Any]:
    """Logical limits only; availability and physical memory fit are separate."""
    report = {
        "status": "not_checked",
        "reason": "package_not_opened",
        "prompt_tokens": prompt_tokens,
        "max_tokens": max_tokens,
        "physical_peak_bytes": None,
        "device_memory_fit": None,
    }
    if runtime is None:
        return report
    config = runtime.model_config
    if backend != MLX_LOCAL_BACKEND:
        return {**report, "status": "not_applicable", "reason": "backend_not_covered"}
    if not config.get("mlx_diffusion_d2f_exact_kv_cache", False):
        return {**report, "status": "not_applicable", "reason": "exact_d2f_not_enabled"}
    if prompt_tokens is None:
        return {**report, "status": "needs_request_bounds", "reason": "token_bounds_required"}
    try:
        capacity = exact_d2f_kv_capacity(
            config, prompt_tokens=prompt_tokens, max_tokens=max_tokens,
        )
    except ValueError as exc:
        return {**report, "status": "invalid", "reason": str(exc)}
    return {**report, **capacity.to_dict(), "status": "available", "reason": None}


def _validate_requested_backend(requested_backend: str) -> None:
    if (
        requested_backend != AUTO_BACKEND
        and requested_backend not in EXPLICIT_BACKEND_IDS
        and requested_backend not in PLANNED_BACKEND_IDS
    ):
        raise ValueError(f"unknown backend: {requested_backend}")


def _validate_cuda_precision(
    requested_backend: str,
    cuda_precision: str | None,
) -> None:
    if cuda_precision is not None and cuda_precision not in CUDA_PRECISION_IDS:
        raise ValueError(f"unknown CUDA precision: {cuda_precision}")
    if requested_backend != CUDA_NATIVE_BACKEND and cuda_precision is not None:
        raise ValueError("CUDA precision requires requested_backend='cuda_native'")


def _backend_capability(
    backend_id: str,
    *,
    environment: CapabilityEnvironment,
    runtime: object | None,
    cuda_precision: str | None,
) -> BackendCapability:
    if backend_id in PLANNED_BACKEND_IDS:
        return BackendCapability(
            backend_id=backend_id,
            selectable=False,
            accelerated=True,
            platform_class="mobile_embedded",
            importable=False,
            machine_usable=False,
            package_compatible=None,
            usable=False,
            reasons=("not_implemented",),
        )

    reasons: list[str] = []
    if backend_id == NUMPY_LOCAL_BACKEND:
        accelerated = False
        platform_class = "portable_reference"
        importable = environment.has_module("numpy")
        machine_usable = environment.python_supported and importable
        if not importable:
            reasons.append(_dependency_reason(environment, "numpy"))
    elif backend_id == MLX_LOCAL_BACKEND:
        accelerated = True
        platform_class = "apple_silicon"
        importable = environment.has_module("mlx")
        platform_supported = (
            environment.system == "Darwin"
            and environment.machine.lower() in {"arm64", "aarch64"}
        )
        machine_usable = (
            environment.python_supported and importable and platform_supported
        )
        if not importable:
            reasons.append(_dependency_reason(environment, "mlx"))
        if not platform_supported:
            reasons.append("platform_unsupported:requires_darwin_arm64")
    elif backend_id == CUDA_NATIVE_BACKEND:
        accelerated = True
        platform_class = "nvidia_single_pc"
        importable = environment.has_module("diffusorrt_runtime")
        platform_supported = (
            environment.system in {"Linux", "Windows"}
            and environment.machine.lower() in {"x86_64", "amd64"}
        )
        machine_usable = (
            environment.python_supported and importable and platform_supported
        )
        if not importable:
            reasons.append(_dependency_reason(environment, "diffusorrt_runtime"))
        if not platform_supported:
            reasons.append("platform_unsupported:requires_x86_64_linux_or_windows")
        if cuda_precision is None:
            reasons.append("cuda_precision_required")
    elif backend_id == HF_DREAM_REFERENCE_BACKEND:
        accelerated = False
        platform_class = "external_oracle"
        required = ("torch", "transformers")
        missing = [
            name for name in required if not environment.has_module(name)
        ]
        importable = not missing
        for name in missing:
            reasons.append(_dependency_reason(environment, name))
        transformers_version = environment.distribution_version("transformers")
        version_supported = (
            transformers_version is not None
            and transformers_version.startswith("4.46.")
        )
        if importable and not version_supported:
            reasons.append(
                "dependency_version_unsupported:transformers_requires_4_46_x"
            )
        machine_usable = (
            environment.python_supported and importable and version_supported
        )
    else:  # Defensive: all callers iterate shared constants.
        raise ValueError(f"unknown backend: {backend_id}")

    package_compatible, package_reasons = _package_compatibility(
        backend_id,
        runtime,
        environment=environment,
        cuda_precision=cuda_precision,
    )
    reasons.extend(package_reasons)
    usable = (
        machine_usable
        and package_compatible is not False
        and not (backend_id == CUDA_NATIVE_BACKEND and cuda_precision is None)
    )
    return BackendCapability(
        backend_id=backend_id,
        selectable=True,
        accelerated=accelerated,
        platform_class=platform_class,
        importable=importable,
        machine_usable=machine_usable,
        package_compatible=package_compatible,
        usable=usable,
        reasons=tuple(reasons),
    )


def _package_compatibility(
    backend_id: str,
    runtime: object | None,
    *,
    environment: CapabilityEnvironment,
    cuda_precision: str | None,
) -> tuple[bool | None, tuple[str, ...]]:
    if runtime is None:
        return None, ()

    # CUDA admits one exact package through its profile below. The descriptor
    # records the backend the package was compiled for, which never names CUDA.
    if backend_id != CUDA_NATIVE_BACKEND:
        try:
            _validate_descriptor_backend(runtime, backend_id)
        except ValueError as exc:
            return False, (str(exc),)

    if backend_id == NUMPY_LOCAL_BACKEND:
        if bool(getattr(runtime, "fixture_mode", False)):
            return True, ()
        if not getattr(runtime, "package_dir", None) or not getattr(
            runtime, "merged_weights_ref", ""
        ):
            return False, ("package_weights_missing",)
        from .backends.model_adapter import select_numpy_executor_adapter

        model_config = getattr(runtime, "model_config", None) or {}
        raw_mask_token_id = model_config.get("mask_token_id")
        mask_token_id = (
            raw_mask_token_id
            if type(raw_mask_token_id) is int and raw_mask_token_id >= 0
            else 0
        )
        try:
            select_numpy_executor_adapter(model_config, mask_token_id)
        except ValueError as exc:
            return False, (str(exc),)
        return True, ()

    if backend_id == MLX_LOCAL_BACKEND:
        try:
            _validate_mlx_package(runtime)
        except ValueError as exc:
            return False, (str(exc),)
        return True, ()

    if backend_id == CUDA_NATIVE_BACKEND:
        package_dir = getattr(runtime, "package_dir", None)
        if not package_dir:
            return False, ("package_weights_missing",)
        from .backends.cuda_profile import admit_cuda_execution_profile

        try:
            admit_cuda_execution_profile(
                package_dir,
                precision=cuda_precision,
                model_config=getattr(runtime, "model_config", None),
            )
        except (OSError, TypeError, ValueError) as exc:
            return False, (str(exc) or type(exc).__name__,)
        return True, ()

    if backend_id == HF_DREAM_REFERENCE_BACKEND:
        base_ref = str(getattr(runtime, "source_base_model_reference", "") or "")
        if not base_ref:
            return False, ("package_reference_model_missing",)
        adapter_ref = str(
            getattr(runtime, "source_checkpoint_reference", "") or ""
        )
        if (
            adapter_ref
            and adapter_ref != base_ref
            and not environment.has_module("peft")
        ):
            return False, ("dependency_missing:peft",)
        return True, ()

    raise ValueError(f"unknown backend: {backend_id}")


def _load_runtime_package(package: Path, *, trust_remote_code: bool = False):
    from .model import load_runtime

    return load_runtime(package, trust_remote_code=trust_remote_code)


def _load_cuda_runtime_package(package: Path, *, trust_remote_code: bool = False):
    from .model import _load_runtime_for_backend

    return _load_runtime_for_backend(
        package,
        requested_backend=CUDA_NATIVE_BACKEND,
        trust_remote_code=trust_remote_code,
    )


def _load_deployment_runtime_package(
    package: Path,
    *,
    requested_backend: str,
    mlx_profile_name: str,
    precision: str | None = None,
    device_memory_budget_bytes: int = 0,
    trust_remote_code: bool = False,
    expected_module: object = UNSPECIFIED,
    model_catalog: ModelCatalog | None = None,
):
    from .model import load_runtime_for_deployment

    device_options = {}
    if requested_backend == CUDA_NATIVE_BACKEND:
        device_options["precision"] = precision
        device_options["device_memory_budget_bytes"] = device_memory_budget_bytes
    return load_runtime_for_deployment(
        package,
        requested_backend=requested_backend,
        mlx_profile_name=mlx_profile_name,
        trust_remote_code=trust_remote_code,
        expected_module=expected_module,
        model_catalog=model_catalog,
        **device_options,
    )


def _resolve_runtime_backend(runtime: object, requested_backend: str) -> str:
    from .model import _resolve_generation_backend

    return _resolve_generation_backend(
        requested_backend,
        fixture_mode=bool(getattr(runtime, "fixture_mode", False)),
        backend_memory_owner=str(
            getattr(runtime, "backend_memory_owner", "numpy")
        ),
        forward_executor=str(
            getattr(runtime, "forward_executor", "python_scaffold")
        ),
    )


def _validate_mlx_package(runtime: object) -> None:
    if not bool(getattr(runtime, "fixture_mode", False)) and (
        not getattr(runtime, "package_dir", None)
        or not getattr(runtime, "merged_weights_ref", "")
    ):
        raise ValueError("package_weights_missing")
    model_config = getattr(runtime, "model_config", None) or {}
    mask_token_id = model_config.get("mask_token_id", 0)
    try:
        resolve_plan = getattr(runtime, "_model_plan", None)
        if callable(resolve_plan):
            resolve_plan(int(mask_token_id))
        else:
            from .model import _select_mlx_adapter

            _select_mlx_adapter(model_config, int(mask_token_id))
    except (TypeError, ValueError) as exc:
        detail = str(exc)
        if detail.startswith("unsupported_architecture_family:"):
            raise ValueError(detail) from exc
        raise ValueError("package_mlx_config_invalid") from exc

    if model_config.get("mlx_diffusion_d2f_exact_kv_cache") is True:
        _validate_native_kv_capability()


def _validate_native_kv_capability() -> None:
    """Prove the private owner needed by exact-D2F can open and close.

    Importability is insufficient: a stale extension, an absent Apple device
    provider, or a manager that cannot shut down all make the requested
    deployment unusable.  Opening an empty manager exercises the same
    acquisition seam as generation without allocating model K/V payloads.
    """
    from .backends.kv_runtime import NativeKVUnavailable, open_manager

    try:
        manager = open_manager(device="mlx")
    except NativeKVUnavailable as exc:
        raise ValueError(f"package_native_kv_unavailable:{exc}") from exc
    try:
        manager.close()
    except Exception as exc:
        detail = str(exc) or type(exc).__name__
        raise ValueError(f"package_native_kv_shutdown_failed:{detail}") from exc


def _validate_descriptor_backend(runtime: object, backend_id: str) -> None:
    """Apply canonical package admission to every selectable backend."""
    descriptor_key = str(getattr(runtime, "model_descriptor_key", "") or "")
    if not descriptor_key:
        return

    from .model_coverage.producer_descriptors import runtime_descriptor_for_key

    descriptor = runtime_descriptor_for_key(descriptor_key)
    if getattr(runtime, "model_descriptor_digest", "") != descriptor.descriptor_digest:
        raise ValueError("package_model_descriptor_digest_mismatch")
    if getattr(runtime, "source_checkpoint_reference", "") != descriptor.model_id:
        raise ValueError("package_model_descriptor_identity_mismatch")
    if getattr(runtime, "source_checkpoint_sha", "") != descriptor.revision:
        raise ValueError("package_model_descriptor_revision_mismatch")
    package_backend = str(getattr(runtime, "admitted_backend", "") or "")
    if package_backend != backend_id:
        raise ValueError(f"package_backend_not_admitted:{backend_id}")
    precision = str(getattr(runtime, "model_precision", "") or "")
    descriptor.resolve_backend(backend_id, precision)


def _module_import_status(name: str) -> str | None:
    """None when ``name`` imports; otherwise ``absent`` or ``unloadable``.

    Only the package itself being missing is absence. A module it needs being
    missing, or a native library the loader cannot open, is an installed
    package that cannot load, which needs a different fix.
    """
    try:
        import_module(name)
    except ModuleNotFoundError as error:
        return "absent" if error.name == name else "unloadable"
    except Exception:
        return "unloadable"
    return None


def _dependency_reason(environment: CapabilityEnvironment, name: str) -> str:
    """A stable reason code; the exception text, which can name host paths, stays out."""
    if environment.module_failure(name) == "unloadable":
        return f"dependency_unloadable:{name}"
    return f"dependency_missing:{name}"


def _distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None
