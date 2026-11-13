"""Local execution-package deployment through the shared model/runtime plan."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from .backend_ids import (
    AUTO_BACKEND,
    CUDA_NATIVE_BACKEND,
    MLX_LOCAL_BACKEND,
    NUMPY_LOCAL_BACKEND,
)
from .trust_identity import UNSPECIFIED

if TYPE_CHECKING:
    from .model_catalog import ModelCatalog


DEFAULT_MLX_DEPLOYMENT_PROFILE = "active-window-exact-d2f-bs8"
HEALTH_PROMPT = "DiffusorRT health check."
HEALTH_MAX_TOKENS = 1
DEPLOYMENT_TARGETS = (
    AUTO_BACKEND,
    CUDA_NATIVE_BACKEND,
    MLX_LOCAL_BACKEND,
    NUMPY_LOCAL_BACKEND,
)


class DeploymentError(RuntimeError):
    def __init__(
        self,
        code: str,
        *,
        detail: str | None = None,
        capability_report: object | None = None,
    ):
        self.code = code
        self.detail = detail
        self.capability_report = capability_report
        message = code if detail is None else f"{code}: {detail}"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class ResolvedDeploymentProfile:
    name: str
    model_config: dict[str, Any]


@dataclass(frozen=True, slots=True)
class DeploymentHealth:
    ok: bool
    backend: str
    max_tokens: int
    finish_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "backend": self.backend,
            "max_tokens": self.max_tokens,
            "finish_reason": self.finish_reason,
        }


@dataclass(slots=True)
class LocalDeploymentSession:
    runtime: object
    backend: str
    profile: str
    health: DeploymentHealth
    capability_report: object
    precision: str | None = None
    device_ordinal: int = 0
    device_memory_budget_bytes: int = 0

    def close(self) -> None:
        """Release the runtime this session holds. Idempotent and retryable.

        The controller dropping ITS reference was not enough: the session that
        created the runtime kept holding it, so every buffer stayed resident
        after a `serve` returned. Whoever owns the object has to let go, and
        both of them own one.
        """
        runtime = self.runtime
        # Direct call: RuntimeHandle always defines close(). The
        # getattr/callable dance existed to tolerate test doubles that did not.
        if runtime is not None:
            runtime.close()
            self.runtime = None


def _health_contract_for_runtime(runtime: object, profile: str) -> tuple[str, int]:
    """Return runtime-owned health input; certification never controls a run."""

    descriptor_key = str(getattr(runtime, "model_descriptor_key", "") or "")
    executable = getattr(runtime, "_resolved_model_plan", None)
    load_request = getattr(executable, "load", None)
    backend = str(
        getattr(load_request, "backend", "")
        or getattr(runtime, "admitted_backend", "")
        or ""
    )
    precision = str(
        getattr(load_request, "precision", "")
        or getattr(runtime, "model_precision", "")
        or ""
    )
    plan = getattr(runtime, "_resolved_build_plan", None)
    definition = getattr(plan, "definition", None)
    if definition is None and descriptor_key:
        from .model_definitions import model_definition_for_catalog_key

        definition = model_definition_for_catalog_key(descriptor_key)
    if definition is None:
        return HEALTH_PROMPT, HEALTH_MAX_TOKENS
    # A CUDA profile is admitted per package and projected into the plan; the
    # model definition does not list it.
    target = getattr(executable, "deployment", None)
    if target is None:
        try:
            target = definition.resolve_deployment(backend, precision)
        except ValueError as exc:
            raise DeploymentError(
                "deployment_health_contract_invalid",
                detail=(
                    f"profile={profile!r} model={definition.key!r} "
                    f"backend={backend!r} precision={precision!r}: {exc}"
                ),
            ) from exc
    if target.deployment is not None and profile != target.deployment.profile:
        raise DeploymentError(
            "deployment_health_contract_invalid",
            detail=f"profile={profile!r} model={definition.key!r}",
        )
    policy = definition.deployment_policy
    return policy.health_prompt, policy.health_max_tokens


def resolve_deployment_profile(
    model_config: Mapping[str, Any],
    *,
    backend: str,
    descriptor_key: str = "",
    precision: str = "fp32",
) -> ResolvedDeploymentProfile:
    from .model import _resolve_deployment_profile_config

    profile, config = _resolve_deployment_profile_config(
        dict(model_config),
        backend=backend,
        mlx_profile_name=DEFAULT_MLX_DEPLOYMENT_PROFILE,
        descriptor_key=descriptor_key,
        precision=precision,
    )
    return ResolvedDeploymentProfile(
        name=profile,
        model_config=config,
    )


def prepare_local_deployment(
    package: str | Path,
    *,
    target: str = AUTO_BACKEND,
    trust_remote_code: bool = False,
    expected_module: object = UNSPECIFIED,
    precision: str | None = None,
    device_ordinal: int = 0,
    device_memory_budget_bytes: int = 0,
    model_catalog: ModelCatalog | None = None,
) -> LocalDeploymentSession:
    """Compatibility adapter into the application-scoped model repository."""

    from .local_models import ModelRepository

    return ModelRepository(catalog=model_catalog).open_deployment(
        package,
        target=target,
        trust_remote_code=trust_remote_code,
        expected_module=expected_module,
        precision=precision,
        device_ordinal=device_ordinal,
        device_memory_budget_bytes=device_memory_budget_bytes,
    )


def _prepare_local_deployment_impl(
    package: str | Path,
    *,
    target: str = AUTO_BACKEND,
    trust_remote_code: bool = False,
    expected_module: object = UNSPECIFIED,
    precision: str | None = None,
    device_ordinal: int = 0,
    device_memory_budget_bytes: int = 0,
    model_catalog: ModelCatalog | None = None,
) -> LocalDeploymentSession:
    """Prepare one runtime for local serving.

    ``trust_remote_code`` defaults to False and is threaded to the single
    runtime load. A package is untrusted data: one that declares an ``auto_map``
    tokenizer module gets that module imported when it loads, so loading it
    executes code the package author wrote. Packages with no such declaration
    need no consent and are not asked for it.
    """
    if target not in DEPLOYMENT_TARGETS:
        raise DeploymentError(
            "deployment_target_unknown",
            detail=target,
        )
    from .backend_ids import validate_device_options

    try:
        validate_device_options(
            target,
            precision,
            device_ordinal,
            device_memory_budget_bytes,
        )
    except ValueError as exc:
        raise DeploymentError(
            "deployment_device_options_invalid",
            detail=str(exc),
        ) from exc
    from .package_lifecycle import acquire_package_load_lease

    package_path = Path(package).expanduser().absolute()
    discovery_lease = acquire_package_load_lease(package_path)

    # Lazy, like _load_runtime_once below: cli.py imports this module at load
    # time, so a module-level `from .tokenizer import ...` would put the
    # tokenizer stack behind `diffusorrt --help`. Caught by
    # test_importing_cli_does_not_import_runtime_dependency_graph.
    from .tokenizer import package_declares_custom_tokenizer_code

    # Ask for consent before calling the loader at all. This is NOT the only
    # defence -- TokenizerSpec.from_package still receives trust_remote_code and
    # still refuses AutoTokenizer's custom-code path without it, and the actual
    # tokenizer load happens further in, inside load_runtime. The value here is
    # a clear, early refusal and zero loader invocations, not a claim that
    # entering load_runtime has already executed anything.
    #
    # The question is "does this package ship a tokenizer module?", not "does it
    # use the Transformers tokenizer": Qwen3-MDLM and LLaDA are real packages
    # with auto_map=null that execute nothing, and gating on the latter refuses
    # them for code they do not have.
    try:
        if not package_path.is_dir():
            raise DeploymentError(
                "deployment_package_not_found",
                detail=str(package_path),
            )
        try:
            custom_module = package_declares_custom_tokenizer_code(package_path)
        except ValueError as exc:
            # Unreadable or unsafe tokenizer_config: refuse rather than treat
            # "cannot tell" as "nothing to consent to".
            raise DeploymentError(
                "deployment_tokenizer_config_invalid",
                detail=str(exc),
            ) from exc
    finally:
        discovery_lease.close()

    if custom_module is not None and not trust_remote_code:
        raise DeploymentError(
            "deployment_remote_code_not_trusted",
            detail=(
                f"{package_path} declares a custom tokenizer module "
                f"({custom_module}) that is imported when the package loads. "
                "Pass trust_remote_code=True to permit that, having satisfied "
                "yourself the package is one you trust."
            ),
        )

    load_expected_module = expected_module
    if trust_remote_code and expected_module is UNSPECIFIED:
        # Bind the caller's package-level grant to what the pre-load discovery
        # observed. None is intentional: if a module appears before the
        # materialized tokenizer is imported, the tokenizer boundary refuses
        # the mismatch instead of widening a boolean grant.
        load_expected_module = custom_module

    try:
        load_device_options = {}
        if target == CUDA_NATIVE_BACKEND:
            load_device_options = {
                "precision": precision,
                "device_ordinal": device_ordinal,
                "device_memory_budget_bytes": device_memory_budget_bytes,
            }
        catalog_options = (
            {} if model_catalog is None else {"model_catalog": model_catalog}
        )
        loaded = _load_runtime_once(
            package_path,
            target=target,
            trust_remote_code=trust_remote_code,
            expected_module=load_expected_module,
            **catalog_options,
            **load_device_options,
        )
    except Exception as exc:
        raise DeploymentError(
            "deployment_package_load_failed",
            detail=str(exc) or type(exc).__name__,
        ) from exc
    runtime = loaded.runtime
    try:
        deployment_profile = loaded.profile
        diagnose_options = {}
        if precision is not None:
            diagnose_options["cuda_precision"] = precision
        report = _diagnose_loaded_runtime(
            runtime,
            package_path=package_path,
            requested_backend=target,
            deployment_profile=deployment_profile,
            **diagnose_options,
        )
        if not report.healthy or report.selected_backend is None:
            reasons = ",".join(report.reasons) or "backend_unavailable"
            raise DeploymentError(
                "deployment_preflight_failed",
                detail=reasons,
                capability_report=report,
            )

        backend = report.selected_backend
        if backend != loaded.backend:
            raise DeploymentError(
                "deployment_backend_resolution_mismatch",
                detail=f"preload={loaded.backend!r} capability={backend!r}",
                capability_report=report,
            )

        health_prompt, health_max_tokens = _health_contract_for_runtime(
            runtime, deployment_profile
        )
        try:
            device_options = {}
            if backend == CUDA_NATIVE_BACKEND:
                device_options = {
                    "precision": precision,
                    "device_ordinal": device_ordinal,
                    "device_memory_budget_bytes": device_memory_budget_bytes,
                }
            result = runtime.generate(
                health_prompt,
                backend=backend,
                max_tokens=health_max_tokens,
                return_debug=True,
                **device_options,
            )
        except Exception as exc:
            raise DeploymentError(
                "deployment_health_inference_failed",
                detail=str(exc) or type(exc).__name__,
                capability_report=report,
            ) from exc

        debug = getattr(result, "debug", None)
        metrics = getattr(debug, "metrics", {}) if debug is not None else {}
        resolved_backend = metrics.get(
            "resolved_backend",
            metrics.get("generation_backend"),
        )
        if resolved_backend != backend:
            raise DeploymentError(
                "deployment_health_backend_mismatch",
                detail=f"expected={backend} actual={resolved_backend!r}",
                capability_report=report,
            )
    except BaseException as primary:
        try:
            runtime.close()
        except BaseException as cleanup:
            if hasattr(primary, "add_note"):
                primary.add_note(
                    "deployment runtime close also failed: "
                    f"{type(cleanup).__name__}: {cleanup}"
                )
        raise

    health = DeploymentHealth(
        ok=True,
        backend=backend,
        max_tokens=health_max_tokens,
        finish_reason=str(getattr(result, "finish_reason", "unknown")),
    )
    return LocalDeploymentSession(
        runtime=runtime,
        backend=backend,
        profile=deployment_profile,
        health=health,
        capability_report=report,
        precision=precision,
        device_ordinal=device_ordinal,
        device_memory_budget_bytes=device_memory_budget_bytes,
    )


def _load_runtime_once(
    package_path: Path,
    *,
    target: str = AUTO_BACKEND,
    precision: str | None = None,
    device_ordinal: int = 0,
    device_memory_budget_bytes: int = 0,
    trust_remote_code: bool = False,
    expected_module: object = UNSPECIFIED,
    model_catalog: ModelCatalog | None = None,
):
    """The single runtime load. Exactly one call site, by design.

    Keyword-only and threaded straight through -- NOT wrapped in a
    `try: load_runtime(..., trust_remote_code=x) except TypeError:` retry.
    Feature-detecting a signature by catching TypeError would also swallow a
    genuine TypeError raised inside the loader and silently fall back to a
    load that ignores the trust decision.
    """
    from .model import load_runtime_for_deployment

    device_options = {}
    if target == CUDA_NATIVE_BACKEND:
        device_options = {
            "precision": precision,
            "device_ordinal": device_ordinal,
            "device_memory_budget_bytes": device_memory_budget_bytes,
        }
    return load_runtime_for_deployment(
        package_path,
        requested_backend=target,
        mlx_profile_name=DEFAULT_MLX_DEPLOYMENT_PROFILE,
        trust_remote_code=trust_remote_code,
        expected_module=expected_module,
        model_catalog=model_catalog,
        **device_options,
    )


def _diagnose_loaded_runtime(
    runtime: object,
    *,
    package_path: Path,
    requested_backend: str,
    deployment_profile: str,
    cuda_precision: str | None = None,
):
    from .capabilities import diagnose_loaded_runtime

    return diagnose_loaded_runtime(
        runtime,
        package_path=package_path,
        requested_backend=requested_backend,
        deployment_profile=deployment_profile,
        cuda_precision=cuda_precision,
    )
