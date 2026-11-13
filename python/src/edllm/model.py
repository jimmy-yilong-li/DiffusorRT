from __future__ import annotations

import contextlib
import copy
import json
from collections.abc import Callable, Mapping
from importlib import resources
import dataclasses
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import jsonschema

from .backend_ids import (
    CUDA_NATIVE_BACKEND,
    EXPLICIT_BACKEND_IDS,
    MLX_LOCAL_BACKEND,
    PLANNED_BACKEND_IDS,
)
from .backends.kernel_table import KernelDispatchPlan, plan_kernel_dispatch
from .decode import run_block_diffusion_decode, run_dream_masked_diffusion_decode
from .executor import (
    ExecutorInput,
    DreamTransformerExecutor,
    FakeExecutor,
    PROBED_MLX_EXECUTOR_SOURCE,
    TRANSFORMER_NUMPY_EXECUTOR_SOURCE,
)
from .graph import (
    DenseTransformerPlan,
    GraphIR,
    build_graph_ir,
    compile_dense_transformer_plan,
    same_mask_kind_family,
    uniform_attention_mask_kind,
)
from .model_definitions import (
    DeploymentTarget,
    ModelAdapter,
    ModelDefinition,
    model_build_digest,
    model_definition_digest,
    model_definition_for_catalog_key,
    model_definition_from_payload,
    runtime_architecture_for_config,
)
from .model_catalog import ModelCatalog
from .package_artifacts import open_package_artifact, read_open_artifact, sha256_stream
from .package_lifecycle import PackageLease, acquire_package_load_lease
from .presets import apply_policy_profile, apply_preset, validate_policy_profile_task
from .result import DebugBundle, GenerateResult, InfillResult
from .runtime_config import (
    ResolvedMLXRuntimeConfig,
    resolve_mlx_runtime_config,
    validate_core_model_config,
    validate_generation_length,
    validate_package_snapshot_config,
    validate_request_sampling,
)
from .control_tokens import valid_token_id
from .tokenizer import TokenizerSpec
from .trust_identity import UNSPECIFIED
from .validation import _passes_text_quality_gate
from .weight_store import (
    SAFETENSORS_SHARDED_V1,
    WeightArtifactSpec,
    open_verified_weight_shard,
    read_weight_store_index,
    weight_artifact_spec_from_manifest,
)


def _close_preserving_primary(
    close: Callable[[], object],
    primary: BaseException,
    *,
    label: str,
) -> None:
    """Run checked runtime cleanup without replacing the active failure."""

    try:
        close()
    except BaseException as cleanup:
        primary.add_note(
            f"{label} also failed: {type(cleanup).__name__}: {cleanup}"
        )


RUNTIME_PACKAGE_VERSION = (2, 0)
LEGACY_PACKAGE_MAJOR = 1
LEGACY_PACKAGE_MINOR = 1
_LOAD_TIME_FALLBACK_MLX_DEFAULTS = {
    "decode_strategy": "block_threshold",
    "block_size": 8,
    "confidence_threshold": 0.9,
    "gamma": 0.1,
}
FULL_FORWARD_EXECUTOR_SOURCES = frozenset({TRANSFORMER_NUMPY_EXECUTOR_SOURCE})
PROBE_EXECUTOR_SOURCE = PROBED_MLX_EXECUTOR_SOURCE
_MAX_PACKAGE_MANIFEST_BYTES = 4 * 1024 * 1024
_MAX_PACKAGE_GRAPH_BYTES = 64 * 1024 * 1024


def _resolve_generation_backend(
    requested_backend: str,
    *,
    fixture_mode: bool,
    backend_memory_owner: str,
    forward_executor: str,
) -> str:
    if requested_backend in EXPLICIT_BACKEND_IDS:
        return requested_backend
    if requested_backend in PLANNED_BACKEND_IDS:
        raise ValueError(
            f"backend {requested_backend!r} is planned but not implemented; "
            "no runtime path exists for it yet."
        )
    if requested_backend != "auto":
        raise ValueError(f"unknown generation backend: {requested_backend}")
    if fixture_mode:
        return "numpy_local"
    if backend_memory_owner == "mlx" and forward_executor == "apple_mlx_backend":
        return "mlx_local"
    return "numpy_local"


def _resolve_caller_snapshot_path(value: str | Path | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, (str, Path)) or isinstance(value, bool):
        raise TypeError("snapshot_save_path must be a path or None")
    if isinstance(value, str) and not value:
        raise ValueError("snapshot_save_path must not be empty")
    return str(Path(value))


def _record_generation_backend_provenance(
    result: GenerateResult,
    *,
    requested_backend: str,
    resolved_backend: str,
) -> GenerateResult:
    if result.debug is not None:
        result.debug.metrics.update(
            {
                "requested_backend": requested_backend,
                "resolved_backend": resolved_backend,
                "generation_backend": resolved_backend,
            }
        )
    return result


class PackageConfigError(ValueError):
    """Internal marker for package model-config compatibility failures."""


@dataclass(slots=True)
class Version:
    major: int
    minor: int
    patch: int


@dataclass(frozen=True, slots=True)
class _ValidatedRuntimeManifestSnapshot:
    """A detached manifest snapshot that passed every pre-tokenizer gate."""

    manifest: dict
    version: Version
    raw_model_config: dict
    fixture_mode: bool
    compatibility_warning: str | None


@dataclass(frozen=True, slots=True)
class _CanonicalRuntimeManifest:
    """The single package/config view consumed by product and parity."""

    manifest: dict
    version: Version
    model_config: Mapping[str, object]
    model_definition: ModelDefinition | None
    model_adapter: ModelAdapter | None
    resolved_mlx_config: ResolvedMLXRuntimeConfig
    compatibility_warning: str | None


def _freeze_plan_value(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_plan_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_plan_value(item) for item in value)
    return copy.deepcopy(value)


def _thaw_plan_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_plan_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_plan_value(item) for item in value]
    return copy.deepcopy(value)


@dataclass(frozen=True, slots=True)
class ResolvedBuildPlan:
    """Validated package/model contract before a device is selected."""

    package_model_config: Mapping[str, object]
    definition: ModelDefinition | None
    descriptor_key: str
    descriptor_digest: str
    definition_digest: str
    deployments: tuple[DeploymentTarget, ...]
    weight_artifact: WeightArtifactSpec
    build_digest: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "deployments", tuple(self.deployments))
        object.__setattr__(
            self,
            "package_model_config",
            _freeze_plan_value(self.package_model_config),
        )
        if self.definition is not None:
            if self.descriptor_key and self.definition.key != self.descriptor_key:
                raise ValueError("build plan descriptor/model definition mismatch")
            if self.descriptor_key and self.definition_digest:
                raise ValueError("build plan model definition scope is ambiguous")
            required = self.definition.deployments
            if self.deployments[: len(required)] != required:
                raise ValueError("build plan deployment contracts mismatch")
        elif self.definition_digest and self.descriptor_key:
            raise ValueError("build plan model definition scope is ambiguous")
        seen: set[tuple[str, str]] = set()
        for target in self.deployments:
            target.validate()
            identity = (target.backend, target.precision)
            if identity in seen:
                raise ValueError("build plan deployment target duplicated")
            seen.add(identity)
        if not isinstance(self.weight_artifact, WeightArtifactSpec):
            raise TypeError("build plan weight artifact invalid")

    def resolve_deployment(self, backend: str, precision: str) -> DeploymentTarget:
        for target in self.deployments:
            if target.backend == backend and target.precision == precision:
                return target
        definition_key = self.definition.key if self.definition is not None else "unknown"
        raise ValueError(
            f"unsupported_backend_precision: {definition_key}:{backend}:{precision}"
        )


# Fixed-quota strategies cuda_native runs without a cache or an execution
# budget, each on a device session scored on the GPU: MDLM over its full
# bidirectional canvas, BD3 over its absolute blocks (see _generate_cuda_native).
_CUDA_FIXED_QUOTA_STRATEGIES = frozenset({"fixed_quota", "bd3_fixed_quota"})

CUDA_FIXED_QUOTA_BUDGET_REFUSAL = (
    "cuda_fixed_quota_execution_budget_unsupported: cuda_native fixed_quota "
    "runs without a request memory budget; pass 0 to run with unbounded "
    "execution memory"
)


def refuse_unsupported_execution_budget(
    *, backend: str, decode_strategy: str, device_memory_budget_bytes: int
) -> None:
    """The one rule for a memory budget a plan cannot honour.

    The native cap covers runtime creation only, and the fixed-quota window
    path has no execution budget, so accepting a value would promise a limit
    nothing enforces. An executable plan applies this when it is built, before
    weights or a device runtime exist, so opening, deploying and diagnosing a
    package all refuse alike; generation applies it to a per-call budget.
    """
    if (
        backend == CUDA_NATIVE_BACKEND
        and decode_strategy in _CUDA_FIXED_QUOTA_STRATEGIES
        and device_memory_budget_bytes != 0
    ):
        raise ValueError(CUDA_FIXED_QUOTA_BUDGET_REFUSAL)


@dataclass(frozen=True, slots=True)
class LoadRequest:
    """One caller's device/backend choice, separate from package construction."""

    requested_backend: str
    backend: str
    precision: str
    deployment_profile: str
    device_ordinal: int = 0
    device_memory_budget_bytes: int = 0


@dataclass(frozen=True, slots=True)
class ExecutableModelPlan:
    """One immutable GraphIR-to-backend decision shared by all calls.

    A ResolvedBuildPlan can be reused for several deployment choices. A
    LoadRequest binds one choice, applies its policy, compiles GraphIR once and
    produces this plan before weights or K/V are acquired.
    """

    build: ResolvedBuildPlan
    load: LoadRequest
    deployment: DeploymentTarget | None
    model_config: Mapping[str, object]
    graph: GraphIR | None
    backend_plan: DenseTransformerPlan | None
    kernel_plan: KernelDispatchPlan | None
    adapter: ModelAdapter | None
    runtime_config: ResolvedMLXRuntimeConfig

    def __post_init__(self) -> None:
        refuse_unsupported_execution_budget(
            backend=self.load.backend,
            decode_strategy=self.runtime_config.decode_strategy,
            device_memory_budget_bytes=self.load.device_memory_budget_bytes,
        )
        object.__setattr__(
            self,
            "model_config",
            _freeze_plan_value(self.model_config),
        )
        if (
            self.build.definition is not None
            and self.adapter is not None
            and self.adapter.architecture_family
            != self.build.definition.runtime.key
        ):
            raise ValueError("resolved model definition/runtime adapter mismatch")
        if self.graph is None:
            if self.backend_plan is not None:
                raise ValueError("backend plan cannot exist without GraphIR")
        elif self.backend_plan is None:
            raise ValueError("GraphIR requires a compiled backend plan")
        elif self.backend_plan.graph is not self.graph:
            raise ValueError("compiled backend plan does not belong to GraphIR")
        if self.load.backend == MLX_LOCAL_BACKEND and self.kernel_plan is None:
            raise ValueError("mlx_local requires a kernel dispatch plan")
        if self.load.backend == CUDA_NATIVE_BACKEND and self.kernel_plan is not None:
            raise ValueError("cuda_native cannot consume an MLX kernel dispatch plan")
        if not self.build.weight_artifact.supports_backend(self.load.backend):
            raise ValueError(
                "backend does not support package weight format: "
                f"backend={self.load.backend!r} "
                f"format={self.build.weight_artifact.format!r}"
            )
        if self.deployment is not None:
            if self.deployment not in self.build.deployments:
                raise ValueError("executable deployment is absent from build plan")
            if (
                self.deployment.backend != self.load.backend
                or self.deployment.precision != self.load.precision
            ):
                raise ValueError("executable deployment/load request mismatch")
            contract = self.deployment.deployment
            if (
                contract is not None
                and contract.profile != self.load.deployment_profile
            ):
                raise ValueError("executable deployment profile mismatch")
            if contract is not None:
                contract.validate_attention_mask_kind(self.attention_mask_kind)

    @property
    def definition(self) -> ModelDefinition | None:
        return self.build.definition

    @property
    def descriptor_key(self) -> str:
        return self.build.descriptor_key

    @property
    def descriptor_digest(self) -> str:
        return self.build.descriptor_digest

    def matches(
        self,
        model_config: dict,
        graph: GraphIR | None,
        mask_token_id: int,
    ) -> bool:
        return (
            self.adapter is not None
            and self.adapter.mask_token_id == mask_token_id
            and self.graph is graph
            and _thaw_plan_value(self.model_config) == model_config
        )

    @property
    def attention_mask_kind(self) -> str:
        """Return executable mask structure from GraphIR when available."""

        if self.graph is None:
            return self.runtime_config.attention_mask_kind
        graph_kind = uniform_attention_mask_kind(self.graph)
        if not same_mask_kind_family(
            graph_kind,
            self.runtime_config.attention_mask_kind,
        ):
            raise ValueError("runtime config contradicts the loaded GraphIR mask")
        return graph_kind


@dataclass(frozen=True, slots=True)
class ChatRequestAdmission:
    """Exact prompt accounting against one immutable executable plan."""

    prompt_tokens: int
    output_tokens_reserved: int
    max_sequence_tokens: int


class ChatRequestCapacityError(ValueError):
    """A valid rendered chat request does not fit the executable sequence."""


@dataclass(frozen=True, slots=True)
class _DeploymentRuntimeLoad:
    """One profiled package load and the policy choices that produced it."""

    runtime: RuntimeHandle
    plan: ExecutableModelPlan

    @property
    def backend(self) -> str:
        return self.plan.load.backend

    @property
    def profile(self) -> str:
        return self.plan.load.deployment_profile


@dataclass(slots=True)
class _CudaNativeRuntimeState:
    """One resident compiled runtime and the immutable request that opened it."""

    runtime: object
    key: tuple[object, ...]

    def close(self) -> None:
        self.runtime.close()


@dataclass(slots=True)
class RuntimeHandle:
    @property
    def cooperative_cancellation_backends(self) -> frozenset[str]:
        """Qualified decode checkpoints for this immutable, cache-free plan.

        cuda_native qualifies only for its cache-free fixed-quota plans
        (fixed_quota over a bidirectional graph, bd3_fixed_quota over a
        d2f_hybrid graph, each on a device session without K/V): unwinding at
        a step boundary closes the session normally. Its block_threshold
        device sessions do not.
        """
        plan = self._resolved_model_plan
        if plan is None:
            return frozenset()
        cfg = plan.runtime_config
        if any((cfg.layer0_kv_cache_enabled, cfg.prefix_kv_cache, cfg.block_kv_cache,
                cfg.d2f_exact_kv_cache, cfg.error_fixing, cfg.canvas_cache,
                cfg.cross_request_cache)):
            return frozenset()
        fixed_quota_plans = {("fixed_quota", "bidirectional"), ("bd3_fixed_quota", "d2f_hybrid")}
        if (getattr(cfg, "decode_strategy", None),
                getattr(plan, "attention_mask_kind", None)) in fixed_quota_plans:
            return frozenset({"mlx_local", "cuda_native"})
        return frozenset({"mlx_local"})

    package_version: Version
    package_dir: Path | None = None
    backend_memory_owner: str = "numpy"
    forward_executor: str = "python_scaffold"
    runtime_impl: str = "python_scaffold"
    fixture_mode: bool = True
    merged_weights_ref: str = ""
    model_config: dict | None = None
    graph: GraphIR | None = None
    tokenizer: TokenizerSpec | None = None
    source_checkpoint_reference: str = ""
    source_checkpoint_sha: str = ""
    source_base_model_reference: str = ""
    # Recorded by the compiler so an artifact can state its own redistribution
    # terms. COPYRIGHT_AND_LICENSING.md's boundary is unenforceable if a package
    # cannot say what it was built from and under what licence. Empty for
    # packages built before the compiler wrote it.
    source_license: str = ""
    model_descriptor_key: str = ""
    model_descriptor_digest: str = ""
    model_definition_key: str = ""
    model_definition_digest: str = ""
    model_build_digest: str = ""
    model_precision: str = ""
    admitted_backend: str = ""
    decode_family: str = ""
    compatibility_warning: str | None = None
    _hf_reference_state: object | None = field(default=None, repr=False)
    _mlx_local_state: object | None = field(default=None, repr=False)
    _cuda_native_state: _CudaNativeRuntimeState | None = field(default=None, repr=False)
    _expanded_mlx_diffusion_policy_profile: str | None = field(default=None, repr=False)
    _resolved_build_plan: ResolvedBuildPlan | None = field(default=None, repr=False)
    _resolved_model_plan: ExecutableModelPlan | None = field(default=None, repr=False)
    _package_lease: PackageLease | None = field(default=None, repr=False)
    _execution_closed: bool = field(default=False, init=False, repr=False)

    def _model_plan(self, mask_token_id: int) -> ExecutableModelPlan:
        """Return the resolved plan; only legacy fixtures may rebuild it."""

        model_config = dict(self.model_config or {})
        current = self._resolved_model_plan
        if current is not None and current.matches(
            model_config,
            self.graph,
            mask_token_id,
        ):
            return current
        if current is not None and not self.fixture_mode:
            raise ValueError(
                "runtime model configuration is immutable after load; "
                "create a new Engine for a different model/deployment contract"
            )
        build = self._resolved_build_plan
        definition = (
            build.definition
            if build is not None
            else (current.definition if current is not None else None)
        )
        adapter, runtime_config = _resolve_package_mlx_runtime_config(
            model_config=model_config,
            mask_token_id=mask_token_id,
            definition=definition,
        )
        if build is None:
            build = ResolvedBuildPlan(
                package_model_config=model_config,
                definition=(current.definition if current is not None else None),
                descriptor_key=self.model_descriptor_key,
                descriptor_digest=self.model_descriptor_digest,
                definition_digest=self.model_definition_digest,
                deployments=(
                    current.build.deployments if current is not None else ()
                ),
                weight_artifact=(
                    current.build.weight_artifact
                    if current is not None
                    else WeightArtifactSpec.legacy_npz(
                        self.merged_weights_ref or "weights.npz"
                    )
                ),
                build_digest=(current.build.build_digest if current is not None else ""),
            )
            self._resolved_build_plan = build
        load = (
            current.load
            if current is not None
            else LoadRequest(
                requested_backend="auto",
                backend="unbound",
                precision=self.model_precision or "fp32",
                deployment_profile="baseline",
            )
        )
        resolved = ExecutableModelPlan(
            build=build,
            load=load,
            deployment=(current.deployment if current is not None else None),
            model_config=copy.deepcopy(model_config),
            graph=self.graph,
            backend_plan=(
                current.backend_plan
                if current is not None and current.graph is self.graph
                else (
                    compile_dense_transformer_plan(self.graph)
                    if self.graph is not None
                    else None
                )
            ),
            kernel_plan=(
                plan_kernel_dispatch(
                    model_config,
                    compute_dtype=runtime_config.compute_dtype,
                )
                if load.backend in {MLX_LOCAL_BACKEND, "unbound"}
                else None
            ),
            adapter=adapter,
            runtime_config=runtime_config,
        )
        # Validate the structural join before an execution backend can touch
        # weights or K/V. This property is also the one source used below.
        _ = resolved.attention_mask_kind
        self._resolved_model_plan = resolved
        return resolved

    def _require_planned_backend(self, backend: str) -> None:
        if self.fixture_mode:
            return
        plan = self._resolved_model_plan
        if plan is None or plan.load.backend == "unbound":
            return
        if plan.load.backend != backend:
            raise ValueError(
                "runtime backend differs from the executable model plan: "
                f"planned={plan.load.backend!r} requested={backend!r}"
            )

    def _require_execution_open(self) -> None:
        if self._execution_closed:
            raise RuntimeError(
                "runtime handle is closed; create a new handle to execute again"
            )

    def close(self) -> None:
        """Release the resident backend state this handle owns. Idempotent.

        There was no way to do this. Engine.close() dropped its reference and
        said so honestly -- "RuntimeHandle exposes no explicit release, so this
        cannot promise deterministic deallocation" -- and the service
        controller called `runtime.close()` defensively against a method that
        did not exist, so a foreground `serve` that returned still pinned every
        MLX buffer the model had allocated.

        The first attempt is terminal for execution even if backend cleanup
        fails.  Metadata stays readable, while backend state and the package
        lease stay reachable only so a later ``close()`` can retry cleanup.
        The lease is released only after every backend owner closes.
        """
        self._execution_closed = True
        cleanup_errors: list[BaseException] = []
        for attribute in (
            "_mlx_local_state",
            "_cuda_native_state",
            "_hf_reference_state",
        ):
            state = getattr(self, attribute, None)
            if state is None:
                continue
            state_close = getattr(state, "close", None)
            try:
                if callable(state_close):
                    state_close()
            except BaseException as error:
                # Keep the failed owner for retry, but continue releasing
                # independent owners. One broken backend must not strand all
                # other resources attached to the same handle.
                cleanup_errors.append(error)
            else:
                setattr(self, attribute, None)
        if cleanup_errors:
            primary = cleanup_errors[0]
            if hasattr(primary, "add_note"):
                for error in cleanup_errors[1:]:
                    primary.add_note(
                        "another runtime owner also failed to close: "
                        f"{type(error).__name__}: {error}"
                    )
            raise primary
        lease = self._package_lease
        if lease is not None:
            lease.close()
            self._package_lease = None

    def __del__(self) -> None:
        # Explicit close is the checked lifecycle.  This fallback prevents a
        # simply abandoned low-level handle from retaining an OS descriptor
        # until process exit; cleanup failure remains suppressed and therefore
        # keeps the still-open descriptor rather than claiming success.
        with contextlib.suppress(Exception):
            self.close()

    def _validate_mlx_policy_profile_integrity(self) -> None:
        expanded_profile = self._expanded_mlx_diffusion_policy_profile
        if expanded_profile is None:
            return
        current_profile = (self.model_config or {}).get(
            "mlx_diffusion_policy_profile",
            "baseline",
        )
        if current_profile != expanded_profile:
            raise ValueError(
                "mlx_diffusion_policy_profile changed after load; reload the "
                "package to expand a different policy profile"
            )

    def _validate_mlx_policy_profile_task(self, task: str) -> None:
        self._validate_mlx_policy_profile_integrity()
        raw_profile = (self.model_config or {}).get(
            "mlx_diffusion_policy_profile",
            "baseline",
        )
        profile_name = "baseline" if raw_profile is None else str(raw_profile)
        validate_policy_profile_task(profile_name, task)

    def admit_chat_messages(
        self,
        chat_messages: list[dict[str, str]],
        *,
        max_tokens: int | None,
    ) -> ChatRequestAdmission:
        """Render and validate one chat request without executing a forward.

        Conversation context selection and generation call this same boundary;
        neither may estimate tokens or reinterpret block geometry.
        """

        self._require_execution_open()
        if max_tokens is None:
            # author:Opus -- a reply without a limit grows chunk by chunk; the
            # request must leave room for its first chunk.
            max_tokens = AUTO_REPLY_CHUNK_TOKENS
        validate_generation_length(max_tokens, field_name="max_tokens")
        if self.tokenizer is None:
            raise ValueError("runtime has no tokenizer")
        model_config = self.model_config or {}
        configured_mask = valid_token_id(model_config.get("mask_token_id"))
        mask_token_id = (
            int(configured_mask)
            if configured_mask is not None
            else self.tokenizer.require_control_token("<|mask|>")
        )
        plan = self._model_plan(mask_token_id)
        adapter = plan.adapter
        if adapter is None:
            raise ValueError("chat requires a registered model definition")
        prompt_token_ids, _ = _mlx_prompt_token_ids(
            self.tokenizer,
            "",
            chat_messages=chat_messages,
            chat_template_kwargs=dict(adapter.chat_template_kwargs),
        )
        if max_tokens:
            _validate_prompt_block_alignment(prompt_token_ids, plan.runtime_config)
        maximum = plan.model_config.get("max_position_embeddings")
        if type(maximum) is not int or maximum <= 0:
            raise ValueError(
                "chat requires a positive max_position_embeddings in the "
                "executable model plan"
            )
        if len(prompt_token_ids) + max_tokens > maximum:
            raise ChatRequestCapacityError(
                "chat request exceeds the executable sequence limit: "
                f"prompt_tokens={len(prompt_token_ids)} max_tokens={max_tokens} "
                f"max_position_embeddings={maximum}"
            )
        return ChatRequestAdmission(
            prompt_tokens=len(prompt_token_ids),
            output_tokens_reserved=max_tokens,
            max_sequence_tokens=maximum,
        )

    def _sync_mlx_local_engine(self) -> None:
        """Mirror a legacy fixture's config/graph into its persistent engine.

        The engine is constructed once and kept resident (weights stay loaded),
        but everything it derives from ``model_config``/``graph`` — the forward
        GraphIR, linear-quant spec and kernel table — resolves from its own
        fields. Non-fixture plans reject post-load mutation in ``_model_plan``;
        this synchronization remains only for mutable compatibility fixtures.
        Weight residency is preserved for those fixtures.
        """
        engine = self._mlx_local_state
        if engine is None:
            return
        live_config = dict(self.model_config or {})
        config_changed = getattr(engine, "model_config", None) != live_config
        if config_changed:
            engine.model_config = live_config
        current = self._resolved_model_plan
        current_backend_plan = (
            current.backend_plan
            if current is not None and current.graph is self.graph
            else None
        )
        current_kernel_plan = current.kernel_plan if current is not None else None
        if getattr(engine, "graph", None) is not self.graph:
            engine.graph = self.graph
            engine._graph = self.graph
            engine.backend_plan = current_backend_plan
        elif config_changed and getattr(engine, "graph", None) is None:
            # No handle-level graph: the engine's cached forward GraphIR was
            # built lazily from the previous config — drop it so the next
            # forward rebuilds from the new config.
            engine._graph = None
            engine.backend_plan = None
        elif getattr(engine, "backend_plan", None) is not current_backend_plan:
            engine.backend_plan = current_backend_plan
        if getattr(engine, "kernel_plan", None) != current_kernel_plan:
            engine.kernel_plan = current_kernel_plan
            if not getattr(engine, "_kernel_table_injected", False):
                engine.kernel_table = None

    def make_executor(self) -> FakeExecutor | DreamTransformerExecutor:
        self._require_execution_open()
        if self.tokenizer is None:
            raise ValueError("runtime has no tokenizer")
        if self.fixture_mode:
            return FakeExecutor(vocab_size=self.tokenizer.vocab_size)
        if self.package_dir is None or not self.merged_weights_ref:
            raise ValueError("numpy_local requires package weights for a real package")

        model_config = self.model_config or {}
        configured_mask_token_id = valid_token_id(model_config.get("mask_token_id"))
        mask_token_id = (
            int(configured_mask_token_id)
            if configured_mask_token_id is not None
            else self.tokenizer.require_control_token("<|mask|>")
        )
        plan = self._model_plan(mask_token_id)
        adapter = plan.adapter
        if adapter is None:
            raise ValueError("numpy_local requires a registered model definition")
        backend_plan = plan.backend_plan
        uses_qk_norm = (
            backend_plan.apply_qk_norm
            if backend_plan is not None
            else adapter.uses_qk_norm
        )
        uses_shifted_logits = (
            backend_plan.logits_convention == "dream_ar_shift"
            if backend_plan is not None
            else adapter.uses_shifted_logits
        )
        if (
            adapter.architecture_family != "qwen2_dream"
            or uses_qk_norm
            or not uses_shifted_logits
        ):
            architecture = str(
                model_config.get("architecture")
                or model_config.get("architecture_family")
                or "<missing>"
            )
            raise ValueError(
                f"numpy_local does not support real architecture {architecture}"
            )
        return DreamTransformerExecutor(
            package_dir=self.package_dir,
            merged_weights_ref=self.merged_weights_ref,
            model_config=model_config,
        )

    def make_executor_input(
        self,
        *,
        prompt_ids: list[int],
        block_token_ids: list[int],
        block_position: int,
        cache_state_ref: str | None = None,
        attention_mask_kind: str | None = None,
    ) -> ExecutorInput:
        return ExecutorInput(
            prompt_token_ids=prompt_ids,
            block_token_ids=block_token_ids,
            block_position=block_position,
            generation_cursor=block_position,
            cache_state_ref=cache_state_ref,
            attention_mask_kind=attention_mask_kind
            or str(
                (self.model_config or {}).get("attention_mask_kind", "bidirectional")
            ),
        )

    def _cuda_runtime(
        self,
        *,
        precision: str,
        device_ordinal: int,
        creation_memory_budget_bytes: int,
        exact_kv: bool,
    ):
        """Open or reuse the one compiled CUDA runtime owned by this handle."""
        self._require_execution_open()
        if self.fixture_mode:
            raise ValueError("cuda_native requires a real execution package")
        if self.package_dir is None:
            raise ValueError("cuda_native requires an execution package directory")

        from diffusorrt_runtime import Runtime

        from .backends.device_runtime import (
            cuda_engine_config,
            cuda_precision_descriptor,
        )

        key = (
            precision,
            device_ordinal,
            creation_memory_budget_bytes,
            exact_kv,
        )
        state = self._cuda_native_state
        if state is not None and state.key != key:
            # A resident runtime owns weights prepared for exactly one precision
            # and engine contract. Close it before opening a different one; if
            # close fails, retain it so the caller can retry cleanup.
            state.close()
            self._cuda_native_state = None
            state = None
        if state is None:
            runtime = Runtime(
                self.package_dir,
                device="cuda",
                engine_config=cuda_engine_config(precision, exact_kv=exact_kv),
                precision=cuda_precision_descriptor(precision),
                device_ordinal=device_ordinal,
                creation_memory_budget_bytes=creation_memory_budget_bytes,
            )
            state = _CudaNativeRuntimeState(runtime=runtime, key=key)
            self._cuda_native_state = state
        return state.runtime

    def _generate_cuda_native(
        self,
        prompt: str,
        *,
        chat_messages: list[dict] | None,
        max_tokens: int,
        return_debug: bool,
        precision: str,
        device_ordinal: int,
        device_memory_budget_bytes: int,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> GenerateResult:
        """Run the shared decode policy over the compiled CUDA device session.

        Above ``temperature`` 0 the fixed-quota policies sample each position's
        token; an unseeded request draws a seed, which the debug bundle reports.
        """
        started_at = time.perf_counter()
        if self.tokenizer is None:
            raise ValueError("cuda_native requires a package tokenizer")
        model_config = self.model_config or {}
        mask_token_id = _resolve_mlx_mask_token_id(model_config, self.tokenizer)
        plan = self._model_plan(mask_token_id)
        adapter = plan.adapter
        if adapter is None:
            raise ValueError("cuda_native requires a registered model definition")
        runtime_config = plan.runtime_config
        decode_strategy = runtime_config.decode_strategy
        if decode_strategy not in {"block_threshold", *_CUDA_FIXED_QUOTA_STRATEGIES}:
            raise ValueError(
                "cuda_native currently implements only the shared "
                "block_threshold, fixed_quota and bd3_fixed_quota decode strategies"
            )
        attention_kind = plan.attention_mask_kind
        if attention_kind not in {"bidirectional", "d2f_hybrid"}:
            raise ValueError(
                "cuda_native requires a bidirectional or d2f_hybrid GraphIR"
            )
        if decode_strategy == "fixed_quota" and attention_kind != "bidirectional":
            raise ValueError("cuda_native fixed_quota requires a bidirectional GraphIR")
        if decode_strategy == "bd3_fixed_quota" and attention_kind != "d2f_hybrid":
            raise ValueError("cuda_native bd3_fixed_quota requires a d2f_hybrid GraphIR")
        if temperature != 0 and decode_strategy not in _CUDA_FIXED_QUOTA_STRATEGIES:
            raise ValueError(
                f"this model decodes with {decode_strategy}, which is greedy; temperature "
                "sampling belongs to the fixed-quota models (MDLM, BD3)"
            )
        if temperature == 0:
            seed = None  # greedy: nothing to key
        elif seed is None:
            import secrets

            seed = secrets.randbits(32)
        # A direct caller passes its own budget per call, so the plan's check at
        # load time is not enough here. Refused before any device allocation.
        refuse_unsupported_execution_budget(
            backend=CUDA_NATIVE_BACKEND,
            decode_strategy=decode_strategy,
            device_memory_budget_bytes=device_memory_budget_bytes,
        )
        # Every strategy runs on a device session: the device scores each
        # frontier row, so no logits row crosses to the host. BD3's absolute
        # blocks reach the session as its decode geometry.
        # The I8 product join makes the accepted D2F path consume the one native
        # K/V owner. Bidirectional models such as LLaDA cannot reuse causal
        # frozen-prefix K/V and remain full-canvas by construction. The
        # fixed-quota plans hold no cache: BD3 recomputes its visible prefix
        # every round.
        exact_kv = attention_kind == "d2f_hybrid" and decode_strategy == "block_threshold"
        prompt_token_ids, _ = _mlx_prompt_token_ids(
            self.tokenizer,
            prompt,
            chat_messages=chat_messages,
            chat_template_kwargs=adapter.chat_template_kwargs,
        )
        if max_tokens:
            _validate_prompt_block_alignment(prompt_token_ids, runtime_config)
        # BD3's session starts its blocks at 0, so its prompt is part of the
        # device canvas. A prompt carrying the mask token (a user can type it)
        # keeps the host window loop, which never treats prompt rows as
        # decodable. Since review F1 the native session no longer refuses MASK
        # in a no-cache frozen prefix, so this fallback could move to the device
        # once that path has its own route check.
        window_loop = (
            decode_strategy == "bd3_fixed_quota" and mask_token_id in prompt_token_ids
        )
        from .backends.device_runtime import (
            generate_device_tokens,
            generate_window_tokens,
        )

        runtime = self._cuda_runtime(
            precision=precision,
            device_ordinal=device_ordinal,
            creation_memory_budget_bytes=device_memory_budget_bytes,
            exact_kv=exact_kv,
        )
        eos_token_id = int(self.tokenizer.eos_token_id)
        effective_steps = _fixed_quota_request_steps(runtime_config, max_tokens)
        if window_loop:
            # A forward-only session: no policy runs on the device. It still
            # names the d2f_hybrid graph's mask kind, the one decode field a
            # forward reads; the runtime refuses a disagreeing session.
            with runtime.session(session_config={"attention_mask_kind": 1}) as session:
                generated = generate_window_tokens(
                    session,
                    prompt_token_ids,
                    max_tokens=max_tokens,
                    mask_token_id=mask_token_id,
                    eos_token_id=eos_token_id,
                    steps=effective_steps,
                    block_size=runtime_config.block_size,
                    decode_strategy=decode_strategy,
                    requires_prompt_block_alignment=(
                        runtime_config.requires_prompt_block_alignment
                    ),
                    temperature=float(temperature),
                    seed=seed or 0,
                )
            device_metrics = {}
        else:
            # BD3's session starts its blocks at 0, so its mutable span, which
            # the runtime checks against a d2f canvas, is the whole canvas. The
            # policy still proposes and commits only reply positions.
            mutable_span = max_tokens + (
                len(prompt_token_ids) if decode_strategy == "bd3_fixed_quota" else 0
            )
            from .abi.enums import DECODE_STRATEGY_BY_NAME, CommitPolicy

            # The policy that runs, by its own name: the runtime admits each one
            # only on the graph it can run on, and lets a fixed-quota policy
            # propose MASK, which its host rule ranks like any token.
            session_config = {
                "decode_strategy": int(DECODE_STRATEGY_BY_NAME[decode_strategy]),
                "commit_policy": int(
                    CommitPolicy.FIXED_QUOTA
                    if decode_strategy in _CUDA_FIXED_QUOTA_STRATEGIES
                    else CommitPolicy.THRESHOLD
                ),
                "attention_mask_kind": 1 if attention_kind == "d2f_hybrid" else 0,
                "block_size": runtime_config.block_size,
                "steps": effective_steps,
                "max_tokens": mutable_span,
            }
            with runtime.session(session_config=session_config) as session:
                generated = generate_device_tokens(
                    session,
                    prompt_token_ids,
                    max_tokens=max_tokens,
                    mask_token_id=mask_token_id,
                    eos_token_id=eos_token_id,
                    steps=effective_steps,
                    block_size=runtime_config.block_size,
                    confidence_threshold=runtime_config.confidence_threshold,
                    exact_kv=exact_kv,
                    memory_budget_bytes=device_memory_budget_bytes,
                    attention_mask_kind=attention_kind,
                    decode_strategy=decode_strategy,
                    requires_prompt_block_alignment=(
                        runtime_config.requires_prompt_block_alignment
                    ),
                    temperature=float(temperature),
                    seed=seed or 0,
                )
                device_metrics = session.device_metrics()
        if decode_strategy in _CUDA_FIXED_QUOTA_STRATEGIES:
            # Refused a budget above: execution memory is unbounded.
            device_metrics = {**device_metrics, "cuda_execution_memory_bounded": False}
        text = self.tokenizer.decode(generated.visible_generated_token_ids)
        debug = None
        if return_debug:
            debug = DebugBundle(
                metrics={
                    "session_id": str(uuid.uuid4()),
                    "generation_backend": CUDA_NATIVE_BACKEND,
                    "backend_provenance": "compiled_cuda_execution_package",
                    "cuda_precision": precision,
                    "cuda_device_ordinal": device_ordinal,
                    "cuda_exact_kv": exact_kv,
                    "total_latency_ms": round(
                        (time.perf_counter() - started_at) * 1000.0, 3
                    ),
                    "block_iteration_count": generated.iteration_count,
                    "cuda_forward_count": generated.forward_count,
                    "cuda_committed_token_count": generated.committed_token_count,
                    "cuda_remaining_mask_count": generated.remaining_mask_count,
                    "decode_loop_kind": generated.decode_loop_kind,
                    # What actually ran, for comparing backends token by token.
                    # Only in a requested debug bundle, never in an ordinary result.
                    "decode_strategy": decode_strategy,
                    "effective_steps": effective_steps,
                    "sampling_temperature": float(temperature),
                    "sampling_seed": seed,
                    "block_size": runtime_config.block_size,
                    "mask_token_id": mask_token_id,
                    "eos_token_id": eos_token_id,
                    "prompt_token_ids": [int(token) for token in prompt_token_ids],
                    "visible_generated_token_ids": [
                        int(token) for token in generated.visible_generated_token_ids
                    ],
                    **device_metrics,
                }
            )
        return GenerateResult(
            text=text,
            finish_reason=generated.finish_reason,
            debug=debug,
            visible_output_tokens=len(generated.visible_generated_token_ids),
        )

    def _generate_cuda_infill(
        self,
        prefix: str,
        suffix: str,
        *,
        max_fill_tokens: int,
        return_debug: bool,
        precision: str,
        device_ordinal: int,
        device_memory_budget_bytes: int,
    ) -> InfillResult:
        """Fill one span using full-canvas CUDA forward and the shared policy."""
        started_at = time.perf_counter()
        if self.tokenizer is None:
            raise ValueError("cuda_native requires a package tokenizer")
        model_config = self.model_config or {}
        mask_token_id = _resolve_mlx_mask_token_id(model_config, self.tokenizer)
        plan = self._model_plan(mask_token_id)
        runtime_config = plan.runtime_config
        if plan.attention_mask_kind not in {"bidirectional", "full"}:
            raise ValueError("cuda_native infill requires bidirectional GraphIR")
        if runtime_config.decode_strategy != "block_threshold":
            raise ValueError(
                "cuda_native infill currently implements only the shared "
                "block_threshold decode strategy"
            )
        from .backends.device_runtime import infill_device_tokens

        prefix_ids = [int(token) for token in self.tokenizer.encode_raw(prefix)]
        suffix_ids = [int(token) for token in self.tokenizer.encode_raw(suffix)]
        runtime = self._cuda_runtime(
            precision=precision,
            device_ordinal=device_ordinal,
            creation_memory_budget_bytes=device_memory_budget_bytes,
            exact_kv=False,
        )
        session_config = {
            "decode_strategy": 0,
            "commit_policy": 2,
            "attention_mask_kind": 0,
            "block_size": runtime_config.block_size,
            "steps": runtime_config.steps,
            "max_tokens": max_fill_tokens,
        }
        with runtime.session(session_config=session_config) as session:
            infilled = infill_device_tokens(
                session,
                prefix_ids,
                suffix_ids,
                max_fill_tokens=max_fill_tokens,
                mask_token_id=mask_token_id,
                eos_token_id=int(self.tokenizer.eos_token_id),
                steps=runtime_config.steps,
                block_size=runtime_config.block_size,
                confidence_threshold=runtime_config.confidence_threshold,
                memory_budget_bytes=device_memory_budget_bytes,
            )
            device_metrics = session.device_metrics()
        filled_text = self.tokenizer.decode(infilled.fill_token_ids, strip=False)
        debug = None
        if return_debug:
            debug = DebugBundle(
                metrics={
                    "session_id": str(uuid.uuid4()),
                    "generation_backend": CUDA_NATIVE_BACKEND,
                    "backend_provenance": "compiled_cuda_execution_package",
                    "cuda_precision": precision,
                    "cuda_device_ordinal": device_ordinal,
                    "cuda_exact_kv": False,
                    "total_latency_ms": round(
                        (time.perf_counter() - started_at) * 1000.0, 3
                    ),
                    "cuda_infill_prefix_token_count": len(prefix_ids),
                    "cuda_infill_suffix_token_count": len(suffix_ids),
                    "cuda_infill_fill_token_count": len(infilled.fill_token_ids),
                    "cuda_forward_count": infilled.forward_count,
                    "cuda_committed_token_count": infilled.committed_token_count,
                    "cuda_remaining_mask_count": infilled.remaining_mask_count,
                    "decode_loop_kind": infilled.decode_loop_kind,
                    **device_metrics,
                }
            )
        return InfillResult(
            filled_text=filled_text,
            full_text=prefix + filled_text + suffix,
            finish_reason=infilled.finish_reason,
            debug=debug,
        )

    def generate(
        self,
        prompt: str = "",
        *,
        chat_messages: list[dict] | None = None,
        max_tokens: int | None = 32,
        return_debug: bool = False,
        backend: str = "auto",
        snapshot_save_path: str | Path | None = None,
        precision: str | None = None,
        device_ordinal: int = 0,
        device_memory_budget_bytes: int = 0,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> GenerateResult:
        self._require_execution_open()
        if max_tokens is not None:
            validate_generation_length(max_tokens, field_name="max_tokens")
        validate_request_sampling(temperature=temperature, seed=seed)
        resolved_snapshot_path = _resolve_caller_snapshot_path(snapshot_save_path)
        requested_backend = backend
        resolved_backend = _resolve_generation_backend(
            requested_backend,
            fixture_mode=self.fixture_mode,
            backend_memory_owner=self.backend_memory_owner,
            forward_executor=self.forward_executor,
        )
        self._require_planned_backend(resolved_backend)
        if resolved_snapshot_path is not None and resolved_backend != "mlx_local":
            raise ValueError("snapshot_save_path requires backend='mlx_local'")
        if max_tokens is None and resolved_backend != "mlx_local":
            # author:Opus -- no limit means "until the model ends" on the MLX path,
            # which decodes in chunks. Other backends keep the reply size they had.
            max_tokens = AUTO_REPLY_CHUNK_TOKENS
        if resolved_backend != CUDA_NATIVE_BACKEND and temperature != 0:
            raise NotImplementedError(
                f"temperature samples on cuda_native only; {resolved_backend} decodes greedily"
            )
        if resolved_backend == CUDA_NATIVE_BACKEND:
            if precision is None:
                raise ValueError("cuda_native requires an explicit precision")
            return _record_generation_backend_provenance(
                self._generate_cuda_native(
                    prompt,
                    chat_messages=chat_messages,
                    max_tokens=max_tokens,
                    return_debug=return_debug,
                    precision=precision,
                    device_ordinal=device_ordinal,
                    device_memory_budget_bytes=device_memory_budget_bytes,
                    temperature=temperature,
                    seed=seed,
                ),
                requested_backend=requested_backend,
                resolved_backend=resolved_backend,
            )
        if resolved_backend == "hf_dream_reference":
            return _record_generation_backend_provenance(
                self._generate_hf_dream_reference(
                    prompt,
                    chat_messages=chat_messages,
                    max_tokens=max_tokens,
                    return_debug=return_debug,
                ),
                requested_backend=requested_backend,
                resolved_backend=resolved_backend,
            )
        if resolved_backend == "mlx_local":
            return _record_generation_backend_provenance(
                self._generate_mlx_local(
                    prompt,
                    chat_messages=chat_messages,
                    max_tokens=max_tokens,
                    return_debug=return_debug,
                    snapshot_save_path=resolved_snapshot_path,
                ),
                requested_backend=requested_backend,
                resolved_backend=resolved_backend,
            )
        if self.tokenizer is None:
            raise ValueError("runtime has no tokenizer")

        started_at = time.perf_counter()
        session_id = str(uuid.uuid4())
        model_config = self.model_config or {}
        # author:GLM — manifest priority: dict.get() eagerly evaluates its
        # default argument, so require_control_token runs even when the
        # manifest already carries mask_token_id. Use an explicit check.
        if valid_token_id(model_config.get("mask_token_id")) is not None:
            model_mask_token_id = int(model_config["mask_token_id"])
        else:
            model_mask_token_id = self.tokenizer.require_control_token("<|mask|>")
        plan = self._model_plan(model_mask_token_id)
        if chat_messages is None:
            # Preserve the legacy NumPy single-prompt contract exactly. Chat
            # templating is needed only for the explicit multi-message API.
            prompt_tokens = self.tokenizer.encode(prompt)
        else:
            # NumPy's v1.1 causal compatibility path needs only tokenizer
            # policy; its executor capability is checked separately below.
            adapter = plan.adapter
            if adapter is None:
                raise ValueError("numpy_local requires a registered model definition")
            prompt_tokens, _prompt_uses_chat_template = _mlx_prompt_token_ids(
                self.tokenizer,
                prompt,
                chat_messages=chat_messages,
                chat_template_kwargs=adapter.chat_template_kwargs,
            )
        executor = self.make_executor()
        if isinstance(executor, DreamTransformerExecutor):
            decode_outcome = run_dream_masked_diffusion_decode(
                prompt_tokens=prompt_tokens,
                executor=executor,
                max_tokens=max_tokens,
                mask_token_id=model_mask_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                diffusion_steps=int(model_config.get("diffusion_steps", 4)),
                eps=float(model_config.get("diffusion_eps", 1e-3)),
                attention_mask_kind=str(
                    model_config.get("attention_mask_kind", "bidirectional")
                ),
                cache_refresh_interval=int(
                    model_config.get("diffusion_cache_refresh_interval", 0)
                ),
                diffusion_alg=str(model_config.get("diffusion_alg", "entropy")),
            )
        else:
            decode_outcome = run_block_diffusion_decode(
                prompt_tokens=prompt_tokens,
                executor=executor,
                max_tokens=max_tokens,
                mask_token_id=model_mask_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                block_size=int(model_config.get("block_size", 2)),
                step_budget=int(model_config.get("step_budget", 4)),
                attention_mask_kind=str(
                    model_config.get("attention_mask_kind", "bidirectional")
                ),
            )
        # The smoke floor must not exceed what was actually requested: with the
        # benchmark fallback max_tokens=4 a hardcoded ">= 5" made forward_real
        # impossible even for a genuine forward. Scale the floor to max_tokens.
        smoke_len = min(5, max_tokens)
        visible_preview = decode_outcome.visible_generated_token_ids[:smoke_len]
        smoke_text = self.tokenizer.decode(visible_preview)
        smoke_ok = (
            len(visible_preview) >= smoke_len
            and len(set(visible_preview)) >= min(3, smoke_len)
            and sum(1 for char in smoke_text if char.isalpha()) >= 2
        )
        tokenizer_real = self.tokenizer.kind == "transformers_auto_tokenizer"
        forward_real = (
            tokenizer_real
            and decode_outcome.executor_source in FULL_FORWARD_EXECUTOR_SOURCES
            and smoke_ok
            and len(decode_outcome.generated_tokens) >= 1
        )
        commit_policy = "runtime_argmax_from_fake_logits"
        baseline_trigger = "entropy_or_confidence"
        if decode_outcome.executor_source == PROBE_EXECUTOR_SOURCE:
            commit_policy = "runtime_argmax_from_probe_logits"
        elif decode_outcome.decode_loop_kind == "dream_masked_diffusion_entropy":
            commit_policy = "dream_entropy_masked_diffusion"
        elif decode_outcome.decode_loop_kind == "dream_masked_diffusion_origin":
            commit_policy = "dream_origin_masked_diffusion"
            baseline_trigger = "origin_random_transfer"
        elif forward_real:
            commit_policy = "runtime_argmax_from_real_logits"
        baseline_status = {
            "tokenizer_real": tokenizer_real,
            "forward_real": forward_real,
            "entropy_real": forward_real and decode_outcome.entropy_used,
            "refresh_real": bool(decode_outcome.refresh_real),
        }
        text = self.tokenizer.decode(decode_outcome.visible_generated_token_ids)
        total_latency_ms = (time.perf_counter() - started_at) * 1000.0

        debug = None
        if return_debug:
            debug = DebugBundle(
                metrics={
                    "session_id": session_id,
                    "block_count": decode_outcome.block_count,
                    "block_iteration_count": decode_outcome.iteration_count,
                    "ttft_ms": round(decode_outcome.first_block_latency_ms, 3),
                    "total_latency_ms": round(total_latency_ms, 3),
                    "decoded_safe_commit_count": decode_outcome.safe_commit_count,
                    "decoded_hold_count": decode_outcome.hold_count,
                    "decoded_committed_count": decode_outcome.committed_count,
                    "peak_memory_bytes": 0,
                    "forward_executor": self.forward_executor,
                    "runtime_impl": self.runtime_impl,
                    "generation_backend": "numpy_local",
                    "backend_provenance": "local_execution_package",
                    "local_executor_verified": bool(baseline_status["forward_real"]),
                    "commit_policy": commit_policy,
                    "baseline_trigger": baseline_trigger,
                    "cache_refresh_count": decode_outcome.cache_refresh_count,
                    "baseline_includes_cache_refresh": True,
                    "decode_loop_kind": decode_outcome.decode_loop_kind,
                    "visible_generated_token_ids": decode_outcome.visible_generated_token_ids,
                    "baseline_status": baseline_status,
                    "row_verified": all(baseline_status.values()),
                }
            )

        return _record_generation_backend_provenance(
            # decode_outcome.finish_reason, not a constant. "stop" was
            # returned for every exit, so a caller could not tell a completed
            # answer from one truncated at max_tokens.
            GenerateResult(
                text=text,
                finish_reason=decode_outcome.finish_reason,
                debug=debug,
            ),
            requested_backend=requested_backend,
            resolved_backend=resolved_backend,
        )

    def _generate_hf_dream_reference(
        self,
        prompt: str,
        *,
        chat_messages: list[dict] | None,
        max_tokens: int,
        return_debug: bool,
    ) -> GenerateResult:
        from .backends import hf_dream

        base_model_ref, adapter_ref = self._hf_reference_model_refs()
        started_at = time.perf_counter()
        if self._hf_reference_state is None:
            self._hf_reference_state = hf_dream.load_reference_state(
                base_model_ref=base_model_ref,
                adapter_ref=adapter_ref,
            )
        reference = hf_dream.generate_reference(
            state=self._hf_reference_state,
            prompt=prompt,
            chat_messages=chat_messages,
            max_tokens=max_tokens,
        )
        total_latency_ms = (time.perf_counter() - started_at) * 1000.0
        baseline_status = {
            "tokenizer_real": False,
            "forward_real": False,
            "entropy_real": False,
            "refresh_real": False,
        }
        debug = None
        if return_debug:
            debug = DebugBundle(
                metrics={
                    "session_id": str(uuid.uuid4()),
                    "generation_backend": "hf_dream_reference",
                    "backend_provenance": "upstream_hf_dream",
                    "local_executor_verified": False,
                    "baseline_status": baseline_status,
                    "row_verified": False,
                    "total_latency_ms": round(total_latency_ms, 3),
                    **reference.metrics,
                }
            )
        return GenerateResult(text=reference.text, finish_reason="stop", debug=debug)

    def _generate_mlx_local(
        self,
        prompt: str,
        *,
        chat_messages: list[dict] | None = None,
        max_tokens: int | None,
        return_debug: bool,
        snapshot_save_path: str | None = None,
    ) -> GenerateResult:
        started_at = time.perf_counter()
        self._validate_mlx_policy_profile_task("chat")
        from .backends import mlx_cache
        from .backends import mlx_dream

        model_config = self.model_config or {}
        # FAIL-CLOSED routing up front: reject unknown architectures BEFORE any
        # engine construction or forward, so foreign weights never run Dream
        # semantics. ``mask_token_id`` is resolved from model_config first (only
        # falling back to the tokenizer when present) so this works pre-load.
        mask_token_id = _resolve_mlx_mask_token_id(model_config, self.tokenizer)
        plan = self._model_plan(mask_token_id)
        adapter = plan.adapter
        if adapter is None:
            raise ValueError("mlx_local requires a registered model definition")
        runtime_config = plan.runtime_config
        if runtime_config.attention_mask_kind == "causal":
            raise ValueError(
                "attention_mask_kind='causal' is supported only by the "
                "legacy numpy_local compatibility path"
            )
        if snapshot_save_path is not None and (
            self.fixture_mode or self.tokenizer is None
        ):
            raise ValueError("snapshot_save_path requires a verified mlx_local runtime")

        if self._mlx_local_state is None:
            self._mlx_local_state = mlx_dream.MLXDreamEngine(
                package_dir=self.package_dir,
                merged_weights_ref=self.merged_weights_ref,
                model_config=dict(self.model_config or {}),
                weight_artifact=plan.build.weight_artifact,
                graph=self.graph,
                backend_plan=plan.backend_plan,
                kernel_plan=plan.kernel_plan,
            )
        self._sync_mlx_local_engine()
        if self.fixture_mode or self.tokenizer is None:
            text = self._mlx_local_state.generate_unverified(
                prompt=prompt,
                max_tokens=AUTO_REPLY_CHUNK_TOKENS if max_tokens is None else max_tokens,
            )
            baseline_status = {
                "tokenizer_real": False,
                "forward_real": False,
                "entropy_real": False,
                "refresh_real": False,
            }
            debug = None
            if return_debug:
                debug = DebugBundle(
                    metrics={
                        "session_id": str(uuid.uuid4()),
                        "generation_backend": "mlx_local",
                        "backend_provenance": "local_mlx_execution_package",
                        "local_executor_verified": False,
                        "baseline_status": baseline_status,
                        "row_verified": False,
                        "total_latency_ms": round(
                            (time.perf_counter() - started_at) * 1000.0, 3
                        ),
                        "mlx_local_state": "unverified_skeleton",
                        "mlx_forward_count": 0,
                        "mlx_committed_token_count": 0,
                        "mlx_output_token_count": 0,
                        "mlx_committed_tokens_per_forward": 0.0,
                        "mlx_s_over_t": None,
                        "mlx_diffusion_policy_profile": runtime_config.policy_profile,
                    }
                )
            return GenerateResult(
                text=text, finish_reason="unverified_skeleton", debug=debug
            )

        # ModelAdapter boundary (selected above, fail-closed): Dream (Qwen2-based)
        # is the shifted AR-style default; Qwen3-MDLM (A2DQwen3) is unshifted with
        # enable_thinking=False. For Dream uses_shifted_logits stays True and
        # chat_template_kwargs stays {}, so behavior is byte-identical.
        #
        # SINGLE chat renderer: when ``chat_messages`` is provided (the chat CLI
        # path), the template is applied to that full message list EXACTLY ONCE.
        # Otherwise ``prompt`` is templated as a lone user turn (unchanged).
        prompt_token_ids, prompt_uses_chat_template = _mlx_prompt_token_ids(
            self.tokenizer,
            prompt,
            chat_messages=chat_messages,
            chat_template_kwargs=adapter.chat_template_kwargs,
        )
        if max_tokens is None or max_tokens:
            _validate_prompt_block_alignment(prompt_token_ids, runtime_config)
        # Resolve the decode-provenance params ONCE and reuse the same values for
        # both the engine call and the debug metrics, so the emitted JSON
        # reflects exactly what ran (reproducible from its own output).
        #
        # Decode defaults belong to the resolved model definition; explicit
        # package settings override them. Do not borrow another checkpoint's
        # faster schedule: block size and quota affect answer semantics.
        def decode_once(token_ids: list[int], budget: int):
            return self._mlx_local_state.generate_entropy(
                prompt_token_ids=token_ids,
                max_tokens=budget,
                mask_token_id=mask_token_id,
                eos_token_id=int(self.tokenizer.eos_token_id),
                steps=_fixed_quota_request_steps(runtime_config, budget),
                eps=runtime_config.eps,
                temperature=runtime_config.temperature,
                top_p=runtime_config.top_p,
                remasking=runtime_config.remasking,
                alg_temp=runtime_config.alg_temp,
                attention_mask_kind=plan.attention_mask_kind,
                enable_layer0_kv_cache=runtime_config.layer0_kv_cache_enabled,
                decode_strategy=runtime_config.decode_strategy,
                requires_prompt_block_alignment=runtime_config.requires_prompt_block_alignment,
                confidence_threshold=runtime_config.confidence_threshold,
                block_size=runtime_config.block_size,
                gamma=runtime_config.gamma,
                # Approximate prefix KV cache (Phase G throughput): OFF by default so
                # the forward stays byte-identical; the chat/long-context path enables
                # it via model_config once validated.
                enable_prefix_kv_cache=runtime_config.prefix_kv_cache,
                prefix_kv_refresh_interval=runtime_config.prefix_kv_refresh_interval,
                # Approximate block KV cache (Phase D throughput): ALSO cache committed-
                # block K/V so each step forwards only the still-masked frontier (the
                # bigger lever — erases the 51–55% per-step re-forward of committed
                # positions). OFF by default -> byte-identical. A strict superset of the
                # prefix cache (caches the prefix too), so it subsumes it when both flags
                # are set.
                enable_block_kv_cache=runtime_config.block_kv_cache,
                block_kv_refresh_interval=runtime_config.block_kv_refresh_interval,
                # EXACT d2f cross-block KV cache (D2F-2): valid only under the
                # d2f_hybrid mask (fail-closed otherwise), exclusive with the
                # approximate caches. OFF by default -> byte-identical; the D2F-3
                # block scheduler (below) is what drives it during a real decode.
                enable_d2f_exact_kv_cache=runtime_config.d2f_exact_kv_cache,
                # D2F-3 pipelined block scheduler thresholds (consumed ONLY by the
                # opt-in decode_strategy="d2f_block_pipeline"; tau_conf is the shared
                # confidence_threshold). Defaults = the D2F reference eval settings.
                d2f_add_threshold=runtime_config.d2f_add_threshold,
                d2f_activate_threshold=runtime_config.d2f_activate_threshold,
                # Commit 5 (the NFE lever): the within-window commit RULE for the d2f
                # pipeline. "threshold" (default) = byte-identical τ_conf; the opt-in
                # "entropy_budget" commits the EB joint-uncertainty set per active
                # block (reuses mlx_diffusion_gamma). Consumed only by
                # decode_strategy="d2f_block_pipeline".
                d2f_commit_policy=runtime_config.d2f_commit_policy,
                # K2-3 compatibility tombstones for the retired row-K/V side
                # cache. The resolver accepts only these neutral values.
                d2f_skip_committed_rows=runtime_config.d2f_skip_committed_rows,
                d2f_skip_refresh_interval=runtime_config.d2f_skip_refresh_interval,
                # Mixed-precision forward compute dtype (Phase D throughput): default
                # fp32 (fail-safe, byte-identical to the pre-bf16 forward). bf16 is an
                # explicit opt-in (a family that passes the bf16 parity gate AND wants
                # the RAM win pins "bf16" in its manifest); bf16's throughput value is
                # ~nil here, so fp32 is the safe default and a dropped pin cannot
                # silently break parity.
                compute_dtype=runtime_config.compute_dtype,
                # Phase F3 error model (bounded detect->remask->re-decode recovery).
                # OFF by default -> exactly today's single decode pass (byte-identical).
                error_fixing=runtime_config.error_fixing_kwargs(),
                trace_canvas=runtime_config.trace_canvas,
                # K2-3 compatibility tombstones. The engine accepts only these
                # neutral values; canvas and cross-request K/V ownership are gone.
                enable_canvas_cache=runtime_config.canvas_cache,
                enable_cross_request_cache=runtime_config.cross_request_cache,
                canvas_cache_budget_bytes=runtime_config.canvas_cache_budget_bytes,
                save_snapshot_path=snapshot_save_path,
                window_exec=runtime_config.window_exec,
            )

        generated, chunk_budget, chunk_count = _decode_until_end(
            decode_once,
            list(prompt_token_ids),
            max_tokens=max_tokens,
            sequence_limit=plan.model_config.get("max_position_embeddings"),
            whole_canvas=plan.attention_mask_kind in {"bidirectional", "full"},
        )
        effective_steps = _fixed_quota_request_steps(runtime_config, chunk_budget)
        text = self.tokenizer.decode(generated.visible_generated_token_ids)
        output_token_count = len(generated.visible_generated_token_ids)
        text_quality_passed = _passes_text_quality_gate(text)
        kv_cache_metrics = dict(generated.kv_cache_metrics or mlx_cache.empty_metrics())
        timing_metrics = dict(generated.timing_metrics or {})
        refresh_real = mlx_cache.derive_refresh_real(
            forward_count=generated.forward_count,
            quality_passed=text_quality_passed,
            reuse_count=int(kv_cache_metrics.get("mlx_kv_reuse_count", 0)),
            refresh_count=int(kv_cache_metrics.get("mlx_kv_refresh_count", 0)),
            invalidated_count=int(kv_cache_metrics.get("mlx_kv_invalidated_count", 0)),
        )
        tokenizer_real = (
            self.tokenizer.kind == "transformers_auto_tokenizer"
            and prompt_uses_chat_template
        )
        # Execution evidence is independent of the legacy English-text heuristic:
        # a numerical answer (or a poor answer) still came from a real forward.
        forward_real = generated.forward_count > 0
        baseline_status = {
            "tokenizer_real": tokenizer_real,
            "forward_real": forward_real,
            "entropy_real": (
                forward_real
                and generated.entropy_used
                and generated.decode_loop_kind == "mlx_dream_masked_diffusion_entropy"
            ),
            "refresh_real": refresh_real,
        }
        debug = None
        if return_debug:
            debug = DebugBundle(
                metrics={
                    "session_id": str(uuid.uuid4()),
                    "generation_backend": "mlx_local",
                    "backend_provenance": "local_mlx_execution_package",
                    "local_executor_verified": bool(baseline_status["forward_real"]),
                    "baseline_status": baseline_status,
                    "row_verified": all(baseline_status.values()),
                    "total_latency_ms": round(
                        (time.perf_counter() - started_at) * 1000.0, 3
                    ),
                    "mlx_local_state": "verified_generation"
                    if forward_real
                    else "no_forward_execution",
                    "mlx_prompt_uses_chat_template": prompt_uses_chat_template,
                    "mlx_text_quality_passed": text_quality_passed,
                    "mlx_text_quality_scope": "surface_heuristic_not_accuracy",
                    "decode_loop_kind": generated.decode_loop_kind,
                    "mlx_forward_count": generated.forward_count,
                    "mlx_committed_token_count": generated.committed_token_count,
                    "mlx_output_token_count": output_token_count,
                    "mlx_committed_tokens_per_forward": (
                        generated.committed_token_count / generated.forward_count
                        if generated.forward_count
                        else 0.0
                    ),
                    # Phase F metrics (neutral when error-fixing is OFF, the F1
                    # default): the aggression measure + the self-correction
                    # counters. ``committed_per_forward`` mirrors the inline
                    # ``mlx_committed_tokens_per_forward`` but is sourced from the
                    # Diffusion Runtime Layer result, so an aggressive CommitPolicy
                    # (F2) / a recovery loop (F3) report through one channel.
                    "mlx_committed_per_forward": generated.committed_per_forward,
                    "mlx_remask_count": generated.remask_count,
                    "mlx_corrected_positions": generated.corrected_positions,
                    "mlx_recovery_rounds": generated.recovery_rounds,
                    # F3 error-model accounting: flags, corrections vs churn, and
                    # the optional objective quality delta (all neutral when
                    # error-fixing is OFF -- the default).
                    "mlx_wrong_commit_count": generated.wrong_commit_count,
                    "mlx_churned_positions": generated.churned_positions,
                    "mlx_quality_delta": generated.quality_delta,
                    "mlx_s_over_t": (
                        generated.forward_count / output_token_count
                        if output_token_count
                        else None
                    ),
                    # Finalization evidence (FIX 1): honest finish reason and the
                    # state of the generated region at termination.
                    "mlx_hit_eos": generated.hit_eos,
                    "mlx_remaining_mask_count": generated.remaining_mask_count,
                    "mlx_finish_reason": generated.finish_reason,
                    "mlx_reply_chunks": chunk_count,
                    # Decode provenance (FIX 2): the ACTUAL params used, so the
                    # CLI `--json` is reproducible from its own output.
                    "mlx_diffusion_policy_profile": runtime_config.policy_profile,
                    "mlx_diffusion_decode_strategy": runtime_config.decode_strategy,
                    "mlx_diffusion_block_size": runtime_config.block_size,
                    "mlx_diffusion_configured_steps": runtime_config.steps,
                    "mlx_diffusion_effective_steps": effective_steps,
                    **(
                        {
                            "mlx_diffusion_temperature": runtime_config.temperature,
                            "mlx_diffusion_top_p": runtime_config.top_p,
                            "mlx_diffusion_remasking": runtime_config.remasking,
                        }
                        if runtime_config.decode_strategy in {"bd3_fixed_quota", "fixed_quota"}
                        else {}
                    ),
                    "mlx_diffusion_confidence_threshold": runtime_config.confidence_threshold,
                    "mlx_diffusion_gamma": runtime_config.gamma,
                    "mlx_diffusion_window_exec": runtime_config.window_exec_label,
                    "mlx_diffusion_snapshot_save_path": snapshot_save_path,
                    "block_iteration_count": generated.iteration_count,
                    **kv_cache_metrics,
                    **timing_metrics,
                    **_mlx_weight_state_metrics(self._mlx_local_state),
                    # MaskCanvas v0 control-plane trace (only when trace_canvas was
                    # requested). Serialized via to_dict() into a NESTED JSON-safe
                    # value, so the scalar timing_metrics above stays flat and
                    # cli.py's json.dumps(debug_metrics) remains valid.
                    **(
                        {"canvas_trace": generated.canvas_trace.to_dict(full=False)}
                        if generated.canvas_trace is not None
                        else {}
                    ),
                    **(
                        {"kv_manager_trace": generated.kv_manager_trace}
                        if generated.kv_manager_trace is not None
                        else {}
                    ),
                }
            )
        return GenerateResult(
            text=text, finish_reason=generated.finish_reason, debug=debug,
            visible_output_tokens=output_token_count,
        )

    def infill(
        self,
        prefix: str,
        suffix: str,
        *,
        max_fill_tokens: int,
        backend: str = "mlx_local",
        return_debug: bool = False,
        precision: str | None = None,
        device_ordinal: int = 0,
        device_memory_budget_bytes: int = 0,
    ) -> InfillResult:
        """dLLM-native infill / span-fill: fill masked holes BETWEEN a fixed
        ``prefix`` and a fixed ``suffix``.

        The bidirectional forward fills the holes respecting BOTH sides -- the
        differentiation from autoregressive models (which could only continue the
        prefix). ``prefix`` and ``suffix`` are tokenized as RAW token ids (NO chat
        template -- infill is span completion, not chat). Returns an
        :class:`InfillResult` with ``filled_text`` (the decoded hole) and
        ``full_text`` (prefix + filled + suffix). Routing is fail-closed; only the
        ``mlx_local`` and the compiled ``cuda_native`` bidirectional path support
        infill today.
        """
        self._require_execution_open()
        validate_generation_length(max_fill_tokens, field_name="max_fill_tokens")
        self._require_planned_backend(backend)
        if backend == CUDA_NATIVE_BACKEND:
            if precision is None:
                raise ValueError("cuda_native requires an explicit precision")
            return self._generate_cuda_infill(
                prefix,
                suffix,
                max_fill_tokens=max_fill_tokens,
                return_debug=return_debug,
                precision=precision,
                device_ordinal=device_ordinal,
                device_memory_budget_bytes=device_memory_budget_bytes,
            )
        if backend != MLX_LOCAL_BACKEND:
            raise ValueError(f"unknown infill backend: {backend}")
        self._validate_mlx_policy_profile_task("infill")
        return self._generate_mlx_infill(
            prefix,
            suffix,
            max_fill_tokens=max_fill_tokens,
            return_debug=return_debug,
        )

    def edit(
        self,
        text: str,
        start: int,
        end: int,
        *,
        max_fill_tokens: int,
        backend: str = "mlx_local",
        return_debug: bool = False,
        precision: str | None = None,
        device_ordinal: int = 0,
        device_memory_budget_bytes: int = 0,
    ) -> InfillResult:
        """dLLM-native span EDIT / rewrite: replace ``text[start:end]`` with newly
        generated content that respects BOTH the surrounding prefix AND suffix.

        A thin wrapper over :meth:`infill` (``prefix = text[:start]``,
        ``suffix = text[end:]``) — the bidirectional rewrite an autoregressive model
        cannot do (AR conditions only on the left context). The returned
        ``InfillResult.filled_text`` is the rewritten span; ``full_text`` is the edited
        document. ``start``/``end`` are character offsets into ``text``
        (``0 <= start <= end <= len(text)``); ``start == end`` is a pure insertion.
        """
        self._require_execution_open()
        validate_generation_length(max_fill_tokens, field_name="max_fill_tokens")
        if not (0 <= start <= end <= len(text)):
            raise ValueError(
                f"edit span [{start}, {end}) out of range for text of length {len(text)}"
            )
        if backend not in {MLX_LOCAL_BACKEND, CUDA_NATIVE_BACKEND}:
            raise ValueError(f"unknown edit backend: {backend}")
        if backend == MLX_LOCAL_BACKEND:
            self._validate_mlx_policy_profile_task("edit")
        return self.infill(
            text[:start],
            text[end:],
            max_fill_tokens=max_fill_tokens,
            backend=backend,
            return_debug=return_debug,
            precision=precision,
            device_ordinal=device_ordinal,
            device_memory_budget_bytes=device_memory_budget_bytes,
        )

    def _generate_mlx_infill(
        self,
        prefix: str,
        suffix: str,
        *,
        max_fill_tokens: int,
        return_debug: bool,
    ) -> InfillResult:
        started_at = time.perf_counter()
        self._validate_mlx_policy_profile_task("infill")
        from .backends import mlx_cache
        from .backends import mlx_dream

        if self.tokenizer is None:
            raise ValueError("runtime has no tokenizer")
        model_config = self.model_config or {}
        # FAIL-CLOSED routing up front (same as generation): reject unknown
        # architectures BEFORE any engine construction or forward.
        mask_token_id = _resolve_mlx_mask_token_id(model_config, self.tokenizer)
        plan = self._model_plan(mask_token_id)
        adapter = plan.adapter
        if adapter is None:
            raise ValueError("mlx_local requires a registered model definition")
        runtime_config = plan.runtime_config
        if runtime_config.attention_mask_kind == "causal":
            raise ValueError(
                "attention_mask_kind='causal' is supported only by the "
                "legacy numpy_local compatibility path"
            )

        # Tokenize prefix/suffix as RAW token ids -- NO chat template. Infill is
        # span completion: the model fills holes between the two raw spans.
        prefix_token_ids = [
            int(token_id) for token_id in self.tokenizer.encode_raw(prefix)
        ]
        suffix_token_ids = [
            int(token_id) for token_id in self.tokenizer.encode_raw(suffix)
        ]

        if self.fixture_mode:
            # Same unverified-skeleton contract as generation: a fixture package
            # has no real weights, so fail soft with the skeleton result instead
            # of a raw FileNotFoundError deep in the weight loader.
            return InfillResult(
                filled_text="",
                full_text=prefix + suffix,
                finish_reason="unverified_skeleton",
            )

        if self._mlx_local_state is None:
            self._mlx_local_state = mlx_dream.MLXDreamEngine(
                package_dir=self.package_dir,
                merged_weights_ref=self.merged_weights_ref,
                model_config=dict(self.model_config or {}),
                weight_artifact=plan.build.weight_artifact,
                graph=self.graph,
                backend_plan=plan.backend_plan,
                kernel_plan=plan.kernel_plan,
            )
        self._sync_mlx_local_engine()
        infilled = self._mlx_local_state.generate_infill(
            prefix_token_ids=prefix_token_ids,
            suffix_token_ids=suffix_token_ids,
            max_fill_tokens=max_fill_tokens,
            mask_token_id=mask_token_id,
            eos_token_id=int(self.tokenizer.eos_token_id),
            steps=_fixed_quota_request_steps(runtime_config, max_fill_tokens),
            eps=runtime_config.eps,
            temperature=runtime_config.temperature,
            top_p=runtime_config.top_p,
            alg_temp=runtime_config.alg_temp,
            attention_mask_kind=plan.attention_mask_kind,
            enable_layer0_kv_cache=runtime_config.layer0_kv_cache_enabled,
            decode_strategy=runtime_config.decode_strategy,
            confidence_threshold=runtime_config.confidence_threshold,
            block_size=runtime_config.block_size,
            gamma=runtime_config.gamma,
            # Default fp32 (fail-safe); bf16 is an explicit manifest opt-in. See the
            # matching note in the generate_entropy path above.
            compute_dtype=runtime_config.compute_dtype,
        )
        # Only the replacement is decoded. Round-tripping the fixed context can
        # strip whitespace, remove control-token literals or normalize the text.
        filled_text = self.tokenizer.decode(infilled.fill_token_ids, strip=False)
        full_text = prefix + filled_text + suffix
        fill_token_count = len(infilled.fill_token_ids)
        kv_cache_metrics = dict(infilled.kv_cache_metrics or mlx_cache.empty_metrics())
        timing_metrics = dict(infilled.timing_metrics or {})
        debug = None
        if return_debug:
            debug = DebugBundle(
                metrics={
                    "session_id": str(uuid.uuid4()),
                    "generation_backend": "mlx_local",
                    "backend_provenance": "local_mlx_execution_package",
                    "total_latency_ms": round(
                        (time.perf_counter() - started_at) * 1000.0, 3
                    ),
                    "mlx_infill_prefix_token_count": len(prefix_token_ids),
                    "mlx_infill_suffix_token_count": len(suffix_token_ids),
                    "mlx_infill_fill_token_count": fill_token_count,
                    "mlx_infill_forward_count": infilled.forward_count,
                    "mlx_infill_committed_token_count": infilled.committed_token_count,
                    "mlx_infill_remaining_mask_count": infilled.remaining_mask_count,
                    # Infill forward economy: forwards per filled token (the
                    # S/T analog for a bounded span).
                    "mlx_infill_s_over_t": (
                        infilled.forward_count / fill_token_count
                        if fill_token_count
                        else None
                    ),
                    "decode_loop_kind": infilled.decode_loop_kind,
                    "mlx_finish_reason": infilled.finish_reason,
                    "mlx_diffusion_decode_strategy": runtime_config.decode_strategy,
                    "mlx_diffusion_block_size": runtime_config.block_size,
                    "mlx_diffusion_confidence_threshold": runtime_config.confidence_threshold,
                    "mlx_diffusion_gamma": runtime_config.gamma,
                    "block_iteration_count": infilled.iteration_count,
                    **kv_cache_metrics,
                    **timing_metrics,
                    **_mlx_weight_state_metrics(self._mlx_local_state),
                }
            )
        return InfillResult(
            filled_text=filled_text,
            full_text=full_text,
            finish_reason=infilled.finish_reason,
            debug=debug,
        )

    def _hf_reference_model_refs(self) -> tuple[str, str | None]:
        base_ref = str(self.source_base_model_reference or "")
        adapter_ref = str(self.source_checkpoint_reference or "")
        if not base_ref:
            raise ValueError("hf_dream_reference requires source_base_model_reference")
        if not adapter_ref or adapter_ref == base_ref:
            adapter_ref = None
        return base_ref, adapter_ref


def _select_mlx_adapter(model_config: dict, mask_token_id: int):
    """Compatibility entry into the unified runtime-definition registry.

    Routing is fail-closed and exact. Product loads call this once while
    creating :class:`ExecutableModelPlan`; repeated calls remain only for mutable
    legacy handles and focused compatibility helpers.
    """
    from .backends.model_adapter import select_adapter

    return select_adapter(model_config, mask_token_id)


def _resolve_package_mlx_runtime_config(
    *,
    model_config: dict,
    mask_token_id: int,
    definition: ModelDefinition | None = None,
):
    adapter = (
        definition.build_runtime_adapter(model_config, mask_token_id)
        if definition is not None
        else _select_mlx_adapter(model_config, mask_token_id)
    )
    resolved = resolve_mlx_runtime_config(
        model_config,
        adapter_defaults=adapter.decode_defaults,
    )
    return adapter, resolved


def _validate_package_mlx_runtime_config(
    *,
    model_config: dict,
    mask_token_id: int,
    package_major: int,
    definition: ModelDefinition | None = None,
) -> tuple[ModelAdapter | None, ResolvedMLXRuntimeConfig]:
    adapter: ModelAdapter | None
    try:
        adapter = (
            definition.build_runtime_adapter(model_config, mask_token_id)
            if definition is not None
            else _select_mlx_adapter(model_config, mask_token_id)
        )
    except ValueError as exc:
        if not str(exc).startswith("unsupported_architecture_family:"):
            raise
        adapter = None
        adapter_defaults = _LOAD_TIME_FALLBACK_MLX_DEFAULTS
    else:
        adapter_defaults = adapter.decode_defaults
    return (
        adapter,
        resolve_mlx_runtime_config(
            model_config,
            adapter_defaults=adapter_defaults,
            allow_legacy_causal=package_major == LEGACY_PACKAGE_MAJOR,
        ),
    )


def _resolve_mlx_decode_params(
    model_config: dict, adapter
) -> tuple[str, int, float, float]:
    """Resolve the mlx_local decode params: adapter defaults, config overrides.

    The per-model decode defaults (``decode_strategy`` / ``block_size`` /
    ``confidence_threshold`` / ``gamma``) come from the definition's runtime
    Adapter. Matching ``model_config`` keys still override them.

    ``gamma`` is the Phase F F2 EB-Sampler budget; it is only consumed by the
    opt-in ``entropy_budget`` strategy, but is resolved here so the engine call and
    the debug provenance metrics use identical values regardless of strategy.

    Returns ``(decode_strategy, block_size, confidence_threshold, gamma)`` resolved
    ONCE.
    """
    resolved = resolve_mlx_runtime_config(
        model_config,
        adapter_defaults=adapter.decode_defaults,
    )
    return (
        resolved.decode_strategy,
        resolved.block_size,
        resolved.confidence_threshold,
        resolved.gamma,
    )


def _resolve_mlx_window_exec(model_config: dict) -> tuple[bool | None, str]:
    """Resolve the public active-window route selector.

    ``None`` / missing / ``"auto"`` keeps the engine default: use
    ``execute_window`` only when the declared-exact regime is active.
    ``"primary"`` / ``True`` force ``execute_window``; ``"oracle"`` / ``False``
    force ``full_forward``. Unknown values fail closed before engine construction.
    """
    resolved = resolve_mlx_runtime_config(
        model_config,
        adapter_defaults={"decode_strategy": "block_threshold", "block_size": 8},
    )
    return resolved.window_exec, resolved.window_exec_label


def _resolve_error_fixing(model_config: dict, adapter) -> dict | None:
    """Resolve the Phase F3 error-model config bundle (recovery loop), or ``None``.

    The master gate ``mlx_diffusion_error_fixing`` defaults to OFF (``False``) ->
    returns ``None`` -> the engine runs exactly today's single decode pass
    (byte-identical). Enabling it assembles the bundle ``RecoveryPolicy``
    consumes: the detector battery + their thresholds + the round budget. As with
    the decode params, ``adapter.decode_defaults`` supplies the per-model default
    and matching ``model_config`` keys OVERRIDE it (model-agnostic; no `if model==`).
    """
    defaults = adapter.decode_defaults
    config = dict(model_config)
    legacy_defaults = {
        "mlx_diffusion_error_fixing": "error_fixing",
        "mlx_diffusion_recovery_max_rounds": "recovery_max_rounds",
        "mlx_diffusion_recovery_detectors": "recovery_detectors",
        "mlx_diffusion_recovery_confidence_floor": "recovery_confidence_floor",
        "mlx_diffusion_recovery_repetition_max_ngram": "recovery_repetition_max_ngram",
        "mlx_diffusion_recovery_kl_threshold": "recovery_kl_threshold",
    }
    for public_key, default_key in legacy_defaults.items():
        if public_key not in config and default_key in defaults:
            config[public_key] = defaults[default_key]
    resolved = resolve_mlx_runtime_config(
        config,
        adapter_defaults=defaults,
    )
    return resolved.error_fixing_kwargs()


def _validate_prompt_block_alignment(
    prompt_token_ids: list[int], runtime_config: ResolvedMLXRuntimeConfig
) -> None:
    from .runtime_config import validate_prompt_block_alignment

    validate_prompt_block_alignment(
        prompt_length=len(prompt_token_ids),
        block_size=runtime_config.block_size,
        required=runtime_config.requires_prompt_block_alignment,
    )


AUTO_REPLY_CHUNK_TOKENS = 128
"""Without a limit, a reply is decoded in chunks of this many positions."""
AUTO_REPLY_LIMIT_TOKENS = 1024
"""Without a limit, a reply stops here if the model has not ended it."""


def _decode_until_end(decode_once, prompt_token_ids: list[int], *, max_tokens: int | None,
                      sequence_limit: object, whole_canvas: bool):
    """Decode one reply. With no limit, decode it until the model ends it.

    author:Opus -- how depends on what the model attends to.

    A block-causal model (block diffusion) writes block after block and stops at
    its end token; room it does not reach costs nothing. Without a limit it gets
    one decode with the whole automatic budget.

    A model that attends over its whole canvas (`whole_canvas`, masked diffusion)
    plans its answer to fit the canvas: a large canvas makes it slower and
    wordier. Without a limit it is decoded one ordinary chunk at a time. When a
    chunk is written to its last position without an end token and without a
    hole, the next chunk is decoded after it, with everything written so far as
    part of the prompt, which such a model attends to in the same way.

    Either way a reply stops at the end token, at `AUTO_REPLY_LIMIT_TOKENS`, or
    at the model's sequence limit. The decode loop itself is unchanged.

    Returns the reply's result, the budget of its last decode, and the number of
    decodes.
    """
    if max_tokens is not None:
        return decode_once(prompt_token_ids, max_tokens), max_tokens, 1
    limit = AUTO_REPLY_LIMIT_TOKENS
    if type(sequence_limit) is int and sequence_limit > 0:
        limit = min(limit, sequence_limit - len(prompt_token_ids))
    limit = max(1, limit)
    if not whole_canvas:
        return decode_once(prompt_token_ids, limit), limit, 1
    budget = min(AUTO_REPLY_CHUNK_TOKENS, limit)
    chunks = [decode_once(prompt_token_ids, budget)]
    written = list(chunks[-1].visible_generated_token_ids)
    while (
        chunks[-1].finish_reason == "length"
        and chunks[-1].remaining_mask_count == 0
        and len(chunks[-1].visible_generated_token_ids) == budget
        and len(written) < limit
    ):
        budget = min(AUTO_REPLY_CHUNK_TOKENS, limit - len(written))
        chunks.append(decode_once(prompt_token_ids + written, budget))
        written.extend(chunks[-1].visible_generated_token_ids)
    if len(chunks) == 1:
        return chunks[0], budget, 1
    last = chunks[-1]
    forward_count = sum(chunk.forward_count for chunk in chunks)
    committed = sum(chunk.committed_token_count for chunk in chunks)
    combined = dataclasses.replace(
        last,
        visible_generated_token_ids=written,
        iteration_count=sum(chunk.iteration_count for chunk in chunks),
        forward_count=forward_count,
        committed_token_count=committed,
        entropy_used=any(chunk.entropy_used for chunk in chunks),
        committed_per_forward=(committed / forward_count) if forward_count else 0.0,
        remask_count=sum(chunk.remask_count for chunk in chunks),
        corrected_positions=sum(chunk.corrected_positions for chunk in chunks),
        wrong_commit_count=sum(chunk.wrong_commit_count for chunk in chunks),
        churned_positions=sum(chunk.churned_positions for chunk in chunks),
    )
    return combined, budget, len(chunks)


def _fixed_quota_request_steps(
    runtime_config: ResolvedMLXRuntimeConfig, max_tokens: int
) -> int:
    # author:Opus -- a request carries an output cap and no step count. Upstream
    # often pairs the two in published evaluation recipes. Holding its budget while
    # the cap grows spreads it over more blocks and commits several tokens per
    # round (four at 512), which degrades the text. This product policy imposes
    # a floor; it is not identical to every upstream steps/cap combination.
    # Never schedule fewer than
    # one round per token of each block.
    if runtime_config.decode_strategy not in {"bd3_fixed_quota", "fixed_quota"}:
        return runtime_config.steps
    block_size = runtime_config.block_size
    blocks = (max_tokens + block_size - 1) // block_size
    return max(runtime_config.steps, blocks * block_size)


def _mlx_prompt_token_ids(
    tokenizer: TokenizerSpec,
    prompt: str,
    *,
    chat_messages: list[dict] | None = None,
    chat_template_kwargs: dict | None = None,
) -> tuple[list[int], bool]:
    """Build local-backend prompt token ids, owning chat templating in ONE place.

    ``chat_messages`` (the chat CLI path) is templated as a full conversation
    EXACTLY ONCE. Without it, ``prompt`` is templated as a lone user turn -- the
    byte-identical single-turn default. The runtime is the single renderer, so
    the CLI must NOT pre-render with ``apply_chat_template``; passing an already
    -templated string here would template it twice.
    """
    # The message list the template (or plain fallback) renders. The chat path
    # passes the full conversation; the single-turn path wraps the prompt once.
    if chat_messages is not None:
        messages = [dict(m) for m in chat_messages]
    else:
        messages = [{"role": "user", "content": prompt}]

    backend = getattr(tokenizer, "backend", tokenizer)
    apply_chat_template = getattr(backend, "apply_chat_template", None)
    if callable(apply_chat_template):
        # Thread the adapter's chat-template kwargs (e.g. Qwen3's
        # ``{"enable_thinking": False}``) into the SINGLE template call. Dream
        # passes ``{}`` -> byte-identical to the pre-adapter behavior.
        inputs = apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            **(chat_template_kwargs or {}),
        )
        return _input_ids_from_tokenizer_output(inputs), True
    # Plain tokenizer (no chat template): encode the prompt directly for the
    # single-turn path, or a plain ``ROLE: content`` transcript for chat so the
    # multi-turn context still reaches a template-less model.
    if chat_messages is not None:
        return tokenizer.encode(_plain_chat_transcript(messages)), False
    return tokenizer.encode(prompt), False


def _plain_chat_transcript(messages: list[dict]) -> str:
    """Render messages as a ``ROLE: content`` transcript ending in ``ASSISTANT:``.

    Mirrors :func:`edllm.chat._plain_prompt` so a template-less tokenizer still
    sees the multi-turn context and continues the assistant turn.
    """
    lines = [f"{str(m['role']).upper()}: {m['content']}" for m in messages]
    lines.append("ASSISTANT:")
    return "\n".join(lines)


def _resolve_mlx_mask_token_id(model_config: dict, tokenizer: object) -> int:
    """Resolve the mask token id for the MLX path without requiring a tokenizer.

    Prefers ``model_config['mask_token_id']`` (always present after
    ``_enriched_model_config``); falls back to the tokenizer's ``<|mask|>`` only
    when a tokenizer is available. Kept tokenizer-optional so the fail-closed
    architecture check can run before the (possibly absent) tokenizer is used.
    """
    config_mask = model_config.get("mask_token_id")
    if isinstance(config_mask, bool):
        raise ValueError(f"mask_token_id must be an int, not bool; got {config_mask!r}")
    if isinstance(config_mask, int):
        return config_mask
    if tokenizer is not None:
        # required=True: a missing <|mask|> used to resolve to the UNK id, and
        # the trailing `return 0` then masked with token 0 when there was no
        # tokenizer at all. Both produce a diffusion loop that runs and returns
        # plausible nonsense. For a framework that claims to be model-general
        # this has to fail closed -- the next family whose mask token is named
        # differently would otherwise generate silently wrong output.
        return tokenizer.token_to_id("<|mask|>", required=True)
    raise ValueError(
        "cannot resolve a mask token id: the package manifest has no "
        "`mask_token_id` and no tokenizer was supplied. Masking with 0 would "
        "run and return plausible nonsense."
    )


def _input_ids_from_tokenizer_output(inputs) -> list[int]:
    values = getattr(inputs, "input_ids", None)
    if values is None and isinstance(inputs, dict):
        values = inputs.get("input_ids")
    if values is None:
        values = inputs
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(token_id) for token_id in values]


def _mlx_weight_state_metrics(engine: object) -> dict:
    metrics = getattr(engine, "weight_state_metrics", None)
    if callable(metrics):
        return dict(metrics())
    return {}


def _read_runtime_manifest_snapshot(package_dir: str | Path) -> object:
    """Safely read and parse ``model.json`` exactly once for package loading."""
    package_dir = Path(package_dir)
    payload = _read_package_artifact_text(
        package_dir=package_dir,
        artifact_ref="model.json",
        artifact_name="manifest",
        max_bytes=_MAX_PACKAGE_MANIFEST_BYTES,
    )
    try:
        return json.loads(payload)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("compatibility_error") from exc


def _validate_runtime_manifest_snapshot(
    manifest_snapshot: object,
    *,
    model_config_override: dict | None = None,
) -> _ValidatedRuntimeManifestSnapshot:
    """Validate one detached manifest value without reopening ``model.json``.

    This is deliberately the pre-tokenizer half of package loading.  Version
    and structural schema failures retain their historical priority, while
    config errors keep their precise ``PackageConfigError`` messages.
    """
    try:
        manifest = copy.deepcopy(manifest_snapshot)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("compatibility_error") from exc
    if type(manifest) is not dict:
        raise ValueError("compatibility_error")

    version = manifest.get("version")
    version_fields = frozenset({"major", "minor", "patch"})
    if type(version) is not dict or set(version) != version_fields:
        raise ValueError("compatibility_error")
    if any(
        type(version[field]) is not int or version[field] < 0
        for field in version_fields
    ):
        raise ValueError("compatibility_error")
    major = version["major"]
    minor = version["minor"]
    patch = version["patch"]
    _preflight_runtime_model_identity(manifest)
    compatibility_warning = None
    deferred_schema_error = None
    if major == LEGACY_PACKAGE_MAJOR:
        if minor > LEGACY_PACKAGE_MINOR:
            raise ValueError("compatibility_error")
        if minor < LEGACY_PACKAGE_MINOR:
            compatibility_warning = "legacy_package_minor_version"
        else:
            deferred_schema_error = _validate_manifest_schema(manifest)
    elif major == RUNTIME_PACKAGE_VERSION[0]:
        if minor > RUNTIME_PACKAGE_VERSION[1]:
            raise ValueError("compatibility_error")
        deferred_schema_error = _validate_manifest_schema(manifest)
    else:
        raise ValueError("compatibility_error")

    if model_config_override is not None:
        raw_model_config = copy.deepcopy(model_config_override)
    elif "model_config" not in manifest and compatibility_warning is not None:
        # Package v1.0 predates model_config and keeps its compatibility-warning
        # verdict. An explicit null/scalar/list is malformed, not legacy absence.
        raw_model_config = {}
    else:
        raw_model_config = manifest.get("model_config")
        if type(raw_model_config) is not dict:
            if deferred_schema_error is not None:
                raise ValueError("compatibility_error") from deferred_schema_error
            raise ValueError("compatibility_error")
    try:
        validate_core_model_config(raw_model_config)
    except ValueError as exc:
        raise PackageConfigError(str(exc)) from exc
    try:
        validate_package_snapshot_config(raw_model_config)
    except ValueError as exc:
        raise PackageConfigError(str(exc)) from exc
    # Structural schema errors keep priority over role/profile/runtime checks.
    if deferred_schema_error is not None:
        raise ValueError("compatibility_error") from deferred_schema_error

    from .dflash_drafter import recognize_drafter, DFlashDrafterConfigError

    try:
        drafter_identity = recognize_drafter(raw_model_config)
    except DFlashDrafterConfigError as exc:
        err = PackageConfigError(str(exc))
        err.code = exc.code
        raise err from exc
    if drafter_identity is not None:
        from .dflash_drafter import DFlashDrafterStandaloneRefused

        raise DFlashDrafterStandaloneRefused(drafter_identity)

    fixture_mode = manifest.get("fixture_mode", True)
    if type(fixture_mode) is not bool:
        raise ValueError("compatibility_error")

    return _ValidatedRuntimeManifestSnapshot(
        manifest=manifest,
        version=Version(major=major, minor=minor, patch=patch),
        raw_model_config=copy.deepcopy(raw_model_config or {}),
        fixture_mode=fixture_mode,
        compatibility_warning=compatibility_warning,
    )


def _preflight_runtime_model_identity(manifest: dict) -> None:
    """Reject mixed or partial Product/Local identity before JSON Schema.

    JSON Schema still owns field types and the complete package shape.  This
    small preflight preserves the actionable boundary error: a package must
    bind either the historical Product descriptor identity or the embedded
    Local definition identity, never a partial mixture of the two.
    """

    product_fields = (
        "model_descriptor_key",
        "model_descriptor_digest",
        "model_precision",
        "admitted_backend",
    )
    local_fields = (
        "model_definition_key",
        "model_definition_digest",
        "model_build_digest",
        "model_definition",
    )
    product_present = tuple(field in manifest for field in product_fields)
    local_present = tuple(field in manifest for field in local_fields)
    if any(product_present) and any(local_present):
        raise PackageConfigError("package_model_definition_scope_ambiguous")
    if any(product_present) and not all(product_present):
        raise PackageConfigError("package_descriptor_identity_incomplete")
    if any(local_present) and not all(local_present):
        raise PackageConfigError("package_model_definition_identity_incomplete")


def _preflight_runtime_descriptor_identity(
    validated_snapshot: _ValidatedRuntimeManifestSnapshot,
) -> object | None:
    """Reject partial descriptor identity before package-specific validation."""
    manifest = validated_snapshot.manifest
    identity_fields = ("model_descriptor_key", "model_descriptor_digest")
    identity_present = tuple(field in manifest for field in identity_fields)
    if not any(identity_present):
        return None
    required_fields = (*identity_fields, "model_precision", "admitted_backend")
    if not all(field in manifest for field in required_fields):
        raise ValueError("package_descriptor_identity_incomplete")
    return manifest["model_descriptor_key"]


def _validate_runtime_descriptor_snapshot(
    package_dir: str | Path,
    validated_snapshot: _ValidatedRuntimeManifestSnapshot,
    *,
    descriptor: object | None = None,
    verify_weights_hash: bool = False,
) -> tuple[dict, str | None] | None:
    """Validate descriptor/package truth against the preflighted snapshot."""
    manifest = validated_snapshot.manifest
    descriptor_key = _preflight_runtime_descriptor_identity(validated_snapshot)
    if descriptor is None:
        if descriptor_key is None:
            return None
        from .model_coverage.producer_descriptors import runtime_descriptor_for_key

        descriptor = runtime_descriptor_for_key(descriptor_key)
    from .model_coverage.producer_descriptors import validate_execution_package

    return validate_execution_package(
        descriptor,
        package_dir,
        verify_weights_hash=verify_weights_hash,
        manifest_snapshot=manifest,
    )


def _resolve_build_plan(
    validated_snapshot: _ValidatedRuntimeManifestSnapshot,
    *,
    provider_deployments: tuple[DeploymentTarget, ...] = (),
    model_catalog: ModelCatalog | None = None,
) -> ResolvedBuildPlan:
    if model_catalog is not None and not isinstance(model_catalog, ModelCatalog):
        raise TypeError("catalog must be a ModelCatalog")
    manifest = validated_snapshot.manifest
    definition_key = str(manifest.get("model_definition_key", "") or "")
    definition_digest = str(manifest.get("model_definition_digest", "") or "")
    build_digest = str(manifest.get("model_build_digest", "") or "")
    descriptor_key = str(manifest.get("model_descriptor_key", "") or "")
    definition = None
    try:
        if descriptor_key:
            if definition_key or definition_digest or build_digest:
                raise PackageConfigError("package_model_definition_scope_ambiguous")
            definition = model_definition_for_catalog_key(descriptor_key)
        elif definition_key:
            payload = manifest.get("model_definition")
            if not definition_digest or not build_digest or not isinstance(
                payload, Mapping
            ):
                raise PackageConfigError(
                    "package_model_definition_identity_incomplete"
                )
            definition = model_definition_from_payload(payload)
            if definition.key != definition_key:
                raise PackageConfigError("package_model_definition_key_mismatch")
            if model_definition_digest(definition) != definition_digest:
                raise PackageConfigError("package_model_definition_digest_mismatch")
            if model_build_digest(definition) != build_digest:
                raise PackageConfigError("package_model_build_digest_mismatch")
            if model_catalog is not None:
                entry = model_catalog.entry_for_key(definition_key)
                if (
                    definition_digest != entry.definition_digest
                    or build_digest != entry.build_digest
                    or definition != entry.definition
                ):
                    raise PackageConfigError("package_model_definition_digest_mismatch")
        elif definition_digest or build_digest or "model_definition" in manifest:
            raise PackageConfigError("package_model_definition_identity_incomplete")
    except (KeyError, ValueError) as exc:
        raise PackageConfigError(str(exc)) from exc
    package_model_config = manifest.get("model_config")
    if type(package_model_config) is not dict:
        package_model_config = validated_snapshot.raw_model_config
    if definition is not None:
        try:
            resolved_runtime = runtime_architecture_for_config(
                package_model_config
            )
        except ValueError as exc:
            raise PackageConfigError(str(exc)) from exc
        if resolved_runtime is not definition.runtime:
            raise PackageConfigError("package_model_definition_runtime_mismatch")
        if not descriptor_key:
            expected_identity = {
                "source_checkpoint_reference": definition.model_id,
                "source_base_model_reference": definition.model_id,
                "source_checkpoint_sha": definition.revision,
                "source_base_model_sha": definition.revision,
                "source_license": definition.source_license,
                "graph_ir_family": definition.graph_ir_family,
            }
            for field, expected in expected_identity.items():
                if manifest.get(field) != expected:
                    raise PackageConfigError(f"package_{field}_mismatch")
            if package_model_config.get("mask_token_id") != definition.mask_token_id:
                raise PackageConfigError("package_mask_token_id_mismatch")
            if (
                package_model_config.get("tie_word_embeddings")
                is not definition.tie_word_embeddings
            ):
                raise PackageConfigError("package_tie_word_embeddings_mismatch")
    if (
        validated_snapshot.compatibility_warning is not None
        and "merged_weights_ref" not in manifest
    ):
        # Package v1.0 predates executable weights. It remains loadable only as
        # the explicit compatibility-warning fixture path and cannot execute.
        weight_artifact = WeightArtifactSpec.legacy_npz("weights.npz")
    else:
        try:
            weight_artifact = weight_artifact_spec_from_manifest(manifest)
        except ValueError as exc:
            raise PackageConfigError(str(exc)) from exc
    return ResolvedBuildPlan(
        package_model_config=package_model_config,
        definition=definition,
        descriptor_key=descriptor_key,
        descriptor_digest=str(manifest.get("model_descriptor_digest", "") or ""),
        definition_digest=definition_digest,
        build_digest=build_digest,
        deployments=(
            *(definition.deployments if definition is not None else ()),
            *provider_deployments,
        ),
        weight_artifact=weight_artifact,
    )


def _canonicalize_runtime_manifest_snapshot(
    package_dir: str | Path,
    manifest_snapshot: object | None = None,
    *,
    tokenizer: TokenizerSpec | None = None,
    validated_snapshot: _ValidatedRuntimeManifestSnapshot | None = None,
    build_plan: ResolvedBuildPlan | None = None,
) -> _CanonicalRuntimeManifest:
    """Return the exact canonical config used by both product and parity.

    The function consumes an in-memory snapshot only.  Supplying an already
    validated snapshot lets :func:`load_runtime` retain its tokenizer consent
    ordering without repeating any path read.
    """
    if validated_snapshot is None:
        validated_snapshot = _validate_runtime_manifest_snapshot(
            manifest_snapshot,
        )
    elif manifest_snapshot is not None:
        raise TypeError("manifest_snapshot and validated_snapshot are exclusive")

    build_plan = build_plan or _resolve_build_plan(validated_snapshot)
    model_definition = build_plan.definition
    try:
        profiled_model_config = apply_policy_profile(
            validated_snapshot.raw_model_config
        )
    except ValueError as exc:
        raise PackageConfigError(str(exc)) from exc
    model_config = _enrich_profiled_model_config(
        profiled_model_config,
        Path(package_dir),
        tokenizer=tokenizer,
        fixture_mode=validated_snapshot.fixture_mode,
        legacy=validated_snapshot.compatibility_warning is not None,
    )
    mask_token_id = (
        None
        if validated_snapshot.compatibility_warning is not None
        and "mask_token_id" not in model_config
        else _resolve_mlx_mask_token_id(model_config, tokenizer)
    )
    try:
        model_adapter, resolved_mlx_config = _validate_package_mlx_runtime_config(
            model_config=model_config,
            mask_token_id=mask_token_id,
            package_major=validated_snapshot.version.major,
            definition=model_definition,
        )
    except ValueError as exc:
        raise PackageConfigError(str(exc)) from exc
    return _CanonicalRuntimeManifest(
        manifest=copy.deepcopy(validated_snapshot.manifest),
        version=Version(
            major=validated_snapshot.version.major,
            minor=validated_snapshot.version.minor,
            patch=validated_snapshot.version.patch,
        ),
        model_config=copy.deepcopy(model_config),
        model_definition=model_definition,
        model_adapter=model_adapter,
        resolved_mlx_config=resolved_mlx_config,
        compatibility_warning=validated_snapshot.compatibility_warning,
    )


def _construct_runtime_from_validated_snapshot(
    package_dir: Path,
    validated_snapshot: _ValidatedRuntimeManifestSnapshot,
    *,
    verify_weights_hash: bool,
    trust_remote_code: bool,
    expected_module: object,
    build_plan: ResolvedBuildPlan | None = None,
    load_request: LoadRequest | None = None,
) -> RuntimeHandle:
    """Construct one runtime without reopening or revalidating its manifest."""
    # Resolve package identity before a tokenizer bundle can execute. New
    # ``model_definition_key`` values are meaningful only beside the validated
    # descriptor identity; old packages may omit the additive field.
    build_plan = build_plan or _resolve_build_plan(validated_snapshot)
    tokenizer_expected_module = expected_module
    if (
        tokenizer_expected_module is UNSPECIFIED
        and build_plan.definition is not None
    ):
        tokenizer_expected_module = build_plan.definition.custom_tokenizer_module
    tokenizer = TokenizerSpec.from_package(
        package_dir,
        fixture_mode=validated_snapshot.fixture_mode,
        trust_remote_code=trust_remote_code,
        expected_module=tokenizer_expected_module,
    )
    canonical = _canonicalize_runtime_manifest_snapshot(
        package_dir,
        tokenizer=tokenizer,
        validated_snapshot=validated_snapshot,
        build_plan=build_plan,
    )
    manifest = canonical.manifest
    version = canonical.version
    major = version.major
    minor = version.minor
    compatibility_warning = canonical.compatibility_warning
    model_config = canonical.model_config
    if verify_weights_hash:
        _verify_package_weights_hash(package_dir, manifest)
    load_request = load_request or LoadRequest(
        requested_backend="auto",
        backend="unbound",
        precision=str(manifest.get("model_precision", "") or "fp32"),
        deployment_profile="baseline",
    )
    deployment = None
    if (
        build_plan.definition is not None
        and load_request.backend in {MLX_LOCAL_BACKEND, CUDA_NATIVE_BACKEND}
    ):
        target = build_plan.resolve_deployment(
            load_request.backend,
            load_request.precision,
        )
        contract = target.deployment
        if contract is None or contract.profile == load_request.deployment_profile:
            deployment = target
        elif load_request.deployment_profile != "baseline":
            raise ValueError("executable deployment profile mismatch")
    graph = None
    if not (major == LEGACY_PACKAGE_MAJOR and minor < LEGACY_PACKAGE_MINOR):
        graph = _load_or_build_package_graph(
            package_dir=package_dir,
            manifest=manifest,
            model_config=model_config,
            definition=canonical.model_definition,
            deployment=deployment,
        )
    resolved_model_plan = ExecutableModelPlan(
        build=build_plan,
        load=load_request,
        deployment=deployment,
        model_config=copy.deepcopy(model_config),
        graph=graph,
        backend_plan=(
            compile_dense_transformer_plan(graph) if graph is not None else None
        ),
        kernel_plan=(
            plan_kernel_dispatch(
                model_config,
                compute_dtype=canonical.resolved_mlx_config.compute_dtype,
            )
            if load_request.backend in {MLX_LOCAL_BACKEND, "unbound"}
            else None
        ),
        adapter=canonical.model_adapter,
        runtime_config=canonical.resolved_mlx_config,
    )
    # A Product load validates the graph/config join now, before a backend can
    # acquire weights or K/V. Unknown legacy/fixture identities retain their
    # historical load-only compatibility and still fail at execution.
    if canonical.model_adapter is not None:
        _ = resolved_model_plan.attention_mask_kind
    return RuntimeHandle(
        package_version=Version(
            major=major,
            minor=minor,
            patch=version.patch,
        ),
        package_dir=package_dir,
        backend_memory_owner=manifest.get("backend_memory_owner", "numpy"),
        forward_executor=manifest.get("forward_executor", "python_scaffold"),
        runtime_impl=manifest.get("runtime_impl", "python_scaffold"),
        fixture_mode=validated_snapshot.fixture_mode,
        merged_weights_ref=manifest.get("merged_weights_ref", ""),
        model_config=model_config,
        graph=graph,
        tokenizer=tokenizer,
        source_checkpoint_reference=manifest.get("source_checkpoint_reference", ""),
        source_checkpoint_sha=manifest.get("source_checkpoint_sha", ""),
        source_base_model_reference=manifest.get("source_base_model_reference", ""),
        source_license=str(manifest.get("source_license", "") or ""),
        model_descriptor_key=str(manifest.get("model_descriptor_key", "") or ""),
        model_descriptor_digest=str(manifest.get("model_descriptor_digest", "") or ""),
        model_definition_key=str(manifest.get("model_definition_key", "") or ""),
        model_definition_digest=str(
            manifest.get("model_definition_digest", "") or ""
        ),
        model_build_digest=str(manifest.get("model_build_digest", "") or ""),
        model_precision=str(manifest.get("model_precision", "") or ""),
        admitted_backend=str(manifest.get("admitted_backend", "") or ""),
        decode_family=manifest.get("decode_family", ""),
        compatibility_warning=compatibility_warning,
        _expanded_mlx_diffusion_policy_profile=(
            canonical.resolved_mlx_config.policy_profile
        ),
        _resolved_build_plan=build_plan,
        _resolved_model_plan=resolved_model_plan,
    )


def _resolve_deployment_profile_config(
    model_config: dict,
    *,
    backend: str,
    mlx_profile_name: str | None,
    descriptor_key: str = "",
    precision: str = "fp32",
    build_plan: ResolvedBuildPlan | None = None,
) -> tuple[str, dict]:
    """Resolve one deployment overlay without mutating package-owned data.

    ``mlx_profile_name=None`` means "use only the exact package/descriptor
    default".  This is the ordinary Engine path: Product models receive their
    admitted optimization contract while an unregistered package keeps the
    configuration it carries.  A non-``None`` name is an explicit fallback
    used by the deployment command for legacy packages without a model
    definition. A Local definition without an optimization overlay still
    owns its decoder; it does not opt into that legacy fallback.
    """
    config = copy.deepcopy(model_config)
    if backend != MLX_LOCAL_BACKEND:
        return "baseline", config

    # ``baseline`` is the explicit no-overlay deployment.  It must remain
    # selectable even when a Product definition owns a faster default profile:
    # certification and callers need one stable package-semantics path without
    # rewriting the model registry or pretending the optimization is absent.
    if mlx_profile_name == "baseline":
        package_profile = config.get("mlx_diffusion_policy_profile")
        if package_profile not in (None, "baseline"):
            raise ValueError(
                "baseline deployment conflicts with the package policy profile"
            )
        return "baseline", config

    definition = build_plan.definition if build_plan is not None else None
    if definition is None and descriptor_key:
        definition = model_definition_for_catalog_key(descriptor_key)
    if definition is not None:
        target = definition.resolve_deployment(backend, precision)
        if target.deployment is not None:
            if target.deployment.profile == "baseline":
                package_profile = config.get("mlx_diffusion_policy_profile")
                if package_profile not in (None, "baseline"):
                    raise ValueError(
                        "baseline deployment conflicts with the package policy profile"
                    )
                target.deployment.validate_attention_mask_kind(
                    config.get("attention_mask_kind", "bidirectional")
                )
                return "baseline", config
            if config.get("mlx_diffusion_policy_profile") is not None:
                raise ValueError(
                    "package policy profile conflicts with the exact model definition "
                    "deployment contract"
                )
            if target.deployment.config_source == "package":
                target.deployment.validate_attention_mask_kind(
                    config.get("attention_mask_kind", "bidirectional")
                )
                return target.deployment.profile, config
            if target.deployment.config_source == "preset":
                resolved = apply_preset(config, target.deployment.profile)
                target.deployment.validate_attention_mask_kind(
                    resolved.get("attention_mask_kind", "bidirectional")
                )
                return target.deployment.profile, resolved
            raise ValueError("deployment_config_source_invalid")
    package_profile = config.get("mlx_diffusion_policy_profile")
    if package_profile is not None:
        return f"policy:{package_profile}", config
    if mlx_profile_name is None or definition is not None:
        return "baseline", config
    return mlx_profile_name, apply_preset(config, mlx_profile_name)


def _replace_validated_model_config(
    snapshot: _ValidatedRuntimeManifestSnapshot,
    model_config: dict,
) -> _ValidatedRuntimeManifestSnapshot:
    """Revalidate an in-memory deployment overlay at the package boundary."""
    config = copy.deepcopy(model_config)
    try:
        validate_core_model_config(config)
        validate_package_snapshot_config(config)
    except ValueError as exc:
        raise PackageConfigError(str(exc)) from exc
    return _ValidatedRuntimeManifestSnapshot(
        manifest=copy.deepcopy(snapshot.manifest),
        version=Version(
            major=snapshot.version.major,
            minor=snapshot.version.minor,
            patch=snapshot.version.patch,
        ),
        raw_model_config=config,
        fixture_mode=snapshot.fixture_mode,
        compatibility_warning=snapshot.compatibility_warning,
    )


def _validated_package_snapshot(package_dir: Path) -> _ValidatedRuntimeManifestSnapshot:
    manifest = _read_runtime_manifest_snapshot(package_dir)
    validated_snapshot = _validate_runtime_manifest_snapshot(manifest)
    try:
        _validate_runtime_descriptor_snapshot(package_dir, validated_snapshot)
    except ValueError as exc:
        raise PackageConfigError(str(exc)) from exc
    return validated_snapshot


def _validated_package_snapshot_for_backend(
    package_dir: Path,
    *,
    requested_backend: str,
) -> tuple[_ValidatedRuntimeManifestSnapshot, object | None]:
    """Validate package truth plus one authenticated backend overlay."""
    manifest = _read_runtime_manifest_snapshot(package_dir)
    cuda_profile = None
    model_config_override = None
    if requested_backend == CUDA_NATIVE_BACKEND:
        if type(manifest) is not dict or type(manifest.get("model_config")) is not dict:
            # Preserve the ordinary compatibility verdict for malformed input.
            validated_snapshot = _validate_runtime_manifest_snapshot(manifest)
        else:
            from .backends.cuda_profile import resolve_cuda_execution_profile_config

            cuda_profile, model_config_override = resolve_cuda_execution_profile_config(
                package_dir,
                model_config=manifest["model_config"],
            )
            validated_snapshot = _validate_runtime_manifest_snapshot(
                manifest,
                model_config_override=model_config_override,
            )
    else:
        validated_snapshot = _validate_runtime_manifest_snapshot(manifest)
    try:
        _validate_runtime_descriptor_snapshot(package_dir, validated_snapshot)
    except ValueError as exc:
        raise PackageConfigError(str(exc)) from exc
    return validated_snapshot, cuda_profile


def load_runtime(
    package_dir: str | Path,
    *,
    verify_weights_hash: bool = False,
    trust_remote_code: bool = False,
    expected_module: object = UNSPECIFIED,
    catalog: ModelCatalog | None = None,
) -> RuntimeHandle:
    """Load the package exactly as written.

    ``expected_module`` is the module the caller obtained consent for. The
    tokenizer loader verifies it against the materialized bundle immediately
    before AutoTokenizer runs. ``catalog`` resolves only an explicitly bound
    Local definition; it never mutates or extends Product admission.
    """
    package_path = Path(package_dir).expanduser().absolute()
    lease = acquire_package_load_lease(package_path)
    try:
        validated_snapshot = _validated_package_snapshot(package_path)
        build_plan = _resolve_build_plan(
            validated_snapshot,
            model_catalog=catalog,
        )
        runtime = _construct_runtime_from_validated_snapshot(
            package_path,
            validated_snapshot,
            verify_weights_hash=verify_weights_hash,
            trust_remote_code=trust_remote_code,
            expected_module=expected_module,
            build_plan=build_plan,
        )
    except BaseException as primary:
        _close_preserving_primary(
            lease.close,
            primary,
            label="package lease release",
        )
        raise
    runtime._package_lease = lease
    return runtime


def _load_runtime_for_backend(
    package_dir: str | Path,
    *,
    requested_backend: str,
    verify_weights_hash: bool = False,
    trust_remote_code: bool = False,
    expected_module: object = UNSPECIFIED,
    model_catalog: ModelCatalog | None = None,
) -> RuntimeHandle:
    """Load one package through its explicitly admitted backend profile."""
    package_path = Path(package_dir).expanduser().absolute()
    lease = acquire_package_load_lease(package_path)
    try:
        validated_snapshot, cuda_profile = _validated_package_snapshot_for_backend(
            package_path,
            requested_backend=requested_backend,
        )
        build_plan = _resolve_build_plan(
            validated_snapshot,
            provider_deployments=(
                cuda_profile.deployment_targets if cuda_profile is not None else ()
            ),
            model_catalog=model_catalog,
        )
        runtime = _construct_runtime_from_validated_snapshot(
            package_path,
            validated_snapshot,
            verify_weights_hash=verify_weights_hash,
            trust_remote_code=trust_remote_code,
            expected_module=expected_module,
            build_plan=build_plan,
        )
    except BaseException as primary:
        _close_preserving_primary(
            lease.close,
            primary,
            label="package lease release",
        )
        raise
    runtime._package_lease = lease
    return runtime


def load_runtime_for_deployment(
    package_dir: str | Path,
    *,
    requested_backend: str,
    mlx_profile_name: str | None = None,
    precision: str | None = None,
    device_ordinal: int = 0,
    device_memory_budget_bytes: int = 0,
    verify_weights_hash: bool = False,
    trust_remote_code: bool = False,
    expected_module: object = UNSPECIFIED,
    model_catalog: ModelCatalog | None = None,
) -> _DeploymentRuntimeLoad:
    """Load a package after resolving its admitted local-deployment profile.

    The manifest and descriptor are validated before the overlay. The overlay
    affects only the detached model-config snapshot consumed by canonical
    validation and graph construction; it never rewrites package bytes.  With
    no explicit ``mlx_profile_name``, an exact definition-owned contract is
    applied when present and all other packages retain their own configuration.
    """
    package_path = Path(package_dir).expanduser().absolute()
    lease = acquire_package_load_lease(package_path)
    try:
        validated_snapshot, cuda_profile = _validated_package_snapshot_for_backend(
            package_path,
            requested_backend=requested_backend,
        )
        manifest = validated_snapshot.manifest
        build_plan = _resolve_build_plan(
            validated_snapshot,
            provider_deployments=(
                cuda_profile.deployment_targets if cuda_profile is not None else ()
            ),
            model_catalog=model_catalog,
        )
        backend = _resolve_generation_backend(
            requested_backend,
            fixture_mode=validated_snapshot.fixture_mode,
            backend_memory_owner=str(manifest.get("backend_memory_owner", "numpy")),
            forward_executor=str(
                manifest.get("forward_executor", "python_scaffold")
            ),
        )
        resolved_precision = str(
            precision or manifest.get("model_precision", "") or "fp32"
        )
        if backend == CUDA_NATIVE_BACKEND:
            if cuda_profile is None:
                raise PackageConfigError("cuda_package_profile_not_admitted")
            profile = cuda_profile.profile_id
            model_config = copy.deepcopy(validated_snapshot.raw_model_config)
        else:
            profile, model_config = _resolve_deployment_profile_config(
                validated_snapshot.raw_model_config,
                backend=backend,
                mlx_profile_name=mlx_profile_name,
                descriptor_key=str(manifest.get("model_descriptor_key", "") or ""),
                precision=resolved_precision,
                build_plan=build_plan,
            )
        load_request = LoadRequest(
            requested_backend=requested_backend,
            backend=backend,
            precision=resolved_precision,
            deployment_profile=profile,
            device_ordinal=device_ordinal,
            device_memory_budget_bytes=device_memory_budget_bytes,
        )
        profiled_snapshot = _replace_validated_model_config(
            validated_snapshot,
            model_config,
        )
        runtime = _construct_runtime_from_validated_snapshot(
            package_path,
            profiled_snapshot,
            verify_weights_hash=verify_weights_hash,
            trust_remote_code=trust_remote_code,
            expected_module=expected_module,
            build_plan=build_plan,
            load_request=load_request,
        )
    except BaseException as primary:
        _close_preserving_primary(
            lease.close,
            primary,
            label="package lease release",
        )
        raise
    runtime._package_lease = lease
    plan = runtime._resolved_model_plan
    if plan is None:
        runtime.close()
        raise RuntimeError("deployment load did not create an executable model plan")
    return _DeploymentRuntimeLoad(
        runtime=runtime,
        plan=plan,
    )


def _verify_package_weights_hash(package_dir: str | Path, manifest: dict) -> None:
    """Verify the declared NPZ file, or the index and every recorded shard.

    This opt-in check streams bytes without loading tensors. Legacy packages
    without an output hash or weights reference have nothing to verify;
    declared but missing artifacts fail closed.
    """
    if manifest.get("package_format") == SAFETENSORS_SHARDED_V1:
        artifact = weight_artifact_spec_from_manifest(manifest)
        index = read_weight_store_index(package_dir, artifact)
        for shard in index.shards:
            # The context hashes the opened descriptor on successful exit.
            with open_verified_weight_shard(package_dir, shard):
                pass
        return
    expected = manifest.get("output_hash")
    weights_ref = manifest.get("merged_weights_ref")
    if not expected or not weights_ref:
        return
    with open_package_artifact(
        package_path=package_dir,
        artifact_ref=weights_ref,
        error_prefix="",
        artifact_name="weights",
    ) as (handle, _weights_path, _weights_stat):
        actual = sha256_stream(handle)
    if actual != expected:
        raise ValueError("weights_hash_mismatch")


def try_load_package(package_dir: str | Path) -> dict:
    runtime = None
    try:
        runtime = load_runtime(package_dir)
    except PackageConfigError:
        return {"kind": "compatibility_error"}
    except ValueError as exc:
        # DFlashDrafterStandaloneRefused is a ValueError subclass; match its
        # precise code so a drafter package maps to compatibility_error rather
        # than raising past the caller.
        from .dflash_drafter import DFlashDrafterStandaloneRefused

        if isinstance(exc, DFlashDrafterStandaloneRefused) or (
            str(exc) == "compatibility_error"
        ):
            return {"kind": "compatibility_error"}
        raise

    try:
        if runtime.compatibility_warning is not None:
            return {
                "kind": "compatibility_warning",
                "warning": runtime.compatibility_warning,
            }
        return {"kind": "load_success"}
    finally:
        runtime.close()


def _validate_manifest_schema(manifest: dict) -> jsonschema.ValidationError | None:
    schema = json.loads(
        resources.files("edllm.schemas")
        .joinpath("execution_package.schema.json")
        .read_text(encoding="utf-8")
    )
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(manifest))
    if not errors:
        return None
    for error in errors:
        path = list(error.absolute_path)
        if not path or path[0] != "model_config":
            raise ValueError("compatibility_error") from error
    return errors[0]


def _load_or_build_package_graph(
    *,
    package_dir: Path,
    manifest: dict,
    model_config: dict,
    definition: ModelDefinition | None = None,
    deployment: DeploymentTarget | None = None,
) -> GraphIR:
    """Load package GraphIR, allowing only a typed deployment mask transform."""
    graph_ref = manifest.get("graph_ref")
    if graph_ref:
        graph_payload = _read_package_artifact_text(
            package_dir=package_dir,
            artifact_ref=graph_ref,
            artifact_name="graph",
            max_bytes=_MAX_PACKAGE_GRAPH_BYTES,
        )
        try:
            graph = GraphIR.from_json(graph_payload)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("compatibility_error") from exc
        manifest_ir = manifest.get("ir_version")
        if manifest_ir is not None and dict(manifest_ir) != graph.ir_version:
            raise ValueError("compatibility_error")
        package_config = manifest.get("model_config")
        if type(package_config) is not dict:
            package_config = model_config
        package_mask_kind = str(
            package_config.get("attention_mask_kind", "bidirectional")
        )
        try:
            graph_mask_kind = uniform_attention_mask_kind(graph)
        except ValueError as exc:
            raise ValueError("compatibility_error") from exc
        if not same_mask_kind_family(package_mask_kind, graph_mask_kind):
            raise ValueError("package GraphIR contradicts package model_config")
        if definition is not None:
            # Compile the backend contract once from the exact definition and
            # require the packaged graph to agree. ``full`` and
            # ``bidirectional`` are equivalent spellings, so compare using the
            # package spelling after the mask-family check above.
            expected_config = copy.deepcopy(package_config)
            expected_config["attention_mask_kind"] = graph_mask_kind
            expected_graph = build_graph_ir(
                expected_config,
                runtime=definition.runtime,
            )
            if graph.to_dict() != expected_graph.to_dict():
                raise ValueError("package GraphIR contradicts ModelDefinition")
        effective_mask_kind = str(
            model_config.get("attention_mask_kind", "bidirectional")
        )
        if not same_mask_kind_family(effective_mask_kind, graph_mask_kind):
            contract = deployment.deployment if deployment is not None else None
            fixture_profile = str(
                model_config.get("mlx_diffusion_policy_profile", "") or ""
            )
            fixture_overlay = bool(manifest.get("fixture_mode")) and bool(
                fixture_profile
            )
            if (
                not fixture_overlay
                and (
                    contract is None
                    or contract.attention_mask_kind != effective_mask_kind
                )
            ):
                raise ValueError(
                    "deployment model_config contradicts package GraphIR"
                )
            return build_graph_ir(
                model_config,
                runtime=(definition.runtime if definition is not None else None),
            )
        return graph
    return build_graph_ir(
        model_config,
        runtime=(definition.runtime if definition is not None else None),
    )


def _read_package_artifact_text(
    *,
    package_dir: Path,
    artifact_ref: object,
    artifact_name: str,
    max_bytes: int,
) -> str:
    with open_package_artifact(
        package_path=package_dir,
        artifact_ref=artifact_ref,
        error_prefix="",
        artifact_name=artifact_name,
        max_bytes=max_bytes,
    ) as (handle, _artifact_path, _source_stat):
        payload = read_open_artifact(
            handle,
            max_bytes=max_bytes,
            error_prefix="",
            artifact_name=artifact_name,
        )
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{artifact_name}_encoding_invalid") from exc


def _enriched_model_config(
    model_config: dict | None,
    package_dir: Path,
    *,
    tokenizer: TokenizerSpec | None = None,
) -> dict:
    return _enrich_profiled_model_config(
        apply_policy_profile(model_config or {}),
        package_dir,
        tokenizer=tokenizer,
    )


def _enrich_profiled_model_config(
    model_config: dict,
    package_dir: Path,
    *,
    tokenizer: TokenizerSpec | None = None,
    fixture_mode: bool = True,
    legacy: bool = False,
) -> dict:
    config = dict(model_config)
    # ONE resolution, no cascade. This used to try <|mask|>, then
    # <|mask_token|>, then <|endoftext|>, then EOS, then 0 -- five different
    # tokens, four of which mean something else entirely, and the last of which
    # is whatever token 0 happens to be. Every one of them lets the diffusion
    # loop run and return plausible nonsense.
    #
    # A manifest value wins when it is a valid int: the compiler wrote it from
    # the source config, and it is the model's own answer.
    raw_mask_token_id = valid_token_id(config.get("mask_token_id"))
    if raw_mask_token_id is not None:
        config["mask_token_id"] = raw_mask_token_id
    elif legacy:
        # Left ABSENT rather than invented. The package already carries a
        # compatibility warning; a decode path that needs the token refuses at
        # the point it needs it, which names the real problem.
        config.pop("mask_token_id", None)
    else:
        tokenizer = tokenizer or TokenizerSpec.from_package(
            package_dir,
            fixture_mode=fixture_mode,
        )
        config["mask_token_id"] = tokenizer.require_control_token("<|mask|>")
    config.setdefault("block_size", 2)
    config.setdefault("step_budget", 4)
    config.setdefault("diffusion_steps", 4)
    config.setdefault("diffusion_eps", 1e-3)
    config.setdefault("attention_mask_kind", "bidirectional")
    if _uses_qwen_geometry_defaults(config):
        config.setdefault(
            "head_dim",
            max(
                1,
                int(config.get("hidden_size", 1))
                // max(1, int(config.get("attention_heads", 1))),
            ),
        )
        config.setdefault("rms_norm_eps", 1e-6)
        config.setdefault("rope_theta", 1000000.0)
        config.setdefault("rope_scaling", {"rope_type": "linear", "factor": 1.0})
        # Dream-7B / Dream-Coder-7B GEOMETRY only. Do NOT inject 7B geometry for
        # non-Dream Qwen-family packages (e.g. Qwen3-MDLM 0.6B) — they must declare
        # their own num_kv_heads / intermediate_size in the manifest. (General
        # platform: a package carries its geometry; only the known Dream family gets
        # the 7B convenience defaults.)
        if str(config.get("architecture", "")).lower() in {
            "dreammodel",
            "dream",
            "dream_stub",
        }:
            config.setdefault("num_kv_heads", 4)
            config.setdefault("intermediate_size", 18944)
    return config


def _uses_qwen_geometry_defaults(config: dict) -> bool:
    """Return true only for exact registered Qwen/Dream compatibility aliases.

    New execution packages carry their geometry. These defaults remain for the
    known pre-GraphIR fixture aliases, but an arbitrary ``qwen*`` or ``dream*``
    string must not inherit executable geometry by prefix coincidence.
    """
    from .model_definitions import runtime_architecture_for_config

    try:
        architecture = runtime_architecture_for_config(config)
    except ValueError:
        return False
    return architecture.key in {"qwen2_dream", "qwen3_mdlm", "qwen3_bd3lm"}
