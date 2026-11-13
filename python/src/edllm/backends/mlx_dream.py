from __future__ import annotations

import contextlib
import hashlib
import json
import os
import stat
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import cache_registry
from . import diffusion_runtime
from . import mask_canvas
from . import mlx_cache
from . import mlx_forward
from . import mlx_ops
from .decode_session import DecodeRequest, DecodeSession, RecoveryPolicy
from .kernel_table import (
    KernelDispatchPlan,
    KernelTable,
    canonical_compute_dtype_name as _canonical_compute_dtype_name,
    materialize_kernel_table,
    plan_kernel_dispatch,
)
from ..graph import (
    DenseTransformerPlan,
    GraphIR,
    compile_dense_transformer_plan,
)
from ..runtime_config import (
    validate_bd3_sampling_contract,
    validate_d2f_recovery_contract,
    validate_fixed_quota_sampling_contract,
    validate_generation_length,
    validate_prompt_block_alignment,
)
from ..weight_store import (
    NPZ_V1,
    SAFETENSORS_SHARDED_V1,
    WeightArtifactSpec,
    open_verified_weight_shard,
    read_weight_store_index,
)
from .backend_runner import BackendRunner, MLXBackendRunner


LAYER0_TRACE_STAGE_ORDER = (
    "embed",
    "input_layernorm",
    "q_linear",
    "k_linear",
    "v_linear",
    "q_after_rope",
    "k_after_rope",
    "attention_output",
    "attention_projected",
    "layer0_output",
)

_WEIGHTS_HASH_CHUNK_BYTES = 1024 * 1024
_MLX_WEIGHT_LOADER_PROVENANCE = object()


@dataclass(slots=True)
class MLXGenerationResult:
    visible_generated_token_ids: list[int]
    iteration_count: int
    forward_count: int
    decode_loop_kind: str
    entropy_used: bool
    committed_token_count: int
    # Finalization evidence: no dataclass defaults (these are honesty-bearing,
    # like the fields above). ``hit_eos`` is True iff an EOS was produced in the
    # generated region; ``remaining_mask_count`` is the number of unresolved
    # masks still in the answer (region before the first EOS, or the whole
    # region at the step cap); ``finish_reason`` is "stop" on EOS else "length".
    hit_eos: bool
    remaining_mask_count: int
    finish_reason: str
    kv_cache_metrics: dict[str, int | bool | str] | None = None
    timing_metrics: dict[str, float | int | bool] | None = None
    # Phase F metrics (neutral when error-fixing is OFF, the F1 default):
    # committed-per-forward (aggression), and the self-correction accounting
    # (F3 error model: flags, re-masks, corrections vs churn, quality delta).
    committed_per_forward: float = 0.0
    remask_count: int = 0
    corrected_positions: int = 0
    recovery_rounds: int = 1
    wrong_commit_count: int = 0
    churned_positions: int = 0
    quality_delta: float | None = None
    # MaskCanvas v0 control-plane trace. ``None`` unless ``trace_canvas=True`` was
    # requested. It is a ``mask_canvas.MaskCanvasTrace`` (not put in the scalar
    # ``timing_metrics``); model.py serializes it via ``to_dict()`` into the debug
    # metrics so the JSON surface stays valid.
    canvas_trace: Any = None
    # K2-2 native-owner evidence. Separate from scalar ``kv_cache_metrics``;
    # this is the validated native wire document, not a second serialization.
    kv_manager_trace: dict[str, Any] | None = None


@dataclass(slots=True)
class MLXInfillResult:
    """Result of a dLLM-native infill / span-fill decode.

    Unlike :class:`MLXGenerationResult` (open-ended trailing generation), infill
    fills a BOUNDED span between fixed prefix/suffix, so there is no first-EOS
    truncation: ``fill_token_ids`` are the decoded tokens at the originally
    fillable (hole) positions in order, and ``full_token_ids`` is the whole
    sequence (prefix + filled holes + suffix). ``remaining_mask_count`` is the
    number of holes the step cap left masked. The honesty-bearing decode labels
    (``decode_loop_kind`` / ``entropy_used``) and forward/commit counters mirror
    the generation result so the same metrics make sense.
    """

    fill_token_ids: list[int]
    full_token_ids: list[int]
    iteration_count: int
    forward_count: int
    committed_token_count: int
    decode_loop_kind: str
    entropy_used: bool
    remaining_mask_count: int
    finish_reason: str
    kv_cache_metrics: dict[str, int | bool | str] | None = None
    timing_metrics: dict[str, float | int | bool] | None = None


@dataclass(slots=True)
class MLXWeightState:
    weights: dict[str, Any]
    source_path: Path
    load_count: int = 1
    source_digest: str | None = field(default=None, init=False, repr=False)
    source_stat: tuple[int, int, int, int, int] | None = field(
        default=None, init=False, repr=False
    )
    _loader_provenance: object | None = field(default=None, init=False, repr=False)


def _validate_retired_kv_direct_config(
    *,
    exact_kv_slice_mode: Any,
    exact_kv_slice_page_budget_bytes: int | None,
    d2f_skip_committed_rows: bool,
    d2f_skip_refresh_interval: int,
    enable_delayed_kv: bool,
    delayed_kv_scope: str,
    delayed_kv_max_stale_steps: int | None,
    delayed_kv_refresh_interval: int | None,
    delayed_kv_upper_layer_range: tuple[int, int] | None,
    delayed_kv_stale_epoch_budget: int | None,
    enable_canvas_cache: bool,
    canvas_cache_budget_bytes: int | None,
    enable_cross_request_cache: bool,
    enable_cross_session_cache: bool,
    cross_session_cache_root: str | Path | None,
) -> None:
    """Keep removed K/V controls as explicit, fail-closed tombstones."""
    neutral = {
        "exact_kv_slice_mode": (exact_kv_slice_mode, "off"),
        "exact_kv_slice_page_budget_bytes": (
            exact_kv_slice_page_budget_bytes,
            None,
        ),
        "d2f_skip_committed_rows": (d2f_skip_committed_rows, False),
        "d2f_skip_refresh_interval": (d2f_skip_refresh_interval, 4),
        "enable_delayed_kv": (enable_delayed_kv, False),
        "delayed_kv_scope": (delayed_kv_scope, "layer0"),
        "delayed_kv_max_stale_steps": (delayed_kv_max_stale_steps, None),
        "delayed_kv_refresh_interval": (delayed_kv_refresh_interval, None),
        "delayed_kv_upper_layer_range": (delayed_kv_upper_layer_range, None),
        "delayed_kv_stale_epoch_budget": (delayed_kv_stale_epoch_budget, None),
        "enable_canvas_cache": (enable_canvas_cache, False),
        "canvas_cache_budget_bytes": (canvas_cache_budget_bytes, None),
        "enable_cross_request_cache": (enable_cross_request_cache, False),
        "enable_cross_session_cache": (enable_cross_session_cache, False),
        "cross_session_cache_root": (cross_session_cache_root, None),
    }
    for name, (actual, expected) in neutral.items():
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(
                f"{name} is no longer supported in DiffusorRT 0.2; only the "
                "neutral compatibility "
                f"value {expected!r} is accepted"
            )


@dataclass(slots=True)
class MLXDreamEngine:
    package_dir: Path | None
    merged_weights_ref: str
    model_config: dict[str, Any]
    weight_artifact: WeightArtifactSpec | None = field(default=None, repr=False)
    graph: GraphIR | None = field(default=None, repr=False)
    backend_plan: DenseTransformerPlan | None = field(default=None, repr=False)
    kernel_plan: KernelDispatchPlan | None = field(default=None, repr=False)
    # The immutable selector/capability plan is produced at package load. The
    # callable table is materialized once from it on first use. Direct low-level
    # callers may omit the plan, in which case __post_init__ creates the same
    # frozen plan from their compatibility config. Tests may inject a table.
    kernel_table: KernelTable | None = field(default=None, repr=False)
    _kernel_table_injected: bool = field(default=False, init=False, repr=False)
    # The execution backend the engine routes its forward through. Defaults to
    # the MLX runner; a ``MetalBackendRunner`` (same BackendRunner protocol) can
    # be injected here with NO other engine/decode/adapter change.
    backend_runner: BackendRunner = field(default_factory=MLXBackendRunner)
    _weight_state: MLXWeightState | None = field(default=None, init=False, repr=False)
    _cleanup_pending: bool = field(default=False, init=False, repr=False)
    _resident_dtype_name: str | None = field(default=None, init=False, repr=False)
    # Number of REAL (re)loads this engine performed; mirrored onto each new
    # weight state's ``load_count`` so ``weight_state_metrics`` reports reloads
    # instead of the constant 1.
    _weight_load_count: int = field(default=0, init=False, repr=False)
    # Quant-1: the linear-quantization spec the CURRENT resident weights were
    # prepared with (e.g. "none" / "int8-g64"). ``None`` means "unknown" (no
    # load ran — e.g. a test-injected weight state, honored as-is, mirroring
    # ``_resident_dtype_name``); a recorded name that differs from the config's
    # spec forces a reload so quantization can never go stale.
    _resident_quant_name: str | None = field(default=None, init=False, repr=False)
    # Legacy/direct callers may build GraphIR lazily. Product construction passes
    # GraphIR plus its already validated DenseTransformerPlan before any weights
    # or cache state are acquired.
    _graph: GraphIR | None = field(default=None, init=False, repr=False)
    # K2-2: one compiled manager per exact-D2F generation.  It stays attached to
    # the engine after the decode so ``close()`` is the checked lifetime boundary;
    # starting another exact decode first closes the previous manager rather than
    # accumulating generation-local identities in an unbounded shared store.
    _native_kv_manager: Any = field(default=None, init=False, repr=False)
    # A failed decode cleanup can still own backend handles that never crossed
    # into the manager.  Keep the scheduling facade too until ``close_decode``
    # succeeds; retaining only the manager loses the sole names of those
    # orphaned handles and makes checked cleanup impossible to retry.
    _native_kv_strategy: Any = field(default=None, init=False, repr=False)
    _native_kv_generation: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        # Direct construction is a bounded fixture/compatibility seam. Leave a
        # missing spec on the established NPZ path so unsafe refs retain the
        # backend's load-time error contract. Production construction always
        # supplies the package-validated immutable plan.
        if (
            self.weight_artifact is not None
            and self.weight_artifact.primary_ref != self.merged_weights_ref
        ):
            raise ValueError("MLX weight artifact disagrees with package weights ref")
        self._kernel_table_injected = self.kernel_table is not None
        if self.backend_plan is not None:
            if self.graph is not None and self.backend_plan.graph is not self.graph:
                raise ValueError("compiled backend plan does not belong to GraphIR")
            self.graph = self.backend_plan.graph
        if self.graph is not None:
            self._graph = self.graph
            if self.backend_plan is None:
                self.backend_plan = compile_dense_transformer_plan(self.graph)
        if self.kernel_plan is None:
            self.kernel_plan = plan_kernel_dispatch(self.model_config)

    def close(self) -> None:
        """Release every engine-owned resident/cache reference. Idempotent.

        After successful close, a later forward rebuilds the GraphIR,
        native manager, and resident weights from package truth. Clearing
        the resident dict before dropping the state also releases MLX arrays when
        an exception traceback temporarily retains the state object itself.
        """
        # The native K/V owner goes first.  If cleanup reports debt, keep the
        # manager and every resident weight reachable so a caller can retry the
        # checked close; clearing them first would turn a reported failure into
        # an unrecoverable leak hidden behind an idempotent-looking method.
        self._cleanup_pending = True
        self._close_native_kv_manager()
        state = self._weight_state
        if state is not None:
            state.weights.clear()
        self._resident_dtype_name = None
        self._resident_quant_name = None
        self._weight_load_count = 0
        self.graph = None
        self._graph = None
        self.backend_plan = None
        self.kernel_table = None
        if state is not None:
            # MLX caches released buffers process-wide. Finish device work
            # before clearing that cache, without importing MLX for an unused
            # engine. Retain the cleared state until cleanup succeeds so a
            # failed checked close retries synchronization and cache release.
            mx = sys.modules.get("mlx.core")
            if mx is not None:
                mx.synchronize()
                mx.clear_cache()
        self._weight_state = None
        self._cleanup_pending = False

    def _close_native_kv_decode(self) -> None:
        strategy = self._native_kv_strategy
        if strategy is not None:
            strategy.close_decode()
            self._native_kv_strategy = None

    def _close_native_kv_manager(self) -> None:
        self._close_native_kv_decode()
        manager = self._native_kv_manager
        if manager is None:
            return
        manager.shutdown()
        manager.close()
        self._native_kv_manager = None

    @staticmethod
    def _native_kv_digest(domain: str, value: Any) -> str:
        canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(domain.encode("utf-8") + canonical).hexdigest()

    def _new_native_d2f_strategy(
        self,
        *,
        prompt_token_ids: list[int],
        max_tokens: int,
        block_size: int,
        decode_strategy: str,
        attention_mask_kind: str,
        compute_dtype: str | None,
        graph: GraphIR,
        d2f_commit_policy: str,
        confidence_threshold: float,
        d2f_add_threshold: float,
        d2f_activate_threshold: float,
        gamma: float,
        error_fixing_enabled: bool,
    ):
        """Open the only K/V owner and bind the generation's exact identity."""
        from .cache_identity import PolicyVersion
        from .kv_capacity import exact_d2f_kv_capacity
        from .kv_runtime import open_manager
        from .native_d2f_cache import (
            NativeD2FExactKVCacheStrategy,
            NativeD2FIdentityContext,
        )

        capacity = exact_d2f_kv_capacity(
            prompt_tokens=len(prompt_token_ids),
            max_tokens=max_tokens,
            backend_plan=self._compiled_backend_plan(graph),
        )
        spec = capacity.tensor_spec
        n_layers = capacity.layer_count
        stored_heads = spec.kv_heads
        head_dim = spec.head_dim
        compute_dtype_name = _canonical_compute_dtype_name(
            compute_dtype or self._config_compute_dtype_name()
        )
        # Storage precision is independent of weight/compute precision. Doctor
        # and admission share the sizing rule; capture enforces its real shape.
        kv_dtype_name = spec.dtype
        kv_quantization = spec.quantization_identity

        # Every K2-2 identity is generation-local.  Rotate before opening the new
        # one, and let a cleanup failure stop the generation rather than silently
        # retaining two managers.
        self._close_native_kv_manager()
        manager = open_manager(
            device="mlx",
            session_reserved_bytes=0,
            shared_cache_budget_bytes=capacity.shared_cache_budget_bytes,
            staging_headroom_bytes=capacity.staging_headroom_bytes,
        )
        generation = self._native_kv_generation + 1
        session = f"mlx-d2f-{generation}"
        try:
            self._bind_resident_source_digest()
            resident = self._weight_state
            weights_digest = (
                resident.source_digest
                if resident is not None and resident.source_digest
                else self._native_kv_digest(
                    "weights:",
                    {
                        "path": (
                            str(resident.source_path) if resident is not None else None
                        ),
                        "stat": (
                            resident.source_stat if resident is not None else None
                        ),
                        "ref": self.merged_weights_ref,
                    },
                )
            )
            policy = PolicyVersion.of(
                decode_strategy=decode_strategy,
                commit_policy=d2f_commit_policy,
                remask_policy=("enabled" if error_fixing_enabled else "none"),
                thresholds=(
                    confidence_threshold,
                    d2f_add_threshold,
                    d2f_activate_threshold,
                    gamma,
                ),
                block_size=block_size,
            )
            identity_context = NativeD2FIdentityContext(
                package_digest=self._native_kv_digest(
                    "package:",
                    {
                        "path": (
                            str(self.package_dir)
                            if self.package_dir is not None
                            else None
                        ),
                        "weights_ref": self.merged_weights_ref,
                    },
                ),
                weights_digest=weights_digest,
                tokenizer_identity=self._native_kv_digest(
                    "tokenizer:",
                    {
                        name: self.model_config.get(name)
                        for name in (
                            "vocab_size",
                            "bos_token_id",
                            "eos_token_id",
                            "pad_token_id",
                            "mask_token_id",
                        )
                    },
                ),
                graph_digest=self._native_kv_digest(
                    "graph:",
                    {
                        "graph": graph.to_dict(),
                        "compute_dtype": compute_dtype_name,
                    },
                ),
                attention_mask_kind=attention_mask_kind,
                model_geometry=(
                    ("head_dim", head_dim),
                    ("kv_heads", stored_heads),
                    ("layers", n_layers),
                ),
                layer_set=tuple(range(n_layers)),
                dtype=kv_dtype_name,
                quantization_identity=kv_quantization,
                policy_version=policy.digest,
                cache_salt=session,
                privacy_scope="session",
            )
            strategy = NativeD2FExactKVCacheStrategy(
                manager=manager,
                session=session,
                identity_context=identity_context,
                prompt_len=len(prompt_token_ids),
                block_size=int(block_size),
                window_mode=(
                    "tail"
                    if decode_strategy == "d2f_block_pipeline"
                    else "single_block"
                ),
                num_layers=n_layers,
                tensor_spec=spec,
            )
        except BaseException as primary:
            try:
                manager.shutdown()
            except BaseException as cleanup:
                primary.add_note(
                    f"shutting down a rejected native K/V manager failed: {cleanup}"
                )
            try:
                manager.close()
            except BaseException as cleanup:
                # Preserve the only retry path.  A local manager whose checked
                # destroy failed must not be handed to GC as if cleanup had
                # completed.
                self._native_kv_manager = manager
                primary.add_note(
                    f"destroying a rejected native K/V manager failed: {cleanup}"
                )
            raise
        self._native_kv_manager = manager
        self._native_kv_strategy = strategy
        self._native_kv_generation = generation
        return strategy

    def _config_compute_dtype_name(self) -> str:
        """The compute-dtype name for this engine: model_config override or fp32.

        fp32 is the fail-safe default (Phase D): byte-identical to the pre-bf16
        forward, so a package with no dtype pin (or one whose manifest regenerated
        and dropped the pin) computes in fp32 and can never silently break the HF
        parity gate. bf16 is an explicit opt-in — a family that passes the bf16
        parity gate AND wants the resident-RAM win pins ``mlx_compute_dtype="bf16"``
        in its manifest. The name is validated without importing the device SDK;
        conversion to an MLX dtype happens only when execution needs it.
        """
        if self.kernel_plan is not None:
            return self.kernel_plan.compute_dtype
        name = str((self.model_config or {}).get("mlx_compute_dtype", "fp32"))
        return _canonical_compute_dtype_name(name)

    def _forward_graph(
        self,
        *,
        uses_qk_norm: bool | None = None,
        uses_shifted_logits: bool | None = None,
    ) -> GraphIR:
        """The GraphIR the forward + decode selection read their flags from.

        Built once from ``self.model_config`` and cached on the engine. A test may
        inject ``self._graph`` to drive the forward independently of the arch
        string. The optional arguments are compatibility assertions for direct
        low-level callers; they can no longer override GraphIR.
        """
        if self._graph is None:
            self._graph = mlx_forward._graph_for_forward(self.model_config)
        graph = self._graph
        if uses_qk_norm is not None:
            if type(uses_qk_norm) is not bool:
                raise ValueError("uses_qk_norm must be a boolean or None")
            if uses_qk_norm != mlx_forward._ir_apply_qk_norm(graph):
                raise ValueError("uses_qk_norm contradicts the loaded GraphIR")
        if uses_shifted_logits is not None:
            if type(uses_shifted_logits) is not bool:
                raise ValueError("uses_shifted_logits must be a boolean or None")
            graph_shifted = graph.logits_convention == "dream_ar_shift"
            if uses_shifted_logits != graph_shifted:
                raise ValueError("uses_shifted_logits contradicts the loaded GraphIR")
        return graph

    def _compiled_backend_plan(
        self,
        graph: GraphIR | None = None,
    ) -> DenseTransformerPlan:
        """Return the single validated lowering consumed by optimized forwards."""

        resolved_graph = graph if graph is not None else self._forward_graph()
        if self.backend_plan is None:
            self.backend_plan = compile_dense_transformer_plan(resolved_graph)
        elif self.backend_plan.graph is not resolved_graph:
            raise ValueError("compiled backend plan does not belong to GraphIR")
        return self.backend_plan

    def _resolved_kernel_table(self) -> KernelTable:
        """Materialize the plan-owned per-op table once for this engine."""

        if self.kernel_table is not None:
            return self.kernel_table
        if self.kernel_plan is None:
            raise ValueError("MLX engine has no kernel dispatch plan")
        self.kernel_table = materialize_kernel_table(self.kernel_plan)
        return self.kernel_table

    def generate_unverified(self, *, prompt: str, max_tokens: int) -> str:
        del prompt, max_tokens
        return ""

    def memory_snapshot(self) -> dict[str, int]:
        """Read existing MLX allocations without initializing or synchronizing Metal."""
        if self._weight_state is None:
            return {}
        import mlx.core as mx

        return {
            "mlx_active_bytes": int(mx.get_active_memory()),
            "mlx_cache_bytes": int(mx.get_cache_memory()),
        }

    def weight_state_metrics(self) -> dict[str, int | bool | str]:
        state = self._weight_state
        return {
            "mlx_weight_resident": state is not None and bool(state.weights),
            "mlx_cleanup_pending": self._cleanup_pending,
            "mlx_weight_load_count": state.load_count if state is not None else 0,
            "mlx_weight_count": len(state.weights) if state is not None else 0,
            "mlx_compute_dtype": self._resident_dtype_name
            if self._resident_dtype_name is not None
            else self._config_compute_dtype_name(),
        }

    def generate_entropy(
        self,
        *,
        prompt_token_ids: list[int],
        max_tokens: int,
        mask_token_id: int,
        eos_token_id: int,
        steps: int = 512,
        eps: float = 1e-3,
        temperature: float = 0.2,
        top_p: float = 0.95,
        remasking: str = "low_confidence",
        alg_temp: float = 0.0,
        attention_mask_kind: str = "bidirectional",
        enable_layer0_kv_cache: bool | None = None,
        decode_strategy: str = "entropy_count",
        requires_prompt_block_alignment: bool | None = None,
        confidence_threshold: float = 0.9,
        block_size: int = 32,
        gamma: float = 0.1,
        uses_shifted_logits: bool | None = None,
        uses_qk_norm: bool | None = None,
        enable_prefix_kv_cache: bool = False,
        prefix_kv_refresh_interval: int = 0,
        enable_block_kv_cache: bool = False,
        block_kv_refresh_interval: int = 0,
        enable_d2f_exact_kv_cache: bool = False,
        # Exact K/V Slice Data Plane v0 call data. Internal and default-off;
        # package/model config deliberately does not expose these controls yet.
        exact_kv_slice_mode: Any = "off",
        exact_kv_slice_page_budget_bytes: int | None = None,
        # D2F-3 pipelined block scheduler thresholds (used ONLY by the opt-in
        # ``d2f_block_pipeline`` strategy; τ_conf is the shared
        # ``confidence_threshold``). Defaults = the reference eval settings
        # (``D2F-eval/eval_dream.sh``: τ_add 0.1, τ_act 0.95).
        d2f_add_threshold: float = 0.1,
        d2f_activate_threshold: float = 0.95,
        # Commit 5 (the NFE lever): the WITHIN-window commit rule for the d2f
        # pipeline. "threshold" (default) = today's per-position τ_conf, BYTE-
        # IDENTICAL; "entropy_budget" = the EB joint-uncertainty budget (reuses
        # ``gamma``), committing a larger SAFE set per forward. Opt-in; consumed
        # ONLY by decode_strategy="d2f_block_pipeline".
        d2f_commit_policy: str = "threshold",
        # K2-3 compatibility tombstones for the retired approximate row-K/V
        # side cache. Only these neutral defaults are accepted.
        d2f_skip_committed_rows: bool = False,
        d2f_skip_refresh_interval: int = 4,
        compute_dtype: str | None = None,
        # Phase F3 error model. ``None`` (the default) -> recovery OFF -> exactly
        # today's single decode pass (byte-identical). A dict enables the bounded
        # detect->remask->re-decode recovery loop (see RecoveryPolicy).
        error_fixing: dict | None = None,
        # MaskCanvas v0 (control-plane capture). OFF by default -> byte-identical:
        # attaches a behavior-neutral CanvasTraceObserver (+ a DecodeEventLog) to
        # the decode loop and builds a JSON-safe MaskCanvasTrace AFTER the run. The
        # observer is a PURE side-effect, so the committed tokens / forward count /
        # finish reason are unchanged whether it is ON or OFF.
        trace_canvas: bool = False,
        # Snapshot save/restore v0. When set, the engine writes a manifest-last
        # JSON snapshot artifact after decode. Saving forces the same behavior-
        # neutral trace capture used by trace_canvas, but does not expose the trace
        # on the returned MLXGenerationResult unless trace_canvas=True.
        save_snapshot_path: str | Path | None = None,
        # Auto/True use BackendRunner.execute_window: preserve the attention
        # context and project only the requested output rows where supported.
        # False retains the dense full_forward oracle for comparisons.
        window_exec: bool | None = None,
        # Active-Window Runtime (C2a, WS3). Optional callable sink receiving one
        # WindowExecutionPlan per decode step (BEFORE the forward). When set, the
        # decode loop emits a plan each step via this sink (the same as
        # DiffusionIterationManager.window_plan_sink). Default None -> byte-identical:
        # no plan is emitted, no classify_window call runs in the hot path.
        window_plan_sink: Any = None,
        # K2-3 compatibility tombstones. Only these neutral defaults are
        # accepted; the research strategy no longer has product wiring.
        enable_delayed_kv: bool = False,
        delayed_kv_scope: str = "layer0",
        delayed_kv_max_stale_steps: int | None = None,
        delayed_kv_refresh_interval: int | None = None,
        delayed_kv_upper_layer_range: "tuple[int, int] | None" = None,
        delayed_kv_stale_epoch_budget: "int | None" = None,
        # Retired canvas/cross-store compatibility tombstones. Snapshot capture
        # remains supported and is independent of these former K/V owners.
        enable_canvas_cache: bool = False,
        enable_cross_request_cache: bool = False,
        enable_cross_session_cache: bool = False,
        cross_session_cache_root: str | Path | None = None,
        canvas_cache_budget_bytes: int | None = None,
    ) -> MLXGenerationResult:
        validate_generation_length(max_tokens, field_name="max_tokens")
        if requires_prompt_block_alignment is None:
            requires_prompt_block_alignment = decode_strategy == "bd3_fixed_quota"
        if max_tokens:
            validate_prompt_block_alignment(
                prompt_length=len(prompt_token_ids),
                block_size=block_size,
                required=requires_prompt_block_alignment,
            )
        _validate_retired_kv_direct_config(
            exact_kv_slice_mode=exact_kv_slice_mode,
            exact_kv_slice_page_budget_bytes=exact_kv_slice_page_budget_bytes,
            d2f_skip_committed_rows=d2f_skip_committed_rows,
            d2f_skip_refresh_interval=d2f_skip_refresh_interval,
            enable_delayed_kv=enable_delayed_kv,
            delayed_kv_scope=delayed_kv_scope,
            delayed_kv_max_stale_steps=delayed_kv_max_stale_steps,
            delayed_kv_refresh_interval=delayed_kv_refresh_interval,
            delayed_kv_upper_layer_range=delayed_kv_upper_layer_range,
            delayed_kv_stale_epoch_budget=delayed_kv_stale_epoch_budget,
            enable_canvas_cache=enable_canvas_cache,
            canvas_cache_budget_bytes=canvas_cache_budget_bytes,
            enable_cross_request_cache=enable_cross_request_cache,
            enable_cross_session_cache=enable_cross_session_cache,
            cross_session_cache_root=cross_session_cache_root,
        )
        if enable_layer0_kv_cache is None:
            # Preserve the historical default for ordinary decodes while
            # making the native exact-D2F regime single-owner by default.
            enable_layer0_kv_cache = not (
                enable_d2f_exact_kv_cache or decode_strategy in {"bd3_fixed_quota", "fixed_quota"}
            )
        elif type(enable_layer0_kv_cache) is not bool:
            raise ValueError(
                "enable_layer0_kv_cache must be a bool or None; got "
                f"{enable_layer0_kv_cache!r}"
            )
        if alg_temp not in {None, 0.0}:
            raise ValueError(
                "mlx_local entropy generation currently supports alg_temp=0.0 only"
            )
        if decode_strategy not in {
            "entropy_count",
            "confidence_threshold",
            "block_threshold",
            "entropy_budget",
            "d2f_block_pipeline",
            "dilated_unmask",
            "bd3_fixed_quota",
            "fixed_quota",
        }:
            raise ValueError(
                "decode_strategy must be 'entropy_count', 'confidence_threshold', "
                "'block_threshold', 'entropy_budget', 'd2f_block_pipeline', "
                "'dilated_unmask', 'fixed_quota', or 'bd3_fixed_quota', "
                f"got {decode_strategy!r}"
            )
        if enable_d2f_exact_kv_cache and decode_strategy not in {
            "block_threshold",
            "d2f_block_pipeline",
        }:
            raise ValueError(
                "enable_d2f_exact_kv_cache=True requires a decode strategy "
                "with an in-order block lifecycle: 'block_threshold' (D2F-2) "
                "or 'd2f_block_pipeline' (D2F-3); got "
                f"{decode_strategy!r}"
            )
        if decode_strategy == "fixed_quota":
            validate_fixed_quota_sampling_contract(
                temperature=temperature, top_p=top_p, remasking=remasking
            )
            if attention_mask_kind not in {"bidirectional", "full"}:
                raise ValueError("fixed_quota requires bidirectional attention")
            if (
                enable_layer0_kv_cache or enable_prefix_kv_cache
                or enable_block_kv_cache or enable_d2f_exact_kv_cache
                or (error_fixing and error_fixing.get("enabled", False))
            ):
                raise ValueError("fixed_quota requires caches and error-fixing recovery off")
        if decode_strategy == "bd3_fixed_quota":
            validate_bd3_sampling_contract(
                temperature=temperature, top_p=top_p, remasking=remasking
            )
            if attention_mask_kind != "d2f_hybrid":
                raise ValueError(
                    "decode_strategy='bd3_fixed_quota' requires "
                    "attention_mask_kind='d2f_hybrid'"
                )
            if (
                enable_layer0_kv_cache
                or enable_prefix_kv_cache
                or enable_block_kv_cache
                or enable_d2f_exact_kv_cache
            ):
                raise ValueError(
                    "decode_strategy='bd3_fixed_quota' is admitted only for "
                    "the no-cache path"
                )
            if (
                error_fixing
                and bool(error_fixing.get("enabled", False))
                and not enable_d2f_exact_kv_cache
            ):
                raise ValueError(
                    "decode_strategy='bd3_fixed_quota' does not yet support "
                    "error-fixing recovery"
                )
        # D2F-3 fail-closed boundaries: the pipelined scheduler's lifecycle and
        # the exactness thesis live in the d2f_hybrid block-causal mask, so any
        # other mask kind refuses to run; the approximate caches assume
        # bidirectional attention (contradictory). Recovery can compose with
        # in-order freezing only through the exact owner's frozen-position view.
        if decode_strategy == "d2f_block_pipeline":
            if attention_mask_kind != "d2f_hybrid":
                raise ValueError(
                    "decode_strategy='d2f_block_pipeline' requires "
                    "attention_mask_kind='d2f_hybrid'; got "
                    f"{attention_mask_kind!r}"
                )
            if enable_block_kv_cache or enable_prefix_kv_cache:
                raise ValueError(
                    "decode_strategy='d2f_block_pipeline' cannot run with the "
                    "approximate prefix/block KV caches (they assume "
                    "bidirectional attention); use the exact d2f KV cache "
                    "(enable_d2f_exact_kv_cache) or no cache"
                )
            validate_d2f_recovery_contract(
                decode_strategy=decode_strategy,
                error_fixing_enabled=bool(error_fixing and error_fixing.get("enabled", False)),
                exact_kv_enabled=enable_d2f_exact_kv_cache,
            )
            if d2f_commit_policy not in {"threshold", "entropy_budget", "dilated"}:
                raise ValueError(
                    "d2f_commit_policy must be 'threshold' (default, "
                    "byte-identical), 'entropy_budget', or 'dilated', got "
                    f"{d2f_commit_policy!r}"
                )
        elif d2f_commit_policy != "threshold":
            # Fail-closed: the within-window commit rules (entropy_budget,
            # dilated) are meaningless without the pipelined scheduler that owns
            # the active window. Only "threshold" (the byte-identical default)
            # is valid for non-D2F strategies.
            raise ValueError(
                f"d2f_commit_policy={d2f_commit_policy!r} requires "
                "decode_strategy='d2f_block_pipeline' (the scheduler whose "
                f"active window it selects within); got {decode_strategy!r}"
            )
        # Thin orchestrator over the Diffusion Runtime Layer: build the
        # MaskState, pick the DecodingStrategy from ``decode_strategy``, run the
        # DiffusionIterationManager, then map its result back onto the legacy
        # MLXGenerationResult. The forward (incl. layer-0 KV cache and the
        # Dream/MDLM logit shift) and the stage timing stay here in mlx_dream;
        # the manager calls into them via the injected callables/hooks, so the
        # model-agnostic decode loop never imports MLX.
        if enable_d2f_exact_kv_cache:
            # Approximate caches remain mutually exclusive with native exact
            # ownership. Retired owner knobs were rejected above for every mode.
            incompatible = []
            if enable_layer0_kv_cache:
                incompatible.append("enable_layer0_kv_cache")
            if enable_prefix_kv_cache:
                incompatible.append("enable_prefix_kv_cache")
            if enable_block_kv_cache:
                incompatible.append("enable_block_kv_cache")
            if incompatible:
                raise ValueError(
                    "the native exact-D2F owner does not yet admit "
                    + ", ".join(incompatible)
                    + "; refusing a dual-owner fallback"
                )
        elif self._native_kv_manager is not None:
            # A completed exact decode deliberately leaves its manager attached
            # until the engine lifetime boundary so checked cleanup remains
            # retryable.  A later non-exact decode is another lifetime boundary:
            # it must retire that owner before admitting unrelated work.  Without
            # this check a direct RuntimeHandle user could continue decoding after
            # close() reported native cleanup debt, because only the next EXACT
            # generation used to retry the manager.
            self._close_native_kv_manager()
        # Resolve structural forward semantics before allocating weights or K/V.
        # Compatibility assertions from low-level callers may confirm GraphIR,
        # but cannot change it.
        graph = self._forward_graph(
            uses_qk_norm=uses_qk_norm,
            uses_shifted_logits=uses_shifted_logits,
        )
        self._compiled_backend_plan(graph)
        weights = self._resident_weights(
            compute_dtype=compute_dtype,
            require_source_identity=save_snapshot_path is not None,
        )
        native_d2f_strategy = None
        if enable_d2f_exact_kv_cache:
            native_d2f_strategy = self._new_native_d2f_strategy(
                prompt_token_ids=prompt_token_ids,
                max_tokens=max_tokens,
                block_size=block_size,
                decode_strategy=decode_strategy,
                attention_mask_kind=attention_mask_kind,
                compute_dtype=compute_dtype,
                graph=graph,
                d2f_commit_policy=d2f_commit_policy,
                confidence_threshold=confidence_threshold,
                d2f_add_threshold=d2f_add_threshold,
                d2f_activate_threshold=d2f_activate_threshold,
                gamma=gamma,
                error_fixing_enabled=bool(
                    error_fixing and error_fixing.get("enabled", False)
                ),
            )
        snapshot_identity = None
        if save_snapshot_path is not None:
            from edllm.backends.cache_identity import PolicyVersion
            from edllm.backends.snapshot_store import build_snapshot_identity

            resident_state = self._weight_state
            if resident_state is None or not resident_state.source_digest:
                raise ValueError("snapshot_resident_weights_digest_required")
            if resident_state._loader_provenance is not _MLX_WEIGHT_LOADER_PROVENANCE:
                raise ValueError("snapshot_resident_weights_provenance_required")
            if resident_state.source_stat is None:
                raise ValueError("snapshot_resident_weights_stat_required")
            snapshot_policy = PolicyVersion.of(
                decode_strategy=decode_strategy,
                commit_policy=(
                    d2f_commit_policy
                    if decode_strategy == "d2f_block_pipeline"
                    else decode_strategy
                ),
                remask_policy="enabled" if error_fixing else "none",
                thresholds=(
                    confidence_threshold,
                    d2f_add_threshold,
                    d2f_activate_threshold,
                    gamma,
                ),
                block_size=block_size,
            )
            snapshot_identity = build_snapshot_identity(
                package_dir=self.package_dir,
                merged_weights_ref=self.merged_weights_ref,
                model_config=self.model_config,
                policy_version=snapshot_policy,
                resident_source_path=resident_state.source_path,
                resident_source_digest=resident_state.source_digest,
                resident_source_stat=resident_state.source_stat,
            )
        timing_metrics = _new_mlx_timing_metrics()
        kv_cache_state = (
            _new_layer0_kv_cache_state(
                self.model_config,
                attention_mask_kind=attention_mask_kind,
            )
            if enable_layer0_kv_cache
            else None
        )
        # Approximate prefix/block K/V or the native exact-D2F strategy is
        # selected by
        # cache_registry.select_state_cache_manager (the focused selection
        # registry; see its docstring for the per-strategy semantics). OFF by
        # default -> returns None -> the forward is byte-identical. Retired K/V
        # controls were already checked as exact neutral tombstones.
        state_cache_manager = cache_registry.select_state_cache_manager(
            enable_d2f_exact_kv_cache=enable_d2f_exact_kv_cache,
            enable_block_kv_cache=enable_block_kv_cache,
            enable_prefix_kv_cache=enable_prefix_kv_cache,
            prompt_len=len(prompt_token_ids),
            block_size=block_size,
            decode_strategy=decode_strategy,
            d2f_skip_committed_rows=d2f_skip_committed_rows,
            d2f_skip_refresh_interval=d2f_skip_refresh_interval,
            block_kv_refresh_interval=block_kv_refresh_interval,
            prefix_kv_refresh_interval=prefix_kv_refresh_interval,
            enable_delayed_kv=enable_delayed_kv,
            delayed_kv_max_stale_steps=delayed_kv_max_stale_steps,
            delayed_kv_refresh_interval=delayed_kv_refresh_interval,
            delayed_kv_scope=delayed_kv_scope,
            delayed_kv_stale_epoch_budget=delayed_kv_stale_epoch_budget,
            delayed_kv_upper_layer_range=delayed_kv_upper_layer_range,
            exact_strategy=native_d2f_strategy,
        )

        capture_canvas = bool(trace_canvas or save_snapshot_path is not None)
        try:
            session = DecodeRequest(
                decode_strategy=decode_strategy,
                steps=int(steps),
                block_size=block_size,
                temperature=temperature,
                top_p=top_p,
                eps=eps,
                confidence_threshold=confidence_threshold,
                requires_prompt_block_alignment=requires_prompt_block_alignment,
                gamma=gamma,
                d2f_add_threshold=d2f_add_threshold,
                d2f_activate_threshold=d2f_activate_threshold,
                d2f_commit_policy=d2f_commit_policy,
                recovery=RecoveryPolicy.from_config(error_fixing),
            ).prepare(
                prefix_token_ids=prompt_token_ids,
                token_count=max_tokens,
                mask_token_id=mask_token_id,
                eos_token_id=eos_token_id,
                kv_cache_state=kv_cache_state,
                state_cache_manager=state_cache_manager,
                window_exec=window_exec,
                capture_canvas=capture_canvas,
            )
            decoded = self._run_decode_session(
                session,
                weights=weights,
                attention_mask_kind=attention_mask_kind,
                kv_cache_state=kv_cache_state,
                timing_metrics=timing_metrics,
                graph=graph,
                state_cache_manager=state_cache_manager,
                window_plan_sink=window_plan_sink,
            )
        except BaseException as primary:
            if native_d2f_strategy is not None:
                try:
                    self._close_native_kv_decode()
                except BaseException as cleanup:
                    primary.add_note(
                        f"closing the native K/V decode after failure failed: {cleanup}"
                    )
            raise

        kv_manager_trace = None
        native_kv_metrics: dict[str, int] = {}
        native_backend_metrics: dict[str, int] = {}
        if native_d2f_strategy is not None:
            # Close the SESSION before evidence is read, so the trace proves
            # release_lease + close_session rather than reporting an in-flight
            # lifecycle as a successful generation.
            self._close_native_kv_decode()
            kv_manager_trace = native_d2f_strategy.manager_trace()
            native_kv_metrics = native_d2f_strategy.manager_metrics()
            native_backend_metrics = native_d2f_strategy.backend_metrics()
        timing_metrics["mlx_timing_measured_forward_count"] = decoded.forward_count
        # active-tokens-per-forward: the Pareto-calibration FLOP-waste metric (mean
        # forwarded positions / forward). ~canvas for full-canvas strategies, < canvas
        # once the active window shrinks (d2f). Sibling of committed_per_forward.
        _fwd = int(decoded.forward_count or 0)
        timing_metrics["mlx_active_tokens_per_forward"] = (
            int(timing_metrics.get("mlx_timing_active_tokens_total", 0)) / _fwd
            if _fwd
            else 0.0
        )
        # MaskCanvas v0: build the JSON-safe trace from the observer + event log +
        # the final canvas. Built when tracing OR snapshot-saving; returned only
        # when trace_canvas=True (saving should not alter the result surface).
        canvas_trace = None
        if capture_canvas:
            canvas = mask_canvas.MaskCanvas(
                session.state,
                observer=session.trace_observer,
                event_log=session.event_log,
                commit_policy=getattr(session.strategy, "commit_policy", None),
                decode_strategy=decode_strategy,
            )
            canvas_trace = canvas.trace(
                forward_count=decoded.forward_count,
                finish_reason=decoded.finish_reason,
            )
        kv_cache_metrics = _merge_cache_metrics(
            kv_cache_state,
            state_cache_manager,
        )
        if native_d2f_strategy is not None:
            kv_cache_metrics["kv_owner"] = "native.core"
            kv_cache_metrics["kv_manager_bytes"] = int(
                native_kv_metrics["shared_resident_bytes"]
            )
            kv_cache_metrics.update(
                {
                    f"kv_manager_{name}": int(value)
                    for name, value in native_kv_metrics.items()
                }
            )
            kv_cache_metrics.update(native_backend_metrics)
        if save_snapshot_path is not None:
            from edllm.backends.snapshot_store import (
                SnapshotBundle,
                SnapshotCacheState,
                SnapshotStore,
            )

            if canvas_trace is None:
                raise RuntimeError(
                    "snapshot save requested but no canvas trace was captured"
                )
            if snapshot_identity is None:
                raise RuntimeError("snapshot durable identity was not resolved")
            SnapshotStore(root=save_snapshot_path).save(
                SnapshotBundle(
                    identity=snapshot_identity,
                    canvas=canvas_trace.to_dict(full=True),
                    resume={
                        "prompt_token_ids": list(prompt_token_ids),
                        "visible_generated_token_ids": list(
                            decoded.visible_generated_token_ids
                        ),
                        "max_tokens": int(max_tokens),
                        "decode_strategy": decode_strategy,
                        "attention_mask_kind": attention_mask_kind,
                        "block_size": int(block_size),
                        "confidence_threshold": float(confidence_threshold),
                    },
                    cache=SnapshotCacheState(
                        exact_only=True,
                        canvas_cache_enabled=False,
                        cross_session_cache_enabled=False,
                        cross_session_cache_root=None,
                        immutable_entry_count=0,
                        page_pool_stats={},
                        metrics=dict(kv_cache_metrics),
                    ),
                )
            )
        return MLXGenerationResult(
            visible_generated_token_ids=decoded.visible_generated_token_ids,
            iteration_count=decoded.iteration_count,
            forward_count=decoded.forward_count,
            decode_loop_kind=decoded.decode_loop_kind,
            entropy_used=decoded.entropy_used,
            committed_token_count=decoded.committed_token_count,
            hit_eos=decoded.hit_eos,
            remaining_mask_count=decoded.remaining_mask_count,
            finish_reason=decoded.finish_reason,
            kv_cache_metrics=kv_cache_metrics,
            timing_metrics=timing_metrics,
            committed_per_forward=decoded.committed_per_forward,
            remask_count=decoded.remask_count,
            corrected_positions=decoded.corrected_positions,
            recovery_rounds=decoded.recovery_rounds,
            wrong_commit_count=decoded.wrong_commit_count,
            churned_positions=decoded.churned_positions,
            quality_delta=decoded.quality_delta,
            canvas_trace=canvas_trace if trace_canvas else None,
            kv_manager_trace=kv_manager_trace,
        )

    def generate_infill(
        self,
        *,
        prefix_token_ids: list[int],
        suffix_token_ids: list[int],
        max_fill_tokens: int,
        mask_token_id: int,
        eos_token_id: int,
        steps: int = 512,
        eps: float = 1e-3,
        temperature: float = 0.2,
        top_p: float = 0.95,
        alg_temp: float = 0.0,
        attention_mask_kind: str = "bidirectional",
        enable_layer0_kv_cache: bool = True,
        decode_strategy: str = "block_threshold",
        confidence_threshold: float = 0.9,
        block_size: int = 32,
        gamma: float = 0.1,
        uses_shifted_logits: bool | None = None,
        uses_qk_norm: bool | None = None,
        compute_dtype: str | None = None,
        # Active-Window Runtime (Task B2, WS2; tri-state since 3A-B). Mirrors
        # generate_entropy: auto/True use output-row selection; False retains
        # the dense oracle. The fixed infill suffix stays in attention context.
        window_exec: bool | None = None,
    ) -> MLXInfillResult:
        """dLLM-native infill / span-fill: fill ``max_fill_tokens`` masked holes
        BETWEEN a fixed prefix and a fixed suffix.

        Builds ``x = prefix + [mask]*max_fill_tokens + suffix`` and runs the SAME
        Diffusion Runtime Layer (MaskState + DecodingStrategy +
        DiffusionIterationManager + the layer-0 KV cache + the Dream/MDLM logit
        shift) the open-ended generation uses -- additively, via
        :meth:`diffusion_runtime.MaskState.for_infill`, which restricts
        maskability to the middle holes so the bidirectional forward fills them
        respecting BOTH sides and the fixed suffix is never overwritten. The
        result is finalized with :meth:`MaskState.filled_span` (NO first-EOS
        truncation -- the span is bounded).
        """
        validate_generation_length(max_fill_tokens, field_name="max_fill_tokens")
        if decode_strategy == "bd3_fixed_quota":
            raise ValueError(
                "This model's fixed-quota block-diffusion policy does not support "
                "infill or edit; use generation or an infill-capable model"
            )
        if alg_temp not in {None, 0.0}:
            raise ValueError("mlx_local infill currently supports alg_temp=0.0 only")
        if decode_strategy not in {
            "entropy_count",
            "confidence_threshold",
            "block_threshold",
            "entropy_budget",
            "fixed_quota",
        }:
            raise ValueError(
                "decode_strategy must be 'entropy_count', 'confidence_threshold', "
                f"'block_threshold', 'fixed_quota', or 'entropy_budget', got {decode_strategy!r}"
            )
        if decode_strategy == "fixed_quota":
            validate_fixed_quota_sampling_contract(
                temperature=temperature, top_p=top_p, remasking="low_confidence"
            )
            if attention_mask_kind not in {"bidirectional", "full"} or enable_layer0_kv_cache:
                raise ValueError("fixed_quota infill requires bidirectional attention and caches off")

        # Infill never consumes the exact-D2F manager.  Retire a manager left by
        # an earlier exact generation before starting new work, and propagate any
        # cleanup debt instead of running beside a closed-but-not-destroyed owner.
        self._close_native_kv_manager()
        # Settle the executable structure before loading weights or allocating
        # cache state. A malformed or contradictory graph must fail cheaply.
        graph = self._forward_graph(
            uses_qk_norm=uses_qk_norm,
            uses_shifted_logits=uses_shifted_logits,
        )
        self._compiled_backend_plan(graph)
        # An empty edit/infill still validates its execution contract, but it
        # has no forward and must not load a model just to preserve the context.
        weights = self._resident_weights(compute_dtype=compute_dtype) if max_fill_tokens else {}
        timing_metrics = _new_mlx_timing_metrics()
        kv_cache_state = (
            _new_layer0_kv_cache_state(
                self.model_config,
                attention_mask_kind=attention_mask_kind,
            )
            if enable_layer0_kv_cache
            else None
        )

        session = DecodeRequest(
            decode_strategy=decode_strategy,
            steps=int(steps),
            block_size=block_size,
            temperature=temperature,
            top_p=top_p,
            eps=eps,
            confidence_threshold=confidence_threshold,
            gamma=gamma,
        ).prepare(
            prefix_token_ids=prefix_token_ids,
            suffix_token_ids=suffix_token_ids,
            token_count=max_fill_tokens,
            mask_token_id=mask_token_id,
            eos_token_id=eos_token_id,
            kv_cache_state=kv_cache_state,
            window_exec=window_exec,
        )
        decoded = self._run_decode_session(
            session,
            weights=weights,
            attention_mask_kind=attention_mask_kind,
            kv_cache_state=kv_cache_state,
            timing_metrics=timing_metrics,
            graph=graph,
        )
        timing_metrics["mlx_timing_measured_forward_count"] = decoded.forward_count
        span = session.state.filled_span()
        # Infill fills a bounded span; an EOS inside the span is kept and the
        # finish reason is whether every hole resolved ("stop") or the step cap
        # left a hole masked ("length").
        finish_reason = "length" if span["remaining_mask_count"] > 0 else "stop"
        return MLXInfillResult(
            fill_token_ids=span["fill_token_ids"],
            full_token_ids=span["full_token_ids"],
            iteration_count=decoded.iteration_count,
            forward_count=decoded.forward_count,
            committed_token_count=decoded.committed_token_count,
            decode_loop_kind=decoded.decode_loop_kind,
            entropy_used=decoded.entropy_used,
            remaining_mask_count=span["remaining_mask_count"],
            finish_reason=finish_reason,
            # Infill keeps the byte-exact full forward (its middle-hole layout does
            # not match the trailing-prefix cache assumption), so the prefix-cache
            # block is the neutral OFF block; the layer-0 KV metrics are unchanged.
            kv_cache_metrics=_merge_cache_metrics(kv_cache_state, None),
            timing_metrics=timing_metrics,
        )

    def _run_decode_session(
        self,
        session: DecodeSession,
        *,
        weights: Any,
        attention_mask_kind: str,
        kv_cache_state: Any | None,
        timing_metrics: dict[str, float | int | bool],
        graph: GraphIR,
        state_cache_manager: Any | None = None,
        window_plan_sink=None,
    ) -> diffusion_runtime.DiffusionIterationResult:
        """Bind the resolved MLX execution path to shared request control state.

        Resolve kernels once for both full and window forward. Graph semantics,
        resident weights and K/V ownership remain with the engine and runner.
        """
        kernels = self._resolved_kernel_table()
        return session.run(
            forward_fn=self._make_forward_fn(
                weights=weights,
                attention_mask_kind=attention_mask_kind,
                kv_cache_state=kv_cache_state,
                timing_metrics=timing_metrics,
                graph=graph,
                state_cache_manager=state_cache_manager,
                mask_token_id=session.state.mask_token_id,
                mask_geometry=session.mask_geometry,
                kernels=kernels,
            ),
            select_transform=lambda logits: mlx_forward._logits_for_selection(
                logits, logits_convention=graph.logits_convention
            ),
            scheduler_phase=lambda: _timing_phase(
                timing_metrics, "mlx_timing_scheduler_overhead_ms"
            ),
            select_phase=lambda: _timing_phase(
                timing_metrics, "mlx_timing_shift_select_ms"
            ),
            commit_phase=lambda: _timing_phase(timing_metrics, "mlx_timing_commit_ms"),
            execute_window_fn=(
                self._make_execute_window_fn(
                    weights=weights,
                    attention_mask_kind=attention_mask_kind,
                    kv_cache_state=kv_cache_state,
                    timing_metrics=timing_metrics,
                    graph=graph,
                    state_cache_manager=state_cache_manager,
                    mask_token_id=session.state.mask_token_id,
                    mask_geometry=session.mask_geometry,
                    kernels=kernels,
                )
                if session.use_execute_window
                else None
            ),
            window_plan_sink=window_plan_sink,
        )

    def _make_forward_fn(
        self,
        *,
        weights: Any,
        attention_mask_kind: str,
        kv_cache_state: Any | None,
        timing_metrics: dict[str, float | int | bool],
        graph: GraphIR,
        state_cache_manager: Any | None = None,
        mask_token_id: int | None = None,
        mask_geometry: Any | None = None,
        kernels: KernelTable | None = None,
    ):
        """Build the backend forward callable the manager drives.

        Delegates to the injected ``backend_runner`` (the MLX runner by default,
        a Metal runner if swapped in), threading the layer-0 KV cache, the
        approximate prefix KV cache (``state_cache_manager``), and the
        forward-stage timing through unchanged. The default ``MLXBackendRunner``
        dereferences ``_forward_full_logits_from_weights`` at call time, so the
        existing monkeypatch seam keeps working THROUGH the runner.

        When a block KV strategy is active the closure derives the still-masked
        frontier from the current ``token_ids`` (vs ``mask_token_id``) and passes
        it to the forward, so the forward can carve the block-only work. With no
        block strategy ``masked_positions`` stays ``None`` and the path is
        byte-identical. ``mask_geometry`` (D2F-3: the d2f_hybrid prompt_len /
        block_size) flows through as forward-call data; ``None`` everywhere else.
        ``kernels`` is the per-op KernelTable resolved at the entry point
        (M2-1); ``None`` lets the forward resolve the default from
        ``model_config`` — byte-identical either way today.
        """

        runner = self.backend_runner
        block_active = getattr(state_cache_manager, "block_kv", None) is not None
        backend_plan = self._compiled_backend_plan(graph)

        def forward_fn(token_ids):
            masked_positions = None
            if block_active and mask_token_id is not None:
                masked_positions = [
                    i for i, t in enumerate(token_ids) if int(t) == int(mask_token_id)
                ]
            return runner.full_forward(
                weights=weights,
                model_config=self.model_config,
                token_ids=token_ids,
                attention_mask_kind=attention_mask_kind,
                kv_cache_state=kv_cache_state,
                timing_metrics=timing_metrics,
                graph=graph,
                backend_plan=backend_plan,
                state_cache_manager=state_cache_manager,
                masked_positions=masked_positions,
                mask_geometry=mask_geometry,
                kernels=kernels,
            )

        return forward_fn

    def _make_execute_window_fn(
        self,
        *,
        weights: Any,
        attention_mask_kind: str,
        kv_cache_state: Any | None,
        timing_metrics: dict[str, float | int | bool],
        graph: "GraphIR",
        state_cache_manager: Any | None = None,
        mask_token_id: int | None = None,
        mask_geometry: Any | None = None,
        kernels: "KernelTable | None" = None,
    ):
        """Build the ``execute_window_fn`` callable the manager's B2 path drives.

        Mirrors ``_make_forward_fn`` in structure. Returns a closure
        ``(window_plan, token_ids) -> WindowExecutionResult`` that calls
        ``backend_runner.execute_window`` with all the runtime arguments bound.
        The convention shift (dream_ar_shift / raw) is applied INSIDE
        ``execute_window`` (by the parity wrapper in ``MLXBackendRunner``), so
        the routed path MUST NOT call ``select_transform`` / ``select_logits``
        on the result — doing so would double-shift Dream (finding 1 pin).

        ``state_cache_manager`` and ``mask_geometry`` default to ``None`` here;
        generate_infill (which has no prefix/block KV cache or d2f hybrid mask)
        passes them explicitly as None.

        author:Opus -- with a block KV strategy the still-masked frontier is
        derived here from the request's ``mask_token_id``, as ``_make_forward_fn``
        does. This path became the default without it, which left the forward to
        guess the mask id from ``model_config``; a config that names none made the
        frontier empty.
        """
        runner = self.backend_runner
        block_active = getattr(state_cache_manager, "block_kv", None) is not None
        backend_plan = self._compiled_backend_plan(graph)

        def execute_window_fn(window_plan, token_ids):
            masked_positions = None
            if block_active and mask_token_id is not None:
                masked_positions = [
                    i for i, t in enumerate(token_ids) if int(t) == int(mask_token_id)
                ]
            return runner.execute_window(
                plan=window_plan,
                weights=weights,
                model_config=self.model_config,
                token_ids=token_ids,
                attention_mask_kind=attention_mask_kind,
                kv_cache_state=kv_cache_state,
                timing_metrics=timing_metrics,
                graph=graph,
                backend_plan=backend_plan,
                state_cache_manager=state_cache_manager,
                masked_positions=masked_positions,
                mask_geometry=mask_geometry,
                kernels=kernels,
            )

        return execute_window_fn

    def _finalize_generation(
        self,
        *,
        x: list[int],
        prompt_token_ids: list[int],
        mask_token_id: int,
        eos_token_id: int,
        iteration_count: int,
        forward_count: int,
        committed_token_count: int,
        kv_cache_state: Any | None,
        timing_metrics: dict[str, float | int | bool],
        decode_loop_kind: str = "mlx_dream_masked_diffusion_entropy",
        entropy_used: bool = True,
    ) -> MLXGenerationResult:
        """Assemble an MLXGenerationResult from a finished working sequence.

        The AR-standard first-EOS truncation / remaining-mask / finish-reason
        logic now lives on :class:`diffusion_runtime.MaskState`; this delegates
        to it so there is a single source of truth, then layers on the
        engine-level metrics (timing, KV cache).
        """
        timing_metrics["mlx_timing_measured_forward_count"] = forward_count
        final = diffusion_runtime.MaskState(
            x=list(x),
            prompt_len=len(prompt_token_ids),
            mask_token_id=mask_token_id,
            eos_token_id=eos_token_id,
        ).finalize()
        return MLXGenerationResult(
            visible_generated_token_ids=final["visible_token_ids"],
            iteration_count=iteration_count,
            forward_count=forward_count,
            decode_loop_kind=decode_loop_kind,
            entropy_used=entropy_used,
            committed_token_count=committed_token_count,
            hit_eos=final["hit_eos"],
            remaining_mask_count=final["remaining_mask_count"],
            finish_reason=final["finish_reason"],
            kv_cache_metrics=kv_cache_state.metrics()
            if kv_cache_state is not None
            else mlx_cache.empty_metrics(),
            timing_metrics=timing_metrics,
        )

    def forward_full_logits(
        self,
        *,
        token_ids: list[int],
        attention_mask_kind: str = "bidirectional",
        mask_geometry: Any | None = None,
        compute_dtype: str | None = None,
    ):
        # The compute dtype is carried by the resident weights: ``compute_dtype``
        # override if given, else the engine's model_config default (bf16). The
        # HF logit-parity gate calls this with no override, so it exercises the
        # bf16 forward by default; pass compute_dtype="fp32" for the A/B baseline.
        # Stage 2: pass the engine's cached Graph IR so THIS standalone parity/debug
        # path is graph-driven exactly like the generation/infill paths — the seam
        # is uniformly fed the IR, never graph=None on one path only.
        graph = self._forward_graph()
        backend_plan = self._compiled_backend_plan(graph)
        weights = self._resident_weights(compute_dtype=compute_dtype)
        return self.backend_runner.full_forward(
            weights=weights,
            model_config=self.model_config,
            token_ids=token_ids,
            attention_mask_kind=attention_mask_kind,
            mask_geometry=mask_geometry,
            graph=graph,
            backend_plan=backend_plan,
            kernels=self._resolved_kernel_table(),
        )

    def _resident_weights(
        self,
        *,
        compute_dtype: Any = None,
        require_source_identity: bool = False,
    ) -> dict[str, Any]:
        """Resident weights loaded ONCE at the engine's compute dtype.

        The dtype is the explicit ``compute_dtype`` override if given, else the
        model_config default (bf16). Loading at that dtype is the throughput lever:
        bf16 halves the resident weight RAM + per-step bandwidth on batch-1 decode.
        If a later call requests a DIFFERENT dtype than the resident one, the
        weights are reloaded so the forward never mixes a stale dtype.

        Quant-1: this is also the weight-prep hook for opt-in linear
        quantization. When ``model_config`` carries ``mlx_linear_quant_bits``
        (8 or 4; default None == OFF == byte-identical fp), the 2D linear
        projection weights are quantized into resident ``QuantizedWeight``
        bundles HERE, once per load — a one-time load cost, never per forward.

        ``require_source_identity`` is enabled only for cross-session caching.
        Ordinary loads bind stat/provenance without a second full-file read;
        enabling cross-session later hashes that unchanged resident source once.
        """
        if self._cleanup_pending:
            raise RuntimeError("MLX engine close cleanup is pending; retry close() before execution")
        dtype_name = (
            str(compute_dtype)
            if compute_dtype is not None
            else self._config_compute_dtype_name()
        )
        # Canonicalize ('bf16' and 'bfloat16' are the same resident dtype).
        canonical = _canonical_compute_dtype_name(dtype_name)
        # Fail-closed quant-spec resolution (None == quantization OFF).
        quant_spec = _resolve_linear_quant_spec(self.model_config)
        quant_name = (
            "none" if quant_spec is None else f"int{quant_spec[0]}-g{quant_spec[1]}"
        )
        # Reload only when we have NO resident state, or when WE previously loaded
        # at a KNOWN-different dtype / quant spec. A weight state injected by a
        # test (dtype/quant not recorded) is honored as-is, so the spy/stub seams
        # keep working.
        needs_load = (
            self._weight_state is None
            or (
                self._resident_dtype_name is not None
                and self._resident_dtype_name != canonical
            )
            or (
                self._resident_quant_name is not None
                and self._resident_quant_name != quant_name
            )
        )
        if needs_load:
            artifact = self.weight_artifact
            if artifact is not None and artifact.format == SAFETENSORS_SHARDED_V1:
                if self.package_dir is None:
                    raise FileNotFoundError("MLXDreamEngine requires package_dir")
                new_state = _load_mlx_sharded_weight_state(
                    package_dir=self.package_dir,
                    artifact=artifact,
                    dtype=canonical,
                    bind_digest=require_source_identity,
                )
            elif artifact is None or artifact.format == NPZ_V1:
                with contextlib.ExitStack() as stack:
                    weights_handle, source_path, source_stat = self._enter_weights_file(
                        stack
                    )
                    new_state = _load_mlx_weight_state(
                        weights_handle,
                        source_path=source_path,
                        source_stat=source_stat,
                        dtype=canonical,
                        bind_digest=require_source_identity,
                    )
            else:  # WeightArtifactSpec rejects this; keep the backend fail-closed.
                raise ValueError(
                    f"mlx weight artifact format unsupported: {artifact.format!r}"
                )
            # I4 mlx half: a bias-declaring model missing its q/k/v biases fails
            # closed here, before the corrupt package ever reaches a forward.
            _validate_declared_biases_present(
                new_state.weights,
                self._forward_graph(),
            )
            if quant_spec is not None:
                _quantize_linear_weights(
                    new_state.weights,
                    bits=quant_spec[0],
                    group_size=quant_spec[1],
                )
            # Commit only AFTER quantization succeeded: a mid-reload failure
            # leaves the previous resident state and its recorded dtype/quant
            # names fully intact (the next call re-derives needs_load against
            # them, so a reverted spec correctly keeps the old resident state).
            self._weight_state = new_state
            self._resident_dtype_name = canonical
            self._resident_quant_name = quant_name
            self._weight_load_count += 1
            self._weight_state.load_count = self._weight_load_count
        if require_source_identity:
            self._bind_resident_source_digest()
        return self._weight_state.weights

    def _open_weights_file(self):
        if self.package_dir is None:
            raise FileNotFoundError("MLXDreamEngine requires package_dir")
        from edllm.package_artifacts import open_package_artifact

        return open_package_artifact(
            package_path=Path(self.package_dir),
            artifact_ref=self.merged_weights_ref,
            error_prefix="mlx",
            artifact_name="weights",
        )

    def _enter_weights_file(self, stack: contextlib.ExitStack):
        try:
            return stack.enter_context(self._open_weights_file())
        except ValueError as exc:
            if str(exc) == "mlx_weights_artifact_missing":
                raise FileNotFoundError(self.merged_weights_ref) from exc
            raise

    def _bind_resident_source_digest(self) -> None:
        state = self._weight_state
        if state is None or state.source_digest:
            return
        if state._loader_provenance is not _MLX_WEIGHT_LOADER_PROVENANCE:
            return
        if state.source_stat is None:
            raise ValueError("snapshot_resident_weights_stat_required")

        with contextlib.ExitStack() as stack:
            weights_handle, source_path, source_stat = self._enter_weights_file(stack)
            if Path(state.source_path).absolute() != source_path:
                raise ValueError("snapshot_resident_weights_source_mismatch")
            current_identity = _weight_source_stat(source_stat)
            if tuple(state.source_stat) != current_identity:
                raise ValueError("snapshot_resident_weights_file_changed")
            weights_handle.seek(0)
            source_digest = _sha256_file_handle(weights_handle)
            if (
                _weight_source_stat(os.fstat(weights_handle.fileno()))
                != current_identity
            ):
                raise ValueError("snapshot_resident_weights_file_changed")
        artifact = self.weight_artifact
        if artifact is not None and artifact.digest and source_digest != artifact.digest:
            raise ValueError("mlx_weights_digest_mismatch")
        state.source_digest = source_digest

    def forward_layer0_trace(
        self,
        *,
        prompt_token_ids: list[int],
        block_token_ids: list[int],
        mask_token_id: int,
        attention_mask_kind: str = "bidirectional",
        compute_dtype: str | None = None,
    ) -> dict[str, dict]:
        if self._cleanup_pending:
            raise RuntimeError("MLX engine close cleanup is pending; retry close() before execution")
        del mask_token_id
        mask_geometry = None
        if attention_mask_kind == "d2f_hybrid":
            model_config = self.model_config or {}
            block_size = model_config.get("block_size")
            if block_size is None:
                block_size = model_config.get("mlx_diffusion_block_size")
            if type(block_size) is not int or block_size <= 0:
                raise ValueError("layer0_trace_d2f_block_size_required")
            mask_geometry = {
                "prompt_len": len(prompt_token_ids),
                "block_size": block_size,
            }
        weights = (
            self._weight_state.weights
            if compute_dtype is None and self._weight_state is not None
            else self._resident_weights(compute_dtype=compute_dtype)
        )
        return self.backend_runner.layer0_trace(
            weights=weights,
            prompt_token_ids=prompt_token_ids,
            block_token_ids=block_token_ids,
            attention_mask_kind=attention_mask_kind,
            mask_geometry=mask_geometry,
            backend_plan=self._compiled_backend_plan(),
            kernels=self._resolved_kernel_table(),
        )


def compare_layer0_trace(
    candidate: dict[str, dict],
    reference: dict[str, dict],
    *,
    atol: float,
    rtol: float,
) -> dict:
    import numpy as np

    comparisons: dict[str, dict] = {}
    missing_stages: list[str] = []
    first_divergent_stage: str | None = None
    for stage in LAYER0_TRACE_STAGE_ORDER:
        if stage not in candidate or stage not in reference:
            missing_stages.append(stage)
            comparisons[stage] = {"passed": False, "shape_match": False}
            if first_divergent_stage is None:
                first_divergent_stage = stage
            continue

        candidate_rows = np.asarray(candidate[stage]["rows"], dtype=np.float32)
        reference_rows = np.asarray(reference[stage]["rows"], dtype=np.float32)
        shape_match = list(candidate_rows.shape) == list(reference_rows.shape)
        if not shape_match:
            comparisons[stage] = {
                "passed": False,
                "shape_match": False,
                "candidate_shape": list(candidate_rows.shape),
                "reference_shape": list(reference_rows.shape),
            }
            if first_divergent_stage is None:
                first_divergent_stage = stage
            continue

        diff = np.abs(candidate_rows - reference_rows)
        max_abs_diff = float(diff.max()) if diff.size else 0.0
        reference_scale = (
            float(np.max(np.abs(reference_rows))) if reference_rows.size else 0.0
        )
        tolerance = float(atol + rtol * reference_scale)
        passed = max_abs_diff <= tolerance
        comparisons[stage] = {
            "passed": bool(passed),
            "shape_match": True,
            "max_abs_diff": max_abs_diff,
            "tolerance": tolerance,
        }
        if not passed and first_divergent_stage is None:
            first_divergent_stage = stage

    return {
        "first_divergent_stage": first_divergent_stage,
        "missing_stages": missing_stages,
        "comparisons": comparisons,
    }


# --- State-cache selection (relocated) ------------------------------------
# The state-cache *selection* factory and its strategy constructors now live in
# the focused ``cache_registry`` module (so future KV-cache work stops piling
# into this monolith). The module-level names below are DEPRECATED aliases kept
# for backward compatibility: several tests reference these old private names on
# ``mlx_dream`` directly (selection unit tests, the d2f-exact fail-closed test,
# the layer-0 stateful-weights test). They are exact aliases of the public
# ``cache_registry`` entry points — behavior is byte-identical. New code should
# call ``cache_registry.*`` directly.
_new_layer0_kv_cache_state = cache_registry.new_layer0_kv_cache_state
_new_prefix_kv_state_cache_manager = cache_registry.new_prefix_kv_state_cache_manager
_new_block_kv_state_cache_manager = cache_registry.new_block_kv_state_cache_manager
_new_d2f_exact_kv_state_cache_manager = (
    cache_registry.new_d2f_exact_kv_state_cache_manager
)
_select_state_cache_manager = cache_registry.select_state_cache_manager


def _merge_cache_metrics(
    kv_cache_state: Any | None,
    state_cache_manager: Any | None,
):
    """Combine layer-0, prefix, block, and native exact-D2F metrics.

    The layer-0 KV cache stays the prior neutral strategy; the prefix/block/d2f
    caches are the StateCacheManager strategies. All four metric blocks are
    emitted so a `--json` dump shows exactly which caches ran. With none enabled
    this is the empty-metrics block plus the neutral prefix + block + d2f blocks.
    """
    from . import mlx_state_cache

    metrics = dict(
        kv_cache_state.metrics()
        if kv_cache_state is not None
        else mlx_cache.empty_metrics()
    )
    metrics.update(
        state_cache_manager.prefix_metrics()
        if state_cache_manager is not None
        else mlx_state_cache.empty_prefix_metrics()
    )
    metrics.update(
        state_cache_manager.block_metrics()
        if state_cache_manager is not None
        else mlx_state_cache.empty_block_metrics()
    )
    metrics.update(
        state_cache_manager.d2f_metrics()
        if state_cache_manager is not None
        else mlx_state_cache.empty_d2f_metrics()
    )
    return metrics


def _rope_signature(model_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "theta": float(model_config["rope_theta"]),
        "scaling": model_config.get("rope_scaling"),
    }


def _new_mlx_timing_metrics() -> dict[str, float | int | bool]:
    return {
        "mlx_timing_enabled": True,
        "mlx_timing_forward_total_ms": 0.0,
        "mlx_timing_layer0_forward_ms": 0.0,
        "mlx_timing_upper_layers_forward_ms": 0.0,
        "mlx_timing_final_norm_lm_head_ms": 0.0,
        "mlx_timing_scheduler_overhead_ms": 0.0,
        "mlx_timing_shift_select_ms": 0.0,
        "mlx_timing_commit_ms": 0.0,
        "mlx_timing_measured_forward_count": 0,
        "mlx_timing_layer_count": 0,
        # Active-window accounting (Commit 1): sum of the per-forward forwarded
        # token count (= active window for d2f / window_end; = full canvas for the
        # no-cache strategies). Surfaced as mlx_active_tokens_per_forward to quantify
        # the full-canvas reforward the active-window runtime exists to cut.
        "mlx_timing_active_tokens_total": 0,
    }


@contextlib.contextmanager
def _timing_phase(metrics: dict[str, float | int | bool], key: str):
    """Accumulate wall time for a decode loop section into ``metrics[key]``.

    The manager wraps each loop section (scheduler / shift-select / commit) in
    one of these so the legacy stage-timing buckets stay populated. Times via
    ``_timer_now`` so the existing deterministic-clock monkeypatch keeps working.
    """
    started_at = mlx_forward._timer_now()
    try:
        yield
    finally:
        mlx_forward._accumulate_timing_ms(metrics, key, started_at)


def _sha256_file_handle(handle: Any) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(_WEIGHTS_HASH_CHUNK_BYTES), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _weight_source_stat(source_stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        source_stat.st_dev,
        source_stat.st_ino,
        source_stat.st_size,
        source_stat.st_mtime_ns,
        source_stat.st_ctime_ns,
    )


def _require_stable_weight_source(
    expected: tuple[int, int, int, int, int], current: os.stat_result
) -> None:
    if _weight_source_stat(current) != expected:
        raise ValueError("mlx_weights_changed_during_load")


def _validate_declared_biases_present(weights: dict, graph: GraphIR) -> None:
    """Require actual projection biases to match the compiled graph exactly."""

    declared: set[str] = set()
    possible: set[str] = set()
    for layer_index, layer in enumerate(graph.layers):
        prefix = f"model.layers.{layer_index}"
        possible.update(
            f"{prefix}.self_attn.{role}_proj.bias" for role in ("q", "k", "v", "o")
        )
        possible.update(
            f"{prefix}.mlp.{role}_proj.bias" for role in ("gate", "up", "down")
        )
        for op in layer.ops:
            if op.type != "linear" or op.params.get("role") not in {"q", "k", "v"}:
                continue
            bias_present = op.params.get("bias_present")
            if type(bias_present) is not bool:
                raise ValueError("GraphIR linear bias_present must be a boolean")
            bias_key = op.weight_keys.get("bias")
            if bias_present:
                if not bias_key:
                    raise ValueError("GraphIR bias-bearing linear has no bias weight key")
                declared.add(bias_key)
            elif bias_key is not None:
                raise ValueError("GraphIR bias-free linear declares a bias weight key")
    missing = sorted(declared - set(weights))
    if missing:
        raise ValueError(
            f"bias_present=True but {len(missing)} declared q/k/v bias tensor(s) "
            f"are missing from the weights (e.g. {missing[0]})"
        )
    unexpected = sorted((possible & set(weights)) - declared)
    if unexpected:
        raise ValueError(
            f"GraphIR declares no bias for {len(unexpected)} present tensor(s) "
            f"(e.g. {unexpected[0]})"
        )


def _load_mlx_weight_state(
    weights_handle: Any,
    *,
    source_path: Path,
    source_stat: os.stat_result,
    dtype: Any = None,
    bind_digest: bool = False,
) -> MLXWeightState:
    """Load the merged npz into resident mlx arrays at the compute ``dtype``.

    Defaults to fp32 (byte-identical to the legacy load). When the forward
    computes in bf16, the resident weights are bf16 — HALVING the resident weight
    RAM and the per-step weight bandwidth on memory-bound batch-1 decode (the
    throughput lever). Non-float tensors are kept as-is. ``_weight`` then casts
    each weight to the compute dtype at use (a no-op when already that dtype).
    The caller supplies the safely opened package-local descriptor. It is fully
    loaded without reopening its display path. Stable pre/post-load stats always
    bind provenance and ``source_stat`` to the resident arrays. ``bind_digest``
    adds one bounded hash pass on the same descriptor only when cross-session
    identity is required.
    """
    import mlx.core as mx
    import numpy as np

    target = mx.float32 if dtype is None else mlx_forward._resolve_compute_dtype(dtype)
    start_stat_result = os.fstat(weights_handle.fileno())
    if not stat.S_ISREG(start_stat_result.st_mode):
        raise ValueError("mlx_weights_not_regular_file")
    source_identity = _weight_source_stat(start_stat_result)
    if source_identity != _weight_source_stat(source_stat):
        raise ValueError("mlx_weights_changed_during_load")

    weights_handle.seek(0)
    with np.load(weights_handle) as source:
        weights = {}
        for key in source.files:
            value = source[key]
            if value.dtype.kind == "f":
                if target is mx.float32:
                    weights[key] = mx.array(value.astype(np.float32))
                else:
                    # numpy has no bf16 -> cast through MLX.
                    weights[key] = mx.array(value).astype(target)
            else:
                weights[key] = mx.array(value)
    _require_stable_weight_source(source_identity, os.fstat(weights_handle.fileno()))
    source_digest = None
    if bind_digest:
        weights_handle.seek(0)
        source_digest = _sha256_file_handle(weights_handle)
        _require_stable_weight_source(
            source_identity, os.fstat(weights_handle.fileno())
        )

    state = MLXWeightState(
        weights=weights,
        source_path=Path(source_path),
    )
    state.source_digest = source_digest
    state.source_stat = source_identity
    state._loader_provenance = _MLX_WEIGHT_LOADER_PROVENANCE
    return state


def _load_mlx_sharded_weight_state(
    *,
    package_dir: Path,
    artifact: WeightArtifactSpec,
    dtype: Any = None,
    bind_digest: bool = False,
) -> MLXWeightState:
    """Load a verified dtype-preserving store into resident MLX arrays.

    Shards are opened through the same package-relative no-follow boundary as
    legacy NPZ.  ``mx.load`` accepts the already-open binary stream; after it has
    materialized the arrays, the context verifies that exact descriptor against
    the digest recorded by the bounded index.
    """

    import mlx.core as mx

    target = mx.float32 if dtype is None else mlx_forward._resolve_compute_dtype(dtype)
    index = read_weight_store_index(package_dir, artifact)
    weights: dict[str, Any] = {}
    for shard in index.shards:
        expected = {tensor.name: tensor for tensor in shard.tensors}
        with open_verified_weight_shard(package_dir, shard) as handle:
            loaded = mx.load(handle, format="safetensors")
            if not isinstance(loaded, dict) or set(loaded) != set(expected):
                raise ValueError("weight_store_shard_tensor_set_mismatch")
            # Bind all arrays to the still-open, subsequently hashed descriptor.
            # This makes the loader correct even if a future MLX release makes
            # safetensors reads lazy.
            mx.eval(*loaded.values())
        for name, record in expected.items():
            value = loaded[name]
            if tuple(int(dimension) for dimension in value.shape) != record.shape:
                raise ValueError("weight_store_shard_tensor_shape_mismatch")
            actual_dtype = str(value.dtype).rsplit(".", 1)[-1].lower()
            if actual_dtype != record.dtype.lower():
                raise ValueError("weight_store_shard_tensor_dtype_mismatch")
            if record.dtype in {"bfloat16", "float16", "float32", "float64"}:
                value = value.astype(target)
            weights[name] = value

    state = MLXWeightState(weights=weights, source_path=index.source_path)
    state.source_stat = index.source_stat
    state.source_digest = index.source_digest if bind_digest else None
    state._loader_provenance = _MLX_WEIGHT_LOADER_PROVENANCE
    return state


# Quant-1: the resident linear-projection weight keys — the 2D matmul operands
# the ``linear`` kernel consumes. Exactly these may quantize: attention
# q/k/v/o_proj, MLP gate/up/down, and the lm_head. NEVER norms (tiny vectors),
# NEVER the embedding (a gather, not a matmul), NEVER biases.
_LINEAR_QUANT_WEIGHT_SUFFIXES = (
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
)

_LINEAR_QUANT_ALLOWED_BITS = (8, 4)


def _resolve_linear_quant_spec(model_config: Any) -> tuple[int, int] | None:
    """Resolve the opt-in linear-quantization spec from ``model_config``.

    ``mlx_linear_quant_bits`` in {None (default, OFF), 8, 4} +
    ``mlx_linear_quant_group_size`` (default 64). Returns ``None`` (quantization
    OFF — the byte-identical fp forward) or ``(bits, group_size)``. Fail-closed:
    any other bits value, or a non-positive / non-int group size, raises — a
    typo must never silently fall back to a different precision.
    """
    cfg = model_config or {}
    bits = cfg.get("mlx_linear_quant_bits")
    if bits is None:
        return None
    # NOTE: bool is an int subclass; True/False must not alias 1/0 silently.
    if isinstance(bits, bool) or bits not in _LINEAR_QUANT_ALLOWED_BITS:
        raise ValueError(
            f"unsupported mlx_linear_quant_bits: {bits!r} (expected None, 8, or 4)"
        )
    group_size_raw = cfg.get("mlx_linear_quant_group_size", 64)
    if isinstance(group_size_raw, bool):
        raise ValueError(
            f"mlx_linear_quant_group_size must be a positive int; got {group_size_raw!r}"
        )
    try:
        group_size = int(group_size_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"mlx_linear_quant_group_size must be a positive int; got {group_size_raw!r}"
        ) from exc
    if group_size < 1:
        raise ValueError(
            f"mlx_linear_quant_group_size must be a positive int; got {group_size_raw!r}"
        )
    return (int(bits), group_size)


def _is_quantizable_linear_key(key: str) -> bool:
    if key == "lm_head.weight":
        return True
    return key.startswith("model.layers.") and key.endswith(
        _LINEAR_QUANT_WEIGHT_SUFFIXES
    )


def _quantize_linear_weights(
    weights: dict[str, Any], *, bits: int, group_size: int
) -> None:
    """Replace the resident fp linear weights with ``QuantizedWeight`` bundles.

    Runs ONCE per weight load (inside the ``_resident_weights`` load branch) —
    quantization is a one-time load cost, never a per-forward one. A linear
    weight whose in-dim does not divide by ``group_size`` (or that is not 2D)
    keeps its fp array — a per-weight fp fallback, not a crash.
    """
    for key, value in weights.items():
        if not _is_quantizable_linear_key(key):
            continue
        shape = tuple(getattr(value, "shape", ()))
        if len(shape) != 2 or int(shape[-1]) % group_size != 0:
            continue  # fp fallback for the indivisible / odd-shaped weight
        weights[key] = mlx_ops.quantize_linear_weight(
            value, group_size=group_size, bits=bits
        )
