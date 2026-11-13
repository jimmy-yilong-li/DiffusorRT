"""Request-scoped decode assembly shared by host and device adapters.

This module binds immutable request policy to a fresh canvas, recovery hooks,
cache notifications and optional tracing. DiffusionIterationManager remains the
only decode loop; backend adapters retain weights, forward execution and device
resource ownership. No logits or K/V tensors are stored here.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable

from . import diffusion_runtime as dr
from . import mask_canvas
from .correction_policy import CorrectionPolicy


@dataclass(frozen=True, slots=True)
class RecoveryPolicy:
    enabled: bool = False
    detectors: tuple[str, ...] = ("low_confidence", "repetition")
    max_rounds: int = 2
    confidence_floor: float = 0.5
    repetition_max_ngram: int = 3
    kl_threshold: float = 1.0

    @classmethod
    def from_config(cls, config: dict | None) -> RecoveryPolicy:
        if not config or not bool(config.get("enabled", False)):
            return cls()
        requested = tuple(config.get("detectors", ("low_confidence", "repetition")))
        return cls(
            enabled=True,
            detectors=requested,
            max_rounds=int(config.get("max_rounds", 2)),
            confidence_floor=(
                float(config.get("confidence_floor", 0.5))
                if "low_confidence" in requested
                else 0.5
            ),
            repetition_max_ngram=(
                int(config.get("repetition_max_ngram", 3))
                if "repetition" in requested
                else 3
            ),
            kl_threshold=(
                float(config.get("kl_threshold", 1.0))
                if "kl_drift" in requested
                else 1.0
            ),
        )

    def build(self, *, frozen_positions_fn=None):
        if not self.enabled:
            return dr.QualityRecoveryLoop(), None, False
        detectors = []
        for name in self.detectors:
            if name == "low_confidence":
                detectors.append(dr.LowConfidenceDetector(floor=self.confidence_floor))
            elif name == "repetition":
                detectors.append(
                    dr.RepetitionDetector(max_ngram=self.repetition_max_ngram)
                )
            elif name == "kl_drift":
                detectors.append(dr.KLDriftDetector(threshold=self.kl_threshold))
            else:
                raise ValueError(
                    f"unknown error-fixing detector {name!r}; expected one of "
                    "'low_confidence', 'repetition', 'kl_drift'"
                )
        detector = (
            detectors[0]
            if len(detectors) == 1
            else dr.CompositeDetector(detectors=detectors)
        )
        remask = dr.ConfWeightedRemask()
        loop = dr.QualityRecoveryLoop(
            detector=detector,
            remask_policy=remask,
            enabled=True,
            max_recovery_rounds=self.max_rounds,
            frozen_positions_fn=frozen_positions_fn,
        )
        return (
            loop,
            CorrectionPolicy(detector=detector, remask_policy=remask),
            "kl_drift" in self.detectors,
        )


@dataclass(frozen=True, slots=True)
class DecodeRequest:
    """Resolved per-call policy, independent of model and execution backend."""

    decode_strategy: str
    steps: int
    block_size: int = 8
    temperature: float = 0.0
    seed: int = 0
    """Keys the draws when ``temperature`` > 0 (see ``diffusion_runtime.sampling_noise``)."""
    top_p: float = 1.0
    eps: float = 1e-8
    confidence_threshold: float = 0.9
    requires_prompt_block_alignment: bool = True
    gamma: float = 0.1
    d2f_add_threshold: float = 0.1
    d2f_activate_threshold: float = 0.95
    d2f_commit_policy: str = "threshold"
    recovery: RecoveryPolicy = RecoveryPolicy()

    def prepare(
        self,
        *,
        prefix_token_ids: list[int],
        token_count: int,
        mask_token_id: int,
        eos_token_id: int,
        suffix_token_ids: list[int] | None = None,
        kv_cache_state: Any = None,
        state_cache_manager: Any = None,
        window_exec: bool | None = None,
        capture_canvas: bool = False,
    ) -> DecodeSession:
        """Create fresh request state; an explicit suffix selects bounded infill.

        Cache objects are borrowed from the existing owner. Frozen positions
        are queried at recovery time, never copied before the base pass.
        """
        prompt_len = len(prefix_token_ids)
        tokens = [*prefix_token_ids, *([mask_token_id] * token_count)]
        if suffix_token_ids is None:
            state = dr.MaskState(
                x=tokens,
                prompt_len=prompt_len,
                mask_token_id=mask_token_id,
                eos_token_id=eos_token_id,
            )
        else:
            state = dr.MaskState.for_infill(
                x=[*tokens, *suffix_token_ids],
                fillable_positions=list(range(prompt_len, prompt_len + token_count)),
                mask_token_id=mask_token_id,
                eos_token_id=eos_token_id,
            )
        exact = getattr(state_cache_manager, "d2f_exact_kv", None)
        layout = None
        if self.decode_strategy in {"d2f_block_pipeline", "bd3_fixed_quota"}:
            layout = dr.BlockLayout(
                start=prompt_len,
                length=token_count,
                block_size=int(self.block_size),
                origin=0 if self.decode_strategy == "bd3_fixed_quota" else prompt_len,
            )
        strategy = dr.build_decode_strategy(
            decode_strategy=self.decode_strategy,
            temperature=self.temperature,
            seed=self.seed,
            top_p=self.top_p,
            eps=self.eps,
            confidence_threshold=self.confidence_threshold,
            block_size=self.block_size,
            block_layout=layout,
            requires_prompt_block_alignment=self.requires_prompt_block_alignment,
            gamma=self.gamma,
            d2f_add_threshold=self.d2f_add_threshold,
            d2f_activate_threshold=self.d2f_activate_threshold,
            d2f_commit_policy=self.d2f_commit_policy,
            d2f_freeze_observer=exact.freeze_block if exact is not None else None,
            d2f_frozen_block_count_fn=(
                (lambda: exact.frozen_block_count) if exact is not None else None
            ),
        )
        frozen_positions = None
        if exact is not None and self.recovery.enabled:
            frozen_positions = lambda: frozenset(
                range(
                    prompt_len,
                    prompt_len + exact.frozen_block_count * int(self.block_size),
                )
            )
        recovery, corrections, record_distributions = self.recovery.build(
            frozen_positions_fn=frozen_positions,
        )
        return DecodeSession(
            request=self,
            state=state,
            strategy=strategy,
            mask_geometry=layout.mask_geometry if layout is not None else None,
            use_execute_window=window_exec if window_exec is not None else True,
            commit_observer=_cache_commit_observer(kv_cache_state, state_cache_manager),
            recovery_loop=recovery,
            correction_policy=corrections,
            record_distributions=record_distributions,
            trace_observer=mask_canvas.CanvasTraceObserver()
            if capture_canvas
            else None,
            event_log=dr.DecodeEventLog() if capture_canvas else None,
            # author:Opus -- None unless the caller scoped a live observer; the
            # session then reports its real canvas, with no effect on decoding.
            observation=mask_canvas.session_observation(
                state=state,
                start=prompt_len,
                length=token_count,
                decode_strategy=self.decode_strategy,
                step_limit=int(self.steps),
                block_layout=layout,
            ),
        )


def _cache_commit_observer(kv_cache_state, state_cache_manager):
    block = getattr(state_cache_manager, "block_kv", None)
    exact = getattr(state_cache_manager, "d2f_exact_kv", None)
    if kv_cache_state is None and block is None and exact is None:
        return None

    def observe(position, token_id):
        if kv_cache_state is not None:
            kv_cache_state.observe_commit(position=position, token_id=token_id)
        if block is not None:
            block.observe_commit(position=position, token_id=token_id)
        if exact is not None and hasattr(exact, "observe_commit"):
            exact.observe_commit(position=position, token_id=token_id)

    return observe


def _host_forward_forbidden(_tokens):
    raise RuntimeError("a device session cannot fall back to host logits")


@dataclass(slots=True)
class DecodeSession:
    """Mutable per-request control state, never a second device/KV owner."""

    request: DecodeRequest
    state: dr.MaskState
    strategy: dr.DecodingStrategy
    mask_geometry: Any
    use_execute_window: bool
    commit_observer: Any
    recovery_loop: dr.QualityRecoveryLoop
    correction_policy: CorrectionPolicy | None
    record_distributions: bool
    trace_observer: Any
    event_log: dr.DecodeEventLog | None
    observation: mask_canvas.SessionObservation | None = None

    def run(
        self,
        *,
        forward_fn: Callable = _host_forward_forbidden,
        select_transform: Callable | None = None,
        execute_window_fn: Callable | None = None,
        window_plan_sink: Callable | None = None,
        scheduler_phase: Callable = nullcontext,
        select_phase: Callable = nullcontext,
        commit_phase: Callable = nullcontext,
        propose_fn: Callable | None = None,
        commit_device_fn: Callable | None = None,
    ) -> dr.DiffusionIterationResult:
        if propose_fn is not None and self.request.decode_strategy not in {
            "block_threshold", "fixed_quota", "bd3_fixed_quota"
        }:
            raise ValueError(
                "device sessions support only the block_threshold, fixed_quota and "
                "bd3_fixed_quota request policies"
            )
        observation = self.observation
        trace_observer = self.trace_observer
        if observation is not None:
            # The session's own trace observer keeps its identity and runs first.
            trace_observer = (
                observation
                if trace_observer is None
                else mask_canvas.TraceFanOut(trace_observer, observation)
            )
            observation.begin()
        manager = dr.DiffusionIterationManager(
            state=self.state,
            strategy=self.strategy,
            steps=self.request.steps,
            forward_fn=forward_fn,
            select_transform=select_transform
            if select_transform is not None
            else lambda logits: logits,
            execute_window_fn=execute_window_fn,
            window_plan_sink=window_plan_sink,
            commit_observer=self.commit_observer,
            scheduler_phase=scheduler_phase,
            select_phase=select_phase,
            commit_phase=commit_phase,
            recovery_loop=self.recovery_loop,
            correction_policy=self.correction_policy,
            record_distributions=self.record_distributions,
            trace_observer=trace_observer,
            event_log=self.event_log,
            propose_fn=propose_fn,
            commit_device_fn=commit_device_fn,
        )
        if observation is None:
            return manager.run()
        try:
            result = manager.run()
            # author:Opus -- a stop that arrives during the last step finds no
            # step left to notice it; the request then ends when its scope
            # closes. Ask here, so the observer is not told that a request which
            # is about to raise GenerationCancelled completed.
            from ..cancellation import check_cancelled

            check_cancelled()
        except BaseException:
            observation.end(completed=False)
            raise
        observation.end(completed=True)
        return result
