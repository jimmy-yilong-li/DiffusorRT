"""Common-policy execution over a compiled device session.

This module owns no GPU allocations, cache, model forward or decode policy.
The token list is a control mirror updated only after a checked device commit.
The caller owns the session and MUST close it on success, cancellation or error.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from ..backend_ids import CUDA_PRECISION_IDS
from .decode_session import DecodeRequest, DecodeSession
from .diffusion_runtime import DiffusionIterationResult
from .window_plan import WindowExecutionResult


def cuda_precision_descriptor(scheme: str) -> dict[str, int | str]:
    """Return the exact callable descriptor accepted by the CUDA provider."""
    if scheme not in CUDA_PRECISION_IDS:
        raise ValueError(
            f"unsupported CUDA precision {scheme!r}; expected one of "
            f"{', '.join(CUDA_PRECISION_IDS)}"
        )
    bf16 = scheme == "bf16"
    return {
        "version": 1,
        "scheme": scheme,
        "weight_type": "bf16" if bf16 else "int8",
        "activation_type": "bf16",
        "accumulation_type": "fp32",
        "scale_granularity": "none" if bf16 else "per_group",
        "scale_layout": "" if bf16 else "row-major-linear-input-groups-g64",
        "calibration_provenance": "" if bf16 else "weight-group-absmax-v1",
        "kernel_provider": (
            "cuBLASLt+diffusorrt_cuda" if bf16 else "diffusorrt_cuda"
        ),
        "kernel_version": (
            "bf16-weights-fp32-residual-v3"
            if bf16
            else "int8-g64-linear-weights-fp32-residual-v4"
        ),
        "packing_version": (
            "row-major-bf16-v1" if bf16 else "row-major-int8-linear-g64-v2"
        ),
    }


def cuda_engine_config(scheme: str, *, exact_kv: bool) -> dict[str, int]:
    """Translate one admitted precision into the native engine request."""
    cuda_precision_descriptor(scheme)
    config = {"compute_dtype": 0, "d2f_exact_kv_cache": int(exact_kv)}
    if scheme == "int8_weight_only":
        config.update(linear_quant_bits=8, linear_quant_group_size=64)
    return config


@dataclass(slots=True)
class DeviceInfillResult:
    fill_token_ids: list[int]
    full_token_ids: list[int]
    iteration_count: int
    forward_count: int
    committed_token_count: int
    remaining_mask_count: int
    finish_reason: str
    decode_loop_kind: str
    entropy_used: bool


def _validate_decode_options(
    *,
    max_tokens: int,
    steps: int,
    block_size: int,
    confidence_threshold: float,
    exact_kv: bool,
    memory_budget_bytes: int,
) -> None:
    for name, value, minimum in (
        ("max_tokens", max_tokens, 0),
        ("steps", steps, 1),
        ("block_size", block_size, 1),
        ("memory_budget_bytes", memory_budget_bytes, 0),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not math.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be finite and in [0, 1]")
    if not isinstance(exact_kv, bool):
        raise TypeError("exact_kv must be bool")


def _forming_reply(state, token_count: int, *, block_end: int | None) -> list[int]:
    """The reply as decided so far, for a window that shows it forming.

    ``block_end`` is where BD3's active block ends, and None for a policy that
    decodes the whole canvas at once. Once BD3 has decided an EOS it stops
    after that block, so the canvas it will decode ends there.
    """
    generated = state.x[state.prompt_len:state.prompt_len + token_count]
    if block_end and state.eos_token_id in generated:
        return state.x[state.prompt_len:block_end]
    return generated


def _run_device_state(
    session,
    decode: DecodeSession,
    *,
    mutable_token_count: int,
    exact_kv: bool,
    memory_budget_bytes: int,
    attention_mask_kind: str,
    cancelled: Callable[[], bool] | None,
) -> DiffusionIterationResult:
    state = decode.state
    block_size = decode.request.block_size
    steps = decode.request.steps
    if attention_mask_kind not in {"bidirectional", "d2f_hybrid"}:
        raise ValueError(
            "device generation supports only bidirectional or d2f_hybrid graphs"
        )
    if attention_mask_kind == "bidirectional" and exact_kv:
        raise ValueError("bidirectional device generation cannot use exact D2F K/V")
    if attention_mask_kind == "d2f_hybrid" and (
        state.fillable_positions is not None
        or state.prompt_len + mutable_token_count != len(state.x)
    ):
        raise ValueError("d2f_hybrid device generation requires a trailing mutable span")

    from ..decode_progress import report_canvas

    # BD3's blocks are absolute: its geometry's origin (0) is the session's
    # block origin. Other policies count blocks from the prompt.
    geometry = decode.mask_geometry
    origin = state.prompt_len if geometry is None else geometry["prompt_len"]
    proposals = 0
    session.start_device(
        state.x,
        prompt_len=origin,
        block_size=block_size,
        mask_token_id=state.mask_token_id,
        exact_kv=exact_kv,
        steps=steps,
        memory_budget_bytes=memory_budget_bytes,
        temperature=float(decode.request.temperature),
        seed=int(decode.request.seed),
    )

    def propose(plan):
        nonlocal proposals
        if cancelled is not None and cancelled():
            raise InterruptedError("device generation cancelled")
        generated = _forming_reply(
            state, mutable_token_count,
            block_end=plan.window_end if geometry is not None else None,
        )
        report_canvas(step=proposals, steps=steps, generated=generated,
                      mask_token_id=state.mask_token_id, eos_token_id=state.eos_token_id)
        proposals += 1
        if attention_mask_kind == "bidirectional":
            # Every LLaDA row attends to both future masks and any fixed suffix.
            # The native provider enforces this full-canvas request as well.
            frozen = state.prompt_len
            end = len(state.x)
        else:
            # Dream's D2F graph may freeze complete generated blocks. This is
            # also the cache-off shape; only exact_kv changes reuse. BD3 forwards
            # the prefix through its active absolute block, as its policy plans.
            block = (min(plan.frontier) - origin) // block_size
            frozen = origin + block * block_size
            end = plan.window_end if geometry is not None and plan.window_end else (
                min(len(state.x), frozen + block_size))
        return session.propose_device(
            plan.frontier, window_end=end, frozen_prefix=frozen
        )

    return decode.run(
        propose_fn=propose,
        commit_device_fn=session.commit_device,
    )


def generate_device_tokens(
    session, prompt_token_ids: list[int], *, max_tokens: int,
    mask_token_id: int, eos_token_id: int, steps: int = 512,
    block_size: int = 8, confidence_threshold: float = 0.9,
    exact_kv: bool = True, memory_budget_bytes: int = 0,
    attention_mask_kind: str = "d2f_hybrid",
    cancelled: Callable[[], bool] | None = None,
    decode_strategy: str = "block_threshold",
    requires_prompt_block_alignment: bool = False,
    temperature: float = 0.0,
    seed: int = 0,
) -> DiffusionIterationResult:
    """Run a shared policy with scoring on the device.

    The device returns each frontier row's token and raw max-probability, so no
    row of logits crosses to the host. ``block_threshold`` is the D2F policy;
    its candidates leave MASK out of the argmax and the softmax, as its host
    rule does. ``fixed_quota`` runs MDLM's full bidirectional canvas without
    K/V, and ``bd3_fixed_quota`` runs BD3 over its absolute blocks on a
    d2f_hybrid graph without K/V: the session's block origin is 0 and each
    proposal forwards the prefix through the active block. Both rank rows over
    the whole vocabulary, MASK included, as their host rules and the upstream
    samplers do (review R7): a candidate may be MASK, and committing it leaves
    the row masked. The session names each policy (review R2), and the runtime
    admits it only on its graph. Scores are in double on the device; they can
    differ from the host's in the last bits, by summation order. Above
    ``temperature`` 0 a fixed-quota policy draws each row's token as its host
    rule does, keyed by ``seed``; the threshold policy refuses a temperature.
    Cancellation is observed before each forward, not by freeing in-flight
    buffers from another thread. Native session close drains and releases
    them. Other strategies are not translated to this contract.
    """
    if decode_strategy not in {"block_threshold", "fixed_quota", "bd3_fixed_quota"}:
        raise ValueError(
            "device sessions run only block_threshold, fixed_quota and bd3_fixed_quota, "
            f"not {decode_strategy!r}"
        )
    if decode_strategy == "bd3_fixed_quota" and (
        attention_mask_kind != "d2f_hybrid" or exact_kv
    ):
        raise ValueError("device bd3_fixed_quota runs only a d2f_hybrid canvas without K/V")
    if decode_strategy == "fixed_quota" and (
        attention_mask_kind != "bidirectional" or exact_kv
    ):
        raise ValueError(
            "device fixed_quota runs only a full bidirectional canvas without K/V"
        )
    _validate_decode_options(
        max_tokens=max_tokens,
        steps=steps,
        block_size=block_size,
        confidence_threshold=confidence_threshold,
        exact_kv=exact_kv,
        memory_budget_bytes=memory_budget_bytes,
    )
    if not prompt_token_ids:
        raise ValueError("device generation requires a non-empty prompt")
    if decode_strategy == "bd3_fixed_quota" and mask_token_id in prompt_token_ids:
        # Block origin 0 makes the prompt part of the session's canvas; such a
        # prompt keeps the host window loop (see RuntimeHandle._generate_cuda_native).
        raise ValueError(
            "device bd3_fixed_quota cannot carry MASK in its prompt; "
            "run it through generate_window_tokens"
        )
    options = ({"confidence_threshold": confidence_threshold}
               if decode_strategy == "block_threshold"
               else {"requires_prompt_block_alignment": requires_prompt_block_alignment})
    decode = DecodeRequest(
        decode_strategy=decode_strategy,
        steps=steps,
        block_size=block_size,
        temperature=temperature,
        seed=seed,
        **options,
    ).prepare(
        prefix_token_ids=prompt_token_ids,
        token_count=max_tokens,
        mask_token_id=mask_token_id,
        eos_token_id=eos_token_id,
    )
    return _run_device_state(
        session,
        decode,
        mutable_token_count=max_tokens,
        exact_kv=exact_kv,
        memory_budget_bytes=memory_budget_bytes,
        attention_mask_kind=attention_mask_kind,
        cancelled=cancelled,
    )


def generate_window_tokens(
    session,
    prompt_token_ids: list[int],
    *,
    max_tokens: int,
    mask_token_id: int,
    eos_token_id: int,
    steps: int,
    block_size: int,
    decode_strategy: str = "fixed_quota",
    requires_prompt_block_alignment: bool = True,
    temperature: float = 0.0,
    seed: int = 0,
) -> DiffusionIterationResult:
    """Run fixed-quota decoding over compiled window execution, without a cache.

    The device computes frontier logits and the shared policy scores them on the
    host. cuda_native runs both strategies on device sessions instead (see
    ``generate_device_tokens``) and keeps this loop only for a BD3 prompt that
    carries the mask token. ``fixed_quota`` forwards the whole bidirectional
    canvas. ``bd3_fixed_quota`` forwards the prefix through its active absolute
    block and hands every window the decode's block geometry, which a d2f_hybrid
    graph requires. A stop is honoured at the decode loop's own step boundary.
    ``temperature`` and ``seed`` sample as the device session does.
    Execution memory is unbounded: this path has no budget, so callers must
    refuse a nonzero one before opening the runtime.
    """
    if decode_strategy not in {"fixed_quota", "bd3_fixed_quota"}:
        raise ValueError(
            "the window loop runs only fixed_quota and bd3_fixed_quota, "
            f"not {decode_strategy!r}"
        )
    _validate_decode_options(
        max_tokens=max_tokens,
        steps=steps,
        block_size=block_size,
        confidence_threshold=0.0,
        exact_kv=False,
        memory_budget_bytes=0,
    )
    if not prompt_token_ids:
        raise ValueError("device generation requires a non-empty prompt")
    from ..decode_progress import report_canvas

    forwards = 0
    decode = DecodeRequest(
        decode_strategy=decode_strategy,
        steps=steps,
        block_size=block_size,
        requires_prompt_block_alignment=requires_prompt_block_alignment,
        temperature=temperature,
        seed=seed,
    ).prepare(
        prefix_token_ids=prompt_token_ids,
        token_count=max_tokens,
        mask_token_id=mask_token_id,
        eos_token_id=eos_token_id,
    )
    geometry = decode.mask_geometry
    window_options = (
        {"mask_geometry": (geometry["prompt_len"], geometry["block_size"])}
        if geometry is not None
        else {}
    )

    def forward(plan, token_ids: list[int]) -> WindowExecutionResult:
        import numpy as np

        nonlocal forwards
        generated = _forming_reply(
            decode.state, max_tokens,
            block_end=plan.window_end if geometry is not None else None,
        )
        report_canvas(step=forwards, steps=steps, generated=generated,
                      mask_token_id=mask_token_id, eos_token_id=eos_token_id)
        forwards += 1
        scope = tuple(plan.forward_scope)
        window = session.execute_window(token_ids, scope, **window_options)
        logits = np.frombuffer(window.logits, dtype="<f4").reshape(
            window.n_rows, window.n_vocab
        )
        return WindowExecutionResult(
            logits=logits,
            positions=scope,
            cache_mutations=(),
            forwarded_rows=window.forwarded_rows,
        )

    return decode.run(execute_window_fn=forward)


def infill_device_tokens(
    session,
    prefix_token_ids: list[int],
    suffix_token_ids: list[int],
    *,
    max_fill_tokens: int,
    mask_token_id: int,
    eos_token_id: int,
    steps: int = 512,
    block_size: int = 8,
    confidence_threshold: float = 0.9,
    memory_budget_bytes: int = 0,
    cancelled: Callable[[], bool] | None = None,
) -> DeviceInfillResult:
    """Fill one contiguous span with the shared policy and full CUDA canvas."""
    _validate_decode_options(
        max_tokens=max_fill_tokens,
        steps=steps,
        block_size=block_size,
        confidence_threshold=confidence_threshold,
        exact_kv=False,
        memory_budget_bytes=memory_budget_bytes,
    )
    decode = DecodeRequest(
        decode_strategy="block_threshold",
        steps=steps,
        block_size=block_size,
        confidence_threshold=confidence_threshold,
    ).prepare(
        prefix_token_ids=prefix_token_ids,
        suffix_token_ids=suffix_token_ids,
        token_count=max_fill_tokens,
        mask_token_id=mask_token_id,
        eos_token_id=eos_token_id,
    )
    if not decode.state.x:
        return DeviceInfillResult(
            fill_token_ids=[],
            full_token_ids=[],
            iteration_count=0,
            forward_count=0,
            committed_token_count=0,
            remaining_mask_count=0,
            finish_reason="stop",
            decode_loop_kind="mlx_dream_block_threshold",
            entropy_used=False,
        )
    decoded = _run_device_state(
        session,
        decode,
        mutable_token_count=max_fill_tokens,
        exact_kv=False,
        memory_budget_bytes=memory_budget_bytes,
        attention_mask_kind="bidirectional",
        cancelled=cancelled,
    )
    span = decode.state.filled_span()
    remaining = int(span["remaining_mask_count"])
    return DeviceInfillResult(
        fill_token_ids=list(span["fill_token_ids"]),
        full_token_ids=list(span["full_token_ids"]),
        iteration_count=decoded.iteration_count,
        forward_count=decoded.forward_count,
        committed_token_count=decoded.committed_token_count,
        remaining_mask_count=remaining,
        finish_reason="length" if remaining else "stop",
        decode_loop_kind=decoded.decode_loop_kind,
        entropy_used=decoded.entropy_used,
    )
