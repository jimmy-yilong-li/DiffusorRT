from __future__ import annotations

import time
from dataclasses import dataclass
import math
import random

from .cache import RefreshSchedule
from .executor import (
    BlockLogits,
    DreamTransformerExecutor,
    ExecutorInput,
    FakeExecutor,
    PROBED_MLX_EXECUTOR_SOURCE,
    PooledEmbeddingProbeExecutor,
)
from .policy import softmax_probabilities, token_commit_decision


@dataclass(slots=True)
class DecodeOutcome:
    generated_tokens: list[int]
    block_count: int
    cache_refresh_count: int
    iteration_count: int
    first_block_latency_ms: float
    hold_count: int
    safe_commit_count: int
    executor_source: str
    decode_loop_kind: str
    visible_generated_token_ids: list[int]
    committed_count: int
    entropy_used: bool
    refresh_real: bool

    finish_reason: str = "length"
    """Why the loop stopped: "stop" (EOS appeared) or "length" (budget reached).

    Recorded, not inferred. The NumPy path returned finish_reason="stop"
    unconditionally, so a caller could not tell a completed answer from a
    truncated one -- and "stop" is the reading that makes a truncation look
    finished. This field changes no selection: it names the branch the loop
    already took.
    """


def run_block_diffusion_decode(
    *,
    prompt_tokens: list[int],
    executor: FakeExecutor | PooledEmbeddingProbeExecutor | DreamTransformerExecutor,
    max_tokens: int,
    mask_token_id: int,
    eos_token_id: int,
    block_size: int,
    step_budget: int,
    attention_mask_kind: str,
) -> DecodeOutcome:
    started_at = time.perf_counter()
    refresh_schedule = RefreshSchedule()
    generated: list[int] = []
    block_count = 0
    cache_refresh_count = 0
    iteration_count = 0
    hold_count = 0
    safe_commit_count = 0
    committed_count = 0
    first_block_latency_ms = 0.0
    executor_source = "python_scaffold_logits"
    entropy_used = False
    # "length" until EOS actually appears. The default is the truthful one:
    # a loop that exits on the token budget did not stop, it ran out.
    finish_reason = "length"

    while len(_visible_tokens(generated, mask_token_id=mask_token_id, eos_token_id=eos_token_id)) < max_tokens:
        block_len = min(block_size, max_tokens - len(_visible_tokens(generated, mask_token_id=mask_token_id, eos_token_id=eos_token_id)))
        block_tokens = [mask_token_id] * block_len
        committed = [False] * block_len
        last_logits: BlockLogits | None = None

        for _ in range(step_budget):
            iteration_count += 1
            last_logits = executor.step(
                ExecutorInput(
                    prompt_token_ids=[*prompt_tokens, *generated],
                    block_token_ids=block_tokens,
                    block_position=len(generated),
                    attention_mask_kind=attention_mask_kind,
                )
            )
            executor_source = last_logits.source
            changed = False
            for idx, row in enumerate(last_logits.values):
                if committed[idx]:
                    continue
                decision = token_commit_decision(row)
                entropy_used = True
                if decision.kind == "commit":
                    block_tokens[idx] = _select_token(
                        row,
                        banned_ids=set(_visible_tokens(generated, mask_token_id=mask_token_id, eos_token_id=eos_token_id)[-4:])
                        | {token_id for token_id in block_tokens if token_id != mask_token_id},
                    )
                    committed[idx] = True
                    committed_count += 1
                    changed = True
                else:
                    hold_count += 1

                if all(committed):
                    break
            if all(committed):
                break
            if not changed and not _supports_multi_iteration(last_logits.source):
                break

        if last_logits is None:
            break

        for idx, row in enumerate(last_logits.values):
            if not committed[idx]:
                block_tokens[idx] = _select_token(
                    row,
                    banned_ids=set(_visible_tokens(generated, mask_token_id=mask_token_id, eos_token_id=eos_token_id)[-4:])
                    | {token_id for token_id in block_tokens if token_id != mask_token_id},
                )
                committed[idx] = True
                safe_commit_count += 1
                committed_count += 1

        generated.extend(block_tokens)
        block_count += 1
        if refresh_schedule.should_refresh(block_count):
            cache_refresh_count += 1
        if block_count == 1:
            first_block_latency_ms = (time.perf_counter() - started_at) * 1000.0
        if eos_token_id in block_tokens:
            finish_reason = "stop"
            break

    visible_generated = _visible_tokens(generated, mask_token_id=mask_token_id, eos_token_id=eos_token_id)
    return DecodeOutcome(
        generated_tokens=generated,
        block_count=block_count,
        cache_refresh_count=cache_refresh_count,
        iteration_count=iteration_count,
        first_block_latency_ms=first_block_latency_ms,
        hold_count=hold_count,
        safe_commit_count=safe_commit_count,
        executor_source=executor_source,
        decode_loop_kind="d2f_block_diffusion",
        committed_count=committed_count,
        visible_generated_token_ids=visible_generated,
        entropy_used=entropy_used,
        refresh_real=False,
        finish_reason=finish_reason,
    )


def run_dream_masked_diffusion_decode(
    *,
    prompt_tokens: list[int],
    executor: DreamTransformerExecutor,
    max_tokens: int,
    mask_token_id: int,
    eos_token_id: int,
    diffusion_steps: int,
    attention_mask_kind: str,
    eps: float = 1e-3,
    cache_refresh_interval: int = 0,
    diffusion_alg: str = "entropy",
) -> DecodeOutcome:
    started_at = time.perf_counter()
    if diffusion_alg not in {"entropy", "origin"}:
        raise ValueError(f"unknown Dream diffusion alg: {diffusion_alg}")
    decode_loop_kind = f"dream_masked_diffusion_{diffusion_alg}"
    generated = [mask_token_id] * max_tokens
    executor_source = "python_scaffold_logits"
    iteration_count = 0
    committed_count = 0
    hold_count = 0
    first_block_latency_ms = 0.0
    cache_refresh_count = 0
    cache_forward_count = 0
    cache_reuse_seen = False
    cache_refresh_seen = False
    cache_state_ref = "dream_bidirectional_layer0_kv" if cache_refresh_interval > 0 else None

    if max_tokens <= 0 or diffusion_steps <= 0:
        return DecodeOutcome(
            generated_tokens=[],
            block_count=0,
            cache_refresh_count=0,
            iteration_count=0,
            first_block_latency_ms=0.0,
            hold_count=0,
            safe_commit_count=0,
            executor_source=executor_source,
            decode_loop_kind=decode_loop_kind,
            committed_count=0,
            visible_generated_token_ids=[],
            entropy_used=False,
            refresh_real=False,
            finish_reason="length",
        )

    timesteps = _linspace(1.0, eps, diffusion_steps + 1)

    for step_idx in range(diffusion_steps):
        mask_positions = [idx for idx, token_id in enumerate(generated) if token_id == mask_token_id]
        if not mask_positions:
            break

        t = timesteps[step_idx]
        s = timesteps[step_idx + 1]
        transfer_count = len(mask_positions) if step_idx == diffusion_steps - 1 else int(len(mask_positions) * (1 - s / t))
        transfer_count = max(0, min(len(mask_positions), transfer_count))
        if diffusion_alg == "entropy" and transfer_count == 0:
            hold_count += len(mask_positions)
            continue

        cache_metadata = _dream_cache_metadata(
            cache_refresh_interval=cache_refresh_interval,
            forward_index=cache_forward_count + 1,
            block_start=len(prompt_tokens),
        )
        if cache_metadata.get("kv_cache_action") == "refresh":
            cache_refresh_count += 1

        block_logits = executor.step(
            ExecutorInput(
                prompt_token_ids=prompt_tokens,
                block_token_ids=list(generated),
                block_position=0,
                mask_pattern=[False] * len(prompt_tokens)
                + [token_id == mask_token_id for token_id in generated],
                cache_state_ref=cache_state_ref,
                attention_mask_kind=attention_mask_kind,
                attention_metadata=cache_metadata,
            )
        )
        cache_forward_count += 1
        iteration_count += 1
        executor_source = block_logits.source
        if int(block_logits.attention_metadata.get("kv_cache_reused", 0)) > 0:
            cache_reuse_seen = True
        if int(block_logits.attention_metadata.get("kv_cache_refreshed", 0)) > 0:
            cache_refresh_seen = True
        if iteration_count == 1:
            first_block_latency_ms = (time.perf_counter() - started_at) * 1000.0

        if diffusion_alg == "origin":
            p_transfer = 1.0 if step_idx == diffusion_steps - 1 else 1 - s / t
            step_committed = 0
            for position in mask_positions:
                if step_idx != diffusion_steps - 1 and random.random() >= p_transfer:
                    continue
                token_id, _confidence = _sample_tokens(block_logits.values[position])
                generated[position] = token_id
                committed_count += 1
                step_committed += 1
            hold_count += len(mask_positions) - step_committed
        else:
            sampled: list[tuple[int, int, float]] = []
            for position in mask_positions:
                token_id, confidence = _sample_tokens(
                    block_logits.values[position],
                    neg_entropy=True,
                )
                sampled.append((position, token_id, confidence))

            sampled.sort(key=lambda item: item[2], reverse=True)
            for position, token_id, _confidence in sampled[:transfer_count]:
                generated[position] = token_id
                committed_count += 1
            hold_count += len(mask_positions) - transfer_count

    visible_generated = _visible_tokens(generated, mask_token_id=mask_token_id, eos_token_id=eos_token_id)
    # This loop fills a fixed canvas, so it always exhausts its budget. EOS
    # appearing among the produced tokens is what makes the answer complete;
    # anything else is a truncation, and calling that "stop" is the reading
    # that makes a cut-off answer look finished.
    finish_reason = "stop" if eos_token_id in generated else "length"
    return DecodeOutcome(
        generated_tokens=generated,
        block_count=1 if max_tokens > 0 else 0,
        cache_refresh_count=cache_refresh_count,
        iteration_count=iteration_count,
        first_block_latency_ms=first_block_latency_ms,
        hold_count=hold_count,
        safe_commit_count=0,
        executor_source=executor_source,
        decode_loop_kind=decode_loop_kind,
        committed_count=committed_count,
        visible_generated_token_ids=visible_generated,
        entropy_used=diffusion_alg == "entropy",
        refresh_real=cache_reuse_seen and cache_refresh_seen,
        finish_reason=finish_reason,
    )


def _dream_cache_metadata(
    *,
    cache_refresh_interval: int,
    forward_index: int,
    block_start: int,
) -> dict[str, str | int]:
    if cache_refresh_interval <= 0:
        return {}
    action = "refresh" if forward_index > 0 and forward_index % cache_refresh_interval == 0 else "reuse"
    return {
        "kv_cache_kind": "bidirectional_layer0_kv",
        "kv_cache_action": action,
        "kv_cache_block_start": block_start,
    }


def _visible_tokens(tokens: list[int], *, mask_token_id: int, eos_token_id: int) -> list[int]:
    visible: list[int] = []
    for token_id in tokens:
        if token_id == eos_token_id:
            break
        if token_id != mask_token_id:
            visible.append(token_id)
    return visible


def _argmax_token(values: list[float]) -> int:
    return max(range(len(values)), key=lambda idx: values[idx])


def _select_token(values: list[float], *, banned_ids: set[int]) -> int:
    ranked = sorted(range(len(values)), key=lambda idx: values[idx], reverse=True)
    for token_id in ranked:
        if token_id not in banned_ids:
            return token_id
    return ranked[0]


def _sample_tokens(
    values: list[float],
    *,
    neg_entropy: bool = False,
) -> tuple[int, float]:
    probabilities = _softmax(values)
    token_id = max(range(len(probabilities)), key=probabilities.__getitem__) if probabilities else 0
    confidence = probabilities[token_id] if probabilities else 1.0
    if neg_entropy:
        confidence = sum(prob * math.log(prob + 1e-10) for prob in probabilities)
    return token_id, confidence


def _softmax(values: list[float]) -> list[float]:
    return softmax_probabilities(values)


def _linspace(start: float, stop: float, count: int) -> list[float]:
    if count <= 1:
        return [start]
    step = (stop - start) / (count - 1)
    return [start + step * idx for idx in range(count)]


def _supports_multi_iteration(executor_source: str) -> bool:
    return executor_source != PROBED_MLX_EXECUTOR_SOURCE
