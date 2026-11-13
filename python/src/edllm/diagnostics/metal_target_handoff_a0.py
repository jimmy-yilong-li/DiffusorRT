from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from . import dflash_mlx_reference as reference
from .metal_target_handoff import (
    ProjectionEvidence,
    array_digest,
    native_split_projection,
    prepare_projection_weights,
)

_TOLERANCES = {
    "float32": (1e-5, 1e-5),
    "float16": (2e-3, 2e-3),
    "bfloat16": (2e-2, 2e-2),
}


@dataclass(frozen=True, slots=True)
class A0ProjectionProbeConfig:
    user_prompt: str = "Explain why the sky appears blue in two concise sentences."
    warmups: int = 5
    trials: int = 30

    def __post_init__(self) -> None:
        if not isinstance(self.user_prompt, str) or not self.user_prompt.strip():
            raise ValueError("user_prompt must be a non-empty string")
        if (
            not isinstance(self.warmups, int)
            or isinstance(self.warmups, bool)
            or self.warmups < 1
        ):
            raise ValueError("warmups must be an int >= 1")
        if (
            not isinstance(self.trials, int)
            or isinstance(self.trials, bool)
            or self.trials < 2
        ):
            raise ValueError("trials must be an int >= 2")


def _timing_summary(samples_ms: Sequence[float]) -> dict[str, float]:
    if not samples_ms:
        raise ValueError("timing samples must not be empty")
    samples = np.asarray(samples_ms, dtype=np.float64)
    if not np.all(np.isfinite(samples)) or np.any(samples < 0):
        raise ValueError("timing samples must be finite and non-negative")
    return {
        "min_ms": float(np.min(samples)),
        "mean_ms": float(np.mean(samples)),
        "p50_ms": float(np.percentile(samples, 50)),
        "p90_ms": float(np.percentile(samples, 90)),
        "max_ms": float(np.max(samples)),
    }


def _comparison(
    oracle: np.ndarray,
    native: np.ndarray,
    *,
    dtype: str,
) -> dict[str, object]:
    try:
        rtol, atol = _TOLERANCES[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported comparison dtype: {dtype}") from exc
    oracle_fp32 = np.asarray(oracle, dtype=np.float32)
    native_fp32 = np.asarray(native, dtype=np.float32)
    if oracle_fp32.shape != native_fp32.shape:
        raise ValueError(
            f"comparison shape mismatch: {oracle_fp32.shape} != {native_fp32.shape}"
        )
    all_finite = bool(
        np.all(np.isfinite(oracle_fp32)) and np.all(np.isfinite(native_fp32))
    )
    if all_finite:
        absolute = np.abs(native_fp32 - oracle_fp32)
        denominator = np.maximum(np.abs(oracle_fp32), np.finfo(np.float32).tiny)
        relative = absolute / denominator
        max_abs_error = float(np.max(absolute, initial=0.0))
        max_relative_error = float(np.max(relative, initial=0.0))
        within_tolerance = bool(
            np.allclose(
                native_fp32,
                oracle_fp32,
                rtol=rtol,
                atol=atol,
                equal_nan=False,
            )
        )
    else:
        max_abs_error = None
        max_relative_error = None
        within_tolerance = False
    return {
        "all_finite": all_finite,
        "within_tolerance": within_tolerance,
        "rtol": rtol,
        "atol": atol,
        "max_abs_error": max_abs_error,
        "max_relative_error": max_relative_error,
    }


def _correctness_reasons(comparison: dict[str, object]) -> tuple[str, ...]:
    reasons = []
    if not comparison["all_finite"]:
        reasons.append("projection_nonfinite")
    if not comparison["within_tolerance"]:
        reasons.append("projection_tolerance_failed")
    return tuple(reasons)


def _committed_round_tokens(
    draft_tokens: Sequence[int],
    target_tokens: Sequence[int],
    *,
    stop_token_ids: Sequence[int],
    remaining_tokens: int,
) -> tuple[int, ...]:
    if len(target_tokens) != len(draft_tokens) + 1:
        raise ValueError("target_tokens must contain one correction position")
    if (
        not isinstance(remaining_tokens, int)
        or isinstance(remaining_tokens, bool)
        or remaining_tokens < 1
    ):
        raise ValueError("remaining_tokens must be an int >= 1")
    accepted = next(
        (
            index
            for index, (draft_token, target_token) in enumerate(
                zip(draft_tokens, target_tokens, strict=False)
            )
            if draft_token != target_token
        ),
        len(draft_tokens),
    )
    committed = tuple(draft_tokens[:accepted]) + (target_tokens[accepted],)
    committed = committed[:remaining_tokens]
    stop_ids = frozenset(stop_token_ids)
    stop_index = next(
        (
            index
            for index, token_id in enumerate(committed)
            if token_id in stop_ids
        ),
        None,
    )
    return committed if stop_index is None else committed[: stop_index + 1]


def _time_once(mx: Any, fn: Callable[[], Any]) -> float:
    mx.synchronize()
    started = time.perf_counter()
    value = fn()
    mx.eval(value)
    mx.synchronize()
    return (time.perf_counter() - started) * 1000.0


def _measure_timings(
    mx: Any,
    *,
    oracle_fn: Callable[[], Any],
    native_fn: Callable[[], Any],
    warmups: int,
    trials: int,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    cold_call_ms = {
        "oracle": _time_once(mx, oracle_fn),
        "native": _time_once(mx, native_fn),
    }
    for _ in range(warmups):
        mx.eval(oracle_fn(), native_fn())
    mx.synchronize()

    samples = {"oracle": [], "native": []}
    for trial in range(trials):
        order = (
            (("oracle", oracle_fn), ("native", native_fn))
            if trial % 2 == 0
            else (("native", native_fn), ("oracle", oracle_fn))
        )
        for name, fn in order:
            samples[name].append(_time_once(mx, fn))
    return (
        {
            name: _timing_summary(values)
            for name, values in samples.items()
        },
        cold_call_ms,
    )


def _measure_variant_timings(
    mx: Any,
    *,
    fn: Callable[[], Any],
    warmups: int,
    trials: int,
) -> dict[str, float]:
    for _ in range(warmups):
        mx.eval(fn())
    mx.synchronize()
    return _timing_summary([_time_once(mx, fn) for _ in range(trials)])


def _measure_memory(
    mx: Any,
    fn: Callable[[], Any],
) -> tuple[np.ndarray, int, int]:
    mx.synchronize()
    mx.clear_cache()
    mx.synchronize()
    before = int(mx.get_active_memory())
    mx.reset_peak_memory()
    value = fn()
    mx.eval(value)
    mx.synchronize()
    active_delta = int(mx.get_active_memory()) - before
    peak_delta = int(mx.get_peak_memory()) - before
    host_value = np.array(value.astype(mx.float32))
    del value
    mx.clear_cache()
    mx.synchronize()
    return host_value, active_delta, peak_delta


def _implementation_commit() -> str:
    repository = Path(__file__).resolve().parents[4]
    paths = (
        "python/src/edllm/diagnostics/metal_target_handoff.py",
        "python/src/edllm/diagnostics/metal_target_handoff_a0.py",
    )
    head = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if head.returncode != 0:
        raise reference.ReferenceEnvironmentError(
            "implementation_commit_unavailable",
            head.stderr.strip() or "cannot resolve the EDLLM implementation commit",
        )
    dirty = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain", "--", *paths],
        check=False,
        capture_output=True,
        text=True,
    )
    if dirty.returncode != 0 or dirty.stdout.strip():
        raise reference.ReferenceEnvironmentError(
            "implementation_dirty",
            "projection implementation must be committed before measurement",
        )
    return head.stdout.strip()


def run_a0_projection_probe(
    upstream_source_dir: str | Path,
    *,
    config: A0ProjectionProbeConfig | None = None,
    cache_dir: str | Path | None = None,
) -> ProjectionEvidence:
    config = config or A0ProjectionProbeConfig()
    reference._verify_host()
    source_dir = reference._verify_upstream_source(
        Path(upstream_source_dir).expanduser()
    )
    reference._verify_distribution_version("mlx", reference.MLX_VERSION)
    reference._verify_distribution_version("mlx-lm", reference.MLX_LM_VERSION)
    implementation_commit = _implementation_commit()

    try:
        import mlx.core as mx
        from dflash import model_mlx
        from huggingface_hub import snapshot_download
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler
    except ImportError as exc:
        raise reference.ReferenceEnvironmentError(
            "optional_dependency_missing",
            f"failed to import the pinned MLX projection probe: {exc}",
        ) from exc

    imported_source = Path(model_mlx.__file__).resolve()
    if source_dir not in imported_source.parents:
        raise reference.ReferenceEnvironmentError(
            "upstream_source_mismatch",
            f"dflash imported from {imported_source}, expected {source_dir}",
        )

    resolved_cache_dir = (
        Path(cache_dir).expanduser() if cache_dir is not None else None
    )
    target_path = reference._resolve_snapshot(
        snapshot_download,
        reference.TARGET_MODEL_ID,
        reference.TARGET_REVISION,
        cache_dir=resolved_cache_dir,
        allow_patterns=("*.json", "*.jinja", "*.safetensors", "*.txt"),
    )
    draft_path = reference._resolve_snapshot(
        snapshot_download,
        reference.DRAFT_MODEL_ID,
        reference.DRAFT_REVISION,
        cache_dir=resolved_cache_dir,
        allow_patterns=("*.json", "*.safetensors"),
    )
    reference_config = reference.DFlashReferenceConfig(
        user_prompt=config.user_prompt,
        block_size=16,
        max_tokens=32,
        warmup_tokens=3,
        temperature=0.0,
        enable_thinking=True,
    )
    draft_config_digest = reference._validate_draft_snapshot(
        draft_path,
        reference_config,
    )
    target_config_digest = reference._config_digest(target_path / "config.json")

    try:
        model, tokenizer = model_mlx.load(str(target_path))
        with reference._pinned_draft_loader(model_mlx, draft_path):
            draft = model_mlx.load_draft(reference.DRAFT_MODEL_ID)
    except Exception as exc:
        raise reference.ReferenceEnvironmentError(
            "model_load_failed",
            f"failed to load the pinned projection pair: "
            f"{type(exc).__name__}: {exc}",
        ) from exc

    target_layer_ids = tuple(int(value) for value in draft.config.target_layer_ids)
    model_mlx._patch_model(model, target_layer_ids)
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": config.user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    add_special_tokens = (
        tokenizer.bos_token is None
        or not prompt_text.startswith(tokenizer.bos_token)
    )
    prompt_token_ids = tuple(
        int(token_id)
        for token_id in tokenizer.encode(
            prompt_text,
            add_special_tokens=add_special_tokens,
        )
    )
    stop_token_ids = tuple(
        sorted(int(token_id) for token_id in tokenizer.eos_token_ids)
    )

    try:
        sampler = make_sampler(temp=0.0)
        target_cache = make_prompt_cache(model)
        prompt = mx.array(prompt_token_ids)[None]
        with mx.stream(model_mlx.generation_stream):
            logits = model(prompt, target_cache)
            prompt_hidden_layers = tuple(model._hidden_states)
        if len(prompt_hidden_layers) != len(target_layer_ids) or any(
            hidden is None for hidden in prompt_hidden_layers
        ):
            raise RuntimeError("prompt target layer capture is incomplete")
        mx.eval(logits, *prompt_hidden_layers, draft.fc.weight)

        draft.bind(model)
        draft_cache = make_prompt_cache(draft)
        prompt_hidden = mx.concatenate(prompt_hidden_layers, axis=-1)
        bootstrap_token = int(sampler(logits[:, -1:])[0, 0].item())
        block = mx.array(
            [
                [bootstrap_token]
                + [int(draft.config.mask_token_id)]
                * (reference_config.block_size - 1)
            ]
        )
        with mx.stream(model_mlx.generation_stream):
            draft_logits = draft(
                block,
                prompt_hidden,
                draft_cache,
                logits_start=1,
            )
            draft_tokens = sampler(draft_logits)
        mx.eval(draft_tokens)

        verify_input = mx.concatenate(
            [mx.array([[bootstrap_token]]), draft_tokens],
            axis=1,
        )
        with mx.stream(model_mlx.generation_stream):
            verify_logits = model(verify_input, target_cache)
            hidden_layers = tuple(model._hidden_states)
            target_tokens = sampler(verify_logits)
        if len(hidden_layers) != len(target_layer_ids) or any(
            hidden is None for hidden in hidden_layers
        ):
            raise RuntimeError("verify target layer capture is incomplete")
        mx.eval(verify_logits, target_tokens, *hidden_layers)

        draft_token_ids = tuple(int(value) for value in draft_tokens[0].tolist())
        target_token_ids = tuple(int(value) for value in target_tokens[0].tolist())
        committed_token_ids = _committed_round_tokens(
            draft_token_ids,
            target_token_ids,
            stop_token_ids=stop_token_ids,
            remaining_tokens=reference_config.max_tokens - 1,
        )
        logical_length = len(committed_token_ids)
        del block, draft_cache, draft_logits, draft_tokens, logits
        del prompt, prompt_hidden, prompt_hidden_layers, target_cache
        del target_tokens, verify_input, verify_logits
        mx.clear_cache()
        mx.synchronize()

        prepare_started = time.perf_counter()
        prepared = prepare_projection_weights(
            draft.fc.weight,
            layer_count=len(hidden_layers),
            hidden_size=int(hidden_layers[0].shape[-1]),
        )
        mx.synchronize()
        prepare_ms = (time.perf_counter() - prepare_started) * 1000.0

        sequence_length = int(hidden_layers[0].shape[1])
        if sequence_length != reference_config.block_size:
            raise RuntimeError(
                "verify capture length does not match the paired block size"
            )

        def oracle_projection():
            return draft.fc(mx.concatenate(hidden_layers, axis=-1))

        def native_projection():
            return native_split_projection(
                hidden_layers,
                prepared,
                start=0,
                length=logical_length,
                target_layer_ids=target_layer_ids,
                expected_target_layer_ids=tuple(draft.config.target_layer_ids),
            ).values

        timing_summaries, cold_call_ms = _measure_timings(
            mx,
            oracle_fn=oracle_projection,
            native_fn=native_projection,
            warmups=config.warmups,
            trials=config.trials,
        )
        oracle_np, oracle_active_delta, oracle_peak_delta = _measure_memory(
            mx,
            oracle_projection,
        )
        native_np, native_active_delta, native_peak_delta = _measure_memory(
            mx,
            native_projection,
        )
        concat_timing = _measure_variant_timings(
            mx,
            fn=lambda: mx.concatenate(hidden_layers, axis=-1),
            warmups=config.warmups,
            trials=config.trials,
        )
        concatenated_hidden = mx.concatenate(hidden_layers, axis=-1)
        mx.eval(concatenated_hidden)
        # noqa on the closure: `concatenated_hidden` is deleted below to free
        # the buffer, which makes ruff treat it as unbound inside the lambda.
        # _measure_variant_timings CALLS fn before returning, so every
        # execution happens while the name is still bound.
        range_timing = _measure_variant_timings(
            mx,
            fn=lambda: concatenated_hidden[  # noqa: F821
                :, :logical_length, :
            ],
            warmups=config.warmups,
            trials=config.trials,
        )
        del concatenated_hidden
        mx.clear_cache()
        mx.synchronize()
    except Exception as exc:
        raise reference.ReferenceExecutionError(
            "projection_probe_failed",
            f"the pinned projection probe failed: {type(exc).__name__}: {exc}",
        ) from exc

    geometry = native_split_projection(
        hidden_layers,
        prepared,
        start=0,
        length=logical_length,
        target_layer_ids=target_layer_ids,
        expected_target_layer_ids=tuple(draft.config.target_layer_ids),
    ).geometry
    comparison = _comparison(oracle_np, native_np, dtype=geometry.dtype)
    reasons = _correctness_reasons(comparison)
    promotion_reasons = []
    if reasons:
        promotion_reasons.append("correctness_gate_failed")
    if (
        timing_summaries["native"]["p50_ms"]
        >= timing_summaries["oracle"]["p50_ms"]
    ):
        promotion_reasons.append("native_projection_not_faster")
    if native_peak_delta >= oracle_peak_delta:
        promotion_reasons.append("native_peak_memory_not_lower")

    timing_values: dict[str, float] = {
        "weight_prepare_once_ms": prepare_ms,
        "oracle_cold_call_ms": cold_call_ms["oracle"],
        "native_cold_call_ms": cold_call_ms["native"],
    }
    for variant, summary in timing_summaries.items():
        for name, value in summary.items():
            timing_values[f"{variant}_{name}"] = value
    for name, value in concat_timing.items():
        timing_values[f"concat_only_{name}"] = value
    for name, value in range_timing.items():
        timing_values[f"logical_range_extraction_{name}"] = value
    memory_values = {
        "hidden_input_bytes": sum(int(value.nbytes) for value in hidden_layers),
        "oracle_concat_theoretical_bytes": sum(
            int(value.nbytes) for value in hidden_layers
        ),
        "prepared_weight_bytes": sum(
            int(value.nbytes) for value in prepared.chunks
        ),
        "oracle_active_delta": oracle_active_delta,
        "oracle_peak_delta": oracle_peak_delta,
        "native_active_delta": native_active_delta,
        "native_peak_delta": native_peak_delta,
    }
    device_info = mx.device_info()
    environment = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mlx_version": metadata.version("mlx"),
        "mlx_lm_version": metadata.version("mlx-lm"),
        "metal_device_name": str(device_info.get("device_name", "unknown")),
        "metal_architecture": str(device_info.get("architecture", "unknown")),
        "edllm_implementation_commit": implementation_commit,
        "upstream_source_dir": str(source_dir),
        "upstream_commit": reference.UPSTREAM_COMMIT,
        "upstream_import_path": str(imported_source),
        "target_model_id": reference.TARGET_MODEL_ID,
        "target_revision": reference.TARGET_REVISION,
        "target_snapshot_path": str(target_path),
        "target_config_sha256": target_config_digest,
        "draft_model_id": reference.DRAFT_MODEL_ID,
        "draft_revision": reference.DRAFT_REVISION,
        "draft_snapshot_path": str(draft_path),
        "draft_config_sha256": draft_config_digest,
        "prompt_token_sha256": array_digest(
            np.asarray(prompt_token_ids, dtype=np.int64)
        ),
        "digest_comparison_dtype": "float32",
        "measurement_order": "alternating_oracle_native",
        "capture_phase": "first_speculative_target_verify",
        "bootstrap_token_id": str(bootstrap_token),
        "draft_proposal_token_count": str(len(draft_token_ids)),
        "accepted_commit_length": str(logical_length),
        "warmups": str(config.warmups),
        "trials": str(config.trials),
    }
    round_summary = {
        "bootstrap_token_id": bootstrap_token,
        "draft_proposal_token_ids": list(draft_token_ids),
        "target_verify_token_ids": list(target_token_ids),
        "stop_token_ids": list(stop_token_ids),
        "committed_token_ids": list(committed_token_ids),
        "logical_range_is_full_sequence": logical_length == sequence_length,
    }
    return ProjectionEvidence(
        verdict="pass" if not reasons else "reference_failed",
        promotion_verdict=(
            "candidate" if not promotion_reasons else "do_not_promote"
        ),
        geometry=geometry,
        oracle_digest=array_digest(oracle_np),
        native_digest=array_digest(native_np),
        max_abs_error=comparison["max_abs_error"],
        max_relative_error=comparison["max_relative_error"],
        comparison={
            "all_finite": comparison["all_finite"],
            "within_tolerance": comparison["within_tolerance"],
            "rtol": comparison["rtol"],
            "atol": comparison["atol"],
        },
        timings_ms=timing_values,
        memory_bytes=memory_values,
        environment=environment,
        round_summary=round_summary,
        reasons=tuple(reasons),
        promotion_reasons=tuple(promotion_reasons),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure the pinned A0 DFlash target-feature projection."
    )
    parser.add_argument("--upstream-source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--trials", type=int, default=30)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        evidence = run_a0_projection_probe(
            args.upstream_source,
            config=A0ProjectionProbeConfig(
                warmups=args.warmups,
                trials=args.trials,
            ),
            cache_dir=args.cache_dir,
        )
        payload = evidence.to_dict()
    except reference.ReferenceEnvironmentError as exc:
        payload = {
            "schema_version": 1,
            "verdict": "environment_blocked",
            "error": {"code": exc.code, "message": str(exc)},
        }
    except reference.ReferenceExecutionError as exc:
        payload = {
            "schema_version": 1,
            "verdict": "reference_failed",
            "error": {"code": exc.code, "message": str(exc)},
        }

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0 if payload["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
