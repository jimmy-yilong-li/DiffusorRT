"""Optional DFlash-on-MLX reference diagnostic.

The report and runner in this module are dependency-free.  The real MLX
adapter is loaded lazily so importing :mod:`edllm` never requires DFlash,
MLX, mlx-lm, or huggingface-hub.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Iterator, Literal, Mapping, Protocol, Sequence


UPSTREAM_REPOSITORY = "https://github.com/z-lab/dflash.git"
UPSTREAM_COMMIT = "94e4abc5e0c31b67bc1a9d30f1cc34ece28a8756"
TARGET_MODEL_ID = "Qwen/Qwen3.5-4B"
TARGET_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
DRAFT_MODEL_ID = "z-lab/Qwen3.5-4B-DFlash"
DRAFT_REVISION = "96899cc270945f554998309580b08a04a05a3187"
MLX_VERSION = "0.31.2"
MLX_LM_VERSION = "0.31.3"
FIXED_USER_PROMPT = "Explain why the sky appears blue in two concise sentences."

AcceptancePhase = Literal["bootstrap", "speculative", "terminal"]


class ReferenceEnvironmentError(RuntimeError):
    """Expected failure to prepare the optional reference environment."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class ReferenceExecutionError(RuntimeError):
    """Expected failure while executing an otherwise available reference."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class DFlashReferenceConfig:
    max_tokens: int = 32
    block_size: int = 16
    warmup_tokens: int = 3
    temperature: float = 0.0
    enable_thinking: bool = True
    user_prompt: str = FIXED_USER_PROMPT

    def validation_error(self) -> tuple[str, str] | None:
        if (
            not isinstance(self.max_tokens, int)
            or isinstance(self.max_tokens, bool)
            or self.max_tokens <= 1
        ):
            return "max_tokens_invalid", "max_tokens must be an integer greater than 1"
        if (
            not isinstance(self.block_size, int)
            or isinstance(self.block_size, bool)
            or self.block_size <= 1
        ):
            return "block_size_invalid", "block_size must be an integer greater than 1"
        if (
            not isinstance(self.warmup_tokens, int)
            or isinstance(self.warmup_tokens, bool)
            or self.warmup_tokens <= 0
        ):
            return "warmup_tokens_invalid", "warmup_tokens must be a positive integer"
        if (
            not isinstance(self.temperature, (int, float))
            or isinstance(self.temperature, bool)
            or not math.isfinite(float(self.temperature))
            or float(self.temperature) != 0.0
        ):
            return "temperature_invalid", "A0 requires greedy temperature 0"
        if not isinstance(self.enable_thinking, bool) or not self.enable_thinking:
            return "thinking_invalid", "A0 requires the supported thinking=True mode"
        if not isinstance(self.user_prompt, str) or not self.user_prompt:
            return "prompt_invalid", "user_prompt must be a non-empty string"
        return None

    def to_dict(self) -> dict[str, object]:
        return {
            "max_tokens": self.max_tokens,
            "block_size": self.block_size,
            "warmup_tokens": self.warmup_tokens,
            "temperature": float(self.temperature),
            "enable_thinking": self.enable_thinking,
            "user_prompt": self.user_prompt,
        }


@dataclass(frozen=True, slots=True)
class AcceptanceEvent:
    phase: AcceptancePhase
    raw_accepted_progress: int
    committed_token_count: int
    finish_reason: str | None = None

    def validation_error(self) -> str | None:
        if self.phase not in {"bootstrap", "speculative", "terminal"}:
            return "acceptance_phase_invalid"
        for value in (self.raw_accepted_progress, self.committed_token_count):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                return "acceptance_count_invalid"
        if self.phase == "terminal" and self.committed_token_count != 0:
            return "terminal_event_has_tokens"
        return None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "phase": self.phase,
            "raw_accepted_progress": self.raw_accepted_progress,
            "committed_token_count": self.committed_token_count,
        }
        if self.phase == "speculative":
            payload["normalized_draft_prefix_accepted"] = max(
                self.raw_accepted_progress - 1,
                0,
            )
        if self.finish_reason is not None:
            payload["finish_reason"] = self.finish_reason
        return payload


@dataclass(frozen=True, slots=True)
class GenerationTrace:
    token_ids: tuple[int, ...]
    elapsed_seconds: float
    reported_generation_tps: float | None = None
    acceptance_events: tuple[AcceptanceEvent, ...] = ()
    finish_reason: str | None = None

    def validation_error(self, *, require_acceptance: bool) -> str | None:
        if not self.token_ids:
            return "output_tokens_empty"
        if any(
            not isinstance(token_id, int)
            or isinstance(token_id, bool)
            or token_id < 0
            for token_id in self.token_ids
        ):
            return "output_token_invalid"
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds <= 0:
            return "elapsed_seconds_invalid"
        if self.reported_generation_tps is not None and (
            not math.isfinite(self.reported_generation_tps)
            or self.reported_generation_tps <= 0
        ):
            return "reported_generation_tps_invalid"
        if require_acceptance:
            if not self.acceptance_events:
                return "acceptance_events_missing"
            for event in self.acceptance_events:
                if error := event.validation_error():
                    return error
            phases = tuple(event.phase for event in self.acceptance_events)
            if phases[0] != "bootstrap" or phases.count("bootstrap") != 1:
                return "bootstrap_event_count_invalid"
            bootstrap = self.acceptance_events[0]
            if (
                bootstrap.raw_accepted_progress != 1
                or bootstrap.committed_token_count != 1
            ):
                return "bootstrap_event_shape_invalid"
            if phases.count("terminal") > 1 or (
                "terminal" in phases and phases[-1] != "terminal"
            ):
                return "terminal_event_order_invalid"
            if phases[-1] == "terminal":
                terminal = self.acceptance_events[-1]
                if terminal.raw_accepted_progress != 0:
                    return "terminal_event_shape_invalid"
            if not any(event.phase == "speculative" for event in self.acceptance_events):
                return "speculative_round_missing"
            if any(
                event.phase == "speculative"
                and (
                    event.raw_accepted_progress < 1
                    or event.committed_token_count < 1
                )
                for event in self.acceptance_events
            ):
                return "speculative_event_shape_invalid"
            if sum(
                event.committed_token_count
                for event in self.acceptance_events
                if event.phase != "terminal"
            ) != len(self.token_ids):
                return "acceptance_committed_count_mismatch"
        elif self.acceptance_events:
            return "target_trace_has_acceptance_events"
        return None

    @property
    def measured_tps(self) -> float:
        return len(self.token_ids) / self.elapsed_seconds

    def to_dict(self, *, include_acceptance: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "token_ids": list(self.token_ids),
            "token_count": len(self.token_ids),
            "elapsed_seconds": self.elapsed_seconds,
            "measured_tokens_per_second": self.measured_tps,
        }
        if self.reported_generation_tps is not None:
            payload["reported_generation_tps"] = self.reported_generation_tps
        if self.finish_reason is not None:
            payload["finish_reason"] = self.finish_reason
        if include_acceptance:
            events = [event.to_dict() for event in self.acceptance_events]
            payload["acceptance_events"] = events
            payload["speculative_round_count"] = sum(
                event.phase == "speculative" for event in self.acceptance_events
            )
        return payload


@dataclass(frozen=True, slots=True)
class ReferenceObservation:
    environment: Mapping[str, object]
    prompt_token_ids: tuple[int, ...]
    stop_token_ids: tuple[int, ...]
    target_runs: tuple[GenerationTrace, GenerationTrace]
    dflash_run: GenerationTrace


class ReferenceAdapter(Protocol):
    def run(self, config: DFlashReferenceConfig) -> ReferenceObservation: ...


@dataclass(frozen=True, slots=True)
class DFlashReferenceReport:
    verdict: str
    reasons: tuple[str, ...]
    config: DFlashReferenceConfig
    environment: Mapping[str, object] = field(default_factory=dict)
    prompt: Mapping[str, object] = field(default_factory=dict)
    target_runs: tuple[GenerationTrace, ...] = ()
    dflash_run: GenerationTrace | None = None
    comparison: Mapping[str, object] = field(default_factory=dict)
    smoke_timing: Mapping[str, object] = field(default_factory=dict)
    error: Mapping[str, str] | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": 1,
            "verdict": self.verdict,
            "reasons": list(self.reasons),
            "reference": {
                "upstream_repository": UPSTREAM_REPOSITORY,
                "upstream_commit": UPSTREAM_COMMIT,
                "target_model_id": TARGET_MODEL_ID,
                "target_revision": TARGET_REVISION,
                "draft_model_id": DRAFT_MODEL_ID,
                "draft_revision": DRAFT_REVISION,
                "mlx_version": MLX_VERSION,
                "mlx_lm_version": MLX_LM_VERSION,
            },
            "config": self.config.to_dict(),
            "environment": dict(self.environment),
            "prompt": dict(self.prompt),
            "target_runs": [
                trace.to_dict(include_acceptance=False) for trace in self.target_runs
            ],
            "dflash_run": (
                self.dflash_run.to_dict(include_acceptance=True)
                if self.dflash_run is not None
                else None
            ),
            "comparison": dict(self.comparison),
            "smoke_timing": dict(self.smoke_timing),
        }
        if self.error is not None:
            payload["error"] = dict(self.error)
        return payload

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        ) + "\n"


def run_dflash_reference(
    adapter: ReferenceAdapter,
    config: DFlashReferenceConfig | None = None,
) -> DFlashReferenceReport:
    config = config or DFlashReferenceConfig()
    if error := config.validation_error():
        code, message = error
        return DFlashReferenceReport(
            verdict="invalid_input",
            reasons=(code,),
            config=config,
            error={"code": code, "message": message},
        )

    try:
        observation = adapter.run(config)
    except ReferenceEnvironmentError as exc:
        return DFlashReferenceReport(
            verdict="environment_blocked",
            reasons=(exc.code,),
            config=config,
            error={"code": exc.code, "message": str(exc)},
        )
    except ReferenceExecutionError as exc:
        return DFlashReferenceReport(
            verdict="reference_failed",
            reasons=(exc.code,),
            config=config,
            error={"code": exc.code, "message": str(exc)},
        )
    return _build_report(config, observation)


def _build_report(
    config: DFlashReferenceConfig,
    observation: ReferenceObservation,
) -> DFlashReferenceReport:
    reasons: list[str] = []
    if len(observation.target_runs) != 2:
        reasons.append("target_run_count_invalid")
    else:
        for trace in observation.target_runs:
            if error := trace.validation_error(require_acceptance=False):
                reasons.append(error)
                break
    if error := observation.dflash_run.validation_error(require_acceptance=True):
        reasons.append(error)
    if any(
        event.phase == "speculative"
        and event.raw_accepted_progress > config.block_size
        for event in observation.dflash_run.acceptance_events
    ):
        reasons.append("acceptance_exceeds_block_size")
    if not observation.prompt_token_ids:
        reasons.append("prompt_tokens_empty")
    if any(
        not isinstance(token_id, int)
        or isinstance(token_id, bool)
        or token_id < 0
        for token_id in observation.prompt_token_ids
    ):
        reasons.append("prompt_token_invalid")
    if any(
        not isinstance(token_id, int)
        or isinstance(token_id, bool)
        or token_id < 0
        for token_id in observation.stop_token_ids
    ):
        reasons.append("stop_token_invalid")
    if not observation.stop_token_ids:
        reasons.append("stop_tokens_empty")

    target_stable = (
        len(observation.target_runs) == 2
        and observation.target_runs[0].token_ids == observation.target_runs[1].token_ids
    )
    if not target_stable:
        reasons.append("target_output_unstable")

    reference_tokens = (
        observation.target_runs[0].token_ids if observation.target_runs else ()
    )
    dflash_tokens = observation.dflash_run.token_ids
    exact_match = reference_tokens == dflash_tokens
    if not exact_match:
        reasons.append("target_dflash_token_mismatch")
    first_divergent_position = _first_divergent_position(
        reference_tokens,
        dflash_tokens,
    )

    prompt_ids = tuple(observation.prompt_token_ids)
    prompt_digest = hashlib.sha256(
        json.dumps(list(prompt_ids), separators=(",", ":")).encode("ascii")
    ).hexdigest()
    comparison = {
        "target_runs_token_identical": target_stable,
        "target_dflash_token_identical": exact_match,
        "first_divergent_position": first_divergent_position,
    }

    smoke_timing: dict[str, object] = {"decision_grade": False}
    if (
        exact_match
        and not reasons
        and observation.target_runs
        and observation.target_runs[0].elapsed_seconds > 0
        and observation.dflash_run.elapsed_seconds > 0
    ):
        target_seconds = sum(
            trace.elapsed_seconds for trace in observation.target_runs
        ) / len(observation.target_runs)
        dflash_seconds = observation.dflash_run.elapsed_seconds
        smoke_timing.update(
            {
                "target_wall_seconds_mean": target_seconds,
                "dflash_wall_seconds": dflash_seconds,
                "target_over_dflash_wall_time_ratio": target_seconds
                / dflash_seconds,
            }
        )

    return DFlashReferenceReport(
        verdict="pass" if not reasons else "reference_failed",
        reasons=tuple(dict.fromkeys(reasons)),
        config=config,
        environment=observation.environment,
        prompt={
            "token_ids": list(prompt_ids),
            "token_count": len(prompt_ids),
            "token_ids_sha256": prompt_digest,
            "stop_token_ids": list(observation.stop_token_ids),
        },
        target_runs=observation.target_runs,
        dflash_run=observation.dflash_run,
        comparison=comparison,
        smoke_timing=smoke_timing,
    )


def _first_divergent_position(
    left: tuple[int, ...],
    right: tuple[int, ...],
) -> int | None:
    # strict=False deliberately: the loop finds the first differing TOKEN, and
    # a length difference is handled explicitly below as a divergence at the
    # shorter length. strict=True would raise instead of reporting it.
    for index, (left_token, right_token) in enumerate(zip(left, right, strict=False)):
        if left_token != right_token:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


class UpstreamMLXAdapter:
    """Run the pinned upstream MLX reference without importing it eagerly."""

    def __init__(
        self,
        upstream_source_dir: str | Path,
        *,
        cache_dir: str | Path | None = None,
    ) -> None:
        self._upstream_source_dir = Path(upstream_source_dir).expanduser()
        self._cache_dir = (
            Path(cache_dir).expanduser() if cache_dir is not None else None
        )

    def run(self, config: DFlashReferenceConfig) -> ReferenceObservation:
        _verify_host()
        source_dir = _verify_upstream_source(self._upstream_source_dir)
        _verify_distribution_version("mlx", MLX_VERSION)
        _verify_distribution_version("mlx-lm", MLX_LM_VERSION)

        try:
            import mlx.core as mx
            from dflash import model_mlx
            from huggingface_hub import snapshot_download
            from mlx_lm import stream_generate as target_stream_generate
            from mlx_lm.sample_utils import make_sampler
        except ImportError as exc:
            raise ReferenceEnvironmentError(
                "optional_dependency_missing",
                f"failed to import the pinned MLX reference: {exc}",
            ) from exc

        imported_source = Path(model_mlx.__file__).resolve()
        if source_dir not in imported_source.parents:
            raise ReferenceEnvironmentError(
                "upstream_source_mismatch",
                f"dflash imported from {imported_source}, expected {source_dir}",
            )

        target_path = _resolve_snapshot(
            snapshot_download,
            TARGET_MODEL_ID,
            TARGET_REVISION,
            cache_dir=self._cache_dir,
            allow_patterns=(
                "*.json",
                "*.jinja",
                "*.safetensors",
                "*.txt",
            ),
        )
        draft_path = _resolve_snapshot(
            snapshot_download,
            DRAFT_MODEL_ID,
            DRAFT_REVISION,
            cache_dir=self._cache_dir,
            allow_patterns=("*.json", "*.safetensors"),
        )
        draft_config_digest = _validate_draft_snapshot(draft_path, config)
        target_config_digest = _config_digest(target_path / "config.json")

        try:
            model, tokenizer = model_mlx.load(str(target_path))
            with _pinned_draft_loader(model_mlx, draft_path):
                draft = model_mlx.load_draft(DRAFT_MODEL_ID)
        except Exception as exc:
            raise ReferenceEnvironmentError(
                "model_load_failed",
                f"failed to load the pinned target/draft pair: "
                f"{type(exc).__name__}: {exc}",
            ) from exc

        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": config.user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=config.enable_thinking,
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
            warmup_prompt = tokenizer.encode("Hi")
            list(
                model_mlx.stream_generate(
                    model,
                    draft,
                    tokenizer,
                    warmup_prompt,
                    block_size=config.block_size,
                    max_tokens=config.warmup_tokens,
                    sampler=make_sampler(temp=0.0),
                )
            )
            mx.synchronize()

            target_runs = tuple(
                _run_target_once(
                    mx=mx,
                    stream_generate=target_stream_generate,
                    make_sampler=make_sampler,
                    model=model,
                    tokenizer=tokenizer,
                    prompt_token_ids=prompt_token_ids,
                    max_tokens=config.max_tokens,
                )
                for _ in range(2)
            )
            dflash_run = _run_dflash_once(
                mx=mx,
                stream_generate=model_mlx.stream_generate,
                make_sampler=make_sampler,
                model=model,
                draft=draft,
                tokenizer=tokenizer,
                prompt_token_ids=prompt_token_ids,
                max_tokens=config.max_tokens,
                block_size=config.block_size,
            )
        except Exception as exc:
            raise ReferenceExecutionError(
                "reference_execution_failed",
                f"the pinned MLX reference failed: {type(exc).__name__}: {exc}",
            ) from exc

        return ReferenceObservation(
            environment={
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "mlx_version": metadata.version("mlx"),
                "mlx_lm_version": metadata.version("mlx-lm"),
                "upstream_source_dir": str(source_dir),
                "upstream_import_path": str(imported_source),
                "target_snapshot_path": str(target_path),
                "target_config_sha256": target_config_digest,
                "draft_snapshot_path": str(draft_path),
                "draft_config_sha256": draft_config_digest,
                "loader_pin_bridge": True,
            },
            prompt_token_ids=prompt_token_ids,
            stop_token_ids=stop_token_ids,
            target_runs=(target_runs[0], target_runs[1]),
            dflash_run=dflash_run,
        )


def _verify_host() -> None:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise ReferenceEnvironmentError(
            "apple_silicon_required",
            "the MLX reference requires macOS on arm64",
        )


def _verify_upstream_source(source_dir: Path) -> Path:
    try:
        resolved = source_dir.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ReferenceEnvironmentError(
            "upstream_source_missing",
            f"upstream source is unavailable: {source_dir}",
        ) from exc
    completed = subprocess.run(
        ["git", "-C", str(resolved), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ReferenceEnvironmentError(
            "upstream_source_not_git",
            completed.stderr.strip() or f"not a git checkout: {resolved}",
        )
    actual_commit = completed.stdout.strip()
    if actual_commit != UPSTREAM_COMMIT:
        raise ReferenceEnvironmentError(
            "upstream_commit_mismatch",
            f"expected {UPSTREAM_COMMIT}, got {actual_commit}",
        )
    dirty = subprocess.run(
        ["git", "-C", str(resolved), "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    if dirty.returncode != 0 or dirty.stdout.strip():
        raise ReferenceEnvironmentError(
            "upstream_source_dirty",
            f"upstream checkout must be clean: {resolved}",
        )
    return resolved


def _verify_distribution_version(distribution: str, expected: str) -> None:
    try:
        actual = metadata.version(distribution)
    except metadata.PackageNotFoundError as exc:
        raise ReferenceEnvironmentError(
            "optional_dependency_missing",
            f"missing distribution: {distribution}",
        ) from exc
    if actual != expected:
        raise ReferenceEnvironmentError(
            "optional_dependency_version_mismatch",
            f"{distribution} must be {expected}, got {actual}",
        )


def _resolve_snapshot(
    snapshot_download,
    repo_id: str,
    revision: str,
    *,
    cache_dir: Path | None,
    allow_patterns: Sequence[str],
) -> Path:
    try:
        path = Path(
            snapshot_download(
                repo_id,
                revision=revision,
                cache_dir=str(cache_dir) if cache_dir is not None else None,
                allow_patterns=list(allow_patterns),
            )
        ).resolve(strict=True)
    except Exception as exc:
        raise ReferenceEnvironmentError(
            "snapshot_resolution_failed",
            f"failed to resolve {repo_id}@{revision}: "
            f"{type(exc).__name__}: {exc}",
        ) from exc
    if path.name != revision:
        raise ReferenceEnvironmentError(
            "snapshot_revision_mismatch",
            f"expected snapshot {revision}, got {path}",
        )
    return path


def _validate_draft_snapshot(
    draft_path: Path,
    config: DFlashReferenceConfig,
) -> str:
    config_path = draft_path / "config.json"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReferenceEnvironmentError(
            "draft_config_invalid",
            f"cannot read draft config: {config_path}",
        ) from exc
    required = {
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "intermediate_size",
        "vocab_size",
        "rms_norm_eps",
        "rope_theta",
        "max_position_embeddings",
        "block_size",
        "num_target_layers",
        "dflash_config",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ReferenceEnvironmentError(
            "draft_config_incompatible",
            f"upstream MLX loader fields missing: {missing}",
        )
    if payload["block_size"] != config.block_size:
        raise ReferenceEnvironmentError(
            "draft_block_size_mismatch",
            f"expected block size {config.block_size}, got {payload['block_size']}",
        )
    dflash_config = payload["dflash_config"]
    if not isinstance(dflash_config, dict) or not {
        "mask_token_id",
        "target_layer_ids",
    }.issubset(dflash_config):
        raise ReferenceEnvironmentError(
            "draft_config_incompatible",
            "dflash_config lacks mask_token_id or target_layer_ids",
        )
    if not (draft_path / "model.safetensors").is_file():
        raise ReferenceEnvironmentError(
            "draft_weights_missing",
            f"draft weights are missing from {draft_path}",
        )
    return _config_digest(config_path)


def _config_digest(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReferenceEnvironmentError(
            "model_config_invalid",
            f"cannot read model config: {path}",
        ) from exc
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@contextmanager
def _pinned_draft_loader(model_mlx, draft_path: Path) -> Iterator[None]:
    original = model_mlx.snapshot_download

    def resolve_pinned(repo_id, *args, **kwargs):
        del args, kwargs
        if repo_id != DRAFT_MODEL_ID:
            raise ReferenceEnvironmentError(
                "draft_loader_repo_mismatch",
                f"unexpected draft repo requested: {repo_id}",
            )
        return str(draft_path)

    model_mlx.snapshot_download = resolve_pinned
    try:
        yield
    finally:
        model_mlx.snapshot_download = original


def _run_target_once(
    *,
    mx,
    stream_generate,
    make_sampler,
    model,
    tokenizer,
    prompt_token_ids: tuple[int, ...],
    max_tokens: int,
) -> GenerationTrace:
    mx.synchronize()
    started = time.perf_counter()
    responses = list(
        stream_generate(
            model,
            tokenizer,
            list(prompt_token_ids),
            max_tokens=max_tokens,
            sampler=make_sampler(temp=0.0),
        )
    )
    mx.synchronize()
    elapsed = time.perf_counter() - started
    if not responses:
        return GenerationTrace(token_ids=(), elapsed_seconds=elapsed)
    return GenerationTrace(
        token_ids=tuple(int(response.token) for response in responses),
        elapsed_seconds=elapsed,
        reported_generation_tps=float(responses[-1].generation_tps),
        finish_reason=responses[-1].finish_reason,
    )


def _run_dflash_once(
    *,
    mx,
    stream_generate,
    make_sampler,
    model,
    draft,
    tokenizer,
    prompt_token_ids: tuple[int, ...],
    max_tokens: int,
    block_size: int,
) -> GenerationTrace:
    mx.synchronize()
    started = time.perf_counter()
    responses = list(
        stream_generate(
            model,
            draft,
            tokenizer,
            list(prompt_token_ids),
            block_size=block_size,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=0.0),
        )
    )
    mx.synchronize()
    elapsed = time.perf_counter() - started
    return _dflash_trace_from_responses(responses, elapsed_seconds=elapsed)


def _dflash_trace_from_responses(
    responses: Sequence[object],
    *,
    elapsed_seconds: float,
) -> GenerationTrace:
    token_ids: list[int] = []
    events: list[AcceptanceEvent] = []
    reported_tps: float | None = None
    finish_reason: str | None = None
    for index, response in enumerate(responses):
        response_tokens = tuple(int(token_id) for token_id in response.tokens)
        token_ids.extend(response_tokens)
        response_finish = getattr(response, "finish_reason", None)
        if index == 0:
            phase: AcceptancePhase = "bootstrap"
        elif not response_tokens and response_finish is not None:
            phase = "terminal"
        else:
            phase = "speculative"
        events.append(
            AcceptanceEvent(
                phase=phase,
                raw_accepted_progress=int(response.accepted),
                committed_token_count=len(response_tokens),
                finish_reason=response_finish,
            )
        )
        reported_tps = float(response.generation_tps)
        if response_finish is not None:
            finish_reason = response_finish
    return GenerationTrace(
        token_ids=tuple(token_ids),
        elapsed_seconds=elapsed_seconds,
        reported_generation_tps=reported_tps,
        acceptance_events=tuple(events),
        finish_reason=finish_reason,
    )


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-source", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    report = run_dflash_reference(
        UpstreamMLXAdapter(
            args.upstream_source,
            cache_dir=args.cache_dir,
        )
    )
    text = report.to_json()
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report.verdict == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
