from __future__ import annotations

import json
import math
import re
import sys
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Literal

from .engine import Engine
from .model import _close_preserving_primary
from .result import EmptyGenerationError


@dataclass(slots=True)
class BenchmarkPrompt:
    id: str
    text: str
    max_tokens: int = 4


@dataclass(slots=True)
class BenchmarkMode:
    """One immutable load request; cache intervals are expectations, not overlays."""

    name: str
    package_dir: Path
    package_kind: str
    cache_refresh_interval: int | None = None
    backend: str = "auto"
    deployment_profile: str | None = None
    precision: str | None = None
    device_ordinal: int = 0
    device_memory_budget_bytes: int = 0
    trust_remote_code: bool = False


@dataclass(slots=True)
class BenchmarkResult:
    mode: str
    package_kind: str
    package_dir: Path
    prompt_id: str
    max_tokens: int
    cache_refresh_interval: int
    decode_loop_kind: str
    block_count: int
    peak_memory_bytes: int | None
    baseline_status: dict[str, bool]
    row_verified: bool
    cache_refresh_count: int
    visible_generated_token_ids: list[int]
    ttft_ms: float
    total_latency_ms: float
    tpf: float
    aup: float | None
    tokens_per_second: float
    text: str
    backend: str = ""
    deployment_profile: str = ""
    precision: str = ""


@dataclass(slots=True)
class BenchmarkReport:
    results: list[BenchmarkResult]


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """A frozen, zero-shot labelled request; the Engine applies its chat template."""

    task: str
    doc_id: int
    id: str | None
    prompt: str
    answer_kind: Literal["choice", "exact", "numeric"]
    gold: tuple[str, ...]
    max_tokens: int
    labels: tuple[str, ...] = ()
    choices: tuple[str, ...] = ()


SCORING_PROTOCOL = "labelled-generation-v5"
_NUMBER = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
_NUMERIC_VALUE = rf"{_NUMBER}(?:\s*/\s*{_NUMBER})?"
_NUMERIC_ANSWER = rf"(?:{_NUMERIC_VALUE}|<\s*{_NUMERIC_VALUE}\s*>)"
_NUMERIC_SUFFIX = r"[ \t]*(?:[a-zA-Z%]+(?:[ \t]+[a-zA-Z%]+)*)?[.!]?[ \t]*"


def _answer_text(text: str) -> str:
    # Remove complete, delimited Markdown emphasis, not arbitrary characters
    # inside an answer token ('1**2**3' must not become the number 123).
    return re.sub(r"(?<!\w)\*\*([^*\n]+)\*\*(?!\w)", r"\1", text).strip()


def _normalized_number(value: str) -> str | None:
    value = value.strip().removeprefix("<").removesuffix(">").strip()
    parts = value.replace(",", "").split("/")
    number = Decimal(parts[0].strip())
    if len(parts) == 2:
        denominator = Decimal(parts[1].strip())
        if not denominator:
            return None
        number /= denominator
    parsed = format(number, "f")
    return parsed.rstrip("0").rstrip(".") if "." in parsed else parsed


def _numeric_answer(text: str) -> str | None:
    # An explicit answer outranks later verification equations. Never infer the
    # answer from an arbitrary trailing '=' in a possibly unfinished derivation.
    text = _answer_text(text).translate(str.maketrans({"−": "-", "－": "-"}))
    # Match a complete answer token before removing its optional wrapper.
    # Global bracket deletion would join '<1>2' into the different number 12.
    value_pattern = (
        rf"[ \t]*\$?[ \t]*({_NUMERIC_ANSWER}){_NUMERIC_SUFFIX}"
        r"(?=\n|$|(?:check|verification)\s*[:=])"
    )
    declared = re.findall(
        r"(?:####|\b(?:the\s+)?(?:final\s+)?answer\b\s*(?:is\s*)?[:=]?)"
        + value_pattern,
        text, re.IGNORECASE,
    )
    if not declared:
        declared = re.findall(rf"\\boxed\{{\s*({_NUMERIC_VALUE})\s*\}}", text)
    if declared:
        values = {_normalized_number(value) for value in declared}
        return values.pop() if len(values) == 1 and None not in values else None
    bare = re.fullmatch(rf"\$?\s*({_NUMERIC_ANSWER}){_NUMERIC_SUFFIX}", text)
    if bare is not None:
        return _normalized_number(bare.group(1))
    final = re.search(
        rf"(?:^|\bso\s+)it\s+(?:costs?|is)\s*\$?\s*"
        rf"({_NUMERIC_ANSWER}){_NUMERIC_SUFFIX}$", text, re.IGNORECASE,
    )
    return _normalized_number(final.group(1)) if final is not None else None


def _choice_answer(case: EvaluationCase, text: str) -> str | None:
    def normalized(value: str) -> str:
        return _answer_text(value).casefold().rstrip(".! ")

    text = _answer_text(text)
    first_line = text.splitlines()[0].strip()

    # Whole-option matching comes first: 'E. coli' is an option's text, not a
    # declaration of label E. This also handles non-person WinoGrande answers.
    options = {
        label: normalized(choice)
        for label, choice in zip(case.labels, case.choices, strict=True)
    } if case.choices else {}
    found = [label for label, choice in options.items() if normalized(first_line) == choice]
    if len(found) > 1:
        return None
    labels = "|".join(re.escape(label) for label in case.labels)
    bare = re.fullmatch(rf"[\[(]?({labels})[\]).]?", first_line, re.IGNORECASE)
    # A declaration needs a delimited label, not the article in 'the answer is
    # a plant'. 'Because' can introduce an explanation after an explicit label.
    label_end = r"(?!\w)(?=\s*(?:$|[.!:,;\])]|because\b))"
    declaration = (
        rf"\b(?:(?:the\s+)?(?:(?:correct|final)\s+)?answer\s*(?:is\s*)?[:=]?|option)"
        rf"\s*[\[(]?({labels}){label_end}"
    )
    explicit = re.match(declaration, first_line, re.IGNORECASE)
    labelled = re.fullmatch(rf"[\[(]?({labels})[\]).:]\s*(.*)", first_line, re.IGNORECASE)
    # Punctuated labels with prose need the actual option text. Otherwise an
    # abbreviation can be mistaken for a label merely because of its period.
    if labelled is not None:
        label, payload = labelled.group(1).upper(), labelled.group(2)
        if payload and normalized(payload) != options.get(label):
            labelled = None
    match = bare or explicit or labelled
    if found or match is not None:
        label = found[0] if found else match.group(1).upper()
        # Explicitly conflicting answers are ambiguous. Do not scan bare letters
        # in option prose: 'vitamin C' is not a second answer declaration.
        declarations = re.findall(declaration, text, re.IGNORECASE)
        if any(other.upper() != label for other in declarations):
            return None
        return label
    return None


def score_evaluation_answer(case: EvaluationCase, text: str) -> tuple[str | None, bool]:
    """Read one answer under the frozen task parser, then compare with gold."""
    if case.answer_kind == "exact":
        from .evaluation import score_triviaqa_answer

        return score_triviaqa_answer(text, case.gold)
    text = text.strip()
    if not text:
        return None, False
    if case.answer_kind == "numeric":
        parsed = _numeric_answer(text)
        return parsed, parsed is not None and Decimal(parsed) in {
            Decimal(answer) for answer in case.gold
        }
    if case.answer_kind == "choice":
        parsed = _choice_answer(case, text)
        return parsed, parsed in case.gold
    raise ValueError(f"unknown evaluation answer kind: {case.answer_kind!r}")


def _process_peak_rss_bytes() -> int | None:
    if sys.platform not in ("darwin", "linux"):
        return None
    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024


def _measure_engine_request(engine: Engine, prompt: str, max_tokens: int) -> dict[str, Any]:
    """The shared measurement seam for direct and harness-driven requests."""
    mx = None
    backend = engine.backend
    if backend == "auto":
        backend = engine.capabilities().get("selected_backend")
    if backend == "mlx_local":
        import mlx.core as mx

        mx.synchronize()
        mx.reset_peak_memory()
    started = time.perf_counter()
    metrics: dict[str, Any] = {}
    text = ""
    finish_reason = None
    generation_error = None
    interrupted = False
    try:
        try:
            result = engine.generate(prompt, max_tokens=max_tokens, return_debug=True)
        except EmptyGenerationError as error:
            if error.result is None:
                raise
            # The public Engine refuses an empty answer. Evaluation must still
            # count it as empty output and retain its actual work/latency data.
            result = error.result
        text, finish_reason = result.text, result.finish_reason
        metrics = _metrics_from_result(result)
        if mx is not None:
            mx.synchronize()
    except (Exception, KeyboardInterrupt) as error:
        generation_error = f"{type(error).__name__}: {error}"
        interrupted = isinstance(error, KeyboardInterrupt)
    latency_ms = (time.perf_counter() - started) * 1000.0
    output_tokens = metrics.get("mlx_output_token_count", metrics.get("cuda_output_token_count"))
    if output_tokens is None and "visible_generated_token_ids" in metrics:
        output_tokens = len(metrics["visible_generated_token_ids"])
    return {
        "output": text, "generation_error": generation_error, "interrupted": interrupted,
        "finish_reason": finish_reason,
        "truncated": finish_reason == "length" if finish_reason else None,
        "remaining_masks": metrics.get(
            "mlx_remaining_mask_count", metrics.get("cuda_remaining_mask_count")
        ),
        "forwards": metrics.get("mlx_forward_count", metrics.get("cuda_forward_count")),
        "output_tokens": output_tokens, "latency_ms": latency_ms,
        "runtime_latency_ms": metrics.get("total_latency_ms"), "ttft_ms": None,
        "effective_tokens_per_second": (
            output_tokens * 1000.0 / latency_ms if output_tokens is not None else None
        ),
        "mlx_peak_active_bytes": mx.get_peak_memory() if mx is not None else None,
        "mlx_active_bytes": mx.get_active_memory() if mx is not None else None,
        "mlx_cache_bytes": mx.get_cache_memory() if mx is not None else None,
        "process_peak_rss_bytes": _process_peak_rss_bytes(),
        "decode_settings": {
            key: value for key, value in metrics.items() if key.startswith("mlx_diffusion_")
        },
        "stage_timings_ms": {
            key: value for key, value in metrics.items()
            if key.startswith("mlx_timing_") and key.endswith("_ms")
        },
    }


def _evaluation_row(
    case: EvaluationCase, measured: dict[str, Any], *,
    score: tuple[str | None, bool] | None = None,
) -> dict[str, Any]:
    text = measured["output"]
    remaining = measured["remaining_masks"]
    error_reason = None
    if measured["generation_error"] is not None:
        error_reason = "generation_error"
    elif remaining is not None and remaining > 0:
        error_reason = "residual_masks"
    elif not text.strip():
        error_reason = "empty"
    parsed, correct = (None, False) if error_reason else (
        score if score is not None else score_evaluation_answer(case, text)
    )
    if error_reason is None and parsed is None:
        error_reason = ("truncated" if measured["truncated"] else
                        "completion_unknown" if measured["truncated"] is None else "unparseable")
    row = {
        **measured, "task": case.task, "doc_id": case.doc_id, "id": case.id,
        "completion_observed": measured["truncated"] is not None,
        "outcome": "output_error" if error_reason else (
            "correct" if correct else "wrong_answer"
        ),
        "error_reason": error_reason, "parsed": parsed,
    }
    if case.answer_kind != "exact":
        row["gold"] = list(case.gold)
    return row


def _record_not_attempted(stream: Any, cases: list[EvaluationCase], attempted: set[tuple[str, int]]) -> None:
    """Complete a partial run's question ledger without inventing measurements."""
    for case in cases:
        if (case.task, case.doc_id) not in attempted:
            row = {"task": case.task, "doc_id": case.doc_id, "id": case.id,
                   "outcome": "not_attempted", "error_reason": None, "parsed": None,
                   "output": None, "latency_ms": None, "finish_reason": None,
                   "truncated": None, "remaining_masks": None, "forwards": None,
                   "output_tokens": None, "completion_observed": False}
            stream.write(json.dumps(row) + "\n")
    stream.flush()


def run_dataset_evaluation(
    *,
    engine: Engine,
    cases: list[EvaluationCase],
    output_dir: str | Path,
    metadata: dict[str, Any],
    on_result: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Measure serial public requests on a caller-owned Engine, retaining failures.

    This is generative answer scoring, not likelihood evaluation. The caller
    owns load/checked close and records those timings separately. MLX peak active
    allocation is reset per request; RSS is a cumulative process peak. A
    non-streaming Engine has no user-visible TTFT, so it is always null here.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Never overwrite a previous run's evidence.
    with (output_dir / "run.json").open("x") as stream:
        json.dump({
            **metadata, "scoring_protocol": SCORING_PROTOCOL,
            "selected_questions": [
                {"task": case.task, "doc_id": case.doc_id, "id": case.id}
                for case in cases
            ],
        }, stream, indent=2)
    rows: list[dict[str, Any]] = []
    primary_error: BaseException | None = None
    try:
        with (output_dir / "rows.jsonl").open("x") as stream:
            try:
                for case in cases:
                    measured = _measure_engine_request(engine, case.prompt, case.max_tokens)
                    row = _evaluation_row(case, measured)
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    rows.append(row)
                    if measured["interrupted"]:
                        raise KeyboardInterrupt("evaluation request interrupted")
                    if on_result is not None:
                        on_result(row)
            except BaseException as error:
                _close_preserving_primary(
                    lambda: _record_not_attempted(stream, cases, {(row["task"], row["doc_id"]) for row in rows}),
                    error, label="partial evaluation ledger",
                )
                raise
    except BaseException as error:
        primary_error = error
        raise
    finally:
        if primary_error is None:
            write_dataset_evaluation_report(output_dir, cases=cases)
        else:
            _close_preserving_primary(
                lambda: write_dataset_evaluation_report(output_dir, cases=cases),
                primary_error, label="evaluation report finalization",
            )
    return rows


def write_dataset_evaluation_report(
    output_dir: str | Path, *, cases: list[EvaluationCase] | None = None
) -> Path:
    """Regenerate from saved rows and IDs, without loading questions or a model.

    Historical v1 headers require their original cases; v2 headers retain the
    selected question IDs themselves. Scores are read, never recomputed here.
    """
    output_dir = Path(output_dir)
    rows = [json.loads(line) for line in (output_dir / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    metadata = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
    questions = ([{"task": case.task, "doc_id": case.doc_id, "id": case.id} for case in cases]
                 if cases is not None else metadata["selected_questions"])
    report_identity = {
        key: value for key, value in metadata.items()
        if key not in ("selected_questions", "selection", "task_definitions")
    }
    lines = [
        "# Labelled generation run", "",
        str(metadata.get("scope", "Generative answers under the recorded evaluation protocol.")), "",
        "TTFT is unavailable (non-streaming). MLX peak active bytes are per-request; "
        "process peak RSS is cumulative. Loading and close are recorded separately.", "",
        "## Run identity", "", "```json", json.dumps(report_identity, indent=2), "```", "",
        "Task definitions and the full selected-ID list are retained in run.json.", "",
        "## Task summary", "",
        "| Task | Selected | Attempted | Correct | Wrong | Output error | Completion unknown | Accuracy | p50 ms | p95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    details: list[str] = []
    for task in dict.fromkeys(question["task"] for question in questions):
        selected = [question for question in questions if question["task"] == task]
        group = [row for row in rows if row["task"] == task and row["outcome"] != "not_attempted"]
        correct = sum(row["outcome"] == "correct" for row in group)
        wrong = sum(row["outcome"] == "wrong_answer" for row in group)
        errors = sum(row["outcome"] == "output_error" for row in group)
        unknown = sum(row["truncated"] is None for row in group)
        latencies = [row["latency_ms"] for row in group]
        p50 = f"{_percentile(latencies, .50):.1f}" if latencies else "—"
        p95 = f"{_percentile(latencies, .95):.1f}" if latencies else "—"
        lines.append(
            f"| {task} | {len(selected)} | {len(group)} | {correct} | {wrong} | {errors} | "
            f"{unknown} | {correct / len(selected):.1%} | {p50} | {p95} |"
        )
        details.extend(["", f"## {task}", ""])
        attempted = {row["doc_id"] for row in group}
        details.append("Status: " + ("complete" if len(group) == len(selected) else "PARTIAL"))
        reasons = ("generation_error", "residual_masks", "empty", "truncated", "unparseable", "completion_unknown")
        counts = Counter(row["error_reason"] for row in group if row["outcome"] == "output_error")
        details.append("Output errors by reason: " + ", ".join(
            f"{reason} {counts[reason]}" for reason in reasons
        ) + ".")
        if len(selected) <= 500:
            details.append("Questions, in frozen order: " + ", ".join(
                str(question["id"] if question["id"] is not None else question["doc_id"])
                for question in selected
            ) + ".")
        else:
            details.append("Question IDs: see run.json and rows.jsonl.")
        details.extend(["", "Wrong answers (parsed -> gold):"])
        wrong_rows = [row for row in group if row["outcome"] == "wrong_answer"]
        if len(selected) > 500 and wrong_rows:
            details.append("- See rows.jsonl.")
        elif not wrong_rows:
            details.append("- None.")
        else:
            for row in wrong_rows:
                label = row["id"] if row["id"] is not None else row["doc_id"]
                details.append(f"- {label}: {row['parsed']!r} -> {row.get('gold', 'dataset aliases')!r}.")
        details.extend(["", "Output errors:"])
        for reason in reasons:
            ids = [str(row["id"] if row["id"] is not None else row["doc_id"])
                   for row in group if row["error_reason"] == reason]
            if ids:
                details.append(f"- {reason}: {', '.join(ids)}.")
        if not errors:
            details.append("- None.")
        missing = [str(question["id"] if question["id"] is not None else question["doc_id"])
                   for question in selected if question["doc_id"] not in attempted]
        details.append("Not attempted: " + (", ".join(missing) or "none") + ".")
        details.append("")
    lines.extend(details)
    path = output_dir / "report.md"
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8", newline="\n")
    return path


@dataclass(slots=True)
class MLXSkipComputeSample:
    mode: str
    prompt_id: str
    repeat_index: int
    cache_enabled: bool
    total_latency_ms: float
    forward_count: int
    layer0_projected_rows: int
    layer0_skipped_projection_rows: int
    baseline_status: dict[str, bool]
    row_verified: bool
    timing_metrics: dict[str, float | int | bool]
    text: str
    deployment_profile: str = ""
    precision: str = ""
    exact_reuses: int = 0
    peak_memory_bytes: int | None = None


@dataclass(slots=True)
class MLXSkipComputeSummary:
    mode: str
    prompt_id: str
    cache_enabled: bool
    run_count: int
    p50_latency_ms: float
    p95_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    total_projected_rows: int
    total_skipped_projection_rows: int


@dataclass(slots=True)
class MLXSkipComputeBenchmarkReport:
    samples: list[MLXSkipComputeSample]
    summaries: list[MLXSkipComputeSummary]
    speedup_claimed: bool = False


def default_benchmark_prompts() -> list[BenchmarkPrompt]:
    fixture_path = (
        Path(__file__).resolve().parents[3]
        / "fixtures"
        / "prompts"
        / "baseline_prompts.json"
    )
    if not fixture_path.exists():
        return [
            BenchmarkPrompt(
                id="known_prompt",
                text="Question: What is the capital of France? Answer:",
                max_tokens=4,
            ),
            BenchmarkPrompt(id="broad_prompt", text="Hello, ", max_tokens=4),
        ]
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    return [
        BenchmarkPrompt(
            id=str(item["id"]),
            text=str(item["text"]),
            max_tokens=int(item.get("max_tokens", 4)),
        )
        for item in payload["prompts"]
    ]


def default_benchmark_modes(
    *,
    stub_package_dir: str | Path,
    real_package_dir: str | Path | None = None,
    cache_refresh_interval: int | None = None,
) -> list[BenchmarkMode]:
    """Measure the stub and a real package's selected profile without changing it.

    Cache comparisons use explicitly configured BenchmarkModes. The historical
    interval argument now checks the real package's configuration; it does not
    create an implicit cache variant.
    """
    modes = [
        BenchmarkMode(
            name="stub_scaffold",
            package_dir=Path(stub_package_dir),
            package_kind="stub",
            cache_refresh_interval=0,
            backend="numpy_local",
            deployment_profile="baseline",
        )
    ]
    if real_package_dir is not None and Path(real_package_dir).exists():
        real_dir = Path(real_package_dir)
        modes.append(
            BenchmarkMode(
                name="real_package_profile",
                package_dir=real_dir,
                package_kind="execution_package",
                cache_refresh_interval=cache_refresh_interval,
            )
        )
    return modes


def _open_benchmark_mode(mode: BenchmarkMode) -> Engine:
    engine = Engine.from_package(
        mode.package_dir,
        backend=mode.backend,
        deployment_profile=mode.deployment_profile,
        precision=mode.precision,
        device_ordinal=mode.device_ordinal,
        device_memory_budget_bytes=mode.device_memory_budget_bytes,
        trust_remote_code=mode.trust_remote_code,
    )
    try:
        plan = engine._handle._resolved_model_plan
        interval = int(plan.model_config.get("diffusion_cache_refresh_interval", 0))
        if (
            mode.cache_refresh_interval is not None
            and interval != mode.cache_refresh_interval
        ):
            raise ValueError(
                "cache_refresh_interval is an expectation, not a runtime override: "
                f"mode={mode.name!r} expected={mode.cache_refresh_interval} actual={interval}; "
                "configure the package/profile before loading"
            )
    except BaseException as error:
        _close_preserving_primary(
            engine.close, error, label="benchmark engine cleanup"
        )
        raise
    return engine


def run_benchmark_matrix(
    *,
    modes: list[BenchmarkMode],
    prompts: list[BenchmarkPrompt],
) -> BenchmarkReport:
    """Run each profile sequentially through the installed product Engine seam."""
    results: list[BenchmarkResult] = []
    for mode in modes:
        engine = _open_benchmark_mode(mode)
        primary_error: BaseException | None = None
        try:
            plan = engine._handle._resolved_model_plan
            interval = int(plan.model_config.get("diffusion_cache_refresh_interval", 0))
            for prompt in prompts:
                result = engine.generate(
                    prompt.text,
                    max_tokens=prompt.max_tokens,
                    return_debug=True,
                )
                metrics = _metrics_from_result(result)
                _warn_if_cache_interval_did_not_refresh(
                    mode=mode,
                    metrics=metrics,
                    interval=interval,
                )
                results.append(
                    BenchmarkResult(
                        mode=mode.name,
                        package_kind=mode.package_kind,
                        package_dir=Path(mode.package_dir),
                        prompt_id=prompt.id,
                        max_tokens=int(prompt.max_tokens),
                        cache_refresh_interval=interval,
                        decode_loop_kind=str(metrics.get("decode_loop_kind", "")),
                        block_count=int(metrics.get("block_count", 0)),
                        peak_memory_bytes=_peak_memory_bytes(metrics),
                        baseline_status={
                            key: bool(value)
                            for key, value in dict(metrics.get("baseline_status", {})).items()
                        },
                        row_verified=bool(metrics.get("row_verified", False)),
                        cache_refresh_count=int(metrics.get("cache_refresh_count", 0)),
                        visible_generated_token_ids=[
                            int(token_id)
                            for token_id in list(metrics.get("visible_generated_token_ids", []))
                        ],
                        ttft_ms=float(metrics.get("ttft_ms", 0.0)),
                        total_latency_ms=float(metrics.get("total_latency_ms", 0.0)),
                        tpf=_tokens_per_forward(metrics),
                        aup=None,
                        tokens_per_second=_tokens_per_second(metrics),
                        text=str(result.text),
                        backend=plan.load.backend,
                        deployment_profile=plan.load.deployment_profile,
                        precision=plan.load.precision,
                    )
                )
        except BaseException as error:
            primary_error = error
            raise
        finally:
            if primary_error is None:
                engine.close()
            else:
                _close_preserving_primary(
                    engine.close,
                    primary_error,
                    label="benchmark engine cleanup",
                )
    return BenchmarkReport(results=results)


def run_mlx_skip_compute_latency_benchmark(
    *,
    package_dir: str | Path | None = None,
    prompts: list[BenchmarkPrompt],
    repeats: int = 5,
    warmup_runs: int = 0,
    generate_once: Callable[..., Any] | None = None,
    cache_off_mode: BenchmarkMode | None = None,
    cache_on_mode: BenchmarkMode | None = None,
) -> MLXSkipComputeBenchmarkReport:
    """Compare explicit MLX cache profiles with at most one resident model.

    Without an injected generator, both modes must be provided and already
    carry their cache configuration. No baseline/cache semantic equivalence or
    speedup is inferred here. The package_dir argument is retained only for the
    historical injected-generator interface.
    """
    if int(repeats) < 5:
        raise ValueError("mlx skip-compute benchmark requires at least 5 repeats per mode")
    if int(warmup_runs) < 0:
        raise ValueError("warmup_runs must be non-negative")

    if generate_once is None and (cache_off_mode is None or cache_on_mode is None):
        raise ValueError(
            "MLX cache comparison requires cache_off_mode and cache_on_mode; "
            "configure both packages/profiles before loading"
        )
    modes = (
        (
            cache_off_mode or BenchmarkMode(
                "mlx_local_no_cache", Path(package_dir or "."), "injected"
            ),
            False,
        ),
        (
            cache_on_mode or BenchmarkMode(
                "mlx_local_cache_on", Path(package_dir or "."), "injected"
            ),
            True,
        ),
    )

    def generate(prompt, cache_enabled, repeat_index, engine):
        if generate_once is not None:
            return generate_once(
                prompt=prompt,
                cache_enabled=cache_enabled,
                repeat_index=repeat_index,
            )
        return engine.generate(
            prompt.text, max_tokens=prompt.max_tokens, return_debug=True
        )

    samples: list[MLXSkipComputeSample] = []
    for mode, cache_enabled in modes:
        engine = _open_benchmark_mode(mode) if generate_once is None else None
        primary_error: BaseException | None = None
        try:
            plan = engine._handle._resolved_model_plan if engine is not None else None
            if plan is not None:
                config = plan.runtime_config
                configured_cache = any((
                    config.layer0_kv_cache_enabled, config.prefix_kv_cache,
                    config.block_kv_cache, config.d2f_exact_kv_cache,
                ))
                if plan.load.backend != "mlx_local" or configured_cache != cache_enabled:
                    raise ValueError(f"MLX cache mode disagrees with its executable plan: {mode.name!r}")

            for prompt in prompts:
                for warmup_index in range(int(warmup_runs)):
                    generate(prompt, cache_enabled, -(warmup_index + 1), engine)
                for repeat_index in range(int(repeats)):
                    result = generate(prompt, cache_enabled, repeat_index, engine)
                    metrics = _metrics_from_result(result)
                    samples.append(
                        MLXSkipComputeSample(
                            mode=mode.name,
                            prompt_id=prompt.id,
                            repeat_index=repeat_index,
                            cache_enabled=cache_enabled,
                            total_latency_ms=float(
                                metrics.get("total_latency_ms", 0.0)
                            ),
                            forward_count=int(metrics.get("mlx_forward_count", 0)),
                            layer0_projected_rows=int(
                                metrics.get("mlx_kv_layer0_projected_rows", 0)
                            ),
                            layer0_skipped_projection_rows=int(
                                metrics.get("mlx_kv_layer0_skipped_projection_rows", 0)
                            ),
                            baseline_status={
                                key: bool(value)
                                for key, value in dict(
                                    metrics.get("baseline_status", {})
                                ).items()
                            },
                            row_verified=bool(metrics.get("row_verified", False)),
                            timing_metrics=_mlx_timing_metrics(metrics),
                            text=str(result.text),
                            deployment_profile=plan.load.deployment_profile if plan else "",
                            precision=plan.load.precision if plan else "",
                            exact_reuses=int(metrics.get("kv_manager_applied_reuses", 0)),
                            peak_memory_bytes=_peak_memory_bytes(metrics),
                        )
                    )
        except BaseException as error:
            primary_error = error
            raise
        finally:
            if engine is not None:
                if primary_error is None:
                    engine.close()
                else:
                    _close_preserving_primary(
                        engine.close, primary_error, label="benchmark engine cleanup"
                    )
    return MLXSkipComputeBenchmarkReport(
        samples=samples,
        summaries=_summarize_mlx_skip_compute_samples(samples),
        speedup_claimed=False,
    )


def _summarize_mlx_skip_compute_samples(
    samples: list[MLXSkipComputeSample],
) -> list[MLXSkipComputeSummary]:
    keys = sorted({(sample.mode, sample.prompt_id, sample.cache_enabled) for sample in samples})
    summaries: list[MLXSkipComputeSummary] = []
    for mode, prompt_id, cache_enabled in keys:
        group = [
            sample
            for sample in samples
            if sample.mode == mode
            and sample.prompt_id == prompt_id
            and sample.cache_enabled is cache_enabled
        ]
        latencies = [sample.total_latency_ms for sample in group]
        summaries.append(
            MLXSkipComputeSummary(
                mode=mode,
                prompt_id=prompt_id,
                cache_enabled=cache_enabled,
                run_count=len(group),
                p50_latency_ms=_percentile(latencies, 0.50),
                p95_latency_ms=_percentile(latencies, 0.95),
                min_latency_ms=min(latencies) if latencies else 0.0,
                max_latency_ms=max(latencies) if latencies else 0.0,
                total_projected_rows=sum(sample.layer0_projected_rows for sample in group),
                total_skipped_projection_rows=sum(
                    sample.layer0_skipped_projection_rows for sample in group
                ),
            )
        )
    return summaries


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = int(math.ceil(float(percentile) * len(ordered))) - 1
    index = max(0, min(index, len(ordered) - 1))
    return ordered[index]


def _metrics_from_result(result: Any) -> dict[str, Any]:
    debug = getattr(result, "debug", None)
    if debug is None:
        return {}
    return dict(getattr(debug, "metrics", {}) or {})


def _mlx_timing_metrics(metrics: dict[str, Any]) -> dict[str, float | int | bool]:
    timing: dict[str, float | int | bool] = {}
    for key, value in metrics.items():
        if not str(key).startswith("mlx_timing_"):
            continue
        if isinstance(value, bool):
            timing[str(key)] = value
        elif isinstance(value, int):
            timing[str(key)] = int(value)
        elif isinstance(value, float):
            timing[str(key)] = float(value)
    return timing


def _tokens_per_second(metrics: dict[str, Any]) -> float:
    total_latency_ms = float(metrics.get("total_latency_ms", 0.0))
    if total_latency_ms <= 0:
        return 0.0
    token_count = len(list(metrics.get("visible_generated_token_ids", [])))
    return token_count / (total_latency_ms / 1000.0)


def _tokens_per_forward(metrics: dict[str, Any]) -> float:
    """Committed tokens per executor forward call.

    No-transfer scheduler holds are excluded because they do not call the
    executor. This is not tokens per diffusion scheduler iteration.
    """
    forward_count = int(metrics.get("block_iteration_count", 0))
    if forward_count <= 0:
        return 0.0
    token_count = int(
        metrics.get(
            "decoded_committed_count",
            len(list(metrics.get("visible_generated_token_ids", []))),
        )
    )
    return token_count / float(forward_count)


def _peak_memory_bytes(metrics: dict[str, Any]) -> int | None:
    peak_memory_bytes = int(metrics.get("peak_memory_bytes", 0))
    if peak_memory_bytes <= 0:
        return None
    return peak_memory_bytes


def _warn_if_cache_interval_did_not_refresh(
    *,
    mode: BenchmarkMode,
    metrics: dict[str, Any],
    interval: int,
) -> None:
    if interval <= 0:
        return
    if int(metrics.get("cache_refresh_count", 0)) > 0:
        return
    warnings.warn(
        (
            f"{mode.name} configured cache_refresh_interval={interval} "
            "but observed cache_refresh_count=0; do not interpret this row as a "
            "cache-refresh measurement."
        ),
        RuntimeWarning,
        stacklevel=2,
    )
