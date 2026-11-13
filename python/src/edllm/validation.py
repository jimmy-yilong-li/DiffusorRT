from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class CacheValidationRow:
    prompt_id: str
    text: str
    output_text: str
    max_tokens: int
    cache_refresh_interval: int
    decode_loop_kind: str
    baseline_status: dict[str, bool]
    row_verified: bool
    cache_refresh_count: int
    visible_generated_token_ids: list[int]
    ttft_ms: float
    total_latency_ms: float


@dataclass(slots=True)
class CacheValidationReport:
    rows: list[CacheValidationRow]
    cache_default_recommendation: bool
    baseline_verified: bool = False


def run_cache_validation_matrix(
    runtime: Any,
    *,
    prompts: list[dict[str, str]],
    cache_refresh_intervals: tuple[int, ...] = (0, 2),
    max_tokens: int = 8,
) -> CacheValidationReport:
    rows: list[CacheValidationRow] = []
    model_config = getattr(runtime, "model_config", None)
    if model_config is None:
        runtime.model_config = {}
        model_config = runtime.model_config
    original_interval = model_config.get("diffusion_cache_refresh_interval")

    try:
        for prompt in prompts:
            for interval in cache_refresh_intervals:
                model_config["diffusion_cache_refresh_interval"] = int(interval)
                result = runtime.generate(prompt["text"], max_tokens=max_tokens, return_debug=True)
                metrics = result.debug.metrics
                rows.append(
                    CacheValidationRow(
                        prompt_id=prompt["id"],
                        text=prompt["text"],
                        output_text=result.text,
                        max_tokens=max_tokens,
                        cache_refresh_interval=int(interval),
                        decode_loop_kind=str(metrics.get("decode_loop_kind", "")),
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
                    )
                )
    finally:
        if original_interval is None:
            model_config.pop("diffusion_cache_refresh_interval", None)
        else:
            model_config["diffusion_cache_refresh_interval"] = original_interval

    report = CacheValidationReport(
        rows=rows,
        cache_default_recommendation=False,
        baseline_verified=False,
    )
    recommendation = cache_validation_recommends_default(report)
    report.cache_default_recommendation = recommendation
    report.baseline_verified = recommendation
    return report


def cache_validation_recommends_default(report: CacheValidationReport) -> bool:
    cache_rows = [row for row in report.rows if row.cache_refresh_interval > 0]
    no_cache_prompt_ids = {
        row.prompt_id for row in report.rows if row.cache_refresh_interval == 0
    }
    cache_prompt_ids = {row.prompt_id for row in cache_rows}
    if not cache_rows:
        return False
    if not no_cache_prompt_ids:
        return False
    if len(no_cache_prompt_ids) < 2:
        return False
    if cache_prompt_ids != no_cache_prompt_ids:
        return False
    no_cache_rows_by_prompt: dict[str, CacheValidationRow] = {}
    for row in report.rows:
        if row.cache_refresh_interval != 0:
            continue
        if row.prompt_id in no_cache_rows_by_prompt:
            return False
        no_cache_rows_by_prompt[row.prompt_id] = row
    for cache_row in cache_rows:
        no_cache_row = no_cache_rows_by_prompt.get(cache_row.prompt_id)
        if no_cache_row is None:
            return False
        if cache_row.output_text != no_cache_row.output_text:
            return False
        if cache_row.visible_generated_token_ids != no_cache_row.visible_generated_token_ids:
            return False
    if not all(row.baseline_status.get("refresh_real", False) for row in cache_rows):
        return False
    if not all(row.cache_refresh_count > 0 for row in cache_rows):
        return False
    # Every row must have tokenizer_real, forward_real, entropy_real all True.
    # These three are not cache-dependent and reflect that the runtime actually
    # loaded a real tokenizer and ran a real forward with real entropy commit.
    required_per_row_bits = ("tokenizer_real", "forward_real", "entropy_real")
    for row in report.rows:
        for bit in required_per_row_bits:
            if not row.baseline_status.get(bit, False):
                return False
    return all(_passes_text_quality_gate(row.output_text) for row in report.rows)


def _passes_text_quality_gate(text: str) -> bool:
    """Stronger than _has_alpha. Rejects numeric-dominated noise like
    '1000 times' that the per-call >=2 alpha gate let through."""
    stripped = text.lstrip()
    if not stripped:
        return False
    if not stripped[0].isalpha():
        return False
    alpha_count = sum(1 for c in text if c.isalpha())
    digit_count = sum(1 for c in text if c.isdigit())
    if alpha_count < 3:
        return False
    if alpha_count <= digit_count:
        return False
    return True
