"""Pure contracts for threshold-vs-entropy commit calibration.

The module deliberately imports no MLX or runtime backend. A runner may collect
real observations, while the evidence validation and verdict remain portable and
unit-testable.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REFERENCE_CONFIG_NAME = "threshold_tau_0.9"
_FAMILIES = frozenset({"threshold", "entropy_budget"})
_CASE_FIELDS = frozenset(
    {
        "id",
        "regime",
        "prompt",
        "max_tokens",
        "required_any",
        "required_all",
        "min_words",
    }
)
_WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)
_MASK_MARKERS = ("<|mask|>", "[mask]")
_FENCED_CODE_RE = re.compile(
    r"```(?:python)?\s*\n(?P<code>.*?)```",
    flags=re.IGNORECASE | re.DOTALL,
)


def _nonempty_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _finite_number(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{field_name} must be finite")
    return resolved


def _nonnegative_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return int(value)


def _string_tuple(value: Any, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a JSON array of strings")
    items = tuple(
        _nonempty_string(item, field_name=f"{field_name} item") for item in value
    )
    if len(set(items)) != len(items):
        raise ValueError(f"{field_name} must not contain duplicates")
    return items


@dataclass(frozen=True, slots=True)
class CalibrationCase:
    case_id: str
    regime: str
    prompt: str
    max_tokens: int
    required_any: tuple[str, ...] = ()
    required_all: tuple[str, ...] = ()
    min_words: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "case_id", _nonempty_string(self.case_id, field_name="case_id")
        )
        object.__setattr__(
            self, "regime", _nonempty_string(self.regime, field_name="regime")
        )
        object.__setattr__(
            self, "prompt", _nonempty_string(self.prompt, field_name="prompt")
        )
        max_tokens = _nonnegative_int(self.max_tokens, field_name="max_tokens")
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        object.__setattr__(self, "max_tokens", max_tokens)
        min_words = _nonnegative_int(self.min_words, field_name="min_words")
        if min_words < 1:
            raise ValueError("min_words must be >= 1")
        object.__setattr__(self, "min_words", min_words)
        for field_name in ("required_any", "required_all"):
            value = getattr(self, field_name)
            if not isinstance(value, tuple) or any(
                not isinstance(item, str) or not item.strip() for item in value
            ):
                raise ValueError(f"{field_name} must be a tuple of non-empty strings")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.case_id,
            "regime": self.regime,
            "prompt": self.prompt,
            "max_tokens": self.max_tokens,
            "required_any": list(self.required_any),
            "required_all": list(self.required_all),
            "min_words": self.min_words,
        }


def _case_from_mapping(raw: Mapping[str, Any], *, index: int) -> CalibrationCase:
    unknown = set(raw) - _CASE_FIELDS
    if unknown:
        raise ValueError(f"case[{index}] has unknown fields: {sorted(unknown)}")
    missing = {"id", "regime", "prompt", "max_tokens"} - set(raw)
    if missing:
        raise ValueError(f"case[{index}] is missing fields: {sorted(missing)}")
    return CalibrationCase(
        case_id=raw["id"],
        regime=raw["regime"],
        prompt=raw["prompt"],
        max_tokens=raw["max_tokens"],
        required_any=_string_tuple(raw.get("required_any"), field_name="required_any"),
        required_all=_string_tuple(raw.get("required_all"), field_name="required_all"),
        min_words=raw.get("min_words", 1),
    )


def load_cases(path: str | Path) -> list[CalibrationCase]:
    """Load and strictly validate a versioned calibration-case manifest."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load calibration cases from {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("calibration case manifest must be a JSON object")
    if set(payload) - {"version", "description", "cases"}:
        raise ValueError(
            "calibration case manifest has unknown fields: "
            f"{sorted(set(payload) - {'version', 'description', 'cases'})}"
        )
    if payload.get("version") != "1.0.0":
        raise ValueError("calibration case manifest version must be '1.0.0'")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("calibration case manifest cases must be a non-empty array")

    cases: list[CalibrationCase] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_cases):
        if not isinstance(raw, dict):
            raise ValueError(f"case[{index}] must be a JSON object")
        case = _case_from_mapping(raw, index=index)
        if case.case_id in seen:
            raise ValueError(f"duplicate case id: {case.case_id!r}")
        seen.add(case.case_id)
        cases.append(case)
    return cases


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    name: str
    family: str
    confidence_threshold: float | None = None
    gamma: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty_string(self.name, field_name="name"))
        if self.family not in _FAMILIES:
            raise ValueError(f"family must be one of {sorted(_FAMILIES)}")
        if self.family == "threshold":
            if self.confidence_threshold is None:
                raise ValueError("threshold config requires confidence_threshold")
            if self.gamma is not None:
                raise ValueError("threshold config must not set gamma")
            threshold = _finite_number(
                self.confidence_threshold, field_name="confidence_threshold"
            )
            if not 0.0 <= threshold <= 1.0:
                raise ValueError("confidence_threshold must be in [0, 1]")
            object.__setattr__(self, "confidence_threshold", threshold)
        else:
            if self.gamma is None:
                raise ValueError("entropy_budget config requires gamma")
            if self.confidence_threshold is not None:
                raise ValueError(
                    "entropy_budget config must not set confidence_threshold"
                )
            gamma = _finite_number(self.gamma, field_name="gamma")
            if gamma < 0.0:
                raise ValueError("gamma must be >= 0")
            object.__setattr__(self, "gamma", gamma)

    def model_config_overlay(self) -> dict[str, Any]:
        """Return the fixed calibration envelope plus this policy's one signal."""
        return {
            "mlx_diffusion_policy_profile": "baseline",
            "mlx_diffusion_decode_strategy": "d2f_block_pipeline",
            "attention_mask_kind": "d2f_hybrid",
            "mlx_diffusion_d2f_exact_kv_cache": True,
            "mlx_diffusion_window_exec": "auto",
            "mlx_diffusion_block_size": 8,
            "mlx_diffusion_d2f_add_threshold": 0.1,
            "mlx_diffusion_d2f_activate_threshold": 0.95,
            "mlx_diffusion_d2f_commit_policy": self.family,
            "mlx_diffusion_confidence_threshold": (
                self.confidence_threshold
                if self.confidence_threshold is not None
                else 0.9
            ),
            "mlx_diffusion_gamma": self.gamma if self.gamma is not None else 0.0,
            "mlx_compute_dtype": "fp32",
            "mlx_linear_quant_bits": None,
            "mlx_layer0_kv_cache_enabled": False,
            "mlx_diffusion_prefix_kv_cache": False,
            "mlx_diffusion_block_kv_cache": False,
            "mlx_diffusion_d2f_skip_committed_rows": False,
            "mlx_diffusion_error_fixing": False,
            "mlx_diffusion_trace_canvas": False,
            "mlx_diffusion_canvas_cache": False,
            "mlx_diffusion_cross_request_cache": False,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_CONFIGS: tuple[CalibrationConfig, ...] = (
    CalibrationConfig(
        name=REFERENCE_CONFIG_NAME, family="threshold", confidence_threshold=0.9
    ),
    CalibrationConfig(
        name="threshold_tau_0.6", family="threshold", confidence_threshold=0.6
    ),
    CalibrationConfig(
        name="threshold_tau_0.7", family="threshold", confidence_threshold=0.7
    ),
    CalibrationConfig(
        name="threshold_tau_0.8", family="threshold", confidence_threshold=0.8
    ),
    CalibrationConfig(name="eb_gamma_0.4", family="entropy_budget", gamma=0.4),
    CalibrationConfig(name="eb_gamma_0.6", family="entropy_budget", gamma=0.6),
    CalibrationConfig(name="eb_gamma_0.8", family="entropy_budget", gamma=0.8),
    CalibrationConfig(name="eb_gamma_1.0", family="entropy_budget", gamma=1.0),
)


@dataclass(frozen=True, slots=True)
class OutputQualityEvidence:
    quality_pass: bool
    objective_pass: bool | None
    min_words_pass: bool
    mask_leak: bool
    repetition_score: float
    distinct_ratio: float
    degeneration_score: float
    word_count: int
    syntax_pass: bool | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalized_text(text: str) -> str:
    return " ".join(text.lower().split())


def _repetition_score(words: Sequence[str], *, ngram: int = 4) -> float:
    if len(words) < 40 or len(words) < ngram:
        return 0.0
    grams = [
        tuple(words[index : index + ngram])
        for index in range(len(words) - ngram + 1)
    ]
    counts: dict[tuple[str, ...], int] = {}
    for gram in grams:
        counts[gram] = counts.get(gram, 0) + 1
    return max(counts.values()) / len(grams)


def evaluate_output(case: CalibrationCase, output_text: str) -> OutputQualityEvidence:
    """Apply bounded objective, structure, and degeneration checks to one output."""
    if not isinstance(output_text, str):
        raise TypeError("output_text must be a string")
    normalized = _normalized_text(output_text)
    words = [word.lower() for word in _WORD_RE.findall(output_text)]
    word_count = len(words)
    min_words_pass = word_count >= case.min_words
    mask_leak = any(marker in normalized for marker in _MASK_MARKERS)
    repetition_score = _repetition_score(words)
    distinct_ratio = len(set(words)) / word_count if word_count else 0.0
    repeated = repetition_score >= 0.12
    low_distinct = word_count >= 20 and distinct_ratio < 0.35
    empty = not normalized
    degeneration_score = float(
        int(mask_leak) + int(repeated) + int(low_distinct) + int(empty)
    )

    required_any_pass = (
        any(item.lower() in normalized for item in case.required_any)
        if case.required_any
        else True
    )
    required_all_pass = all(
        item.lower() in normalized for item in case.required_all
    )
    has_objective = bool(case.required_any or case.required_all)
    objective_pass = (
        required_any_pass and required_all_pass if has_objective else None
    )
    syntax_pass: bool | None = None
    if case.regime == "code":
        match = _FENCED_CODE_RE.search(output_text)
        if match is None:
            syntax_pass = False
        else:
            try:
                compile(match.group("code"), f"<{case.case_id}>", "exec")
            except SyntaxError:
                syntax_pass = False
            else:
                syntax_pass = True
    quality_pass = (
        not empty
        and min_words_pass
        and objective_pass is not False
        and syntax_pass is not False
        and degeneration_score == 0.0
    )
    return OutputQualityEvidence(
        quality_pass=quality_pass,
        objective_pass=objective_pass,
        min_words_pass=min_words_pass,
        mask_leak=mask_leak,
        repetition_score=repetition_score,
        distinct_ratio=distinct_ratio,
        degeneration_score=degeneration_score,
        word_count=word_count,
        syntax_pass=syntax_pass,
    )


@dataclass(frozen=True, slots=True)
class CalibrationObservation:
    config_name: str
    case_id: str
    output_text: str
    quality_pass: bool
    objective_pass: bool | None
    degeneration_score: float
    repetition_score: float
    distinct_ratio: float
    forwarded_rows: int
    committed_tokens: int
    forward_count: int
    output_tokens: int
    latency_ms: float
    syntax_pass: bool | None = None

    def __post_init__(self) -> None:
        _nonempty_string(self.config_name, field_name="config_name")
        _nonempty_string(self.case_id, field_name="case_id")
        if not isinstance(self.output_text, str):
            raise ValueError("output_text must be a string")
        if type(self.quality_pass) is not bool:
            raise ValueError("quality_pass must be a boolean")
        if self.objective_pass is not None and type(self.objective_pass) is not bool:
            raise ValueError("objective_pass must be a boolean or null")
        if self.syntax_pass is not None and type(self.syntax_pass) is not bool:
            raise ValueError("syntax_pass must be a boolean or null")
        for field_name in (
            "degeneration_score",
            "repetition_score",
            "distinct_ratio",
            "latency_ms",
        ):
            value = _finite_number(getattr(self, field_name), field_name=field_name)
            if value < 0.0:
                raise ValueError(f"{field_name} must be >= 0")
            if field_name == "distinct_ratio" and value > 1.0:
                raise ValueError("distinct_ratio must be <= 1")
            object.__setattr__(self, field_name, value)
        for field_name in (
            "forwarded_rows",
            "committed_tokens",
            "forward_count",
            "output_tokens",
        ):
            object.__setattr__(
                self,
                field_name,
                _nonnegative_int(getattr(self, field_name), field_name=field_name),
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_unique(items: Iterable[Any], *, attr: str, label: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in items:
        key = getattr(item, attr)
        if key in result:
            raise ValueError(f"duplicate {label}: {key!r}")
        result[key] = item
    if not result:
        raise ValueError(f"{label} collection must not be empty")
    return result


def _config_regime_summary(
    observations: Sequence[CalibrationObservation],
    reference_by_case: Mapping[str, CalibrationObservation],
) -> dict[str, Any]:
    forwarded_rows = sum(item.forwarded_rows for item in observations)
    committed_tokens = sum(item.committed_tokens for item in observations)
    forward_count = sum(item.forward_count for item in observations)
    output_tokens = sum(item.output_tokens for item in observations)
    exact_matches = sum(
        _normalized_text(item.output_text)
        == _normalized_text(reference_by_case[item.case_id].output_text)
        for item in observations
    )
    objective = [item for item in observations if item.objective_pass is not None]
    syntax = [item for item in observations if item.syntax_pass is not None]
    return {
        "case_count": len(observations),
        "quality_pass": all(item.quality_pass for item in observations),
        "quality_pass_count": sum(item.quality_pass for item in observations),
        "objective_pass_count": sum(item.objective_pass is True for item in objective),
        "objective_case_count": len(objective),
        "syntax_pass_count": sum(item.syntax_pass is True for item in syntax),
        "syntax_case_count": len(syntax),
        "degeneration_count": sum(
            item.degeneration_score > 0.0 for item in observations
        ),
        "mean_degeneration_score": (
            sum(item.degeneration_score for item in observations) / len(observations)
        ),
        "exact_text_match_rate": exact_matches / len(observations),
        "forwarded_rows": forwarded_rows,
        "committed_tokens": committed_tokens,
        "rows_per_committed_token": forwarded_rows / max(1, committed_tokens),
        "forward_count": forward_count,
        "output_tokens": output_tokens,
        "forwards_per_output_token": forward_count / max(1, output_tokens),
        "latency_ms": sum(item.latency_ms for item in observations),
    }


def summarize_calibration(
    cases: Sequence[CalibrationCase],
    configs: Sequence[CalibrationConfig],
    observations: Sequence[CalibrationObservation],
    *,
    reference_config_name: str = REFERENCE_CONFIG_NAME,
    material_reduction: float = 0.03,
    min_cases_per_regime: int = 3,
) -> dict[str, Any]:
    """Validate a calibration matrix and return its deterministic verdict."""
    case_by_id = _validate_unique(cases, attr="case_id", label="case id")
    config_by_name = _validate_unique(configs, attr="name", label="config name")
    if reference_config_name not in config_by_name:
        raise ValueError(f"unknown reference config: {reference_config_name!r}")
    if config_by_name[reference_config_name].family != "threshold":
        raise ValueError("reference config must belong to the threshold family")
    material_reduction = _finite_number(
        material_reduction, field_name="material_reduction"
    )
    if not 0.0 <= material_reduction < 1.0:
        raise ValueError("material_reduction must be in [0, 1)")
    min_cases_per_regime = _nonnegative_int(
        min_cases_per_regime, field_name="min_cases_per_regime"
    )
    if min_cases_per_regime < 1:
        raise ValueError("min_cases_per_regime must be >= 1")

    observation_by_key: dict[tuple[str, str], CalibrationObservation] = {}
    for observation in observations:
        if observation.config_name not in config_by_name:
            raise ValueError(f"observation has unknown config: {observation.config_name!r}")
        if observation.case_id not in case_by_id:
            raise ValueError(f"observation has unknown case: {observation.case_id!r}")
        key = (observation.config_name, observation.case_id)
        if key in observation_by_key:
            raise ValueError(
                "duplicate observation for config/case: "
                f"{observation.config_name!r}/{observation.case_id!r}"
            )
        observation_by_key[key] = observation

    missing = [
        {"case_id": case_id, "config_name": config_name}
        for config_name in config_by_name
        for case_id in case_by_id
        if (config_name, case_id) not in observation_by_key
    ]
    missing.sort(key=lambda item: (item["config_name"], item["case_id"]))
    missing_reference_cases = sorted(
        case_id
        for case_id in case_by_id
        if (reference_config_name, case_id) not in observation_by_key
    )
    if missing_reference_cases:
        return {
            "verdict": "insufficient_evidence",
            "reference_config": reference_config_name,
            "missing_observations": missing,
            "regimes": {},
        }

    reference_by_case = {
        case_id: observation_by_key[(reference_config_name, case_id)]
        for case_id in case_by_id
    }
    failed_reference_cases = sorted(
        case_id
        for case_id, observation in reference_by_case.items()
        if not observation.quality_pass
    )
    if failed_reference_cases:
        return {
            "verdict": "reference_failed",
            "reference_config": reference_config_name,
            "failed_reference_cases": failed_reference_cases,
            "missing_observations": [],
            "candidate_observations_skipped": len(missing),
            "regimes": {},
        }
    if missing:
        return {
            "verdict": "insufficient_evidence",
            "reference_config": reference_config_name,
            "missing_observations": missing,
            "regimes": {},
        }

    cases_by_regime: dict[str, list[CalibrationCase]] = {}
    for case in cases:
        cases_by_regime.setdefault(case.regime, []).append(case)
    undersized = {
        regime: len(regime_cases)
        for regime, regime_cases in cases_by_regime.items()
        if len(regime_cases) < min_cases_per_regime
    }
    if undersized:
        return {
            "verdict": "insufficient_evidence",
            "reference_config": reference_config_name,
            "missing_observations": [],
            "undersized_regimes": undersized,
            "regimes": {},
        }

    regime_reports: dict[str, Any] = {}
    any_nonreference_eligible = False
    family_winners: set[str] = set()
    for regime in sorted(cases_by_regime):
        regime_case_ids = [case.case_id for case in cases_by_regime[regime]]
        config_reports: dict[str, Any] = {}
        eligible_by_family: dict[str, list[tuple[str, dict[str, Any]]]] = {
            family: [] for family in _FAMILIES
        }
        for config_name, config in config_by_name.items():
            selected = [
                observation_by_key[(config_name, case_id)]
                for case_id in regime_case_ids
            ]
            summary = _config_regime_summary(selected, reference_by_case)
            eligible = bool(summary["quality_pass"])
            summary["eligible"] = eligible
            summary["family"] = config.family
            config_reports[config_name] = summary
            if eligible:
                eligible_by_family[config.family].append((config_name, summary))
                if config_name != reference_config_name:
                    any_nonreference_eligible = True

        best_by_family: dict[str, dict[str, Any]] = {}
        for family, candidates in eligible_by_family.items():
            if not candidates:
                continue
            name, summary = min(
                candidates,
                key=lambda item: (
                    item[1]["rows_per_committed_token"],
                    item[1]["forwards_per_output_token"],
                    item[0],
                ),
            )
            best_by_family[family] = {
                "config_name": name,
                "rows_per_committed_token": summary["rows_per_committed_token"],
                "forwards_per_output_token": summary["forwards_per_output_token"],
            }

        winner_family: str | None
        if len(best_by_family) == 1:
            winner_family = next(iter(best_by_family))
        elif len(best_by_family) == 2:
            threshold_rows = best_by_family["threshold"]["rows_per_committed_token"]
            entropy_rows = best_by_family["entropy_budget"][
                "rows_per_committed_token"
            ]
            slower = max(threshold_rows, entropy_rows)
            relative_reduction = (
                abs(threshold_rows - entropy_rows) / slower if slower else 0.0
            )
            if relative_reduction < material_reduction:
                winner_family = None
            else:
                winner_family = (
                    "threshold" if threshold_rows < entropy_rows else "entropy_budget"
                )
        else:
            winner_family = None
        if winner_family is not None:
            family_winners.add(winner_family)
        regime_reports[regime] = {
            "case_ids": regime_case_ids,
            "configs": config_reports,
            "best_by_family": best_by_family,
            "winner_family": winner_family,
        }

    if not any_nonreference_eligible:
        verdict = "no_eligible_candidate"
    elif family_winners == _FAMILIES:
        verdict = "crossover_observed"
    elif family_winners == {"threshold"}:
        verdict = "threshold_only"
    elif family_winners == {"entropy_budget"}:
        verdict = "entropy_budget_only"
    else:
        verdict = "no_material_difference"
    return {
        "verdict": verdict,
        "reference_config": reference_config_name,
        "material_reduction": material_reduction,
        "missing_observations": [],
        "regimes": regime_reports,
    }


def write_json_report(
    path: str | Path, payload: Mapping[str, Any], *, overwrite: bool = False
) -> None:
    """Write JSON in one filesystem transaction, refusing overwrite by default."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary_path, destination)
            temporary_path = None
        else:
            os.link(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
