"""D2-lite quality harness (WS3 D2-lite).

A standalone open-core module that calls the engine's PUBLIC API
(``MLXDreamEngine.generate_entropy`` / ``generate_infill``) and produces a
structured :class:`QualityReport`. Exact and approximate runs carry derived
per-task provenance. An approximate path passes its run as *candidate* and an
exact-path report as *reference*, and the report carries the token-level
``exact_match_rate`` between them without trusting a caller-supplied exact label.

Generation uses the same public API as tests and the CLI. Evidence preflight also
uses the MLX engine's resident-weight seam to bind Task 1 loader provenance; it
does not infer weight identity from a display path.
"""

from __future__ import annotations

import hashlib
import json
import math
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from edllm import presets


_TASK_KINDS = ("chat", "infill", "edit")
_PROFILE_SELECTOR = "mlx_diffusion_policy_profile"
_MODEL_IDENTITY_REQUIRED_CONFIG_KEYS = frozenset(
    {"architecture", "hidden_size", "intermediate_size", "vocab_size"}
)
_GRAPH_IR_REQUIRED_CONFIG_KEYS = frozenset(
    {
        "layer_count",
        "attention_heads",
        "num_kv_heads",
        "head_dim",
        "rms_norm_eps",
        "rope_theta",
    }
)
_QUALITY_REPORT_STATE_VERSION = 1
_QUALITY_REPORT_STATE_FIELDS = (
    "tasks",
    "aggregate",
    "_canonical_baseline_specs",
    "_self_reference_kind",
    "_self_profile_name",
    "_run_identity",
    "_reference_task_evidence",
    "_evidence_seal",
)


class _Unset:
    __slots__ = ()

    def __repr__(self) -> str:
        return "_UNSET"


_UNSET = _Unset()
_LEGACY_SHARED_DEFAULTS: dict[str, Any] = {
    "decode_strategy": "block_threshold",
    "confidence_threshold": 0.9,
    "block_size": 8,
    "steps": 64,
    "uses_shifted_logits": True,
    "uses_qk_norm": False,
}
_POLICY_OVERLAY_CONFIG_KEYS = frozenset(
    key
    for overlay in presets.POLICY_PROFILES.values()
    for key in overlay
)

# These describe package state rather than call-time engine arguments. They stay
# in the evidence mapping but are removed before invoking the public engine API.
_EVIDENCE_ONLY_ENGINE_KEYS = frozenset(
    {"linear_quant_bits", "linear_quant_group_size"}
)
_CALL_ONLY_ENGINE_KEYS = frozenset({"window_plan_sink"})

_PROFILE_ENGINE_KEYS = frozenset(
    {
        "attention_mask_kind",
        "block_size",
        "confidence_threshold",
        "d2f_activate_threshold",
        "d2f_add_threshold",
        "d2f_commit_policy",
        "decode_strategy",
        "enable_d2f_exact_kv_cache",
        "error_fixing",
        "gamma",
        "window_exec",
    }
)

_CHAT_ENGINE_KEYS = frozenset(
    {
        "alg_temp",
        "attention_mask_kind",
        "block_kv_refresh_interval",
        "block_size",
        "canvas_cache_budget_bytes",
        "compute_dtype",
        "confidence_threshold",
        "cross_session_cache_root",
        "d2f_activate_threshold",
        "d2f_add_threshold",
        "d2f_commit_policy",
        "d2f_skip_committed_rows",
        "d2f_skip_refresh_interval",
        "decode_strategy",
        "delayed_kv_max_stale_steps",
        "delayed_kv_refresh_interval",
        "delayed_kv_scope",
        "delayed_kv_stale_epoch_budget",
        "delayed_kv_upper_layer_range",
        "enable_block_kv_cache",
        "enable_canvas_cache",
        "enable_cross_request_cache",
        "enable_cross_session_cache",
        "enable_d2f_exact_kv_cache",
        "enable_delayed_kv",
        "enable_layer0_kv_cache",
        "enable_prefix_kv_cache",
        "eps",
        "error_fixing",
        "gamma",
        "prefix_kv_refresh_interval",
        "save_snapshot_path",
        "steps",
        "temperature",
        "top_p",
        "trace_canvas",
        "uses_qk_norm",
        "uses_shifted_logits",
        "window_exec",
        "window_plan_sink",
        "exact_kv_slice_mode",
        "exact_kv_slice_page_budget_bytes",
    }
)

_INFILL_ENGINE_KEYS = frozenset(
    {
        "alg_temp",
        "attention_mask_kind",
        "block_size",
        "compute_dtype",
        "confidence_threshold",
        "decode_strategy",
        "enable_layer0_kv_cache",
        "eps",
        "gamma",
        "steps",
        "temperature",
        "top_p",
        "uses_qk_norm",
        "uses_shifted_logits",
        "window_exec",
    }
)

_RETIRED_KV_ENGINE_DEFAULTS: dict[str, Any] = {
    "exact_kv_slice_mode": "off",
    "exact_kv_slice_page_budget_bytes": None,
    "d2f_skip_committed_rows": False,
    "d2f_skip_refresh_interval": 4,
    "enable_delayed_kv": False,
    "delayed_kv_scope": "layer0",
    "delayed_kv_max_stale_steps": None,
    "delayed_kv_refresh_interval": None,
    "delayed_kv_upper_layer_range": None,
    "delayed_kv_stale_epoch_budget": None,
    "enable_canvas_cache": False,
    "canvas_cache_budget_bytes": None,
    "enable_cross_request_cache": False,
    "enable_cross_session_cache": False,
    "cross_session_cache_root": None,
}

_CHAT_IDENTITY_FIELDS = frozenset(
    {"prompt_token_ids", "max_tokens", "mask_token_id", "eos_token_id"}
)
_INFILL_IDENTITY_FIELDS = frozenset(
    {
        "prefix_token_ids",
        "suffix_token_ids",
        "max_fill_tokens",
        "mask_token_id",
        "eos_token_id",
    }
)

_BOOL_ENGINE_KEYS = frozenset(
    {
        "d2f_skip_committed_rows",
        "enable_block_kv_cache",
        "enable_canvas_cache",
        "enable_cross_request_cache",
        "enable_cross_session_cache",
        "enable_d2f_exact_kv_cache",
        "enable_delayed_kv",
        "enable_layer0_kv_cache",
        "enable_prefix_kv_cache",
        "trace_canvas",
        "uses_shifted_logits",
    }
)
_INT_ENGINE_KEYS = frozenset(
    {
        "block_kv_refresh_interval",
        "block_size",
        "canvas_cache_budget_bytes",
        "d2f_skip_refresh_interval",
        "delayed_kv_max_stale_steps",
        "delayed_kv_refresh_interval",
        "delayed_kv_stale_epoch_budget",
        "prefix_kv_refresh_interval",
        "steps",
    }
)
_FLOAT_ENGINE_KEYS = frozenset(
    {
        "alg_temp",
        "confidence_threshold",
        "d2f_activate_threshold",
        "d2f_add_threshold",
        "eps",
        "gamma",
        "temperature",
        "top_p",
    }
)


# ---------------------------------------------------------------------------
# Repetition scan (mirrors RepetitionDetector logic, applied to a flat list)
# ---------------------------------------------------------------------------


def _count_repetitions(tokens: list[int], *, max_ngram: int = 3) -> int:
    """Return the number of flagged (repeated) positions in *tokens*.

    Matches the ``RepetitionDetector`` logic in ``diffusion_runtime``:

    * Adjacent unigram runs: flag every repeat after the first.
    * Local repeated n-grams (length 2..max_ngram): if an n-gram is immediately
      followed by the same n-gram, flag the second occurrence's positions.

    The flagged *count* (not the flag list itself) is what the report needs.
    """
    n = len(tokens)
    flagged: set[int] = set()
    for i in range(1, n):
        if tokens[i] == tokens[i - 1]:
            flagged.add(i)
    for gram in range(2, max_ngram + 1):
        for i in range(0, n - 2 * gram + 1):
            if tokens[i : i + gram] == tokens[i + gram : i + 2 * gram]:
                for j in range(i + gram, i + 2 * gram):
                    flagged.add(j)
    return len(flagged)


# ---------------------------------------------------------------------------
# Per-task degeneration analysis
# ---------------------------------------------------------------------------


def _degeneration(
    tokens: list[int],
    text: str,
    *,
    mask_token_id: int,
    eos_token_id: int,
) -> dict[str, Any]:
    """Compute the degeneration component dict for a token sequence.

    Components:

    * ``repetition_rate`` — fraction of flagged-repeated positions (0.0 = clean).
    * ``mask_leak`` — True if ``mask_token_id`` appears in the visible output.
    * ``empty`` — True if there are zero visible tokens.
    * ``printable_ratio`` — fraction of characters in *text* that are printable
      ASCII (string.printable).  A low ratio signals garbled / garbage output.
    * ``premature_eos`` — True if EOS appears before any real (non-mask, non-EOS)
      content token.

    The aggregate ``degeneration_score`` accumulates:
      +1.0 if mask_leak, +1.0 if empty, +repetition_rate (0..1),
      +max(0, 1 - printable_ratio) * 0.5 (penalty for low printable ratio),
      +0.5 if premature_eos.
    Higher = worse.
    """
    n = len(tokens)
    mask_leak = mask_token_id in tokens
    empty = n == 0

    repetition_rate = 0.0
    if n > 0:
        flagged_count = _count_repetitions(tokens)
        repetition_rate = flagged_count / n

    printable_set = set(string.printable)
    if text:
        printable_ratio = sum(1 for ch in text if ch in printable_set) / len(text)
    else:
        printable_ratio = 0.0

    # Premature EOS: EOS appears before any real content token.
    premature_eos = False
    seen_real = False
    for tok in tokens:
        if tok == eos_token_id:
            if not seen_real:
                premature_eos = True
            break
        if tok != mask_token_id:
            seen_real = True

    score = 0.0
    if mask_leak:
        score += 1.0
    if empty:
        score += 1.0
    score += repetition_rate
    score += max(0.0, 1.0 - printable_ratio) * 0.5
    if premature_eos:
        score += 0.5

    return {
        "repetition_rate": repetition_rate,
        "mask_leak": mask_leak,
        "empty": empty,
        "printable_ratio": printable_ratio,
        "premature_eos": premature_eos,
        "degeneration_score": score,
    }


# ---------------------------------------------------------------------------
# Task-success rules (hard structural checks)
# ---------------------------------------------------------------------------

_CHAT_MIN_VISIBLE = 1
_CHAT_MAX_VISIBLE_MULTIPLIER = 4   # at most 4× the max_tokens cap is absurd
_CHAT_MAX_REPETITION_RATE = 0.5    # above 50% flagged = degenerate
_CHAT_MIN_PRINTABLE_RATIO = 0.5    # below 50% printable = garbage


def _chat_task_success(
    tokens: list[int],
    text: str,
    degen: dict[str, Any],
    *,
    max_tokens: int,
    mask_token_id: int,
    eos_token_id: int,
) -> tuple[bool, dict[str, Any]]:
    """Hard chat success checks; returns (pass, detail_dict)."""
    n = len(tokens)
    checks = {
        "visible_count_positive": n >= _CHAT_MIN_VISIBLE,
        "not_absurdly_long": n <= max_tokens * _CHAT_MAX_VISIBLE_MULTIPLIER,
        "repetition_ok": degen["repetition_rate"] < _CHAT_MAX_REPETITION_RATE,
        "no_mask_leak": not degen["mask_leak"],
        "no_eos_leak": eos_token_id not in tokens,
        "printable_ok": degen["printable_ratio"] >= _CHAT_MIN_PRINTABLE_RATIO,
    }
    return all(checks.values()), checks


def _infill_task_success(
    fill_token_ids: list[int],
    full_token_ids: list[int],
    prefix_token_ids: list[int],
    suffix_token_ids: list[int],
    *,
    mask_token_id: int,
    eos_token_id: int,
) -> tuple[bool, dict[str, Any]]:
    """Hard infill success checks; returns (pass, detail_dict).

    Rules:
    * prefix byte-identical: full_token_ids[:len(prefix)] == prefix_token_ids
    * suffix byte-identical: full_token_ids[-len(suffix):] == suffix_token_ids
      (vacuously True when suffix is empty)
    * fill contains no mask token
    * fill contains no eos token
    * fill is non-empty
    """
    np = len(prefix_token_ids)
    ns = len(suffix_token_ids)

    prefix_ok = list(full_token_ids[:np]) == list(prefix_token_ids)
    suffix_ok = (
        list(full_token_ids[len(full_token_ids) - ns:]) == list(suffix_token_ids)
        if ns > 0
        else True
    )
    no_mask = mask_token_id not in fill_token_ids
    no_eos = eos_token_id not in fill_token_ids
    fill_nonempty = len(fill_token_ids) > 0

    checks = {
        "prefix_byte_identical": prefix_ok,
        "suffix_byte_identical": suffix_ok,
        "fill_no_mask": no_mask,
        "fill_no_eos": no_eos,
        "fill_nonempty": fill_nonempty,
    }
    return all(checks.values()), checks


def _edit_task_success(
    fill_token_ids: list[int],
    full_token_ids: list[int],
    prefix_token_ids: list[int],
    suffix_token_ids: list[int],
    *,
    mask_token_id: int,
    eos_token_id: int,
) -> tuple[bool, dict[str, Any]]:
    """Hard edit success checks; identical rules to infill (edit is implemented
    as infill(prefix=text[:start], suffix=text[end:])) plus an edit-specific
    label for the detail dict."""
    np = len(prefix_token_ids)
    ns = len(suffix_token_ids)

    outside_prefix_ok = list(full_token_ids[:np]) == list(prefix_token_ids)
    outside_suffix_ok = (
        list(full_token_ids[len(full_token_ids) - ns:]) == list(suffix_token_ids)
        if ns > 0
        else True
    )
    no_mask = mask_token_id not in fill_token_ids
    no_eos = eos_token_id not in fill_token_ids
    span_nonempty = len(fill_token_ids) > 0

    checks = {
        "outside_prefix_byte_identical": outside_prefix_ok,
        "outside_suffix_byte_identical": outside_suffix_ok,
        "span_no_mask": no_mask,
        "span_no_eos": no_eos,
        "span_nonempty": span_nonempty,
    }
    return all(checks.values()), checks


# ---------------------------------------------------------------------------
# Policy run specification and provenance
# ---------------------------------------------------------------------------


class _FrozenMapping(dict[str, Any]):
    """Small immutable mapping that remains safe under standard serializers."""

    __slots__ = ()

    def __init__(self, values: Any = ()) -> None:
        items = values.items() if isinstance(values, Mapping) else values
        dict.__init__(self)
        for key, value in items:
            dict.__setitem__(self, str(key), _freeze_value(value))

    def _immutable(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise TypeError("_FrozenMapping is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable

    def __deepcopy__(self, memo: dict[int, Any]) -> "_FrozenMapping":
        del memo
        return self

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return type(self), (dict(self),)


def _freeze_value(value: Any) -> Any:
    if isinstance(value, _FrozenMapping):
        return value
    if isinstance(value, Mapping):
        return _FrozenMapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass(frozen=True, slots=True)
class PolicyRunSpec:
    """Derived, immutable evidence for one quality-report task invocation."""

    profile_name: str
    engine_kwargs: Mapping[str, Any]
    task_kind: str
    applicable: bool
    exact_reference: bool = field(init=False)

    def __post_init__(self) -> None:
        if self.task_kind not in _TASK_KINDS:
            raise ValueError(f"unsupported quality-report task kind: {self.task_kind!r}")
        frozen_kwargs = _freeze_value(self.engine_kwargs)
        object.__setattr__(self, "engine_kwargs", frozen_kwargs)
        object.__setattr__(self, "exact_reference", _is_exact_reference(frozen_kwargs))


@dataclass(frozen=True, slots=True)
class _QualityRunIdentity:
    """Private, deterministic identity for model/package and task inputs."""

    model_identity: Mapping[str, Any]
    task_inputs: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_identity", _freeze_value(self.model_identity))
        object.__setattr__(self, "task_inputs", _freeze_value(self.task_inputs))


def _profile_name(model_config: Mapping[str, Any]) -> str:
    value = model_config.get(_PROFILE_SELECTOR, "baseline")
    return "baseline" if value is None else str(value)


def _without_policy_overlay(model_config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in model_config.items()
        if key != _PROFILE_SELECTOR and key not in _POLICY_OVERLAY_CONFIG_KEYS
    }


def _without_profile_selector(model_config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in model_config.items()
        if key != _PROFILE_SELECTOR
    }


def _adapter_defaults(model_config: Mapping[str, Any]) -> tuple[dict[str, Any], bool, bool]:
    config = dict(model_config)
    has_architecture = (
        "architecture" in config and config["architecture"] is not None
    ) or (
        "architecture_family" in config
        and config["architecture_family"] is not None
    )
    if has_architecture:
        from edllm.model_definitions import (
            default_decode_policy_for_runtime,
            runtime_architecture_for_config,
        )

        runtime = runtime_architecture_for_config(config)
        decode_policy = default_decode_policy_for_runtime(runtime.key)
        return (
            decode_policy.resolve(config),
            runtime.uses_shifted_logits,
            runtime.uses_qk_norm,
        )

    uses_shifted_logits = config.get("uses_shifted_logits", True)
    uses_qk_norm = config.get("uses_qk_norm", False)
    if type(uses_shifted_logits) is not bool:
        raise ValueError("uses_shifted_logits must be a JSON boolean")
    if type(uses_qk_norm) is not bool:
        raise ValueError("uses_qk_norm must be a JSON boolean")
    return (
        {
            "decode_strategy": "block_threshold",
            "block_size": 8,
            "confidence_threshold": 0.9,
            "gamma": 0.1,
        },
        uses_shifted_logits,
        uses_qk_norm,
    )


def _normalize_compute_dtype(value: Any) -> str:
    normalized = "fp32" if value is None else str(value).strip().lower()
    if normalized in {"fp32", "float32"}:
        return "fp32"
    if normalized in {"bf16", "bfloat16"}:
        return "bf16"
    raise ValueError(
        f"unsupported mlx_compute_dtype: {value!r} (expected 'bf16' or 'fp32')"
    )


def _normalize_window_exec(value: Any) -> bool | None:
    if value is None or value is True or value is False:
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "auto":
            return None
        if normalized == "primary":
            return True
        if normalized == "oracle":
            return False
    raise ValueError(
        "mlx_diffusion_window_exec must be one of 'auto', 'primary', 'oracle', "
        "True, False, or None"
    )


def _resolved_runtime_config(model_config: Mapping[str, Any]):
    from edllm.runtime_config import resolve_mlx_runtime_config

    defaults, uses_shifted_logits, uses_qk_norm = _adapter_defaults(model_config)
    resolved = resolve_mlx_runtime_config(
        model_config,
        adapter_defaults=defaults,
    )
    return resolved, uses_shifted_logits, uses_qk_norm


def _common_engine_kwargs(
    model_config: Mapping[str, Any],
    *,
    _resolved: Any = None,
    _adapter_flags: tuple[bool, bool] | None = None,
) -> dict[str, Any]:
    if _resolved is None or _adapter_flags is None:
        resolved, uses_shifted_logits, uses_qk_norm = _resolved_runtime_config(
            model_config
        )
    else:
        resolved = _resolved
        uses_shifted_logits, uses_qk_norm = _adapter_flags
    return {
        "steps": resolved.steps,
        "eps": resolved.eps,
        "temperature": resolved.temperature,
        "top_p": resolved.top_p,
        "alg_temp": resolved.alg_temp,
        "attention_mask_kind": resolved.attention_mask_kind,
        "enable_layer0_kv_cache": resolved.layer0_kv_cache_enabled,
        "decode_strategy": resolved.decode_strategy,
        "confidence_threshold": resolved.confidence_threshold,
        "block_size": resolved.block_size,
        "gamma": resolved.gamma,
        "uses_shifted_logits": uses_shifted_logits,
        "uses_qk_norm": uses_qk_norm,
        "compute_dtype": _normalize_compute_dtype(resolved.compute_dtype),
        "window_exec": resolved.window_exec,
        "linear_quant_bits": resolved.linear_quant_bits,
        "linear_quant_group_size": resolved.linear_quant_group_size,
    }


def _engine_kwargs_for_task(
    model_config: Mapping[str, Any],
    task_kind: str,
) -> dict[str, Any]:
    resolved, uses_shifted_logits, uses_qk_norm = _resolved_runtime_config(
        model_config
    )
    kwargs = _common_engine_kwargs(
        model_config,
        _resolved=resolved,
        _adapter_flags=(uses_shifted_logits, uses_qk_norm),
    )
    if task_kind != "chat":
        return kwargs

    kwargs.update(
        {
            "enable_prefix_kv_cache": resolved.prefix_kv_cache,
            "prefix_kv_refresh_interval": resolved.prefix_kv_refresh_interval,
            "enable_block_kv_cache": resolved.block_kv_cache,
            "block_kv_refresh_interval": resolved.block_kv_refresh_interval,
            "enable_d2f_exact_kv_cache": resolved.d2f_exact_kv_cache,
            "d2f_add_threshold": resolved.d2f_add_threshold,
            "d2f_activate_threshold": resolved.d2f_activate_threshold,
            "d2f_commit_policy": resolved.d2f_commit_policy,
            "d2f_skip_committed_rows": resolved.d2f_skip_committed_rows,
            "d2f_skip_refresh_interval": resolved.d2f_skip_refresh_interval,
            "enable_delayed_kv": False,
            "delayed_kv_scope": "layer0",
            "delayed_kv_max_stale_steps": None,
            "delayed_kv_refresh_interval": None,
            "delayed_kv_upper_layer_range": None,
            "delayed_kv_stale_epoch_budget": None,
            "error_fixing": resolved.error_fixing_kwargs(),
            "trace_canvas": resolved.trace_canvas,
            "save_snapshot_path": resolved.snapshot_save_path,
            "enable_canvas_cache": resolved.canvas_cache,
            "enable_cross_request_cache": resolved.cross_request_cache,
            "canvas_cache_budget_bytes": resolved.canvas_cache_budget_bytes,
        }
    )
    return kwargs


def _is_exact_reference(engine_kwargs: Mapping[str, Any]) -> bool:
    if bool(engine_kwargs.get("enable_prefix_kv_cache", False)):
        return False
    if bool(engine_kwargs.get("enable_block_kv_cache", False)):
        return False
    if bool(engine_kwargs.get("d2f_skip_committed_rows", False)):
        return False
    if engine_kwargs.get("linear_quant_bits") is not None:
        return False
    compute_dtype = str(engine_kwargs.get("compute_dtype", "fp32")).lower()
    if compute_dtype not in {"fp32", "float32"}:
        return False
    return not bool(engine_kwargs.get("enable_delayed_kv", False))


def _policy_run_spec(
    *,
    profile_name: str,
    engine_kwargs: Mapping[str, Any],
    task_kind: str,
    applicable: bool,
) -> PolicyRunSpec:
    return PolicyRunSpec(
        profile_name=profile_name,
        engine_kwargs=dict(engine_kwargs),
        task_kind=task_kind,
        applicable=applicable,
    )


def build_policy_run_specs(
    model_config: Mapping[str, Any],
) -> list[PolicyRunSpec]:
    """Build canonical per-task policy evidence from a complete model config."""
    source_config = dict(model_config)
    requested_profile = _profile_name(source_config)

    profile_config = dict(source_config)
    profile_config[_PROFILE_SELECTOR] = requested_profile
    resolved_profile_config = presets.apply_policy_profile(profile_config)

    execution_baseline_config = _without_profile_selector(source_config)
    resolved_execution_baseline_config = presets.apply_policy_profile(
        execution_baseline_config
    )

    specs = []
    for task_kind in _TASK_KINDS:
        applicable = (
            requested_profile != "baseline"
            and presets.policy_profile_supports_task(requested_profile, task_kind)
        )
        execution_config = (
            resolved_profile_config
            if applicable
            else resolved_execution_baseline_config
        )
        specs.append(
            _policy_run_spec(
                profile_name=requested_profile if applicable else "baseline",
                engine_kwargs=_engine_kwargs_for_task(execution_config, task_kind),
                task_kind=task_kind,
                applicable=applicable,
            )
        )
    return specs


# ---------------------------------------------------------------------------
# Exact-match rate
# ---------------------------------------------------------------------------


def _exact_match_rate(
    candidate_tokens: list[int],
    reference_tokens: list[int],
) -> tuple[float, int]:
    """Compare candidate to reference token-by-token at aligned positions.

    Returns (rate, match_count). Rate = matches / max(len_c, len_r, 1), so an
    unequal length is penalized (a short candidate cannot score 1.0 against a
    longer reference). Equal non-empty sequences have rate 1.0. Self-reference
    is handled by the task-record builder and remains 1.0 even for an empty
    output.

    Two genuinely-identical EMPTY outputs are a FULL match (1.0): the
    max-denominator alone would map them to ``0/1 == 0.0``, reporting a
    candidate that reproduced the reference exactly (both emitted nothing —
    immediate EOS, an empty infill/edit) as complete disagreement and failing
    gates that assert 1.0 for byte-identical outputs. Only the both-empty case
    is special-cased; empty-vs-non-empty stays 0.0.
    """
    if not candidate_tokens and not reference_tokens:
        return 1.0, 0
    aligned_count = min(len(candidate_tokens), len(reference_tokens))
    matches = sum(
        1
        for i in range(aligned_count)
        if candidate_tokens[i] == reference_tokens[i]
    )
    denominator = max(len(candidate_tokens), len(reference_tokens), 1)
    return matches / denominator, matches


# ---------------------------------------------------------------------------
# QualityReport (frozen dataclass)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TaskQualityRecord:
    """Quality record for a single task invocation."""

    task: str                          # "chat" | "infill" | "edit"
    output_tokens: list[int]           # visible token ids (fill for infill/edit)
    output_text: str                   # decoded text of the output span
    # Honesty flag: True iff a REAL tokenizer ``decode`` produced ``output_text``.
    # When False (the decimal-id fallback), the printable/garbage dimension of
    # ``degeneration`` (``printable_ratio`` and the chat ``printable_ok`` check)
    # is NOT meaningful — decimal digits are always printable.
    text_is_real_decode: bool
    # Layer 1: exact-match
    exact_match_rate: float            # 1.0 when reference=None (self)
    # Layer 2: task success
    task_success: bool
    task_detail: dict[str, Any]        # per-check booleans
    # Layer 3: degeneration
    degeneration: dict[str, Any]       # components + degeneration_score
    # Rows-forwarded metrics
    forwarded_tokens: int              # mlx_timing_active_tokens_total
    committed_tokens: int              # committed_token_count
    active_tokens: int                 # mlx_active_tokens_per_forward (rounded)
    rows_forwarded_per_correct_token: float  # forwarded_tokens / max(1, exact_match_count)
    candidate_length: int = 0
    reference_length: int = 0
    effective_profile_name: str = "baseline"
    profile_applicable: bool = False
    effective_engine_kwargs: Mapping[str, Any] = field(default_factory=dict)
    exact_reference: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "effective_engine_kwargs",
            _freeze_value(self.effective_engine_kwargs),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "output_tokens": list(self.output_tokens),
            "output_text": self.output_text,
            "text_is_real_decode": self.text_is_real_decode,
            "exact_match_rate": self.exact_match_rate,
            "task_success": self.task_success,
            "task_detail": dict(self.task_detail),
            "degeneration": dict(self.degeneration),
            "forwarded_tokens": self.forwarded_tokens,
            "committed_tokens": self.committed_tokens,
            "active_tokens": self.active_tokens,
            "rows_forwarded_per_correct_token": self.rows_forwarded_per_correct_token,
            "candidate_length": self.candidate_length,
            "reference_length": self.reference_length,
            "effective_profile_name": self.effective_profile_name,
            "profile_applicable": self.profile_applicable,
            "effective_engine_kwargs": _json_safe(self.effective_engine_kwargs),
            "exact_reference": self.exact_reference,
        }


@dataclass(frozen=True, slots=True)
class _ReferenceTaskEvidence:
    task: str
    output_tokens: tuple[int, ...]
    output_text: str
    text_is_real_decode: bool
    exact_match_rate: float
    task_success: bool
    task_detail: Mapping[str, Any]
    degeneration: Mapping[str, Any]
    forwarded_tokens: int
    committed_tokens: int
    active_tokens: int
    rows_forwarded_per_correct_token: float
    candidate_length: int
    reference_length: int
    effective_profile_name: str
    profile_applicable: bool
    effective_engine_kwargs: Mapping[str, Any]
    exact_reference: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_detail", _freeze_value(self.task_detail))
        object.__setattr__(self, "degeneration", _freeze_value(self.degeneration))
        object.__setattr__(
            self,
            "effective_engine_kwargs",
            _freeze_value(self.effective_engine_kwargs),
        )


@dataclass(frozen=True, slots=True)
class QualityReport:
    """Structured quality report for a chat + infill + edit task set.

    With ``reference=None``, ``exact_match_rate == 1.0`` because each task is its
    own comparison target; ``TaskQualityRecord.exact_reference`` independently
    records whether its resolved math is exact. With ``reference=<run>``, exact
    match measures aligned token agreement and penalizes length differences.
    """

    tasks: list[TaskQualityRecord]      # one record per task, in order
    aggregate: dict[str, Any]           # summary across all tasks
    _canonical_baseline_specs: tuple[PolicyRunSpec, ...] = field(
        default_factory=tuple,
        init=False,
        repr=False,
        compare=False,
    )
    _self_reference_kind: str = field(
        default="unverified_reference",
        init=False,
        repr=False,
        compare=False,
    )
    _self_profile_name: str = field(
        default="unknown",
        init=False,
        repr=False,
        compare=False,
    )
    _run_identity: _QualityRunIdentity | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _reference_task_evidence: tuple[_ReferenceTaskEvidence, ...] = field(
        default_factory=tuple,
        init=False,
        repr=False,
        compare=False,
    )
    _evidence_seal: str | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tasks": [t.to_dict() for t in self.tasks],
            "aggregate": _json_safe(self.aggregate),
        }

    def __getstate__(self) -> dict[str, Any]:
        return {
            "version": _QUALITY_REPORT_STATE_VERSION,
            "values": tuple(
                getattr(self, name) for name in _QUALITY_REPORT_STATE_FIELDS
            ),
        }

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, Mapping):
            version = state.get("version")
            values = state.get("values")
            if (
                version != _QUALITY_REPORT_STATE_VERSION
                or not isinstance(values, (list, tuple))
                or len(values) != len(_QUALITY_REPORT_STATE_FIELDS)
            ):
                raise ValueError("unsupported QualityReport pickle state")
        elif isinstance(state, (list, tuple)) and len(state) == 6:
            # 4a4d7b0: snapshot was the sixth slot, before _run_identity existed.
            tasks, aggregate, specs, _kind, profile, snapshots = state
            values = (
                tasks,
                aggregate,
                specs,
                "unverified_reference",
                profile,
                None,
                snapshots,
                None,
            )
        elif isinstance(state, (list, tuple)) and len(state) == 7:
            # Pre-seal run-identity state is readable but cannot remain verified.
            tasks, aggregate, specs, _kind, profile, identity, snapshots = state
            values = (
                tasks,
                aggregate,
                specs,
                "unverified_reference",
                profile,
                identity,
                snapshots,
                None,
            )
        else:
            raise ValueError("unsupported QualityReport pickle state")

        for name, value in zip(_QUALITY_REPORT_STATE_FIELDS, values, strict=True):
            object.__setattr__(self, name, value)


def _policy_run_spec_payload(spec: PolicyRunSpec) -> dict[str, Any]:
    return {
        "profile_name": spec.profile_name,
        "engine_kwargs": _json_safe(spec.engine_kwargs),
        "task_kind": spec.task_kind,
        "applicable": spec.applicable,
        "exact_reference": spec.exact_reference,
    }


def _reference_task_evidence_payload(
    evidence: _ReferenceTaskEvidence,
) -> dict[str, Any]:
    return {
        "task": evidence.task,
        "output_tokens": list(evidence.output_tokens),
        "output_text": evidence.output_text,
        "text_is_real_decode": evidence.text_is_real_decode,
        "exact_match_rate": evidence.exact_match_rate,
        "task_success": evidence.task_success,
        "task_detail": _json_safe(evidence.task_detail),
        "degeneration": _json_safe(evidence.degeneration),
        "forwarded_tokens": evidence.forwarded_tokens,
        "committed_tokens": evidence.committed_tokens,
        "active_tokens": evidence.active_tokens,
        "rows_forwarded_per_correct_token": (
            evidence.rows_forwarded_per_correct_token
        ),
        "candidate_length": evidence.candidate_length,
        "reference_length": evidence.reference_length,
        "effective_profile_name": evidence.effective_profile_name,
        "profile_applicable": evidence.profile_applicable,
        "effective_engine_kwargs": _json_safe(evidence.effective_engine_kwargs),
        "exact_reference": evidence.exact_reference,
    }


def _compute_evidence_seal(report: QualityReport) -> str:
    identity = report._run_identity
    payload = {
        "self_reference_kind": report._self_reference_kind,
        "self_profile_name": report._self_profile_name,
        "run_identity": (
            None
            if identity is None
            else {
                "model_identity": _json_safe(identity.model_identity),
                "task_inputs": _json_safe(identity.task_inputs),
            }
        ),
        "reference_task_evidence": [
            _reference_task_evidence_payload(item)
            for item in report._reference_task_evidence
        ],
        "canonical_baseline_specs": [
            _policy_run_spec_payload(spec)
            for spec in report._canonical_baseline_specs
        ],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(b"edllm-quality-evidence-v1\0" + encoded).hexdigest()


# ---------------------------------------------------------------------------
# Engine call helpers
# ---------------------------------------------------------------------------


def _run_chat(
    engine: Any,
    *,
    prompt_token_ids: list[int],
    max_tokens: int,
    mask_token_id: int,
    eos_token_id: int,
    decode_strategy: str,
    confidence_threshold: float,
    block_size: int,
    steps: int,
    uses_shifted_logits: bool,
    uses_qk_norm: bool,
    generate_entropy_kwargs: Mapping[str, Any] | None = None,
) -> tuple[list[int], dict]:
    """Run chat (generate_entropy) and return (visible_token_ids, timing_metrics)."""
    call_kwargs = _merge_runtime_kwargs(
        {
            "prompt_token_ids": prompt_token_ids,
            "max_tokens": max_tokens,
            "mask_token_id": mask_token_id,
            "eos_token_id": eos_token_id,
            "steps": steps,
            "decode_strategy": decode_strategy,
            "confidence_threshold": confidence_threshold,
            "block_size": block_size,
            "uses_shifted_logits": uses_shifted_logits,
            "uses_qk_norm": uses_qk_norm,
            "enable_layer0_kv_cache": False,
        },
        generate_entropy_kwargs,
        forbidden={
            "prompt_token_ids",
            "max_tokens",
            "mask_token_id",
            "eos_token_id",
        },
    )
    result = engine.generate_entropy(**call_kwargs)
    tm = dict(result.timing_metrics or {})
    return list(result.visible_generated_token_ids), tm, int(result.committed_token_count)


def _run_infill(
    engine: Any,
    *,
    prefix_token_ids: list[int],
    suffix_token_ids: list[int],
    max_fill_tokens: int,
    mask_token_id: int,
    eos_token_id: int,
    decode_strategy: str,
    confidence_threshold: float,
    block_size: int,
    steps: int,
    uses_shifted_logits: bool,
    uses_qk_norm: bool,
    generate_infill_kwargs: Mapping[str, Any] | None = None,
) -> tuple[list[int], list[int], dict, int]:
    """Run infill (generate_infill); return (fill_ids, full_ids, timing, committed)."""
    call_kwargs = _merge_runtime_kwargs(
        {
            "prefix_token_ids": prefix_token_ids,
            "suffix_token_ids": suffix_token_ids,
            "max_fill_tokens": max_fill_tokens,
            "mask_token_id": mask_token_id,
            "eos_token_id": eos_token_id,
            "steps": steps,
            "decode_strategy": decode_strategy,
            "confidence_threshold": confidence_threshold,
            "block_size": block_size,
            "uses_shifted_logits": uses_shifted_logits,
            "uses_qk_norm": uses_qk_norm,
            "enable_layer0_kv_cache": False,
        },
        generate_infill_kwargs,
        forbidden={
            "prefix_token_ids",
            "suffix_token_ids",
            "max_fill_tokens",
            "mask_token_id",
            "eos_token_id",
        },
    )
    result = engine.generate_infill(**call_kwargs)
    tm = dict(result.timing_metrics or {})
    return (
        list(result.fill_token_ids),
        list(result.full_token_ids),
        tm,
        int(result.committed_token_count),
    )


def _merge_runtime_kwargs(
    base: dict[str, Any],
    extra: Mapping[str, Any] | None,
    *,
    forbidden: set[str],
) -> dict[str, Any]:
    merged = dict(base)
    if not extra:
        return merged
    blocked = sorted(set(extra).intersection(forbidden))
    if blocked:
        raise ValueError(
            "quality_report runtime kwargs cannot override task identity fields: "
            + ", ".join(blocked)
        )
    merged.update(dict(extra))
    return merged


def _normalized_legacy_engine_value(key: str, value: Any) -> Any:
    if key == "compute_dtype":
        return _normalize_compute_dtype(value)
    if key == "window_exec":
        return _normalize_window_exec(value)
    if key in _BOOL_ENGINE_KEYS:
        return bool(value)
    if key == "uses_qk_norm":
        return None if value is None else bool(value)
    if key in _INT_ENGINE_KEYS:
        return None if value is None else int(value)
    if key in _FLOAT_ENGINE_KEYS:
        return float(value)
    if key == "delayed_kv_upper_layer_range":
        return None if value is None else tuple(int(item) for item in value)
    if key in {"cross_session_cache_root", "save_snapshot_path"}:
        return None if value is None else str(value)
    if key == "error_fixing":
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise ValueError("quality_report error_fixing compatibility value must be a mapping")
        return _json_safe(value)
    return str(value) if key in {
        "attention_mask_kind",
        "d2f_commit_policy",
        "decode_strategy",
        "delayed_kv_scope",
    } else value


def _replace_spec_kwargs(
    spec: PolicyRunSpec,
    engine_kwargs: Mapping[str, Any],
) -> PolicyRunSpec:
    return _policy_run_spec(
        profile_name=spec.profile_name,
        engine_kwargs=engine_kwargs,
        task_kind=spec.task_kind,
        applicable=spec.applicable,
    )


def _legacy_base_engine_kwargs(
    *,
    task_kind: str,
    decode_strategy: str,
    confidence_threshold: float,
    block_size: int,
    steps: int,
    uses_shifted_logits: bool,
    uses_qk_norm: bool,
) -> dict[str, Any]:
    del task_kind
    return {
        "decode_strategy": str(decode_strategy),
        "confidence_threshold": float(confidence_threshold),
        "block_size": int(block_size),
        "steps": int(steps),
        "uses_shifted_logits": bool(uses_shifted_logits),
        "uses_qk_norm": bool(uses_qk_norm),
        "enable_layer0_kv_cache": False,
    }


def _apply_legacy_base(
    specs: list[PolicyRunSpec],
    *,
    decode_strategy: str,
    confidence_threshold: float,
    block_size: int,
    steps: int,
    uses_shifted_logits: bool,
    uses_qk_norm: bool,
) -> list[PolicyRunSpec]:
    resolved: list[PolicyRunSpec] = []
    for spec in specs:
        base = _legacy_base_engine_kwargs(
            task_kind=spec.task_kind,
            decode_strategy=decode_strategy,
            confidence_threshold=confidence_threshold,
            block_size=block_size,
            steps=steps,
            uses_shifted_logits=uses_shifted_logits,
            uses_qk_norm=uses_qk_norm,
        )
        merged = dict(spec.engine_kwargs)
        merged.update(base)
        if spec.applicable:
            for key in _PROFILE_ENGINE_KEYS:
                if key in spec.engine_kwargs:
                    merged[key] = spec.engine_kwargs[key]
        resolved.append(_replace_spec_kwargs(spec, merged))
    return resolved


def _explicit_legacy_base_values(**values: Any) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not _UNSET}


def _has_evidence_bearing_runtime_kwargs(
    *runtime_kwargs: Mapping[str, Any] | None,
) -> bool:
    return any(
        bool(set(kwargs) - _CALL_ONLY_ENGINE_KEYS)
        for kwargs in runtime_kwargs
        if kwargs
    )


def _apply_explicit_legacy_base(
    specs: list[PolicyRunSpec],
    explicit_values: Mapping[str, Any],
) -> list[PolicyRunSpec]:
    resolved: list[PolicyRunSpec] = []
    for spec in specs:
        merged = dict(spec.engine_kwargs)
        for key, raw_value in explicit_values.items():
            merged[key] = _normalized_legacy_engine_value(key, raw_value)
        if spec.applicable:
            for key in _PROFILE_ENGINE_KEYS:
                if key in spec.engine_kwargs:
                    merged[key] = spec.engine_kwargs[key]
        resolved.append(_replace_spec_kwargs(spec, merged))
    return resolved


def _validate_legacy_base_agreement(
    specs: list[PolicyRunSpec],
    explicit_values: Mapping[str, Any],
) -> None:
    for spec in specs:
        for key, raw_value in explicit_values.items():
            value = _normalized_legacy_engine_value(key, raw_value)
            expected = spec.engine_kwargs.get(key)
            if _freeze_value(value) != expected:
                raise ValueError(
                    f"quality_report top-level compatibility value {key!r} "
                    f"disagrees with the derived {spec.task_kind} spec: "
                    f"{value!r} != {expected!r}"
                )


def _adapt_legacy_runtime_kwargs(
    specs: list[PolicyRunSpec],
    *,
    requested_profile: str,
    generate_entropy_kwargs: Mapping[str, Any] | None,
    generate_infill_kwargs: Mapping[str, Any] | None,
    complete_model_config: bool,
) -> tuple[list[PolicyRunSpec], dict[str, dict[str, Any]]]:
    by_task = {spec.task_kind: spec for spec in specs}
    call_extras: dict[str, dict[str, Any]] = {
        task_kind: {} for task_kind in _TASK_KINDS
    }

    def _adapt(
        task_kind: str,
        extra: Mapping[str, Any] | None,
        allowed: frozenset[str],
        identity_fields: frozenset[str],
    ) -> None:
        if not extra:
            return
        blocked = sorted(set(extra).intersection(identity_fields))
        if blocked:
            raise ValueError(
                "quality_report runtime kwargs cannot override task identity fields: "
                + ", ".join(blocked)
            )
        unknown = sorted(set(extra) - allowed)
        if unknown:
            raise ValueError(
                f"quality_report {task_kind} compatibility kwargs are unsupported: "
                + ", ".join(unknown)
            )
        spec = by_task[task_kind]
        merged = dict(spec.engine_kwargs)
        for key, raw_value in extra.items():
            if key in _RETIRED_KV_ENGINE_DEFAULTS:
                expected_retired = _RETIRED_KV_ENGINE_DEFAULTS[key]
                if (
                    type(raw_value) is not type(expected_retired)
                    or raw_value != expected_retired
                ):
                    raise ValueError(
                        f"quality_report compatibility kwarg {key!r} is no "
                        "longer supported in DiffusorRT 0.2; only the neutral value "
                        f"{expected_retired!r} is accepted"
                    )
            if key not in spec.engine_kwargs:
                call_extras[task_kind][key] = raw_value
                continue
            value = _normalized_legacy_engine_value(key, raw_value)
            expected = spec.engine_kwargs.get(key)
            must_agree = complete_model_config or (
                spec.applicable
                and key in _PROFILE_ENGINE_KEYS
            )
            if must_agree and _freeze_value(value) != expected:
                raise ValueError(
                    f"quality_report {task_kind} compatibility kwarg {key!r} "
                    f"disagrees with resolved profile {requested_profile!r}"
                )
            merged[key] = value
        by_task[task_kind] = _replace_spec_kwargs(spec, merged)

    _adapt(
        "chat",
        generate_entropy_kwargs,
        _CHAT_ENGINE_KEYS,
        _CHAT_IDENTITY_FIELDS,
    )
    _adapt(
        "infill",
        generate_infill_kwargs,
        _INFILL_ENGINE_KEYS,
        _INFILL_IDENTITY_FIELDS,
    )
    _adapt(
        "edit",
        generate_infill_kwargs,
        _INFILL_ENGINE_KEYS,
        _INFILL_IDENTITY_FIELDS,
    )
    return [by_task[task_kind] for task_kind in _TASK_KINDS], call_extras


def _canonical_baseline_specs(
    model_config: Mapping[str, Any],
) -> list[PolicyRunSpec]:
    baseline_config = _without_policy_overlay(model_config)
    baseline_config[_PROFILE_SELECTOR] = "baseline"
    return build_policy_run_specs(baseline_config)


def _has_complete_canonical_model_config(
    model_config: Mapping[str, Any],
) -> bool:
    required = (
        _MODEL_IDENTITY_REQUIRED_CONFIG_KEYS | _GRAPH_IR_REQUIRED_CONFIG_KEYS
    )
    if any(key not in model_config for key in required):
        return False

    architecture = model_config["architecture"]
    if not isinstance(architecture, str) or architecture.strip() == "":
        return False

    positive_integer_keys = required - {
        "architecture",
        "rms_norm_eps",
        "rope_theta",
    }
    if any(
        type(model_config[key]) is not int or model_config[key] <= 0
        for key in positive_integer_keys
    ):
        return False

    for key in ("rms_norm_eps", "rope_theta"):
        value = model_config[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            return False

    from edllm.graph import build_graph_ir

    try:
        build_graph_ir(_without_policy_overlay(model_config))
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _snapshot_task_inputs(
    *,
    mask_token_id: int,
    eos_token_id: int,
    chat_prompt_token_ids: list[int],
    chat_max_tokens: int,
    infill_prefix_token_ids: list[int],
    infill_suffix_token_ids: list[int],
    infill_max_fill_tokens: int,
    edit_prefix_token_ids: list[int],
    edit_suffix_token_ids: list[int],
    edit_max_fill_tokens: int,
) -> Mapping[str, Any]:
    return _FrozenMapping(
        {
            "mask_token_id": int(mask_token_id),
            "eos_token_id": int(eos_token_id),
            "chat": {
                "prompt_token_ids": tuple(int(token) for token in chat_prompt_token_ids),
                "max_tokens": int(chat_max_tokens),
            },
            "infill": {
                "prefix_token_ids": tuple(
                    int(token) for token in infill_prefix_token_ids
                ),
                "suffix_token_ids": tuple(
                    int(token) for token in infill_suffix_token_ids
                ),
                "max_fill_tokens": int(infill_max_fill_tokens),
            },
            "edit": {
                "prefix_token_ids": tuple(
                    int(token) for token in edit_prefix_token_ids
                ),
                "suffix_token_ids": tuple(
                    int(token) for token in edit_suffix_token_ids
                ),
                "max_fill_tokens": int(edit_max_fill_tokens),
            },
        }
    )


def _build_run_identity(
    engine: Any,
    *,
    model_config: Mapping[str, Any],
    task_inputs: Mapping[str, Any],
) -> _QualityRunIdentity:
    model_identity: dict[str, Any] = {
        "model_config": _without_policy_overlay(model_config),
    }
    for attribute in ("package_dir", "merged_weights_ref"):
        try:
            value = getattr(engine, attribute)
        except AttributeError:
            continue
        if attribute == "package_dir" and value is not None:
            value = str(Path(value))
        elif attribute == "merged_weights_ref" and value is not None:
            value = str(value)
        model_identity[attribute] = value
    model_identity.update(_resident_weight_identity(engine))
    return _QualityRunIdentity(
        model_identity=model_identity,
        task_inputs=task_inputs,
    )


def _resident_weight_identity(engine: Any) -> dict[str, Any]:
    from edllm.backends import mlx_dream

    if not isinstance(engine, mlx_dream.MLXDreamEngine):
        return {}

    engine._resident_weights(require_source_identity=True)
    state = engine._weight_state

    digest = getattr(state, "source_digest", None)
    source_stat = getattr(state, "source_stat", None)
    provenance = getattr(state, "_loader_provenance", None)
    digest_is_sha256 = (
        isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
    )
    stat_is_bound = (
        isinstance(source_stat, tuple)
        and len(source_stat) == 5
        and all(type(item) is int for item in source_stat)
    )
    if (
        state is None
        or provenance is not mlx_dream._MLX_WEIGHT_LOADER_PROVENANCE
        or not digest_is_sha256
        or not stat_is_bound
    ):
        raise ValueError(
            "quality_report requires loader-backed resident weight identity "
            "with source_digest and source_stat"
        )

    identity: dict[str, Any] = {
        "resident_source_digest": digest,
        "resident_source_stat": source_stat,
    }
    source_path = getattr(state, "source_path", None)
    if source_path is not None:
        identity["resident_source_path"] = str(Path(source_path))
    return identity


def _validate_engine_graph_ir(engine: Any) -> None:
    graphs: list[Any] = []
    for attribute in ("graph", "_graph"):
        graph = getattr(engine, attribute, None)
        if graph is not None and all(graph is not item for item in graphs):
            graphs.append(graph)
    if not graphs:
        return

    engine_model_config = getattr(engine, "model_config", None)
    if not isinstance(engine_model_config, Mapping):
        raise ValueError(
            "quality_report engine GraphIR validation requires a complete "
            "engine.model_config"
        )

    from edllm.graph import build_graph_ir

    try:
        expected = build_graph_ir(
            presets.apply_policy_profile(dict(engine_model_config))
        ).to_dict()
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "quality_report engine GraphIR validation requires a complete "
            "engine.model_config"
        ) from exc

    for graph in graphs:
        to_dict = getattr(graph, "to_dict", None)
        if not callable(to_dict):
            raise ValueError(
                "quality_report engine GraphIR disagrees with engine.model_config"
            )
        try:
            actual = to_dict()
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "quality_report engine GraphIR disagrees with engine.model_config"
            ) from exc
        if actual != expected:
            raise ValueError(
                "quality_report engine GraphIR disagrees with engine.model_config"
            )


def _normalized_base_model_config(
    model_config: Mapping[str, Any],
) -> Mapping[str, Any]:
    base = _without_policy_overlay(model_config)
    resolved, _, _ = _resolved_runtime_config(base)
    normalized = dict(base)
    normalized["mlx_compute_dtype"] = _normalize_compute_dtype(
        resolved.compute_dtype
    )
    normalized["mlx_linear_quant_bits"] = resolved.linear_quant_bits
    normalized["mlx_linear_quant_group_size"] = resolved.linear_quant_group_size
    return _freeze_value(normalized)


def _validate_model_config_evidence_agreement(
    engine_model_config: Mapping[str, Any],
    supplied_model_config: Mapping[str, Any],
) -> None:
    # Resolve the full engine config as well as the caller config. This preserves
    # profile-overlay freedom while keeping the strict public validator in charge.
    build_policy_run_specs(engine_model_config)
    engine_base = _normalized_base_model_config(engine_model_config)
    supplied_base = _normalized_base_model_config(supplied_model_config)
    if engine_base == supplied_base:
        return

    differing_keys = sorted(
        key
        for key in set(engine_base) | set(supplied_base)
        if engine_base.get(key, _UNSET) != supplied_base.get(key, _UNSET)
    )
    raise ValueError(
        "quality_report engine model_config base model disagrees with the supplied "
        "model_config after removing policy profile overlays: "
        + ", ".join(differing_keys)
    )


def _classify_self_reference(
    *,
    requested_profile: str,
    run_specs: list[PolicyRunSpec],
    canonical_specs: list[PolicyRunSpec],
    legacy_runtime_override: bool,
    has_complete_model_config: bool,
    compared_reference: bool,
) -> str:
    run_by_task = {spec.task_kind: spec for spec in run_specs}
    canonical_by_task = {spec.task_kind: spec for spec in canonical_specs}
    if set(run_by_task) != set(_TASK_KINDS):
        return "unverified_reference"
    if not all(spec.exact_reference for spec in run_by_task.values()):
        return "unverified_reference"
    if (
        not has_complete_model_config
        or compared_reference
        or requested_profile != "baseline"
        or legacy_runtime_override
    ):
        return "verified_profile_reference"
    if set(canonical_by_task) != set(_TASK_KINDS):
        return "unverified_reference"
    canonical_match = all(
        run_by_task[task_kind].profile_name == canonical_by_task[task_kind].profile_name
        and run_by_task[task_kind].applicable == canonical_by_task[task_kind].applicable
        and run_by_task[task_kind].engine_kwargs
        == canonical_by_task[task_kind].engine_kwargs
        for task_kind in _TASK_KINDS
    )
    return "exact_baseline" if canonical_match else "verified_profile_reference"


def _engine_call_kwargs(
    spec: PolicyRunSpec,
    call_extras: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    kwargs = {
        key: _json_safe(value)
        for key, value in spec.engine_kwargs.items()
        if key not in _EVIDENCE_ONLY_ENGINE_KEYS
    }
    if call_extras:
        kwargs.update(call_extras)
    return kwargs


def _make_task_record(
    *,
    task: str,
    output_tokens: list[int],
    output_text: str,
    text_is_real_decode: bool,
    reference_tokens: list[int] | None,
    timing_metrics: dict,
    committed_tokens: int,
    mask_token_id: int,
    eos_token_id: int,
    task_success: bool,
    task_detail: dict[str, Any],
    run_spec: PolicyRunSpec,
) -> TaskQualityRecord:
    degen = _degeneration(
        output_tokens,
        output_text,
        mask_token_id=mask_token_id,
        eos_token_id=eos_token_id,
    )

    if reference_tokens is None:
        emr = 1.0
        match_count = len(output_tokens)
        reference_length = len(output_tokens)
    else:
        emr, match_count = _exact_match_rate(output_tokens, reference_tokens)
        reference_length = len(reference_tokens)

    forwarded = int(timing_metrics.get("mlx_timing_active_tokens_total", 0))
    active_per_fwd = timing_metrics.get("mlx_active_tokens_per_forward", 0)
    active_tok = round(active_per_fwd)
    rows_fwd_per_correct = forwarded / max(1, match_count)

    return TaskQualityRecord(
        task=task,
        output_tokens=output_tokens,
        output_text=output_text,
        text_is_real_decode=text_is_real_decode,
        exact_match_rate=emr,
        task_success=task_success,
        task_detail=task_detail,
        degeneration=degen,
        forwarded_tokens=forwarded,
        committed_tokens=committed_tokens,
        active_tokens=active_tok,
        rows_forwarded_per_correct_token=rows_fwd_per_correct,
        candidate_length=len(output_tokens),
        reference_length=reference_length,
        effective_profile_name=run_spec.profile_name,
        profile_applicable=run_spec.applicable,
        effective_engine_kwargs=_json_safe(run_spec.engine_kwargs),
        exact_reference=run_spec.exact_reference,
    )


def _snapshot_task_record(record: TaskQualityRecord) -> _ReferenceTaskEvidence:
    return _ReferenceTaskEvidence(
        task=record.task,
        output_tokens=tuple(record.output_tokens),
        output_text=record.output_text,
        text_is_real_decode=record.text_is_real_decode,
        exact_match_rate=record.exact_match_rate,
        task_success=record.task_success,
        task_detail=record.task_detail,
        degeneration=record.degeneration,
        forwarded_tokens=record.forwarded_tokens,
        committed_tokens=record.committed_tokens,
        active_tokens=record.active_tokens,
        rows_forwarded_per_correct_token=record.rows_forwarded_per_correct_token,
        candidate_length=record.candidate_length,
        reference_length=record.reference_length,
        effective_profile_name=record.effective_profile_name,
        profile_applicable=record.profile_applicable,
        effective_engine_kwargs=record.effective_engine_kwargs,
        exact_reference=record.exact_reference,
    )


def _copy_reference_task_record(
    evidence: _ReferenceTaskEvidence,
) -> TaskQualityRecord:
    output_length = len(evidence.output_tokens)
    return TaskQualityRecord(
        task=evidence.task,
        output_tokens=list(evidence.output_tokens),
        output_text=evidence.output_text,
        text_is_real_decode=evidence.text_is_real_decode,
        exact_match_rate=1.0,
        task_success=evidence.task_success,
        task_detail=dict(evidence.task_detail),
        degeneration=dict(evidence.degeneration),
        forwarded_tokens=evidence.forwarded_tokens,
        committed_tokens=evidence.committed_tokens,
        active_tokens=evidence.active_tokens,
        rows_forwarded_per_correct_token=(
            evidence.forwarded_tokens / max(1, output_length)
        ),
        candidate_length=output_length,
        reference_length=output_length,
        effective_profile_name=evidence.effective_profile_name,
        profile_applicable=False,
        effective_engine_kwargs=evidence.effective_engine_kwargs,
        exact_reference=evidence.exact_reference,
    )


def _require_complete_reference_tasks(records: list[Any]) -> dict[str, Any]:
    task_names: list[str] = []
    for record in records:
        if isinstance(record, Mapping):
            task_name = record.get("task")
        else:
            task_name = getattr(record, "task", None)
        task_names.append(str(task_name) if task_name is not None else "")
    if (
        len(records) != len(_TASK_KINDS)
        or set(task_names) != set(_TASK_KINDS)
        or any(task_names.count(task_kind) != 1 for task_kind in _TASK_KINDS)
    ):
        raise ValueError(
            "quality_report reference tasks must contain exactly one chat, "
            "infill, and edit record"
        )
    return dict(zip(task_names, records, strict=True))


def _has_consistent_private_evidence(reference: Any) -> bool:
    """Check accidental corruption, not hostile same-process mutation.

    Code with ``object.__setattr__`` can also recompute this unkeyed digest. The
    seal is a deterministic consistency guard for serialization and object drift,
    not a security boundary against code executing in this process.
    """
    if not isinstance(reference, QualityReport):
        return False
    try:
        identity = reference._run_identity
        evidence = reference._reference_task_evidence
        seal = reference._evidence_seal
        evidence_is_complete = (
            len(evidence) == len(_TASK_KINDS)
            and {item.task for item in evidence} == set(_TASK_KINDS)
            and all(
                sum(item.task == task_kind for item in evidence) == 1
                for task_kind in _TASK_KINDS
            )
        )
        return (
            identity is not None
            and evidence_is_complete
            and isinstance(seal, str)
            and seal == _compute_evidence_seal(reference)
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _resolve_reference_tasks(
    reference: QualityReport | dict | None,
) -> tuple[
    dict[str, list[int] | None],
    dict[str, _ReferenceTaskEvidence],
]:
    ref_tokens: dict[str, list[int] | None] = dict.fromkeys(_TASK_KINDS)
    if reference is None:
        return ref_tokens, {}

    if isinstance(reference, QualityReport) and _has_consistent_private_evidence(
        reference
    ):
        private_by_task = _require_complete_reference_tasks(
            list(reference._reference_task_evidence)
        )
        for task_kind, evidence in private_by_task.items():
            ref_tokens[task_kind] = list(evidence.output_tokens)
        return ref_tokens, private_by_task

    if isinstance(reference, QualityReport):
        records = list(reference.tasks)
    elif isinstance(reference, Mapping):
        raw_records = reference.get("tasks")
        if not isinstance(raw_records, (list, tuple)):
            records = []
        else:
            records = list(raw_records)
    else:
        raise TypeError("quality_report reference must be a QualityReport or mapping")

    public_by_task = _require_complete_reference_tasks(records)
    for task_kind, record in public_by_task.items():
        if isinstance(record, Mapping):
            output_tokens = record.get("output_tokens")
        else:
            output_tokens = getattr(record, "output_tokens", None)
        if not isinstance(output_tokens, (list, tuple)):
            raise ValueError(
                f"quality_report reference task {task_kind!r} has invalid output_tokens"
            )
        ref_tokens[task_kind] = [int(token) for token in output_tokens]
    return ref_tokens, {}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def quality_report(
    engine: Any,
    *,
    # Token-level identifiers (required to evaluate mask/eos leakage + infill).
    mask_token_id: int,
    eos_token_id: int,
    # Chat task.
    chat_prompt_token_ids: list[int],
    chat_max_tokens: int = 24,
    # Infill task.
    infill_prefix_token_ids: list[int],
    infill_suffix_token_ids: list[int],
    infill_max_fill_tokens: int = 12,
    # Edit task: drives infill(prefix=text[:start], suffix=text[end:]).
    edit_prefix_token_ids: list[int],
    edit_suffix_token_ids: list[int],
    edit_max_fill_tokens: int = 12,
    # Shared decode params.
    decode_strategy: Any = _UNSET,
    confidence_threshold: Any = _UNSET,
    block_size: Any = _UNSET,
    steps: Any = _UNSET,
    uses_shifted_logits: Any = _UNSET,
    uses_qk_norm: Any = _UNSET,
    # Real tokenizer decode (the maintainer gate MUST pass this). When supplied,
    # ``output_text`` is ``decode(tokens)`` (real text), so the printable/garbage
    # and mask-leak-in-text dimensions of degeneration are meaningful and each
    # task record's ``text_is_real_decode`` is True. When None (the unit gate's
    # tiny seeded weights with no tokenizer), the structural decimal fallback is
    # used and ``text_is_real_decode`` is False (printable check is NOT real).
    decode: "Callable[[list[int]], str] | None" = None,
    # Reference run (exact-path outputs): when None, this IS the reference.
    reference: "QualityReport | dict | None" = None,
    # Preferred evidence source. Profile resolution and task kwargs are derived
    # from this complete model config through apply_policy_profile().
    model_config: Mapping[str, Any] | None = None,
    # Legacy provenance/runtime fields. They are accepted only through the
    # validated compatibility adapter below and cannot override derived evidence.
    profile_name: str | None = None,
    reference_profile_name: str | None = None,
    # Optional runtime kwargs let productized profile gates prove that the report
    # did not merely attach labels: profile-derived decode/cache knobs must reach
    # the public engine calls. Task identity fields (tokens, masks, max lengths)
    # are intentionally blocked by _merge_runtime_kwargs.
    generate_entropy_kwargs: Mapping[str, Any] | None = None,
    generate_infill_kwargs: Mapping[str, Any] | None = None,
) -> QualityReport:
    """Run the chat / infill / edit task set and return a :class:`QualityReport`.

    When ``reference`` is ``None``, every task's ``exact_match_rate`` is 1.0
    because the run is its own comparison target. Derived ``exact_reference``
    evidence is separate from this self-comparison metric.

    When ``reference`` is a prior :class:`QualityReport` (or its ``to_dict()``
    output), ``exact_match_rate`` is computed per-task against the reference's
    output tokens (token-position agreement).  This is how an approximate run
    (C2b) calls ``quality_report``:

        exact_run = quality_report(exact_engine, ...)
        approx_run = quality_report(approx_engine, reference=exact_run, ...)

    ``decode`` makes the text layer pluggable: pass a real tokenizer's
    ``decode`` (e.g. ``runtime.tokenizer.decode``) so ``printable_ratio`` /
    mask-leak-in-text / repetition run on REAL decoded text. Without it the
    decimal-id fallback is used and ``text_is_real_decode`` is False, signalling
    that the printable dimension is NOT meaningful.

    The ``rows_forwarded_per_correct_token`` metric is always present alongside
    ``task_success`` (they are fields of the same :class:`TaskQualityRecord`).
    """
    if decode is not None and not callable(decode):
        raise TypeError("quality_report decode must be callable or None")

    task_inputs = _snapshot_task_inputs(
        mask_token_id=mask_token_id,
        eos_token_id=eos_token_id,
        chat_prompt_token_ids=chat_prompt_token_ids,
        chat_max_tokens=chat_max_tokens,
        infill_prefix_token_ids=infill_prefix_token_ids,
        infill_suffix_token_ids=infill_suffix_token_ids,
        infill_max_fill_tokens=infill_max_fill_tokens,
        edit_prefix_token_ids=edit_prefix_token_ids,
        edit_suffix_token_ids=edit_suffix_token_ids,
        edit_max_fill_tokens=edit_max_fill_tokens,
    )
    mask_token_id = int(task_inputs["mask_token_id"])
    eos_token_id = int(task_inputs["eos_token_id"])
    chat_prompt_token_ids = list(task_inputs["chat"]["prompt_token_ids"])
    chat_max_tokens = int(task_inputs["chat"]["max_tokens"])
    infill_prefix_token_ids = list(
        task_inputs["infill"]["prefix_token_ids"]
    )
    infill_suffix_token_ids = list(
        task_inputs["infill"]["suffix_token_ids"]
    )
    infill_max_fill_tokens = int(task_inputs["infill"]["max_fill_tokens"])
    edit_prefix_token_ids = list(task_inputs["edit"]["prefix_token_ids"])
    edit_suffix_token_ids = list(task_inputs["edit"]["suffix_token_ids"])
    edit_max_fill_tokens = int(task_inputs["edit"]["max_fill_tokens"])

    explicit_legacy_base = _explicit_legacy_base_values(
        decode_strategy=decode_strategy,
        confidence_threshold=confidence_threshold,
        block_size=block_size,
        steps=steps,
        uses_shifted_logits=uses_shifted_logits,
        uses_qk_norm=uses_qk_norm,
    )
    legacy_runtime_override = bool(
        explicit_legacy_base
        or _has_evidence_bearing_runtime_kwargs(
            generate_entropy_kwargs,
            generate_infill_kwargs,
        )
    )
    engine_model_config = getattr(engine, "model_config", None)
    has_engine_model_config = isinstance(engine_model_config, Mapping)
    supplied_model_config = model_config is not None
    has_model_config_evidence = supplied_model_config or has_engine_model_config
    if has_model_config_evidence:
        source_model_config = dict(
            model_config if supplied_model_config else engine_model_config
        )
        requested_profile = _profile_name(source_model_config)
        if profile_name is not None and str(profile_name) != requested_profile:
            raise ValueError(
                "quality_report profile_name disagrees with the model_config "
                f"policy profile: {profile_name!r} != {requested_profile!r}"
            )
        run_specs = build_policy_run_specs(source_model_config)
        canonical_specs = _canonical_baseline_specs(source_model_config)
        if supplied_model_config and has_engine_model_config:
            _validate_model_config_evidence_agreement(
                engine_model_config,
                source_model_config,
            )
        if supplied_model_config:
            _validate_legacy_base_agreement(run_specs, explicit_legacy_base)
        elif explicit_legacy_base:
            run_specs = _apply_explicit_legacy_base(
                run_specs,
                explicit_legacy_base,
            )
    else:
        requested_profile = "baseline" if profile_name is None else str(profile_name)
        source_model_config = {_PROFILE_SELECTOR: requested_profile}
        run_specs = build_policy_run_specs(source_model_config)
        canonical_specs = _canonical_baseline_specs(source_model_config)
        legacy_base = {
            key: explicit_legacy_base.get(key, default)
            for key, default in _LEGACY_SHARED_DEFAULTS.items()
        }
        run_specs = _apply_legacy_base(run_specs, **legacy_base)

    has_complete_model_config = (
        has_model_config_evidence
        and _has_complete_canonical_model_config(source_model_config)
    )

    _validate_engine_graph_ir(engine)
    run_identity = _build_run_identity(
        engine,
        model_config=(
            engine_model_config if has_engine_model_config else source_model_config
        ),
        task_inputs=task_inputs,
    )

    run_specs, call_extras = _adapt_legacy_runtime_kwargs(
        run_specs,
        requested_profile=requested_profile,
        generate_entropy_kwargs=generate_entropy_kwargs,
        generate_infill_kwargs=generate_infill_kwargs,
        complete_model_config=supplied_model_config,
    )
    specs_by_task = {spec.task_kind: spec for spec in run_specs}
    product_profile_run = any(spec.applicable for spec in run_specs)
    ref_tokens, reference_evidence = _resolve_reference_tasks(reference)
    reference_kind = _reference_kind(reference)
    if product_profile_run and reference_kind != "exact_baseline":
        raise ValueError(
            "quality_report product profiles require a verified exact-baseline "
            "QualityReport reference before candidate execution"
        )
    if product_profile_run and (
        not isinstance(reference, QualityReport)
        or reference._run_identity != run_identity
    ):
        raise ValueError(
            "quality_report product profile exact-baseline reference run identity "
            "does not match the current model/package and task inputs"
        )

    text_is_real = decode is not None

    def _render(tokens: list[int]) -> str:
        return decode(tokens) if decode is not None else _tokens_to_text(tokens)

    derived_reference_profile = _reference_profile_name(
        profile_name=requested_profile,
        reference=reference,
    )
    if (
        reference_profile_name is not None
        and str(reference_profile_name) != derived_reference_profile
    ):
        raise ValueError(
            "quality_report reference_profile_name disagrees with the reference "
            f"report: {reference_profile_name!r} != {derived_reference_profile!r}"
        )

    records: list[TaskQualityRecord] = []

    # ------------------------------------------------------------------
    # 1. Chat task
    # ------------------------------------------------------------------
    chat_tokens, chat_tm, chat_committed = _run_chat(
        engine,
        prompt_token_ids=chat_prompt_token_ids,
        max_tokens=chat_max_tokens,
        mask_token_id=mask_token_id,
        eos_token_id=eos_token_id,
        decode_strategy=decode_strategy,
        confidence_threshold=confidence_threshold,
        block_size=block_size,
        steps=steps,
        uses_shifted_logits=uses_shifted_logits,
        uses_qk_norm=uses_qk_norm,
        generate_entropy_kwargs=_engine_call_kwargs(
            specs_by_task["chat"],
            call_extras["chat"],
        ),
    )
    chat_text = _render(chat_tokens)
    chat_degen = _degeneration(
        chat_tokens, chat_text,
        mask_token_id=mask_token_id,
        eos_token_id=eos_token_id,
    )
    chat_success, chat_detail = _chat_task_success(
        chat_tokens, chat_text, chat_degen,
        max_tokens=chat_max_tokens,
        mask_token_id=mask_token_id,
        eos_token_id=eos_token_id,
    )
    records.append(_make_task_record(
        task="chat",
        output_tokens=chat_tokens,
        output_text=chat_text,
        text_is_real_decode=text_is_real,
        reference_tokens=ref_tokens["chat"],
        timing_metrics=chat_tm,
        committed_tokens=chat_committed,
        mask_token_id=mask_token_id,
        eos_token_id=eos_token_id,
        task_success=chat_success,
        task_detail=chat_detail,
        run_spec=specs_by_task["chat"],
    ))

    # ------------------------------------------------------------------
    # 2. Infill task
    # ------------------------------------------------------------------
    if product_profile_run:
        records.append(_copy_reference_task_record(reference_evidence["infill"]))
    else:
        infill_fill, infill_full, infill_tm, infill_committed = _run_infill(
            engine,
            prefix_token_ids=infill_prefix_token_ids,
            suffix_token_ids=infill_suffix_token_ids,
            max_fill_tokens=infill_max_fill_tokens,
            mask_token_id=mask_token_id,
            eos_token_id=eos_token_id,
            decode_strategy=decode_strategy,
            confidence_threshold=confidence_threshold,
            block_size=block_size,
            steps=steps,
            uses_shifted_logits=uses_shifted_logits,
            uses_qk_norm=uses_qk_norm,
            generate_infill_kwargs=_engine_call_kwargs(
                specs_by_task["infill"],
                call_extras["infill"],
            ),
        )
        infill_text = _render(infill_fill)
        infill_success, infill_detail = _infill_task_success(
            infill_fill, infill_full,
            infill_prefix_token_ids, infill_suffix_token_ids,
            mask_token_id=mask_token_id,
            eos_token_id=eos_token_id,
        )
        records.append(_make_task_record(
            task="infill",
            output_tokens=infill_fill,
            output_text=infill_text,
            text_is_real_decode=text_is_real,
            reference_tokens=ref_tokens["infill"],
            timing_metrics=infill_tm,
            committed_tokens=infill_committed,
            mask_token_id=mask_token_id,
            eos_token_id=eos_token_id,
            task_success=infill_success,
            task_detail=infill_detail,
            run_spec=specs_by_task["infill"],
        ))

    # ------------------------------------------------------------------
    # 3. Edit task (infill with a different prefix/suffix)
    # ------------------------------------------------------------------
    if product_profile_run:
        records.append(_copy_reference_task_record(reference_evidence["edit"]))
    else:
        edit_fill, edit_full, edit_tm, edit_committed = _run_infill(
            engine,
            prefix_token_ids=edit_prefix_token_ids,
            suffix_token_ids=edit_suffix_token_ids,
            max_fill_tokens=edit_max_fill_tokens,
            mask_token_id=mask_token_id,
            eos_token_id=eos_token_id,
            decode_strategy=decode_strategy,
            confidence_threshold=confidence_threshold,
            block_size=block_size,
            steps=steps,
            uses_shifted_logits=uses_shifted_logits,
            uses_qk_norm=uses_qk_norm,
            generate_infill_kwargs=_engine_call_kwargs(
                specs_by_task["edit"],
                call_extras["edit"],
            ),
        )
        edit_text = _render(edit_fill)
        edit_success, edit_detail = _edit_task_success(
            edit_fill, edit_full,
            edit_prefix_token_ids, edit_suffix_token_ids,
            mask_token_id=mask_token_id,
            eos_token_id=eos_token_id,
        )
        records.append(_make_task_record(
            task="edit",
            output_tokens=edit_fill,
            output_text=edit_text,
            text_is_real_decode=text_is_real,
            reference_tokens=ref_tokens["edit"],
            timing_metrics=edit_tm,
            committed_tokens=edit_committed,
            mask_token_id=mask_token_id,
            eos_token_id=eos_token_id,
            task_success=edit_success,
            task_detail=edit_detail,
            run_spec=specs_by_task["edit"],
        ))

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    all_success = all(r.task_success for r in records)
    mean_emr = sum(r.exact_match_rate for r in records) / max(1, len(records))
    mean_degen = sum(r.degeneration["degeneration_score"] for r in records) / max(1, len(records))
    aggregate = {
        "profile_name": requested_profile,
        "profile_applied_tasks": [
            spec.task_kind for spec in run_specs if spec.applicable
        ],
        "reference_profile_name": derived_reference_profile,
        "reference_kind": reference_kind,
        "all_tasks_success": all_success,
        "mean_exact_match_rate": mean_emr,
        "mean_degeneration_score": mean_degen,
        "task_count": len(records),
    }

    report = QualityReport(
        tasks=records,
        aggregate=aggregate,
    )
    object.__setattr__(report, "_canonical_baseline_specs", tuple(canonical_specs))
    object.__setattr__(
        report,
        "_self_reference_kind",
        _classify_self_reference(
            requested_profile=requested_profile,
            run_specs=run_specs,
            canonical_specs=canonical_specs,
            legacy_runtime_override=legacy_runtime_override,
            has_complete_model_config=has_complete_model_config,
            compared_reference=reference is not None,
        ),
    )
    object.__setattr__(report, "_self_profile_name", requested_profile)
    object.__setattr__(report, "_run_identity", run_identity)
    object.__setattr__(
        report,
        "_reference_task_evidence",
        tuple(_snapshot_task_record(record) for record in records),
    )
    object.__setattr__(report, "_evidence_seal", _compute_evidence_seal(report))
    return report


def _reference_kind(reference: QualityReport | dict | None) -> str:
    if reference is None:
        return "self"
    if not _has_consistent_private_evidence(reference):
        return "unverified_reference"
    return reference._self_reference_kind


def _reference_profile_name(
    *,
    profile_name: str,
    reference: QualityReport | dict | None,
) -> str:
    if reference is None:
        return profile_name
    if _has_consistent_private_evidence(reference):
        return reference._self_profile_name
    return "unknown"


def _tokens_to_text(tokens: list[int]) -> str:
    """Structural fallback rendering of token ids as decimal strings — NOT real
    text.

    The harness itself does not own a tokenizer (that lives in RuntimeHandle).
    When no ``decode`` callable is supplied (the unit gate's tiny seeded weights),
    ids are rendered as their decimal string so the text fields are non-empty.

    IMPORTANT: this is a structural fallback, NOT real text. The printable /
    garbage degeneration check (``printable_ratio`` and the chat ``printable_ok``
    check) is NOT meaningful without a real tokenizer decode — decimal digits are
    always printable, so ``printable_ratio`` is vacuously ~1.0. A caller that
    needs the printable dimension to mean something MUST pass ``decode=`` to
    ``quality_report`` (then ``text_is_real_decode`` is True).
    """
    return " ".join(str(t) for t in tokens) if tokens else ""
