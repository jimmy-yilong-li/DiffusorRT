from __future__ import annotations

from dataclasses import dataclass
import math
import sys
from typing import Mapping


_DECODE_STRATEGIES = frozenset(
    {
        "entropy_count",
        "confidence_threshold",
        "block_threshold",
        "entropy_budget",
        "d2f_block_pipeline",
        "dilated_unmask",
        "bd3_fixed_quota",
        "fixed_quota",
    }
)
_DTYPES = frozenset({"fp32", "float32", "bf16", "bfloat16"})
_COMMIT_POLICIES = frozenset({"threshold", "entropy_budget", "dilated"})
_POLICY_PROFILES = frozenset(
    {
        "baseline",
        "active_window_exact",
        "robust_entropy_budget",
        "correction_diagnostics",
    }
)
_RECOVERY_DETECTORS = frozenset({"low_confidence", "repetition", "kl_drift"})
_ATTENTION_MASK_KINDS = frozenset({"bidirectional", "full", "d2f_hybrid"})
_LEGACY_ATTENTION_MASK_KINDS = _ATTENTION_MASK_KINDS | {"causal"}
_FORBIDDEN_MODEL_CONFIG_KEYS = (
    "mlx_diffusion_cross_session_cache",
    "mlx_diffusion_cross_session_cache_root",
)
# Every mlx_* model_config key the runtime understands. The resolver fails
# closed on an unknown mlx_* key: a typo (e.g. "mlx_diffusion_confidence_
# threshhold") must never silently decode at the defaults while the user
# believes it took effect. Non-mlx_ keys stay permissive — forward-compatible
# converter metadata is the package schema's domain. Drift-guarded against the
# package schema's mlx_* properties by tests/runtime/test_runtime_config.py.
_KNOWN_MLX_CONFIG_KEYS = frozenset(
    {
        "mlx_compute_dtype",
        "mlx_diffusion_alg_temp",
        "mlx_diffusion_block_kv_cache",
        "mlx_diffusion_block_kv_refresh_interval",
        "mlx_diffusion_block_size",
        "mlx_diffusion_canvas_cache",
        "mlx_diffusion_canvas_cache_budget_bytes",
        "mlx_diffusion_confidence_threshold",
        "mlx_diffusion_cross_request_cache",
        "mlx_diffusion_cross_session_cache",
        "mlx_diffusion_cross_session_cache_root",
        "mlx_diffusion_d2f_activate_threshold",
        "mlx_diffusion_d2f_add_threshold",
        "mlx_diffusion_d2f_commit_policy",
        "mlx_diffusion_d2f_exact_kv_cache",
        "mlx_diffusion_d2f_skip_committed_rows",
        "mlx_diffusion_d2f_skip_refresh_interval",
        "mlx_diffusion_decode_strategy",
        "mlx_diffusion_eps",
        "mlx_diffusion_error_fixing",
        "mlx_diffusion_gamma",
        "mlx_diffusion_policy_profile",
        "mlx_diffusion_prefix_kv_cache",
        "mlx_diffusion_prefix_kv_refresh_interval",
        "mlx_diffusion_recovery_confidence_floor",
        "mlx_diffusion_recovery_detectors",
        "mlx_diffusion_recovery_kl_threshold",
        "mlx_diffusion_recovery_max_rounds",
        "mlx_diffusion_recovery_repetition_max_ngram",
        "mlx_diffusion_remasking",
        "mlx_diffusion_snapshot_save_path",
        "mlx_diffusion_steps",
        "mlx_diffusion_temperature",
        "mlx_diffusion_top_p",
        "mlx_diffusion_trace_canvas",
        "mlx_diffusion_window_exec",
        "mlx_layer0_kv_cache_enabled",
        "mlx_linear_quant_bits",
        "mlx_linear_quant_group_size",
    }
)
_CORE_JSON_BOOLEAN_FIELDS = (
    "uses_qk_norm",
    "uses_shifted_logits",
    "bias_present",
)
_CORE_POSITIVE_INTEGER_FIELDS = (
    "layer_count",
    "hidden_size",
    "attention_heads",
    "vocab_size",
    "head_dim",
    "num_kv_heads",
    "intermediate_size",
    "block_size",
    "step_budget",
    "diffusion_steps",
)
_CORE_NONNEGATIVE_INTEGER_FIELDS = ("diffusion_cache_refresh_interval",)
_CORE_POSITIVE_NUMBER_FIELDS = (
    "diffusion_eps",
    "rms_norm_eps",
    "rope_theta",
)


def normalize_rope_scaling(value: object) -> dict[str, float | str]:
    """Return the one implemented RoPE scaling form or refuse it.

    ``rope_scaling`` is executable model semantics.  Unknown fields and values
    cannot be preserved as inert metadata because an older runtime would then
    silently run a future algorithm as today's linear scaling.
    """

    if value is None:
        return {"rope_type": "linear", "factor": 1.0}
    if not isinstance(value, Mapping):
        raise ValueError("rope_scaling must be a mapping or None")

    unknown_keys = set(value) - {"rope_type", "type", "factor"}
    if unknown_keys:
        raise ValueError(
            "rope_scaling contains unsupported fields: "
            + ", ".join(sorted(map(str, unknown_keys)))
        )

    modern_type = value.get("rope_type")
    legacy_type = value.get("type")
    if (
        modern_type is not None
        and legacy_type is not None
        and modern_type != legacy_type
    ):
        raise ValueError("rope_scaling type and rope_type must not disagree")
    rope_type = modern_type if modern_type is not None else legacy_type
    if rope_type is None:
        if "rope_type" in value or "type" in value:
            raise ValueError("rope_scaling type must be a non-empty string")
        rope_type = "linear"
    if not isinstance(rope_type, str) or not rope_type:
        raise ValueError("rope_scaling type must be a non-empty string")
    if rope_type not in {"linear", "default"}:
        raise ValueError(
            f"unsupported rope_scaling type {rope_type!r}; "
            "only linear/default are implemented"
        )

    factor = value.get("factor", 1.0)
    if (
        type(factor) not in (int, float)
        or not math.isfinite(factor)
        or (rope_type == "default" and float(factor) != 1.0)
        or (rope_type == "linear" and float(factor) < 1.0)
    ):
        if rope_type == "default":
            raise ValueError("rope_scaling factor must be 1 when type is default")
        raise ValueError("rope_scaling factor must be a finite number >= 1")
    return {
        "rope_type": "linear",
        "factor": 1.0 if rope_type == "default" else float(factor),
    }


def validate_generation_length(value: int, *, field_name: str) -> None:
    """Require an exact non-boolean integer generation length."""
    if type(value) is not int or value < 0:
        raise ValueError(
            f"{field_name} must be a non-boolean integer >= 0; got {value!r}"
        )


def validate_bd3_sampling_contract(
    *, temperature: object, top_p: object, remasking: object
) -> None:
    validate_fixed_quota_sampling_contract(
        temperature=temperature, top_p=top_p, remasking=remasking,
        strategy="bd3_fixed_quota",
    )


def validate_fixed_quota_sampling_contract(
    *, temperature: object, top_p: object, remasking: object,
    strategy: str = "fixed_quota",
) -> None:
    """Refuse sampling settings the fixed-quota greedy path cannot execute.

    This is the package and deployment contract: a package declares greedy
    decoding. A caller may still ask one request for a temperature; see
    ``validate_fixed_quota_request_sampling``.
    """
    if (
        type(temperature) not in (int, float)
        or temperature != 0.0
        or type(top_p) not in (int, float)
        or top_p != 1.0
        or remasking != "low_confidence"
    ):
        raise ValueError(
            f"{strategy} requires temperature=0.0, top_p=1.0 and "
            "remasking='low_confidence'"
        )


def _is_temperature(value: object) -> bool:
    if type(value) is int:  # exact: math.isfinite cannot take one past float's range
        return 0 <= value <= sys.float_info.max
    return type(value) is float and math.isfinite(value) and value >= 0.0


def validate_fixed_quota_request_sampling(
    *, temperature: object, top_p: object, strategy: str = "fixed_quota",
) -> None:
    """One request's sampling: the model cards' temperature, and no top-p.

    0 is the argmax; above 0 each position's token is a Gumbel draw at that
    temperature, ranked by its raw probability, as the cards' samplers do.
    """
    if not _is_temperature(temperature):
        raise ValueError(f"{strategy} requires a finite temperature >= 0; got {temperature!r}")
    if type(top_p) not in (int, float) or top_p != 1.0:
        raise ValueError(f"{strategy} samples without top_p; it must be 1.0")


def sampling_arguments(temperature: object, seed: object) -> dict:
    """The sampling keywords to hand on. None for a greedy, unseeded request, so
    such a call reaches every layer exactly as it did before sampling existed;
    anything else, a bad value included, goes on to be checked."""
    if type(temperature) in (int, float) and temperature == 0 and seed is None:
        return {}
    return {"temperature": temperature, "seed": seed}


def validate_request_sampling(*, temperature: object, seed: object) -> None:
    """A generation request's sampling options, checked before any backend runs.

    ``temperature`` 0 decodes greedily; above 0 the fixed-quota models draw each
    position's token at that temperature. ``seed`` keys the draws; ``None``
    lets the runtime draw one and report it.
    """
    if not _is_temperature(temperature):
        raise ValueError(f"temperature must be a finite number >= 0; got {temperature!r}")
    if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
        raise ValueError(f"seed must be None or an integer in [0, 2**64); got {seed!r}")


@dataclass(frozen=True, slots=True)
class ResolvedMLXRuntimeConfig:
    layer0_kv_cache_enabled: bool
    prefix_kv_cache: bool
    block_kv_cache: bool
    d2f_exact_kv_cache: bool
    d2f_skip_committed_rows: bool
    error_fixing: bool
    trace_canvas: bool
    canvas_cache: bool
    cross_request_cache: bool
    steps: int
    block_size: int
    prefix_kv_refresh_interval: int
    block_kv_refresh_interval: int
    d2f_skip_refresh_interval: int
    recovery_max_rounds: int
    recovery_repetition_max_ngram: int
    canvas_cache_budget_bytes: int | None
    linear_quant_bits: int | None
    linear_quant_group_size: int
    eps: float
    temperature: float
    top_p: float
    remasking: str
    alg_temp: float
    confidence_threshold: float
    d2f_add_threshold: float
    d2f_activate_threshold: float
    recovery_confidence_floor: float
    gamma: float
    recovery_kl_threshold: float
    compute_dtype: str
    decode_strategy: str
    d2f_commit_policy: str
    window_exec: bool | None
    window_exec_label: str
    policy_profile: str
    recovery_detectors: tuple[str, ...] | None
    snapshot_save_path: str | None
    attention_mask_kind: str
    requires_prompt_block_alignment: bool = False

    def error_fixing_kwargs(self) -> dict | None:
        if not self.error_fixing:
            return None
        config: dict = {
            "enabled": True,
            "max_rounds": self.recovery_max_rounds,
            "confidence_floor": self.recovery_confidence_floor,
            "repetition_max_ngram": self.recovery_repetition_max_ngram,
            "kl_threshold": self.recovery_kl_threshold,
        }
        if self.recovery_detectors is not None:
            config["detectors"] = list(self.recovery_detectors)
        return config


def validate_prompt_block_alignment(
    *, prompt_length: int, block_size: int, required: bool
) -> None:
    """Apply a package's prompt-alignment contract before weight access."""
    if type(required) is not bool:
        raise ValueError("requires_prompt_block_alignment must be a boolean")
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    if required and prompt_length % block_size:
        raise ValueError(
            "decode policy requires the templated prompt length to be an exact "
            "multiple of block_size: "
            f"prompt_length={prompt_length} block_size={block_size}. "
            "For arbitrary prompts, choose a compatible ragged decode policy "
            "(BD3 Local catalog: decode_policy = \"fixed_quota_ragged_v1\") "
            "and rebuild the package."
        )


def validate_core_model_config(model_config: Mapping[str, object]) -> None:
    """Validate shared model/graph fields without coercing JSON values."""
    for key in _CORE_JSON_BOOLEAN_FIELDS:
        if key in model_config and type(model_config[key]) is not bool:
            raise ValueError(
                f"{key} must be a JSON boolean; got {model_config[key]!r}"
            )

    for key in _CORE_POSITIVE_INTEGER_FIELDS:
        if key not in model_config:
            continue
        value = model_config[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{key} must be a positive integer; got {value!r}")
        if value < 1:
            raise ValueError(f"{key} must be >= 1; got {value!r}")

    for key in _CORE_NONNEGATIVE_INTEGER_FIELDS:
        if key not in model_config:
            continue
        value = model_config[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{key} must be a nonnegative integer; got {value!r}")
        if value < 0:
            raise ValueError(f"{key} must be >= 0; got {value!r}")

    for key in _CORE_POSITIVE_NUMBER_FIELDS:
        if key not in model_config:
            continue
        value = model_config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be a finite number > 0; got {value!r}")
        resolved = float(value)
        if not math.isfinite(resolved) or resolved <= 0.0:
            raise ValueError(f"{key} must be a finite number > 0; got {value!r}")

    if "rope_scaling" in model_config:
        normalize_rope_scaling(model_config["rope_scaling"])

    if "attention_heads" in model_config and "num_kv_heads" in model_config:
        attention_heads = model_config["attention_heads"]
        num_kv_heads = model_config["num_kv_heads"]
        if attention_heads % num_kv_heads != 0:
            raise ValueError("attention_heads must be divisible by num_kv_heads")


def _validate_retired_kv_package_config(config: Mapping[str, object]) -> None:
    """Reject non-neutral spellings of removed K/V implementations."""
    neutral = {
        "mlx_diffusion_d2f_skip_committed_rows": False,
        "mlx_diffusion_d2f_skip_refresh_interval": 4,
        # Check the user-facing owner request before its former supporting
        # canvas flag so a compound legacy config names the capability the
        # caller actually asked to enable.
        "mlx_diffusion_cross_request_cache": False,
        "mlx_diffusion_canvas_cache": False,
        "mlx_diffusion_canvas_cache_budget_bytes": None,
    }
    for key, expected in neutral.items():
        if key not in config:
            continue
        actual = config[key]
        # ``bool`` subclasses ``int`` and Python also treats ``4.0 == 4``.
        # A retired setting accepts its exact compatibility spelling, not a
        # merely equal value that would otherwise be accepted and ignored.
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(
                f"{key} is no longer supported in DiffusorRT 0.2; only the "
                "neutral compatibility "
                f"value {expected!r} is accepted"
            )


def resolve_mlx_runtime_config(
    model_config: Mapping[str, object],
    *,
    adapter_defaults: Mapping[str, object],
    allow_legacy_causal: bool = False,
) -> ResolvedMLXRuntimeConfig:
    """Resolve and validate the public MLX model configuration without side effects."""
    config = dict(model_config)
    validate_core_model_config(config)
    for key in _FORBIDDEN_MODEL_CONFIG_KEYS:
        if key in config:
            raise ValueError(
                f"{key} is caller runtime data and is not accepted in model_config"
            )
    unknown_mlx = sorted(
        key
        for key in config
        if isinstance(key, str)
        and key.startswith("mlx_")
        and key not in _KNOWN_MLX_CONFIG_KEYS
    )
    if unknown_mlx:
        import difflib

        parts = []
        for key in unknown_mlx:
            close = difflib.get_close_matches(
                key, sorted(_KNOWN_MLX_CONFIG_KEYS), n=1, cutoff=0.6
            )
            suggestion = f" (did you mean {close[0]!r}?)" if close else ""
            parts.append(f"{key!r}{suggestion}")
        raise ValueError(
            "unknown mlx_* model_config key(s): "
            + ", ".join(parts)
            + "; mlx_* keys fail closed against typos"
        )

    _validate_retired_kv_package_config(config)

    prefix_kv_cache = _resolve_bool(
        config, "mlx_diffusion_prefix_kv_cache", False
    )
    block_kv_cache = _resolve_bool(
        config, "mlx_diffusion_block_kv_cache", False
    )
    d2f_exact_kv_cache = _resolve_bool(
        config, "mlx_diffusion_d2f_exact_kv_cache", False
    )
    # K2-2 makes native core the only exact-D2F K/V owner.  The historical
    # layer-0 cache default is still correct for ordinary bidirectional
    # decodes, but enabling it implicitly beside the native exact owner left a
    # second live K/V store.  Default it off only in the exact regime; an
    # explicit true value is rejected below rather than silently ignored.
    layer0_kv_cache_enabled = _resolve_bool(
        config,
        "mlx_layer0_kv_cache_enabled",
        _adapter_default_or(
            adapter_defaults, "layer0_kv_cache_enabled", not d2f_exact_kv_cache
        ),
    )
    d2f_skip_committed_rows = _resolve_bool(
        config, "mlx_diffusion_d2f_skip_committed_rows", False
    )
    error_fixing = _resolve_bool(
        config,
        "mlx_diffusion_error_fixing",
        _adapter_default_or(adapter_defaults, "error_fixing", False),
    )
    trace_canvas = _resolve_bool(config, "mlx_diffusion_trace_canvas", False)
    canvas_cache = _resolve_bool(config, "mlx_diffusion_canvas_cache", False)
    cross_request_cache = _resolve_bool(
        config, "mlx_diffusion_cross_request_cache", False
    )

    steps = _resolve_int(
        config,
        "mlx_diffusion_steps",
        _adapter_default_or(adapter_defaults, "steps", 512),
        minimum=1,
    )
    block_size = _resolve_int(
        config,
        "mlx_diffusion_block_size",
        _adapter_default(adapter_defaults, "block_size", "mlx_diffusion_block_size"),
        minimum=1,
    )
    prefix_kv_refresh_interval = _resolve_int(
        config, "mlx_diffusion_prefix_kv_refresh_interval", 0, minimum=0
    )
    block_kv_refresh_interval = _resolve_int(
        config, "mlx_diffusion_block_kv_refresh_interval", 0, minimum=0
    )
    d2f_skip_refresh_interval = _resolve_int(
        config, "mlx_diffusion_d2f_skip_refresh_interval", 4, minimum=1
    )
    recovery_max_rounds = _resolve_int(
        config,
        "mlx_diffusion_recovery_max_rounds",
        _adapter_default_or(adapter_defaults, "recovery_max_rounds", 2),
        minimum=0,
    )
    recovery_repetition_max_ngram = _resolve_int(
        config,
        "mlx_diffusion_recovery_repetition_max_ngram",
        _adapter_default_or(
            adapter_defaults,
            "recovery_repetition_max_ngram",
            3,
        ),
        minimum=1,
    )
    canvas_cache_budget_bytes = _resolve_optional_int(
        config, "mlx_diffusion_canvas_cache_budget_bytes", minimum=0
    )
    linear_quant_bits = _resolve_optional_quant_bits(config)
    linear_quant_group_size = _resolve_int(
        config, "mlx_linear_quant_group_size", 64, minimum=1
    )

    eps = _resolve_float(
        config, "mlx_diffusion_eps", 1e-3, minimum=0.0, exclusive_minimum=True
    )
    temperature = _resolve_float(
        config,
        "mlx_diffusion_temperature",
        _adapter_default_or(adapter_defaults, "temperature", 0.2),
        minimum=0.0,
    )
    top_p = _resolve_float(
        config,
        "mlx_diffusion_top_p",
        _adapter_default_or(adapter_defaults, "top_p", 0.95),
        minimum=0.0,
        maximum=1.0,
        exclusive_minimum=True,
    )
    remasking = _resolve_enum(
        config,
        "mlx_diffusion_remasking",
        _adapter_default_or(adapter_defaults, "remasking", "low_confidence"),
        frozenset({"low_confidence"}),
    )
    alg_temp = _resolve_float(config, "mlx_diffusion_alg_temp", 0.0)
    if alg_temp != 0.0:
        raise ValueError(
            f"mlx_diffusion_alg_temp must be exactly 0.0; got {alg_temp!r}"
        )
    confidence_threshold = _resolve_float(
        config,
        "mlx_diffusion_confidence_threshold",
        _adapter_default_or(adapter_defaults, "confidence_threshold", 0.9),
        minimum=0.0,
        maximum=1.0,
    )
    d2f_add_threshold = _resolve_float(
        config,
        "mlx_diffusion_d2f_add_threshold",
        0.1,
        minimum=0.0,
        maximum=1.0,
    )
    d2f_activate_threshold = _resolve_float(
        config,
        "mlx_diffusion_d2f_activate_threshold",
        0.95,
        minimum=0.0,
        maximum=1.0,
    )
    recovery_confidence_floor = _resolve_float(
        config,
        "mlx_diffusion_recovery_confidence_floor",
        _adapter_default_or(adapter_defaults, "recovery_confidence_floor", 0.5),
        minimum=0.0,
        maximum=1.0,
    )
    gamma = _resolve_float(
        config,
        "mlx_diffusion_gamma",
        _adapter_default_or(adapter_defaults, "gamma", 0.1),
        minimum=0.0,
    )
    recovery_kl_threshold = _resolve_float(
        config,
        "mlx_diffusion_recovery_kl_threshold",
        _adapter_default_or(adapter_defaults, "recovery_kl_threshold", 1.0),
        minimum=0.0,
    )

    compute_dtype = _resolve_enum(config, "mlx_compute_dtype", "fp32", _DTYPES)
    decode_strategy = _resolve_enum(
        config,
        "mlx_diffusion_decode_strategy",
        _adapter_default(
            adapter_defaults,
            "decode_strategy",
            "mlx_diffusion_decode_strategy",
        ),
        _DECODE_STRATEGIES,
    )
    requires_prompt_block_alignment = _resolve_bool(
        config,
        "bd3_requires_prompt_block_alignment",
        decode_strategy == "bd3_fixed_quota",
    )
    d2f_commit_policy = _resolve_enum(
        config,
        "mlx_diffusion_d2f_commit_policy",
        "threshold",
        _COMMIT_POLICIES,
    )
    window_exec, window_exec_label = _resolve_window_exec(config)
    policy_profile = _resolve_enum(
        config,
        "mlx_diffusion_policy_profile",
        "baseline",
        _POLICY_PROFILES,
    )
    recovery_detectors = _resolve_detectors(
        config,
        _adapter_default_or(adapter_defaults, "recovery_detectors", None),
    )
    snapshot_save_path = _resolve_snapshot_path(config)
    attention_mask_kind = _resolve_enum(
        config,
        "attention_mask_kind",
        "bidirectional",
        _LEGACY_ATTENTION_MASK_KINDS
        if allow_legacy_causal
        else _ATTENTION_MASK_KINDS,
    )

    _validate_combinations(
        decode_strategy=decode_strategy,
        attention_mask_kind=attention_mask_kind,
        layer0_kv_cache=layer0_kv_cache_enabled,
        prefix_kv_cache=prefix_kv_cache,
        block_kv_cache=block_kv_cache,
        d2f_exact_kv_cache=d2f_exact_kv_cache,
        d2f_commit_policy=d2f_commit_policy,
        error_fixing=error_fixing,
        temperature=temperature,
        top_p=top_p,
        remasking=remasking,
    )

    return ResolvedMLXRuntimeConfig(
        layer0_kv_cache_enabled=layer0_kv_cache_enabled,
        prefix_kv_cache=prefix_kv_cache,
        block_kv_cache=block_kv_cache,
        d2f_exact_kv_cache=d2f_exact_kv_cache,
        d2f_skip_committed_rows=d2f_skip_committed_rows,
        error_fixing=error_fixing,
        trace_canvas=trace_canvas,
        canvas_cache=canvas_cache,
        cross_request_cache=cross_request_cache,
        steps=steps,
        block_size=block_size,
        prefix_kv_refresh_interval=prefix_kv_refresh_interval,
        block_kv_refresh_interval=block_kv_refresh_interval,
        d2f_skip_refresh_interval=d2f_skip_refresh_interval,
        recovery_max_rounds=recovery_max_rounds,
        recovery_repetition_max_ngram=recovery_repetition_max_ngram,
        canvas_cache_budget_bytes=canvas_cache_budget_bytes,
        linear_quant_bits=linear_quant_bits,
        linear_quant_group_size=linear_quant_group_size,
        eps=eps,
        temperature=temperature,
        top_p=top_p,
        remasking=remasking,
        alg_temp=alg_temp,
        confidence_threshold=confidence_threshold,
        d2f_add_threshold=d2f_add_threshold,
        d2f_activate_threshold=d2f_activate_threshold,
        recovery_confidence_floor=recovery_confidence_floor,
        gamma=gamma,
        recovery_kl_threshold=recovery_kl_threshold,
        compute_dtype=compute_dtype,
        decode_strategy=decode_strategy,
        d2f_commit_policy=d2f_commit_policy,
        window_exec=window_exec,
        window_exec_label=window_exec_label,
        policy_profile=policy_profile,
        recovery_detectors=recovery_detectors,
        snapshot_save_path=snapshot_save_path,
        attention_mask_kind=attention_mask_kind,
        requires_prompt_block_alignment=requires_prompt_block_alignment,
    )


def _adapter_default(
    adapter_defaults: Mapping[str, object],
    key: str,
    public_key: str,
) -> object:
    if key not in adapter_defaults:
        raise ValueError(f"{public_key} has no adapter default")
    return adapter_defaults[key]


def _adapter_default_or(
    adapter_defaults: Mapping[str, object],
    key: str,
    fallback: object,
) -> object:
    return adapter_defaults[key] if key in adapter_defaults else fallback


def _resolve_bool(config: Mapping[str, object], key: str, default: object) -> bool:
    value = config[key] if key in config else default
    if type(value) is not bool:
        raise ValueError(f"{key} must be a JSON boolean; got {value!r}")
    return value


def _resolve_int(
    config: Mapping[str, object],
    key: str,
    default: object,
    *,
    minimum: int,
) -> int:
    value = config[key] if key in config else default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer >= {minimum}; got {value!r}")
    if value < minimum:
        raise ValueError(f"{key} must be >= {minimum}; got {value!r}")
    return int(value)


def _resolve_optional_int(
    config: Mapping[str, object], key: str, *, minimum: int
) -> int | None:
    value = config[key] if key in config else None
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{key} must be null or an integer >= {minimum}; got {value!r}"
        )
    if value < minimum:
        raise ValueError(f"{key} must be >= {minimum}; got {value!r}")
    return int(value)


def _resolve_optional_quant_bits(config: Mapping[str, object]) -> int | None:
    key = "mlx_linear_quant_bits"
    value = config[key] if key in config else None
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value not in {4, 8}:
        raise ValueError(f"{key} must be null, 4, or 8; got {value!r}")
    return int(value)


def _resolve_float(
    config: Mapping[str, object],
    key: str,
    default: object,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_minimum: bool = False,
) -> float:
    value = config[key] if key in config else default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a finite number; got {value!r}")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{key} must be finite; got {value!r}")
    if minimum is not None and (
        resolved <= minimum if exclusive_minimum else resolved < minimum
    ):
        operator = ">" if exclusive_minimum else ">="
        raise ValueError(f"{key} must be {operator} {minimum}; got {value!r}")
    if maximum is not None and resolved > maximum:
        raise ValueError(f"{key} must be <= {maximum}; got {value!r}")
    return resolved


def _resolve_enum(
    config: Mapping[str, object],
    key: str,
    default: object,
    allowed: frozenset[str],
) -> str:
    value = config[key] if key in config else default
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{key} must be one of {sorted(allowed)}; got {value!r}")
    return value


def _resolve_window_exec(
    config: Mapping[str, object],
) -> tuple[bool | None, str]:
    key = "mlx_diffusion_window_exec"
    value = config[key] if key in config else None
    if value is None or value == "auto":
        return None, "auto"
    if type(value) is bool:
        return (value, "primary" if value else "oracle")
    if value == "primary":
        return True, "primary"
    if value == "oracle":
        return False, "oracle"
    raise ValueError(
        f"{key} must be null, a JSON boolean, 'auto', 'primary', or 'oracle'; "
        f"got {value!r}"
    )


def _resolve_detectors(
    config: Mapping[str, object],
    default: object = None,
) -> tuple[str, ...] | None:
    key = "mlx_diffusion_recovery_detectors"
    value = config[key] if key in config else default
    if value is None:
        return None
    if type(value) is not list:
        raise ValueError(f"{key} must be null or a JSON array; got {value!r}")
    if any(type(detector) is not str or detector not in _RECOVERY_DETECTORS for detector in value):
        raise ValueError(
            f"{key} entries must be one of {sorted(_RECOVERY_DETECTORS)}; got {value!r}"
        )
    return tuple(value)


def _resolve_snapshot_path(config: Mapping[str, object]) -> str | None:
    validate_package_snapshot_config(config)
    return None


def validate_package_snapshot_config(config: Mapping[str, object]) -> None:
    # K2-3 tombstones are compatibility verdicts, not merely schema shape.
    # Validate them here as well as in the full runtime resolver so package
    # loading preserves the precise field-level refusal instead of allowing a
    # deferred JSON-schema error to collapse it to ``compatibility_error``.
    _validate_retired_kv_package_config(config)
    key = "mlx_diffusion_snapshot_save_path"
    value = config[key] if key in config else None
    if value is None:
        return
    raise ValueError(
        f"{key} is no longer package-owned; pass caller-owned "
        "snapshot_save_path to RuntimeHandle.generate()"
    )


def validate_d2f_recovery_contract(
    *, decode_strategy: str, error_fixing_enabled: bool, exact_kv_enabled: bool,
) -> None:
    """Share the frozen-block recovery requirement with the execution boundary."""
    if decode_strategy == "d2f_block_pipeline" and error_fixing_enabled and not exact_kv_enabled:
        raise ValueError(
            "D2F error-fixing recovery requires mlx_diffusion_d2f_exact_kv_cache=true "
            "(the frozen-block-safe remask contract)"
        )


def _validate_combinations(
    *,
    decode_strategy: str,
    attention_mask_kind: str,
    layer0_kv_cache: bool,
    prefix_kv_cache: bool,
    block_kv_cache: bool,
    d2f_exact_kv_cache: bool,
    d2f_commit_policy: str,
    error_fixing: bool,
    temperature: float,
    top_p: float,
    remasking: str,
) -> None:
    if d2f_exact_kv_cache and layer0_kv_cache:
        raise ValueError(
            "mlx_layer0_kv_cache_enabled=true is incompatible with the native "
            "exact-D2F owner; the runtime refuses a second production K/V owner"
        )
    if decode_strategy == "bd3_fixed_quota":
        validate_bd3_sampling_contract(
            temperature=temperature, top_p=top_p, remasking=remasking
        )
        if attention_mask_kind != "d2f_hybrid":
            raise ValueError(
                "bd3_fixed_quota requires attention_mask_kind='d2f_hybrid'"
            )
        if layer0_kv_cache or prefix_kv_cache or block_kv_cache:
            raise ValueError(
                "bd3_fixed_quota Candidate requires every legacy MLX cache off"
            )
        if d2f_exact_kv_cache:
            raise ValueError(
                "bd3_fixed_quota does not yet admit the native exact-D2F K/V "
                "owner; ragged absolute first-block support is required"
            )
        if error_fixing:
            raise ValueError(
                "bd3_fixed_quota does not yet support error-fixing recovery"
            )
    if decode_strategy == "fixed_quota":
        validate_fixed_quota_sampling_contract(
            temperature=temperature, top_p=top_p, remasking=remasking
        )
        if attention_mask_kind not in {"bidirectional", "full"}:
            raise ValueError("fixed_quota requires bidirectional attention")
        if layer0_kv_cache or prefix_kv_cache or block_kv_cache or d2f_exact_kv_cache or error_fixing:
            raise ValueError("fixed_quota requires caches and error-fixing recovery off")
    if decode_strategy == "d2f_block_pipeline":
        if attention_mask_kind != "d2f_hybrid":
            raise ValueError(
                "attention_mask_kind must be 'd2f_hybrid' when "
                "mlx_diffusion_decode_strategy='d2f_block_pipeline'"
            )
        if prefix_kv_cache:
            raise ValueError(
                "mlx_diffusion_prefix_kv_cache is incompatible with "
                "mlx_diffusion_decode_strategy='d2f_block_pipeline'"
            )
        if block_kv_cache:
            raise ValueError(
                "mlx_diffusion_block_kv_cache is incompatible with "
                "mlx_diffusion_decode_strategy='d2f_block_pipeline'"
            )

    if d2f_exact_kv_cache and attention_mask_kind != "d2f_hybrid":
        raise ValueError(
            "attention_mask_kind must be 'd2f_hybrid' when "
            "mlx_diffusion_d2f_exact_kv_cache=true"
        )
    if d2f_exact_kv_cache and decode_strategy not in {
        "block_threshold",
        "d2f_block_pipeline",
    }:
        raise ValueError(
            "mlx_diffusion_d2f_exact_kv_cache=true requires a decode strategy "
            "with an in-order block lifecycle: 'block_threshold' (D2F-2) or "
            "'d2f_block_pipeline' (D2F-3); got "
            f"{decode_strategy!r}"
        )
    if d2f_commit_policy != "threshold" and decode_strategy != "d2f_block_pipeline":
        raise ValueError(
            f"mlx_diffusion_d2f_commit_policy={d2f_commit_policy!r} requires "
            "mlx_diffusion_decode_strategy='d2f_block_pipeline'"
        )
    validate_d2f_recovery_contract(
        decode_strategy=decode_strategy,
        error_fixing_enabled=error_fixing,
        exact_kv_enabled=d2f_exact_kv_cache,
    )
    if d2f_exact_kv_cache and prefix_kv_cache:
        raise ValueError(
            "mlx_diffusion_prefix_kv_cache is mutually exclusive with "
            "mlx_diffusion_d2f_exact_kv_cache"
        )
    if d2f_exact_kv_cache and block_kv_cache:
        raise ValueError(
            "mlx_diffusion_block_kv_cache is mutually exclusive with "
            "mlx_diffusion_d2f_exact_kv_cache"
        )

    # The converse of the first rule in this function. The hybrid mask consumes
    # per-forward geometry (prompt_len / block_size), and there are exactly TWO
    # suppliers of it:
    #
    #   D2F-3  decode_strategy='d2f_block_pipeline' -- mlx_dream.py builds
    #          mask_geometry directly from the prompt length and block size.
    #   D2F-2  block_threshold + d2f_exact_kv_cache=true -- the exact cache OWNS
    #          the geometry, mlx_dream.py derives it when the caller omits it,
    #          and cache_registry.py uses the sequential single-block window.
    #          The lifecycle guard above rejects every other non-pipeline
    #          strategy because it cannot freeze completed blocks in order.
    #
    # With neither, mask_geometry stays None and validate_attention_kind refuses
    # the forward -- a config error that arrived AFTER the weights were loaded
    # and read as a model failure. That, and only that, is what this rejects.
    #
    # An earlier version of this rule demanded the pipeline outright and made
    # D2F-2 unreachable from the public config surface while the runtime still
    # implemented it. A guard wider than the defect it closes is its own defect.
    #
    # Checked last, so a config that is also wrong for another reason still
    # reports that reason first and existing diagnostics keep their priority.
    if (
        attention_mask_kind == "d2f_hybrid"
        and decode_strategy not in {"d2f_block_pipeline", "bd3_fixed_quota"}
        and not d2f_exact_kv_cache
    ):
        raise ValueError(
            "attention_mask_kind='d2f_hybrid' requires a source of per-forward "
            "mask geometry: either mlx_diffusion_decode_strategy="
            "'d2f_block_pipeline' (D2F-3) or mlx_diffusion_d2f_exact_kv_cache="
            f"true (D2F-2). Got decode_strategy={decode_strategy!r} with "
            "d2f_exact_kv_cache=false, which fails at the first forward"
        )
