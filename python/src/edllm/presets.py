"""Named, validated inference presets — opt-in `model_config` overlays.

A preset is a partial `model_config` you MERGE into a package's config (defaults are
never changed by importing this module). Each preset is a config that has been
measured + quality-validated; the docstring states the number and the caveat.

Usage while constructing a package or deployment profile:

    from edllm.presets import PRESETS
    package_config.update(PRESETS["throughput-d2f-bs8-tau0.6"])

Product runtimes resolve an immutable executable plan at load. Do not mutate a
loaded runtime; select a deployment profile or create a new Engine instead.

Debug metrics the D2F runtime surfaces (via `return_debug=True` -> `result.debug.metrics`):
`mlx_forward_count` (NFE), `mlx_committed_per_forward`, `mlx_s_over_t`,
`mlx_active_tokens_per_forward`, `mlx_output_token_count`.
"""
from __future__ import annotations

import copy
from typing import Any, Mapping

# Exact-only active-window product preset. This is the stable baseline surface:
# D2F block pipeline + block-causal mask + exact prompt/frozen-block KV reuse,
# routed through the primary execute_window path in auto mode. It deliberately
# does NOT inherit the throughput preset because that preset opts into int8.
ACTIVE_WINDOW_EXACT_D2F_BS8: dict[str, Any] = {
    "mlx_diffusion_decode_strategy": "d2f_block_pipeline",
    "attention_mask_kind": "d2f_hybrid",
    "mlx_diffusion_d2f_exact_kv_cache": True,
    # Explicit even though runtime_config defaults this off in the exact
    # regime: the preset documents that native core is the only K/V owner.
    "mlx_layer0_kv_cache_enabled": False,
    "mlx_diffusion_window_exec": "auto",
    "mlx_diffusion_block_size": 8,
    "mlx_diffusion_confidence_threshold": 0.8,
    "mlx_diffusion_d2f_add_threshold": 0.1,
    "mlx_diffusion_d2f_activate_threshold": 0.95,
}

# The strongest VALIDATED single-stream throughput config (2026-06-13): the D2F block-pipeline
# runtime at block_size=8 + confidence_threshold(tau_conf)=0.6, with int8 quantized linear.
# Measured **39.72 tok/s on Dream-7B (AR gap ~2.0x; AR Qwen2.5-7B 4-bit = 79.9 single-stream)**,
# EM 8/9 on the AR prompts; quality DE-RISKED on a 36-prompt x 6-category eval where it was
# IDENTICAL to the conservative anchor (zero real repetition). `results/2026-06-12-{pareto-
# calibration,quality-eval,frontier-headroom-probe,commit5-eb-vs-threshold}.md`.
#
# CAVEAT: opt-in, NOT a default. Validated on Dream-7B + the AR prompt set + bounded categories;
# block_size=8 is aggressive — a manifest-default promotion should add a larger held-out + an
# LLM/human coherence judge. int8 holds top-1 parity on Dream-7B (Quant-1) but not every family.
THROUGHPUT_D2F_BS8_TAU06: dict[str, Any] = {
    "mlx_diffusion_decode_strategy": "d2f_block_pipeline",
    "attention_mask_kind": "d2f_hybrid",
    "mlx_diffusion_d2f_exact_kv_cache": True,
    "mlx_diffusion_block_size": 8,
    "mlx_diffusion_confidence_threshold": 0.6,   # tau_conf — the aggressive-commit frontier
    "mlx_diffusion_d2f_add_threshold": 0.1,       # tau_add
    "mlx_diffusion_d2f_activate_threshold": 0.95,  # tau_act
    "mlx_linear_quant_bits": 8,                    # parity-safe int8 (top-1 on Dream-7B)
    "mlx_linear_quant_group_size": 64,
}

# A more conservative, also-validated point (bigger block, less aggressive commit): ~30.4 tok/s,
# same-quality, for callers who want extra margin on harder/longer generations.
THROUGHPUT_D2F_BS8_TAU08: dict[str, Any] = {
    **THROUGHPUT_D2F_BS8_TAU06,
    "mlx_diffusion_confidence_threshold": 0.8,
}

# The robustness-oriented commit policy (Commit 5): same window, entropy-budget commit — a
# SAFER set per forward (its value shows on harder inputs, not raw speed on easy ones).
ROBUST_D2F_BS8_EB: dict[str, Any] = {
    **THROUGHPUT_D2F_BS8_TAU08,
    "mlx_diffusion_d2f_commit_policy": "entropy_budget",
    "mlx_diffusion_gamma": 0.4,
}

PRESETS: dict[str, dict[str, Any]] = {
    "active-window-exact-d2f-bs8": ACTIVE_WINDOW_EXACT_D2F_BS8,
    "throughput-d2f-bs8-tau0.6": THROUGHPUT_D2F_BS8_TAU06,
    "throughput-d2f-bs8-tau0.8": THROUGHPUT_D2F_BS8_TAU08,
    "robust-d2f-bs8-eb": ROBUST_D2F_BS8_EB,
}

POLICY_PROFILES: dict[str, dict[str, Any]] = {
    "baseline": {},
    "active_window_exact": ACTIVE_WINDOW_EXACT_D2F_BS8,
    "robust_entropy_budget": {
        **ACTIVE_WINDOW_EXACT_D2F_BS8,
        "mlx_diffusion_d2f_commit_policy": "entropy_budget",
        "mlx_diffusion_gamma": 0.4,
    },
    "correction_diagnostics": {
        **ACTIVE_WINDOW_EXACT_D2F_BS8,
        "mlx_diffusion_error_fixing": True,
        "mlx_diffusion_recovery_detectors": ["low_confidence", "repetition"],
        "mlx_diffusion_recovery_max_rounds": 2,
    },
}

POLICY_PROFILE_TASKS: dict[str, frozenset[str]] = {
    "baseline": frozenset({"chat", "infill", "edit"}),
    "active_window_exact": frozenset({"chat"}),
    "robust_entropy_budget": frozenset({"chat"}),
    "correction_diagnostics": frozenset({"chat"}),
}
_POLICY_TASKS = frozenset({"chat", "infill", "edit"})
if set(POLICY_PROFILE_TASKS) != set(POLICY_PROFILES):
    raise RuntimeError("policy profile task declarations are incomplete")


def policy_profile_supports_task(profile_name: str, task: str) -> bool:
    """Return whether a declared policy profile supports a public task.

    Unknown profiles and task names fail closed rather than becoming an implicit
    fallback. The mapping is deliberately separate from the profile overlays:
    a caller override cannot make a chat-only D2F graph safe for infill/edit.
    """
    if profile_name not in POLICY_PROFILES:
        raise ValueError(
            f"unknown policy profile {profile_name!r}; "
            f"available: {sorted(POLICY_PROFILES)}"
        )
    if task not in _POLICY_TASKS:
        raise ValueError(
            f"unknown policy profile task {task!r}; available: {sorted(_POLICY_TASKS)}"
        )
    return task in POLICY_PROFILE_TASKS[profile_name]


def validate_policy_profile_task(profile_name: str, task: str) -> None:
    """Fail early when a profile cannot execute the requested public task."""
    if policy_profile_supports_task(profile_name, task):
        return
    raise ValueError(
        "policy_profile_task_unsupported: "
        f"profile={profile_name!r} task={task!r}; "
        f"supported={sorted(POLICY_PROFILE_TASKS[profile_name])}"
    )


def public_operation_support(
    *, backend: str, decode_strategy: str, policy_profile: str = "baseline",
    attention_mask_kind: str = "bidirectional",
) -> dict[str, bool]:
    """Describe declared operations, not device readiness or answer quality."""
    span_strategies = {
        "entropy_count", "confidence_threshold", "block_threshold", "entropy_budget", "fixed_quota",
    }
    span_backend = backend == "mlx_local" or (
        backend == "cuda_native" and decode_strategy == "block_threshold"
        and attention_mask_kind in {"bidirectional", "full"}
    )
    span_supported = span_backend and decode_strategy in span_strategies
    return {
        "generate": True,
        "chat": policy_profile_supports_task(policy_profile, "chat"),
        "infill": span_supported and policy_profile_supports_task(policy_profile, "infill"),
        "edit": span_supported and policy_profile_supports_task(policy_profile, "edit"),
    }


def apply_preset(model_config: Mapping[str, Any], name: str) -> dict[str, Any]:
    """Return a NEW model_config = ``model_config`` overlaid with preset ``name``.

    Fail-closed on an unknown name and does not mutate the input. Apply the
    result before loading; a non-fixture runtime plan is immutable.
    """
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; available: {sorted(PRESETS)}")
    return {**dict(model_config), **PRESETS[name]}


def apply_policy_profile(model_config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a NEW model_config with an optional policy profile overlay applied.

    Unlike ``apply_preset()``, policy profiles are defaults: explicit keys in the
    caller's config win over profile values. Missing profile and ``"baseline"``
    are behavior-neutral no-ops. Unknown profiles fail closed.
    """
    config = dict(model_config)
    profile_name = config.get("mlx_diffusion_policy_profile")
    if profile_name is None:
        return config
    profile_name = str(profile_name)
    if profile_name not in POLICY_PROFILES:
        raise ValueError(
            f"unknown policy profile {profile_name!r}; available: {sorted(POLICY_PROFILES)}"
        )
    return {**copy.deepcopy(POLICY_PROFILES[profile_name]), **copy.deepcopy(config)}
