"""Reusable model conformance suite for the dLLM engine.

The dLLM engine is model-general: Dream is adapter #1, Qwen3-0.6B-diffusion-mdlm
is next, and LLaDA-8B is on the roadmap. Every masked-diffusion model adapter
must satisfy the SAME contract to "work" on the engine. This module defines that
contract as a single reusable check, `run_dllm_conformance`, with Dream as the
reference that passes.

Two tiers of checks:

- **structural** (always run): cheap, no 14GB weights required. They validate
  that a package loads, the tokenizer round-trips, a mask token is resolvable,
  and the model_config exposes the geometry keys the engine indexes. These run
  on a fixture/stub package by default.

- **generation** (opt-in, ``run_generation=True``): runs a real ``mlx_local``
  generation and validates the output text quality and the debug/provenance
  metrics. Requires real merged weights and `mlx`.

Design rules:

- No top-level ``mlx`` import. The ``mlx_local`` backend is imported lazily by
  the engine only when a generation check runs.
- Never raise out of ``run_dllm_conformance``: every check captures its own
  failure as ``ConformanceCheck(passed=False, detail=...)`` so a single broken
  package yields a report, not a traceback.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from .model import load_runtime
from .runtime_config import validate_fixed_quota_sampling_contract
from .validation import _passes_text_quality_gate

# Geometry required by the loaded runtime contract. This remains a local literal
# so conformance stays decoupled from the MLX backend and never imports it.
# ``vocab_size`` is included because the decode head and mask resolution depend
# on it. ``load_runtime`` enriches Dream/Qwen-like configs with derived geometry
# before this check runs.
_REQUIRED_GEOMETRY_KEYS: tuple[str, ...] = (
    "hidden_size",
    "attention_heads",
    "head_dim",
    "num_kv_heads",
    "rms_norm_eps",
    "rope_theta",
    "vocab_size",
)

# The provenance keys the generation debug bundle must expose so a `--json` dump
# is reproducible from its own output.
_PROVENANCE_KEYS: tuple[str, ...] = (
    "mlx_diffusion_decode_strategy",
    "mlx_diffusion_block_size",
    "mlx_diffusion_confidence_threshold",
    "mlx_diffusion_gamma",
)

_ROUND_TRIP_PROBE = "draft response for the model"
_GENERATION_PROMPT = "Explain why the sky is blue, briefly."


@dataclass(frozen=True, slots=True)
class ConformanceCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class ConformanceReport:
    package_dir: str
    checks: tuple[ConformanceCheck, ...]

    @property
    def passed(self) -> bool:
        """True iff there is at least one check and EVERY check passed.

        An empty check tuple is NOT passed: conformance must mean evidence, so a
        report with zero checks (e.g. a silently-skipped suite) must not look
        conformant.
        """
        return bool(self.checks) and all(check.passed for check in self.checks)


def run_dllm_conformance(
    package_dir: str | Path,
    *,
    run_generation: bool = False,
    trust_remote_code: bool = False,
) -> ConformanceReport:
    """Run the dLLM model conformance suite against ``package_dir``.

    Always runs the structural checks. When ``run_generation`` is True, also runs
    the (slow, real-weights) generation checks via the ``mlx_local`` backend.
    Never raises: each check captures its own failure as ``passed=False``.
    ``trust_remote_code`` threads to ``load_runtime`` (C2): pass True for a
    trusted, locally-provisioned package whose tokenizer needs custom code.
    """
    package_dir = str(package_dir)
    checks: list[ConformanceCheck] = []

    runtime, load_check = _check_load_runtime(
        package_dir, trust_remote_code=trust_remote_code
    )
    checks.append(load_check)

    if runtime is None:
        # Cannot evaluate any structural/generation checks without a runtime;
        # the load failure detail already explains why. Return early so the
        # report is honest about what was (not) evaluated.
        return ConformanceReport(package_dir=package_dir, checks=tuple(checks))

    try:
        checks.append(_check_tokenizer_round_trip(runtime))
        checks.append(_check_mask_token_id_resolvable(runtime))
        checks.append(_check_model_config_geometry_keys(runtime))

        descriptor = None
        descriptor_key = str(getattr(runtime, "model_descriptor_key", "") or "")
        if descriptor_key:
            from .model_coverage.producer_descriptors import descriptor_for_key

            try:
                descriptor = descriptor_for_key(descriptor_key)
            except ValueError as exc:
                checks.append(
                    ConformanceCheck(
                        name="model_descriptor_identity",
                        passed=False,
                        detail=str(exc),
                    )
                )
            else:
                checks.append(_check_model_descriptor_identity(runtime, descriptor))

        if run_generation:
            checks.extend(_run_generation_checks(runtime, descriptor=descriptor))
    finally:
        try:
            runtime.close()
        except Exception as exc:  # noqa: BLE001 - conformance reports failures
            checks.append(
                ConformanceCheck(
                    name="runtime_close",
                    passed=False,
                    detail=f"runtime close failed: {type(exc).__name__}: {exc}",
                )
            )

    return ConformanceReport(package_dir=package_dir, checks=tuple(checks))


# --------------------------------------------------------------------------- #
# Structural checks
# --------------------------------------------------------------------------- #


def _check_load_runtime(package_dir: str, *, trust_remote_code: bool = False):
    name = "load_runtime"
    try:
        runtime = load_runtime(package_dir, trust_remote_code=trust_remote_code)
    except Exception as exc:  # noqa: BLE001 - capture, never raise
        return None, ConformanceCheck(
            name=name,
            passed=False,
            detail=f"load_runtime failed: {type(exc).__name__}: {exc}",
        )
    if runtime.tokenizer is None:
        try:
            runtime.close()
        except Exception as exc:  # noqa: BLE001 - conformance reports failures
            detail = (
                "load_runtime succeeded without a tokenizer and runtime close "
                f"failed: {type(exc).__name__}: {exc}"
            )
        else:
            detail = "load_runtime succeeded but runtime.tokenizer is None"
        return None, ConformanceCheck(
            name=name,
            passed=False,
            detail=detail,
        )
    return runtime, ConformanceCheck(
        name=name,
        passed=True,
        detail=(
            f"loaded package v{runtime.package_version.major}."
            f"{runtime.package_version.minor}, tokenizer.kind={runtime.tokenizer.kind}"
        ),
    )


def _check_tokenizer_round_trip(runtime) -> ConformanceCheck:
    name = "tokenizer_round_trip"
    try:
        ids = runtime.tokenizer.encode(_ROUND_TRIP_PROBE)
        if not ids or not all(isinstance(token_id, int) for token_id in ids):
            return ConformanceCheck(
                name=name,
                passed=False,
                detail=f"encode did not return non-empty int ids: {ids!r}",
            )
        text = runtime.tokenizer.decode(ids)
        if not isinstance(text, str) or "draft" not in text:
            return ConformanceCheck(
                name=name,
                passed=False,
                detail=f"decode did not round-trip a known token: {text!r}",
            )
    except Exception as exc:  # noqa: BLE001
        return ConformanceCheck(
            name=name,
            passed=False,
            detail=f"round-trip raised: {type(exc).__name__}: {exc}",
        )
    return ConformanceCheck(
        name=name,
        passed=True,
        detail=f"encode->decode round-trips {_ROUND_TRIP_PROBE!r} ({len(ids)} tokens)",
    )


def _check_mask_token_id_resolvable(runtime) -> ConformanceCheck:
    name = "mask_token_id_resolvable"
    try:
        from .control_tokens import valid_token_id

        model_config = runtime.model_config or {}
        config_mask = valid_token_id(model_config.get("mask_token_id"))
        if config_mask is not None:
            return ConformanceCheck(
                name=name,
                passed=True,
                detail=f"resolved from model_config['mask_token_id']={config_mask}",
            )
        # require_control_token, not token_to_id: the plain lookup returns the
        # UNK id for an absent token, so a package with no mask token was
        # reported as CONFORMANT with mask_token_id = UNK. Measured: a
        # tokenizer answering UNK=3 gave passed=True.
        tokenizer_mask = runtime.tokenizer.require_control_token("<|mask|>")
        if valid_token_id(tokenizer_mask) is not None:
            return ConformanceCheck(
                name=name,
                passed=True,
                detail=f"resolved from tokenizer '<|mask|>'={tokenizer_mask}",
            )
    except Exception as exc:  # noqa: BLE001
        return ConformanceCheck(
            name=name,
            passed=False,
            detail=f"mask resolution raised: {type(exc).__name__}: {exc}",
        )
    return ConformanceCheck(
        name=name,
        passed=False,
        detail="no mask_token_id in model_config and tokenizer has no '<|mask|>'",
    )


def _check_model_config_geometry_keys(runtime) -> ConformanceCheck:
    name = "model_config_geometry_keys"
    model_config = runtime.model_config or {}
    missing = [
        key
        for key in _REQUIRED_GEOMETRY_KEYS
        if key not in model_config
        or model_config[key] is None
        or model_config[key] == ""
    ]
    if missing:
        return ConformanceCheck(
            name=name,
            passed=False,
            detail=f"model_config missing geometry keys: {missing}",
        )
    return ConformanceCheck(
        name=name,
        passed=True,
        detail="model_config exposes " + ", ".join(_REQUIRED_GEOMETRY_KEYS),
    )


def _check_model_descriptor_identity(runtime, descriptor) -> ConformanceCheck:
    name = "model_descriptor_identity"
    actual = {
        "model_id": getattr(runtime, "source_checkpoint_reference", ""),
        "revision": getattr(runtime, "source_checkpoint_sha", ""),
        "descriptor_digest": getattr(runtime, "model_descriptor_digest", ""),
        "backend": getattr(runtime, "admitted_backend", ""),
        "precision": getattr(runtime, "model_precision", ""),
    }
    expected = {
        "model_id": descriptor.model_id,
        "revision": descriptor.revision,
        "descriptor_digest": descriptor.descriptor_digest,
    }
    problems = [
        f"{field} expected={value!r} actual={actual[field]!r}"
        for field, value in expected.items()
        if actual[field] != value
    ]
    try:
        descriptor.resolve_backend(actual["backend"], actual["precision"])
    except ValueError as exc:
        problems.append(str(exc))
    if problems:
        return ConformanceCheck(name=name, passed=False, detail="; ".join(problems))
    return ConformanceCheck(
        name=name,
        passed=True,
        detail=(
            f"{descriptor.key}@{descriptor.revision} "
            f"{actual['backend']}/{actual['precision']}"
        ),
    )


# --------------------------------------------------------------------------- #
# Generation checks (opt-in)
# --------------------------------------------------------------------------- #


def _run_generation_checks(runtime, *, descriptor=None) -> list[ConformanceCheck]:
    """Run one mlx_local generation and derive every generation check from it.

    A single generation is shared across the checks so the slow forward runs
    once. If generation itself raises, that is captured as the
    `generation_runs` failure and the dependent checks report that they could
    not be evaluated.
    """
    capability = descriptor.capability_case if descriptor is not None else None
    result, run_check = _generate_for_conformance(runtime, capability=capability)
    if result is None:
        return [
            run_check,
            ConformanceCheck(
                name="generation_text_quality",
                passed=False,
                detail="not evaluated: generation did not produce a result",
            ),
            ConformanceCheck(
                name="generation_debug_metrics",
                passed=False,
                detail="not evaluated: generation did not produce a result",
            ),
            ConformanceCheck(
                name="generation_provenance_metrics",
                passed=False,
                detail="not evaluated: generation did not produce a result",
            ),
            ConformanceCheck(
                name="generation_baseline_status_honest",
                passed=False,
                detail="not evaluated: generation did not produce a result",
            ),
            ConformanceCheck(
                name="generation_capability_case",
                passed=False,
                detail=(
                    "not evaluated: ordinary generation did not produce a result"
                ),
            ),
        ]

    metrics = result.debug.metrics if result.debug is not None else {}
    ordinary_checks = [
        _check_generation_text_quality(result),
        _check_generation_debug_metrics(metrics),
        _check_generation_provenance_metrics(metrics),
        _check_generation_baseline_status(metrics),
    ]
    failed_ordinary_checks = [
        check.name for check in ordinary_checks if not check.passed
    ]
    if failed_ordinary_checks:
        capability_check = ConformanceCheck(
            name="generation_capability_case",
            passed=False,
            detail=(
                "not evaluated: ordinary generation checks failed: "
                + ", ".join(failed_ordinary_checks)
            ),
        )
    else:
        capability_check = _run_capability_case(
            runtime,
            generation_result=result,
            capability=capability,
        )
    return [
        run_check,
        *ordinary_checks,
        capability_check,
    ]


def _generate_for_conformance(runtime, *, capability=None):
    name = "generation_runs"
    prompt = capability.prompt if capability is not None else _GENERATION_PROMPT
    max_tokens = capability.max_tokens if capability is not None else 40
    try:
        result = runtime.generate(
            prompt,
            backend="mlx_local",
            max_tokens=max_tokens,
            return_debug=True,
        )
    except Exception as exc:  # noqa: BLE001
        return None, ConformanceCheck(
            name=name,
            passed=False,
            detail=f"generate raised: {type(exc).__name__}: {exc}",
        )
    if result.debug is None:
        return None, ConformanceCheck(
            name=name,
            passed=False,
            detail="generate returned no debug bundle (return_debug ignored)",
        )
    return result, ConformanceCheck(
        name=name,
        passed=True,
        detail=f"mlx_local generation returned finish_reason={result.finish_reason!r}",
    )


def _check_generation_text_quality(result) -> ConformanceCheck:
    name = "generation_text_quality"
    text = result.text
    if not text or not _passes_text_quality_gate(text):
        return ConformanceCheck(
            name=name,
            passed=False,
            detail=f"text failed quality gate: {text!r}",
        )
    return ConformanceCheck(
        name=name,
        passed=True,
        detail=f"non-empty text passes quality gate ({len(text)} chars)",
    )


def _check_generation_debug_metrics(metrics: dict) -> ConformanceCheck:
    name = "generation_debug_metrics"
    problems: list[str] = []

    forward_count = metrics.get("mlx_forward_count")
    if not isinstance(forward_count, int) or forward_count <= 0:
        problems.append(f"mlx_forward_count not >0: {forward_count!r}")

    output_token_count = metrics.get("mlx_output_token_count")
    if not isinstance(output_token_count, int) or output_token_count <= 0:
        problems.append(f"mlx_output_token_count not >0: {output_token_count!r}")

    s_over_t = metrics.get("mlx_s_over_t")
    if not isinstance(s_over_t, float):
        problems.append(f"mlx_s_over_t not a float: {s_over_t!r}")

    finish_reason = metrics.get("mlx_finish_reason")
    if not isinstance(finish_reason, str) or not finish_reason:
        problems.append(f"mlx_finish_reason not a non-empty str: {finish_reason!r}")

    if problems:
        return ConformanceCheck(name=name, passed=False, detail="; ".join(problems))
    return ConformanceCheck(
        name=name,
        passed=True,
        detail=(
            f"mlx_forward_count={forward_count}, "
            f"mlx_output_token_count={output_token_count}, "
            f"mlx_s_over_t={s_over_t:.4f}, mlx_finish_reason={finish_reason!r}"
        ),
    )


def _check_generation_provenance_metrics(metrics: dict) -> ConformanceCheck:
    name = "generation_provenance_metrics"
    keys = _PROVENANCE_KEYS
    if metrics.get("mlx_diffusion_decode_strategy") in {"bd3_fixed_quota", "fixed_quota"}:
        keys += (
            "mlx_diffusion_temperature",
            "mlx_diffusion_top_p",
            "mlx_diffusion_remasking",
        )
        problem = _fixed_quota_sampling_problem(metrics)
        if problem is not None:
            return ConformanceCheck(name=name, passed=False, detail=problem)
    missing = [key for key in keys if metrics.get(key) is None]
    if missing:
        return ConformanceCheck(
            name=name,
            passed=False,
            detail=f"missing provenance keys: {missing}",
        )
    return ConformanceCheck(
        name=name,
        passed=True,
        detail=", ".join(f"{key}={metrics[key]!r}" for key in keys),
    )


def _check_generation_baseline_status(metrics: dict) -> ConformanceCheck:
    """The baseline_status must be HONEST: forward_real True (the runtime ran a
    real, quality-passing forward), and entropy_real must match the active decode
    strategy -- False for block_threshold (confidence, not entropy, decoding)."""
    name = "generation_baseline_status_honest"
    baseline_status = metrics.get("baseline_status")
    if not isinstance(baseline_status, dict):
        return ConformanceCheck(
            name=name,
            passed=False,
            detail=f"baseline_status not a dict: {baseline_status!r}",
        )

    strategy = str(metrics.get("mlx_diffusion_decode_strategy", ""))
    problems: list[str] = []

    if baseline_status.get("forward_real") is not True:
        problems.append(f"forward_real not True: {baseline_status.get('forward_real')!r}")

    entropy_real = baseline_status.get("entropy_real")
    # entropy_real is only honest-True for the entropy strategy; for
    # block_threshold (default) it must be False.
    if strategy == "block_threshold":
        if entropy_real is not False:
            problems.append(
                f"entropy_real must be False for block_threshold: {entropy_real!r}"
            )
    elif strategy == "entropy_count":
        if entropy_real is not True:
            problems.append(
                f"entropy_real must be True for entropy_count: {entropy_real!r}"
            )
    elif not isinstance(entropy_real, bool):
        problems.append(f"entropy_real not a bool: {entropy_real!r}")

    if problems:
        return ConformanceCheck(name=name, passed=False, detail="; ".join(problems))
    return ConformanceCheck(
        name=name,
        passed=True,
        detail=(
            f"strategy={strategy!r}, forward_real=True, "
            f"entropy_real={entropy_real!r} (matches strategy)"
        ),
    )


def _check_generation_capability_case(result, *, capability=None) -> ConformanceCheck:
    name = "generation_capability_case"
    if capability is None:
        return ConformanceCheck(
            name=name,
            passed=True,
            detail="no descriptor-specific capability declared",
        )
    if capability.kind != "chat":
        return ConformanceCheck(
            name=name,
            passed=False,
            detail=f"unknown_capability_kind:{capability.kind}",
        )
    metrics = result.debug.metrics if result.debug is not None else {}
    problems: list[str] = []
    if metrics.get("mlx_diffusion_decode_strategy") in {"bd3_fixed_quota", "fixed_quota"}:
        problem = _fixed_quota_sampling_problem(metrics)
        if problem is not None:
            problems.append(problem)
    for check in capability.required_checks:
        if check == "nonempty":
            if not result.text.strip():
                problems.append("empty_text")
        elif check == "no_mask_token":
            marker = getattr(capability, "mask_token_text", None) or "<|mask|>"
            if marker in result.text:
                problems.append("mask_token_in_text")
            remaining = metrics.get("mlx_remaining_mask_count")
            if remaining != 0:
                problems.append(f"remaining_mask_count={remaining!r}")
        else:
            problems.append(f"unknown_capability_check:{check}")
    if problems:
        return ConformanceCheck(name=name, passed=False, detail="; ".join(problems))
    return ConformanceCheck(
        name=name,
        passed=True,
        detail=_capability_success_detail(capability, metrics=metrics),
    )


def _run_capability_case(
    runtime,
    *,
    generation_result,
    capability=None,
) -> ConformanceCheck:
    name = "generation_capability_case"
    if capability is None:
        return ConformanceCheck(
            name=name,
            passed=True,
            detail="no descriptor-specific capability declared",
        )
    if capability.kind == "chat":
        if generation_result is None:
            return ConformanceCheck(
                name=name,
                passed=False,
                detail="not evaluated: chat generation did not produce a result",
            )
        return _check_generation_capability_case(
            generation_result, capability=capability
        )
    if capability.kind not in {"infill", "code-infill"}:
        return ConformanceCheck(
            name=name,
            passed=False,
            detail=f"unknown_capability_kind:{capability.kind}",
        )
    if not isinstance(capability.prefix, str) or not isinstance(
        capability.suffix, str
    ):
        return ConformanceCheck(
            name=name,
            passed=False,
            detail="infill capability missing prefix/suffix",
        )
    mask_token_text = getattr(capability, "mask_token_text", None)
    if not isinstance(mask_token_text, str) or not mask_token_text:
        return ConformanceCheck(
            name=name,
            passed=False,
            detail="infill capability missing mask_token_text",
        )
    try:
        result = runtime.infill(
            capability.prefix,
            capability.suffix,
            backend="mlx_local",
            max_fill_tokens=capability.max_tokens,
            return_debug=True,
        )
    except Exception as exc:
        return ConformanceCheck(
            name=name,
            passed=False,
            detail=f"infill raised: {type(exc).__name__}: {exc}",
        )
    if result.debug is None:
        return ConformanceCheck(
            name=name,
            passed=False,
            detail="infill returned no debug bundle (return_debug ignored)",
        )
    return _check_infill_capability_case(result, capability=capability)


def _check_infill_capability_case(result, *, capability) -> ConformanceCheck:
    name = "generation_capability_case"
    metrics = result.debug.metrics if result.debug is not None else {}
    problems: list[str] = []
    for check in capability.required_checks:
        if check == "nonempty":
            if not result.filled_text.strip():
                problems.append("empty_fill")
        elif check == "no_mask_token":
            marker = capability.mask_token_text
            if marker in result.filled_text or marker in result.full_text:
                problems.append("mask_token_in_fill")
        elif check == "prefix_preserved":
            if not result.full_text.startswith(capability.prefix):
                problems.append("prefix_not_preserved")
        elif check == "suffix_preserved":
            if not result.full_text.endswith(capability.suffix):
                problems.append("suffix_not_preserved")
        elif check == "holes_filled":
            remaining = metrics.get("mlx_infill_remaining_mask_count")
            if remaining != 0:
                problems.append(f"remaining_infill_mask_count={remaining!r}")
        elif check == "fixed_signature":
            signature_problem = _fixed_signature_problem(
                result.filled_text,
                capability.required_signature,
            )
            if signature_problem is not None:
                problems.append(signature_problem)
        elif check == "python_parse":
            try:
                ast.parse(result.full_text)
            except (SyntaxError, ValueError, TypeError) as exc:
                problems.append(f"python_parse_failed:{type(exc).__name__}")
        else:
            problems.append(f"unknown_capability_check:{check}")
    if problems:
        return ConformanceCheck(name=name, passed=False, detail="; ".join(problems))
    return ConformanceCheck(
        name=name,
        passed=True,
        detail=_capability_success_detail(capability),
    )


def _fixed_quota_sampling_problem(metrics: dict) -> str | None:
    try:
        validate_fixed_quota_sampling_contract(
            temperature=metrics.get("mlx_diffusion_temperature"),
            top_p=metrics.get("mlx_diffusion_top_p"),
            remasking=metrics.get("mlx_diffusion_remasking"),
            strategy=metrics["mlx_diffusion_decode_strategy"],
        )
    except ValueError as exc:
        return str(exc)
    return None


def _capability_success_detail(capability, *, metrics: dict | None = None) -> str:
    sampling = ""
    if metrics is not None and metrics.get("mlx_diffusion_decode_strategy") in {
        "bd3_fixed_quota", "fixed_quota"
    }:
        sampling = (
            f"temperature={metrics['mlx_diffusion_temperature']!r}, "
            f"top_p={metrics['mlx_diffusion_top_p']!r}, "
            f"remasking={metrics['mlx_diffusion_remasking']!r}, "
        )
    return (
        f"kind={capability.kind!r}, decode=deterministic_greedy, "
        f"{sampling}"
        f"seed_field=reserved-zero, checks={capability.required_checks!r}"
    )


def _fixed_signature_problem(
    filled_text: str,
    required_signature: str | None,
) -> str | None:
    if not isinstance(required_signature, str) or not required_signature:
        return "required_signature_missing"
    from .model_coverage.producer_descriptors import (
        _canonical_required_signature_function,
    )

    expected = _canonical_required_signature_function(required_signature)
    if expected is None:
        return "required_signature_invalid"
    try:
        filled_module = ast.parse(filled_text)
    except (SyntaxError, ValueError, TypeError):
        return "required_signature_missing"

    if len(filled_module.body) != 1:
        return "required_signature_extra_statements"
    generated = filled_module.body[0]
    if not isinstance(generated, ast.FunctionDef) or generated.name != expected.name:
        return "required_signature_missing"
    if generated.decorator_list:
        return "required_signature_decorated"
    if getattr(generated, "type_params", ()):
        return "required_signature_type_parameters"
    expected_interface = _function_interface(expected)
    if _function_interface(generated) != expected_interface:
        return "required_signature_mismatch"
    return None


def _function_interface(function: ast.FunctionDef) -> tuple[object, ...]:
    arguments = function.args
    return (
        function.name,
        tuple(_ast_node_key(arg) for arg in arguments.posonlyargs),
        tuple(_ast_node_key(arg) for arg in arguments.args),
        _ast_node_key(arguments.vararg),
        tuple(_ast_node_key(arg) for arg in arguments.kwonlyargs),
        _ast_node_key(arguments.kwarg),
        tuple(_ast_node_key(default) for default in arguments.defaults),
        tuple(_ast_node_key(default) for default in arguments.kw_defaults),
        _ast_node_key(function.returns),
    )


def _ast_node_key(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    return ast.dump(node, annotate_fields=True, include_attributes=False)
