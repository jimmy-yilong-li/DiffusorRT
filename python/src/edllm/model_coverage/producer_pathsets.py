"""Versioned, verifier-owned producer boundaries for model admission."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Mapping


QWEN_CANDIDATE_PATHSET_ID = "qwen3-mdlm-candidate-mlx-fp32-v10"
QWEN_PUBLIC_PATHSET_ID = "qwen3-mdlm-public-mlx-fp32-bidirectional-v11"
QWEN_BD3_CANDIDATE_PATHSET_ID = "qwen3-bd3lm-candidate-mlx-fp32-v10"
QWEN_BD3_PUBLIC_PATHSET_ID = (
    "qwen3-bd3lm-public-mlx-fp32-ragged-fixed-quota-no-cache-v11"
)
LLADA_CANDIDATE_PATHSET_ID = "llada-8b-instruct-candidate-mlx-fp32-v9"
LLADA_PUBLIC_PATHSET_ID = "llada-8b-instruct-public-mlx-fp32-baseline-v7"
DREAM_CODER_CANDIDATE_PATHSET_ID = "dream-coder-v0-instruct-7b-candidate-mlx-fp32-v9"
_DREAM_CODER_CANDIDATE_V1_PATHSET_ID = (
    "dream-coder-v0-instruct-7b-candidate-mlx-fp32-v1"
)

_QWEN_PROFILE = (
    (
        ("backend", "mlx_local"),
        ("minimum_device", "darwin_arm64_apple_m4_max_64gb"),
        ("precision", "fp32"),
    ),
)


@dataclass(frozen=True, slots=True)
class ProducerPathset:
    pathset_id: str
    candidate_key: str
    run_kind: str
    profiles: tuple[tuple[tuple[str, str], ...], ...]
    paths: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.pathset_id or not self.candidate_key or not self.run_kind:
            raise ValueError("producer_pathset_identity_invalid")
        if not self.paths or tuple(sorted(self.paths)) != self.paths:
            raise ValueError("producer_pathset_paths_not_canonical")
        if len(set(self.paths)) != len(self.paths):
            raise ValueError("producer_pathset_path_duplicate")
        for raw_path in self.paths:
            path = PurePosixPath(raw_path)
            if (
                not raw_path
                or path.is_absolute()
                or ".." in path.parts
                or "\\" in raw_path
                or "\x00" in raw_path
            ):
                raise ValueError("producer_pathset_path_invalid")


# Frozen SDK 0.2.0 boundaries. Additions belong to a new active version below,
# never to these tuples: receipts bind both the path list and the Git blobs.
_QWEN_CANDIDATE_PATHS = tuple(
    sorted(
        {
            "python/build-backend.lock.json",
            "python/pyproject.toml",
            "python/src/edllm/__init__.py",
            "python/src/edllm/backend_ids.py",
            "python/src/edllm/backends/__init__.py",
            "python/src/edllm/backends/backend_runner.py",
            "python/src/edllm/backends/cache_registry.py",
            "python/src/edllm/backends/diffusion_runtime.py",
            "python/src/edllm/backends/kernel_table.py",
            "python/src/edllm/backends/kv_capacity.py",
            "python/src/edllm/backends/kv_slices.py",
            "python/src/edllm/backends/mask_canvas.py",
            "python/src/edllm/backends/mlx_cache.py",
            "python/src/edllm/backends/mlx_dream.py",
            "python/src/edllm/backends/mlx_ops.py",
            "python/src/edllm/backends/mlx_state_cache.py",
            "python/src/edllm/backends/model_adapter.py",
            "python/src/edllm/backends/token_state.py",
            "python/src/edllm/backends/window_plan.py",
            "python/src/edllm/cache.py",
            "python/src/edllm/chat.py",
            "python/src/edllm/compiler/__init__.py",
            "python/src/edllm/compiler/build.py",
            "python/src/edllm/compiler/errors.py",
            "python/src/edllm/compiler/planner.py",
            "python/src/edllm/compiler/registry.py",
            "python/src/edllm/compiler/types.py",
            "python/src/edllm/conformance.py",
            "python/src/edllm/control_tokens.py",
            "python/src/edllm/converter/__init__.py",
            "python/src/edllm/converter/convert_hf_masked_diffusion.py",
            "python/src/edllm/converter/qwen3_chat_template.jinja",
            "python/src/edllm/decode.py",
            "python/src/edllm/dflash_drafter.py",
            "python/src/edllm/executor.py",
            "python/src/edllm/graph/__init__.py",
            "python/src/edllm/graph/backend_plan.py",
            "python/src/edllm/graph/emitter.py",
            "python/src/edllm/graph/ir.py",
            "python/src/edllm/graph/numpyref.py",
            "python/src/edllm/engine.py",
            "python/src/edllm/local_models.py",
            "python/src/edllm/model.py",
            "python/src/edllm/model_catalog.py",
            "python/src/edllm/model_definitions.py",
            "python/src/edllm/model_coverage/data/cases/qwen3_06b_mdlm_v0_1.json",
            "python/src/edllm/model_coverage/data/locks/qwen3_hf_457.json",
            "python/src/edllm/model_coverage/__init__.py",
            "python/src/edllm/model_coverage/admission.py",
            "python/src/edllm/model_coverage/admission_catalog.py",
            "python/src/edllm/model_coverage/admission_types.py",
            "python/src/edllm/model_coverage/candidate_evidence.py",
            "python/src/edllm/model_coverage/candidate_gate_worker.py",
            "python/src/edllm/model_coverage/parity.py",
            "python/src/edllm/model_coverage/producer_descriptors.py",
            "python/src/edllm/model_coverage/producer_pathsets.py",
            "python/src/edllm/model_coverage/reference.py",
            "python/src/edllm/model_coverage/data/locks/qwen3_product_311_mlx031.json",
            "python/src/edllm/model_coverage/descriptors.py",
            "python/src/edllm/package_artifacts.py",
            "python/src/edllm/policy.py",
            "python/src/edllm/presets.py",
            "python/src/edllm/result.py",
            "python/src/edllm/runtime_config.py",
            "python/src/edllm/schemas/__init__.py",
            "python/src/edllm/schemas/execution_package.schema.json",
            "python/src/edllm/tokenizer.py",
            "python/src/edllm/trust_identity.py",
            "python/src/edllm/validation.py",
            "python/src/edllm/weight_store.py",
            "scripts/installed_wheel_gate.py",
            "scripts/model_coverage_candidate_gate.py",
        }
    )
)

# Experimental admission is a control-plane transition performed only after a
# Candidate run has been committed.  These verifier/state files are therefore
# deliberately outside the new Candidate execution boundaries: changing a
# receipt pointer must not invalidate the model execution it points at.  The
# gate records whether the model is already runtime-configured as Product so
# the same Candidate evidence path can revalidate an existing Product without
# mutating runtime configuration. Model/runtime/converter/reference files
# remain bound below.
_EXPERIMENTAL_CONTROL_PATHS = {
    "python/src/edllm/model_coverage/admission.py",
    "python/src/edllm/model_coverage/admission_types.py",
    "python/src/edllm/model_coverage/producer_pathsets.py",
}

_QWEN_MDL_MODEL_INPUTS = {
    "python/src/edllm/converter/qwen3_chat_template.jinja",
    "python/src/edllm/model_coverage/data/cases/qwen3_06b_mdlm_v0_1.json",
    "python/src/edllm/model_coverage/data/locks/qwen3_hf_457.json",
}

_EXPERIMENTAL_CANDIDATE_BASE = (
    set(_QWEN_CANDIDATE_PATHS) - _EXPERIMENTAL_CONTROL_PATHS - _QWEN_MDL_MODEL_INPUTS
)

_QWEN_BD3_CANDIDATE_PATHS = tuple(
    sorted(
        _EXPERIMENTAL_CANDIDATE_BASE
        | {
            "python/src/edllm/converter/qwen3_chat_template.jinja",
            "python/src/edllm/model_coverage/data/cases/qwen3_06b_bd3lm_v0_1.json",
            "python/src/edllm/model_coverage/data/locks/qwen3_hf_457.json",
        }
    )
)

_LLADA_CANDIDATE_PATHS = tuple(
    sorted(
        _EXPERIMENTAL_CANDIDATE_BASE
        | {
            "python/src/edllm/model_coverage/data/cases/llada_8b_instruct.json",
            "python/src/edllm/model_coverage/data/locks/dream_hf_446.json",
        }
    )
)

_DREAM_CODER_CANDIDATE_PATHS = tuple(
    sorted(
        _EXPERIMENTAL_CANDIDATE_BASE
        | {
            "python/src/edllm/model_coverage/data/cases/dream_coder_v0_instruct_7b.json",
            "python/src/edllm/model_coverage/data/locks/dream_hf_446.json",
        }
    )
)

# The retained Dream-Coder receipt predates the unified model plan.  Preserve
# that exact verifier boundary for reading its historical Experimental record;
# new evidence is always produced against the active pathset below.
_DREAM_CODER_CANDIDATE_V1_PATHS = tuple(
    path
    for path in _DREAM_CODER_CANDIDATE_PATHS
    if path
    not in {
        "python/src/edllm/graph/backend_plan.py",
        "python/src/edllm/local_models.py",
        "python/src/edllm/model_catalog.py",
        "python/src/edllm/model_definitions.py",
        "python/src/edllm/weight_store.py",
    }
)

_CANDIDATE_ONLY_PATHS = {
    "python/src/edllm/conformance.py",
    "python/src/edllm/model_coverage/candidate_gate_worker.py",
    "python/src/edllm/model_coverage/parity.py",
    "python/src/edllm/model_coverage/reference.py",
    "scripts/model_coverage_candidate_gate.py",
}

_PUBLIC_PRODUCT_ADDITIONS = {
    "python/src/diffusor_rt/__init__.py",
    "python/src/diffusor_rt/__main__.py",
    "python/src/diffusor_rt/cli.py",
    "python/src/edllm/__main__.py",
    "python/src/edllm/_log.py",
    "python/src/edllm/backends/mlx_d2f_forward.py",
    "python/src/edllm/capabilities.py",
    "python/src/edllm/chat.py",
    "python/src/edllm/cli.py",
    "python/src/edllm/compiler/__init__.py",
    "python/src/edllm/compiler/build.py",
    "python/src/edllm/compiler/errors.py",
    "python/src/edllm/compiler/planner.py",
    "python/src/edllm/compiler/registry.py",
    "python/src/edllm/compiler/types.py",
    "python/src/edllm/deployment.py",
    "python/src/edllm/engine.py",
    "python/src/edllm/local_models.py",
    "python/src/edllm/model_coverage/descriptors.py",
    "python/src/edllm/service_asgi.py",
    "python/src/edllm/service_contracts.py",
    "python/src/edllm/service_runtime.py",
    "scripts/installed_wheel_gate.py",
    "scripts/private_runtime_wheel_gate.py",
}

_QWEN_PUBLIC_PATHS = tuple(
    sorted(
        (set(_QWEN_CANDIDATE_PATHS) - _CANDIDATE_ONLY_PATHS) | _PUBLIC_PRODUCT_ADDITIONS
    )
)

_QWEN_BD3_PUBLIC_PATHS = tuple(
    sorted(
        (set(_QWEN_BD3_CANDIDATE_PATHS) - _CANDIDATE_ONLY_PATHS)
        | _PUBLIC_PRODUCT_ADDITIONS
    )
)

_LLADA_PUBLIC_PATHS = tuple(
    sorted(
        (set(_LLADA_CANDIDATE_PATHS) - _CANDIDATE_ONLY_PATHS)
        | _PUBLIC_PRODUCT_ADDITIONS
    )
)

_SDK020_PATHSETS: Mapping[str, ProducerPathset] = MappingProxyType(
    {
        pathset_id: ProducerPathset(
            pathset_id=pathset_id,
            candidate_key=candidate_key,
            run_kind=run_kind,
            profiles=_QWEN_PROFILE,
            paths=paths,
        )
        for pathset_id, candidate_key, run_kind, paths in (
            (
                "dream-coder-v0-instruct-7b-candidate-mlx-fp32-v4",
                "dream_coder_v0_instruct_7b",
                "candidate-four-gate",
                _DREAM_CODER_CANDIDATE_PATHS,
            ),
            (
                "llada-8b-instruct-candidate-mlx-fp32-v4",
                "llada_8b_instruct",
                "candidate-four-gate",
                _LLADA_CANDIDATE_PATHS,
            ),
            (
                "llada-8b-instruct-public-mlx-fp32-baseline-v1",
                "llada_8b_instruct",
                "postpromotion-public-product-path",
                _LLADA_PUBLIC_PATHS,
            ),
            (
                "qwen3-bd3lm-candidate-mlx-fp32-v5",
                "qwen3_06b_bd3lm_v0_1",
                "candidate-four-gate",
                _QWEN_BD3_CANDIDATE_PATHS,
            ),
            (
                "qwen3-bd3lm-public-mlx-fp32-ragged-fixed-quota-no-cache-v5",
                "qwen3_06b_bd3lm_v0_1",
                "postpromotion-public-product-path",
                _QWEN_BD3_PUBLIC_PATHS,
            ),
            (
                "qwen3-mdlm-candidate-mlx-fp32-v5",
                "qwen3_06b_mdlm_v0_1",
                "candidate-four-gate",
                _QWEN_CANDIDATE_PATHS,
            ),
            (
                "qwen3-mdlm-public-mlx-fp32-exact-d2f-v5",
                "qwen3_06b_mdlm_v0_1",
                "postpromotion-public-product-path",
                _QWEN_PUBLIC_PATHS,
            ),
        )
    }
)

_SDK021_LOCK = (
    "python/src/edllm/model_coverage/data/locks/qwen3_product_311_mlx031_sdk021.json"
)
_SDK021_PUBLIC_ADDITIONS = {
    "python/src/edllm/application_state.py",
    "python/src/edllm/conversation.py",
    "python/src/edllm/_conversation_store.py",
    "python/src/edllm/package_lifecycle.py",
}
_SDK021_PATHSETS: Mapping[str, ProducerPathset] = MappingProxyType(
    {
        active_id: replace(
            _SDK020_PATHSETS[previous_id],
            pathset_id=active_id,
            paths=tuple(
                sorted(
                    set(_SDK020_PATHSETS[previous_id].paths)
                    | {_SDK021_LOCK}
                    | (
                        _SDK021_PUBLIC_ADDITIONS
                        if _SDK020_PATHSETS[previous_id].run_kind
                        == "postpromotion-public-product-path"
                        else set()
                    )
                )
            ),
        )
        for previous_id, active_id in (
            (
                "dream-coder-v0-instruct-7b-candidate-mlx-fp32-v4",
                "dream-coder-v0-instruct-7b-candidate-mlx-fp32-v5",
            ),
            ("llada-8b-instruct-candidate-mlx-fp32-v4", "llada-8b-instruct-candidate-mlx-fp32-v5"),
            ("llada-8b-instruct-public-mlx-fp32-baseline-v1", "llada-8b-instruct-public-mlx-fp32-baseline-v2"),
            ("qwen3-bd3lm-candidate-mlx-fp32-v5", "qwen3-bd3lm-candidate-mlx-fp32-v6"),
            (
                "qwen3-bd3lm-public-mlx-fp32-ragged-fixed-quota-no-cache-v5",
                "qwen3-bd3lm-public-mlx-fp32-ragged-fixed-quota-no-cache-v6",
            ),
            ("qwen3-mdlm-candidate-mlx-fp32-v5", "qwen3-mdlm-candidate-mlx-fp32-v6"),
            ("qwen3-mdlm-public-mlx-fp32-exact-d2f-v5", "qwen3-mdlm-public-mlx-fp32-exact-d2f-v6"),
        )
    }
)

_SDK022_LOCK = "python/src/edllm/model_coverage/data/locks/qwen3_product_311_mlx031_sdk022.json"
_SDK022_PATHSETS: Mapping[str, ProducerPathset] = MappingProxyType({
    active_id: replace(
        _SDK021_PATHSETS[previous_id],
        pathset_id=active_id,
        paths=tuple(sorted(set(_SDK021_PATHSETS[previous_id].paths) | {_SDK022_LOCK})),
    )
    for previous_id, active_id in (
        ("dream-coder-v0-instruct-7b-candidate-mlx-fp32-v5", "dream-coder-v0-instruct-7b-candidate-mlx-fp32-v6"),
        ("llada-8b-instruct-candidate-mlx-fp32-v5", "llada-8b-instruct-candidate-mlx-fp32-v6"),
        ("llada-8b-instruct-public-mlx-fp32-baseline-v2", "llada-8b-instruct-public-mlx-fp32-baseline-v3"),
        ("qwen3-bd3lm-candidate-mlx-fp32-v6", "qwen3-bd3lm-candidate-mlx-fp32-v7"),
        ("qwen3-bd3lm-public-mlx-fp32-ragged-fixed-quota-no-cache-v6", "qwen3-bd3lm-public-mlx-fp32-ragged-fixed-quota-no-cache-v7"),
        ("qwen3-mdlm-candidate-mlx-fp32-v6", "qwen3-mdlm-candidate-mlx-fp32-v7"),
        ("qwen3-mdlm-public-mlx-fp32-exact-d2f-v6", "qwen3-mdlm-public-mlx-fp32-bidirectional-v7"),
    )
})

# The forward extraction changes the producer file boundary, not the evidence.
# Preserve every historical path list; future runs bind the backend-owned math.
_M1_FORWARD_PATHSETS: Mapping[str, ProducerPathset] = MappingProxyType({
    active_id: replace(
        _SDK022_PATHSETS[previous_id],
        pathset_id=active_id,
        paths=tuple(sorted(
            set(_SDK022_PATHSETS[previous_id].paths)
            | {"python/src/edllm/backends/mlx_forward.py"}
        )),
    )
    for previous_id, active_id in (
        ("dream-coder-v0-instruct-7b-candidate-mlx-fp32-v6", "dream-coder-v0-instruct-7b-candidate-mlx-fp32-v7"),
        ("llada-8b-instruct-candidate-mlx-fp32-v6", "llada-8b-instruct-candidate-mlx-fp32-v7"),
        ("llada-8b-instruct-public-mlx-fp32-baseline-v3", "llada-8b-instruct-public-mlx-fp32-baseline-v4"),
        ("qwen3-bd3lm-candidate-mlx-fp32-v7", "qwen3-bd3lm-candidate-mlx-fp32-v8"),
        ("qwen3-bd3lm-public-mlx-fp32-ragged-fixed-quota-no-cache-v7", "qwen3-bd3lm-public-mlx-fp32-ragged-fixed-quota-no-cache-v8"),
        ("qwen3-mdlm-candidate-mlx-fp32-v7", "qwen3-mdlm-candidate-mlx-fp32-v8"),
        ("qwen3-mdlm-public-mlx-fp32-bidirectional-v7", "qwen3-mdlm-public-mlx-fp32-bidirectional-v8"),
    )
})

# Session assembly is part of every active producer, including cache-OFF paths.
# Keep the earlier forward-only boundary readable without rewriting its receipts.
_M1_SESSION_PATHSETS: Mapping[str, ProducerPathset] = MappingProxyType({
    active_id: replace(
        _M1_FORWARD_PATHSETS[previous_id],
        pathset_id=active_id,
        paths=tuple(sorted(
            set(_M1_FORWARD_PATHSETS[previous_id].paths)
            | {"python/src/edllm/backends/decode_session.py"}
        )),
    )
    for previous_id, active_id in (
        ("qwen3-mdlm-candidate-mlx-fp32-v8", "qwen3-mdlm-candidate-mlx-fp32-v9"),
        ("qwen3-mdlm-public-mlx-fp32-bidirectional-v8", "qwen3-mdlm-public-mlx-fp32-bidirectional-v9"),
        ("qwen3-bd3lm-candidate-mlx-fp32-v8", "qwen3-bd3lm-candidate-mlx-fp32-v9"),
        ("qwen3-bd3lm-public-mlx-fp32-ragged-fixed-quota-no-cache-v8", "qwen3-bd3lm-public-mlx-fp32-ragged-fixed-quota-no-cache-v9"),
        ("llada-8b-instruct-candidate-mlx-fp32-v7", "llada-8b-instruct-candidate-mlx-fp32-v8"),
        ("llada-8b-instruct-public-mlx-fp32-baseline-v4", "llada-8b-instruct-public-mlx-fp32-baseline-v5"),
        ("dream-coder-v0-instruct-7b-candidate-mlx-fp32-v7", "dream-coder-v0-instruct-7b-candidate-mlx-fp32-v8"),
    )
})

# Shared cancellation checkpoints execute even when no token is supplied.
# Bind the dependency in a new boundary; never mutate historical receipts.
_CANCELLATION_PATHSETS: Mapping[str, ProducerPathset] = MappingProxyType({
    active_id: replace(
        _M1_SESSION_PATHSETS[previous_id], pathset_id=active_id,
        paths=tuple(sorted(set(_M1_SESSION_PATHSETS[previous_id].paths)
                           | {"python/src/edllm/cancellation.py"})),
    )
    for previous_id, active_id in (
        ("qwen3-mdlm-candidate-mlx-fp32-v9", QWEN_CANDIDATE_PATHSET_ID),
        ("qwen3-mdlm-public-mlx-fp32-bidirectional-v9", "qwen3-mdlm-public-mlx-fp32-bidirectional-v10"),
        ("qwen3-bd3lm-candidate-mlx-fp32-v9", QWEN_BD3_CANDIDATE_PATHSET_ID),
        ("qwen3-bd3lm-public-mlx-fp32-ragged-fixed-quota-no-cache-v9", "qwen3-bd3lm-public-mlx-fp32-ragged-fixed-quota-no-cache-v10"),
        ("llada-8b-instruct-candidate-mlx-fp32-v8", LLADA_CANDIDATE_PATHSET_ID),
        ("llada-8b-instruct-public-mlx-fp32-baseline-v5", "llada-8b-instruct-public-mlx-fp32-baseline-v6"),
        ("dream-coder-v0-instruct-7b-candidate-mlx-fp32-v8", DREAM_CODER_CANDIDATE_PATHSET_ID),
    )
})

# author:Opus -- the CLI opens a model through model_opening.open_model, the code
# it has shared with the chat windows since 80e0cdb2. That work used to sit in
# cli.py, inside this boundary; it runs on every public product path. Bind the
# dependency in a new boundary; never mutate historical receipts. A Candidate run
# does not go through the CLI, so its boundaries are the ones above.
PRODUCER_PATHSETS: Mapping[str, ProducerPathset] = MappingProxyType({
    **{
        pathset_id: pathset
        for pathset_id, pathset in _CANCELLATION_PATHSETS.items()
        if pathset.run_kind == "candidate-four-gate"
    },
    **{
        active_id: replace(
            _CANCELLATION_PATHSETS[previous_id], pathset_id=active_id,
            paths=tuple(sorted(set(_CANCELLATION_PATHSETS[previous_id].paths)
                               | {"python/src/edllm/model_opening.py"})),
        )
        for previous_id, active_id in (
            ("qwen3-mdlm-public-mlx-fp32-bidirectional-v10", QWEN_PUBLIC_PATHSET_ID),
            ("qwen3-bd3lm-public-mlx-fp32-ragged-fixed-quota-no-cache-v10", QWEN_BD3_PUBLIC_PATHSET_ID),
            ("llada-8b-instruct-public-mlx-fp32-baseline-v6", LLADA_PUBLIC_PATHSET_ID),
        )
    },
})

RECEIPT_PATHSETS: Mapping[str, ProducerPathset] = MappingProxyType(
    {
        _DREAM_CODER_CANDIDATE_V1_PATHSET_ID: ProducerPathset(
            pathset_id=_DREAM_CODER_CANDIDATE_V1_PATHSET_ID,
            candidate_key="dream_coder_v0_instruct_7b",
            run_kind="candidate-four-gate",
            profiles=_QWEN_PROFILE,
            paths=_DREAM_CODER_CANDIDATE_V1_PATHS,
        ),
        **_SDK020_PATHSETS,
        **_SDK021_PATHSETS,
        **_SDK022_PATHSETS,
        **_M1_FORWARD_PATHSETS,
        **_M1_SESSION_PATHSETS,
        **_CANCELLATION_PATHSETS,
        **PRODUCER_PATHSETS,
    }
)

# Every local import reachable syntactically from a producer pathset must either
# be in that pathset or be classified here. These are import-only, lazy, or
# disabled-profile modules; the regression test fails when a new dependency is
# added without an explicit producer-boundary decision.
PATHSET_IMPORT_EXCLUSIONS: Mapping[str, str] = MappingProxyType(
    {
        "python/src/edllm/abi/enums.py": (
            "imported in RuntimeHandle._generate_cuda_native for native session "
            "configuration; these producer pathsets execute the MLX backend"
        ),
        "python/src/edllm/decode_progress.py": (
            "imported only when a caller passes on_progress (the chat window); "
            "no model-coverage producer does"
        ),
        "python/src/edllm/gui/__init__.py": (
            "loaded only by `diffusorrt gui`, the desktop chat window; no "
            "model-coverage producer opens it"
        ),
        "python/src/edllm/gui/preferences.py": (
            "read only by the CLI GUI option validator; model-coverage "
            "producers do not launch a chat window"
        ),
        "python/src/edllm/_windows_files.py": (
            "Windows file layer, imported only when os.name == 'nt'; every "
            "model-coverage producer runs on macOS"
        ),
        "python/src/edllm/application_state.py": (
            "application state is outside Candidate forward execution; "
            "SDK 0.2.1 public-product pathsets bind it for durable chat"
        ),
        "python/src/edllm/backends/cache_identity.py": (
            "snapshot-only policy identity loaded lazily when snapshot I/O is requested"
        ),
        "python/src/edllm/backends/correction_policy.py": "error fixing disabled",
        "python/src/edllm/backends/cuda_profile.py": (
            "loaded only for explicit cuda_native package admission; every "
            "model-coverage profile in this verifier is mlx_local"
        ),
        "python/src/edllm/backends/device_runtime.py": (
            "loaded only after explicit cuda_native selection; every "
            "model-coverage profile in this verifier is mlx_local"
        ),
        "python/src/edllm/backends/metal_ops.py": "unselected generic metal backend",
        "python/src/edllm/backends/kv_manager.py": (
            "native exact-D2F owner disabled by the frozen candidate profile"
        ),
        "python/src/edllm/backends/kv_runtime.py": (
            "native exact-D2F owner disabled by the frozen candidate profile"
        ),
        "python/src/edllm/backends/mlx_d2f_forward.py": (
            "public deployment exact-d2f; candidate evidence binds baseline-no-d2f"
        ),
        "python/src/edllm/backends/native_d2f_cache.py": (
            "native exact-D2F owner disabled by the frozen candidate profile"
        ),
        "python/src/edllm/backends/snapshot_store.py": (
            "snapshot I/O is optional and disabled in the admitted generation profile"
        ),
        "python/src/edllm/converter/hf_import.py": "lazy unrelated converter export",
        "python/src/edllm/converter/merge.py": "lazy unrelated converter export",
        "python/src/edllm/converter/package_writer.py": "lazy unrelated converter export",
        "python/src/edllm/conversation.py": (
            "application conversation orchestration is not part of model "
            "conversion or Candidate numerics; SDK 0.2.1 public-product "
            "pathsets bind it and its SQLite store for chat restart"
        ),
        "python/src/edllm/capabilities.py": (
            "candidate installed gate uses Engine generation, not optional diagnostics"
        ),
        "python/src/edllm/deployment.py": (
            "loaded only by local-model diagnostics; the candidate gate opens "
            "an explicit execution package"
        ),
        "python/src/edllm/model_coverage/admission.py": (
            "post-run receipt verification and Product control state"
        ),
        "python/src/edllm/model_coverage/admission_types.py": (
            "post-run admission state and receipt-pointer control plane"
        ),
        "python/src/edllm/models_tui.py": (
            "optional interactive model management is not imported by the "
            "non-interactive admission workflow"
        ),
        "python/src/edllm/package_lifecycle.py": (
            "cross-process locking and atomic publication coordinate package "
            "lifetime without changing Candidate numerics; SDK 0.2.1 "
            "public-product pathsets bind this executed lifecycle dependency"
        ),
        "python/src/edllm/dilated_schedule.py": "dilated decode disabled",
        "python/src/edllm/mask_fixing.py": "unselected CLI command",
    }
)


__all__ = [
    "DREAM_CODER_CANDIDATE_PATHSET_ID",
    "LLADA_CANDIDATE_PATHSET_ID",
    "LLADA_PUBLIC_PATHSET_ID",
    "PATHSET_IMPORT_EXCLUSIONS",
    "PRODUCER_PATHSETS",
    "RECEIPT_PATHSETS",
    "QWEN_BD3_CANDIDATE_PATHSET_ID",
    "QWEN_BD3_PUBLIC_PATHSET_ID",
    "QWEN_CANDIDATE_PATHSET_ID",
    "QWEN_PUBLIC_PATHSET_ID",
    "ProducerPathset",
]
