"""Exact product admission for compiled CUDA execution packages.

Model descriptors describe model semantics and remain independent of device
backends.  This module is the smaller join: it admits one exact package shape
for one compiled backend profile without adding CUDA to the model's Apple
receipt or pretending that every structurally compatible package was tested.

The large weights file is not re-hashed on every Engine construction.  Its
digest is bound by the exact compile-receipt bytes below; the native package
loader verifies the artifact set, the small control-plane artifacts and the
manifest/receipt weight-digest agreement before it maps the weights.  Full
artifact hashing remains an acquisition/install gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from ..backend_ids import CUDA_NATIVE_BACKEND, CUDA_PRECISION_IDS
from ..model_definitions import DeploymentContract, DeploymentTarget
from ..package_artifacts import open_package_artifact, sha256_stream


@dataclass(frozen=True, slots=True)
class CudaExecutionProfile:
    profile_id: str
    model_id: str
    revision: str
    descriptor_key: str
    descriptor_digest: str
    compile_receipt_sha256: str
    manifest_sha256: str
    graph_sha256: str
    weights_sha256: str
    precisions: tuple[str, ...]
    minimum_device: str
    # Geometry and mask facts the in-memory config must still report.
    runtime_config: Mapping[str, object] = field(hash=False)
    # Applied to the detached config once this exact package is admitted.
    runtime_overlay: Mapping[str, object] = field(hash=False)

    @property
    def deployment_targets(self) -> tuple[DeploymentTarget, ...]:
        """Project the authenticated provider profile into the model plan."""

        contract = DeploymentContract(
            profile=self.profile_id,
            config_source="package",
        )
        return tuple(
            DeploymentTarget(
                backend=CUDA_NATIVE_BACKEND,
                precision=precision,
                minimum_device=self.minimum_device,
                deployment=contract,
            )
            for precision in self.precisions
        )


_DREAM_RUNTIME_CONFIG = {
    "architecture": "DreamModel",
    "attention_mask_kind": "d2f_hybrid",
    "block_size": 8,
    "mask_token_id": 151666,
    "layer_count": 28,
    "hidden_size": 3584,
    "attention_heads": 28,
    "num_kv_heads": 4,
    "head_dim": 128,
    "vocab_size": 152064,
}

_DREAM_RUNTIME_OVERLAY = {
    # The compiled package records graph/model truth, while this flag selects
    # the already-admitted native K/V owner for the shared block-threshold
    # decoder.  Keep the overlay here, beside the exact package admission,
    # rather than weakening canonical validation for every backend.
    "mlx_diffusion_d2f_exact_kv_cache": True,
}

_QWEN3_MDLM_RUNTIME_CONFIG = {
    "architecture": "A2DQwen3",
    "attention_mask_kind": "bidirectional",
    "mask_token_id": 151669,
    "layer_count": 28,
    "hidden_size": 1024,
    "attention_heads": 16,
    "num_kv_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
}

_DREAM_D2F_A100_V1 = CudaExecutionProfile(
    profile_id="dream-v0-instruct-7b-d2f-a100-v1",
    model_id="Dream-org/Dream-v0-Instruct-7B",
    revision="05334cb9faaf763692dcf9d8737c642be2b2a6ae",
    descriptor_key="dream_v0_instruct_7b",
    descriptor_digest=(
        "0cb9c169da557bc050f90da9f3107cbecb94d9e1b5747b2e735a5a5ff4dce780"
    ),
    compile_receipt_sha256=(
        "624c6e267140653f50a12442fe13531068c9fcb8518a5303a2ce0182044ecc31"
    ),
    manifest_sha256=(
        "70bb5f519774a68ec1e78d02ec6398b5d2edd657b8bf005d11c4acd69fbfcf4f"
    ),
    graph_sha256=(
        "b60b3313fedaf7f8b91bb015ce00e2c0ea0c1cc15d5b82aff02b1e54d3ebf2d1"
    ),
    weights_sha256=(
        "909430f2635de4335a0eda1de5d00d078b3fca507e052b02325f10e8e0e3437e"
    ),
    precisions=("bf16", "int8_weight_only"),
    minimum_device="linux_x86_64_nvidia_sm80_40gb",
    runtime_config=_DREAM_RUNTIME_CONFIG,
    runtime_overlay=_DREAM_RUNTIME_OVERLAY,
)

# Experimentally gated: forward parity and generation on one named SM89 laptop,
# not product admission. minimum_device labels that evidence; nothing checks the
# live GPU against it.
_QWEN3_MDLM_SM89_V1 = CudaExecutionProfile(
    profile_id="qwen3-0.6b-mdlm-bidirectional-sm89-experimental-v1",
    model_id="dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1",
    revision="c8d24a3f4adaeef46881b450e1bf7d1005203bd7",
    descriptor_key="qwen3_06b_mdlm_v0_1",
    descriptor_digest=(
        "943b42875ea5fac57feeea939edda94e948fc6063eb76460984f0719d340672b"
    ),
    compile_receipt_sha256=(
        "6d3f91ada07e130f565bc38482b8245a15217ef06e83d948373d0beb121654a1"
    ),
    manifest_sha256=(
        "dc5f585ef96aac0d7eeebe253e889dfc5dfdd4de7ffc573b7d5edacb4fdc14ee"
    ),
    graph_sha256=(
        "1d21ecf5bc33f212782eef76bc49a804455f5c4c5f3a9808b2a7c36b37b9b7cc"
    ),
    weights_sha256=(
        "654d0c40526200f26802cbcb76a7c3467f1015513a0d031b576b3b62c925d118"
    ),
    precisions=("bf16",),
    minimum_device="linux_x86_64_nvidia_sm89_8gb",
    runtime_config=_QWEN3_MDLM_RUNTIME_CONFIG,
    runtime_overlay={},
)

_QWEN3_BD3LM_RUNTIME_CONFIG = {
    "architecture": "A2DQwen3BD3LM",
    "attention_mask_kind": "d2f_hybrid",
    "block_size": 32,
    "mask_token_id": 151669,
    "layer_count": 28,
    "hidden_size": 1024,
    "attention_heads": 16,
    "num_kv_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
}

# Experimentally gated like the MDLM profile: window parity and generation on
# one named SM89 laptop. bd3_fixed_quota runs on a device session over its
# absolute block geometry with no cache, so no K/V overlay is admitted.
_QWEN3_BD3LM_SM89_V1 = CudaExecutionProfile(
    profile_id="qwen3-0.6b-bd3lm-d2f-hybrid-sm89-experimental-v1",
    model_id="dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1",
    revision="60f77bba67ba94231c16f5fb29bc66c500b038f1",
    descriptor_key="qwen3_06b_bd3lm_v0_1",
    descriptor_digest=(
        "a1234df92ca9705a6ba1d3b8aecc7740f70d21c7ca13cbd028b21b0eca83e4c1"
    ),
    compile_receipt_sha256=(
        "ff119416fe1b8f31acf00fd48f0b91bc6ab47a16e9ee15b5898dabf415d06630"
    ),
    manifest_sha256=(
        "f5960692030edca1ef8b48deffd3b0c4cfff40d58a8fc39b0a8d7a9d6c699ee7"
    ),
    graph_sha256=(
        "4f59e57c245be76f39344b9d00dc3a13ba99defa6d4f125ed8efa588d6ede63e"
    ),
    weights_sha256=(
        "6d78a7c1f75a68065d56ea192f14e7baaf34152ba26627f0bf6de658faf5bc11"
    ),
    precisions=("bf16",),
    minimum_device="linux_x86_64_nvidia_sm89_8gb",
    runtime_config=_QWEN3_BD3LM_RUNTIME_CONFIG,
    runtime_overlay={},
)

CUDA_EXECUTION_PROFILES = (_DREAM_D2F_A100_V1, _QWEN3_MDLM_SM89_V1, _QWEN3_BD3LM_SM89_V1)

_CONTROL_ARTIFACT_LIMITS = {
    ".diffusorrt-compile.json": 4 * 1024 * 1024,
    "model.json": 4 * 1024 * 1024,
    "graph.json": 64 * 1024 * 1024,
}


def _artifact_sha256(package_dir: Path, name: str) -> str:
    with open_package_artifact(
        package_path=package_dir,
        artifact_ref=name,
        error_prefix="cuda_profile",
        artifact_name=name,
        max_bytes=_CONTROL_ARTIFACT_LIMITS[name],
    ) as (handle, _path, _source_stat):
        return sha256_stream(handle)


def admit_cuda_execution_profile(
    package_dir: str | Path,
    *,
    precision: str | None,
    model_config: Mapping[str, object] | None = None,
) -> CudaExecutionProfile:
    """Return the exact admitted CUDA profile or fail closed.

    ``precision=None`` is useful to ask whether a package has any CUDA product
    profile.  An execution path must provide a concrete precision.
    """
    if precision is not None and precision not in CUDA_PRECISION_IDS:
        raise ValueError(
            f"unsupported CUDA precision {precision!r}; expected one of "
            f"{', '.join(CUDA_PRECISION_IDS)}"
        )

    root = Path(package_dir)
    receipt_sha256 = _artifact_sha256(root, ".diffusorrt-compile.json")
    profile = next(
        (
            candidate
            for candidate in CUDA_EXECUTION_PROFILES
            if candidate.compile_receipt_sha256 == receipt_sha256
        ),
        None,
    )
    if profile is None:
        raise ValueError("cuda_package_profile_not_admitted")
    if precision is not None and precision not in profile.precisions:
        raise ValueError(
            f"cuda_precision_not_admitted:{profile.profile_id}:{precision}"
        )

    expected = {
        "model.json": profile.manifest_sha256,
        "graph.json": profile.graph_sha256,
    }
    for name, expected_sha256 in expected.items():
        if _artifact_sha256(root, name) != expected_sha256:
            raise ValueError(f"cuda_profile_artifact_mismatch:{name}")

    # The product decoder reads this in-memory config while the native provider
    # reads the package's verified graph.  Refuse a caller-mutated handle rather
    # than letting the two halves execute different geometry or mask semantics.
    if model_config is not None:
        for key, expected_value in profile.runtime_config.items():
            if model_config.get(key) != expected_value:
                raise ValueError(f"cuda_profile_runtime_config_mismatch:{key}")
        for key, expected_value in profile.runtime_overlay.items():
            value = model_config.get(key)
            if value is not None and value is not expected_value:
                raise ValueError(f"cuda_profile_runtime_config_mismatch:{key}")
    return profile


def resolve_cuda_execution_profile_config(
    package_dir: str | Path,
    *,
    model_config: Mapping[str, object],
) -> tuple[CudaExecutionProfile, dict[str, object]]:
    """Authenticate one package, then return its detached CUDA config.

    The package bytes stay unchanged.  Only the exact admitted package can
    reach this overlay, and the ordinary loader continues to validate the
    package exactly as written.
    """
    profile = admit_cuda_execution_profile(
        package_dir,
        precision=None,
        model_config=model_config,
    )
    return profile, {**model_config, **profile.runtime_overlay}
