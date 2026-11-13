"""Stable backend identifiers shared by runtime, CLI, and diagnostics."""

from __future__ import annotations


AUTO_BACKEND = "auto"
NUMPY_LOCAL_BACKEND = "numpy_local"
MLX_LOCAL_BACKEND = "mlx_local"
HF_DREAM_REFERENCE_BACKEND = "hf_dream_reference"
CUDA_NATIVE_BACKEND = "cuda_native"

CUDA_PRECISION_IDS = ("bf16", "int8_weight_only")

SELECTABLE_BACKEND_IDS = (
    MLX_LOCAL_BACKEND,
    CUDA_NATIVE_BACKEND,
    HF_DREAM_REFERENCE_BACKEND,
    NUMPY_LOCAL_BACKEND,
)
EXPLICIT_BACKEND_IDS = frozenset(SELECTABLE_BACKEND_IDS)
ACCEPTED_BACKEND_IDS = (AUTO_BACKEND, *SELECTABLE_BACKEND_IDS)

# Product-roadmap entries shown by diagnostics. They are deliberately absent
# from SELECTABLE_BACKEND_IDS until real BackendRunner implementations land.
PLANNED_BACKEND_IDS = ("mobile_native",)


def validate_backend(backend: str) -> str:
    """Validate a user-selected backend before package or model I/O."""

    if backend in ACCEPTED_BACKEND_IDS:
        return backend
    if backend in PLANNED_BACKEND_IDS:
        raise ValueError(
            f"backend {backend!r} is planned but not implemented; no BackendRunner "
            f"exists for it yet. Available: {', '.join(ACCEPTED_BACKEND_IDS)}"
        )
    raise ValueError(
        f"unknown backend {backend!r}; expected one of "
        f"{', '.join(ACCEPTED_BACKEND_IDS)}"
    )


def validate_device_options(
    backend: str,
    precision: str | None,
    device_ordinal: int,
    device_memory_budget_bytes: int,
) -> None:
    """Validate device options at the common request boundary."""

    for name, value in (
        ("device_ordinal", device_ordinal),
        ("device_memory_budget_bytes", device_memory_budget_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if backend == CUDA_NATIVE_BACKEND:
        if precision not in CUDA_PRECISION_IDS:
            raise ValueError(
                "backend cuda_native requires precision="
                f"{' or '.join(repr(value) for value in CUDA_PRECISION_IDS)}"
            )
        return
    if precision is not None or device_ordinal != 0 or device_memory_budget_bytes != 0:
        raise ValueError(
            "precision, device_ordinal and device_memory_budget_bytes are "
            "accepted only with backend='cuda_native'"
        )


def validate_deployment_profile(profile: str | None) -> None:
    if profile is not None and (type(profile) is not str or not profile):
        raise ValueError("deployment_profile must be a non-empty string")
