"""The DiffusorRT boundary ABI, version 0.1.0.

Apache-2.0 SDK surface. Per COPYRIGHT_AND_LICENSING.md the optimized native
engine that implements this contract lives in a separate private repository;
no proprietary source belongs here.

Every type is restricted to what a C struct can carry. Importing this package
must not import mlx, torch, or transformers.
"""
from __future__ import annotations

from .lifetimes import UNSET, EngineConfig, ModelGeometry, SessionDecodeConfig

ABI_VERSION = (0, 1, 0)

__all__ = [
    "ABI_VERSION", "UNSET",
    "EngineConfig", "ModelGeometry", "SessionDecodeConfig",
]

from .buffers import (  # noqa: E402
    MAX_NDIM, BufferView, Device, DType, Ownership, TensorDesc, describe_array,
)
from .handles import NULL_HANDLE, HandleKind, HandleRegistry  # noqa: E402

__all__ += [
    "MAX_NDIM", "BufferView", "Device", "DType", "Ownership", "TensorDesc",
    "describe_array", "NULL_HANDLE", "HandleKind", "HandleRegistry",
]
