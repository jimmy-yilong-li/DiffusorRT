"""Reference Compiler v0 — one exact catalog model into an execution package.

This is the **Reference Compiler** (equivalently, the Reference Package Builder):
checkpoint resolution and pinning, licence propagation, disk budgeting,
download, conversion, and cache publication. Product admission and an explicit
Local ``ModelCatalog`` are separate scopes that feed this same compiler. It is
Apache-2.0 and stays that way, so ``Engine.from_pretrained()`` resolves with no
proprietary component.

It is NOT the Optimizing Native Compiler — graph optimization, backend lowering,
quantization/tuning, device-specific execution plans, and signed backend
artifacts are the proprietary parts, and none of them live here. See
COPYRIGHT_AND_LICENSING.md §3 and §4; "compiler" unqualified is ambiguous and
should not be used in licensing prose.

Importing this package pulls in NO heavy dependency: planning is offline and must
work on a machine without torch, transformers, safetensors, or huggingface_hub.
The build path imports them lazily, inside the functions that need them, and
raises CompilerDependencyError naming the extra when one is absent.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

from .build import (
    VerifiedCompileReceiptSnapshot,
    cached_snapshot_path,
    compile_receipt_digest,
    compile_pretrained,
    sweep_stale_partials,
    verify_compile_receipt,
    verify_compile_receipt_snapshot,
    verify_package_receipt_snapshot,
)
from .errors import (
    CompilerDependencyError,
    CompilerError,
    InsufficientDiskError,
    ModelAccessError,
    RemoteCodeNotTrustedError,
    RevisionMismatchError,
    SnapshotNotCachedError,
    UnsupportedModelError,
)
from .planner import plan_pretrained
from .types import CompilePlan, CompileResult

__all__ = [
    "PINNED_MODEL_ID",
    "REGISTRY",
    "CompilePlan",
    "CompileResult",
    "CompilerDependencyError",
    "CompilerError",
    "InsufficientDiskError",
    "ModelAccessError",
    "RemoteCodeNotTrustedError",
    "RevisionMismatchError",
    "SnapshotNotCachedError",
    "UnsupportedModelError",
    "VerifiedCompileReceiptSnapshot",
    "cached_snapshot_path",
    "compile_receipt_digest",
    "compile_pretrained",
    "plan_pretrained",
    "sweep_stale_partials",
    "verify_compile_receipt",
    "verify_compile_receipt_snapshot",
    "verify_package_receipt_snapshot",
]


def __getattr__(name: str) -> Any:
    if name not in {"REGISTRY", "PINNED_MODEL_ID"}:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    registry = import_module(".registry", __name__)
    value = getattr(registry, name)
    globals()[name] = value
    return value
