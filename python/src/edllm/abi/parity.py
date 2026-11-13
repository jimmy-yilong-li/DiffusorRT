"""Three-tier parity comparison.

Tier 0 CONTROL_PLANE       — bit-exact. Integer logic; any difference is a bug.
Tier 1 SAME_BACKEND_TENSOR — bit-exact digest. What the project's existing
                             byte-identity gates already assert.
Tier 2 CROSS_BACKEND       — tolerance AND committed-token equality. Reduction
                             order legitimately differs across MLX/Metal/CUDA;
                             demanding bit-exactness there produces a gate that
                             gets waived, which is worse than no gate.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class ParityTier(IntEnum):
    CONTROL_PLANE = 0
    SAME_BACKEND_TENSOR = 1
    CROSS_BACKEND = 2


@dataclass(frozen=True, slots=True)
class ParityResult:
    tier: int
    passed: bool
    detail: str


def digest(array) -> str:
    contiguous = np.ascontiguousarray(array)
    if contiguous.dtype.byteorder == ">":
        contiguous = contiguous.astype(contiguous.dtype.newbyteorder("<"))
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def compare_control_plane(a: dict, b: dict) -> ParityResult:
    """Bit-exact structural comparison; reports the FIRST differing key so a
    failure names the field instead of dumping two structures."""
    for key in sorted(set(a) | set(b)):
        if key not in a or key not in b:
            return ParityResult(
                ParityTier.CONTROL_PLANE, False, f"{key}: present on one side only")
        if a[key] != b[key]:
            return ParityResult(
                ParityTier.CONTROL_PLANE, False, f"{key}: {a[key]!r} != {b[key]!r}")
    return ParityResult(ParityTier.CONTROL_PLANE, True, "identical")


def compare_tensor_exact(a, b) -> ParityResult:
    da, db = digest(a), digest(b)
    return ParityResult(
        ParityTier.SAME_BACKEND_TENSOR, da == db,
        "identical" if da == db else f"digest {da[:12]} != {db[:12]}")


def compare_tensor_tolerant(a, b, *, rtol: float, atol: float) -> ParityResult:
    a_arr, b_arr = np.asarray(a), np.asarray(b)
    if a_arr.shape != b_arr.shape:
        return ParityResult(
            ParityTier.CROSS_BACKEND, False, f"shape {a_arr.shape} != {b_arr.shape}")
    if np.allclose(a_arr, b_arr, rtol=rtol, atol=atol):
        return ParityResult(ParityTier.CROSS_BACKEND, True, "within tolerance")
    diff = np.max(np.abs(a_arr.astype(np.float64) - b_arr.astype(np.float64)))
    return ParityResult(
        ParityTier.CROSS_BACKEND, False,
        f"max abs diff {diff:.3e} exceeds rtol={rtol} atol={atol}")


def compare_tokens(a, b) -> ParityResult:
    """Committed-token equality. Two arrays can be within tolerance and still
    argmax differently; this is what the product actually promises."""
    a_arr, b_arr = np.asarray(a), np.asarray(b)
    if a_arr.shape != b_arr.shape:
        return ParityResult(
            ParityTier.CROSS_BACKEND, False,
            f"token shape {a_arr.shape} != {b_arr.shape}")
    mismatched = np.nonzero(a_arr != b_arr)[0]
    if mismatched.size == 0:
        return ParityResult(ParityTier.CROSS_BACKEND, True, "tokens identical")
    index = int(mismatched[0])
    return ParityResult(
        ParityTier.CROSS_BACKEND, False,
        f"first token divergence at index {index}: "
        f"{a_arr.flat[index]} != {b_arr.flat[index]}")
