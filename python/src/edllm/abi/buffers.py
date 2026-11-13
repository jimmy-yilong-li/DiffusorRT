"""Buffers a C consumer can actually use.

Every field answers a question a bare pointer leaves open: byte_length (how
long), device (where), ownership (who frees), strides (is it packed). Buffers
cross as a kinded handle plus an explicit length — never a raw pointer with an
implied size.

v0 scope: HOST buffers. METAL/CUDA enum values are reserved now so adding them
later is not an ABI break; a non-HOST buffer raises today rather than being
silently treated as host memory.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from .handles import HandleKind, HandleRegistry

MAX_NDIM = 4


class DType(IntEnum):
    F32 = 0
    F64 = 1
    BF16 = 2
    I32 = 3
    I64 = 4
    U8 = 5


class Device(IntEnum):
    HOST = 0
    METAL = 1
    CUDA = 2


class Ownership(IntEnum):
    BORROWED = 0            # valid for the call only; callee must not retain
    OWNED_BY_CALLER = 1     # caller frees after the callee returns
    OWNED_BY_CALLEE = 2     # callee frees via diffusorrt_release_handle


_NUMPY_DTYPES = {
    "float32": DType.F32, "float64": DType.F64, "int32": DType.I32,
    "int64": DType.I64, "uint8": DType.U8,
}
_ITEMSIZE = {
    DType.F32: 4, DType.F64: 8, DType.BF16: 2,
    DType.I32: 4, DType.I64: 8, DType.U8: 1,
}


@dataclass(frozen=True, slots=True)
class BufferView:
    handle: int
    byte_length: int
    device: int
    ownership: int


@dataclass(frozen=True, slots=True)
class TensorDesc:
    dtype: int
    ndim: int
    shape: tuple[int, ...]
    strides: tuple[int, ...]   # ELEMENT strides, C order
    buffer: BufferView

    @property
    def element_count(self) -> int:
        count = 1
        for dim in self.shape:
            count *= dim
        return count


def describe_array(array, *, registry: HandleRegistry,
                   device: Device = Device.HOST,
                   ownership: Ownership = Ownership.BORROWED) -> TensorDesc:
    """Describe an array. Fails closed on anything a packed C consumer cannot
    read: unknown dtype, rank above MAX_NDIM, non-contiguous layout."""
    shape = tuple(int(d) for d in array.shape)
    if len(shape) > MAX_NDIM:
        raise ValueError(
            f"ndim {len(shape)} exceeds MAX_NDIM {MAX_NDIM}; the ABI shape array "
            "is fixed-size"
        )
    name = str(getattr(array.dtype, "name", array.dtype))
    # mlx dtypes stringify as "mlx.core.float32"; numpy as "float32".
    name = name.rsplit(".", 1)[-1]
    dtype = DType.BF16 if name == "bfloat16" else _NUMPY_DTYPES.get(name)
    if dtype is None:
        raise ValueError(
            f"dtype {name!r} has no ABI encoding; add it to DType before it can "
            "cross the boundary"
        )
    flags = getattr(array, "flags", None)
    if flags is not None and not flags["C_CONTIGUOUS"]:
        raise ValueError(
            "array is not C-contiguous; a packed C consumer would misread it. "
            "Copy before describing, explicitly, so the copy is visible."
        )
    strides = []
    running = 1
    for dim in reversed(shape):
        strides.append(running)
        running *= dim
    strides.reverse()
    element_count = 1
    for dim in shape:
        element_count *= dim
    return TensorDesc(
        dtype=int(dtype), ndim=len(shape), shape=shape, strides=tuple(strides),
        buffer=BufferView(
            handle=registry.register(array, kind=HandleKind.BUFFER),
            byte_length=element_count * _ITEMSIZE[dtype],
            device=int(device), ownership=int(ownership),
        ),
    )
