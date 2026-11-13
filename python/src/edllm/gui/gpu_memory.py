"""Whole-device GPU memory, read from the NVIDIA driver's NVML library.

NVML ships with the driver (`nvml.dll` in System32 on Windows), so no package
is needed. The reading is the whole device's, every process included; it is not
this program's allocation and not a peak.
"""
from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass

_NVML_SUCCESS = 0
_NAME_BYTES = 96


@dataclass(frozen=True)
class GpuMemory:
    name: str
    used_bytes: int
    total_bytes: int


class _Memory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


def _load_nvml():
    return ctypes.CDLL("nvml.dll" if os.name == "nt" else "libnvidia-ml.so.1")


def read_gpu_memory(device_index: int = 0) -> GpuMemory | None:
    """The device's name and memory, or None when the driver cannot be read."""
    try:
        nvml = _load_nvml()
    except OSError:
        return None
    try:
        if nvml.nvmlInit_v2() != _NVML_SUCCESS:
            return None
    except (OSError, AttributeError):
        return None
    try:
        handle = ctypes.c_void_p()
        if nvml.nvmlDeviceGetHandleByIndex_v2(
            ctypes.c_uint(device_index), ctypes.byref(handle)
        ) != _NVML_SUCCESS:
            return None
        name = ctypes.create_string_buffer(_NAME_BYTES)
        if nvml.nvmlDeviceGetName(handle, name, ctypes.c_uint(_NAME_BYTES)) != _NVML_SUCCESS:
            return None
        memory = _Memory()
        if nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)) != _NVML_SUCCESS:
            return None
        return GpuMemory(
            name=name.value.decode("utf-8", "replace"),
            used_bytes=int(memory.used),
            total_bytes=int(memory.total),
        )
    except (OSError, AttributeError):
        return None
    finally:
        try:
            nvml.nvmlShutdown()
        except (OSError, AttributeError):
            pass
