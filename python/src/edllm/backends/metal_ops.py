from __future__ import annotations

from typing import Any

__all__ = ["rms_norm"]


def _load_metal_kernel():
    from mlx.core.fast import metal_kernel

    return metal_kernel


def rms_norm(x: Any, weight: Any, eps: float):
    """RMSNorm implemented as the first real Metal primitive.

    This module is intentionally a primitive proof, not a full Metal backend:
    importing it must not import MLX, and failures to access/compile Metal fail
    closed instead of silently falling back to the MLX implementation.
    """
    try:
        metal_kernel = _load_metal_kernel()
    except Exception as exc:  # pragma: no cover - exercised by monkeypatch.
        raise RuntimeError("Metal RMSNorm unavailable") from exc

    hidden_size = int(weight.shape[0])
    if hidden_size <= 0:
        raise ValueError("rms_norm weight must have a non-empty hidden dimension")
    if int(x.shape[-1]) != hidden_size:
        raise ValueError(
            f"rms_norm hidden dimension mismatch: x.shape[-1]={x.shape[-1]} "
            f"weight.shape[0]={hidden_size}"
        )

    source = """
        uint elem = thread_position_in_grid.x;
        uint row = elem / hidden_size;
        uint col = elem - row * hidden_size;
        uint base = row * hidden_size;

        float sum_sq = 0.0f;
        for (uint i = 0; i < hidden_size; ++i) {
            float value = float(x[base + i]);
            sum_sq += value * value;
        }

        float variance = sum_sq / float(hidden_size);
        float inv_rms = metal::rsqrt(variance + float(eps[0]));
        out[elem] = T(float(x[elem]) * inv_rms * float(weight[col]));
    """
    kernel = metal_kernel(
        name="edllm_rms_norm_m1",
        input_names=["x", "weight", "eps"],
        output_names=["out"],
        source=source,
    )

    mx = __import__("mlx.core", fromlist=["array", "float32"])
    outputs = kernel(
        inputs=[x, weight, mx.array([float(eps)], dtype=mx.float32)],
        template=[("T", x.dtype), ("hidden_size", hidden_size)],
        grid=(int(x.size), 1, 1),
        threadgroup=(min(256, int(x.size)), 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )
    return outputs[0]
