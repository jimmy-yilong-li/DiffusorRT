"""Per-op kernel dispatch table for the MLX forward (M2-1, the op-dispatch seam).

The forward variants in ``mlx_dream`` dispatch their numerically swappable ops
through a :class:`KernelTable` — a frozen table of callables resolved ONCE at
forward setup — instead of calling ``mlx_ops.*`` directly. That is the seam a
per-op backend (a Metal ``rms_norm``, a quantized matmul) slots into the LIVE
forward through, without touching the engine, the decode loop, or the adapters.

The swappable surface is exactly the op PRIMITIVES:

* ``rms_norm``   — :func:`mlx_ops.rms_norm`
* ``linear``     — :func:`mlx_ops.linear` (the matmul boundary)
* ``rope``       — :func:`mlx_ops.apply_split_half_rope`
* ``swiglu``     — :func:`mlx_ops.swiglu`
* ``attention``  — :func:`mlx_ops.scaled_dot_product_attention`

Deliberately NOT here: the attention mask builders / validators
(``validate_attention_kind`` / the d2f bias tables) — those are fail-closed mask
SEMANTICS, not kernels a backend may reimplement — and
``shift_logits_for_dream``, which is decode-selection convention, not forward
math.

Attention callables consume either raw ``mask_geometry``/``q_positions`` or
the shared builder's ``prepared_mask``. That forward-local value is reused
across layers; selecting a kernel must not rebuild or reinterpret its mask.

Package loading first resolves the selectors into an immutable
:class:`KernelDispatchPlan`.  Product execution receives that plan and never
reinterprets model configuration.  Materialization then binds the named
backends to callables.  Direct low-level callers retain
:func:`resolve_kernel_table` as a compatibility helper.

The ``mlx_ops`` attributes are read at RESOLVE time (late-bound), so the
long-standing test seams that monkeypatch ``mlx_ops.rms_norm`` / ``linear`` /
``shift_logits_for_dream`` on the module keep driving the forward through the
table exactly as they drove the direct calls.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

from . import mlx_ops

# The swappable op names, in forward order. This is the table's full surface;
# the override validation below is keyed on it.
KERNEL_OP_NAMES: tuple[str, ...] = (
    "rms_norm",
    "linear",
    "rope",
    "swiglu",
    "attention",
)

DEFAULT_KERNEL_BACKEND = "mlx"
_COMPUTE_DTYPE_ALIASES = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp32": "float32",
    "float32": "float32",
}


def canonical_compute_dtype_name(name: Any) -> str:
    """Return the device-independent MLX dtype name or fail closed."""

    resolved = _COMPUTE_DTYPE_ALIASES.get(str(name).lower())
    if resolved is None:
        raise ValueError(
            f"unsupported mlx_compute_dtype: {name!r} "
            "(expected 'bf16' or 'fp32')"
        )
    return resolved


@dataclass(frozen=True, slots=True)
class KernelDispatchPlan:
    """Immutable backend selection and the layouts its kernels consume.

    The layout fields are deliberately honest about the current implementation:
    linear weights use ``[out, in]`` and K/V remain compact at the model's
    declared K/V-head count. The pinned MLX fast-SDPA API consumes that grouped
    layout directly; changing either fact requires changing this plan and its
    consumer together. Whether MLX selects its fused Metal kernel remains a
    function of device and shape and is verified by Product admission rather
    than guessed here.
    """

    op_backends: tuple[tuple[str, str], ...]
    compute_dtype: str
    supported_compute_dtypes: tuple[str, ...] = ("float32", "bfloat16")
    linear_weight_layout: str = "out_in"
    attention_layout: str = "sequence_heads_head_dim"
    attention_kernel: str = "mlx_fast_sdpa_api"
    gqa_kv_layout: str = "compact_kv_heads"

    def __post_init__(self) -> None:
        if tuple(name for name, _backend in self.op_backends) != KERNEL_OP_NAMES:
            raise ValueError("kernel dispatch plan must name every op exactly once")
        if self.compute_dtype not in {"float32", "bfloat16"}:
            raise ValueError("kernel dispatch plan compute dtype is unsupported")
        if self.compute_dtype not in self.supported_compute_dtypes:
            raise ValueError("kernel dispatch plan does not support its compute dtype")
        if self.attention_kernel != "mlx_fast_sdpa_api":
            raise ValueError("kernel dispatch plan names an unsupported attention kernel")
        if self.gqa_kv_layout != "compact_kv_heads":
            raise ValueError("kernel dispatch plan names an unsupported GQA K/V layout")

    def backend_for(self, op_name: str) -> str:
        for name, backend in self.op_backends:
            if name == op_name:
                return backend
        raise ValueError(f"kernel dispatch plan has no op {op_name!r}")


@dataclass(frozen=True, slots=True)
class KernelTable:
    """The resolved per-op kernel callables the production forward dispatches."""

    rms_norm: Callable[..., Any]
    linear: Callable[..., Any]
    rope: Callable[..., Any]
    swiglu: Callable[..., Any]
    attention: Callable[..., Any]


def _mlx_kernel_ops() -> dict[str, Callable[..., Any]]:
    """The all-MLX op set, dereferenced from ``mlx_ops`` NOW (late-bound)."""

    return {
        "rms_norm": mlx_ops.rms_norm,
        "linear": mlx_ops.linear,
        "rope": mlx_ops.apply_split_half_rope,
        "swiglu": mlx_ops.swiglu,
        "attention": mlx_ops.scaled_dot_product_attention,
    }


# Registered kernel backends: name -> zero-arg factory of the full op set.
# M2-2+ registers per-op alternatives ("metal", a quantized matmul backend);
# this slice ships MLX only, so the default resolution is byte-identical.
_KERNEL_BACKENDS: dict[str, Callable[[], dict[str, Callable[..., Any]]]] = {
    "mlx": _mlx_kernel_ops,
}


def plan_kernel_dispatch(
    model_config: Mapping[str, Any] | None,
    *,
    compute_dtype: str | None = None,
) -> KernelDispatchPlan:
    """Validate config selectors and freeze the resulting kernel contract."""

    cfg = model_config or {}
    backend = str(cfg.get("kernel_backend", DEFAULT_KERNEL_BACKEND))
    if backend not in _KERNEL_BACKENDS:
        raise ValueError(
            f"unknown kernel_backend {backend!r}; registered backends: "
            f"{sorted(_KERNEL_BACKENDS)}"
        )
    selected = dict.fromkeys(KERNEL_OP_NAMES, backend)

    overrides = cfg.get("kernel_op_overrides") or {}
    if not isinstance(overrides, Mapping):
        raise ValueError(
            "kernel_op_overrides must be a mapping of op name -> backend name; "
            f"got {overrides!r}"
        )
    for op_name, op_backend in overrides.items():
        if op_name not in KERNEL_OP_NAMES:
            raise ValueError(
                f"kernel_op_overrides names an unknown op {op_name!r}; "
                f"swappable ops: {list(KERNEL_OP_NAMES)}"
            )
        if op_backend not in _KERNEL_BACKENDS:
            raise ValueError(
                f"kernel_op_overrides maps {op_name!r} to an unregistered "
                f"backend {op_backend!r}; registered backends: "
                f"{sorted(_KERNEL_BACKENDS)}"
            )
        selected[op_name] = op_backend

    raw_dtype = compute_dtype
    if raw_dtype is None:
        raw_dtype = str(cfg.get("mlx_compute_dtype", "fp32"))
    canonical_dtype = canonical_compute_dtype_name(raw_dtype)
    return KernelDispatchPlan(
        op_backends=tuple((name, selected[name]) for name in KERNEL_OP_NAMES),
        compute_dtype=canonical_dtype,
    )


def materialize_kernel_table(plan: KernelDispatchPlan) -> KernelTable:
    """Bind one validated dispatch plan to the current registered callables."""

    ops: dict[str, Callable[..., Any]] = {}
    for op_name, backend in plan.op_backends:
        factory = _KERNEL_BACKENDS.get(backend)
        if factory is None:
            raise ValueError(
                f"kernel dispatch plan names unregistered backend {backend!r}"
            )
        ops[op_name] = factory()[op_name]

    return KernelTable(**ops)


def resolve_kernel_table(model_config: Mapping[str, Any] | None) -> KernelTable:
    """Compatibility helper: plan and materialize directly from config."""

    return materialize_kernel_table(plan_kernel_dispatch(model_config))
