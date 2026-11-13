"""Graph IR v0 — the portable forward-math spine of the Engine Foundation.

Pure dataclasses + JSON (no mlx / numpy / torch). The IR describes the
transformer-decoder forward math only. An optimized backend consumes one
validated lowering; the portable reference interprets the IR directly. See
:mod:`edllm.graph.ir`, :mod:`edllm.graph.emitter`, and
:mod:`edllm.graph.backend_plan`.
"""

from __future__ import annotations

from .backend_plan import (
    DenseLayerPlan,
    DenseTransformerPlan,
    compile_dense_transformer_plan,
)
from .emitter import build_graph_ir
from .ir import (
    IR_VERSION,
    LOGITS_CONVENTIONS,
    NO_MASK_ATTENTION_KINDS,
    OP_TYPES,
    GraphIR,
    LayerSubgraph,
    Op,
    same_mask_kind_family,
    uniform_attention_mask_kind,
)

__all__ = [
    "IR_VERSION",
    "LOGITS_CONVENTIONS",
    "NO_MASK_ATTENTION_KINDS",
    "OP_TYPES",
    "DenseLayerPlan",
    "DenseTransformerPlan",
    "GraphIR",
    "LayerSubgraph",
    "Op",
    "build_graph_ir",
    "compile_dense_transformer_plan",
    "same_mask_kind_family",
    "uniform_attention_mask_kind",
]
