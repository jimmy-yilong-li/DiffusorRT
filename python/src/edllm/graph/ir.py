"""Graph IR v0 — a typed, serializable, EXECUTABLE description of the transformer
decoder forward math (model forward only).

This is the portable spine of the Engine Foundation: the unit of cross-backend
parity. It describes "given these token rows, produce these logits" — and nothing
about the diffusion dynamics (mask state, commit/remask policy, KV cache, block
scheduling), which live ABOVE the graph in the Diffusion Runtime Layer.

The top-level graph is::

    embedding -> [N x layer_subgraph] -> final_norm -> lm_head

plus package-level ``logits_convention`` metadata (``dream_ar_shift`` | ``raw``)
applied by the runtime's decode-time selection step, NOT an in-graph op.

The ``layer_subgraph`` op order is exactly the ops the four shipped families need::

    rms_norm(input) -> linear q (+bias?) -> qk_norm? -> rope q -> linear k (+bias?)
    -> qk_norm? -> rope k -> linear v (+bias?) -> gqa_repeat -> attention(mask_kind)
    -> linear o -> residual_add -> rms_norm(post_attn) -> mlp_swiglu(gate,up,down)
    -> residual_add

Family-conditional bits (``apply_qk_norm`` / ``bias_present`` /
``logits_convention``) are DATA carried by op params, never ``if family ==``.

This module is intentionally dependency-light: pure dataclasses + JSON, NO mlx /
numpy / torch. It can be imported anywhere (loader, hot path, tests, a future
Metal backend) without pulling in a backend.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

# The Graph IR schema version. Bump ``minor`` for additive ops/params, ``major``
# for a breaking shape change.
IR_VERSION: Mapping[str, int] = MappingProxyType({"major": 0, "minor": 0})


def _validate_ir_version(value: Mapping[str, Any]) -> None:
    if set(value) != {"major", "minor"}:
        raise ValueError("ir_version must contain exactly major and minor")
    major = value["major"]
    minor = value["minor"]
    if type(major) is not int or type(minor) is not int or minor < 0:
        raise ValueError("ir_version must contain nonnegative integer major and minor")
    if major != IR_VERSION["major"] or minor > IR_VERSION["minor"]:
        raise ValueError(
            f"unsupported ir_version: {major}.{minor} "
            f"(this build supports through "
            f"{IR_VERSION['major']}.{IR_VERSION['minor']})"
        )

# The op set (v0). Anything outside this set is rejected on construction so an
# emitter typo cannot silently produce an un-interpretable graph.
OP_TYPES: frozenset[str] = frozenset(
    {
        "embedding",
        "rms_norm",
        "linear",
        "qk_norm",
        "rope",
        "gqa_repeat",
        "attention",
        "residual_add",
        "mlp_swiglu",
        "lm_head",
    }
)

# GraphIR parameters are executable semantics, not an open metadata bag.  A
# runtime that does not recognize a field cannot know whether ignoring it
# changes the forward, so every v0 op accepts only this declared vocabulary.
OP_PARAM_KEYS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "embedding": frozenset(),
        "rms_norm": frozenset({"eps"}),
        "linear": frozenset({"role", "bias_present"}),
        "qk_norm": frozenset({"role", "apply_qk_norm", "eps", "head_dim"}),
        "rope": frozenset({"role", "head_dim", "rope_theta", "rope_scaling"}),
        "gqa_repeat": frozenset({"kv_repeat", "num_kv_heads", "n_heads"}),
        "attention": frozenset({"mask_kind", "head_dim", "n_heads"}),
        "residual_add": frozenset(),
        "mlp_swiglu": frozenset(),
        "lm_head": frozenset(),
    }
)

# The two package-level logits conventions (decode-time, family-data).
LOGITS_CONVENTIONS: frozenset[str] = frozenset({"dream_ar_shift", "raw"})

# Attention mask kinds that mean "no mask" (every position attends everywhere).
# They are one FAMILY: a kwarg saying "full" against an IR saying "bidirectional"
# is not a disagreement. ``d2f_hybrid`` (block-causal) is its own family.
NO_MASK_ATTENTION_KINDS: frozenset[str] = frozenset({"bidirectional", "full"})

# The v0 canonical per-layer op-TYPE order (Pillar 1). This is the single source
# of truth for the layer contract: the emitter produces exactly this sequence and
# the NumpyRef interpreter ASSERTS it at the start of every layer (so a type-level
# corruption — a reordered or wrong-typed op — fails loudly instead of being
# mis-run positionally). The emitter test imports this same constant.
CANONICAL_LAYER_OP_ORDER: tuple[str, ...] = (
    "rms_norm",      # input_layernorm
    "linear",        # q_proj (+bias?)
    "qk_norm",       # q_norm (present-or-absent)
    "rope",          # rope q
    "linear",        # k_proj (+bias?)
    "qk_norm",       # k_norm (present-or-absent)
    "rope",          # rope k
    "linear",        # v_proj (+bias?)
    "gqa_repeat",    # repeat kv
    "attention",     # bidirectional SDPA
    "linear",        # o_proj
    "residual_add",  # + attn
    "rms_norm",      # post_attention_layernorm
    "mlp_swiglu",    # gate/up/down
    "residual_add",  # + mlp
)


def uniform_attention_mask_kind(graph: "GraphIR") -> str:
    """The single ``mask_kind`` carried by every layer's ``attention`` op.

    The mask FAMILY is Graph IR data (op params); the forward applies ONE kind to
    every layer, so a graph whose layers disagree would be silently mis-run by a
    layer-uniform backend. Fail-closed (the ``_ir_apply_qk_norm`` precedent):
    validate uniformity and raise ``ValueError`` on a non-uniform graph instead.
    An empty / attention-less graph reports the historical default
    (``"bidirectional"``).
    """
    kinds: list[str] = []
    for layer in graph.layers:
        attn_op = next((op for op in layer.ops if op.type == "attention"), None)
        if attn_op is None:
            raise ValueError("layer subgraph carries no attention op")
        kinds.append(str(attn_op.params.get("mask_kind", "bidirectional")))
    if len(set(kinds)) > 1:
        raise ValueError(
            "non-uniform attention mask_kind across layers is unsupported by the "
            f"layer-uniform forward; per-layer kinds={kinds}"
        )
    return kinds[0] if kinds else "bidirectional"


def same_mask_kind_family(a: str, b: str) -> bool:
    """Whether two mask kinds belong to the same mask FAMILY.

    ``bidirectional`` and ``full`` are both "no mask" and interchangeable; any
    other kind (``d2f_hybrid``) only matches itself.
    """
    if a in NO_MASK_ATTENTION_KINDS and b in NO_MASK_ATTENTION_KINDS:
        return True
    return a == b


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class Op:
    """A single forward op.

    Attributes:
        type: One of :data:`OP_TYPES`.
        params: Scalar op parameters (``eps``, ``head_dim``, ``n_heads``,
            ``num_kv_heads``, ``rope_theta``, ``rope_scaling``, ``bias_present``,
            ``apply_qk_norm``, ``mask_kind``, ...). All JSON-serializable.
        weight_keys: The exact ``weights.npz`` keys this op reads, by role, e.g.
            ``{"weight": "model.layers.3.self_attn.q_proj.weight", "bias": ...}``.
            Empty for ops that read no weights (``rope``, ``gqa_repeat``,
            ``attention``, ``residual_add``).
    """

    type: str
    params: Mapping[str, Any] = field(default_factory=dict)
    weight_keys: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.type not in OP_TYPES:
            raise ValueError(f"unknown graph op type: {self.type!r}")
        unknown_params = set(self.params) - OP_PARAM_KEYS[self.type]
        if unknown_params:
            raise ValueError(
                f"unknown graph op parameters for {self.type!r}: "
                f"{sorted(unknown_params)!r}"
            )
        object.__setattr__(self, "params", _freeze_json(self.params))
        object.__setattr__(self, "weight_keys", _freeze_json(self.weight_keys))


@dataclass(frozen=True, slots=True)
class LayerSubgraph:
    """One transformer-decoder layer as an ordered list of :class:`Op`."""

    ops: tuple[Op, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "ops", tuple(self.ops))


@dataclass(frozen=True, slots=True)
class GraphIR:
    """The whole forward graph: embedding -> N layers -> final_norm -> lm_head.

    ``logits_convention`` is package metadata (applied by the runtime's selection
    step), not an in-graph op. ``ir_version`` pins the schema.
    """

    embedding: Op
    layers: tuple[LayerSubgraph, ...]
    final_norm: Op
    lm_head: Op
    logits_convention: str
    ir_version: Mapping[str, int] = field(default_factory=lambda: dict(IR_VERSION))

    def __post_init__(self) -> None:
        if self.logits_convention not in LOGITS_CONVENTIONS:
            raise ValueError(
                f"unknown logits_convention: {self.logits_convention!r}"
            )
        _validate_ir_version(self.ir_version)
        object.__setattr__(self, "ir_version", _freeze_json(self.ir_version))
        object.__setattr__(self, "layers", tuple(self.layers))

    # ------------------------------------------------------------------ #
    # Serialization — round-trippable to/from graph.json.
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return {
            "ir_version": dict(self.ir_version),
            "logits_convention": self.logits_convention,
            "embedding": {
                "type": self.embedding.type,
                "params": _thaw_json(self.embedding.params),
                "weight_keys": _thaw_json(self.embedding.weight_keys),
            },
            "layers": [
                {
                    "ops": [
                        {
                            "type": op.type,
                            "params": _thaw_json(op.params),
                            "weight_keys": _thaw_json(op.weight_keys),
                        }
                        for op in layer.ops
                    ]
                }
                for layer in self.layers
            ],
            "final_norm": {
                "type": self.final_norm.type,
                "params": _thaw_json(self.final_norm.params),
                "weight_keys": _thaw_json(self.final_norm.weight_keys),
            },
            "lm_head": {
                "type": self.lm_head.type,
                "params": _thaw_json(self.lm_head.params),
                "weight_keys": _thaw_json(self.lm_head.weight_keys),
            },
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @staticmethod
    def _op_from_dict(blob: dict[str, Any]) -> Op:
        if type(blob) is not dict or set(blob) != {"type", "params", "weight_keys"}:
            raise ValueError("graph op must contain exactly type, params, weight_keys")
        if type(blob["type"]) is not str:
            raise ValueError("graph op type must be a string")
        if type(blob["params"]) is not dict or type(blob["weight_keys"]) is not dict:
            raise ValueError("graph op params and weight_keys must be objects")
        return Op(
            type=blob["type"],
            params=dict(blob["params"]),
            weight_keys=dict(blob["weight_keys"]),
        )

    @classmethod
    def from_dict(cls, blob: dict[str, Any]) -> "GraphIR":
        graph_fields = {
            "ir_version",
            "logits_convention",
            "embedding",
            "layers",
            "final_norm",
            "lm_head",
        }
        if type(blob) is not dict or set(blob) != graph_fields:
            raise ValueError("graph must contain exactly the v0 GraphIR fields")
        if type(blob["ir_version"]) is not dict:
            raise ValueError("graph ir_version must be an object")
        if type(blob["layers"]) is not list:
            raise ValueError("graph layers must be an array")
        ir_version = dict(blob["ir_version"])
        # A newer minor may add executable parameters.  Older runtimes must not
        # silently ignore them, so consumers accept only versions they implement.
        _validate_ir_version(ir_version)
        layers_list: list[LayerSubgraph] = []
        for layer in blob["layers"]:
            if type(layer) is not dict or set(layer) != {"ops"}:
                raise ValueError("graph layer must contain exactly ops")
            if type(layer["ops"]) is not list:
                raise ValueError("graph layer ops must be an array")
            layers_list.append(
                LayerSubgraph(
                    ops=tuple(cls._op_from_dict(op) for op in layer["ops"])
                )
            )
        layers = tuple(layers_list)
        return cls(
            embedding=cls._op_from_dict(blob["embedding"]),
            layers=layers,
            final_norm=cls._op_from_dict(blob["final_norm"]),
            lm_head=cls._op_from_dict(blob["lm_head"]),
            logits_convention=blob["logits_convention"],
            ir_version=ir_version,
        )

    @classmethod
    def from_json(cls, text: str) -> "GraphIR":
        return cls.from_dict(json.loads(text))
