"""Declarative model architecture and conversion definitions.

This module is the shared vocabulary between the compiler, GraphIR emitter and
runtime. A checkpoint contributes one exact ModelDefinition and reuses
conversion/runtime components wherever possible; it does not get a separate
loader, forward pipeline, generation loop, CLI or service route.

The module deliberately has no MLX, Torch, Hugging Face, or Textual imports.
Importing the model catalog must stay cheap on a machine that has not installed
any execution backend yet.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Protocol

from .runtime_config import normalize_rope_scaling


def _freeze_definition_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_definition_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_definition_value(item) for item in value)
    return value


_BLOCK_THRESHOLD_DECODE_DEFAULTS: Mapping[str, Any] = _freeze_definition_value({
    "decode_strategy": "block_threshold",
    "block_size": 8,
    "confidence_threshold": 0.9,
    "gamma": 0.1,
})

_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_KNOWN_ATTENTION_MASK_KINDS = frozenset(
    {"bidirectional", "full", "d2f_hybrid"}
)
_KNOWN_DECODE_STRATEGIES = frozenset(
    {
        "entropy_count",
        "confidence_threshold",
        "block_threshold",
        "entropy_budget",
        "d2f_block_pipeline",
        "dilated_unmask",
        "bd3_fixed_quota",
        "fixed_quota",
    }
)


@dataclass(frozen=True, slots=True)
class DecodeContract:
    """Model-owned diffusion policy written into the execution package."""

    name: str
    decode_strategy: str
    attention_mask_kind: str
    block_size: int
    steps: int
    mask_geometry_prompt_len: int
    requires_prompt_block_alignment: bool
    layer0_kv_cache: bool
    d2f_exact_kv_cache: bool
    temperature: float
    top_p: float
    remasking: str

    def validate(self) -> None:
        for name in (
            "name",
            "decode_strategy",
            "attention_mask_kind",
            "remasking",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"decode_contract_{name}_invalid")
        if self.decode_strategy not in _KNOWN_DECODE_STRATEGIES:
            raise ValueError(
                f"decode_contract_strategy_unknown: {self.decode_strategy!r}"
            )
        if self.attention_mask_kind not in _KNOWN_ATTENTION_MASK_KINDS:
            raise ValueError(
                "decode_contract_attention_mask_unknown: "
                f"{self.attention_mask_kind!r}"
            )
        if type(self.block_size) is not int or self.block_size <= 0:
            raise ValueError("decode_contract_block_size_invalid")
        if type(self.steps) is not int or self.steps <= 0:
            raise ValueError("decode_contract_steps_invalid")
        if (
            type(self.mask_geometry_prompt_len) is not int
            or self.mask_geometry_prompt_len < 0
        ):
            raise ValueError("decode_contract_mask_geometry_invalid")
        for name, value in (
            ("requires_prompt_block_alignment", self.requires_prompt_block_alignment),
            ("layer0_kv_cache", self.layer0_kv_cache),
            ("d2f_exact_kv_cache", self.d2f_exact_kv_cache),
        ):
            if type(value) is not bool:
                raise ValueError(f"decode_contract_{name}_invalid")
        if (
            self.decode_strategy != "bd3_fixed_quota"
            or self.attention_mask_kind != "d2f_hybrid"
            or self.mask_geometry_prompt_len != 0
            or self.layer0_kv_cache
            or self.d2f_exact_kv_cache
            or type(self.temperature) not in (int, float)
            or self.temperature != 0.0
            or type(self.top_p) not in (int, float)
            or self.top_p != 1.0
            or self.remasking != "low_confidence"
        ):
            raise ValueError("decode_contract_combination_unsupported")

    def package_model_config(self) -> dict[str, object]:
        self.validate()
        return {
            "block_size": self.block_size,
            "step_budget": self.steps,
            "attention_mask_kind": self.attention_mask_kind,
            "block_schedule": {
                "kind": "absolute-fixed-quota",
                "block_size": self.block_size,
                "mask_geometry_prompt_len": self.mask_geometry_prompt_len,
            },
            "mlx_diffusion_decode_strategy": self.decode_strategy,
            "mlx_diffusion_block_size": self.block_size,
            "mlx_diffusion_steps": self.steps,
            "mlx_diffusion_temperature": self.temperature,
            "mlx_diffusion_top_p": self.top_p,
            "mlx_diffusion_remasking": self.remasking,
            "mlx_layer0_kv_cache_enabled": self.layer0_kv_cache,
            "mlx_diffusion_d2f_exact_kv_cache": self.d2f_exact_kv_cache,
            "bd3_requires_prompt_block_alignment": (
                self.requires_prompt_block_alignment
            ),
        }

    def evidence_policy(self) -> dict[str, object]:
        self.validate()
        return {
            "name": self.name,
            "decode_strategy": self.decode_strategy,
            "attention_mask_kind": self.attention_mask_kind,
            "d2f_exact_kv_cache": self.d2f_exact_kv_cache,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "remasking": self.remasking,
        }


@dataclass(frozen=True, slots=True)
class DeploymentContract:
    """Configuration source selected for one backend/precision target."""

    profile: str
    config_source: str
    attention_mask_kind: str | None = None

    def validate(self) -> None:
        if type(self.profile) is not str or not self.profile:
            raise ValueError("deployment_profile_invalid")
        if (
            type(self.config_source) is not str
            or self.config_source not in {"package", "preset"}
        ):
            raise ValueError("deployment_config_source_invalid")
        if (
            self.attention_mask_kind is not None
            and self.attention_mask_kind not in _KNOWN_ATTENTION_MASK_KINDS
        ):
            raise ValueError("deployment_attention_mask_kind_invalid")

    def validate_attention_mask_kind(self, actual: object) -> None:
        """Require an execution plan to honor this deployment's mask family."""

        from .graph.ir import same_mask_kind_family

        self.validate()
        expected = self.attention_mask_kind
        if expected is None:
            return
        if type(actual) is not str:
            raise ValueError("deployment_attention_mask_kind_mismatch")
        if same_mask_kind_family(expected, actual):
            return
        raise ValueError(
            "deployment_attention_mask_kind_mismatch: "
            f"expected={expected!r} actual={actual!r}"
        )


@dataclass(frozen=True, slots=True)
class DeploymentTarget:
    """A usable deployment choice, independent of admission evidence."""

    backend: str
    precision: str
    minimum_device: str
    deployment: DeploymentContract | None = None

    def validate(self) -> None:
        for name in ("backend", "precision", "minimum_device"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"deployment_target_{name}_invalid")
        if self.deployment is not None and not isinstance(
            self.deployment, DeploymentContract
        ):
            raise ValueError("deployment_target_contract_invalid")
        if self.deployment is not None:
            self.deployment.validate()


@dataclass(frozen=True, slots=True)
class ModelAdapter:
    """Resolved decode/tokenization policy for one execution package.

    Structural values remain as compatibility metadata for the portable NumPy
    capability gate. Product forwards consume those values from emitted
    :class:`GraphIR`; this object cannot override a loaded graph.
    """

    architecture_family: str
    uses_shifted_logits: bool
    mask_token_id: int
    uses_qk_norm: bool = False
    decode_defaults: Mapping[str, Any] = field(default_factory=dict)
    chat_template_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # ``frozen=True`` alone does not freeze nested dicts. Runtime plans are
        # shared for the lifetime of a loaded model, so make their policy data
        # genuinely immutable rather than relying on caller discipline.
        object.__setattr__(
            self,
            "decode_defaults",
            _freeze_definition_value(self.decode_defaults),
        )
        object.__setattr__(
            self,
            "chat_template_kwargs",
            _freeze_definition_value(self.chat_template_kwargs),
        )

_BD3_DECODE = DecodeContract(
    name="bd3-aligned-no-cache-candidate",
    decode_strategy="bd3_fixed_quota",
    attention_mask_kind="d2f_hybrid",
    block_size=32,
    steps=128,
    mask_geometry_prompt_len=0,
    requires_prompt_block_alignment=True,
    layer0_kv_cache=False,
    d2f_exact_kv_cache=False,
    temperature=0.0,
    top_p=1.0,
    remasking="low_confidence",
)


@dataclass(frozen=True, slots=True)
class DecodePolicy:
    """Decode semantics composed with a checkpoint independently of its forward.

    ``defaults`` covers ordinary shared strategies. ``contract`` is present only
    when a package must carry an exact, stricter policy such as BD3 fixed quota.
    The policy is data: it cannot create a model-specific generation loop.
    """

    key: str
    defaults: Mapping[str, Any]
    contract: DecodeContract | None = None

    def __post_init__(self) -> None:
        if type(self.key) is not str or not self.key:
            raise ValueError("decode_policy_key_invalid")
        if not isinstance(self.defaults, Mapping):
            raise TypeError("decode_policy_defaults_invalid")
        object.__setattr__(
            self,
            "defaults",
            _freeze_definition_value(self.defaults),
        )
        if self.contract is not None:
            if not isinstance(self.contract, DecodeContract):
                raise TypeError("decode_policy_contract_invalid")
            self.contract.validate()

    def resolve(self, config: Mapping[str, Any]) -> dict[str, Any]:
        """Resolve defaults while proving an exact package contract when present."""

        if self.contract is None:
            return dict(self.defaults)
        contract = self.contract
        strategy = config.get("mlx_diffusion_decode_strategy")
        if strategy != contract.decode_strategy:
            raise ValueError(
                f"decode policy {self.key!r} requires "
                "mlx_diffusion_decode_strategy="
                f"{contract.decode_strategy!r}"
            )
        block_size = config.get("mlx_diffusion_block_size")
        if block_size != contract.block_size:
            raise ValueError(
                f"decode policy {self.key!r} requires mlx_diffusion_block_size="
                f"{contract.block_size}"
            )
        if (
            config.get("bd3_requires_prompt_block_alignment")
            is not contract.requires_prompt_block_alignment
        ):
            raise ValueError(
                f"decode policy {self.key!r} requires "
                "bd3_requires_prompt_block_alignment="
                f"{contract.requires_prompt_block_alignment!r}"
            )
        return {
            "decode_strategy": contract.decode_strategy,
            "block_size": contract.block_size,
            "steps": contract.steps,
            "temperature": contract.temperature,
            "top_p": contract.top_p,
            "remasking": contract.remasking,
            "layer0_kv_cache_enabled": contract.layer0_kv_cache,
            "confidence_threshold": float(
                config.get("mlx_diffusion_confidence_threshold", 0.9)
            ),
            "gamma": float(config.get("mlx_diffusion_gamma", 0.1)),
        }


DECODE_POLICIES: Mapping[str, DecodePolicy] = MappingProxyType({
    "bidirectional_fixed_quota_v1": DecodePolicy(
        key="bidirectional_fixed_quota_v1",
        defaults={
            "decode_strategy": "fixed_quota", "block_size": 64, "steps": 256,
            "temperature": 0.0, "top_p": 1.0, "remasking": "low_confidence",
            "layer0_kv_cache_enabled": False,
        },
    ),
    "block_threshold_v1": DecodePolicy(
        key="block_threshold_v1",
        defaults=_BLOCK_THRESHOLD_DECODE_DEFAULTS,
    ),
    "bd3_fixed_quota_v1": DecodePolicy(
        key="bd3_fixed_quota_v1",
        defaults={},
        contract=_BD3_DECODE,
    ),
    "fixed_quota_ragged_v1": DecodePolicy(
        key="fixed_quota_ragged_v1",
        defaults={},
        contract=replace(
            _BD3_DECODE,
            name="absolute-fixed-quota-ragged-no-cache",
            requires_prompt_block_alignment=False,
        ),
    ),
})


@dataclass(frozen=True, slots=True)
class RuntimeArchitectureSpec:
    """One reusable forward/tokenization contract, selected by exact aliases."""

    key: str
    aliases: tuple[str, ...]
    uses_shifted_logits: bool
    uses_qk_norm: bool
    qkv_bias_present: bool
    chat_template_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "aliases", tuple(self.aliases))
        object.__setattr__(
            self,
            "chat_template_kwargs",
            _freeze_definition_value(self.chat_template_kwargs),
        )

    def build(
        self,
        config: Mapping[str, Any],
        mask_token_id: int,
        *,
        decode_policy: DecodePolicy,
    ) -> ModelAdapter:
        if type(mask_token_id) is not int or mask_token_id < 0:
            raise ValueError("mask_token_id must be a non-negative integer")
        if not isinstance(decode_policy, DecodePolicy):
            raise TypeError("decode_policy must be a DecodePolicy")
        return ModelAdapter(
            architecture_family=self.key,
            uses_shifted_logits=self.uses_shifted_logits,
            mask_token_id=mask_token_id,
            uses_qk_norm=self.uses_qk_norm,
            decode_defaults=decode_policy.resolve(config),
            chat_template_kwargs=dict(self.chat_template_kwargs),
        )


RUNTIME_ARCHITECTURES: Mapping[str, RuntimeArchitectureSpec] = MappingProxyType({
    "qwen2_dream": RuntimeArchitectureSpec(
        key="qwen2_dream",
        aliases=("DreamModel", "qwen2_dream", "dream", "qwen2", "dream_stub"),
        uses_shifted_logits=True,
        uses_qk_norm=False,
        qkv_bias_present=True,
    ),
    "qwen3_mdlm": RuntimeArchitectureSpec(
        key="qwen3_mdlm",
        aliases=("A2DQwen3", "a2d-qwen3", "qwen3_mdlm"),
        uses_shifted_logits=False,
        uses_qk_norm=True,
        qkv_bias_present=False,
        chat_template_kwargs={"enable_thinking": False},
    ),
    "qwen3_bd3lm": RuntimeArchitectureSpec(
        key="qwen3_bd3lm",
        aliases=("A2DQwen3BD3LM", "qwen3_bd3lm"),
        uses_shifted_logits=False,
        uses_qk_norm=True,
        qkv_bias_present=False,
        chat_template_kwargs={"enable_thinking": False},
    ),
    "llada": RuntimeArchitectureSpec(
        key="llada",
        aliases=("LLaDAModelLM", "llada"),
        uses_shifted_logits=False,
        uses_qk_norm=False,
        qkv_bias_present=False,
    ),
})


# Legacy packages identify only the runtime architecture. Keep their decode
# choice in a separate table so forward structure does not own generation
# policy. New Local packages bind a DecodePolicy in ModelDefinition instead.
_LEGACY_DECODE_POLICY_BY_RUNTIME: Mapping[str, DecodePolicy] = MappingProxyType({
    "qwen2_dream": DECODE_POLICIES["block_threshold_v1"],
    "qwen3_mdlm": DECODE_POLICIES["bidirectional_fixed_quota_v1"],
    "qwen3_bd3lm": DECODE_POLICIES["bd3_fixed_quota_v1"],
    "llada": DECODE_POLICIES["block_threshold_v1"],
})


def _runtime_aliases() -> dict[str, RuntimeArchitectureSpec]:
    aliases: dict[str, RuntimeArchitectureSpec] = {}
    for spec in RUNTIME_ARCHITECTURES.values():
        if spec.key not in spec.aliases:
            raise RuntimeError(f"runtime spec {spec.key!r} does not name itself")
        for alias in spec.aliases:
            normalized = alias.lower()
            if normalized in aliases:
                raise RuntimeError(f"duplicate runtime architecture alias: {alias}")
            aliases[normalized] = spec
    return aliases


_RUNTIME_ALIASES: Mapping[str, RuntimeArchitectureSpec] = MappingProxyType(
    _runtime_aliases()
)


def runtime_architecture_for_key(key: str) -> RuntimeArchitectureSpec:
    try:
        return RUNTIME_ARCHITECTURES[key]
    except KeyError:
        raise ValueError(f"unsupported_architecture_family: {key.lower()}") from None


def runtime_architecture_for_config(
    config: Mapping[str, Any],
) -> RuntimeArchitectureSpec:
    """Resolve exact architecture and family fields, rejecting disagreement."""

    architecture = str(config.get("architecture", "") or "")
    family = str(config.get("architecture_family", "") or "")

    architecture_spec = None
    if architecture:
        architecture_spec = _RUNTIME_ALIASES.get(architecture.lower())
        if architecture_spec is None:
            raise ValueError(f"unsupported_architecture_family: {architecture.lower()}")

    family_spec = None
    if family:
        family_spec = _RUNTIME_ALIASES.get(family.lower())
        if family_spec is None:
            raise ValueError(f"unsupported_architecture_family: {family.lower()}")

    if (
        architecture_spec is not None
        and family_spec is not None
        and architecture_spec is not family_spec
    ):
        raise ValueError(
            "architecture_family_conflict: "
            f"architecture={architecture!r} family={family!r}"
        )
    spec = architecture_spec or family_spec
    if spec is None:
        raise ValueError("unsupported_architecture_family: ")
    return spec


def build_runtime_adapter(
    config: Mapping[str, Any], mask_token_id: int
) -> ModelAdapter:
    runtime = runtime_architecture_for_config(config)
    return runtime.build(
        config,
        mask_token_id,
        decode_policy=_LEGACY_DECODE_POLICY_BY_RUNTIME[runtime.key],
    )


def default_decode_policy_for_runtime(runtime_key: str) -> DecodePolicy:
    """Compatibility default for a Local definition using a known runtime."""

    try:
        return _LEGACY_DECODE_POLICY_BY_RUNTIME[runtime_key]
    except KeyError:
        raise ValueError(
            f"unsupported_architecture_family: {runtime_key.lower()}"
        ) from None


class TokenizerLayout(StrEnum):
    VERBATIM = "verbatim"
    DREAM_QWEN2_BUILTIN = "dream_qwen2_builtin"


TensorNameMapper = Callable[[str], str]


class ModelConfigMapper(Protocol):
    """Pure upstream-config to canonical package-config Adapter."""

    def __call__(
        self,
        config: Mapping[str, Any],
        *,
        mask_token_id: int,
        runtime_architecture: str,
        tie_word_embeddings: bool,
    ) -> dict[str, Any]: ...


def _first_int(value: object, *, default: int) -> int:
    if isinstance(value, int):
        return value
    if (
        isinstance(value, (list, tuple))
        and value
        and isinstance(value[0], int)
    ):
        return value[0]
    return default


def _context_limit_config(
    config: Mapping[str, Any], source_field: str
) -> dict[str, int]:
    """Preserve the source sequence ceiling without guessing a missing limit."""
    if source_field not in config:
        return {}
    limit = config[source_field]
    if type(limit) is not int or limit <= 0:
        raise ValueError(f"{source_field} must be a positive integer")
    return {"max_position_embeddings": limit}


def build_qwen_model_config(
    config: Mapping[str, Any],
    *,
    mask_token_id: int,
    runtime_architecture: str,
    tie_word_embeddings: bool,
) -> dict[str, Any]:
    """Normalize the shared Dream/Qwen upstream geometry."""

    hidden_size = int(config["hidden_size"])
    attention_heads = int(config["num_attention_heads"])
    head_dim = int(config.get("head_dim") or (hidden_size // attention_heads))
    num_kv_heads = int(config.get("num_key_value_heads", attention_heads))
    layer_count = int(config["num_hidden_layers"])
    return {
        "architecture": runtime_architecture,
        "layer_count": layer_count,
        "num_hidden_layers": layer_count,
        "hidden_size": hidden_size,
        "attention_heads": attention_heads,
        "num_key_value_heads": num_kv_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "intermediate_size": int(config["intermediate_size"]),
        "vocab_size": int(config["vocab_size"]),
        **_context_limit_config(config, "max_position_embeddings"),
        "rope_theta": float(config.get("rope_theta", 1000000.0)),
        "rope_scaling": normalize_rope_scaling(config.get("rope_scaling")),
        "rms_norm_eps": float(config.get("rms_norm_eps", 1e-6)),
        "mask_token_id": int(mask_token_id),
        "tie_word_embeddings": tie_word_embeddings,
        "bos_token_id": int(config.get("bos_token_id", 0)),
        "eos_token_id": int(_first_int(config.get("eos_token_id"), default=0)),
        "pad_token_id": int(config.get("pad_token_id", 0)),
        "block_size": 32,
        "step_budget": 4,
        "attention_mask_kind": "bidirectional",
        "block_schedule": {"kind": "fixed", "block_size": 32},
    }


def build_llada_model_config(
    config: Mapping[str, Any],
    *,
    mask_token_id: int,
    runtime_architecture: str,
    tie_word_embeddings: bool,
) -> dict[str, Any]:
    """Normalize the LLaDA/OLMo upstream geometry."""

    hidden_size = int(config["d_model"])
    attention_heads = int(config["n_heads"])
    num_kv_heads = int(config.get("n_kv_heads", attention_heads))
    head_dim = int(config.get("head_dim") or (hidden_size // attention_heads))
    layer_count = int(config["n_layers"])
    return {
        "architecture": runtime_architecture,
        "layer_count": layer_count,
        "num_hidden_layers": layer_count,
        "hidden_size": hidden_size,
        "attention_heads": attention_heads,
        "num_key_value_heads": num_kv_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "intermediate_size": int(config["mlp_hidden_size"]),
        "vocab_size": int(config.get("vocab_size") or config["embedding_size"]),
        **_context_limit_config(config, "max_sequence_length"),
        "rope_theta": float(config.get("rope_theta", 500000.0)),
        "rope_scaling": {"rope_type": "linear", "factor": 1.0},
        "rms_norm_eps": float(config.get("rms_norm_eps", 1e-5)),
        "mask_token_id": int(mask_token_id),
        "tie_word_embeddings": tie_word_embeddings,
        "bos_token_id": int(_first_int(config.get("bos_token_id"), default=0)),
        "eos_token_id": int(_first_int(config.get("eos_token_id"), default=0)),
        "pad_token_id": int(_first_int(config.get("pad_token_id"), default=0)),
        "block_size": 32,
        "step_budget": 4,
        "attention_mask_kind": "bidirectional",
        "block_schedule": {"kind": "fixed", "block_size": 32},
    }


def _identity_tensor_name(source_key: str) -> str:
    return source_key


_LLADA_BLOCK_SUFFIX_MAP: Mapping[str, str] = MappingProxyType({
    "q_proj.weight": "self_attn.q_proj.weight",
    "k_proj.weight": "self_attn.k_proj.weight",
    "v_proj.weight": "self_attn.v_proj.weight",
    "attn_out.weight": "self_attn.o_proj.weight",
    "ff_proj.weight": "mlp.gate_proj.weight",
    "up_proj.weight": "mlp.up_proj.weight",
    "ff_out.weight": "mlp.down_proj.weight",
    "attn_norm.weight": "input_layernorm.weight",
    "ff_norm.weight": "post_attention_layernorm.weight",
})

_LLADA_TOP_LEVEL_MAP: Mapping[str, str] = MappingProxyType({
    "model.transformer.wte.weight": "model.embed_tokens.weight",
    "model.transformer.ff_out.weight": "lm_head.weight",
    "model.transformer.ln_f.weight": "model.norm.weight",
})


def map_llada_tensor_name(source_key: str) -> str:
    """Map one LLaDA OLMo tensor name to the canonical GraphIR namespace."""

    mapped_top = _LLADA_TOP_LEVEL_MAP.get(source_key)
    if mapped_top is not None:
        return mapped_top
    prefix = "model.transformer.blocks."
    if source_key.startswith(prefix):
        remainder = source_key[len(prefix) :]
        layer_text, separator, suffix = remainder.partition(".")
        mapped_suffix = _LLADA_BLOCK_SUFFIX_MAP.get(suffix)
        if separator and mapped_suffix is not None and layer_text.isdigit():
            return f"model.layers.{int(layer_text)}.{mapped_suffix}"
    raise ValueError(f"unmapped LLaDA tensor name: {source_key!r}")


@dataclass(frozen=True, slots=True)
class ConverterAdapter:
    """Declarative translation from one upstream layout into canonical GraphIR."""

    key: str
    upstream_architectures: tuple[str, ...]
    upstream_model_types: tuple[str, ...]
    runtime_architecture: str
    runtime_adapter: str
    tensor_name_mapper: TensorNameMapper
    model_config_mapper: ModelConfigMapper
    upstream_tying_field: str = "tie_word_embeddings"
    required_tie_word_embeddings: bool | None = None
    qwen_chat_template_fallback: bool = False
    tokenizer_layout: TokenizerLayout = TokenizerLayout.VERBATIM
    tied_weight_keys: tuple[str, ...] = (
        "model.embed_tokens.weight",
        "lm_head.weight",
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "upstream_architectures",
            tuple(self.upstream_architectures),
        )
        object.__setattr__(
            self,
            "upstream_model_types",
            tuple(self.upstream_model_types),
        )
        object.__setattr__(self, "tied_weight_keys", tuple(self.tied_weight_keys))

    def validate_config(self, config: Mapping[str, Any]) -> None:
        architectures = config.get("architectures")
        actual = (
            tuple(str(value) for value in architectures)
            if isinstance(architectures, list)
            else ()
        )
        if actual != self.upstream_architectures:
            raise ValueError(
                f"converter_architecture_mismatch: {self.key}: "
                f"expected={self.upstream_architectures!r} actual={actual!r}"
            )
        model_type = str(config.get("model_type", ""))
        if model_type not in self.upstream_model_types:
            raise ValueError(
                f"converter_model_type_mismatch: {self.key}: "
                f"expected={self.upstream_model_types!r} actual={model_type!r}"
            )

    def build_model_config(
        self,
        config: Mapping[str, Any],
        *,
        mask_token_id: int,
        tie_word_embeddings: bool,
    ) -> dict[str, Any]:
        return self.model_config_mapper(
            config,
            mask_token_id=mask_token_id,
            runtime_architecture=self.runtime_architecture,
            tie_word_embeddings=tie_word_embeddings,
        )


CONVERTER_ADAPTERS: Mapping[str, ConverterAdapter] = MappingProxyType({
    "qwen3_mdlm_verbatim_v1": ConverterAdapter(
        key="qwen3_mdlm_verbatim_v1",
        upstream_architectures=("A2DQwen3LMHeadModel",),
        upstream_model_types=("a2d-qwen3",),
        runtime_architecture="A2DQwen3",
        runtime_adapter="qwen3_mdlm",
        tensor_name_mapper=_identity_tensor_name,
        model_config_mapper=build_qwen_model_config,
        qwen_chat_template_fallback=True,
    ),
    "qwen3_bd3lm_verbatim_v1": ConverterAdapter(
        key="qwen3_bd3lm_verbatim_v1",
        upstream_architectures=("A2DQwen3LMHeadModel",),
        upstream_model_types=("a2d-qwen3",),
        runtime_architecture="A2DQwen3BD3LM",
        runtime_adapter="qwen3_bd3lm",
        tensor_name_mapper=_identity_tensor_name,
        model_config_mapper=build_qwen_model_config,
        qwen_chat_template_fallback=True,
    ),
    "dream_verbatim_v1": ConverterAdapter(
        key="dream_verbatim_v1",
        upstream_architectures=("DreamModel",),
        upstream_model_types=("Dream",),
        runtime_architecture="DreamModel",
        runtime_adapter="qwen2_dream",
        tensor_name_mapper=_identity_tensor_name,
        model_config_mapper=build_qwen_model_config,
    ),
    "dream_coder_verbatim_v1": ConverterAdapter(
        key="dream_coder_verbatim_v1",
        upstream_architectures=("DreamModel",),
        upstream_model_types=("Dream",),
        runtime_architecture="DreamModel",
        runtime_adapter="qwen2_dream",
        tensor_name_mapper=_identity_tensor_name,
        model_config_mapper=build_qwen_model_config,
        tokenizer_layout=TokenizerLayout.DREAM_QWEN2_BUILTIN,
    ),
    "llada_olmo_remap_v1": ConverterAdapter(
        key="llada_olmo_remap_v1",
        upstream_architectures=("LLaDAModelLM",),
        upstream_model_types=("llada",),
        runtime_architecture="LLaDAModelLM",
        runtime_adapter="llada",
        tensor_name_mapper=map_llada_tensor_name,
        model_config_mapper=build_llada_model_config,
        upstream_tying_field="weight_tying",
        required_tie_word_embeddings=False,
        tied_weight_keys=(),
    ),
})


@dataclass(frozen=True, slots=True)
class ArchitectureDefinition:
    """Reusable model structure and source-to-GraphIR translation.

    This value never contains a checkpoint id, a device requirement, or Product
    admission evidence.  A compatible checkpoint can therefore reuse it without
    inheriting another checkpoint's tokenizer or deployment policy.
    """

    key: str
    model_family: str
    converter: ConverterAdapter
    runtime: RuntimeArchitectureSpec
    graph_ir_family: str
    features: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("key", "model_family", "graph_ir_family"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"architecture_definition_{name}_invalid")
        if not isinstance(self.converter, ConverterAdapter):
            raise TypeError("architecture_definition_converter_invalid")
        if not isinstance(self.runtime, RuntimeArchitectureSpec):
            raise TypeError("architecture_definition_runtime_invalid")
        if not isinstance(self.features, (list, tuple)) or any(
            type(feature) is not str or not feature for feature in self.features
        ):
            raise ValueError("architecture_definition_features_invalid")
        object.__setattr__(self, "features", tuple(self.features))
        if self.converter.runtime_adapter != self.runtime.key:
            raise RuntimeError("architecture definition converter/runtime mismatch")
        if self.converter.runtime_architecture not in self.runtime.aliases:
            raise RuntimeError("architecture definition package/runtime mismatch")


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    """Exact source identity and tokenizer facts for one checkpoint."""

    key: str
    model_id: str
    revision: str
    source_license: str
    custom_tokenizer_module: str | None
    mask_token_id: int
    tie_word_embeddings: bool

    def __post_init__(self) -> None:
        for name in ("key", "model_id", "revision", "source_license"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"checkpoint_metadata_{name}_invalid")
        if not _SHA40.fullmatch(self.revision):
            raise ValueError("checkpoint_metadata_revision_invalid")
        if self.custom_tokenizer_module is not None and (
            type(self.custom_tokenizer_module) is not str
            or not self.custom_tokenizer_module
        ):
            raise ValueError("checkpoint_metadata_tokenizer_module_invalid")
        if type(self.mask_token_id) is not int or self.mask_token_id < 0:
            raise ValueError("checkpoint_metadata_mask_token_id_invalid")
        if type(self.tie_word_embeddings) is not bool:
            raise ValueError("checkpoint_metadata_tying_invalid")


@dataclass(frozen=True, slots=True)
class DeploymentPolicy:
    """Operational choices for a checkpoint, separate from certification."""

    targets: tuple[DeploymentTarget, ...]
    health_prompt: str = "DiffusorRT health check."
    health_max_tokens: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.targets, (list, tuple)) or not self.targets:
            raise ValueError("deployment_policy_targets_invalid")
        if any(not isinstance(target, DeploymentTarget) for target in self.targets):
            raise TypeError("deployment_policy_target_invalid")
        object.__setattr__(self, "targets", tuple(self.targets))
        if type(self.health_prompt) is not str or not self.health_prompt:
            raise ValueError("deployment_policy_health_prompt_invalid")
        if type(self.health_max_tokens) is not int or self.health_max_tokens <= 0:
            raise ValueError("deployment_policy_health_max_tokens_invalid")
        seen: set[tuple[str, str]] = set()
        for target in self.targets:
            target.validate()
            identity = (target.backend, target.precision)
            if identity in seen:
                raise RuntimeError("duplicate model deployment target")
            seen.add(identity)


@dataclass(frozen=True, slots=True)
class ModelDefinition:
    """Composition of checkpoint, architecture, and deployment contracts.

    The compatibility properties keep callers simple, while the stored state
    has one owner for each concern.  Coverage and admission records are not part
    of this value and cannot steer runtime behavior.
    """

    checkpoint: CheckpointMetadata
    architecture: ArchitectureDefinition
    deployment_policy: DeploymentPolicy
    decode_policy: DecodePolicy

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint, CheckpointMetadata):
            raise TypeError("model_definition_checkpoint_invalid")
        if not isinstance(self.architecture, ArchitectureDefinition):
            raise TypeError("model_definition_architecture_invalid")
        if not isinstance(self.deployment_policy, DeploymentPolicy):
            raise TypeError("model_definition_deployment_policy_invalid")
        if not isinstance(self.decode_policy, DecodePolicy):
            raise TypeError("model_definition_decode_policy_invalid")
        if self.decode_policy.contract is not None and self.runtime.uses_shifted_logits:
            raise ValueError(
                "model_definition_decode_architecture_mismatch: "
                "this fixed-quota contract supports only unshifted logits"
            )
        required_tying = self.converter.required_tie_word_embeddings
        if required_tying is not None and required_tying is not self.tie_word_embeddings:
            raise RuntimeError("model definition converter/tying mismatch")

    @property
    def key(self) -> str:
        return self.checkpoint.key

    @property
    def model_id(self) -> str:
        return self.checkpoint.model_id

    @property
    def revision(self) -> str:
        return self.checkpoint.revision

    @property
    def source_license(self) -> str:
        return self.checkpoint.source_license

    @property
    def custom_tokenizer_module(self) -> str | None:
        return self.checkpoint.custom_tokenizer_module

    @property
    def mask_token_id(self) -> int:
        return self.checkpoint.mask_token_id

    @property
    def tie_word_embeddings(self) -> bool:
        return self.checkpoint.tie_word_embeddings

    @property
    def model_family(self) -> str:
        return self.architecture.model_family

    @property
    def converter(self) -> ConverterAdapter:
        return self.architecture.converter

    @property
    def runtime(self) -> RuntimeArchitectureSpec:
        return self.architecture.runtime

    @property
    def graph_ir_family(self) -> str:
        return self.architecture.graph_ir_family

    @property
    def architecture_features(self) -> tuple[str, ...]:
        return self.architecture.features

    @property
    def decode_contract(self) -> DecodeContract | None:
        return self.decode_policy.contract

    @property
    def deployments(self) -> tuple[DeploymentTarget, ...]:
        return self.deployment_policy.targets

    @property
    def expected_architectures(self) -> tuple[str, ...]:
        return self.converter.upstream_architectures

    @property
    def expected_model_types(self) -> tuple[str, ...]:
        return self.converter.upstream_model_types

    @property
    def runtime_architecture(self) -> str:
        return self.converter.runtime_architecture

    @property
    def runtime_adapter(self) -> str:
        return self.runtime.key

    @property
    def upstream_tying_field(self) -> str:
        return self.converter.upstream_tying_field

    def resolve_deployment(self, backend: str, precision: str) -> DeploymentTarget:
        for target in self.deployments:
            if target.backend == backend and target.precision == precision:
                return target
        raise ValueError(
            f"unsupported_backend_precision: {self.key}:{backend}:{precision}"
        )

    def build_runtime_adapter(
        self,
        config: Mapping[str, Any],
        mask_token_id: int,
    ) -> ModelAdapter:
        """Resolve runtime policy from this one composed model definition."""

        return self.runtime.build(
            config,
            mask_token_id,
            decode_policy=self.decode_policy,
        )


ARCHITECTURE_DEFINITIONS: Mapping[str, ArchitectureDefinition] = MappingProxyType({
    "dream": ArchitectureDefinition(
        key="dream",
        model_family="dream",
        converter=CONVERTER_ADAPTERS["dream_verbatim_v1"],
        runtime=RUNTIME_ARCHITECTURES["qwen2_dream"],
        graph_ir_family="dense_qwen2_bidirectional_v1",
        features=("shifted_logits", "attention_bias", "untied_lm_head"),
    ),
    "dream_coder": ArchitectureDefinition(
        key="dream_coder",
        model_family="dream_coder",
        converter=CONVERTER_ADAPTERS["dream_coder_verbatim_v1"],
        runtime=RUNTIME_ARCHITECTURES["qwen2_dream"],
        graph_ir_family="dense_qwen2_bidirectional_v1",
        features=("shifted_logits", "attention_bias", "gqa_28_4", "untied_lm_head"),
    ),
    "qwen3_mdlm": ArchitectureDefinition(
        key="qwen3_mdlm",
        model_family="qwen3_mdlm",
        converter=CONVERTER_ADAPTERS["qwen3_mdlm_verbatim_v1"],
        runtime=RUNTIME_ARCHITECTURES["qwen3_mdlm"],
        graph_ir_family="dense_qwen3_bidirectional_qk_norm_gqa_v1",
        features=(
            "unshifted_logits",
            "qk_norm",
            "gqa_16_8",
            "explicit_head_dim_128",
            "tied_embeddings_declared",
        ),
    ),
    "qwen3_bd3lm": ArchitectureDefinition(
        key="qwen3_bd3lm",
        model_family="qwen3_bd3lm",
        converter=CONVERTER_ADAPTERS["qwen3_bd3lm_verbatim_v1"],
        runtime=RUNTIME_ARCHITECTURES["qwen3_bd3lm"],
        graph_ir_family="dense_qwen3_absolute_block_causal_v1",
        features=(
            "unshifted_logits",
            "qk_norm",
            "gqa_16_8",
            "explicit_head_dim_128",
            "absolute_block_causal_attention",
            "fixed_quota_block_diffusion",
            "zero_transfer_noop_elision",
            "tied_embeddings_declared",
        ),
    ),
    "llada": ArchitectureDefinition(
        key="llada",
        model_family="llada",
        converter=CONVERTER_ADAPTERS["llada_olmo_remap_v1"],
        runtime=RUNTIME_ARCHITECTURES["llada"],
        graph_ir_family="dense_llama_bidirectional_v1",
        features=(
            "unshifted_logits",
            "full_mha_32_32",
            "no_qk_norm",
            "no_attention_bias",
            "untied_lm_head",
        ),
    ),
})


def _mlx_policy(
    minimum_device: str,
    *,
    health_prompt: str,
    deployment: DeploymentContract | None = None,
) -> DeploymentPolicy:
    return DeploymentPolicy(
        targets=(
            DeploymentTarget(
                backend="mlx_local",
                precision="fp32",
                minimum_device=minimum_device,
                deployment=deployment,
            ),
        ),
        health_prompt=health_prompt,
    )


MODEL_DEFINITIONS: Mapping[str, ModelDefinition] = MappingProxyType({
    "dream_v0_instruct_7b": ModelDefinition(
        checkpoint=CheckpointMetadata(
            key="dream_v0_instruct_7b",
            model_id="Dream-org/Dream-v0-Instruct-7B",
            revision="05334cb9faaf763692dcf9d8737c642be2b2a6ae",
            source_license="apache-2.0",
            custom_tokenizer_module="tokenization_dream.py",
            mask_token_id=151666,
            tie_word_embeddings=False,
        ),
        architecture=ARCHITECTURE_DEFINITIONS["dream"],
        deployment_policy=_mlx_policy(
            "darwin_arm64_apple_silicon",
            health_prompt="The capital of France is",
        ),
        decode_policy=DECODE_POLICIES["block_threshold_v1"],
    ),
    "qwen3_06b_mdlm_v0_1": ModelDefinition(
        checkpoint=CheckpointMetadata(
            key="qwen3_06b_mdlm_v0_1",
            model_id="dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1",
            revision="c8d24a3f4adaeef46881b450e1bf7d1005203bd7",
            source_license="apache-2.0",
            custom_tokenizer_module=None,
            mask_token_id=151669,
            tie_word_embeddings=True,
        ),
        architecture=ARCHITECTURE_DEFINITIONS["qwen3_mdlm"],
        deployment_policy=_mlx_policy(
            "darwin_arm64_apple_m4_max_64gb",
            health_prompt="Hello, who are you?",
            deployment=DeploymentContract(
                profile="baseline",
                config_source="preset",
                attention_mask_kind="bidirectional",
            ),
        ),
        decode_policy=DECODE_POLICIES["bidirectional_fixed_quota_v1"],
    ),
    "qwen3_06b_bd3lm_v0_1": ModelDefinition(
        checkpoint=CheckpointMetadata(
            key="qwen3_06b_bd3lm_v0_1",
            model_id="dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1",
            revision="60f77bba67ba94231c16f5fb29bc66c500b038f1",
            source_license="apache-2.0",
            custom_tokenizer_module=None,
            mask_token_id=151669,
            tie_word_embeddings=True,
        ),
        architecture=ARCHITECTURE_DEFINITIONS["qwen3_bd3lm"],
        deployment_policy=_mlx_policy(
            "darwin_arm64_apple_m4_max_64gb",
            health_prompt="Hello, who are you?",
            deployment=DeploymentContract(
                profile="bd3-ragged-fixed-quota-no-cache",
                config_source="package",
                attention_mask_kind="d2f_hybrid",
            ),
        ),
        decode_policy=DECODE_POLICIES["fixed_quota_ragged_v1"],
    ),
    "llada_8b_instruct": ModelDefinition(
        checkpoint=CheckpointMetadata(
            key="llada_8b_instruct",
            model_id="GSAI-ML/LLaDA-8B-Instruct",
            revision="08b83a6feb34df1a6011b80c3c00c7563e963b07",
            source_license="mit",
            custom_tokenizer_module=None,
            mask_token_id=126336,
            tie_word_embeddings=False,
        ),
        architecture=ARCHITECTURE_DEFINITIONS["llada"],
        deployment_policy=_mlx_policy(
            "darwin_arm64_apple_m4_max_64gb",
            health_prompt="Explain bidirectional text infilling in one sentence.",
            deployment=DeploymentContract(
                profile="baseline",
                config_source="preset",
                attention_mask_kind="bidirectional",
            ),
        ),
        decode_policy=DECODE_POLICIES["block_threshold_v1"],
    ),
    "dream_coder_v0_instruct_7b": ModelDefinition(
        checkpoint=CheckpointMetadata(
            key="dream_coder_v0_instruct_7b",
            model_id="Dream-org/Dream-Coder-v0-Instruct-7B",
            revision="5d9e88c723af9045f362748b5284bdf43d9c501e",
            source_license="apache-2.0",
            custom_tokenizer_module=None,
            mask_token_id=151666,
            tie_word_embeddings=False,
        ),
        architecture=ARCHITECTURE_DEFINITIONS["dream_coder"],
        deployment_policy=_mlx_policy(
            "darwin_arm64_apple_m4_max_64gb",
            health_prompt="Write a short Python function that adds one to an integer.",
        ),
        decode_policy=DECODE_POLICIES["block_threshold_v1"],
    ),
})

# Runtime product configuration is deliberately separate from certification.
# Admission receipts must prove these exact entries before release, but a
# missing or stale receipt cannot change what an installed runtime exposes.
PRODUCT_MODEL_DEFINITION_KEYS = (
    "dream_v0_instruct_7b",
    "llada_8b_instruct",
    "qwen3_06b_bd3lm_v0_1",
    "qwen3_06b_mdlm_v0_1",
)


def architecture_definition_for_config(
    config: Mapping[str, Any], *, hint: str | None = None
) -> ArchitectureDefinition:
    """Resolve an upstream config to one reusable architecture contract.

    Config and safetensor metadata can identify ordinary layouts.  When two
    diffusion semantics intentionally share the same upstream class (MDLM and
    BD3), an explicit architecture hint is required; a model name is never used
    as a behavioral guess.
    """

    architectures = config.get("architectures")
    actual_architectures = (
        tuple(str(item) for item in architectures)
        if isinstance(architectures, list)
        else ()
    )
    model_type = str(config.get("model_type", "") or "")
    candidates = tuple(
        definition
        for definition in ARCHITECTURE_DEFINITIONS.values()
        if definition.converter.upstream_architectures == actual_architectures
        and model_type in definition.converter.upstream_model_types
    )
    if hint is not None:
        if type(hint) is not str or not hint:
            raise ValueError("architecture_hint_invalid")
        try:
            selected = ARCHITECTURE_DEFINITIONS[hint]
        except KeyError:
            raise ValueError(f"unknown_architecture_definition: {hint}") from None
        if selected not in candidates:
            raise ValueError(
                f"architecture_hint_conflicts_with_snapshot: {hint}"
            )
        return selected
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(
            "unsupported_snapshot_architecture: "
            f"architectures={actual_architectures!r} model_type={model_type!r}"
        )
    names = ", ".join(sorted(item.key for item in candidates))
    raise ValueError(f"architecture_hint_required: choose one of {names}")


def _decode_payload(contract: DecodeContract | None) -> dict[str, object] | None:
    if contract is None:
        return None
    return {
        "name": contract.name,
        "decode_strategy": contract.decode_strategy,
        "attention_mask_kind": contract.attention_mask_kind,
        "block_size": contract.block_size,
        "steps": contract.steps,
        "mask_geometry_prompt_len": contract.mask_geometry_prompt_len,
        "requires_prompt_block_alignment": contract.requires_prompt_block_alignment,
        "layer0_kv_cache": contract.layer0_kv_cache,
        "d2f_exact_kv_cache": contract.d2f_exact_kv_cache,
        "temperature": contract.temperature,
        "top_p": contract.top_p,
        "remasking": contract.remasking,
    }


def _checkpoint_payload(value: CheckpointMetadata) -> dict[str, object]:
    return {
        "key": value.key,
        "model_id": value.model_id,
        "revision": value.revision,
        "source_license": value.source_license,
        "custom_tokenizer_module": value.custom_tokenizer_module,
        "mask_token_id": value.mask_token_id,
        "tie_word_embeddings": value.tie_word_embeddings,
    }


def _converter_payload(value: ConverterAdapter) -> dict[str, object]:
    """Stable declarative semantics behind one converter key.

    Callable implementation changes require a versioned key; they cannot be
    serialized safely. The package still binds every declarative input/output
    contract so a same-name semantic change cannot reuse an old Local cache.
    """

    return {
        "key": value.key,
        "upstream_architectures": list(value.upstream_architectures),
        "upstream_model_types": list(value.upstream_model_types),
        "runtime_architecture": value.runtime_architecture,
        "runtime_adapter": value.runtime_adapter,
        "upstream_tying_field": value.upstream_tying_field,
        "required_tie_word_embeddings": value.required_tie_word_embeddings,
        "qwen_chat_template_fallback": value.qwen_chat_template_fallback,
        "tokenizer_layout": value.tokenizer_layout.value,
        "tied_weight_keys": list(value.tied_weight_keys),
    }


def _runtime_payload(value: RuntimeArchitectureSpec) -> dict[str, object]:
    return {
        "key": value.key,
        "aliases": list(value.aliases),
        "uses_shifted_logits": value.uses_shifted_logits,
        "uses_qk_norm": value.uses_qk_norm,
        "qkv_bias_present": value.qkv_bias_present,
        "chat_template_kwargs": dict(value.chat_template_kwargs),
    }


def _architecture_payload(value: ArchitectureDefinition) -> dict[str, object]:
    return {
        "key": value.key,
        "model_family": value.model_family,
        "converter": _converter_payload(value.converter),
        "runtime": _runtime_payload(value.runtime),
        "graph_ir_family": value.graph_ir_family,
        "features": list(value.features),
    }


def _decode_policy_payload(value: DecodePolicy) -> dict[str, object]:
    return {
        "key": value.key,
        "defaults": dict(value.defaults),
        "contract": _decode_payload(value.contract),
    }


def _deployment_payload(value: DeploymentPolicy) -> dict[str, object]:
    targets: list[dict[str, object]] = []
    for target in value.targets:
        contract = target.deployment
        targets.append(
            {
                "backend": target.backend,
                "precision": target.precision,
                "minimum_device": target.minimum_device,
                "deployment": (
                    None
                    if contract is None
                    else {
                        "profile": contract.profile,
                        "config_source": contract.config_source,
                        "attention_mask_kind": contract.attention_mask_kind,
                    }
                ),
            }
        )
    return {
        "targets": targets,
        "health_prompt": value.health_prompt,
        "health_max_tokens": value.health_max_tokens,
    }


def model_definition_payload(definition: ModelDefinition) -> dict[str, object]:
    """Callable-free package identity for one complete model definition."""

    return {
        "schema_version": 2,
        "checkpoint": _checkpoint_payload(definition.checkpoint),
        "architecture": _architecture_payload(definition.architecture),
        "decode_policy": _decode_policy_payload(definition.decode_policy),
        "deployment_policy": _deployment_payload(definition.deployment_policy),
    }


def model_build_payload(definition: ModelDefinition) -> dict[str, object]:
    """The bytes-producing identity, excluding device/deployment choices."""

    return {
        "schema_version": 2,
        "checkpoint": _checkpoint_payload(definition.checkpoint),
        "architecture": _architecture_payload(definition.architecture),
        "decode_policy": _decode_policy_payload(definition.decode_policy),
    }


def _payload_digest(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def model_definition_digest(definition: ModelDefinition) -> str:
    return _payload_digest(model_definition_payload(definition))


def model_build_digest(definition: ModelDefinition) -> str:
    return _payload_digest(model_build_payload(definition))


def model_definition_from_payload(payload: Mapping[str, object]) -> ModelDefinition:
    """Reconstruct a Local package definition and verify registered semantics."""

    if set(payload) != {
        "schema_version",
        "checkpoint",
        "architecture",
        "decode_policy",
        "deployment_policy",
    } or payload.get("schema_version") != 2:
        raise ValueError("model_definition_payload_invalid")
    checkpoint_data = payload.get("checkpoint")
    architecture_data = payload.get("architecture")
    decode_policy_data = payload.get("decode_policy")
    deployment_data = payload.get("deployment_policy")
    if not isinstance(checkpoint_data, Mapping):
        raise ValueError("model_definition_payload_invalid")
    if not isinstance(architecture_data, Mapping):
        raise ValueError("model_definition_payload_invalid")
    if not isinstance(decode_policy_data, Mapping):
        raise ValueError("model_definition_payload_invalid")
    if not isinstance(deployment_data, Mapping):
        raise ValueError("model_definition_payload_invalid")
    checkpoint_fields = {
        "key", "model_id", "revision", "source_license",
        "custom_tokenizer_module", "mask_token_id", "tie_word_embeddings",
    }
    if set(checkpoint_data) != checkpoint_fields:
        raise ValueError("model_definition_checkpoint_payload_invalid")
    architecture_fields = {
        "key", "model_family", "converter", "runtime", "graph_ir_family",
        "features",
    }
    if set(architecture_data) != architecture_fields:
        raise ValueError("model_definition_architecture_payload_invalid")
    architecture_key = architecture_data["key"]
    converter_data = architecture_data["converter"]
    runtime_data = architecture_data["runtime"]
    if type(architecture_key) is not str:
        raise ValueError("model_definition_architecture_payload_invalid")
    if not isinstance(converter_data, Mapping) or not isinstance(
        runtime_data, Mapping
    ):
        raise ValueError("model_definition_architecture_payload_invalid")
    converter_key = converter_data.get("key")
    runtime_key = runtime_data.get("key")
    if type(converter_key) is not str or type(runtime_key) is not str:
        raise ValueError("model_definition_architecture_payload_invalid")
    try:
        converter = CONVERTER_ADAPTERS[converter_key]
        runtime = RUNTIME_ARCHITECTURES[runtime_key]
    except KeyError as exc:
        raise ValueError(f"unknown_architecture_component: {exc.args[0]}") from None
    if dict(converter_data) != _converter_payload(converter):
        raise ValueError("model_definition_converter_contract_mismatch")
    if dict(runtime_data) != _runtime_payload(runtime):
        raise ValueError("model_definition_runtime_contract_mismatch")
    raw_features = architecture_data["features"]
    if not isinstance(raw_features, list):
        raise ValueError("model_definition_architecture_payload_invalid")
    if set(decode_policy_data) != {"key", "defaults", "contract"}:
        raise ValueError("model_definition_decode_policy_payload_invalid")
    raw_defaults = decode_policy_data["defaults"]
    raw_decode = decode_policy_data["contract"]
    if type(decode_policy_data["key"]) is not str or not isinstance(
        raw_defaults, Mapping
    ):
        raise ValueError("model_definition_decode_policy_payload_invalid")
    decode = None
    if raw_decode is not None:
        if not isinstance(raw_decode, Mapping):
            raise ValueError("model_definition_decode_payload_invalid")
        decode_fields = {
            "name", "decode_strategy", "attention_mask_kind", "block_size",
            "steps", "mask_geometry_prompt_len",
            "requires_prompt_block_alignment", "layer0_kv_cache",
            "d2f_exact_kv_cache", "temperature", "top_p", "remasking",
        }
        if set(raw_decode) != decode_fields:
            raise ValueError("model_definition_decode_payload_invalid")
        decode = DecodeContract(**dict(raw_decode))
    architecture = ArchitectureDefinition(
        key=architecture_key,
        model_family=architecture_data["model_family"],
        converter=converter,
        runtime=runtime,
        graph_ir_family=architecture_data["graph_ir_family"],
        features=tuple(raw_features),
    )
    registered = ARCHITECTURE_DEFINITIONS.get(architecture.key)
    if registered is not None and architecture != registered:
        raise ValueError("model_definition_architecture_contract_mismatch")
    if set(deployment_data) != {"targets", "health_prompt", "health_max_tokens"}:
        raise ValueError("model_definition_deployment_payload_invalid")
    raw_targets = deployment_data["targets"]
    if not isinstance(raw_targets, list):
        raise ValueError("model_definition_deployment_payload_invalid")
    targets: list[DeploymentTarget] = []
    for raw in raw_targets:
        if not isinstance(raw, Mapping) or set(raw) != {
            "backend", "precision", "minimum_device", "deployment"
        }:
            raise ValueError("model_definition_deployment_target_payload_invalid")
        raw_contract = raw["deployment"]
        contract = None
        if raw_contract is not None:
            if not isinstance(raw_contract, Mapping) or set(raw_contract) != {
                "profile", "config_source", "attention_mask_kind"
            }:
                raise ValueError("model_definition_deployment_contract_payload_invalid")
            contract = DeploymentContract(
                profile=raw_contract["profile"],
                config_source=raw_contract["config_source"],
                attention_mask_kind=raw_contract["attention_mask_kind"],
            )
        targets.append(DeploymentTarget(
            backend=raw["backend"],
            precision=raw["precision"],
            minimum_device=raw["minimum_device"],
            deployment=contract,
        ))
    decode_policy = DecodePolicy(
        key=decode_policy_data["key"],
        defaults=dict(raw_defaults),
        contract=decode,
    )
    registered_decode_policy = DECODE_POLICIES.get(decode_policy.key)
    if (
        registered_decode_policy is not None
        and decode_policy != registered_decode_policy
    ):
        raise ValueError("model_definition_decode_policy_contract_mismatch")
    return ModelDefinition(
        checkpoint=CheckpointMetadata(**dict(checkpoint_data)),
        architecture=architecture,
        deployment_policy=DeploymentPolicy(
            targets=tuple(targets),
            health_prompt=deployment_data["health_prompt"],
            health_max_tokens=deployment_data["health_max_tokens"],
        ),
        decode_policy=decode_policy,
    )

def model_definition_for_key(key: str) -> ModelDefinition:
    """Resolve one exact definition; aliases are deliberately not accepted."""

    try:
        return MODEL_DEFINITIONS[key]
    except KeyError:
        raise ValueError(f"unknown_model_definition: {key}") from None


def model_definition_for_catalog_key(key: str) -> ModelDefinition:
    """Compatibility spelling for exact model-key resolution."""

    return model_definition_for_key(key)


def validate_model_snapshot(
    definition: ModelDefinition,
    snapshot: str | Path,
    *,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prove exact HF cache identity and architecture before opening weights."""

    path = Path(snapshot).resolve()
    if path.name != definition.revision:
        raise ValueError(
            "snapshot_revision_mismatch: "
            f"expected={definition.revision} actual={path.name}"
        )
    expected_repo_dir = f"models--{definition.model_id.replace('/', '--')}"
    if path.parent.name != "snapshots" or path.parent.parent.name != expected_repo_dir:
        raise ValueError(f"snapshot_model_id_mismatch: expected={definition.model_id}")
    if config is None:
        config_path = path / "config.json"
        if not config_path.is_file():
            raise ValueError("snapshot_config_missing")
        config = json.loads(config_path.read_text(encoding="utf-8"))
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or not architectures:
        raise ValueError("snapshot_architecture_missing")
    actual_architectures = tuple(str(item) for item in architectures)
    if actual_architectures != definition.expected_architectures:
        raise ValueError(
            "snapshot_architecture_mismatch: "
            f"expected={definition.expected_architectures!r} "
            f"actual={actual_architectures!r}"
        )
    model_type = str(config.get("model_type", ""))
    if model_type not in definition.expected_model_types:
        raise ValueError(
            "snapshot_model_type_mismatch: "
            f"expected={definition.expected_model_types!r} actual={model_type!r}"
        )
    # Coverage tooling still passes its historical flat descriptor projection.
    # Both projections expose the same effective field; the descriptor stores
    # ``None`` for the default only to keep existing receipt digests stable.
    tying_field = getattr(
        definition,
        "effective_upstream_tying_field",
        definition.upstream_tying_field,
    )
    if tying_field not in config:
        raise ValueError(f"snapshot_{tying_field}_missing")
    tie_word_embeddings = config[tying_field]
    if type(tie_word_embeddings) is not bool:
        raise ValueError(
            f"snapshot_{tying_field}_invalid: "
            f"expected=bool actual={tie_word_embeddings!r}"
        )
    if tie_word_embeddings is not definition.tie_word_embeddings:
        raise ValueError(
            f"snapshot_{tying_field}_mismatch: "
            f"expected={definition.tie_word_embeddings!r} "
            f"actual={tie_word_embeddings!r}"
        )
    return dict(config)


def converter_adapter_for_key(key: str) -> ConverterAdapter:
    """Compatibility view over the unified model-definition registry."""

    try:
        return CONVERTER_ADAPTERS[key]
    except KeyError:
        raise ValueError(f"unknown_converter_adapter: {key}") from None


class _DescriptorBinding(Protocol):
    key: str
    converter_adapter: str
    runtime_adapter: str
    expected_architectures: tuple[str, ...]
    expected_model_types: tuple[str, ...]
    runtime_architecture: str
    effective_upstream_tying_field: str


def model_definition_for_descriptor(
    descriptor: _DescriptorBinding,
) -> ModelDefinition:
    """Resolve and validate the one definition named by an exact descriptor."""

    converter_key = str(descriptor.converter_adapter)
    runtime_key = str(descriptor.runtime_adapter)
    definition = model_definition_for_catalog_key(str(descriptor.key))
    if converter_key != definition.converter.key:
        raise ValueError("descriptor_converter_adapter_disagrees")
    converter = definition.converter
    runtime = definition.runtime
    if runtime_key != runtime.key:
        raise ValueError("descriptor_converter_runtime_adapter_disagrees")
    checks = (
        (converter.upstream_architectures, "expected_architectures"),
        (converter.upstream_model_types, "expected_model_types"),
        (converter.runtime_architecture, "runtime_architecture"),
        (converter.runtime_adapter, "runtime_adapter"),
        (converter.upstream_tying_field, "effective_upstream_tying_field"),
    )
    for actual, attribute in checks:
        expected = getattr(descriptor, attribute)
        if actual != expected:
            raise ValueError(f"descriptor_converter_{attribute}_disagrees")
    return definition


def validate_descriptor_binding(descriptor: _DescriptorBinding) -> None:
    """Prove a descriptor resolves through the unified ModelDefinition Interface."""

    model_definition_for_descriptor(descriptor)


__all__ = [
    "ARCHITECTURE_DEFINITIONS",
    "CONVERTER_ADAPTERS",
    "DECODE_POLICIES",
    "MODEL_DEFINITIONS",
    "PRODUCT_MODEL_DEFINITION_KEYS",
    "RUNTIME_ARCHITECTURES",
    "ArchitectureDefinition",
    "CheckpointMetadata",
    "ConverterAdapter",
    "DecodeContract",
    "DecodePolicy",
    "DeploymentContract",
    "DeploymentPolicy",
    "DeploymentTarget",
    "ModelAdapter",
    "ModelDefinition",
    "RuntimeArchitectureSpec",
    "TokenizerLayout",
    "architecture_definition_for_config",
    "build_llada_model_config",
    "build_qwen_model_config",
    "build_runtime_adapter",
    "converter_adapter_for_key",
    "default_decode_policy_for_runtime",
    "map_llada_tensor_name",
    "model_build_digest",
    "model_build_payload",
    "model_definition_digest",
    "model_definition_for_catalog_key",
    "model_definition_for_descriptor",
    "model_definition_for_key",
    "model_definition_from_payload",
    "model_definition_payload",
    "runtime_architecture_for_config",
    "runtime_architecture_for_key",
    "validate_descriptor_binding",
    "validate_model_snapshot",
]
