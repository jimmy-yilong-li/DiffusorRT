"""Application-scoped model definitions for the unified DiffusorRT loader.

The built-in Product registry answers which exact checkpoints DiffusorRT has
admitted.  This module answers a different question: which compatible model
definitions has one application explicitly chosen to use locally?

Catalogs are immutable values.  Loading a TOML file constructs a catalog; it
does not mutate module state, scan entry points, or execute plugin code.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
import hashlib
from pathlib import Path
from types import MappingProxyType
import tomllib
from typing import Any

from .model_definitions import (
    ARCHITECTURE_DEFINITIONS,
    CONVERTER_ADAPTERS,
    DECODE_POLICIES,
    RUNTIME_ARCHITECTURES,
    ArchitectureDefinition,
    CheckpointMetadata,
    DecodeContract,
    DecodePolicy,
    DeploymentContract,
    DeploymentPolicy,
    DeploymentTarget,
    ModelDefinition,
    default_decode_policy_for_runtime,
    model_build_digest,
    model_definition_digest,
)


@dataclass(frozen=True, slots=True)
class ModelResources:
    """Planning estimates for one exact checkpoint.

    These values keep disk/RSS planning available for local catalogs without
    mixing empirical Product-admission evidence into ``ModelDefinition``.
    """

    snapshot_bytes: int
    package_bytes: int
    peak_rss_bytes: int

    def __post_init__(self) -> None:
        for name in ("snapshot_bytes", "package_bytes", "peak_rss_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"model_resource_{name}_invalid")


@dataclass(frozen=True, slots=True)
class LocalModelEntry:
    """One application-selected checkpoint and its planning envelope."""

    definition: ModelDefinition
    resources: ModelResources

    def __post_init__(self) -> None:
        if not isinstance(self.definition, ModelDefinition):
            raise TypeError("local model definition must be a ModelDefinition")
        if not isinstance(self.resources, ModelResources):
            raise TypeError("local model resources must be ModelResources")

    @property
    def key(self) -> str:
        return self.definition.key

    @property
    def model_id(self) -> str:
        return self.definition.model_id

    @property
    def revision(self) -> str:
        return self.definition.revision

    @property
    def source_license(self) -> str:
        return self.definition.source_license

    @property
    def custom_tokenizer_module(self) -> str | None:
        return self.definition.custom_tokenizer_module

    @property
    def snapshot_bytes(self) -> int:
        return self.resources.snapshot_bytes

    @property
    def package_bytes(self) -> int:
        return self.resources.package_bytes

    @property
    def peak_rss_bytes(self) -> int:
        return self.resources.peak_rss_bytes

    @property
    def definition_digest(self) -> str:
        return model_definition_digest(self.definition)

    @property
    def build_digest(self) -> str:
        return model_build_digest(self.definition)


@dataclass(frozen=True, slots=True, init=False)
class ModelCatalog(Mapping[str, LocalModelEntry]):
    """Immutable, explicit Local model scope.

    Mapping lookup is by exact Hugging Face model ID.  Definition-key lookup is
    separate so package identity cannot be confused with a model-ID alias.
    """

    _by_id: Mapping[str, LocalModelEntry]
    _by_key: Mapping[str, LocalModelEntry]

    def __init__(self, entries: Iterable[LocalModelEntry] = ()) -> None:
        by_id: dict[str, LocalModelEntry] = {}
        by_key: dict[str, LocalModelEntry] = {}
        for entry in entries:
            if not isinstance(entry, LocalModelEntry):
                raise TypeError("model catalog entries must be LocalModelEntry values")
            if entry.model_id in by_id:
                raise ValueError(f"duplicate_model_id: {entry.model_id}")
            if entry.key in by_key:
                raise ValueError(f"duplicate_model_definition_key: {entry.key}")
            by_id[entry.model_id] = entry
            by_key[entry.key] = entry
        object.__setattr__(self, "_by_id", MappingProxyType(by_id))
        object.__setattr__(self, "_by_key", MappingProxyType(by_key))

    def __getitem__(self, model_id: str) -> LocalModelEntry:
        return self._by_id[model_id]

    def __iter__(self) -> Iterator[str]:
        return iter(self._by_id)

    def __len__(self) -> int:
        return len(self._by_id)

    def entry_for_key(self, key: str) -> LocalModelEntry:
        try:
            return self._by_key[key]
        except KeyError:
            raise ValueError(f"unknown_model_definition: {key}") from None

    def extend(self, entries: Iterable[LocalModelEntry]) -> "ModelCatalog":
        """Return a new catalog; this instance is never mutated."""

        return ModelCatalog((*self._by_id.values(), *tuple(entries)))

    @classmethod
    def architectures(cls) -> tuple[str, ...]:
        """Architecture templates accepted by the concise TOML format."""

        return tuple(sorted(_ARCHITECTURE_TEMPLATES))

    @classmethod
    def decode_policies(cls) -> tuple[str, ...]:
        """Registered decode-policy keys accepted by both TOML formats."""

        return tuple(sorted(DECODE_POLICIES))

    @classmethod
    def from_toml(cls, path: str | Path) -> "ModelCatalog":
        """Load declarative definitions that reuse built-in component keys."""

        source = Path(path)
        try:
            with source.open("rb") as handle:
                document = tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(f"model_catalog_toml_invalid: {source}: {exc}") from exc
        if set(document) != {"schema_version", "models"}:
            raise ValueError(f"model_catalog_top_level_fields_invalid: source={source}")
        if type(document["schema_version"]) is not int or document["schema_version"] != 1:
            raise ValueError(
                f"model_catalog_schema_version_unsupported: source={source}"
            )
        models = document["models"]
        if not isinstance(models, list) or not models:
            raise ValueError(f"model_catalog_models_missing: source={source}")
        entries: list[LocalModelEntry] = []
        model_ids: set[str] = set()
        definition_keys: set[str] = set()
        for index, item in enumerate(models, start=1):
            model_id = item.get("model_id") if isinstance(item, Mapping) else None
            context = f"source={source}; model_index={index}"
            if model_id is not None:
                context += f"; model_id={model_id!r}"
            try:
                entry = _entry_from_toml(item)
                if entry.model_id in model_ids:
                    raise ValueError(f"duplicate_model_id: {entry.model_id}")
                if entry.key in definition_keys:
                    raise ValueError(
                        f"duplicate_model_definition_key: {entry.key}"
                    )
            except (ValueError, TypeError) as exc:
                raise type(exc)(f"{exc}: {context}") from exc
            entries.append(entry)
            model_ids.add(entry.model_id)
            definition_keys.add(entry.key)
        return cls(entries)


_MODEL_FIELDS = frozenset(
    {
        "key",
        "model_id",
        "revision",
        "source_license",
        "model_family",
        "converter",
        "runtime",
        "custom_tokenizer_module",
        "mask_token_id",
        "tie_word_embeddings",
        "graph_ir_family",
        "architecture_features",
        "deployments",
        "decode_contract",
        "decode_policy",
        "resources",
    }
)
_REQUIRED_MODEL_FIELDS = _MODEL_FIELDS - {
    "custom_tokenizer_module",
    "decode_contract",
    "decode_policy",
}

_ARCHITECTURE_TEMPLATES: Mapping[str, ArchitectureDefinition] = (
    ARCHITECTURE_DEFINITIONS
)
_TEMPLATE_FIELDS = frozenset(
    {
        "architecture",
        "key",
        "model_id",
        "revision",
        "license",
        "custom_tokenizer_module",
        "mask_token_id",
        "tie_word_embeddings",
        "deployments",
        "decode_policy",
        "resources",
    }
)
_REQUIRED_TEMPLATE_FIELDS = frozenset(
    {
        "architecture",
        "model_id",
        "revision",
        "license",
        "mask_token_id",
        "tie_word_embeddings",
        "resources",
    }
)


def _table(value: object, *, error: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(error)
    return value


def _validate_fields(
    table: Mapping[str, Any],
    *,
    allowed: set[str] | frozenset[str],
    required: set[str] | frozenset[str],
    error: str,
) -> None:
    missing = sorted(required - set(table))
    unknown = sorted(set(table) - allowed)
    if missing or unknown:
        raise ValueError(
            f"{error}: missing_keys={missing!r}; unknown_keys={unknown!r}"
        )


def _decode_policy_from_toml(value: object) -> DecodePolicy:
    supported = ", ".join(sorted(DECODE_POLICIES))
    if type(value) is not str or not value:
        raise ValueError(
            f"model_catalog_decode_policy_invalid: {value!r}; "
            f"expected one of {supported}"
        )
    try:
        return DECODE_POLICIES[value]
    except KeyError:
        raise ValueError(
            f"unknown_decode_policy: {value!r}; expected one of {supported}"
        ) from None


def _deployment_from_toml(value: object) -> DeploymentTarget:
    table = _table(value, error="model_catalog_deployment_invalid")
    allowed = {"backend", "precision", "minimum_device", "deployment"}
    required = allowed - {"deployment"}
    _validate_fields(
        table,
        allowed=allowed,
        required=required,
        error="model_catalog_deployment_fields_invalid",
    )
    contract = None
    if "deployment" in table:
        deployment = _table(
            table["deployment"], error="model_catalog_deployment_contract_invalid"
        )
        allowed_contract_fields = {
            "profile", "config_source", "attention_mask_kind"
        }
        _validate_fields(
            deployment,
            allowed=allowed_contract_fields,
            required={"profile", "config_source"},
            error="model_catalog_deployment_contract_fields_invalid",
        )
        contract = DeploymentContract(
            profile=deployment["profile"],
            config_source=deployment["config_source"],
            attention_mask_kind=deployment.get("attention_mask_kind"),
        )
    return DeploymentTarget(
        backend=table["backend"],
        precision=table["precision"],
        minimum_device=table["minimum_device"],
        deployment=contract,
    )


def _decode_from_toml(value: object) -> DecodeContract:
    table = _table(value, error="model_catalog_decode_contract_invalid")
    fields = {
        "name",
        "decode_strategy",
        "attention_mask_kind",
        "block_size",
        "steps",
        "mask_geometry_prompt_len",
        "requires_prompt_block_alignment",
        "layer0_kv_cache",
        "d2f_exact_kv_cache",
        "temperature",
        "top_p",
        "remasking",
    }
    _validate_fields(
        table,
        allowed=fields,
        required=fields,
        error="model_catalog_decode_contract_fields_invalid",
    )
    return DecodeContract(**{name: table[name] for name in fields})


def _resources_from_toml(value: object) -> ModelResources:
    resources = _table(value, error="model_catalog_resources_invalid")
    fields = {"snapshot_bytes", "package_bytes", "peak_rss_bytes"}
    _validate_fields(
        resources,
        allowed=fields,
        required=fields,
        error="model_catalog_resource_fields_invalid",
    )
    return ModelResources(
        snapshot_bytes=resources["snapshot_bytes"],
        package_bytes=resources["package_bytes"],
        peak_rss_bytes=resources["peak_rss_bytes"],
    )


def _entry_from_template_toml(table: Mapping[str, Any]) -> LocalModelEntry:
    _validate_fields(
        table,
        allowed=_TEMPLATE_FIELDS,
        required=_REQUIRED_TEMPLATE_FIELDS,
        error="model_catalog_template_fields_invalid",
    )
    architecture = table["architecture"]
    if type(architecture) is not str or not architecture:
        raise ValueError("model_catalog_architecture_invalid")
    try:
        template = _ARCHITECTURE_TEMPLATES[architecture]
    except KeyError:
        supported = ", ".join(sorted(_ARCHITECTURE_TEMPLATES))
        raise ValueError(
            f"unsupported_architecture_template: {architecture!r}; "
            f"expected one of {supported}"
        ) from None

    model_id = table["model_id"]
    revision = table["revision"]
    key = table.get("key")
    if key is None:
        identity = f"{model_id}@{revision}".encode("utf-8")
        key = f"local_{hashlib.sha256(identity).hexdigest()[:16]}"
    deployments = table.get(
        "deployments",
        (
            DeploymentTarget(
                backend="mlx_local",
                precision="fp32",
                minimum_device="darwin_arm64_apple_silicon",
            ),
        ),
    )
    if isinstance(deployments, list):
        deployments = tuple(_deployment_from_toml(item) for item in deployments)
    definition = ModelDefinition(
        checkpoint=CheckpointMetadata(
            key=key,
            model_id=model_id,
            revision=revision,
            source_license=table["license"],
            custom_tokenizer_module=table.get("custom_tokenizer_module"),
            mask_token_id=table["mask_token_id"],
            tie_word_embeddings=table["tie_word_embeddings"],
        ),
        architecture=template,
        deployment_policy=DeploymentPolicy(targets=deployments),
        decode_policy=(
            _decode_policy_from_toml(table["decode_policy"])
            if "decode_policy" in table
            else default_decode_policy_for_runtime(template.runtime.key)
        ),
    )
    return LocalModelEntry(
        definition=definition,
        resources=_resources_from_toml(table["resources"]),
    )


def _entry_from_toml(value: object) -> LocalModelEntry:
    table = _table(value, error="model_catalog_model_invalid")
    if "architecture" in table:
        return _entry_from_template_toml(table)
    _validate_fields(
        table,
        allowed=_MODEL_FIELDS,
        required=_REQUIRED_MODEL_FIELDS,
        error="model_catalog_model_fields_invalid",
    )
    if "decode_contract" in table and "decode_policy" in table:
        raise ValueError("model_catalog_decode_policy_conflict")
    try:
        converter = CONVERTER_ADAPTERS[table["converter"]]
    except (KeyError, TypeError):
        raise ValueError(
            f"unknown_converter_adapter: {table.get('converter')!r}"
        ) from None
    try:
        runtime = RUNTIME_ARCHITECTURES[table["runtime"]]
    except (KeyError, TypeError):
        raise ValueError(
            f"unsupported_architecture_family: {table.get('runtime')!r}"
        ) from None
    deployments = table["deployments"]
    if not isinstance(deployments, list) or not deployments:
        raise ValueError("model_catalog_deployments_missing")
    architecture_features = table["architecture_features"]
    if not isinstance(architecture_features, list) or any(
        not isinstance(item, str) or not item for item in architecture_features
    ):
        raise ValueError("model_catalog_architecture_features_invalid")
    declared_decode = (
        _decode_from_toml(table["decode_contract"])
        if "decode_contract" in table
        else None
    )
    architecture = ArchitectureDefinition(
        key=f"local:{table['key']}",
        model_family=table["model_family"],
        converter=converter,
        runtime=runtime,
        graph_ir_family=table["graph_ir_family"],
        features=tuple(architecture_features),
    )
    decode_policy = default_decode_policy_for_runtime(runtime.key)
    if "decode_policy" in table:
        decode_policy = _decode_policy_from_toml(table["decode_policy"])
    elif declared_decode is not None:
        decode_policy = DecodePolicy(
            key=f"local:{table['key']}:decode_v1",
            defaults={},
            contract=declared_decode,
        )
    definition = ModelDefinition(
        checkpoint=CheckpointMetadata(
            key=table["key"],
            model_id=table["model_id"],
            revision=table["revision"],
            source_license=table["source_license"],
            custom_tokenizer_module=table.get("custom_tokenizer_module"),
            mask_token_id=table["mask_token_id"],
            tie_word_embeddings=table["tie_word_embeddings"],
        ),
        architecture=architecture,
        deployment_policy=DeploymentPolicy(
            targets=tuple(_deployment_from_toml(item) for item in deployments)
        ),
        decode_policy=decode_policy,
    )
    return LocalModelEntry(
        definition=definition,
        resources=_resources_from_toml(table["resources"]),
    )


__all__ = [
    "LocalModelEntry",
    "ModelCatalog",
    "ModelResources",
    "model_build_digest",
    "model_definition_digest",
]
