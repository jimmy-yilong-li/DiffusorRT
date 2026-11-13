"""Durable, relocatable application configuration.

This module owns user choices only.  Model definitions and deployment
contracts remain in their existing registries, while credentials and
remote-code consent are deliberately absent from the saved schema.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass, fields
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Iterator

from .backend_ids import (
    validate_backend,
    validate_deployment_profile,
    validate_device_options,
)
from .runtime_config import validate_generation_length


DATA_ROOT_ENV = "DIFFUSORRT_DATA_ROOT"
CONFIG_SCHEMA_VERSION = 1
DEFAULT_CONTEXT_POLICY = "recent_only"
CONTEXT_POLICIES = frozenset({DEFAULT_CONTEXT_POLICY, "bounded_summary"})

_CONFIG_FILE = "config.json"
_CONFIG_LOCK_FILE = ".config.lock"
_MAX_CONFIG_BYTES = 4 * 1024 * 1024
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ConfigurationError(ValueError):
    """Saved application configuration is malformed or incompatible."""


def _nonempty_string(value: object, *, field_name: str) -> str:
    if type(value) is not str or not value:
        raise ConfigurationError(f"run_profile_{field_name}_invalid")
    return value


@dataclass(frozen=True, slots=True)
class RunProfile:
    """A named, portable set of user run choices bound to exact model bytes.

    Registry and Local-catalog models bind ``revision`` plus
    ``model_definition_digest``.  A package reference instead binds
    ``package_digest``.  Runtime consent and credentials are intentionally not
    representable.
    """

    name: str
    model: str
    revision: str | None = None
    model_definition_digest: str | None = None
    package_digest: str | None = None
    catalog_path: str | None = None
    backend: str | None = None
    precision: str | None = None
    device_ordinal: int | None = None
    device_memory_budget_bytes: int | None = None
    offline: bool | None = None
    max_tokens: int | None = None
    context_policy: str | None = None
    system_prompt: str | None = None
    deployment_profile_id: str | None = None

    def __post_init__(self) -> None:
        name = _nonempty_string(self.name, field_name="name")
        if (
            name != name.strip()
            or len(name) > 128
            or any(ord(character) < 32 for character in name)
        ):
            raise ConfigurationError("run_profile_name_invalid")
        _nonempty_string(self.model, field_name="model")

        package_bound = self.package_digest is not None
        definition_bound = (
            self.revision is not None or self.model_definition_digest is not None
        )
        if package_bound == definition_bound:
            raise ConfigurationError("run_profile_identity_ambiguous")
        if package_bound:
            if type(self.package_digest) is not str or not _SHA256.fullmatch(
                self.package_digest
            ):
                raise ConfigurationError("run_profile_package_digest_invalid")
        elif (
            type(self.revision) is not str
            or not _SHA40.fullmatch(self.revision)
            or type(self.model_definition_digest) is not str
            or not _SHA256.fullmatch(self.model_definition_digest)
        ):
            raise ConfigurationError("run_profile_model_identity_invalid")

        for field_name in ("catalog_path", "backend", "precision"):
            value = getattr(self, field_name)
            if value is not None:
                _nonempty_string(value, field_name=field_name)
        for field_name in ("device_ordinal", "device_memory_budget_bytes"):
            value = getattr(self, field_name)
            if value is not None and (
                type(value) is not int or value < 0
            ):
                raise ConfigurationError(f"run_profile_{field_name}_invalid")
        if self.offline is not None and type(self.offline) is not bool:
            raise ConfigurationError("run_profile_offline_invalid")
        if self.max_tokens is not None:
            try:
                validate_generation_length(self.max_tokens, field_name="max_tokens")
            except ValueError as exc:
                raise ConfigurationError(str(exc)) from exc
        if self.context_policy is not None:
            if type(self.context_policy) is not str:
                raise ConfigurationError("run_profile_context_policy_invalid")
            if self.context_policy not in CONTEXT_POLICIES:
                raise ConfigurationError(
                    f"run_profile_context_policy_unknown: {self.context_policy!r}"
                )
        if self.system_prompt is not None and type(self.system_prompt) is not str:
            raise ConfigurationError("run_profile_system_prompt_invalid")
        try:
            validate_deployment_profile(self.deployment_profile_id)
        except ValueError as exc:
            raise ConfigurationError(str(exc).replace("deployment_profile", "deployment_profile_id")) from exc

    @property
    def device(self) -> int | None:
        """Semantic device selection, represented by the existing ordinal API."""

        return self.device_ordinal

    @property
    def memory_budget_bytes(self) -> int | None:
        return self.device_memory_budget_bytes

    @property
    def generation_limit(self) -> int | None:
        return self.max_tokens

    def to_dict(self) -> dict[str, object]:
        """Return the version-independent profile record payload."""

        result: dict[str, object] = {}
        for profile_field in fields(self):
            value = getattr(self, profile_field.name)
            if value is not None:
                result[profile_field.name] = value
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "RunProfile":
        if not isinstance(value, Mapping):
            raise ConfigurationError("run_profile_record_invalid")
        allowed = {profile_field.name for profile_field in fields(cls)}
        unknown = set(value) - allowed
        if unknown:
            raise ConfigurationError(
                f"run_profile_fields_unknown: {sorted(unknown)!r}"
            )
        required = {"name", "model"}
        missing = required - set(value)
        if missing:
            raise ConfigurationError(
                f"run_profile_fields_missing: {sorted(missing)!r}"
            )
        return cls(**dict(value))


_RUN_PROFILE_FIELDS = frozenset(
    profile_field.name for profile_field in fields(RunProfile) if profile_field.name != "name"
)
_IMMUTABLE_PROFILE_FIELDS = frozenset(
    {
        "model",
        "revision",
        "model_definition_digest",
        "package_digest",
        "catalog_path",
    }
)
FRAMEWORK_RUN_DEFAULTS: Mapping[str, object] = {
    "backend": "auto",
    "precision": None,
    "device_ordinal": 0,
    "device_memory_budget_bytes": 0,
    "offline": False,
    "max_tokens": None,
    "context_policy": DEFAULT_CONTEXT_POLICY,
    "system_prompt": None,
    "deployment_profile_id": None,
}


def _validated_layer(
    value: Mapping[str, object] | None,
    *,
    layer_name: str,
) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{layer_name} must be a mapping")
    unknown = set(value) - _RUN_PROFILE_FIELDS
    if unknown:
        raise ConfigurationError(
            f"run_configuration_{layer_name}_fields_unknown: {sorted(unknown)!r}"
        )
    immutable = set(value) & _IMMUTABLE_PROFILE_FIELDS
    if immutable:
        raise ConfigurationError(
            f"run_configuration_{layer_name}_identity_forbidden: "
            f"{sorted(immutable)!r}"
        )
    return dict(value)


def resolve_run_config(
    profile: RunProfile,
    *,
    explicit: Mapping[str, object] | None = None,
    deployment_defaults: Mapping[str, object] | None = None,
    framework_defaults: Mapping[str, object] | None = None,
) -> RunProfile:
    """Resolve caller choices with one fixed, typed precedence.

    ``explicit`` contains only values the caller actually supplied.  Parser
    defaults belong in ``framework_defaults``; putting them in ``explicit``
    would intentionally give them explicit precedence.
    """

    if not isinstance(profile, RunProfile):
        raise TypeError("profile must be a RunProfile")
    explicit_values = _validated_layer(explicit, layer_name="explicit")

    layers = (
        _validated_layer(
            FRAMEWORK_RUN_DEFAULTS if framework_defaults is None else framework_defaults,
            layer_name="framework_defaults",
        ),
        _validated_layer(deployment_defaults, layer_name="deployment_defaults"),
        {key: value for key, value in profile.to_dict().items() if key != "name"},
        explicit_values,
    )
    resolved: dict[str, object] = {}
    for layer in layers:
        resolved.update(layer)
    result = RunProfile(name=profile.name, **resolved)
    if result.backend is None:
        raise ConfigurationError("run_configuration_backend_missing")
    try:
        backend = validate_backend(result.backend)
        device_ordinal = result.device_ordinal or 0
        device_memory_budget_bytes = result.device_memory_budget_bytes or 0
        if backend == "cuda_native":
            validate_device_options(
                backend,
                result.precision,
                device_ordinal,
                device_memory_budget_bytes,
            )
        elif backend != "auto" and (
            device_ordinal != 0 or device_memory_budget_bytes != 0
        ):
            raise ValueError(
                "device_ordinal and device_memory_budget_bytes are accepted "
                "only with backend='cuda_native'"
            )
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc
    return result


def resolve_data_root(
    root: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve explicit root, then environment, then platform user-data root."""

    environment = os.environ if environ is None else environ
    candidate: str | Path | None = root
    if candidate is None:
        candidate = environment.get(DATA_ROOT_ENV) or None
    if candidate is not None:
        if isinstance(candidate, str) and not candidate:
            raise ValueError("application data root must not be empty")
        if not isinstance(candidate, (str, Path)) or isinstance(candidate, bool):
            raise TypeError("application data root must be a path")
        return Path(candidate).expanduser().resolve(strict=False)

    # The home directory is asked for only where it is the answer: a process
    # may have LOCALAPPDATA or XDG_DATA_HOME and no home at all.
    if sys.platform == "darwin":
        return (Path.home() / "Library" / "Application Support" / "DiffusorRT").resolve(
            strict=False
        )
    if os.name == "nt":
        local_app_data = environment.get("LOCALAPPDATA") or environment.get("APPDATA")
        base = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
        return (base / "DiffusorRT").expanduser().resolve(strict=False)
    xdg_data_home = environment.get("XDG_DATA_HOME")
    base = (
        Path(xdg_data_home).expanduser()
        if xdg_data_home
        else Path.home() / ".local" / "share"
    )
    return (base / "diffusorrt").expanduser().resolve(strict=False)


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ConfigurationError(f"configuration_duplicate_field: {key!r}")
        result[key] = value
    return result


def _ensure_owner_only_directory(path: Path) -> None:
    missing: list[Path] = []
    probe = path
    while not probe.exists():
        missing.append(probe)
        if probe == probe.parent:
            break
        probe = probe.parent
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if os.name != "nt":
        for created in reversed(missing):
            os.chmod(created, 0o700)


def _retry_while_busy(operation):
    """Windows refuses to replace or read a file another handle is replacing."""
    if os.name != "nt":
        return operation()
    from ._windows_files import retry_while_busy

    return retry_while_busy(operation)


class ApplicationState:
    """One lazy application-state root and its versioned run-profile store."""

    __slots__ = ("_root",)

    def __init__(self, root: str | Path | None = None) -> None:
        self._root = resolve_data_root(root)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def config_path(self) -> Path:
        return self._root / _CONFIG_FILE

    @contextmanager
    def _write_lock(self) -> Iterator[None]:
        _ensure_owner_only_directory(self._root)
        lock_path = self._root / _CONFIG_LOCK_FILE
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(lock_path, flags, 0o600)
        try:
            if os.name != "nt":
                os.fchmod(descriptor, 0o600)
            if os.name == "nt":
                from ._windows_files import lock_descriptor

                # Released when the descriptor closes below.
                lock_descriptor(descriptor, exclusive=True, blocking=True)
                yield
                return
            try:
                import fcntl
            except ImportError as exc:  # pragma: no cover - unsupported platform
                raise RuntimeError(
                    "cross-process application configuration locking is unavailable"
                ) from exc
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def _read_document(self) -> dict[str, RunProfile]:
        path = self.config_path
        if not path.exists():
            return {}
        try:
            if path.stat().st_size > _MAX_CONFIG_BYTES:
                raise ConfigurationError("configuration_file_too_large")
            raw = json.loads(
                _retry_while_busy(lambda: path.read_text(encoding="utf-8")),
                object_pairs_hook=_object_without_duplicate_keys,
            )
        except ConfigurationError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ConfigurationError("configuration_file_invalid") from exc
        if type(raw) is not dict:
            raise ConfigurationError("configuration_document_invalid")
        unknown = set(raw) - {"schema_version", "run_profiles"}
        if unknown:
            raise ConfigurationError(
                f"configuration_fields_unknown: {sorted(unknown)!r}"
            )
        if set(raw) != {"schema_version", "run_profiles"}:
            raise ConfigurationError("configuration_fields_missing")
        schema_version = raw["schema_version"]
        if type(schema_version) is not int:
            raise ConfigurationError("configuration_schema_version_invalid")
        if schema_version > CONFIG_SCHEMA_VERSION:
            raise ConfigurationError(
                f"configuration_schema_newer: {schema_version}"
            )
        if schema_version != CONFIG_SCHEMA_VERSION:
            raise ConfigurationError(
                f"configuration_schema_unsupported: {schema_version}"
            )
        records = raw["run_profiles"]
        if type(records) is not list:
            raise ConfigurationError("configuration_run_profiles_invalid")
        profiles: dict[str, RunProfile] = {}
        for record in records:
            profile = RunProfile.from_dict(record)
            if profile.name in profiles:
                raise ConfigurationError(
                    f"configuration_run_profile_duplicate: {profile.name!r}"
                )
            profiles[profile.name] = profile
        return profiles

    def _replace_document(self, profiles: Mapping[str, RunProfile]) -> None:
        payload = {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "run_profiles": [profiles[name].to_dict() for name in sorted(profiles)],
        }
        encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
        if len(encoded) > _MAX_CONFIG_BYTES:
            raise ConfigurationError("configuration_file_too_large")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".config.",
            suffix=".tmp",
            dir=self._root,
        )
        temporary_path = Path(temporary_name)
        try:
            if os.name != "nt":
                os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = -1
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            _retry_while_busy(lambda: os.replace(temporary_path, self.config_path))
            if os.name != "nt":
                os.chmod(self.config_path, 0o600)
            # Windows cannot open a directory this way, and NTFS journals the rename.
            if os.name != "nt":
                directory_descriptor = os.open(self._root, os.O_RDONLY)
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            with suppress(FileNotFoundError):
                temporary_path.unlink()

    def save_run_profile(self, profile: RunProfile) -> None:
        if not isinstance(profile, RunProfile):
            raise TypeError("profile must be a RunProfile")
        with self._write_lock():
            profiles = self._read_document()
            profiles[profile.name] = profile
            self._replace_document(profiles)

    def load_run_profile(self, name: str) -> RunProfile:
        _nonempty_string(name, field_name="name")
        profiles = self._read_document()
        try:
            return profiles[name]
        except KeyError:
            raise KeyError(f"unknown run profile: {name!r}") from None

    def list_run_profiles(self) -> tuple[RunProfile, ...]:
        profiles = self._read_document()
        return tuple(profiles[name] for name in sorted(profiles))

    def delete_run_profile(self, name: str) -> None:
        _nonempty_string(name, field_name="name")
        if not self.config_path.exists():
            raise KeyError(f"unknown run profile: {name!r}")
        with self._write_lock():
            profiles = self._read_document()
            if name not in profiles:
                raise KeyError(f"unknown run profile: {name!r}")
            del profiles[name]
            self._replace_document(profiles)


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "DATA_ROOT_ENV",
    "ApplicationState",
    "ConfigurationError",
    "RunProfile",
    "resolve_data_root",
    "resolve_run_config",
]
