"""Which model a launch opens.

Command mode opens what it was given. A double-click opens the model the window
last opened, then the only prepared model, and otherwise asks the user.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..backend_ids import AUTO_BACKEND, CUDA_NATIVE_BACKEND
from .preferences import GuiPreferences

_log = logging.getLogger("edllm.gui")

SOURCE_COMMAND = "command"
SOURCE_REMEMBERED = "remembered"
SOURCE_ONLY_AVAILABLE = "only_available"
SOURCE_PICKED = "picked"


@dataclass(frozen=True)
class LaunchChoice:
    spec: str
    backend: str | None
    """None: decide with `default_backend` on the worker, because deciding loads the package."""
    precision: str | None
    source: str


@dataclass(frozen=True)
class NeedsPicker:
    candidates: tuple[Any, ...]
    reason: str
    """"no_models", "several" or "remembered_missing"."""


def resolve_launch(
    *,
    command_spec: str | None,
    command_backend: str,
    command_precision: str | None,
    preferences: GuiPreferences,
    repository: Any,
) -> LaunchChoice | NeedsPicker:
    if command_spec is not None:
        return LaunchChoice(command_spec, _chosen(command_backend), command_precision, SOURCE_COMMAND)
    candidates = prepared_models(repository)
    if preferences.last_model is not None:
        if _still_present(repository, preferences.last_model):
            return LaunchChoice(
                preferences.last_model, _chosen(preferences.backend), preferences.precision,
                SOURCE_REMEMBERED,
            )
        return NeedsPicker(candidates, "remembered_missing")
    if len(candidates) == 1:
        return LaunchChoice(candidates[0].model_id, None, None, SOURCE_ONLY_AVAILABLE)
    return NeedsPicker(candidates, "several" if candidates else "no_models")


def _chosen(backend: str | None) -> str | None:
    """"auto" is no choice: the worker decides with `default_backend`."""
    return None if backend in (None, AUTO_BACKEND) else backend


def admitted_cuda_precision(package_dir: Path) -> str | None:
    """The precision this package's CUDA profile admits first, or None without one."""
    from ..backends import cuda_profile

    try:
        return cuda_profile.admit_cuda_execution_profile(package_dir, precision=None).precisions[0]
    except (OSError, ValueError):
        return None


def default_backend(package_dir: Path, repository: Any) -> tuple[str, str | None]:
    """cuda_native with the package's admitted precision when this machine can run it."""
    precision = admitted_cuda_precision(package_dir)
    if precision is None:
        return AUTO_BACKEND, None
    report = repository.report(
        str(package_dir),
        requested_backend=CUDA_NATIVE_BACKEND,
        check_backend=True,
        cuda_precision=precision,
    )
    if report.ready:
        return CUDA_NATIVE_BACKEND, precision
    return AUTO_BACKEND, None


def prepared_models(repository: Any) -> tuple[Any, ...]:
    """The prepared models, or none when the cache cannot be listed (a drive that is gone)."""
    from ..local_models import CACHE_CACHED

    try:
        models = repository.cached_models()
    except (OSError, ValueError):
        _log.exception("listing prepared models failed")
        return ()
    return tuple(model for model in models if model.status == CACHE_CACHED)


def _still_present(repository: Any, spec: str) -> bool:
    from ..compiler import CompilerError
    from ..local_models import ModelSpecError

    try:
        return bool(repository.plan(spec, allow_local_dir=True, offline=True).package_present)
    except (CompilerError, ModelSpecError, OSError, ValueError):
        return False
