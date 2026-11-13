"""Which model a launch opens.

Command mode opens what it was given. A double-click opens the model the window
last opened, then the only prepared model, and otherwise asks the user.

author:Opus -- the Mac windows' own launch rules; no other platform's backend is considered here.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ...backend_ids import AUTO_BACKEND, MLX_LOCAL_BACKEND
from .preferences import GuiPreferences

SOURCE_COMMAND = "command"
SOURCE_REMEMBERED = "remembered"
SOURCE_ONLY_AVAILABLE = "only_available"
SOURCE_PICKED = "picked"


@dataclass(frozen=True)
class LaunchChoice:
    spec: str
    backend: str | None
    """None: decide with `preferred_backend` on the worker, because deciding loads the package."""
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
    command_backend: str | None,
    command_precision: str | None,
    preferences: GuiPreferences,
    repository: Any,
) -> LaunchChoice | NeedsPicker:
    def overrides(choice: LaunchChoice) -> LaunchChoice:
        backend = command_backend if command_backend is not None else choice.backend
        precision = command_precision if command_precision is not None else (
            choice.precision if backend == choice.backend else None
        )
        return replace(choice, backend=backend, precision=precision)

    # `auto` is what the command line says when nobody chose: the window decides then.
    if command_backend == AUTO_BACKEND:
        command_backend = None
    if command_spec is not None:
        return LaunchChoice(command_spec, command_backend, command_precision, SOURCE_COMMAND)
    candidates = _prepared_models(repository)
    if preferences.last_model is not None:
        if _still_present(repository, preferences.last_model):
            return overrides(LaunchChoice(
                preferences.last_model, preferences.backend, preferences.precision,
                SOURCE_REMEMBERED,
            ))
        return NeedsPicker(candidates, "remembered_missing")
    if len(candidates) == 1:
        return overrides(LaunchChoice(candidates[0].model_id, None, None, SOURCE_ONLY_AVAILABLE))
    return NeedsPicker(candidates, "several" if candidates else "no_models")


def preferred_backend(package_dir: Path, repository: Any) -> tuple[str, str | None]:
    """The backend a Mac window opens when the user chose none.

    `mlx_local` when this Mac can run the package with it, so the window shows
    the real backend and a reply can be stopped; `auto` otherwise.
    """
    try:
        report = repository.report(
            str(package_dir), requested_backend=MLX_LOCAL_BACKEND, check_backend=True
        )
    except (OSError, ValueError):
        return AUTO_BACKEND, None
    return (MLX_LOCAL_BACKEND, None) if report.ready else (AUTO_BACKEND, None)


def _prepared_models(repository: Any) -> tuple[Any, ...]:
    from ...local_models import CACHE_CACHED

    return tuple(model for model in repository.cached_models() if model.status == CACHE_CACHED)


def _still_present(repository: Any, spec: str) -> bool:
    from ...compiler import CompilerError
    from ...local_models import ModelSpecError

    try:
        return bool(repository.plan(spec, allow_local_dir=True, offline=True).package_present)
    except (CompilerError, ModelSpecError, OSError, ValueError):
        return False
