"""What the window remembers between launches: the last model and the panel state.

One small JSON file in the application-data root, beside the conversation
store. A missing or damaged file means the defaults; the window never refuses
to start over a preference.
"""
from __future__ import annotations

import json
import os
import tempfile
from contextlib import suppress
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from ...application_state import (
    ConfigurationError,
    _ensure_owner_only_directory,
    _object_without_duplicate_keys,
)

PREFERENCES_FILE = "gui-mac-preferences.json"
_MAX_BYTES = 64 * 1024


@dataclass(frozen=True)
class GuiPreferences:
    last_model: str | None = None
    backend: str | None = None
    precision: str | None = None
    resources_expanded: bool = True
    live_decode: bool = True
    """Whether a window shows the decode as it happens."""


_TYPES = {"last_model": str, "backend": str, "precision": str, "resources_expanded": bool,
          "live_decode": bool}


def load_preferences(data_root: Path) -> GuiPreferences:
    """The saved preferences, or the defaults when there are none or they are damaged."""
    path = Path(data_root) / PREFERENCES_FILE
    try:
        raw = path.read_bytes()
        if len(raw) > _MAX_BYTES:
            return GuiPreferences()
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=_object_without_duplicate_keys)
    except (OSError, UnicodeDecodeError, ValueError, ConfigurationError):
        return GuiPreferences()
    if not isinstance(document, dict):
        return GuiPreferences()
    values = {}
    for field in fields(GuiPreferences):
        if field.name not in document:
            continue
        value = document[field.name]
        expected = _TYPES[field.name]
        if value is not None and type(value) is not expected:
            return GuiPreferences()
        if value is None and expected is bool:
            return GuiPreferences()
        values[field.name] = value
    return GuiPreferences(**values)


def save_preferences(data_root: Path, preferences: GuiPreferences) -> None:
    """Replace the saved preferences in one step; a reader never sees half a file."""
    root = Path(data_root)
    _ensure_owner_only_directory(root)
    encoded = (json.dumps(asdict(preferences), indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".gui-mac-preferences.", suffix=".tmp", dir=root)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, root / PREFERENCES_FILE)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        with suppress(FileNotFoundError):
            temporary.unlink()
