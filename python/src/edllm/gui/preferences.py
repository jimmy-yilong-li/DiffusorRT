"""What the window remembers between launches: the last model, the panel state
and the temperature replies are sampled at.

One small JSON file in the application-data root, beside the conversation
store. A missing or damaged file means the defaults; the window never refuses
to start over a preference.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from contextlib import suppress
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from ..application_state import (
    ConfigurationError,
    _ensure_owner_only_directory,
    _object_without_duplicate_keys,
    _retry_while_busy,
)

PREFERENCES_FILE = "gui-preferences.json"
_MAX_BYTES = 64 * 1024
WINDOW_TEMPERATURE = 0.0
"""The temperature a window starts at when neither the command line nor a saved
preference names one: greedy, as the model cards' samplers default to. Only models
that sample (cuda_native MDLM and BD3) use the control. At 0.7 the 0.6B models
answered short chat turns with noise ("I am a dog") that greedy decoding did not."""
MAX_TEMPERATURE = 2.0
"""The window's control goes no higher; above it a reply is mostly noise."""


def valid_temperature(value: object) -> bool:
    if type(value) is int:  # exact: math.isfinite cannot take one past float's range
        return 0 <= value <= MAX_TEMPERATURE
    return type(value) is float and math.isfinite(value) and 0.0 <= value <= MAX_TEMPERATURE


@dataclass(frozen=True)
class GuiPreferences:
    last_model: str | None = None
    backend: str | None = None
    precision: str | None = None
    resources_expanded: bool = False
    temperature: float | None = None
    """None until the user sets one: the window then starts at WINDOW_TEMPERATURE."""


_TYPES = {"last_model": str, "backend": str, "precision": str, "resources_expanded": bool,
          "temperature": float}


def load_preferences(data_root: Path) -> GuiPreferences:
    """The saved preferences, or the defaults when there are none or they are damaged."""
    path = Path(data_root) / PREFERENCES_FILE
    try:
        raw = _retry_while_busy(path.read_bytes)
        if len(raw) > _MAX_BYTES:
            return GuiPreferences()
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=_object_without_duplicate_keys)
    except (OSError, UnicodeDecodeError, ValueError, RecursionError, ConfigurationError):
        return GuiPreferences()
    if not isinstance(document, dict):
        return GuiPreferences()
    values = {}
    for field in fields(GuiPreferences):
        if field.name not in document:
            continue
        value = document[field.name]
        expected = _TYPES[field.name]
        if expected is float and value is not None:
            if not valid_temperature(value):
                return GuiPreferences()
            value = float(value)
        elif value is not None and type(value) is not expected:
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
    descriptor, temporary_name = tempfile.mkstemp(prefix=".gui-preferences.", suffix=".tmp", dir=root)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _retry_while_busy(lambda: os.replace(temporary, root / PREFERENCES_FILE))
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        with suppress(FileNotFoundError):
            temporary.unlink()
