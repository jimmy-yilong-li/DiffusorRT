"""What a Mac window may ask for, mapped onto the controller and the preferences.

Used by both Mac windows: the page asks through its server, the native app
through its engine host.

author:Opus
"""
from __future__ import annotations

import logging
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

from . import events as ev
from .decode_frames import DecodeFrames
from .launch import SOURCE_PICKED, LaunchChoice
from .preferences import GuiPreferences, save_preferences

_log = logging.getLogger("edllm.gui")


class MacActions:
    def __init__(self, repository: Any, data_root: Path, preferences: GuiPreferences) -> None:
        self._repository = repository
        self._data_root = data_root
        self._preferences = preferences
        self._lock = threading.Lock()
        self._controller: Any = None
        self._server: Any = None
        self.closed = threading.Event()
        self.close_failed = threading.Event()
        self._picker: ev.PickerRequested | None = None
        self.choose_folder_callback = None
        self.initial_backend = None
        self.initial_precision = None
        self._live_decode = preferences.live_decode
        self._busy = False

    def attach(self, controller: Any, server: Any) -> None:
        self._controller, self._server = controller, server

    def on_event(self, event: object) -> None:
        if isinstance(event, ev.PickerRequested):
            self._picker = event
        if isinstance(event, ev.ModelLoading):
            self._picker = None
        if isinstance(event, ev.ModelReady):
            self.initial_backend = self.initial_precision = None
            self._remember(last_model=event.spec, backend=event.backend, precision=event.precision)
        if isinstance(event, ev.Closed):
            self.closed.set()
        if isinstance(event, ev.Closing):
            self.close_failed.clear()
        if isinstance(event, ev.CloseFailed):
            self.close_failed.set()
        if isinstance(event, ev.Snapshot):
            prefix = (ev.PreferencesShown(self._preferences.resources_expanded),)
            if self._picker is not None:
                prefix += (self._picker,)
            # A rebuilt view says whether the worker is busy, as the live events do.
            event = ev.Snapshot((*prefix, *event.events, ev.Busy(self._busy)))
        if self._server is not None:
            self._server.post(event)

    def on_busy(self, busy: bool) -> None:
        self._busy = busy
        self.on_event(ev.Busy(busy))

    @property
    def live_decode(self) -> bool:
        return self._live_decode

    def decode_observer(self) -> DecodeFrames | None:
        """One reply's live observer. None while the live view is off: nothing is collected."""
        if not self._live_decode or self._server is None:
            return None
        # A transport that queues frames behind a pipe says when it has caught up;
        # one that keeps them for a cursor (the page server) has no such limit to keep.
        return DecodeFrames(self._server.post_decode, ready=getattr(self._server, "decode_ready", None))

    def set_live_decode(self, enabled: bool, *, remember: bool = True) -> dict[str, bool]:
        self._live_decode = bool(enabled)
        if remember:
            self._remember(live_decode=self._live_decode)
        return {"live_decode": self._live_decode}

    def cancel(self) -> bool:
        return self._controller.cancel()

    def send(self, text: str) -> bool:
        return self._controller.send(text)

    def new_chat(self) -> bool:
        return self._controller.new_chat()

    def open_chat(self, session_id: str) -> bool:
        return self._controller.open_chat(session_id)

    def reload_chat(self) -> bool:
        return self._controller.reload_chat()

    def answer_trust(self, granted: bool) -> bool:
        self._controller.answer_trust(granted)
        return True

    def switch_model(self, spec: str, precision: str | None = None) -> bool:
        """Open another model. The launch's own backend and precision, when given, still apply."""
        if precision is None:
            precision = self.initial_precision
        return self._controller.switch_model(
            LaunchChoice(spec, self.initial_backend, precision, SOURCE_PICKED))

    def settings(self) -> dict[str, Any]:
        return {**self._controller.settings(), "native_folder": self.choose_folder_callback is not None,
                "cache_root": str(self._repository.cache_root), "data_root": str(self._data_root),
                "live_decode": self._live_decode}

    def configure(self, max_tokens: int | None, system: str | None) -> bool:
        return self._controller.configure(max_tokens=max_tokens, system_prompt=system)

    def choose_folder(self) -> dict[str, Any]:
        if self.choose_folder_callback is None:
            raise ValueError("Folder selection is available in the window; enter a path here.")
        return {"path": self.choose_folder_callback()}

    def set_resources_expanded(self, expanded: bool) -> bool:
        self._remember(resources_expanded=expanded)
        return True

    def close(self) -> bool:
        self._controller.request_close()
        return True  # Accepted intent, not completion; Closed is the completion event.

    def resync(self) -> bool:
        return self._controller.request_snapshot()

    def resources(self) -> dict[str, Any]:
        return self._controller.resources()

    def candidates(self) -> list[dict[str, str]]:
        from ...local_models import CACHE_CACHED

        try:
            models = self._repository.cached_models()
        except (OSError, ValueError):
            _log.exception("listing prepared models failed")
            return []
        return [{"model_id": str(model.model_id)} for model in models if model.status == CACHE_CACHED]

    def _remember(self, **changes: Any) -> None:
        with self._lock:
            self._preferences = replace(self._preferences, **changes)
            try:
                save_preferences(self._data_root, self._preferences)
            except OSError:
                _log.exception("saving window preferences failed")
