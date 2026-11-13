"""Start the chat window: one entry for `diffusorrt gui` and the double-click launcher.

The page is served on 127.0.0.1 and shown in a standalone Edge window. The chat
ends when that window closes: the browser process exits, the page says goodbye
and no reloaded page follows, or the page stops polling. Ctrl-C on the terminal
ends it too.
"""
from __future__ import annotations

import argparse
import logging
import logging.handlers
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..application_state import resolve_data_root
from ..backend_ids import AUTO_BACKEND, CUDA_NATIVE_BACKEND
from ..model_opening import OpenRequest
from . import browser
from . import events as ev
from .controller import ChatController
from .gpu_memory import read_gpu_memory
from .launch import SOURCE_PICKED, LaunchChoice, NeedsPicker, prepared_models, resolve_launch
from .preferences import (
    WINDOW_TEMPERATURE,
    GuiPreferences,
    load_preferences,
    save_preferences,
    valid_temperature,
)
from .server import GuiServer

_log = logging.getLogger("edllm.gui")
LOG_BYTES = 1024 * 1024
HEARTBEAT_SECONDS = 10.0
"""A page silent this long, with no poll waiting, has been closed."""
LEAVE_GRACE_SECONDS = 3.0
"""A page that said goodbye has closed unless another page (a reload) calls within this."""
FIRST_CONTACT_SECONDS = 120.0
"""How long a window we opened may take to load the page before the chat gives up."""
HANDOFF_SECONDS = 5.0
"""An Edge process that exits this soon handed its window to an Edge already running."""
WAIT_STEP_SECONDS = 0.25
CLOSE_WAIT_SECONDS = 600.0
RELEASE_ATTEMPTS = 3
"""How often closing tries to release a model whose cleanup failed."""
WEB_DIR = Path(__file__).with_name("web")


def run(command_spec: str | None, *, options: argparse.Namespace | None, log_to_file: bool) -> int:
    """Open the chat window and return once it has closed."""
    # The log first: with no console, it is where a failure from here on is told.
    log_path = _log_to_file(resolve_data_root()) if log_to_file else None
    try:
        repository = _repository(options)
    except (OSError, ValueError) as error:
        _log.error("model repository configuration failed: %s", error)
        if log_path is not None:
            _tell_user(f"DiffusorRT could not start: {error}\n\nDetails are in {log_path}.")
        return 2
    data_root = Path(repository.data_root)

    preferences = load_preferences(data_root)
    device_ordinal = int(getattr(options, "device_ordinal", 0) or 0)
    # The command line's, then the one the user last set, then the window's own.
    temperature = next(value for value in (
        getattr(options, "temperature", None), preferences.temperature, WINDOW_TEMPERATURE)
        if value is not None)
    actions = _Actions(repository, data_root, preferences, device_ordinal=device_ordinal,
                       temperature=temperature)
    controller = ChatController(
        open_request=_open_request(options),
        repository=repository,
        post=actions.on_event,
        max_tokens=getattr(options, "max_tokens", 128),
        system_prompt=getattr(options, "system", None),
        context_policy=getattr(options, "context_policy", "recent_only"),
        trust_remote_code=bool(getattr(options, "trust_remote_code", False)),
        gpu_reader=lambda: read_gpu_memory(device_ordinal),
        temperature=temperature,
        seed=getattr(options, "seed", None),
    )
    server = GuiServer(actions, web_dir=WEB_DIR)
    actions.attach(controller, server)
    url = server.start()
    # Whatever fails from here, the model is released and the server stops.
    try:
        server.post(ev.PreferencesShown(preferences.resources_expanded, temperature))
        choice = resolve_launch(
            command_spec=command_spec,
            command_backend=getattr(options, "backend", AUTO_BACKEND),
            command_precision=getattr(options, "precision", None),
            preferences=preferences,
            repository=repository,
        )
        if isinstance(choice, NeedsPicker):
            server.post(ev.PickerRequested(choice.candidates, choice.reason))
        else:
            controller.start(choice)

        window = None
        opened_a_window = not getattr(options, "no_browser", False)
        if opened_a_window:
            window = browser.open_window(url, profile_dir=data_root / "gui-browser")
        else:
            print(f"Open the chat at {url}", flush=True)
        try:
            _wait_until_closed(actions, server, window, expect_page=opened_a_window)
        except KeyboardInterrupt:
            _log.info("interrupted; closing the chat")
    finally:
        _close(actions, controller)
        server.stop()
    return 0


def report_launch_failure(error: BaseException) -> None:
    """The double-click launcher's last word: in the log, and in a box the user sees."""
    _log.error("the chat window failed", exc_info=error)
    _tell_user(f"DiffusorRT stopped unexpectedly: {error}\n\n"
               "Details are in gui.log under the DiffusorRT application data folder.")


def _tell_user(text: str) -> None:
    import ctypes

    windll = getattr(ctypes, "windll", None)
    if windll is not None:
        windll.user32.MessageBoxW(None, text, "DiffusorRT", 0x10)  # MB_ICONERROR


def _wait_until_closed(actions: "_Actions", server: Any, window: Any, *, expect_page: bool) -> None:
    started = time.monotonic()
    while not actions.closed.is_set():
        now = time.monotonic()
        if window is not None and window.poll() is not None:
            if now - started < HANDOFF_SECONDS:
                # Same profile as a window still open: the page now lives in that Edge.
                _log.info("Edge handed the window to a running Edge; following the page instead")
                window = None
            else:
                _log.info("the chat window was closed")
                return
        if server.page_gone(LEAVE_GRACE_SECONDS):
            _log.info("the chat window was closed")
            return
        if server.last_seen() > started:
            if now - server.last_seen() > HEARTBEAT_SECONDS:
                _log.info("the chat page stopped responding; closing")
                return
        elif expect_page and now - started > FIRST_CONTACT_SECONDS:
            _log.error("the chat page never loaded; closing")
            return
        actions.closed.wait(WAIT_STEP_SECONDS)


def _close(actions: "_Actions", controller: Any) -> None:
    """Close the controller: wait out a running job, retry a release that failed.

    The window is gone by now, so a model that will not release is retried a few
    times and then left to the process exit, which frees the GPU in any case.
    """
    from .controller import ModelReleaseError

    deadline = time.monotonic() + CLOSE_WAIT_SECONDS
    failures = 0
    while time.monotonic() < deadline:
        try:
            if controller.request_close():
                return
        except ModelReleaseError as error:
            failures += 1
            if failures >= RELEASE_ATTEMPTS:
                _log.error("the model could not be released (%s); exiting frees the GPU", error)
                return
        # A reply or load is running and closes the chat when it returns, or a
        # release just failed and is tried again.
        if actions.closed.wait(WAIT_STEP_SECONDS):
            return


class _Actions:
    """What the page may ask for, mapped onto the controller and the preferences."""

    def __init__(self, repository: Any, data_root: Path, preferences: GuiPreferences, *,
                 device_ordinal: int = 0, temperature: float = WINDOW_TEMPERATURE) -> None:
        self._repository = repository
        self._data_root = data_root
        self._preferences = preferences
        self._device_ordinal = device_ordinal
        self._temperature = temperature
        self._lock = threading.Lock()
        self._controller: Any = None
        self._server: Any = None
        self.closed = threading.Event()

    def attach(self, controller: Any, server: Any) -> None:
        self._controller, self._server = controller, server

    def on_event(self, event: object) -> None:
        if isinstance(event, ev.ModelReady):
            # What "auto" picked is not a choice to repeat: the next launch decides again.
            self._remember(last_model=event.spec, backend=event.chosen_backend,
                           precision=event.precision if event.chosen_backend else None)
        if isinstance(event, ev.Closed):
            self.closed.set()
        if self._server is not None:
            self._server.post(event)

    def send(self, text: str) -> bool:
        return self._controller.send(text)

    def stop(self) -> bool:
        return self._controller.stop()

    def delete_chat(self, session_id: str, revision: int) -> bool:
        return self._controller.delete_chat(session_id, revision)

    def new_chat(self) -> bool:
        return self._controller.new_chat()

    def open_chat(self, session_id: str) -> bool:
        return self._controller.open_chat(session_id)

    def answer_trust(self, granted: bool) -> bool:
        self._controller.answer_trust(granted)
        return True

    def switch_model(self, spec: str, precision: str | None) -> bool:
        backend = CUDA_NATIVE_BACKEND if precision is not None else None
        return self._controller.switch_model(LaunchChoice(spec, backend, precision, SOURCE_PICKED))

    def set_resources_expanded(self, expanded: bool) -> bool:
        self._remember(resources_expanded=expanded)
        self._show_preferences()
        return True

    def set_temperature(self, temperature: float) -> bool:
        if not valid_temperature(temperature):
            return False
        temperature = float(temperature)
        # One step under the lock: two pages' changes cannot leave the controller at
        # one value and the saved preference, or its echo, at the other.
        with self._lock:
            self._temperature = temperature
            self._controller.set_temperature(temperature)
            self._remember_locked(temperature=temperature)
            self._show_preferences()
        return True

    def _show_preferences(self) -> None:
        if self._server is not None:  # A reload replays the latest.
            self._server.post(ev.PreferencesShown(self._preferences.resources_expanded,
                                                  self._temperature))

    def resources(self) -> dict[str, int] | None:
        # NVML numbers devices in PCI order and CUDA fastest first; on one GPU they agree.
        reading = read_gpu_memory(self._device_ordinal)
        if reading is None:
            return None
        return {"gpu_memory_used_bytes": reading.used_bytes,
                "gpu_memory_total_bytes": reading.total_bytes}

    def candidates(self) -> list[dict[str, str]]:
        return [{"model_id": str(model.model_id)} for model in prepared_models(self._repository)]

    def _remember(self, **changes: Any) -> None:
        with self._lock:
            self._remember_locked(**changes)

    def _remember_locked(self, **changes: Any) -> None:
        self._preferences = replace(self._preferences, **changes)
        try:
            save_preferences(self._data_root, self._preferences)
        except OSError:
            _log.exception("saving window preferences failed")


def _repository(options: argparse.Namespace | None) -> Any:
    if options is not None:
        from ..cli import _repository_for_args

        return _repository_for_args(options)
    from ..local_models import ModelRepository

    return ModelRepository()


def _open_request(options: argparse.Namespace | None):
    def build(choice: LaunchChoice) -> OpenRequest:
        return OpenRequest(
            spec=choice.spec,
            backend=choice.backend or AUTO_BACKEND,
            precision=choice.precision,
            device_ordinal=getattr(options, "device_ordinal", 0),
            device_memory_budget_bytes=getattr(options, "device_memory_budget_bytes", 0),
            offline=True,
            allow_prepare=False,  # The window never downloads or compiles a model.
        )

    return build


def _log_to_file(data_root: Path) -> Path:
    """With no console, everything goes to one rotating file under the data root."""
    directory = data_root / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "gui.log"
    handler = logging.handlers.RotatingFileHandler(
        path, maxBytes=LOG_BYTES, backupCount=1, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
    return path
