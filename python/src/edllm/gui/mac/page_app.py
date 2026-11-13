"""Start the chat page on a Mac: what `diffusorrt gui` runs there.

The page is served on 127.0.0.1 and shown in a WebKit window, or in the default
browser with `--browser`, or its address is printed with `--no-browser`. The
chat ends when the window closes, the page stops polling, or on Ctrl-C.

author:Opus
"""
from __future__ import annotations

import argparse
import logging
import logging.handlers
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any

from ...backend_ids import AUTO_BACKEND
from ...model_opening import OpenRequest
from . import events as ev
from .actions import MacActions
from .controller import MacChatController
from .launch import LaunchChoice, NeedsPicker, resolve_launch
from .preferences import load_preferences
from .server import MacPageServer

_log = logging.getLogger("edllm.gui")
LOG_BYTES = 1024 * 1024
HEARTBEAT_SECONDS = 10.0
"""A page silent this long, with no poll waiting, has been closed."""
FIRST_CONTACT_SECONDS = 120.0
"""How long a page we opened may take to load before the chat gives up."""
WAIT_STEP_SECONDS = 0.25
WEB_DIR = Path(__file__).with_name("web")


def run(command_spec: str | None, *, options: argparse.Namespace | None, log_to_file: bool) -> int:
    """Open the chat page and return once it has closed."""
    try:
        repository = _repository(options)
    except (OSError, ValueError) as error:
        _log.error("model repository configuration failed: %s", error)
        return 2
    data_root = Path(repository.data_root)
    if log_to_file:
        _log_to_file(data_root)

    preferences = load_preferences(data_root)
    actions = MacActions(repository, data_root, preferences)
    controller = MacChatController(
        open_request=_open_request(options),
        repository=repository,
        post=actions.on_event,
        # author:Opus -- the Mac windows have no output limit: a reply runs until the model ends it.
        max_tokens=None,
        system_prompt=getattr(options, "system", None),
        context_policy=getattr(options, "context_policy", "recent_only"),
        trust_remote_code=bool(getattr(options, "trust_remote_code", False)),
        decode_observer=actions.decode_observer,
        on_busy_changed=actions.on_busy,
    )
    server = MacPageServer(actions, web_dir=WEB_DIR)
    actions.attach(controller, server)
    result = 0
    try:
        url = server.start()
        actions.on_event(ev.PreferencesShown(preferences.resources_expanded))
        actions.on_event(ev.SettingsShown(**controller.settings()))
        # `auto` is what the command line says when nobody chose: the window decides then.
        backend = getattr(options, "backend", None)
        if backend == AUTO_BACKEND:
            backend = None
        choice = resolve_launch(
            command_spec=command_spec,
            command_backend=backend,
            command_precision=getattr(options, "precision", None),
            preferences=preferences,
            repository=repository,
        )
        if isinstance(choice, NeedsPicker):
            actions.initial_backend = backend
            actions.initial_precision = getattr(options, "precision", None)
            actions.on_event(ev.PickerRequested(choice.candidates, choice.reason))
        else:
            controller.start(choice)
        opened_a_page = not getattr(options, "no_browser", False)
        if opened_a_page and not getattr(options, "browser", False):
            from .window import MacWindow

            window = MacWindow(url, actions.close, actions.closed)
            actions.choose_folder_callback = window.choose_folder
            watcher = threading.Thread(target=_watch_window_page, args=(actions, server), daemon=True)
            watcher.start()
            window.run()
        else:
            if opened_a_page:
                webbrowser.open(url)
            else:
                print(f"Open the chat at {url}", flush=True)
            # A browser tab outlives the command that opened it; the page's polling says when it is gone.
            _wait_until_closed(actions, server, expect_page=opened_a_page)
    except KeyboardInterrupt:
        _log.info("interrupted; closing the chat")
    except Exception as error:
        _log.error("chat launch failed: %s", error)
        result = 2
    finally:
        _close(actions, controller)
        server.stop()
    return result


def _watch_window_page(actions: MacActions, server: Any) -> None:
    _wait_until_closed(actions, server, expect_page=True)
    actions.close()


def _wait_until_closed(actions: MacActions, server: Any, *, expect_page: bool) -> None:
    started = time.monotonic()
    while not actions.closed.is_set():
        now = time.monotonic()
        if server.last_seen() > started:
            if now - server.last_seen() > HEARTBEAT_SECONDS:
                _log.info("the chat page stopped responding; closing")
                return
        elif expect_page and now - started > FIRST_CONTACT_SECONDS:
            _log.error("the chat page never loaded; closing")
            return
        actions.closed.wait(WAIT_STEP_SECONDS)


def _close(actions: MacActions, controller: Any) -> None:
    actions.close()
    reported = False
    while not actions.closed.is_set():
        try:
            actions.closed.wait(WAIT_STEP_SECONDS)
            if actions.close_failed.is_set() and not reported:
                _log.error("cleanup incomplete; owner and transport retained. Retry Close in the page or press Ctrl-C")
                reported = True
        except KeyboardInterrupt:
            actions.close()
            reported = False


def _repository(options: argparse.Namespace | None) -> Any:
    if options is not None:
        from ...cli import _repository_for_args

        return _repository_for_args(options)
    from ...local_models import ModelRepository

    return ModelRepository()


def _open_request(options: argparse.Namespace | None):
    def build(choice: LaunchChoice) -> OpenRequest:
        return OpenRequest(
            spec=choice.spec,
            backend=choice.backend or AUTO_BACKEND,
            precision=choice.precision,
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
