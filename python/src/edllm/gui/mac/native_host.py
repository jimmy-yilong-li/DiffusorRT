"""The engine host behind the native Mac window.

The window starts this process and owns its lifetime. Commands arrive as one
JSON object per line on stdin; events leave as one JSON object per line on
stdout. There is no network listener and no page. When stdin closes, because
the window quit or died, the host releases the model and exits.

Model, history and cleanup ownership stay in `ChatController`, as for every
other window. This module adds the transport and the live decode frames.

author:Opus
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import signal
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

from ...backend_ids import AUTO_BACKEND
from ...model_opening import OpenRequest
from . import events as ev
from .controller import MacChatController
from .launch import LaunchChoice, NeedsPicker, resolve_launch
from .preferences import load_preferences
from .wire import event_to_json

_log = logging.getLogger("edllm.gui")

PROTOCOL = 1
RESOURCE_POLL_SECONDS = 0.5
CLOSE_WAIT_SECONDS = 600.0
ORPHAN_GRACE_SECONDS = 15.0
"""How long the host may outlive its window before it stops waiting for a clean release."""
MAX_COMMAND_BYTES = 1024 * 1024


class Writer:
    """Serializes events onto the protocol stream from one thread.

    Callers never block on the pipe: a slow reader delays this thread, not the
    decode loop. What waits here is not limited by this class; the one producer
    that could fill it, the live decode, asks `caught_up` before it adds a frame.
    """

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._queue: queue.SimpleQueue[dict[str, Any] | None] = queue.SimpleQueue()
        self._unwritten = 0
        self._count = threading.Lock()
        self._thread = threading.Thread(target=self._drain, name="diffusorrt-host-writer", daemon=True)
        self._thread.start()

    def post(self, payload: dict[str, Any]) -> None:
        with self._count:
            self._unwritten += 1
        self._queue.put(payload)

    def caught_up(self) -> bool:
        """Whether everything posted so far has been handed to the pipe."""
        with self._count:
            return self._unwritten == 0

    def close(self, timeout: float = 5.0) -> None:
        self._queue.put(None)
        self._thread.join(timeout)

    def _drain(self) -> None:
        while True:
            payload = self._queue.get()
            if payload is None:
                return
            try:
                self._stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
                self._stream.write("\n")
                self._stream.flush()
            except (OSError, ValueError):
                # The window is gone. stdin's end-of-file closes the host. The count
                # stays above zero, so no more frames are made for nobody.
                return
            with self._count:
                self._unwritten -= 1


class NativeHost:
    """Maps the window's commands onto the controller and its events onto the wire."""

    def __init__(self, repository: Any, post: Callable[[dict[str, Any]], None], *,
                 options: argparse.Namespace, controller_factory: Callable[..., Any] = MacChatController,
                 actions_factory: Callable[..., Any] | None = None,
                 caught_up: Callable[[], bool] | None = None) -> None:
        if actions_factory is None:
            from .actions import MacActions as actions_factory
        self._repository = repository
        self._post = post
        self._caught_up = caught_up
        self._options = options
        data_root = Path(repository.data_root)
        self._preferences = load_preferences(data_root)
        # The same actions as the page window: one set of rules for every window.
        self.actions = actions_factory(repository, data_root, self._preferences)
        if options.no_live_decode:
            self.actions.set_live_decode(False, remember=False)
        self.controller = controller_factory(
            open_request=self._open_request,
            repository=repository,
            post=self.actions.on_event,
            max_tokens=options.max_tokens,
            system_prompt=options.system,
            context_policy=options.context_policy,
            trust_remote_code=options.trust_remote_code,
            decode_observer=self.actions.decode_observer,
            on_busy_changed=self.actions.on_busy,
        )
        self.actions.attach(self.controller, self)
        self._resource_stamp: object = None

    # The transport the actions post to.

    def post(self, event: object) -> None:
        self._post(event_to_json(event))

    def post_decode(self, payload: dict[str, Any]) -> None:
        self._post(payload)

    def decode_ready(self) -> bool:
        """Whether the window has taken what was sent: a frame made now would not wait behind others."""
        return self._caught_up is None or self._caught_up()

    def start(self) -> None:
        self._post({
            "type": "Hello",
            "protocol": PROTOCOL,
            "pid": os.getpid(),
            "sdk_version": _sdk_version(),
            "python": sys.version.split()[0],
            "cache_root": str(self._repository.cache_root),
            "data_root": str(self._repository.data_root),
            "live_decode": self.actions.live_decode,
        })
        self.actions.on_event(ev.SettingsShown(**self.controller.settings()))
        choice = resolve_launch(
            command_spec=self._options.model,
            command_backend=self._options.backend,
            command_precision=self._options.precision,
            preferences=self._preferences,
            repository=self._repository,
        )
        if isinstance(choice, NeedsPicker):
            self.actions.initial_backend = self._options.backend
            self.actions.initial_precision = self._options.precision
            self.actions.on_event(ev.PickerRequested(choice.candidates, choice.reason))
        else:
            self.controller.start(choice)

    def handle_line(self, line: str) -> None:
        """Run one command line and acknowledge it. A bad line is refused, never fatal."""
        identifier: object = None
        try:
            message = json.loads(line)
            if not isinstance(message, dict):
                raise ValueError("a command must be a JSON object")
            identifier = message.get("id")
            name = message.get("command")
            handler = self._commands().get(name) if isinstance(name, str) else None
            if handler is None:
                raise ValueError(f"unknown command: {name!r}")
            result = handler(message)
        except (ValueError, KeyError, TypeError) as error:
            reason = f"missing field: {error}" if isinstance(error, KeyError) else str(error)
            self._post({"type": "Ack", "id": identifier, "accepted": False, "error": reason})
            return
        except Exception as error:  # the window must hear about it, and keep running
            _log.exception("command failed")
            self._post({"type": "Ack", "id": identifier, "accepted": False,
                        "error": f"unexpected error: {error}"})
            return
        accepted = result is not False
        self._post({"type": "Ack", "id": identifier, "accepted": accepted,
                    "result": None if isinstance(result, bool) else result})

    def poll_resources(self) -> None:
        """Forward the controller's latest completed resource reading, once."""
        reading = self.controller.resources()
        stamp = reading.get("sampled_at")
        if stamp != self._resource_stamp:
            self._resource_stamp = stamp
            self._post({"type": "ResourcesShown", **reading})

    # Commands.

    def _commands(self) -> dict[str, Callable[[dict[str, Any]], Any]]:
        actions = self.actions
        return {
            "open_model": lambda m: actions.switch_model(_text(m, "spec"), _optional_text(m, "precision")),
            "send": lambda m: actions.send(_text(m, "text")),
            "cancel": lambda m: actions.cancel(),
            "new_chat": lambda m: actions.new_chat(),
            "open_chat": lambda m: actions.open_chat(_text(m, "session_id")),
            "reload_chat": lambda m: actions.reload_chat(),
            "answer_trust": lambda m: actions.answer_trust(_flag(m, "granted")),
            "configure": lambda m: actions.configure(_optional_positive(m, "max_tokens"), _optional_text(m, "system")),
            "live_decode": lambda m: actions.set_live_decode(_flag(m, "enabled")),
            "candidates": lambda m: actions.candidates(),
            "settings": lambda m: actions.settings(),
            "resync": lambda m: actions.resync(),
            "close": lambda m: actions.close(),
        }

    def _open_request(self, choice: LaunchChoice) -> OpenRequest:
        return OpenRequest(
            spec=choice.spec,
            backend=choice.backend or AUTO_BACKEND,
            precision=choice.precision,
            offline=True,
            allow_prepare=False,  # The window never downloads or compiles a model.
        )


class OrphanDeadline:
    """When the host must give up on a clean release after its window is gone."""

    def __init__(self, grace: float = ORPHAN_GRACE_SECONDS,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._grace = grace
        self._clock = clock
        self._deadline: float | None = None

    def start(self) -> None:
        if self._deadline is None:
            self._deadline = self._clock() + self._grace

    def expired(self) -> bool:
        return self._deadline is not None and self._clock() >= self._deadline


def _text(message: dict[str, Any], name: str) -> str:
    value = message[name]
    if not isinstance(value, str):
        raise ValueError(f"{name} must be text")
    return value


def _optional_text(message: dict[str, Any], name: str) -> str | None:
    value = message.get(name)
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{name} must be text")
    return value


def _flag(message: dict[str, Any], name: str) -> bool:
    value = message[name]
    if type(value) is not bool:
        raise ValueError(f"{name} must be true or false")
    return value


def _optional_positive(message: dict[str, Any], name: str) -> int | None:
    value = message.get(name)
    if value is not None and (type(value) is not int or value < 1):
        raise ValueError(f"{name} must be a positive integer, or none")
    return value


def _sdk_version() -> str | None:
    from importlib import metadata

    try:
        return metadata.version("diffusor-rt")
    except metadata.PackageNotFoundError:
        return None


def read_command_lines(descriptor: int, *, limit: int = MAX_COMMAND_BYTES):
    """Command lines from a raw descriptor; None stands for one that was too long.

    Reads the descriptor directly. A buffered reader would hold its lock while
    this thread waits, and interpreter shutdown aborts when it cannot take it.
    """
    pending = b""
    discarding = False
    while True:
        try:
            chunk = os.read(descriptor, 65536)
        except OSError:
            return
        if not chunk:
            return
        pending += chunk
        while True:
            line, newline, rest = pending.partition(b"\n")
            if not newline:
                break
            pending = rest
            if discarding:
                discarding = False
            elif len(line) > limit:
                yield None
            else:
                yield line
        if len(pending) > limit:
            # Drop the rest of this line as it arrives, and refuse it once.
            pending = b""
            if not discarding:
                discarding = True
                yield None


def take_protocol_stream() -> TextIO:
    """Keep the original stdout for the protocol; send every other write to stderr.

    A library that prints to stdout would otherwise corrupt the event stream.
    """
    protocol = os.fdopen(os.dup(1), "w", encoding="utf-8", newline="\n")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return protocol


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m edllm.gui.mac.native_host",
        description="Engine host for the DiffusorRT Mac window. Started by the window, not by hand.",
    )
    parser.add_argument("--model", default=None, help="Open this model instead of the remembered one.")
    parser.add_argument("--backend", default=None)
    parser.add_argument("--precision", default=None)
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="A limit for every reply. Without it a reply runs until the model ends it.")
    parser.add_argument("--system", default=None)
    parser.add_argument("--context-policy", choices=("recent_only", "bounded_summary"), default="recent_only")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--model-catalog", default=None)
    parser.add_argument("--no-live-decode", action="store_true",
                        help="Start with the live decode view off; no decode data is collected.")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    options = build_parser().parse_args(argv)
    stream = take_protocol_stream()
    logging.basicConfig(
        level=getattr(logging, str(options.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    writer = Writer(stream)
    try:
        from ...local_models import ModelRepository

        repository = ModelRepository(
            cache_root=options.cache_root, data_root=options.data_root,
            catalog_path=options.model_catalog,
        )
        host = NativeHost(repository, writer.post, options=options, caught_up=writer.caught_up)
    except Exception as error:  # the window shows this instead of a silent exit
        _log.exception("the engine host could not start")
        writer.post({"type": "HostFailed", "reason": str(error)})
        writer.close()
        return 2

    actions = host.actions
    orphan = OrphanDeadline()

    def abandon(reason: str) -> None:
        # Nobody is waiting for a reply any more: stop one that is running and
        # release the model. If a job is stuck (a blocked read, say), the
        # deadline ends the process instead of leaving it behind.
        _log.info("%s; releasing the model", reason)
        orphan.start()
        actions.cancel()
        actions.close()

    def read_commands() -> None:
        for raw in read_command_lines(0):
            if raw is None:
                writer.post({"type": "Ack", "id": None, "accepted": False,
                             "error": "command too large"})
                continue
            line = raw.decode("utf-8", "replace").strip()
            if line:
                host.handle_line(line)
        abandon("the window closed its end")

    def on_signal(_number: int, _frame: object) -> None:
        abandon("asked to stop")

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    host.start()
    threading.Thread(target=read_commands, name="diffusorrt-host-reader", daemon=True).start()

    reported = False
    deadline: float | None = None
    while not actions.closed.is_set():
        actions.closed.wait(RESOURCE_POLL_SECONDS)
        if orphan.expired():
            _log.error("the model was not released within %.0f s of the window closing; exiting",
                       ORPHAN_GRACE_SECONDS)
            writer.close(timeout=1.0)
            # A blocked worker cannot be joined; leave without waiting for it.
            os._exit(3)
        try:
            host.poll_resources()
        except Exception:
            _log.debug("resource poll failed", exc_info=True)
        if actions.close_failed.is_set():
            if not reported:
                _log.error("cleanup incomplete; the model owner is retained for a retry")
                reported = True
            deadline = deadline or time.monotonic() + CLOSE_WAIT_SECONDS
            if time.monotonic() > deadline:
                writer.close()
                return 1
        else:
            reported, deadline = False, None
    writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
