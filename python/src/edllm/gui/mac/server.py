"""The loopback server between the Mac chat page and the controller.

Bound to 127.0.0.1 on a free port. Every request must carry the launch token, as
the `token` query, an `Authorization: Bearer` header, or the HttpOnly cookie the
page receives, and must name this server in its Host header. The page reads
controller events by long-polling `/api/events`.

This is the Mac page's own server, with the frame channel for the live decode
view, stop, the live view switch, and a last answer to a page that is still
polling when the server stops.

author:Opus
"""
from __future__ import annotations

import http.cookies
import json
import logging
import mimetypes
import secrets
import threading
import time
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from . import events as ev
from .wire import event_to_json

_log = logging.getLogger("edllm.gui")
_COOKIE = "drt_token"
_POLL_SECONDS = 20.0
_MAX_BODY = 64 * 1024
_MAX_EVENTS = 256
_REQUEST_SECONDS = 5.0
_FAREWELL_SECONDS = 1.0
"""How long a stopping server waits for a page that is still polling to read the last events."""
_PAGE_NEAR_SECONDS = 2.0
_FLUSH_SECONDS = 0.1
_STATIC = {"/": "index.html", "/index.html": "index.html", "/app.js": "app.js", "/app.css": "app.css",
           "/decode.js": "decode.js", "/rich.js": "rich.js", "/model.js": "model.js",
           "/icon.svg": "icon.svg"}
_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
          ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml"}


class MacPageServer:
    def __init__(self, actions: Any, *, web_dir: Path, token: str | None = None) -> None:
        self._actions = actions
        self._web_dir = Path(web_dir)
        self._token = token or secrets.token_urlsafe(32)
        self._events: deque[dict[str, Any]] = deque(maxlen=_MAX_EVENTS)
        self._next_event = 0
        self._delivered = 0
        # The decode of the newest reply: a view, never history. Its beginning is
        # an ordered event, so a page knows which reply the canvas belongs to. Its
        # frames are not: a page reads them by cursor, however late it joins.
        self._decode: dict[str, Any] | None = None
        self._decode_id = 0
        self._decode_version = 0
        self._stopping = False
        self._changed = threading.Condition()
        self._last_seen = time.monotonic()
        self._waiting_polls = 0
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> str:
        server = self

        class Handler(_Handler):
            owner = server

        self._httpd = _QuietServer(("127.0.0.1", 0), Handler)
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, name="diffusorrt-gui-server", daemon=True
        )
        self._thread.start()
        return f"http://{self.host}/?token={self._token}"

    @property
    def host(self) -> str:
        return f"127.0.0.1:{self._httpd.server_address[1]}"

    def _cookie_name(self) -> str:
        """One name per port: a browser keeps one cookie jar for 127.0.0.1, and two windows can be open at once."""
        return f"{_COOKIE}_{self._httpd.server_address[1]}"

    def post(self, event: object) -> None:
        payload = event_to_json(event)
        with self._changed:
            if (isinstance(event, (ev.ChatShown, ev.ModelLoading, ev.Generating))
                    and self._decode is not None):
                # Another conversation, model or reply is on screen: the old canvas is not its.
                self._decode = None
                self._decode_version += 1
            if isinstance(event, ev.Snapshot):
                self._add_decode_in_flight(payload)
            self._events.append(payload)
            self._next_event += 1
            self._changed.notify_all()

    def post_decode(self, payload: dict[str, Any]) -> None:
        """Take one decode frame from the reply in flight."""
        kind = payload.get("type")
        with self._changed:
            if kind == "DecodeBegan":
                self._decode_id += 1
                begin = {**payload, "id": self._decode_id}
                self._decode = {"id": self._decode_id, "begin": begin, "frames": [], "end": None}
                self._events.append(begin)
                self._next_event += 1
            elif self._decode is None:
                return
            elif kind == "DecodeFrame":
                self._decode["frames"].append(payload)
            elif kind == "DecodeEnded":
                self._decode["end"] = payload
            else:
                return
            self._decode_version += 1
            self._changed.notify_all()

    def last_seen(self) -> float:
        """When the page was last heard from; now, while one of its polls is waiting."""
        with self._changed:
            return time.monotonic() if self._waiting_polls else self._last_seen

    def stop(self) -> None:
        with self._changed:
            self._stopping = True
            self._changed.notify_all()
            # A page that is still polling reads the last events first, `Closed`
            # among them. Otherwise all it could say is that the engine was lost.
            deadline = time.monotonic() + _FAREWELL_SECONDS
            waited = False
            while self._delivered < self._next_event and (
                    self._waiting_polls or time.monotonic() - self._last_seen < _PAGE_NEAR_SECONDS):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                waited = True
                self._changed.wait(remaining)
        if waited:
            time.sleep(_FLUSH_SECONDS)  # the answer is still on its way to the socket
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(5)

    # Used by the handler.

    def _authorized(self, handler: "_Handler", query: dict[str, list[str]], *,
                    cookie_allowed: bool) -> bool:
        """The token must come in the query or a Bearer header.

        The cookie admits only the page's own files: browsers also send it from a page
        on another 127.0.0.1 port, so it must never reach the API.
        """
        if handler.headers.get("Host") != self.host:
            return False
        offered = query.get("token", [None])[0]
        authorization = handler.headers.get("Authorization", "")
        if authorization.startswith("Bearer "):
            offered = offered or authorization[len("Bearer "):]
        if cookie_allowed:
            try:
                cookies = http.cookies.SimpleCookie(handler.headers.get("Cookie", ""))
            except http.cookies.CookieError:
                return False
            if self._cookie_name() in cookies:
                offered = offered or cookies[self._cookie_name()].value
        return offered is not None and offered.isascii() and secrets.compare_digest(offered, self._token)

    def _wait_for_events(self, after: int, *, decode_version: int = -1, decode_id: int = 0,
                         frames_seen: int = 0) -> dict[str, Any]:
        """Events after `after`, and what the page has not seen of the live decode.

        A page that passes no decode cursor (`decode_version` -1) is not woken by
        frames and gets none.
        """
        deadline = time.monotonic() + _POLL_SECONDS
        with self._changed:
            self._waiting_polls += 1
            try:
                while (self._next_event == after and not self._stopping
                       and decode_version in (-1, self._decode_version)):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._changed.wait(remaining)
                first = self._next_event - len(self._events)
                if not first <= after <= self._next_event:
                    return {"events": [], "next": self._next_event, "resync": True}
                payload: dict[str, Any] = {
                    "events": list(self._events)[after - first:], "next": self._next_event,
                }
                if decode_version != -1:
                    payload["decode_version"] = self._decode_version
                    payload["decode"] = self._decode_delta(decode_id, frames_seen)
                if self._next_event > self._delivered:
                    self._delivered = self._next_event
                    self._changed.notify_all()
                return payload
            finally:
                self._waiting_polls -= 1
                self._last_seen = time.monotonic()

    def _decode_delta(self, decode_id: int, frames_seen: int) -> dict[str, Any] | None:
        decode = self._decode
        if decode is None:
            return None
        # A page that has seen none of this decode gets every frame of it. `from`
        # says where these frames start, so a page never applies them out of place.
        frames = decode["frames"]
        start = min(max(0, frames_seen), len(frames)) if decode_id == decode["id"] else 0
        return {"id": decode["id"], "from": start, "frames": frames[start:], "end": decode["end"]}

    def _add_decode_in_flight(self, snapshot: dict[str, Any]) -> None:
        """A rebuilt view keeps the canvas of a reply that is still being decoded."""
        decode = self._decode
        if decode is None or decode["end"] is not None:
            return
        events = snapshot["events"]
        for index, item in enumerate(events):
            if item.get("type") == "Generating":
                events.insert(index + 1, decode["begin"])
                return

    def _act(self, path: str, body: dict[str, Any]) -> Any:
        actions = self._actions
        if path == "/api/send":
            return actions.send(str(body.get("text", "")))
        if path == "/api/cancel":
            return actions.cancel()
        if path == "/api/live_decode":
            enabled = body["enabled"]
            if type(enabled) is not bool:
                raise ValueError("enabled must be true or false")
            return actions.set_live_decode(enabled)
        if path == "/api/new_chat":
            return actions.new_chat()
        if path == "/api/reload_chat":
            return actions.reload_chat()
        if path == "/api/open_chat":
            return actions.open_chat(str(body["session_id"]))
        if path == "/api/answer_trust":
            return actions.answer_trust(bool(body.get("granted", False)))
        if path == "/api/switch_model":
            precision = body.get("precision")
            return actions.switch_model(str(body["spec"]), None if precision is None else str(precision))
        if path == "/api/preferences":
            return actions.set_resources_expanded(bool(body.get("resources_expanded", True)))
        if path == "/api/settings":
            return actions.configure(body.get("max_tokens"), body.get("system"))
        if path == "/api/choose_folder":
            return actions.choose_folder()
        if path == "/api/close":
            return actions.close()
        raise KeyError(path)


class _QuietServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        import sys

        error = sys.exc_info()[1]
        if isinstance(error, (ConnectionError, TimeoutError)):
            # A closed window drops its waiting poll; that is not a fault.
            _log.debug("gui http: %s dropped the connection: %s", client_address, error)
            return
        _log.error("gui http request failed", exc_info=True)


class _Handler(BaseHTTPRequestHandler):
    owner: MacPageServer
    server_version = "DiffusorRT"
    sys_version = ""

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(_REQUEST_SECONDS)

    def log_message(self, format: str, *args: Any) -> None:  # BaseHTTPRequestHandler's name
        _log.debug("gui http: " + format, *args)

    def do_GET(self) -> None:  # http.server's naming
        parts = urlsplit(self.path)
        query = parse_qs(parts.query)
        if not self.owner._authorized(self, query, cookie_allowed=parts.path in _STATIC):
            return self._send_json(HTTPStatus.FORBIDDEN, {"error": "forbidden"})
        self.owner._last_seen = time.monotonic()
        if parts.path == "/api/events":
            try:
                after = int(query.get("after", ["0"])[0])
                cursor = {
                    name: int(query.get(name, [default])[0])
                    for name, default in (("dv", "-1"), ("decode", "0"), ("frames", "0"))
                }
                if after < 0 or cursor["dv"] < -1 or cursor["decode"] < 0 or cursor["frames"] < 0:
                    raise ValueError
            except ValueError:
                return self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid event cursor"})
            payload = self.owner._wait_for_events(
                after, decode_version=cursor["dv"], decode_id=cursor["decode"],
                frames_seen=cursor["frames"])
            if payload.get("resync"):
                self.owner._actions.resync()
            return self._send_json(HTTPStatus.OK, payload)
        if parts.path == "/api/resources":
            return self._send_json(HTTPStatus.OK, self.owner._actions.resources())
        if parts.path == "/api/settings":
            return self._send_json(HTTPStatus.OK, self.owner._actions.settings())
        if parts.path == "/api/candidates":
            return self._send_json(HTTPStatus.OK, self.owner._actions.candidates())
        if parts.path in _STATIC:
            return self._send_file(_STATIC[parts.path], set_cookie="token" in query)
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:  # http.server's naming
        parts = urlsplit(self.path)
        lengths = self.headers.get_all("Content-Length", [])
        raw_length = lengths[0].strip() if lengths else "0"
        if (len(lengths) > 1 or not raw_length.isascii() or not raw_length.isdigit()
                or self.headers.get("Transfer-Encoding") is not None):
            self.close_connection = True
            return self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid content length"})
        try:
            length = int(raw_length)
        except ValueError:
            self.close_connection = True
            return self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid content length"})
        if length > _MAX_BODY:
            self.close_connection = True
            return self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "too large"})
        # Read the body before answering, even a refusal: on Windows, answering first
        # resets the connection while the client is still sending.
        raw = bytearray()
        deadline = time.monotonic() + _REQUEST_SECONDS
        try:
            while len(raw) < length:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError
                self.connection.settimeout(remaining)
                chunk = self.rfile.read1(length - len(raw))
                if not chunk:
                    self.close_connection = True
                    return self._send_json(HTTPStatus.BAD_REQUEST, {"error": "incomplete body"})
                raw.extend(chunk)
        except TimeoutError:
            self.close_connection = True
            return self._send_json(HTTPStatus.REQUEST_TIMEOUT, {"error": "request timed out"})
        if not self.owner._authorized(self, parse_qs(parts.query), cookie_allowed=False):
            return self._send_json(HTTPStatus.FORBIDDEN, {"error": "forbidden"})
        self.owner._last_seen = time.monotonic()
        try:
            body = json.loads(raw.decode("utf-8")) if raw.strip() else {}
            if not isinstance(body, dict):
                raise ValueError("body must be an object")
            accepted = self.owner._act(parts.path, body)
        except KeyError:
            return self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except (ValueError, UnicodeDecodeError) as error:
            return self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        self._send_json(HTTPStatus.OK, accepted if isinstance(accepted, dict) else {"accepted": bool(accepted)})

    def _send_json(self, status: HTTPStatus, payload: Any) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, name: str, *, set_cookie: bool) -> None:
        path = self.owner._web_dir / name
        try:
            data = path.read_bytes()
        except OSError:
            return self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        self.send_response(HTTPStatus.OK)
        content_type = _TYPES.get(path.suffix) or mimetypes.guess_type(name)[0] or "application/octet-stream"
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'")
        if set_cookie:
            self.send_header("Set-Cookie",
                             f"{self.owner._cookie_name()}={self.owner._token}; HttpOnly; SameSite=Strict; Path=/")
        self.end_headers()
        self.wfile.write(data)
