"""The loopback server between the chat page and the controller.

Bound to 127.0.0.1 on a free port. Every API request must carry the launch
token, as the `token` query or an `Authorization: Bearer` header, and must name
this server in its Host header. The page's own files also accept the HttpOnly
cookie the page receives, which holds a secret of its own, never the token. The
page reads controller events by long-polling `/api/events`.

The page names itself in every API call (`X-DiffusorRT-Page`), so a page that
leaves can be told from one that reloads: leaving ends the chat only when no
other page has come instead within a short grace.
"""
from __future__ import annotations

import json
import logging
import mimetypes
import secrets
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from .wire import event_to_json

_log = logging.getLogger("edllm.gui")
_COOKIE = "drt_token"
_PAGE_HEADER = "X-DiffusorRT-Page"
_LOADING_PAGE = "\0loading"
"""Contact from a page whose script has not named itself yet: its own files."""
_POLL_SECONDS = 20.0
_MAX_BODY = 64 * 1024
_STATIC = {"/": "index.html", "/index.html": "index.html", "/app.js": "app.js", "/app.css": "app.css",
           "/logo.png": "logo.png"}
_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
          ".css": "text/css; charset=utf-8", ".png": "image/png"}


_ENDS_FORMING = frozenset({"ReplyForming", "ReplyShown", "ReplyFailed", "ReplyStopped"})


def _same(offered: str | None, expected: str) -> bool:
    # Bytes: a token with characters outside ASCII is simply wrong, never an error.
    return offered is not None and secrets.compare_digest(
        offered.encode("utf-8", "replace"), expected.encode("utf-8"))


class GuiServer:
    def __init__(self, actions: Any, *, web_dir: Path, token: str | None = None) -> None:
        self._actions = actions
        self._web_dir = Path(web_dir)
        self._token = token or secrets.token_urlsafe(32)
        # The page-files cookie gets its own secret: every other 127.0.0.1 server
        # the browser visits receives it, so it must not be the API token.
        self._cookie_secret = secrets.token_urlsafe(32)
        self._events: list[dict[str, Any]] = []
        self._forming: int | None = None
        self._changed = threading.Condition()
        self._last_seen = time.monotonic()
        self._waiting_polls = 0
        self._page: str | None = None
        self._left: tuple[str | None, float] | None = None
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

    def post(self, event: object) -> None:
        payload = event_to_json(event)
        with self._changed:
            # A reply forms in up to one state per decode step; only the newest
            # matters, so an older one becomes a stub and the log stays small.
            if self._forming is not None and payload["type"] in _ENDS_FORMING:
                self._events[self._forming] = {"type": "Superseded"}
                self._forming = None
            if payload["type"] == "ReplyForming":
                self._forming = len(self._events)
            self._events.append(payload)
            self._changed.notify_all()

    def last_seen(self) -> float:
        """When the page was last heard from; now, while one of its polls is waiting."""
        with self._changed:
            return time.monotonic() if self._waiting_polls else self._last_seen

    def page_gone(self, grace_seconds: float) -> bool:
        """Whether the page left `grace_seconds` ago and no other page came instead."""
        with self._changed:
            return self._left is not None and time.monotonic() - self._left[1] >= grace_seconds

    def stop(self) -> None:
        with self._changed:
            self._changed.notify_all()
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
        if offered is None and cookie_allowed:
            return _same(self._cookie_from(handler), self._cookie_secret)
        return _same(offered, self._token)

    def _cookie_from(self, handler: "_Handler") -> str | None:
        """This server's cookie, read past the ones other local sites set.

        SimpleCookie gives up on a header with a name it rejects ("a/b"), and
        the browser sends every 127.0.0.1 site's cookies here.
        """
        prefix = self._cookie_name() + "="
        for pair in handler.headers.get("Cookie", "").split(";"):
            pair = pair.strip()
            if pair.startswith(prefix):
                return pair[len(prefix):]
        return None

    def _cookie_name(self) -> str:
        """Per port: browsers share cookies across ports, and two windows can run at once."""
        return f"{_COOKIE}_{self._httpd.server_address[1]}"

    def _contact(self, page: str | None) -> None:
        """A request from `page` started; another page's contact undoes a leave.

        A caller that names no page is not a page (a script holding the token), so
        it neither replaces the page nor keeps a closed window open.
        """
        with self._changed:
            self._last_seen = time.monotonic()
            if page is None:
                return
            self._page = page
            if self._left is not None and page != self._left[0]:
                self._left = None

    def _page_left(self, page: str | None) -> None:
        with self._changed:
            # A beacon that lost the race to its own reload names a page already replaced.
            if self._page == page:
                self._left = (page, time.monotonic())

    def _wait_for_events(self, after: int) -> tuple[list[dict[str, Any]], int]:
        deadline = time.monotonic() + _POLL_SECONDS
        with self._changed:
            self._waiting_polls += 1
            try:
                while len(self._events) <= after:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._changed.wait(remaining)
                return self._events[after:], len(self._events)
            finally:
                self._waiting_polls -= 1
                self._last_seen = time.monotonic()

    def _act(self, path: str, body: dict[str, Any], page: str | None) -> Any:
        actions = self._actions
        if path == "/api/send":
            return actions.send(str(body.get("text", "")))
        if path == "/api/new_chat":
            return actions.new_chat()
        if path == "/api/open_chat":
            return actions.open_chat(_text_field(body, "session_id"))
        if path == "/api/answer_trust":
            granted = body.get("granted")
            if type(granted) is not bool:
                raise ValueError("answer_trust needs granted: true or false")
            return actions.answer_trust(granted)
        if path == "/api/switch_model":
            precision = body.get("precision")
            if precision is not None and not isinstance(precision, str):
                raise ValueError("switch_model's precision must be text or null")
            return actions.switch_model(_text_field(body, "spec"), precision)
        if path == "/api/preferences":
            # Each preference the page names; a temperature it cannot take is refused.
            accepted = True
            if "resources_expanded" in body:
                accepted = actions.set_resources_expanded(bool(body["resources_expanded"]))
            if "temperature" in body:
                temperature = body["temperature"]
                if type(temperature) not in (int, float):
                    raise ValueError("preferences' temperature must be a number")
                accepted = actions.set_temperature(temperature) and accepted
            return accepted
        if path == "/api/stop":
            return actions.stop()
        if path == "/api/delete_chat":
            session_id, revision = body.get("session_id"), body.get("revision")
            if not isinstance(session_id, str) or not session_id:
                raise ValueError("delete_chat needs the conversation's session_id")
            # The revision the list showed: a conversation changed since is kept.
            if type(revision) is not int or revision < 0:
                raise ValueError("delete_chat needs the revision the list showed")
            return actions.delete_chat(session_id, revision)
        if path == "/api/close":
            self._page_left(page)  # The page is unloading; a reload comes back at once.
            return True
        raise _UnknownPath(path)


class _UnknownPath(Exception):
    pass


def _text_field(body: dict[str, Any], name: str) -> str:
    value = body.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} is required")
    return value


def _content_length(value: str | None) -> int | None:
    """The declared body length; None when it is not a plain non-negative integer."""
    if value is None or value == "":
        return 0
    if not value.isascii() or not value.isdigit():
        return None
    return int(value)


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
    owner: GuiServer
    server_version = "DiffusorRT"
    sys_version = ""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - BaseHTTPRequestHandler's name
        _log.debug("gui http: " + format, *args)

    def do_GET(self) -> None:  # noqa: N802 - http.server's naming
        parts = urlsplit(self.path)
        query = parse_qs(parts.query)
        if not self.owner._authorized(self, query, cookie_allowed=parts.path in _STATIC):
            return self._send_json(HTTPStatus.FORBIDDEN, {"error": "forbidden"})
        self.owner._contact(self._arriving_page(parts.path) if parts.path in _STATIC else self._page(query))
        if parts.path in _STATIC:
            return self._send_file(_STATIC[parts.path], set_cookie="token" in query)
        if parts.path == "/api/events":
            try:
                after = max(0, int(query.get("after", ["0"])[0] or 0))
            except ValueError:
                return self._send_json(HTTPStatus.BAD_REQUEST, {"error": "after must be a number"})
            events, next_index = self.owner._wait_for_events(after)
            return self._send_json(HTTPStatus.OK, {"events": events, "next": next_index})
        readers = {"/api/resources": self.owner._actions.resources,
                   "/api/candidates": self.owner._actions.candidates}
        if parts.path not in readers:
            return self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        try:
            reading = readers[parts.path]()
        except Exception:  # noqa: BLE001 - answered, and logged once
            return self._fail(parts.path)
        self._send_json(HTTPStatus.OK, reading)

    def do_POST(self) -> None:  # noqa: N802 - http.server's naming
        parts = urlsplit(self.path)
        length = _content_length(self.headers.get("Content-Length"))
        if length is None:
            # Where the body ends is unknown, so the connection cannot be reused.
            self.close_connection = True
            return self._send_json(HTTPStatus.BAD_REQUEST, {"error": "bad Content-Length"})
        if length > _MAX_BODY:
            self.close_connection = True
            return self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "too large"})
        # Read the body before answering, even a refusal: on Windows, answering first
        # resets the connection while the client is still sending.
        raw = self.rfile.read(length) if length else b""
        query = parse_qs(parts.query)
        if not self.owner._authorized(self, query, cookie_allowed=False):
            return self._send_json(HTTPStatus.FORBIDDEN, {"error": "forbidden"})
        page = self._page(query)
        if parts.path != "/api/close":  # A goodbye is not a sign of life.
            self.owner._contact(page)
        try:
            body = json.loads(raw.decode("utf-8")) if raw.strip() else {}
            if not isinstance(body, dict):
                raise ValueError("body must be an object")
            accepted = self.owner._act(parts.path, body, page)
        except _UnknownPath:
            return self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except (ValueError, UnicodeDecodeError) as error:
            return self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except Exception:  # noqa: BLE001 - answered, and logged once
            return self._fail(parts.path)
        self._send_json(HTTPStatus.OK, {"accepted": bool(accepted)})

    def _arriving_page(self, path: str) -> str | None:
        """The document loaded by this origin is a page arriving (a reload); its files are not.

        Another 127.0.0.1 port sends the cookie too, but its browser says it is another site.
        """
        if path not in ("/", "/index.html"):
            return None
        if self.headers.get("Sec-Fetch-Site", "same-origin") not in ("same-origin", "none"):
            return None
        return _LOADING_PAGE

    def _page(self, query: dict[str, list[str]]) -> str | None:
        """The page making this call: a header from its script, the query from its beacon."""
        return self.headers.get(_PAGE_HEADER) or query.get("page", [None])[0]

    def _fail(self, path: str) -> None:
        _log.exception("gui %s failed", path)
        self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "internal error"})

    def _send_json(self, status: HTTPStatus, payload: Any) -> None:
        # ASCII escapes: a message can hold half a surrogate pair, which UTF-8
        # cannot encode, and one such event would fail every poll after it.
        data = json.dumps(payload).encode("ascii")
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
            self.send_header("Set-Cookie", f"{self.owner._cookie_name()}={self.owner._cookie_secret}; "
                             "HttpOnly; SameSite=Strict; Path=/")
        self.end_headers()
        self.wfile.write(data)
