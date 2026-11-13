"""The Mac windows' logic, with no GUI toolkit.

Every Engine and Conversation call runs on one worker thread, in order, so a
reply, a model load and a close can never overlap. The window calls the public
methods from its own thread and draws the events posted back to it.

author:Opus -- the Mac windows' own controller. It began as the controller of the
window these windows grew out of; nothing outside this package uses it now, and
it knows one platform.
"""
from __future__ import annotations

import concurrent.futures
import logging
import threading
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from ...cancellation import CancellationToken, GenerationCancelled
from ...conversation import ConversationConflictError, ConversationError
from ...model_opening import ModelOpenError, OpenedModel, OpenRequest
from ...result import EmptyGenerationError
from . import events as ev
from .launch import LaunchChoice, preferred_backend

_log = logging.getLogger("edllm.gui")

EMPTY_ANSWER = ("The model ended its reply without writing anything, so nothing was saved. "
                "The same message is likely to end the same way; try saying more.")
"""What a window shows when the model's first token ends its turn. An outcome, not a fault."""


class _CleanupError(RuntimeError):
    pass


class MacChatController:
    def __init__(
        self,
        *,
        open_request: Callable[[LaunchChoice], OpenRequest],
        repository: Any,
        post: Callable[[object], None],
        executor: concurrent.futures.Executor | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        context_policy: str = "recent_only",
        open_model: Callable[..., OpenedModel] | None = None,
        choose_backend: Callable[[Any, Any], tuple[str, str | None]] = preferred_backend,
        trust_remote_code: bool = False,
        decode_observer: Callable[[], Any] | None = None,
        on_busy_changed: Callable[[bool], None] | None = None,
    ) -> None:
        if open_model is None:
            from ...model_opening import open_model as open_model
        self._open_request = open_request
        self._repository = repository
        self._post = post
        self._owns_executor = executor is None
        self._executor = executor or concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="diffusorrt-chat"
        )
        self._max_tokens = max_tokens
        self._system_prompt = system_prompt
        self._context_policy = context_policy
        self._open_model = open_model
        self._choose_backend = choose_backend
        self._trust_remote_code = trust_remote_code
        # author:Opus -- called once per reply; returns that reply's live
        # observer, or None when the window is not showing the decode.
        self._decode_observer = decode_observer
        self._cancel_token: CancellationToken | None = None
        # Told when the one worker becomes busy or free. A window enables its
        # controls from this, not from guessing which event ends a job.
        self._on_busy_changed = on_busy_changed
        self._reported_busy = False
        self._lock = threading.RLock()
        self._busy = False
        self._close_requested = False
        self._closed = False
        self._opened: OpenedModel | None = None
        self._conversation: Any = None
        self._pending_conversation: Any = None
        self._cleanup_failed = False
        self._snapshot_requested = False
        self._model_event: object | None = None
        self._state_event: object | None = None
        self._trust_request: object | None = None
        self._pending_text: str | None = None
        self._trust_event = threading.Event()
        self._trust_granted = False
        from .resources import snapshot
        self._resource_reading: dict[str, Any] = snapshot(None)
        self._history_conflict = False

    # The window's calls. Each refuses rather than queueing behind a running job.

    def start(self, choice: LaunchChoice) -> bool:
        if not self._claim():
            return False
        self._emit(ev.ModelLoading(choice.spec))
        self._submit(self._load, choice)
        return True

    def switch_model(self, choice: LaunchChoice) -> bool:
        if not self._claim():
            return False
        self._emit(ev.ModelLoading(choice.spec))
        self._submit(self._switch, choice)
        return True

    def answer_trust(self, granted: bool) -> None:
        self._trust_granted = bool(granted)
        self._trust_event.set()

    def new_chat(self) -> bool:
        if not self._claim(needs_model=True):
            return False
        self._submit(self._new_chat)
        return True

    def open_chat(self, session_id: str) -> bool:
        if not self._claim(needs_model=True):
            return False
        self._submit(self._open_chat, session_id)
        return True

    def send(self, text: str) -> bool:
        if self._history_conflict or not text.strip() or not self._claim(needs_model=True):
            return False
        self._pending_text = text
        # The window shows the reply as generating from here on, so it can be
        # stopped from here on: the token exists before the worker begins.
        if getattr(self._opened.engine, "cancellation_supported", False):
            with self._lock:
                self._cancel_token = CancellationToken()
        self._emit(ev.UserMessageShown(text))
        self._emit(ev.Generating())
        self._submit(self._turn, text)
        return True

    def cancel(self) -> bool:
        """Ask the running reply to stop at its next decode boundary.

        True when a reply that can be stopped is running. The reply then ends
        with ReplyCancelled and commits nothing.
        """
        with self._lock:
            token = self._cancel_token
        if token is None:
            return False
        token.cancel()
        return True

    def request_close(self) -> bool:
        """Request checked cleanup without waiting; completion is the Closed event."""
        with self._lock:
            if self._closed:
                return True
            first = not self._close_requested or self._cleanup_failed
            self._close_requested = True
            self._cleanup_failed = False
            if self._busy:
                queue_close = False
            else:
                self._busy = True
                queue_close = True
        self._report_busy()
        if first:
            self._emit(ev.Closing())
        if queue_close:
            self._submit(self._close_now)
        return self._closed

    def request_snapshot(self) -> bool:
        """Coalesce resync requests; all Conversation reads stay on the worker."""
        with self._lock:
            if self._snapshot_requested:
                return True
            if self._closed:
                self._emit(ev.Snapshot((ev.Closed(),)))
                return True
            self._snapshot_requested = True
            self._executor.submit(self._snapshot)
        return True

    def settings(self) -> dict[str, Any]:
        with self._lock:
            return {"max_tokens": self._max_tokens, "system": self._system_prompt, "context_policy": self._context_policy}

    def reload_chat(self) -> bool:
        if not self._claim(needs_model=True):
            return False
        self._submit(self._reload_chat)
        return True

    def _reload_chat(self) -> None:
        if self._conversation is not None:
            self._conversation.reload()
            self._emit(ev.ChatShown(self._conversation.session_id, tuple(self._conversation.messages())))
        self._history_conflict = False

    def configure(self, *, max_tokens: int | None, system_prompt: str | None) -> bool:
        # author:Opus -- the Mac windows send no limit: a reply runs until the model ends it.
        if max_tokens is not None and (type(max_tokens) is not int or max_tokens < 1):
            raise ValueError("Output limit must be a positive integer, or none.")
        if system_prompt is not None and not isinstance(system_prompt, str):
            raise ValueError("System prompt must be text.")
        if not self._claim():
            return False
        self._submit(self._configure, max_tokens, system_prompt)
        return True

    def _configure(self, max_tokens: int | None, system_prompt: str | None) -> None:
        if system_prompt != self._system_prompt:
            self._new_chat()
        with self._lock:
            self._max_tokens, self._system_prompt = max_tokens, system_prompt
        self._emit(ev.SettingsShown(max_tokens, system_prompt, self._context_policy))
        if self._opened is not None:
            self._emit(ev.RecentChanged(self._recent()))

    def resources(self) -> dict[str, Any]:
        # HTTP reads only the last completed owner snapshot; no device polling
        # or second inference worker while a request is active.
        with self._lock:
            return dict(self._resource_reading)

    def _sample_resources(self) -> None:
        from .resources import snapshot

        engine = self._opened.engine if self._opened is not None else None
        try:
            reading = snapshot(engine)
        except Exception:
            _log.debug("resource snapshot unavailable", exc_info=True)
            return
        with self._lock:
            self._resource_reading = reading

    # Worker jobs.

    def _load(self, choice: LaunchChoice) -> None:
        try:
            if (Path(choice.spec).is_absolute() or choice.spec.startswith(("./", "../", "~"))) and not Path(choice.spec).expanduser().is_dir():
                raise ValueError("Package folder does not exist. Choose a prepared package folder or refresh the model list.")
            choice = self._decide_backend(choice)
            opened = self._open_model(
                self._repository, self._open_request(choice), consent=self._ask_trust
            )
        except ModelOpenError as error:
            self._emit(ev.ModelFailed(str(error), error.next_action))
            self._emit(ev.RecentChanged(()))
            return
        except Exception as error:  # shown to the user, not swallowed
            _log.exception("could not open %s", choice.spec)
            self._emit(ev.ModelFailed(f"could not open {choice.spec}: {error}", None))
            self._emit(ev.RecentChanged(()))
            return
        self._opened = opened
        self._history_conflict = False
        capabilities = opened.engine.capabilities()
        actual_backend = capabilities.get("selected_backend") or opened.backend
        actual_precision = opened.precision or capabilities.get("compute_dtype")
        _log.info("opened %s with %s %s", choice.spec, opened.backend, opened.precision or "")
        self._emit(ev.ModelReady(
            spec=choice.spec,
            model_name=_model_name(opened),
            backend=actual_backend,
            precision=actual_precision,
            gpu_name=None,
            precisions=(),
            cancellable=bool(getattr(opened.engine, "cancellation_supported", False)),
            package_dir=str(opened.package_dir),
        ))
        self._emit(ev.RecentChanged(self._recent()))
        self._emit(ev.ChatShown(None, ()))

    def _switch(self, choice: LaunchChoice) -> None:
        self._release()
        self._load(choice)

    def _new_chat(self) -> None:
        self._close_conversation()
        self._history_conflict = False
        self._emit(ev.ChatShown(None, ()))

    def _open_chat(self, session_id: str) -> None:
        # Open first: if it fails, the chat on screen stays the one that receives messages.
        try:
            conversation = self._opened.engine.conversation(
                session_id=session_id,
                system_prompt=self._system_prompt,
                context_policy=self._context_policy,
            )
        except ConversationError as error:
            self._emit(ev.ReplyFailed(str(error), self._history_conflict))
            return
        self._pending_conversation = conversation
        try:
            messages = tuple(conversation.messages())
        except Exception:
            try:
                conversation.close()
            except Exception as error:
                raise _CleanupError(f"candidate conversation close failed: {error}") from error
            self._pending_conversation = None
            raise
        self._close_conversation()
        self._conversation, self._pending_conversation = conversation, None
        self._history_conflict = False
        self._emit(ev.ChatShown(session_id, messages))

    def _turn(self, text: str) -> None:
        try:
            if self._conversation is None:
                # Created on the first message, so an untouched new chat is never stored.
                self._conversation = self._opened.engine.conversation(
                    system_prompt=self._system_prompt, context_policy=self._context_policy
                )
            reply = self._conversation.send(
                text, max_tokens=self._max_tokens, **self._reply_options()
            )
        except GenerationCancelled:
            self._emit(ev.ReplyCancelled())
            return
        except ConversationConflictError as error:
            self._history_conflict = True
            self._emit(ev.ReplyFailed(str(error), reload_required=True))
            return
        except ConversationError as error:
            self._emit(ev.ReplyFailed(str(error)))
            return
        except EmptyGenerationError:
            # author:Opus -- the model's first token ended its turn. The engine stores no
            # empty reply, so the message is still the user's; say that, not "failed".
            _log.info("the model produced no visible answer; nothing was stored")
            self._emit(ev.ReplyFailed(EMPTY_ANSWER))
            return
        except Exception as error:  # shown to the user, not swallowed
            _log.exception("the reply failed")
            self._emit(ev.ReplyFailed(f"the reply failed: {error}"))
            return
        self._emit(ev.ReplyShown(
            reply.text, reply.elapsed_seconds, getattr(reply, "visible_output_tokens", None),
            asdict(reply.context) if getattr(reply, "context", None) is not None else None,
            getattr(self._conversation, "session_id", None),
            getattr(reply, "finish_reason", None),
        ))
        self._emit(ev.RecentChanged(self._recent()))

    def _reply_options(self) -> dict[str, Any]:
        """Cancellation and live observation for one reply, when available."""
        options: dict[str, Any] = {}
        with self._lock:
            token = self._cancel_token  # made when Send was accepted
        if token is not None:
            options["cancellation_token"] = token
        if self._decode_observer is not None:
            observer = self._decode_observer()
            if observer is not None:
                options["observer"] = observer
        return options

    def _close_now(self) -> None:
        if self._closed:
            return
        self._release()
        self._closed = True
        self._emit(ev.Closed())
        if self._owns_executor:
            self._executor.shutdown(wait=False)

    # Helpers.

    def _claim(self, *, needs_model: bool = False) -> bool:
        with self._lock:
            if self._busy or self._closed or self._close_requested:
                return False
            if needs_model and self._opened is None:
                return False
            self._busy = True
        self._report_busy()
        return True

    def _report_busy(self) -> None:
        if self._on_busy_changed is None:
            return
        with self._lock:
            busy = self._busy
            if busy == self._reported_busy:
                return
            self._reported_busy = busy
        try:
            self._on_busy_changed(busy)
        except Exception:  # a window's bug must not stop the worker
            _log.exception("busy notification failed")

    def _submit(self, job: Callable[..., None], *args: Any) -> concurrent.futures.Future:
        return self._executor.submit(self._run, job, *args)

    def _run(self, job: Callable[..., None], *args: Any) -> None:
        try:
            job(*args)
        except _CleanupError as error:
            self._close_failed(error)
        except Exception as error:  # a dead worker would freeze the window
            _log.exception("chat job failed")
            self._emit(ev.ReplyFailed(f"unexpected error: {error}", self._history_conflict))
        finally:
            if job == self._turn:
                self._pending_text = None
                with self._lock:
                    self._cancel_token = None
            if not self._close_requested:
                self._sample_resources()
            with self._lock:
                close = self._close_requested and not self._closed and not self._cleanup_failed
                if not close:
                    self._busy = False
            if close:
                try:
                    self._close_now()
                except _CleanupError as error:
                    self._close_failed(error)
                finally:
                    with self._lock:
                        self._busy = False
            self._report_busy()

    def _close_failed(self, error: Exception) -> None:
        self._cleanup_failed = True
        self._close_requested = True
        _log.error("cleanup failed; owners retained for retry: %s", error)
        self._emit(ev.CloseFailed(str(error)))

    def _snapshot(self) -> None:
        with self._lock:
            if not self._snapshot_requested:
                return
            self._snapshot_requested = False
        events = [self._model_event] if self._model_event is not None else []
        events.append(ev.SettingsShown(**self.settings()))
        conversation = self._conversation
        try:
            events.append(ev.ChatShown(
                conversation.session_id if conversation is not None else None,
                tuple(conversation.messages()) if conversation is not None else (),
            ))
        except Exception:
            # A failed close may have invalidated the store while retaining its owner.
            events.append(ev.ChatShown(None, ()))
        if self._opened is not None and not self._close_requested:
            events.append(ev.RecentChanged(self._recent()))
        if self._trust_request is not None:
            events.append(self._trust_request)
        if self._pending_text is not None:
            events.extend((ev.UserMessageShown(self._pending_text), ev.Generating()))
        if self._state_event is not None:
            events.append(self._state_event)
        if self._history_conflict:
            events.append(ev.ReplyFailed("History changed elsewhere. Reload it before continuing.", True))
        self._emit(ev.Snapshot(tuple(events)))

    def _emit(self, event: object) -> None:
        if isinstance(event, (ev.ModelLoading, ev.ModelReady, ev.ModelFailed)):
            self._model_event = event
        if isinstance(event, (ev.Closing, ev.CloseFailed, ev.Closed)):
            self._state_event = event
        self._post(event)

    def _decide_backend(self, choice: LaunchChoice) -> LaunchChoice:
        if choice.backend is not None:
            return choice
        plan = self._repository.plan(choice.spec, allow_local_dir=True, offline=True)
        backend, precision = self._choose_backend(plan.package_dir, self._repository)
        return replace(choice, backend=backend, precision=precision)

    def _ask_trust(self, spec: str, module: str) -> bool:
        if self._trust_remote_code:
            return True  # Granted on the command line, as `chat --trust-remote-code` does.
        self._trust_event.clear()
        self._trust_granted = False
        self._trust_request = ev.TrustRequested(spec, module)
        self._emit(self._trust_request)
        while not self._trust_event.wait(0.1):
            if self._snapshot_requested:
                self._snapshot()
            if self._close_requested:
                self._trust_request = None
                return False
        self._trust_request = None
        return self._trust_granted

    def _recent(self) -> tuple:
        try:
            return tuple(self._opened.engine.recent_conversations(
                system_prompt=self._system_prompt, context_policy=self._context_policy
            ))
        except Exception:  # a missing list must not stop the chat
            _log.exception("listing recent conversations failed")
            return ()

    def _close_conversation(self) -> None:
        conversation = self._conversation
        if conversation is not None:
            try:
                conversation.close()
            except Exception as error:
                raise _CleanupError(f"conversation close failed: {error}") from error
            self._conversation = None

    def _release(self) -> None:
        errors = []
        for attribute in ("_conversation", "_pending_conversation", "_opened"):
            owner = getattr(self, attribute)
            if owner is None:
                continue
            try:
                (owner.engine if attribute == "_opened" else owner).close()
            except Exception as error:
                errors.append(f"{attribute.lstrip('_')} close failed: {error}")
            else:
                setattr(self, attribute, None)
        if errors:
            raise _CleanupError("; ".join(errors))


def _model_name(opened: OpenedModel) -> str:
    try:
        reference = opened.engine.capabilities().get("source_reference")
    except Exception:  # the folder name is a fine label
        reference = None
    if isinstance(reference, str) and reference.strip():
        return reference
    name = opened.package_dir.name
    return name.split("__", 1)[-1].split("@", 1)[0] or name
