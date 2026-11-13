"""The chat window's logic, with no GUI toolkit.

Every Engine and Conversation call runs on one worker thread, in order, so a
reply, a model load and a close can never overlap. The window calls the public
methods from its own thread and draws the events posted back to it.
"""
from __future__ import annotations

import concurrent.futures
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..backend_ids import AUTO_BACKEND, CUDA_NATIVE_BACKEND
from ..cancellation import CancellationToken, GenerationCancelled
from ..conversation import ConversationConflictError, ConversationError
from ..model_opening import ModelOpenError, OpenedModel, OpenRequest
from . import events as ev
from .gpu_memory import read_gpu_memory
from .launch import LaunchChoice, admitted_cuda_precision, default_backend

_log = logging.getLogger("edllm.gui")


class ModelReleaseError(RuntimeError):
    """The model could not be released; the controller still holds it, and a retry may."""


class ChatController:
    def __init__(
        self,
        *,
        open_request: Callable[[LaunchChoice], OpenRequest],
        repository: Any,
        post: Callable[[object], None],
        executor: concurrent.futures.Executor | None = None,
        max_tokens: int = 128,
        system_prompt: str | None = None,
        context_policy: str = "recent_only",
        open_model: Callable[..., OpenedModel] | None = None,
        gpu_reader: Callable[[], Any] = read_gpu_memory,
        choose_backend: Callable[[Any, Any], tuple[str, str | None]] = default_backend,
        trust_remote_code: bool = False,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> None:
        if open_model is None:
            from ..model_opening import open_model as open_model
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
        self._gpu_reader = gpu_reader
        self._choose_backend = choose_backend
        self._trust_remote_code = trust_remote_code
        self._temperature = float(temperature)
        self._seed = seed
        """None draws a seed for each reply; a number repeats a sampled reply."""
        self._can_sample = False
        self._lock = threading.Lock()
        self._busy = False
        self._close_requested = False
        self._closed = False
        self._opened: OpenedModel | None = None
        self._loaded: LaunchChoice | None = None
        """The choice the open model came from, to go back to when a switch fails."""
        self._unreleased: OpenedModel | None = None
        """A model whose close failed. Engine.close keeps its handle for a retry, so
        it is held here, not dropped: nothing new opens until a retry succeeds."""
        self._release_error: str | None = None
        self._conversation: Any = None
        self._chat_locked = False
        """The open chat changed elsewhere and could not be read again: no send until
        another chat is opened or a new one started."""
        self._trust_event = threading.Event()
        self._trust_granted = False
        self._trusted: set[tuple[str, str]] = set()
        """(spec, module) granted in this run: going back to a model does not ask again."""
        self._can_stop = False
        self._token: CancellationToken | None = None
        self._ready: ev.ModelReady | None = None
        self._settled: list[object] = []
        """A job's last events, which tell the page it may act again: posted by _run
        once the worker is free, so the action they enable is accepted."""

    # The window's calls. Each refuses rather than queueing behind a running job.

    def start(self, choice: LaunchChoice) -> bool:
        if not self._claim():
            return False
        self._post(ev.ModelLoading(choice.spec))
        self._submit(self._start, choice)
        return True

    def switch_model(self, choice: LaunchChoice) -> bool:
        if not self._claim():
            return False
        self._post(ev.ModelLoading(choice.spec))
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

    def delete_chat(self, session_id: str, revision: int) -> bool:
        """Delete a stored conversation as the list showed it, at `revision`.

        Refused while a reply or a load runs.
        """
        if not self._claim(needs_model=True):
            return False
        self._submit(self._delete_chat, session_id, revision)
        return True

    def send(self, text: str) -> bool:
        if not text.strip() or self._chat_locked or not self._claim(needs_model=True):
            return False
        # The token exists before Generating reaches the page, which enables Stop:
        # a Stop pressed before the turn starts must still find it.
        token = CancellationToken() if self._can_stop else None
        with self._lock:
            self._token = token
        self._post(ev.UserMessageShown(text))
        self._post(ev.Generating(started=time.time()))
        self._submit(self._turn, text, token)
        return True

    def set_temperature(self, temperature: float) -> None:
        """The temperature the next reply is sampled at; a running reply keeps its own."""
        with self._lock:
            self._temperature = float(temperature)

    def stop(self) -> bool:
        """Stop the running reply at its next decode step; False when none runs."""
        with self._lock:
            token = self._token
        if token is None:
            return False
        token.cancel()
        return True

    def request_close(self) -> bool:
        """True when closed now; False when a running job must finish first.

        Raises ModelReleaseError when the model could not be released; it stays
        held, and calling again retries the release.
        """
        with self._lock:
            if self._closed:
                return True
            if self._busy:
                if not self._close_requested:
                    self._close_requested = True
                    self._post(ev.Closing())
                return False
            self._busy = True
        if not self._submit(self._close_now).result():
            raise ModelReleaseError(self._release_error or "the model could not be released")
        return True

    # Worker jobs.

    def _start(self, choice: LaunchChoice) -> None:
        failure = self._load(choice)
        if failure is not None:
            self._settle_load_failure(failure)

    def _load(self, choice: LaunchChoice) -> ev.ModelFailed | None:
        """Open `choice`; return its failure for the caller to settle or recover."""
        try:
            choice = self._decide_backend(choice)
            opened = self._open_model(
                self._repository, self._open_request(choice), consent=self._ask_trust
            )
        except ModelOpenError as error:
            return ev.ModelFailed(str(error), error.next_action)
        except Exception as error:  # noqa: BLE001 - shown to the user, not swallowed
            _log.exception("could not open %s", choice.spec)
            reason = f"could not open {choice.spec}: {error}"
            return ev.ModelFailed(reason, None)
        self._opened, self._loaded = opened, choice
        self._can_stop = _can_stop(opened.engine)
        self._can_sample = _can_sample(opened.engine)
        capabilities = _capabilities(opened)
        backend = _running_backend(opened, capabilities)
        _log.info("opened %s with %s %s", choice.spec, backend, opened.precision or "")
        failure = self._warm_up(opened)
        if failure is not None:
            # Release before reporting failure; a failed release remains held.
            # The caller publishes the final failure only once its job is free.
            reason = f"could not start {choice.spec}: {failure}"
            if self._release():
                return ev.ModelFailed(reason, "Try again, or choose another model.",
                                      stage="start")
            return ev.ModelFailed(
                f"{reason}; then it could not be released: {self._release_error}",
                "Try again, or close DiffusorRT to free the GPU.", stage="release")
        gpu = self._gpu_reader()
        self._post(ev.RecentChanged(self._recent()))
        self._settle(ev.ChatShown(None, ()))
        # Posted by _run once this job no longer holds the worker: the page
        # takes ModelReady as leave to send, and a send while busy is refused.
        self._ready = ev.ModelReady(
            spec=_lasting_spec(choice.spec),
            model_name=_model_name(opened, capabilities),
            backend=backend,
            precision=opened.precision,
            gpu_name=None if gpu is None else gpu.name,
            precisions=_admitted_precisions(opened),
            can_stop=self._can_stop,
            chosen_backend=None if opened.backend == AUTO_BACKEND else opened.backend,
            can_sample=self._can_sample,
        )
        return None

    def _warm_up(self, opened: OpenedModel) -> str | None:
        """Pay the first generation's one-time cost while the window says Loading.

        cuda_native puts the weights on the GPU and starts its kernels at the
        first generation, which otherwise lands on the user's first question.
        Nothing is stored. Returns why the runtime could not start (a failed
        allocation, a native build that does not fit), or None.
        """
        from ..result import EmptyGenerationError

        started = time.monotonic()
        try:
            opened.engine.generate("Hi", max_tokens=1)
        except EmptyGenerationError:
            pass  # One token can be an end of turn; the start-up work is done.
        except Exception as error:  # noqa: BLE001 - shown as the load's failure
            _log.exception("the model could not start")
            return str(error) or type(error).__name__
        _log.info("warmed up in %.1f s", time.monotonic() - started)
        return None

    def _switch(self, choice: LaunchChoice) -> None:
        # Released first: an 8 GB GPU may not hold both. A failure goes back.
        previous = self._loaded
        if not self._release():
            # Still held: opening another now could need memory it has not given
            # back. Trying the switch again retries the release first.
            self._post_release_failure()
            return
        failure = self._load(choice)
        if failure is None:
            return
        if previous is None or self._close_requested or self._unreleased is not None:
            self._settle_load_failure(failure)
            return  # A held failed model or a closing window must not open another.
        self._post(ev.ModelLoading(previous.spec))
        fallback_failure = self._load(previous)
        if fallback_failure is None:
            self._settle(ev.ReplyFailed(_back_again(choice, previous), details=failure.reason))
        else:
            # One final card keeps both causes; it must not enable Retry mid-recovery.
            self._settle_load_failure(ev.ModelFailed(
                f'Couldn\'t open "{choice.spec}": {failure.reason}. '
                f'The previous model could not reopen: {fallback_failure.reason}',
                fallback_failure.next_action, stage=fallback_failure.stage,
            ))

    def _new_chat(self) -> None:
        self._close_conversation()
        self._settle(ev.ChatShown(None, ()))

    def _open_chat(self, session_id: str) -> None:
        # Open first: if it fails, the chat on screen stays the one that receives messages.
        try:
            conversation = self._opened.engine.conversation(
                session_id=session_id,
                system_prompt=self._system_prompt,
                context_policy=self._context_policy,
            )
        except ConversationError as error:
            self._settle(ev.ReplyFailed(str(error)))
            return
        self._close_conversation()
        self._conversation = conversation
        self._settle(ev.ChatShown(session_id, tuple(conversation.messages())))

    def _delete_chat(self, session_id: str, revision: int) -> None:
        # Every outcome settles: the page acts on these events (Send on ChatShown),
        # and the worker is free only once this job has returned.
        current = self._conversation is not None and self._conversation.session_id == session_id
        try:
            if current:
                # The open handle checks the revision and generation this window shows.
                self._conversation.delete()
            else:
                # By id, at the revision the list showed. Opening it first would
                # check nothing, and would create a conversation that is gone.
                self._repository.delete_conversation(session_id, expected_revision=revision)
        except KeyError:
            _log.info("conversation %s was already deleted", session_id)
        except ConversationConflictError:
            recent = self._recent()
            # Shown in whichever chat is open, so name another one by its title.
            title = next((item.title for item in recent if item.session_id == session_id), None)
            subject = "This conversation" if current else (
                f'"{title}"' if title else "That conversation")
            self._settle(ev.ReplyFailed(
                f"{subject} was changed somewhere else, so it was not deleted. "
                "Open it to see the change, then delete it again."
            ))
            self._settle(ev.RecentChanged(recent))
            return
        except ConversationError as error:
            self._settle(ev.ReplyFailed(f"Couldn't delete the conversation: {error}"))
            return
        except Exception as error:  # noqa: BLE001 - shown to the user, not swallowed
            _log.exception("deleting a conversation failed")
            self._settle(ev.ReplyFailed(f"Couldn't delete the conversation: {error}"))
            return
        else:
            _log.info("deleted conversation %s", session_id)
        self._settle(ev.ChatDeleted(session_id))
        if current:
            self._close_conversation()  # Already closed by a delete; not by a KeyError.
            self._settle(ev.ChatShown(None, ()))
        self._settle(ev.RecentChanged(self._recent()))

    def _turn(self, text: str, token: CancellationToken | None) -> None:
        try:
            if self._conversation is None:
                # Created on the first message, so an untouched new chat is never stored.
                self._conversation = self._opened.engine.conversation(
                    system_prompt=self._system_prompt, context_policy=self._context_policy
                )
            # Only a model that samples is handed a temperature; the others refuse one.
            with self._lock:
                sampling = ({"temperature": self._temperature, "seed": self._seed}
                            if self._can_sample else {})
            reply = self._conversation.send(
                text, max_tokens=self._max_tokens, on_progress=self._forming,
                cancellation_token=token, **sampling,
            )
        except GenerationCancelled:
            # Nothing was committed; the user's text goes back into the box.
            self._settle(ev.ReplyStopped(text))
            return
        except ConversationConflictError as error:
            self._show_after_conflict(text, error)
            return
        except ConversationError as error:
            self._settle(ev.ReplyFailed(str(error)))
            return
        except Exception as error:  # noqa: BLE001 - shown to the user, not swallowed
            _log.exception("the reply failed")
            self._settle(ev.ReplyFailed(f"the reply failed: {error}"))
            return
        finally:
            with self._lock:
                self._token = None
        self._settle(ev.ReplyShown(
            reply.text, reply.elapsed_seconds, getattr(reply, "visible_output_tokens", None),
            session_id=self._conversation.session_id,
        ))
        self._settle(ev.RecentChanged(self._recent()))

    def _show_after_conflict(self, text: str, error: ConversationConflictError) -> None:
        """Another writer moved the session on. Show the history as it is stored,
        so the next send continues the conversation on the screen."""
        conversation = self._conversation
        try:
            conversation.reload()
            messages = tuple(conversation.messages())
        except Exception:  # noqa: BLE001 - shown as a locked chat, with the way out
            _log.exception("reload after a conflict failed")
            self._chat_locked = True
            self._settle(ev.ReplyFailed(str(error), text=text))
            self._settle(ev.ChatLocked(
                "This conversation was changed somewhere else and could not be read "
                "again, so it cannot continue here. Open it again from Recent, or "
                "start a new chat."
            ))
            return
        self._settle(ev.ChatShown(conversation.session_id, messages))
        self._settle(ev.ReplyFailed(
            "This conversation was changed somewhere else, so your message was not "
            "sent. It now shows that change.",
            details=str(error),
            text=text,
        ))
        self._settle(ev.RecentChanged(self._recent()))

    def _forming(self, progress) -> None:
        self._post(ev.ReplyForming(
            step=progress.step, steps=progress.steps, segments=tuple(progress.segments),
            decided=progress.decided, positions=progress.positions,
            canvas_decided=progress.canvas_decided, canvas_positions=progress.canvas_positions,
        ))

    def _close_now(self) -> bool:
        """Close if the model can be released; False, still open, when it cannot."""
        if self._closed:
            return True
        if not self._release():
            with self._lock:
                self._close_requested = False  # the window's next request retries
            self._post_release_failure()
            return False
        self._closed = True
        self._post(ev.Closed())
        if self._owns_executor:
            self._executor.shutdown(wait=False)
        return True

    # Helpers.

    def _settle(self, event: object) -> None:
        """Post ``event`` when this job has ended and the worker is free."""
        self._settled.append(event)

    def _settle_load_failure(self, failure: ev.ModelFailed) -> None:
        # Retry becomes actionable last, after the failed load's list update.
        self._settle(ev.RecentChanged(()))
        self._settle(failure)

    def _claim(self, *, needs_model: bool = False) -> bool:
        with self._lock:
            if self._busy or self._closed or self._close_requested:
                return False
            if needs_model and self._opened is None:
                return False
            self._busy = True
            return True

    def _submit(self, job: Callable[..., Any], *args: Any) -> concurrent.futures.Future:
        return self._executor.submit(self._run, job, *args)

    def _run(self, job: Callable[..., Any], *args: Any) -> Any:
        try:
            return job(*args)
        except Exception as error:  # noqa: BLE001 - a dead worker would freeze the window
            _log.exception("chat job failed")
            self._settle(ev.ReplyFailed(f"unexpected error: {error}"))
        finally:
            with self._lock:
                self._busy = False
                close = self._close_requested and not self._closed
                ready, self._ready = self._ready, None
                settled, self._settled = self._settled, []
            for event in settled:
                self._post(event)
            if ready is not None and not close:
                self._post(ready)
            if close:
                self._close_now()

    def _decide_backend(self, choice: LaunchChoice) -> LaunchChoice:
        cuda_without_precision = choice.backend == CUDA_NATIVE_BACKEND and choice.precision is None
        if choice.backend is not None and not cuda_without_precision:
            return choice
        from ..compiler import CompilerError
        from ..local_models import ModelSpecError

        try:
            plan = self._repository.plan(choice.spec, allow_local_dir=True, offline=True)
        except (CompilerError, ModelSpecError, OSError, ValueError):
            # Not a model this machine knows: opening it says so in its own words.
            return choice if cuda_without_precision else replace(choice, backend=AUTO_BACKEND)
        if cuda_without_precision:
            # cuda_native was asked for: open it at what the package admits, and
            # let the load say why when it admits nothing.
            return replace(choice, precision=admitted_cuda_precision(plan.package_dir))
        backend, precision = self._choose_backend(plan.package_dir, self._repository)
        if backend == CUDA_NATIVE_BACKEND and choice.precision is not None:
            precision = choice.precision  # Asked for on the command line.
        return replace(choice, backend=backend, precision=precision)

    def _ask_trust(self, spec: str, module: str) -> bool:
        if self._trust_remote_code:
            return True  # Granted on the command line, as `chat --trust-remote-code` does.
        if (spec, module) in self._trusted:
            return True
        self._trust_event.clear()
        self._trust_granted = False
        self._post(ev.TrustRequested(spec, module))
        while not self._trust_event.wait(0.1):
            if self._close_requested:
                self._post(ev.TrustAnswered(False))
                return False
        self._post(ev.TrustAnswered(self._trust_granted))
        if self._trust_granted:
            self._trusted.add((spec, module))
        return self._trust_granted

    def _recent(self) -> tuple:
        try:
            return tuple(self._opened.engine.recent_conversations(
                system_prompt=self._system_prompt, context_policy=self._context_policy
            ))
        except Exception:  # noqa: BLE001 - a missing list must not stop the chat
            _log.exception("listing recent conversations failed")
            return ()

    def _close_conversation(self) -> None:
        # Whatever chat comes next is read afresh, so a lock on this one ends here.
        self._chat_locked = False
        conversation, self._conversation = self._conversation, None
        if conversation is not None:
            try:
                conversation.close()
            except Exception:  # noqa: BLE001 - closing the engine still matters more
                _log.exception("closing the conversation failed")

    def _release(self) -> bool:
        """Close the open model; False when a cleanup failed and a model is still held.

        A model whose close failed stays in `_unreleased`, and the next release
        retries that same Engine first (Engine.close keeps its handle for exactly
        that), so no other model ever takes its place there. The open model then
        leaves `_opened`, so nothing generates with it, and is closed the same way.
        """
        self._close_conversation()
        while True:
            # A held model is retried first, so another never takes its place.
            if self._unreleased is not None:
                try:
                    self._unreleased.engine.close()
                except Exception as error:  # noqa: BLE001 - shown, held and retried, not swallowed
                    _log.exception("releasing the model failed")
                    self._release_error = str(error) or type(error).__name__
                    return False
                self._unreleased, self._release_error = None, None
            if self._opened is None:
                return True
            self._unreleased, self._opened, self._loaded = self._opened, None, None

    def _post_release_failure(self) -> None:
        failure = ev.ModelFailed(
            f"the model in use could not be released: {self._release_error}",
            "Try again, or close DiffusorRT to free the GPU.",
            stage="release",
        )
        # Deferred close runs after _run has drained settled events; a close or
        # switch submitted as a job still has to release the busy flag first.
        if self._busy:
            self._settle(failure)
        else:
            self._post(failure)


def _lasting_spec(spec: str) -> str:
    """A folder by its full path: the window remembers it, and the next launch
    may start in another working directory. A model id stays as it is."""
    folder = Path(spec)
    return str(folder.resolve()) if folder.is_dir() else spec


def _back_again(choice: LaunchChoice, previous: LaunchChoice) -> str:
    if choice.spec == previous.spec:
        return (f"Couldn't switch to {choice.precision}, so the model is open again "
                f"at {previous.precision}.")
    return f'Couldn\'t open "{choice.spec}", so the model you were using is open again.'


def _admitted_precisions(opened: OpenedModel) -> tuple[str, ...]:
    """What this package may run at on cuda_native: its admitted profile decides."""
    from ..backends import cuda_profile

    if opened.backend != CUDA_NATIVE_BACKEND:
        return ()
    try:
        return tuple(cuda_profile.admit_cuda_execution_profile(
            opened.package_dir, precision=None
        ).precisions)
    except (OSError, ValueError):
        return ()


def _can_stop(engine: Any) -> bool:
    """Whether the loaded plan admits cooperative cancellation, as Engine checks it."""
    try:
        return engine._backend in engine._handle.cooperative_cancellation_backends
    except Exception:  # noqa: BLE001 - an engine that cannot say cannot stop
        return False


def _can_sample(engine: Any) -> bool:
    """Whether the loaded plan samples at a temperature, as RuntimeHandle.generate admits it."""
    try:
        strategy = engine._handle._resolved_model_plan.runtime_config.decode_strategy
        return engine._backend == CUDA_NATIVE_BACKEND and strategy in {
            "fixed_quota", "bd3_fixed_quota"}
    except Exception:  # noqa: BLE001 - a plan that cannot say decodes greedily
        return False


def _capabilities(opened: OpenedModel) -> dict[str, Any]:
    try:
        capabilities = opened.engine.capabilities()
    except Exception:  # noqa: BLE001 - labels fall back to what the window already knows
        _log.warning("reading the model's capabilities failed", exc_info=True)
        return {}
    return capabilities if isinstance(capabilities, dict) else {}


def _running_backend(opened: OpenedModel, capabilities: dict[str, Any]) -> str:
    """The backend replies run on: "auto" resolved as generation resolves it."""
    if opened.backend != AUTO_BACKEND:
        return opened.backend
    resolved = capabilities.get("auto_backend")
    return resolved if isinstance(resolved, str) and resolved else opened.backend


def _model_name(opened: OpenedModel, capabilities: dict[str, Any]) -> str:
    reference = capabilities.get("source_reference")
    if isinstance(reference, str) and reference.strip():
        return reference
    name = opened.package_dir.name
    return name.split("__", 1)[-1].split("@", 1)[0] or name
