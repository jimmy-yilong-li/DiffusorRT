"""Bounded, transport-neutral controller for one prepared local runtime."""

from __future__ import annotations

from collections import OrderedDict, deque
from concurrent.futures import Future, InvalidStateError
from dataclasses import dataclass
import logging
import math
import threading
import time
from typing import Any
import weakref

from .result import EmptyGenerationError, require_nonempty_generation
from .cancellation import CancellationToken, GenerationCancelled, cancellation_scope
from .service_contracts import (
    CancellationOutcome,
    DEFAULT_MAX_TOKENS_LIMIT,
    RequestState,
    ServiceCancellationResult,
    ServiceError,
    ServiceGenerateRequest,
    ServiceGenerateResult,
    ServiceState,
    validate_request_id,
    validate_request_transition,
    validate_service_transition,
)

# The client-facing error is deliberately opaque ("generation_failed"), so the
# server side is the ONLY place the cause can survive. Without this logger a
# model crash, an OOM and a config bug were indistinguishable from the outside
# and left no trace anywhere. A library configures no handlers: at ERROR this
# reaches stderr through logging's last-resort handler, and an embedding app
# that calls logging.basicConfig() gets it wherever it routes its own logs.
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ServiceSnapshot:
    """JSON-native controller state for the future health endpoint."""

    state: str
    backend: str
    profile: str
    running_count: int
    queued_count: int
    queue_limit: int
    max_tokens_limit: int
    terminal_record_limit: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "backend": self.backend,
            "profile": self.profile,
            "running_count": self.running_count,
            "queued_count": self.queued_count,
            "queue_limit": self.queue_limit,
            "max_tokens_limit": self.max_tokens_limit,
            "terminal_record_limit": self.terminal_record_limit,
        }


class ServiceRuntimeError(RuntimeError):
    """Stable operational failure safe to expose through a transport adapter."""

    def __init__(
        self,
        code: str,
        detail: str,
        *,
        request_id: str | None = None,
    ) -> None:
        error = ServiceError(
            code=code,
            detail=detail,
            request_id=request_id,
        )
        self.code = error.code
        self.detail = error.detail
        self.request_id = error.request_id
        super().__init__(f"{code}: {detail}")

    def to_error(self) -> ServiceError:
        return ServiceError(
            code=self.code,
            detail=self.detail,
            request_id=self.request_id,
        )


class ServiceRequestHandle:
    """Controlled completion handle whose cancellation uses the controller."""

    __slots__ = ("_controller_ref", "_future", "_request_id")

    def __init__(
        self,
        controller: LocalServiceController,
        request_id: str,
    ) -> None:
        self._controller_ref = weakref.ref(controller)
        self._future: Future[ServiceGenerateResult] = Future()
        self._request_id = request_id

    @property
    def request_id(self) -> str:
        return self._request_id

    def cancel(self) -> bool:
        controller = self._controller_ref()
        if controller is None:
            return False
        outcome = controller.cancel(self._request_id).outcome
        return outcome is CancellationOutcome.CANCELLED

    def cancelled(self) -> bool:
        return self._future.cancelled()

    def done(self) -> bool:
        return self._future.done()

    def result(self, timeout: float | None = None) -> ServiceGenerateResult:
        return self._future.result(timeout=timeout)

    def exception(self, timeout: float | None = None) -> BaseException | None:
        return self._future.exception(timeout=timeout)

    def _cancel_from_controller(self) -> bool:
        return self._future.cancel()

    def _set_result(self, result: ServiceGenerateResult) -> None:
        self._future.set_result(result)

    def _set_exception(self, error: BaseException) -> None:
        self._future.set_exception(error)


@dataclass(slots=True)
class _RequestRecord:
    request: ServiceGenerateRequest | None
    client_handle: ServiceRequestHandle | None
    state: RequestState = RequestState.QUEUED
    cancellation: CancellationToken | None = None


class LocalServiceController:
    """Serialize generation over one prepared ``LocalDeploymentSession``."""

    def __init__(
        self,
        session: object,
        *,
        queue_limit: int = 1,
        max_tokens_limit: int = DEFAULT_MAX_TOKENS_LIMIT,
        terminal_record_limit: int = 128,
    ) -> None:
        _validate_non_negative_int(queue_limit, name="queue_limit")
        _validate_positive_int(max_tokens_limit, name="max_tokens_limit")
        _validate_non_negative_int(
            terminal_record_limit,
            name="terminal_record_limit",
        )

        self._runtime = session.runtime
        self._backend = session.backend
        self._profile = session.profile
        self._cooperative_cancel = (
            self._backend in getattr(self._runtime, "cooperative_cancellation_backends", ())
        )
        self._precision = getattr(session, "precision", None)
        self._device_ordinal = getattr(session, "device_ordinal", 0)
        self._device_memory_budget_bytes = getattr(
            session, "device_memory_budget_bytes", 0
        )
        self._queue_limit = queue_limit
        self._max_tokens_limit = max_tokens_limit
        self._terminal_record_limit = terminal_record_limit

        self._condition = threading.Condition(threading.Lock())
        self._state = validate_service_transition(
            ServiceState.STARTING,
            ServiceState.READY,
        )
        self._live: dict[str, _RequestRecord] = {}
        self._pending: deque[str] = deque()
        self._terminal: OrderedDict[str, RequestState] = OrderedDict()
        self._worker = threading.Thread(
            target=self._worker_loop,
            name=f"diffusorrt-service-{id(self):x}",
            daemon=False,
        )
        self._worker.start()

    def submit(
        self,
        request: ServiceGenerateRequest,
    ) -> ServiceRequestHandle:
        if not isinstance(request, ServiceGenerateRequest):
            raise ServiceRuntimeError(
                "request_type_invalid",
                "request must be a ServiceGenerateRequest",
            )
        if request.max_tokens > self._max_tokens_limit:
            raise ServiceRuntimeError(
                "max_tokens_limit_exceeded",
                "request max_tokens exceeds the service limit",
                request_id=request.request_id,
            )

        with self._condition:
            if self._state is not ServiceState.READY:
                raise ServiceRuntimeError(
                    "service_not_ready",
                    "service is not accepting requests",
                    request_id=request.request_id,
                )
            if (
                request.request_id in self._live
                or request.request_id in self._terminal
            ):
                raise ServiceRuntimeError(
                    "duplicate_request_id",
                    "request_id is already live or recently completed",
                    request_id=request.request_id,
                )
            if len(self._live) >= 1 + self._queue_limit:
                raise ServiceRuntimeError(
                    "queue_full",
                    "service queue is full",
                    request_id=request.request_id,
                )

            client_handle = ServiceRequestHandle(
                self,
                request.request_id,
            )
            record = _RequestRecord(
                request=request,
                client_handle=client_handle,
                cancellation=CancellationToken() if self._cooperative_cancel else None,
            )
            self._live[request.request_id] = record
            self._pending.append(request.request_id)
            self._condition.notify_all()
            return client_handle

    def cancel(self, request_id: str) -> ServiceCancellationResult:
        request_id = validate_request_id(request_id)
        client_handle: ServiceRequestHandle | None = None
        with self._condition:
            record = self._live.get(request_id)
            if record is None:
                return self._cancel_outcome_for_missing_locked(request_id)
            if record.state is RequestState.RUNNING:
                if record.cancellation is not None:
                    record.cancellation.cancel()
                    return ServiceCancellationResult(
                        request_id=request_id,
                        outcome=CancellationOutcome.CANCELLATION_REQUESTED,
                    )
                return ServiceCancellationResult(
                    request_id=request_id,
                    outcome=CancellationOutcome.RUNNING_CANCEL_UNSUPPORTED,
                )
            self._pending.remove(request_id)
            client_handle = record.client_handle
            self._terminalize_locked(request_id, RequestState.CANCELLED)

        if client_handle is not None:
            client_handle._cancel_from_controller()
        return ServiceCancellationResult(
            request_id=request_id,
            outcome=CancellationOutcome.CANCELLED,
        )

    def snapshot(self) -> ServiceSnapshot:
        with self._condition:
            running_count = sum(
                record.state is RequestState.RUNNING
                for record in self._live.values()
            )
            queued_count = sum(
                record.state is RequestState.QUEUED
                for record in self._live.values()
            )
            return ServiceSnapshot(
                state=self._state.value,
                backend=self._backend,
                profile=self._profile,
                running_count=running_count,
                queued_count=queued_count,
                queue_limit=self._queue_limit,
                max_tokens_limit=self._max_tokens_limit,
                terminal_record_limit=self._terminal_record_limit,
            )

    def shutdown(self, timeout: float | None = None) -> bool:
        deadline = _shutdown_deadline(timeout)
        handles_to_cancel: list[ServiceRequestHandle] = []
        with self._condition:
            if self._state is ServiceState.STOPPED:
                stopped = True
            else:
                stopped = False
            if not stopped and self._state is ServiceState.READY:
                self._state = validate_service_transition(
                    self._state,
                    ServiceState.DRAINING,
                )
            if not stopped:
                for record in self._live.values():
                    if record.state is RequestState.RUNNING and record.cancellation is not None:
                        record.cancellation.cancel()
                while self._pending:
                    request_id = self._pending.popleft()
                    record = self._live.get(request_id)
                    if (
                        record is None
                        or record.state is not RequestState.QUEUED
                    ):
                        continue
                    if record.client_handle is not None:
                        handles_to_cancel.append(record.client_handle)
                    self._terminalize_locked(
                        request_id,
                        RequestState.CANCELLED,
                    )

                if not self._live:
                    self._state = validate_service_transition(
                        self._state,
                        ServiceState.STOPPED,
                    )
                self._condition.notify_all()

        for handle in handles_to_cancel:
            handle._cancel_from_controller()

        with self._condition:
            while self._state is not ServiceState.STOPPED:
                remaining = _remaining(deadline)
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(timeout=remaining)

        if self._worker is threading.current_thread():
            return True
        self._worker.join(timeout=_remaining(deadline))
        joined = not self._worker.is_alive()
        if joined:
            self._release_runtime()
        return joined

    def _release_runtime(self) -> None:
        """Drop this controller's reference to the runtime, after the worker has
        joined and can no longer touch it.

        Draining the worker was only half of shutdown: the controller kept the
        runtime alive afterwards, so a foreground `serve` that returned still
        pinned every MLX buffer the model had allocated -- gigabytes, held by an
        object nobody could reach to close. Only released once the worker is
        confirmed joined; dropping it earlier would pull the object out from
        under a generation still in flight.
        """
        runtime = self._runtime
        # A failed close keeps the owner reachable: a caller can retry shutdown
        # once the worker is already stopped, and no successful status is
        # returned for memory that may still be resident.
        if runtime is not None:
            runtime.close()
            self._runtime = None

    @property
    def runtime_released(self) -> bool:
        """True once shutdown has dropped the runtime reference."""
        return self._runtime is None

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while not self._pending:
                    if self._state is ServiceState.STOPPED:
                        return
                    if (
                        self._state is ServiceState.DRAINING
                        and not self._live
                    ):
                        self._state = validate_service_transition(
                            self._state,
                            ServiceState.STOPPED,
                        )
                        self._condition.notify_all()
                        return
                    self._condition.wait()

                request_id = self._pending.popleft()
                record = self._live.get(request_id)
                if (
                    record is None
                    or record.state is not RequestState.QUEUED
                ):
                    continue
                record.state = validate_request_transition(
                    record.state,
                    RequestState.RUNNING,
                )
                request = record.request
                self._condition.notify_all()

            if request is None:
                continue

            result, error = self._run_generation(request, record.cancellation)

            with self._condition:
                record = self._live.get(request_id)
                if record is None:
                    continue
                client_handle = record.client_handle
                cancelled = record.cancellation is not None and record.cancellation.cancelled
                terminal_state = RequestState.CANCELLED if cancelled else (
                    RequestState.SUCCEEDED
                    if error is None
                    else RequestState.FAILED
                )
                self._terminalize_locked(request_id, terminal_state)
                should_stop = (
                    self._state is ServiceState.DRAINING
                    and not self._live
                    and not self._pending
                )
                if should_stop:
                    self._state = validate_service_transition(
                        self._state,
                        ServiceState.STOPPED,
                    )
                    self._condition.notify_all()

            if client_handle is not None:
                try:
                    if cancelled:
                        client_handle._cancel_from_controller()
                    elif error is None and result is not None:
                        client_handle._set_result(result)
                    else:
                        client_handle._set_exception(
                            error
                            or ServiceRuntimeError(
                                "generation_failed",
                                "generation failed",
                                request_id=request_id,
                            )
                        )
                except InvalidStateError:
                    # Controller cancellation may win after terminalization.
                    pass
            if should_stop:
                return
            del request, result, error, client_handle

    def _run_generation(
        self,
        request: ServiceGenerateRequest,
        cancellation: CancellationToken | None = None,
    ) -> tuple[ServiceGenerateResult | None, ServiceRuntimeError | None]:
        try:
            device_options = {}
            if self._backend == "cuda_native":
                device_options = {
                    "precision": self._precision,
                    "device_ordinal": self._device_ordinal,
                    "device_memory_budget_bytes": self._device_memory_budget_bytes,
                }
            with cancellation_scope(cancellation):
                generated = self._runtime.generate(
                    request.prompt,
                    backend=self._backend,
                    max_tokens=request.max_tokens,
                    return_debug=False,
                    **device_options,
                )
            require_nonempty_generation(generated.text)
            return (
                ServiceGenerateResult(
                    request_id=request.request_id,
                    text=generated.text,
                    finish_reason=generated.finish_reason,
                    backend=self._backend,
                    profile=self._profile,
                ),
                None,
            )
        except GenerationCancelled:
            return None, ServiceRuntimeError(
                "request_cancelled", "generation was cancelled", request_id=request.request_id,
            )
        except EmptyGenerationError:
            return (
                None,
                ServiceRuntimeError(
                    "empty_generation",
                    "model produced no visible answer",
                    request_id=request.request_id,
                ),
            )
        except BaseException:
            # BaseException escaping a worker thread would silently kill the
            # only worker and strand live requests. This boundary converts it
            # into the same sanitized request failure as ordinary model errors.
            #
            # Record the cause BEFORE sanitizing: the returned error carries no
            # type, message or traceback, so anything not logged here is gone.
            # Identifiers only -- the prompt is user data and never logged.
            _LOGGER.exception(
                "generation failed: request_id=%s backend=%s profile=%s",
                request.request_id,
                self._backend,
                self._profile,
            )
            return (
                None,
                ServiceRuntimeError(
                    "generation_failed",
                    "generation failed",
                    request_id=request.request_id,
                ),
            )

    def _terminalize_locked(
        self,
        request_id: str,
        state: RequestState,
    ) -> None:
        record = self._live.pop(request_id)
        validate_request_transition(record.state, state)
        record.request = None
        record.client_handle = None

        if self._terminal_record_limit > 0:
            self._terminal[request_id] = state
            self._terminal.move_to_end(request_id)
            while len(self._terminal) > self._terminal_record_limit:
                self._terminal.popitem(last=False)
        self._condition.notify_all()

    def _cancel_outcome_for_missing_locked(
        self,
        request_id: str,
    ) -> ServiceCancellationResult:
        outcome = (
            CancellationOutcome.ALREADY_TERMINAL
            if request_id in self._terminal
            else CancellationOutcome.NOT_FOUND
        )
        return ServiceCancellationResult(
            request_id=request_id,
            outcome=outcome,
        )


def _validate_non_negative_int(value: object, *, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ServiceRuntimeError(
            f"{name}_invalid",
            f"{name} must be a non-negative integer",
        )


def _validate_positive_int(value: object, *, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ServiceRuntimeError(
            f"{name}_invalid",
            f"{name} must be a positive integer",
        )


def _shutdown_deadline(timeout: float | None) -> float | None:
    if timeout is None:
        return None
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout < 0
    ):
        raise ServiceRuntimeError(
            "shutdown_timeout_invalid",
            "shutdown timeout must be a finite non-negative number",
        )
    return time.monotonic() + timeout


def _remaining(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    return max(0.0, deadline - time.monotonic())


__all__ = [
    "LocalServiceController",
    "ServiceRequestHandle",
    "ServiceRuntimeError",
    "ServiceSnapshot",
]
