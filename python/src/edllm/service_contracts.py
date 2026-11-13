"""Standard-library-only contracts for the bounded local service."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
from enum import Enum
import re
from typing import Any
import uuid


DEFAULT_MAX_TOKENS_LIMIT = 512
DEFAULT_PROMPT_CHARS_LIMIT = 1_048_576
MAX_REQUEST_ID_CHARS = 128

_REQUEST_FIELDS = frozenset({"prompt", "max_tokens", "request_id"})
_REQUIRED_REQUEST_FIELDS = frozenset({"prompt", "max_tokens"})
_REQUEST_ID_PATTERN = re.compile(
    rf"[A-Za-z0-9][A-Za-z0-9._:-]{{0,{MAX_REQUEST_ID_CHARS - 1}}}"
)


class RequestState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ServiceState(str, Enum):
    STARTING = "starting"
    READY = "ready"
    DRAINING = "draining"
    STOPPED = "stopped"


class CancellationOutcome(str, Enum):
    CANCELLED = "cancelled"
    CANCELLATION_REQUESTED = "cancellation_requested"
    RUNNING_CANCEL_UNSUPPORTED = "running_cancel_unsupported"
    NOT_FOUND = "not_found"
    ALREADY_TERMINAL = "already_terminal"


class ServiceValidationError(ValueError):
    """Stable validation failure for service and transport adapters."""

    def __init__(
        self,
        code: str,
        detail: str,
        *,
        request_id: str | None = None,
    ) -> None:
        self.code = code
        self.detail = detail
        self.request_id = _request_id_if_string(request_id)
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class ServiceGenerateRequest:
    prompt: str
    max_tokens: int
    request_id: str = field(default_factory=lambda: f"req_{uuid.uuid4().hex}")
    max_tokens_limit: InitVar[int] = DEFAULT_MAX_TOKENS_LIMIT
    prompt_chars_limit: InitVar[int] = DEFAULT_PROMPT_CHARS_LIMIT

    def __post_init__(
        self,
        max_tokens_limit: int,
        prompt_chars_limit: int,
    ) -> None:
        _validate_positive_limit(
            max_tokens_limit,
            name="max_tokens_limit",
            code="max_tokens_limit_invalid",
        )
        _validate_positive_limit(
            prompt_chars_limit,
            name="prompt_chars_limit",
            code="prompt_chars_limit_invalid",
        )
        if not isinstance(self.prompt, str):
            raise ServiceValidationError(
                "prompt_type_invalid",
                "prompt must be a string",
                request_id=_request_id_if_string(self.request_id),
            )
        if not self.prompt.strip():
            raise ServiceValidationError(
                "prompt_empty",
                "prompt must contain non-whitespace text",
                request_id=_request_id_if_string(self.request_id),
            )
        if len(self.prompt) > prompt_chars_limit:
            raise ServiceValidationError(
                "prompt_too_large",
                f"prompt exceeds the {prompt_chars_limit}-character limit",
                request_id=_request_id_if_string(self.request_id),
            )
        if not _is_strict_int(self.max_tokens):
            raise ServiceValidationError(
                "max_tokens_type_invalid",
                "max_tokens must be an integer",
                request_id=_request_id_if_string(self.request_id),
            )
        if not 1 <= self.max_tokens <= max_tokens_limit:
            raise ServiceValidationError(
                "max_tokens_out_of_range",
                f"max_tokens must be between 1 and {max_tokens_limit}",
                request_id=_request_id_if_string(self.request_id),
            )
        _validate_request_id(self.request_id)

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        *,
        max_tokens_limit: int = DEFAULT_MAX_TOKENS_LIMIT,
        prompt_chars_limit: int = DEFAULT_PROMPT_CHARS_LIMIT,
    ) -> ServiceGenerateRequest:
        if not isinstance(payload, Mapping):
            raise ServiceValidationError(
                "request_not_object",
                "request payload must be an object",
            )
        unknown_fields = [key for key in payload if key not in _REQUEST_FIELDS]
        if unknown_fields:
            rendered_fields = sorted(repr(field) for field in unknown_fields)
            raise ServiceValidationError(
                "request_unknown_fields",
                f"unknown request fields: {', '.join(rendered_fields)}",
                request_id=_request_id_from_payload(payload),
            )
        missing_fields = sorted(_REQUIRED_REQUEST_FIELDS.difference(payload))
        if missing_fields:
            raise ServiceValidationError(
                "request_field_missing",
                f"missing request fields: {', '.join(missing_fields)}",
                request_id=_request_id_from_payload(payload),
            )

        values: dict[str, object] = {
            "prompt": payload["prompt"],
            "max_tokens": payload["max_tokens"],
            "max_tokens_limit": max_tokens_limit,
            "prompt_chars_limit": prompt_chars_limit,
        }
        if "request_id" in payload:
            values["request_id"] = payload["request_id"]
        return cls(**values)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "max_tokens": self.max_tokens,
            "request_id": self.request_id,
        }


@dataclass(frozen=True, slots=True)
class ServiceGenerateResult:
    request_id: str
    text: str
    finish_reason: str
    backend: str
    profile: str

    def __post_init__(self) -> None:
        _validate_request_id(self.request_id)
        _validate_string(self.text, name="text")
        _validate_string(self.finish_reason, name="finish_reason", allow_empty=False)
        _validate_string(self.backend, name="backend", allow_empty=False)
        _validate_string(self.profile, name="profile", allow_empty=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "request_id": self.request_id,
            "text": self.text,
            "finish_reason": self.finish_reason,
            "backend": self.backend,
            "profile": self.profile,
        }


@dataclass(frozen=True, slots=True)
class ServiceError:
    code: str
    detail: str
    request_id: str | None = None

    def __post_init__(self) -> None:
        _validate_string(self.code, name="code", allow_empty=False)
        _validate_string(self.detail, name="detail", allow_empty=False)
        if self.request_id is not None:
            _validate_request_id(self.request_id)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "status": "error",
            "error": self.code,
            "detail": self.detail,
        }
        if self.request_id is not None:
            payload["request_id"] = self.request_id
        return payload


@dataclass(frozen=True, slots=True)
class ServiceCancellationResult:
    request_id: str
    outcome: CancellationOutcome

    def __post_init__(self) -> None:
        _validate_request_id(self.request_id)
        if not isinstance(self.outcome, CancellationOutcome):
            raise ServiceValidationError(
                "cancellation_outcome_invalid",
                "outcome must be a CancellationOutcome",
                request_id=self.request_id,
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "outcome": self.outcome.value,
        }


_REQUEST_TRANSITIONS = {
    RequestState.QUEUED: frozenset(
        {RequestState.RUNNING, RequestState.CANCELLED}
    ),
    RequestState.RUNNING: frozenset(
        {RequestState.SUCCEEDED, RequestState.FAILED, RequestState.CANCELLED}
    ),
    RequestState.SUCCEEDED: frozenset(),
    RequestState.FAILED: frozenset(),
    RequestState.CANCELLED: frozenset(),
}

_SERVICE_TRANSITIONS = {
    ServiceState.STARTING: frozenset(
        {ServiceState.READY, ServiceState.STOPPED}
    ),
    ServiceState.READY: frozenset({ServiceState.DRAINING}),
    ServiceState.DRAINING: frozenset({ServiceState.STOPPED}),
    ServiceState.STOPPED: frozenset(),
}


def validate_request_transition(
    current: RequestState,
    target: RequestState,
) -> RequestState:
    if not isinstance(current, RequestState) or not isinstance(target, RequestState):
        raise ServiceValidationError(
            "request_state_invalid",
            "request states must be RequestState values",
        )
    if target not in _REQUEST_TRANSITIONS[current]:
        raise ServiceValidationError(
            "request_state_transition_invalid",
            f"cannot transition request from {current.value} to {target.value}",
        )
    return target


def validate_service_transition(
    current: ServiceState,
    target: ServiceState,
) -> ServiceState:
    if not isinstance(current, ServiceState) or not isinstance(target, ServiceState):
        raise ServiceValidationError(
            "service_state_invalid",
            "service states must be ServiceState values",
        )
    if target not in _SERVICE_TRANSITIONS[current]:
        raise ServiceValidationError(
            "service_state_transition_invalid",
            f"cannot transition service from {current.value} to {target.value}",
        )
    return target


def validate_request_id(request_id: object) -> str:
    """Return a valid request ID or raise a stable validation error."""

    _validate_request_id(request_id)
    return request_id  # type: ignore[return-value]


def _validate_positive_limit(value: object, *, name: str, code: str) -> None:
    if not _is_strict_int(value) or value <= 0:
        raise ServiceValidationError(
            code,
            f"{name} must be a positive integer",
        )


def _validate_request_id(request_id: object) -> None:
    if not isinstance(request_id, str):
        raise ServiceValidationError(
            "request_id_type_invalid",
            "request_id must be a string",
        )
    if _REQUEST_ID_PATTERN.fullmatch(request_id) is None:
        raise ServiceValidationError(
            "request_id_invalid",
            (
                "request_id must be 1-128 characters using only letters, "
                "digits, '.', '_', ':', or '-'"
            ),
        )


def _validate_string(
    value: object,
    *,
    name: str,
    allow_empty: bool = True,
) -> None:
    if not isinstance(value, str) or (
        not allow_empty and not value.strip()
    ):
        raise ServiceValidationError(
            f"{name}_invalid",
            f"{name} must be a string"
            + ("" if allow_empty else " containing at least one character"),
        )


def _is_strict_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _request_id_if_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return value if _REQUEST_ID_PATTERN.fullmatch(value) is not None else None


def _request_id_from_payload(payload: Mapping[str, object]) -> str | None:
    return _request_id_if_string(payload.get("request_id"))


__all__ = [
    "CancellationOutcome",
    "DEFAULT_MAX_TOKENS_LIMIT",
    "DEFAULT_PROMPT_CHARS_LIMIT",
    "MAX_REQUEST_ID_CHARS",
    "RequestState",
    "ServiceCancellationResult",
    "ServiceError",
    "ServiceGenerateRequest",
    "ServiceGenerateResult",
    "ServiceState",
    "ServiceValidationError",
    "validate_request_transition",
    "validate_request_id",
    "validate_service_transition",
]
