"""Lazy optional ASGI transport for the bounded local service."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import logging
import math

from .service_contracts import (
    CancellationOutcome,
    ServiceError,
    ServiceGenerateRequest,
    ServiceValidationError,
)
from .service_runtime import ServiceRuntimeError


DEFAULT_MAX_REQUEST_BYTES = 1_048_576
DEFAULT_REQUEST_TIMEOUT_S = 300.0

_LOGGER = logging.getLogger(__name__)

_HEALTH_FIELDS = (
    "state",
    "backend",
    "profile",
    "running_count",
    "queued_count",
    "queue_limit",
    "max_tokens_limit",
    "terminal_record_limit",
)
_RUNTIME_STATUS = {
    "empty_generation": 500,
    "duplicate_request_id": 409,
    "max_tokens_limit_exceeded": 400,
    "queue_full": 429,
    "request_type_invalid": 400,
    "service_not_ready": 503,
}


def create_service_app(
    controller: object,
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
    *, request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
) -> object:
    """Create the optional Starlette adapter without eager dependencies."""

    _validate_positive_int(
        max_request_bytes,
        name="max_request_bytes",
    )
    if (isinstance(request_timeout_s, bool) or not isinstance(request_timeout_s, (int, float))
            or not math.isfinite(request_timeout_s) or request_timeout_s <= 0):
        raise ServiceRuntimeError("request_timeout_invalid", "request timeout must be a finite positive number")
    try:
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.responses import JSONResponse
        from starlette.routing import Route
    except ImportError as error:
        raise _dependency_missing() from error

    def error_response(
        code: str,
        detail: str,
        status_code: int,
        *,
        request_id: str | None = None,
    ):
        return JSONResponse(
            ServiceError(
                code=code,
                detail=detail,
                request_id=request_id,
            ).to_dict(),
            status_code=status_code,
        )

    async def health(_request: Request):
        try:
            raw_snapshot = controller.snapshot().to_dict()  # type: ignore[attr-defined]
            snapshot = {
                field: raw_snapshot[field]
                for field in _HEALTH_FIELDS
            }
        except Exception:
            _LOGGER.exception("ASGI health snapshot failed")
            return error_response(
                "service_unavailable",
                "service health is unavailable",
                503,
            )
        ready = snapshot["state"] == "ready"
        return JSONResponse(
            {
                "status": "ok" if ready else "unavailable",
                "service": snapshot,
            },
            status_code=200 if ready else 503,
        )

    async def generate(request: Request):
        body_or_response = await _read_limited_body(
            request,
            max_request_bytes=max_request_bytes,
            error_response=error_response,
        )
        if not isinstance(body_or_response, bytes):
            return body_or_response

        try:
            payload = _load_strict_json(body_or_response)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return error_response(
                "malformed_json",
                "request body must be valid finite JSON",
                400,
            )
        if not isinstance(payload, dict):
            return error_response(
                "request_not_object",
                "request payload must be an object",
                400,
            )

        try:
            snapshot = controller.snapshot()  # type: ignore[attr-defined]
            service_request = ServiceGenerateRequest.from_dict(
                payload,
                max_tokens_limit=snapshot.max_tokens_limit,
                prompt_chars_limit=max_request_bytes,
            )
            handle = controller.submit(service_request)  # type: ignore[attr-defined]
        except ServiceValidationError as error:
            return error_response(
                error.code,
                error.detail,
                400,
                request_id=error.request_id,
            )
        except ServiceRuntimeError as error:
            return _runtime_error_response(error, error_response)
        except Exception:
            request_id = _request_id_from_payload(payload)
            _LOGGER.exception(
                "ASGI request submission failed (request_id=%s)",
                request_id,
            )
            return error_response(
                "internal_error",
                "internal service error",
                500,
                request_id=request_id,
            )

        try:
            result = await asyncio.to_thread(handle.result, timeout=request_timeout_s)
        except TimeoutError:
            # Request cooperative stop where supported. Capacity/ownership stay
            # with the worker until it acknowledges a safe execution boundary.
            handle.cancel()
            return error_response(
                "request_timeout", "response wait deadline exceeded; cancellation requested where supported",
                504, request_id=service_request.request_id,
            )
        except asyncio.CancelledError:
            if not handle.cancelled():
                raise
            return error_response(
                "request_cancelled",
                "request was cancelled",
                409,
                request_id=service_request.request_id,
            )
        except ServiceRuntimeError as error:
            return _runtime_error_response(error, error_response)
        except Exception:
            _LOGGER.exception(
                "ASGI generation result failed (request_id=%s)",
                service_request.request_id,
            )
            return error_response(
                "internal_error",
                "internal service error",
                500,
                request_id=service_request.request_id,
            )
        return JSONResponse(result.to_dict(), status_code=200)

    async def cancel(request: Request):
        request_id = request.path_params["request_id"]
        try:
            result = controller.cancel(request_id)  # type: ignore[attr-defined]
        except ServiceValidationError as error:
            return error_response(
                error.code,
                error.detail,
                400,
                request_id=error.request_id,
            )
        except Exception:
            safe_request_id = _request_id_from_payload(
                {"request_id": request_id}
            )
            _LOGGER.exception(
                "ASGI cancellation failed (request_id=%s)",
                safe_request_id,
            )
            return error_response(
                "internal_error",
                "internal service error",
                500,
            )

        if result.outcome is CancellationOutcome.CANCELLATION_REQUESTED:
            return JSONResponse({"status": "ok", **result.to_dict()}, status_code=202)
        if result.outcome is CancellationOutcome.CANCELLED:
            return JSONResponse(
                {"status": "ok", **result.to_dict()},
                status_code=200,
            )
        if result.outcome is CancellationOutcome.RUNNING_CANCEL_UNSUPPORTED:
            return error_response(
                result.outcome.value,
                "running request cancellation is not supported",
                409,
                request_id=result.request_id,
            )
        if result.outcome is CancellationOutcome.ALREADY_TERMINAL:
            return error_response(
                result.outcome.value,
                "request is already terminal",
                409,
                request_id=result.request_id,
            )
        return error_response(
            "not_found",
            "request was not found",
            404,
            request_id=result.request_id,
        )

    @asynccontextmanager
    async def lifespan(_app):
        try:
            yield
        finally:
            await asyncio.to_thread(controller.shutdown)  # type: ignore[attr-defined]

    return Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/v1/generate", generate, methods=["POST"]),
            Route(
                "/v1/requests/{request_id}",
                cancel,
                methods=["DELETE"],
            ),
        ],
        lifespan=lifespan,
    )


def run_service_app(
    app: object,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    log_level: str = "info",
) -> None:
    """Run an ASGI app with the optional Uvicorn dependency."""

    try:
        import uvicorn
    except ImportError as error:
        raise _dependency_missing() from error
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level,
    )


async def _read_limited_body(
    request: object,
    *,
    max_request_bytes: int,
    error_response,
) -> bytes | object:
    content_length = request.headers.get("content-length")
    if (
        content_length is not None
        and content_length.isdigit()
        and int(content_length) > max_request_bytes
    ):
        return error_response(
            "request_body_too_large",
            f"request body exceeds the {max_request_bytes}-byte limit",
            413,
        )

    body = bytearray()
    try:
        async for chunk in request.stream():
            if len(chunk) > max_request_bytes - len(body):
                return error_response(
                    "request_body_too_large",
                    f"request body exceeds the {max_request_bytes}-byte limit",
                    413,
                )
            body.extend(chunk)
    except Exception:
        _LOGGER.exception("ASGI request body read failed")
        return error_response(
            "request_body_incomplete",
            "request body could not be read",
            400,
        )
    return bytes(body)


def _load_strict_json(body: bytes) -> object:
    return json.loads(
        body.decode("utf-8"),
        parse_constant=_reject_non_finite_json,
    )


def _reject_non_finite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _runtime_error_response(error: ServiceRuntimeError, error_response):
    if error.code == "generation_failed":
        return error_response(
            "generation_failed",
            "generation failed",
            500,
            request_id=error.request_id,
        )
    status = _RUNTIME_STATUS.get(error.code)
    if status is not None:
        return error_response(
            error.code,
            error.detail,
            status,
            request_id=error.request_id,
        )
    return error_response(
        "internal_error",
        "internal service error",
        500,
        request_id=error.request_id,
    )


def _request_id_from_payload(payload: dict[str, object]) -> str | None:
    request_id = payload.get("request_id")
    if not isinstance(request_id, str):
        return None
    try:
        ServiceGenerateRequest.from_dict(
            {
                "prompt": "validation-only",
                "max_tokens": 1,
                "request_id": request_id,
            }
        )
    except ServiceValidationError:
        return None
    return request_id


def _validate_positive_int(value: object, *, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ServiceRuntimeError(
            f"{name}_invalid",
            f"{name} must be a positive integer",
        )


def _dependency_missing() -> ServiceRuntimeError:
    from . import _extra_install_command

    return ServiceRuntimeError(
        "service_dependency_missing",
        "install diffusor-rt[serve] from the release wheel to use the local service. "
        f"Run: {_extra_install_command('serve')}",
    )


__all__ = [
    "DEFAULT_MAX_REQUEST_BYTES",
    "create_service_app",
    "run_service_app",
]
