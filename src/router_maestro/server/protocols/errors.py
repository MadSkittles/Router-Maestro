"""Safe protocol-native error normalization and encoding."""

from dataclasses import dataclass
from typing import Literal

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from router_maestro.providers import (
    ProviderError,
    ProviderFailureKind,
    ProviderFailureSignal,
    RequestOptionError,
)

ProtocolName = Literal["openai", "anthropic", "gemini"]
ProtocolSurface = Literal["openai_chat", "openai_responses", "anthropic", "gemini"]
ProtocolSelector = ProtocolName | ProtocolSurface
PROVIDER_FAILURE_SIGNAL_HEADER = "X-Router-Maestro-Error-Signal"
_PROVIDER_SIGNAL_HEADERS = {
    ProviderFailureSignal.COPILOT_BARE_BAD_REQUEST: "copilot_bare_bad_request",
}


@dataclass(frozen=True, slots=True)
class NormalizedProtocolError:
    """Client-safe error fields shared by public protocol encoders."""

    status_code: int
    message: str
    kind: ProviderFailureKind
    parameter: str | None = None
    code: str | None = None
    headers: dict[str, str] | None = None


def protocol_surface_for_path(path: str) -> ProtocolSurface | None:
    """Classify a public inference namespace without requiring a matched route."""
    responses_prefixes = ("/api/openai/v1/responses", "/api/openai/beta/v1/responses")
    if any(path == prefix or path.startswith(f"{prefix}/") for prefix in responses_prefixes):
        return "openai_responses"
    if path == "/api/openai" or path.startswith("/api/openai/"):
        return "openai_chat"
    if path == "/v1/messages" or path.startswith("/v1/messages/"):
        return "anthropic"
    if path == "/api/anthropic" or path.startswith("/api/anthropic/"):
        return "anthropic"
    if path == "/api/gemini" or path.startswith("/api/gemini/"):
        return "gemini"
    return None


def normalize_protocol_error(error: Exception) -> NormalizedProtocolError:
    """Reduce supported exception types to fields safe for a downstream client."""
    if isinstance(error, RequestValidationError):
        return NormalizedProtocolError(
            status_code=422,
            message="Invalid request",
            kind=ProviderFailureKind.CLIENT_REQUEST,
            parameter=_validation_parameter(error),
            code="invalid_request",
        )

    if isinstance(error, ProviderError):
        signal_value = _PROVIDER_SIGNAL_HEADERS.get(error.signal)
        return NormalizedProtocolError(
            status_code=error.downstream_status_code,
            message=error.safe_message,
            kind=error.kind,
            parameter=error.parameter,
            code=(
                "unsupported_parameter"
                if isinstance(error, RequestOptionError) and error.parameter
                else _error_code(error.downstream_status_code, error.kind)
            ),
            headers=(
                {PROVIDER_FAILURE_SIGNAL_HEADER: signal_value} if signal_value is not None else None
            ),
        )

    if isinstance(error, StarletteHTTPException):
        status_code = error.status_code
        return NormalizedProtocolError(
            status_code=status_code,
            message=(
                error.detail if isinstance(error.detail, str) else _default_message(status_code)
            ),
            kind=_failure_kind(status_code),
            code=_error_code(status_code, _failure_kind(status_code)),
            headers=dict(error.headers) if error.headers else None,
        )

    return NormalizedProtocolError(
        status_code=500,
        message="Internal server error",
        kind=ProviderFailureKind.UNKNOWN,
        code="server_error",
    )


def protocol_error_response(
    error: Exception,
    protocol: ProtocolSelector,
) -> JSONResponse:
    """Encode an exception as a native non-stream response for one public surface."""
    surface = _coerce_surface(protocol)
    normalized = normalize_protocol_error(error)
    if surface in {"openai_chat", "openai_responses"}:
        content = {"error": _openai_error(normalized)}
    elif surface == "anthropic":
        content = {
            "type": "error",
            "error": {
                "type": _anthropic_error_type(normalized),
                "message": normalized.message,
            },
        }
    else:
        details = []
        if normalized.parameter:
            details.append(
                {
                    "reason": normalized.code or "invalid_request",
                    "parameter": normalized.parameter,
                }
            )
        content = {
            "error": {
                "code": normalized.status_code,
                "message": normalized.message,
                "status": _gemini_error_status(normalized),
                "details": details,
            }
        }
    return JSONResponse(
        status_code=normalized.status_code,
        content=content,
        headers=normalized.headers,
    )


def postcommit_error_data(
    error: Exception,
    protocol: ProtocolSurface,
) -> dict:
    """Return protocol-native error data for an already-committed stream."""
    normalized = normalize_protocol_error(error)
    if protocol == "openai_chat":
        openai = _openai_error(normalized)
        error_data = {
            "message": openai["message"],
            "type": openai["type"],
        }
        if openai["code"] is not None:
            error_data["code"] = openai["code"]
        return {"error": error_data}
    if protocol == "openai_responses":
        return {
            "code": normalized.code or "server_error",
            "message": normalized.message,
        }
    if protocol == "anthropic":
        return {
            "type": "error",
            "error": {
                "type": _anthropic_error_type(normalized),
                "message": normalized.message,
            },
        }
    return {
        "error": {
            "code": normalized.status_code,
            "message": normalized.message,
            "status": _gemini_error_status(normalized),
            "details": [],
        }
    }


def client_error_response(error: RequestOptionError, protocol: ProtocolName) -> JSONResponse:
    """Encode an unsupported request option in the selected public protocol."""
    if not isinstance(error, RequestOptionError):
        raise TypeError("client_error_response requires a RequestOptionError")
    return protocol_error_response(error, protocol)


def _coerce_surface(protocol: ProtocolSelector) -> ProtocolSurface:
    if protocol == "openai":
        return "openai_chat"
    return protocol


def _validation_parameter(error: RequestValidationError) -> str | None:
    errors = error.errors()
    if not errors:
        return None
    location = errors[0].get("loc", ())
    for part in reversed(location):
        if isinstance(part, str) and part not in {"body", "query", "path", "header"}:
            return part
    return None


def _failure_kind(status_code: int) -> ProviderFailureKind:
    if status_code in {401, 403}:
        return ProviderFailureKind.AUTHENTICATION
    if status_code in {429, 529}:
        return ProviderFailureKind.RATE_LIMIT
    if 400 <= status_code < 500:
        return ProviderFailureKind.CLIENT_REQUEST
    return ProviderFailureKind.UNKNOWN


def _default_message(status_code: int) -> str:
    if status_code in {400, 422}:
        return "Invalid request"
    if status_code == 401:
        return "Authentication failed"
    if status_code == 403:
        return "Permission denied"
    if status_code == 404:
        return "Not found"
    if status_code == 405:
        return "Method not allowed"
    if status_code == 429:
        return "Rate limit exceeded"
    if status_code == 529:
        return "Server overloaded"
    if status_code >= 500:
        return "Internal server error"
    return "Request failed"


def _error_code(status_code: int, kind: ProviderFailureKind) -> str:
    if status_code in {400, 422, 405}:
        return "invalid_request"
    if status_code == 401:
        return "authentication_error"
    if status_code == 403:
        return "permission_denied"
    if status_code == 404:
        return "not_found"
    if status_code == 529:
        return "overloaded"
    if status_code == 429 or kind is ProviderFailureKind.RATE_LIMIT:
        return "rate_limit_exceeded"
    if kind is ProviderFailureKind.UPSTREAM_PROTOCOL:
        return "upstream_protocol_error"
    return "server_error"


def _openai_error(error: NormalizedProtocolError) -> dict[str, str | None]:
    if error.status_code == 401:
        error_type = "authentication_error"
    elif error.status_code == 403:
        error_type = "permission_error"
    elif error.status_code in {429, 529} or error.kind is ProviderFailureKind.RATE_LIMIT:
        error_type = "rate_limit_error"
    elif 400 <= error.status_code < 500:
        error_type = "invalid_request_error"
    else:
        error_type = "api_error"
    return {
        "message": error.message,
        "type": error_type,
        "param": error.parameter,
        "code": error.code,
    }


def _anthropic_error_type(error: NormalizedProtocolError) -> str:
    if error.status_code == 401:
        return "authentication_error"
    if error.status_code == 403:
        return "permission_error"
    if error.status_code == 404:
        return "not_found_error"
    if error.status_code == 413:
        return "request_too_large"
    if error.status_code == 529:
        return "overloaded_error"
    if error.status_code == 429 or error.kind is ProviderFailureKind.RATE_LIMIT:
        return "rate_limit_error"
    if 400 <= error.status_code < 500:
        return "invalid_request_error"
    return "api_error"


def _gemini_error_status(error: NormalizedProtocolError) -> str:
    if error.status_code in {400, 405, 422}:
        return "INVALID_ARGUMENT"
    if error.status_code == 401:
        return "UNAUTHENTICATED"
    if error.status_code == 403:
        return "PERMISSION_DENIED"
    if error.status_code == 404:
        return "NOT_FOUND"
    if error.status_code in {429, 529} or error.kind is ProviderFailureKind.RATE_LIMIT:
        return "RESOURCE_EXHAUSTED"
    if error.status_code in {503, 504}:
        return "UNAVAILABLE"
    return "INTERNAL"
