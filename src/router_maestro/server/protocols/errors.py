"""Minimal protocol-native encoding for provider client-request failures."""

from typing import Literal

from fastapi.responses import JSONResponse
from pydantic import BaseModel

from router_maestro.providers import RequestOptionError

ProtocolName = Literal["openai", "anthropic", "gemini"]


def unrepresented_option_error(request: BaseModel) -> RequestOptionError | None:
    """Reject retained extras on an explicitly extensible request/config model."""
    parameter = _first_unrepresented_option(request)
    if parameter is None:
        return None
    return RequestOptionError(
        f"Unsupported request option '{parameter}'",
        parameter=parameter,
    )


def _first_unrepresented_option(value: object, prefix: str = "") -> str | None:
    if isinstance(value, BaseModel):
        extras = value.model_extra or {}
        if extras:
            name = next(iter(extras))
            return f"{prefix}.{name}" if prefix else name

        for name, field in type(value).model_fields.items():
            wire_name = field.alias or name
            nested_prefix = f"{prefix}.{wire_name}" if prefix else wire_name
            if parameter := _first_unrepresented_option(
                getattr(value, name, None),
                nested_prefix,
            ):
                return parameter
        return None

    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            nested_prefix = f"{prefix}[{index}]"
            if parameter := _first_unrepresented_option(item, nested_prefix):
                return parameter
        return None

    if isinstance(value, dict):
        # Plain dictionaries are opaque vendor payloads. Recurse only so a
        # model nested inside a typed mapping cannot hide retained extras;
        # dictionary keys themselves are not interpreted as request options.
        for name, item in value.items():
            nested_prefix = f"{prefix}.{name}" if prefix else str(name)
            if parameter := _first_unrepresented_option(item, nested_prefix):
                return parameter
    return None


def client_error_response(error: RequestOptionError, protocol: ProtocolName) -> JSONResponse:
    """Encode an unsupported request option in the selected public protocol."""
    if not isinstance(error, RequestOptionError):
        raise TypeError("client_error_response requires a RequestOptionError")
    parameter = error.parameter
    if protocol == "openai":
        content = {
            "error": {
                "message": error.safe_message,
                "type": "invalid_request_error",
                "param": parameter,
                "code": "unsupported_parameter" if parameter else "invalid_request",
            }
        }
    elif protocol == "anthropic":
        content = {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": error.safe_message,
            },
        }
    else:
        details = []
        if parameter:
            details.append({"reason": "unsupported_parameter", "parameter": parameter})
        content = {
            "error": {
                "code": error.status_code,
                "message": error.safe_message,
                "status": "INVALID_ARGUMENT",
                "details": details,
            }
        }
    return JSONResponse(status_code=error.status_code, content=content)
