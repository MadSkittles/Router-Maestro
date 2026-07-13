"""Shared protocol-boundary helpers."""

from router_maestro.server.protocols.errors import (
    client_error_response,
    unrepresented_option_error,
)
from router_maestro.server.protocols.responses_reducer import (
    ResponsesReducer,
    build_nonstream_snapshot,
)

__all__ = [
    "ResponsesReducer",
    "build_nonstream_snapshot",
    "client_error_response",
    "unrepresented_option_error",
]
