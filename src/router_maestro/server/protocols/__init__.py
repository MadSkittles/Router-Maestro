"""Shared protocol-boundary helpers."""

from router_maestro.server.protocols.anthropic_reducer import (
    AnthropicReducer,
    AnthropicStreamProtocolError,
    build_anthropic_response,
)
from router_maestro.server.protocols.errors import (
    client_error_response,
)
from router_maestro.server.protocols.responses_reducer import (
    ResponsesReducer,
    build_nonstream_snapshot,
)

__all__ = [
    "AnthropicReducer",
    "AnthropicStreamProtocolError",
    "ResponsesReducer",
    "build_anthropic_response",
    "build_nonstream_snapshot",
    "client_error_response",
]
