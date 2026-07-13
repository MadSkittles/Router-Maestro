"""Shared protocol-boundary helpers."""

from router_maestro.server.protocols.errors import (
    client_error_response,
    unrepresented_option_error,
)

__all__ = ["client_error_response", "unrepresented_option_error"]
