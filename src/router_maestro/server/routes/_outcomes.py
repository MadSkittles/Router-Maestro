"""Request-context outcome recording for non-stream protocol routes."""

from router_maestro.providers import ChatResponse, TerminalOutcome, resolve_terminal_outcome
from router_maestro.runtime import get_current_request_context


def record_terminal_outcome(outcome: TerminalOutcome) -> None:
    """Record provider semantics without conflating them with the HTTP status."""
    context = get_current_request_context()
    if context is not None:
        context.outcome = outcome


def record_chat_response_outcome(response: ChatResponse) -> TerminalOutcome | None:
    """Resolve canonical or legacy Chat terminal data and record it when present."""
    outcome = resolve_terminal_outcome(response.terminal_outcome, response.finish_reason)
    if outcome is not None:
        record_terminal_outcome(outcome)
    return outcome
