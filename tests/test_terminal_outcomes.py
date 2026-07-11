"""Domain tests for canonical stream terminal outcome resolution."""

from __future__ import annotations

import pytest

from router_maestro.providers.base import (
    ResponseStatus,
    TerminalError,
    TerminalOutcome,
    TransportTermination,
    resolve_terminal_outcome,
)


def _outcome(
    transport: TransportTermination,
    status: ResponseStatus,
    *,
    finish_reason: str | None = None,
    incomplete_details: dict[str, str] | None = None,
    error: TerminalError | None = None,
) -> TerminalOutcome:
    return TerminalOutcome(
        transport=transport,
        response_status=status,
        finish_reason=finish_reason,
        incomplete_details=incomplete_details,
        error=error,
    )


def _assert_protocol_error(outcome: TerminalOutcome | None) -> None:
    assert outcome is not None
    assert outcome.transport is TransportTermination.EXCEPTION
    assert outcome.response_status is ResponseStatus.FAILED
    assert outcome.finish_reason is None
    assert outcome.error is not None
    assert outcome.error.code == "upstream_protocol_error"
    assert outcome.error.message


@pytest.mark.parametrize(
    ("transport", "status"),
    [
        (TransportTermination.UNEXPECTED_EOF, ResponseStatus.UNKNOWN),
        (TransportTermination.CLIENT_CANCELLED, ResponseStatus.CANCELLED),
        (TransportTermination.EXCEPTION, ResponseStatus.FAILED),
        (TransportTermination.EXPLICIT_TERMINAL, ResponseStatus.COMPLETED),
        (TransportTermination.EXPLICIT_TERMINAL, ResponseStatus.INCOMPLETE),
        (TransportTermination.EXPLICIT_TERMINAL, ResponseStatus.FAILED),
        (TransportTermination.EXPLICIT_TERMINAL, ResponseStatus.CANCELLED),
    ],
)
def test_resolver_accepts_every_legal_transport_status_pair(
    transport: TransportTermination,
    status: ResponseStatus,
):
    canonical = _outcome(transport, status)

    assert resolve_terminal_outcome(canonical, None) is canonical


@pytest.mark.parametrize(
    ("transport", "status"),
    [
        (transport, status)
        for transport in TransportTermination
        for status in ResponseStatus
        if (transport, status)
        not in {
            (TransportTermination.UNEXPECTED_EOF, ResponseStatus.UNKNOWN),
            (TransportTermination.CLIENT_CANCELLED, ResponseStatus.CANCELLED),
            (TransportTermination.EXCEPTION, ResponseStatus.FAILED),
            (TransportTermination.EXPLICIT_TERMINAL, ResponseStatus.COMPLETED),
            (TransportTermination.EXPLICIT_TERMINAL, ResponseStatus.INCOMPLETE),
            (TransportTermination.EXPLICIT_TERMINAL, ResponseStatus.FAILED),
            (TransportTermination.EXPLICIT_TERMINAL, ResponseStatus.CANCELLED),
        }
    ],
)
def test_resolver_converts_illegal_transport_status_pairs_to_protocol_error(
    transport: TransportTermination,
    status: ResponseStatus,
):
    _assert_protocol_error(resolve_terminal_outcome(_outcome(transport, status), None))


@pytest.mark.parametrize(
    ("finish_reason", "status"),
    [
        ("stop", ResponseStatus.COMPLETED),
        ("tool_calls", ResponseStatus.COMPLETED),
        ("length", ResponseStatus.INCOMPLETE),
        ("content_filter", ResponseStatus.INCOMPLETE),
    ],
)
def test_resolver_maps_known_legacy_finish_reasons(
    finish_reason: str,
    status: ResponseStatus,
):
    outcome = resolve_terminal_outcome(None, finish_reason)

    assert outcome == _outcome(
        TransportTermination.EXPLICIT_TERMINAL,
        status,
        finish_reason=finish_reason,
    )


def test_resolver_converts_unknown_legacy_finish_reason_to_protocol_error():
    _assert_protocol_error(resolve_terminal_outcome(None, "failed"))


@pytest.mark.parametrize(
    ("status", "finish_reason"),
    [
        (ResponseStatus.COMPLETED, "stop"),
        (ResponseStatus.COMPLETED, "tool_calls"),
        (ResponseStatus.INCOMPLETE, "length"),
        (ResponseStatus.INCOMPLETE, "content_filter"),
    ],
)
def test_resolver_accepts_matching_canonical_and_legacy_finish_reasons(
    status: ResponseStatus,
    finish_reason: str,
):
    canonical = _outcome(
        TransportTermination.EXPLICIT_TERMINAL,
        status,
        finish_reason=finish_reason,
    )

    assert resolve_terminal_outcome(canonical, finish_reason) is canonical


def test_resolver_enriches_canonical_outcome_with_consistent_legacy_finish_reason():
    canonical = _outcome(
        TransportTermination.EXPLICIT_TERMINAL,
        ResponseStatus.INCOMPLETE,
        incomplete_details={"reason": "max_output_tokens"},
    )

    resolved = resolve_terminal_outcome(canonical, "length")

    assert resolved == _outcome(
        TransportTermination.EXPLICIT_TERMINAL,
        ResponseStatus.INCOMPLETE,
        finish_reason="length",
        incomplete_details={"reason": "max_output_tokens"},
    )


def test_resolver_accepts_neutral_chat_projection_for_vendor_incomplete_reason():
    canonical = _outcome(
        TransportTermination.EXPLICIT_TERMINAL,
        ResponseStatus.INCOMPLETE,
        incomplete_details={"reason": "vendor_limit"},
    )

    assert resolve_terminal_outcome(canonical, "stop") is canonical


def test_resolver_accepts_neutral_chat_projection_without_incomplete_details():
    canonical = _outcome(
        TransportTermination.EXPLICIT_TERMINAL,
        ResponseStatus.INCOMPLETE,
    )

    assert resolve_terminal_outcome(canonical, "stop") is canonical


@pytest.mark.parametrize("reason", ["max_output_tokens", "content_filter"])
def test_resolver_rejects_neutral_projection_for_known_incomplete_reason(reason: str):
    canonical = _outcome(
        TransportTermination.EXPLICIT_TERMINAL,
        ResponseStatus.INCOMPLETE,
        incomplete_details={"reason": reason},
    )

    _assert_protocol_error(resolve_terminal_outcome(canonical, "stop"))


@pytest.mark.parametrize("canonical_finish", ["length", "content_filter"])
def test_resolver_rejects_neutral_projection_with_explicit_canonical_finish(
    canonical_finish: str,
):
    reason = "max_output_tokens" if canonical_finish == "length" else "content_filter"
    canonical = _outcome(
        TransportTermination.EXPLICIT_TERMINAL,
        ResponseStatus.INCOMPLETE,
        finish_reason=canonical_finish,
        incomplete_details={"reason": reason},
    )

    _assert_protocol_error(resolve_terminal_outcome(canonical, "stop"))


def test_resolver_converts_different_canonical_and_legacy_finish_reasons_to_error():
    canonical = _outcome(
        TransportTermination.EXPLICIT_TERMINAL,
        ResponseStatus.COMPLETED,
        finish_reason="stop",
    )

    _assert_protocol_error(resolve_terminal_outcome(canonical, "tool_calls"))


@pytest.mark.parametrize(
    ("status", "finish_reason"),
    [
        (ResponseStatus.COMPLETED, "length"),
        (ResponseStatus.INCOMPLETE, "stop"),
        (ResponseStatus.FAILED, "stop"),
        (ResponseStatus.CANCELLED, "stop"),
    ],
)
def test_resolver_converts_status_finish_mismatch_to_protocol_error(
    status: ResponseStatus,
    finish_reason: str,
):
    canonical = _outcome(
        TransportTermination.EXPLICIT_TERMINAL,
        status,
        finish_reason=finish_reason,
    )

    _assert_protocol_error(resolve_terminal_outcome(canonical, None))


@pytest.mark.parametrize(
    ("finish_reason", "details_reason"),
    [
        ("length", "content_filter"),
        ("content_filter", "max_output_tokens"),
    ],
)
def test_resolver_converts_inconsistent_incomplete_details_to_protocol_error(
    finish_reason: str,
    details_reason: str,
):
    canonical = _outcome(
        TransportTermination.EXPLICIT_TERMINAL,
        ResponseStatus.INCOMPLETE,
        finish_reason=finish_reason,
        incomplete_details={"reason": details_reason},
    )

    _assert_protocol_error(resolve_terminal_outcome(canonical, None))


@pytest.mark.parametrize("details_reason", ["max_output_tokens", "content_filter", "other"])
def test_resolver_allows_native_incomplete_details_without_legacy_finish_reason(
    details_reason: str,
):
    canonical = _outcome(
        TransportTermination.EXPLICIT_TERMINAL,
        ResponseStatus.INCOMPLETE,
        incomplete_details={"reason": details_reason},
    )

    assert resolve_terminal_outcome(canonical, None) is canonical


@pytest.mark.parametrize(
    "status",
    [ResponseStatus.COMPLETED, ResponseStatus.FAILED, ResponseStatus.CANCELLED],
)
def test_resolver_rejects_incomplete_details_on_non_incomplete_status(
    status: ResponseStatus,
):
    canonical = _outcome(
        TransportTermination.EXPLICIT_TERMINAL,
        status,
        incomplete_details={"reason": "max_output_tokens"},
    )

    _assert_protocol_error(resolve_terminal_outcome(canonical, None))


@pytest.mark.parametrize(
    ("status", "finish_reason", "incomplete_details"),
    [
        (ResponseStatus.COMPLETED, "stop", None),
        (
            ResponseStatus.INCOMPLETE,
            "length",
            {"reason": "max_output_tokens"},
        ),
    ],
)
def test_resolver_rejects_success_or_incomplete_outcome_with_terminal_error(
    status: ResponseStatus,
    finish_reason: str,
    incomplete_details: dict[str, str] | None,
):
    canonical = _outcome(
        TransportTermination.EXPLICIT_TERMINAL,
        status,
        finish_reason=finish_reason,
        incomplete_details=incomplete_details,
        error=TerminalError(code="upstream_failed", message="upstream failed"),
    )

    _assert_protocol_error(resolve_terminal_outcome(canonical, None))


@pytest.mark.parametrize(
    ("transport", "status"),
    [
        (TransportTermination.EXCEPTION, ResponseStatus.FAILED),
        (TransportTermination.EXPLICIT_TERMINAL, ResponseStatus.FAILED),
        (TransportTermination.CLIENT_CANCELLED, ResponseStatus.CANCELLED),
        (TransportTermination.EXPLICIT_TERMINAL, ResponseStatus.CANCELLED),
    ],
)
@pytest.mark.parametrize(
    "error",
    [None, TerminalError(code="terminal_error", message="terminal error")],
    ids=["without-error", "with-error"],
)
def test_resolver_allows_failed_and_cancelled_outcomes_with_or_without_error(
    transport: TransportTermination,
    status: ResponseStatus,
    error: TerminalError | None,
):
    canonical = _outcome(transport, status, error=error)

    assert resolve_terminal_outcome(canonical, None) is canonical


def test_resolver_allows_unexpected_eof_unknown_with_error():
    canonical = _outcome(
        TransportTermination.UNEXPECTED_EOF,
        ResponseStatus.UNKNOWN,
        error=TerminalError(code="unexpected_eof", message="unexpected EOF"),
    )

    assert resolve_terminal_outcome(canonical, None) is canonical
