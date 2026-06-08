"""Tests for server observability metrics helpers."""

from router_maestro.server.observability.metrics import (
    API_KINDS,
    BOOL_LABEL_FALSE,
    BOOL_LABEL_TRUE,
    UNMATCHED_ROUTE_PATH_TEMPLATE,
    bool_label,
    normalize_http_method,
    path_template_from_scope,
)


class _Route:
    def __init__(self, path: str | None):
        self.path = path


def test_bool_label_uses_canonical_strings():
    assert bool_label(True) == BOOL_LABEL_TRUE
    assert bool_label(False) == BOOL_LABEL_FALSE
    assert BOOL_LABEL_TRUE == "true"
    assert BOOL_LABEL_FALSE == "false"


def test_normalize_http_method_folds_unknown_methods():
    assert normalize_http_method("GET") == "GET"
    assert normalize_http_method("post") == "POST"
    assert normalize_http_method("OPTIONS") == "OPTIONS"
    assert normalize_http_method("PROPFIND") == "OTHER"
    assert normalize_http_method("ZZARBITRARY") == "OTHER"


def test_path_template_from_scope_uses_route_template():
    assert path_template_from_scope({"route": _Route("/api/openai/v1/chat/completions")}) == (
        "/api/openai/v1/chat/completions"
    )


def test_path_template_from_scope_falls_back_for_unmatched_route():
    assert path_template_from_scope({}) == UNMATCHED_ROUTE_PATH_TEMPLATE
    assert path_template_from_scope({"route": None}) == UNMATCHED_ROUTE_PATH_TEMPLATE
    assert path_template_from_scope({"route": _Route(None)}) == UNMATCHED_ROUTE_PATH_TEMPLATE
    assert path_template_from_scope({"route": _Route("")}) == UNMATCHED_ROUTE_PATH_TEMPLATE
    assert UNMATCHED_ROUTE_PATH_TEMPLATE == "unmatched"


def test_api_kind_constants_are_complete():
    assert API_KINDS == (
        "openai_chat",
        "openai_responses",
        "openai_models",
        "anthropic_messages",
        "anthropic_count_tokens",
        "anthropic_models",
        "gemini_generate",
        "gemini_stream",
        "gemini_count_tokens",
        "admin",
    )
    assert len(API_KINDS) == 10
    assert len(set(API_KINDS)) == len(API_KINDS)
