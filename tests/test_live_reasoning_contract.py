"""Regression checks for live reasoning response contracts."""

import importlib

import pytest


def test_empty_max_token_response_accepts_upstream_usage_below_requested_cap():
    """The terminal reason proves exhaustion; billable usage need not equal the cap."""
    matrix = importlib.import_module("integration_tests.test_live_reasoning_matrix")

    matrix._assert_anthropic_reasoning_result(
        {
            "content": [],
            "stop_reason": "max_tokens",
            "usage": {"input_tokens": 170, "output_tokens": 2045},
        },
        requested_max_tokens=2048,
        requested_thinking_budget=None,
    )


def test_empty_max_token_response_rejects_zero_output_usage():
    matrix = importlib.import_module("integration_tests.test_live_reasoning_matrix")

    with pytest.raises(AssertionError):
        matrix._assert_anthropic_reasoning_result(
            {
                "content": [],
                "stop_reason": "max_tokens",
                "usage": {"input_tokens": 170, "output_tokens": 0},
            },
            requested_max_tokens=2048,
            requested_thinking_budget=None,
        )


def test_empty_response_rejects_non_max_token_terminal():
    matrix = importlib.import_module("integration_tests.test_live_reasoning_matrix")

    with pytest.raises(AssertionError):
        matrix._assert_anthropic_reasoning_result(
            {
                "content": [],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 170, "output_tokens": 2045},
            },
            requested_max_tokens=2048,
            requested_thinking_budget=None,
        )
