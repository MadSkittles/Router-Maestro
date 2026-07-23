"""Regression: the two Responses function-call schemas must not share a name.

Before the fix, two different classes were both named ``ResponsesFunctionCallOutput``
(the input-side ``function_call_output`` result at module line ~54, and the
output-side ``function_call`` item at ~203). ``ResponsesRequest.input`` still bound
correctly, because its union captured the input-side class at class-definition
time — request parsing was never broken. The defect was the *exported* name:
after the module finished executing, ``ResponsesFunctionCallOutput`` resolved to
the output-side class, so any ``isinstance(item, ResponsesFunctionCallOutput)``
check on an input item would wrongly return False (a latent hazard; no consumer
existed yet). The fix renames the output-side class to
``ResponsesFunctionCallOutputItem`` so the name is unambiguous.
"""

from router_maestro.server.schemas.responses import (
    ResponsesFunctionCallOutput,
    ResponsesFunctionCallOutputItem,
    ResponsesRequest,
    ResponsesResponse,
)


def test_input_side_function_call_output_binds_to_typed_model():
    # A client returning a tool result (function_call_output) as input.
    req = ResponsesRequest(
        model="gpt-5-codex",
        input=[{"type": "function_call_output", "call_id": "call_1", "output": "42"}],
    )
    item = req.input[0]
    # It binds to the input-side typed model, not a plain dict fallthrough.
    assert isinstance(item, ResponsesFunctionCallOutput)
    assert item.call_id == "call_1"
    assert item.output == "42"


def test_output_side_function_call_binds_to_output_item_model():
    resp = ResponsesResponse(
        id="resp_1",
        model="gpt-5-codex",
        output=[
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": "{}",
            }
        ],
    )
    item = resp.output[0]
    assert isinstance(item, ResponsesFunctionCallOutputItem)
    assert item.name == "get_weather"


def test_two_classes_are_distinct():
    assert ResponsesFunctionCallOutput is not ResponsesFunctionCallOutputItem
    assert ResponsesFunctionCallOutput.model_fields["type"].default == "function_call_output"
    assert ResponsesFunctionCallOutputItem.model_fields["type"].default == "function_call"
