import json

from proxy import (
    parse_leaked_tool_calls,
    repair_message,
    repair_nonstream_body,
    repair_stream_body,
)

# Real leaked samples pulled from Nanbeige4.2-3B transcripts (boxxo-model-bench eval).
NO_PARAMS = "<tool_call> \n<function=get_active_downloads>\n</function>\n</tool_call>"
ONE_PARAM = "<tool_call> <function=get_events>\n<parameter=range>\nweek\n</parameter>\n</function>\n</tool_call>"
MULTI_PARAM = (
    "<tool_call> <function=set_light>\n<parameter=target>\nvanity\n</parameter>\n"
    "<parameter=state>\noff\n</parameter>\n</function>\n</tool_call>"
)
TWO_CALLS = NO_PARAMS + "\n" + ONE_PARAM

# Degenerate variant seen in a second eval (HA Assist, 13-tool schema): no
# opening <function=> tag, dangling unmatched closing </function>.
BARE_NO_WRAPPER = (
    "<tool_call> HassTurnOn\n<parameter=name>\nKitchen Island\n</parameter>\n"
    "<parameter=area>\nKitchen\n</parameter>\n</function>\n</tool_call>"
)


def test_no_params():
    calls = parse_leaked_tool_calls(NO_PARAMS)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_active_downloads"
    assert json.loads(calls[0]["function"]["arguments"]) == {}


def test_one_param():
    calls = parse_leaked_tool_calls(ONE_PARAM)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_events"
    assert json.loads(calls[0]["function"]["arguments"]) == {"range": "week"}


def test_multi_param():
    calls = parse_leaked_tool_calls(MULTI_PARAM)
    args = json.loads(calls[0]["function"]["arguments"])
    assert args == {"target": "vanity", "state": "off"}


def test_two_calls():
    calls = parse_leaked_tool_calls(TWO_CALLS)
    assert len(calls) == 2
    assert calls[0]["function"]["name"] == "get_active_downloads"
    assert calls[1]["function"]["name"] == "get_events"


def test_call_ids_unique():
    calls = parse_leaked_tool_calls(TWO_CALLS)
    assert calls[0]["id"] != calls[1]["id"]


def test_bare_no_wrapper_variant():
    calls = parse_leaked_tool_calls(BARE_NO_WRAPPER)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "HassTurnOn"
    assert json.loads(calls[0]["function"]["arguments"]) == {"name": "Kitchen Island", "area": "Kitchen"}


def test_plain_text_not_matched():
    assert parse_leaked_tool_calls("The weather today is sunny.") is None


def test_empty_content_not_matched():
    assert parse_leaked_tool_calls("") is None
    assert parse_leaked_tool_calls(None) is None


def test_mixed_prose_and_tag_not_matched():
    # A model explaining tool-call syntax in prose shouldn't be misparsed.
    text = "You can use a <tool_call> block like this to call a function."
    assert parse_leaked_tool_calls(text) is None


def test_repair_message_rewrites_leaked_call():
    msg = {"role": "assistant", "content": ONE_PARAM}
    changed = repair_message(msg)
    assert changed is True
    assert msg["content"] is None
    assert msg["tool_calls"][0]["function"]["name"] == "get_events"


def test_repair_message_noop_when_already_structured():
    msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "x", "type": "function", "function": {"name": "get_events", "arguments": "{}"}}],
    }
    changed = repair_message(dict(msg))
    assert changed is False


def test_repair_message_noop_on_plain_text():
    msg = {"role": "assistant", "content": "Hello there."}
    changed = repair_message(msg)
    assert changed is False
    assert msg["content"] == "Hello there."


def test_repair_nonstream_body_rewrites_and_sets_finish_reason():
    body = json.dumps(
        {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": ONE_PARAM},
                    "finish_reason": "stop",
                }
            ]
        }
    ).encode()
    out = repair_nonstream_body(body)
    payload = json.loads(out)
    choice = payload["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "get_events"


def test_repair_nonstream_body_passthrough_on_normal_response():
    body = json.dumps(
        {"choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello there."}, "finish_reason": "stop"}]}
    ).encode()
    out = repair_nonstream_body(body)
    assert out == body


def test_repair_nonstream_body_passthrough_on_malformed_json():
    body = b"not json"
    assert repair_nonstream_body(body) == body


def _sse(events):
    out = "".join(f"data: {json.dumps(e)}\n\n" for e in events)
    out += "data: [DONE]\n\n"
    return out.encode()


def test_repair_stream_body_converts_leaked_call():
    events = [
        {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"content": "<tool_call> <function=get_events>\n"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"content": "<parameter=range>\nweek\n</parameter>\n</function>\n</tool_call>"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    out = repair_stream_body(_sse(events))
    text = out.decode()
    data_lines = [l[len("data: "):] for l in text.split("\n\n") if l.startswith("data: ") and l != "data: [DONE]"]
    parsed = [json.loads(l) for l in data_lines]
    tool_call_chunks = [p for p in parsed if p["choices"][0]["delta"].get("tool_calls")]
    assert len(tool_call_chunks) == 1
    fn = tool_call_chunks[0]["choices"][0]["delta"]["tool_calls"][0]["function"]
    assert fn["name"] == "get_events"
    assert json.loads(fn["arguments"]) == {"range": "week"}
    assert parsed[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_repair_stream_body_passthrough_on_normal_text():
    events = [
        {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"content": "Hello there."}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    raw = _sse(events)
    assert repair_stream_body(raw) == raw


def test_repair_stream_body_passthrough_when_already_structured():
    events = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "get_events", "arguments": "{}"}}]},
                    "finish_reason": None,
                }
            ]
        },
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
    ]
    raw = _sse(events)
    assert repair_stream_body(raw) == raw


def test_repair_stream_body_passthrough_on_empty():
    assert repair_stream_body(b"") == b""
