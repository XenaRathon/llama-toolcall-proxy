import gzip
import json
import os
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

from proxy import (
    ProxyHandler,
    _filter_headers,
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


def test_filter_headers_drops_hop_by_hop():
    headers = [
        ("Connection", "Keep-Alive, close"),
        ("Keep-Alive", "timeout=5, max=1000"),
        ("TE", "trailers, deflate"),
        ("Trailer", "Max-Forwards"),
        ("Transfer-Encoding", "chunked"),
        ("Upgrade", "h2c"),
        ("Proxy-Authenticate", "Basic"),
        ("Accept", "*/*"),
        ("Accept-Encoding", "gzip"),
        ("Host", "example.com"),
    ]
    filtered = _filter_headers(headers)
    assert filtered == [("Accept", "*/*"), ("Accept-Encoding", "gzip"), ("Host", "example.com")]


def test_filter_headers_drops_connection_tokens():
    headers = [
        ("Connection", "X-Custom-Header, Keep-Alive"),
        ("X-Custom-Header", "foo"),
        ("Keep-Alive", "timeout=5"),
        ("Accept", "*/*"),
    ]
    filtered = _filter_headers(headers)
    assert filtered == [("Accept", "*/*")]


class MockUpstream(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        pass

    def _handle(self):
        body = b""
        if "Content-Length" in self.headers:
            body = self.rfile.read(int(self.headers["Content-Length"]))

        if self.path == "/v1/chat/completions/gzipped":
            # Test verifies that proxy sends identity
            assert self.headers.get("Accept-Encoding") == "identity"
            resp = json.dumps(
                {
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": ONE_PARAM},
                            "finish_reason": "stop",
                        }
                    ]
                }
            ).encode("utf-8")
            # Upstream respects identity and returns raw uncompressed response
            self.send_response(200)
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
            return

        if self.path == "/v1/test_methods":
            self.send_response(200)
            self.send_header("X-Method-Received", self.command)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(200)
        self.send_header("Connection", "Keep-Alive")
        self.send_header("Keep-Alive", "timeout=5")
        self.send_header("X-Received-Method", self.command)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"OK")

    def do_GET(self): self._handle()
    def do_POST(self): self._handle()
    def do_PUT(self): self._handle()
    def do_DELETE(self): self._handle()


def test_proxy_e2e():
    upstream = HTTPServer(("127.0.0.1", 0), MockUpstream)
    upstream_port = upstream.server_port
    threading.Thread(target=upstream.serve_forever, daemon=True).start()

    ProxyHandler.upstream = f"http://127.0.0.1:{upstream_port}"
    ProxyHandler.timeout = 5.0
    proxy = HTTPServer(("127.0.0.1", 0), ProxyHandler)
    proxy_port = proxy.server_port
    threading.Thread(target=proxy.serve_forever, daemon=True).start()

    try:

        # Test 1: gzipped JSON body is requested as identity and thus repaired successfully
        req = urllib.request.Request(
            f"http://127.0.0.1:{proxy_port}/v1/chat/completions/gzipped",
            data=b"{}",
            method="POST",
        )
        req.add_header("Accept-Encoding", "gzip")
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
            payload = json.loads(data.decode("utf-8"))
            # Repaired!
            assert payload["choices"][0]["finish_reason"] == "tool_calls"
            assert payload["choices"][0]["message"]["content"] is None

        # Test 2: Hop-by-hop headers dropped and proxy works for other endpoints
        req = urllib.request.Request(
            f"http://127.0.0.1:{proxy_port}/v1/other",
            data=b"{}",
            method="POST",
        )
        req.add_header("Connection", "Keep-Alive")
        with urllib.request.urlopen(req) as resp:
            assert resp.getheader("Connection") is None
            assert resp.getheader("Keep-Alive") is None

        # Test 3: PUT/DELETE methods pass through
        for method in ("PUT", "DELETE"):
            req = urllib.request.Request(
                f"http://127.0.0.1:{proxy_port}/v1/test_methods",
                data=b"{}",
                method=method,
            )
            with urllib.request.urlopen(req) as resp:
                assert resp.getheader("X-Method-Received") == method
                assert resp.read() == b"{}"
    finally:
        upstream.shutdown()
        proxy.shutdown()


@patch("os.environ.get")
@patch("sys.stderr.write")
@patch("proxy.ThreadingHTTPServer")
def test_timeout_env_var_read(mock_server, mock_stderr, mock_environ):
    def env_side_effect(key, default=None):
        if key == "UPSTREAM_URL":
            return "http://example.com"
        if key == "UPSTREAM_TIMEOUT":
            return "42.5"
        return default

    mock_environ.side_effect = env_side_effect

    # Run main which will start the server blockingly, so we need to mock serve_forever
    mock_server.return_value.serve_forever.side_effect = KeyboardInterrupt
    from proxy import main
    main()

    # Test that the timeout was set on the class correctly
    assert ProxyHandler.timeout == 42.5
