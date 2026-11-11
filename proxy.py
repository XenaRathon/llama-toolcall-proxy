#!/usr/bin/env python3
"""
Reverse proxy that sits in front of an OpenAI-compatible chat completions
endpoint (llama-server, vLLM, etc.) and repairs a specific failure mode:
a model that decides on a tool call correctly (right name, right arguments)
but whose inference stack fails to parse it into the API's structured
`tool_calls` field, leaking it instead as literal text in `content`, e.g.:

    <tool_call>
    <function=get_events>
    <parameter=range>
    week
    </parameter>
    </function>
    </tool_call>

This proxy detects that exact shape and rewrites the response into a
proper OpenAI tool_calls payload before it reaches the caller. Everything
else passes through untouched.

Env vars:
  UPSTREAM_URL   Root URL of the real server, e.g. http://192.168.1.196:8090
                 (NOT including /v1 -- the client's own request path,
                 including /v1/..., is forwarded onto this root unchanged.
                 That way clients just swap base_url from
                 http://<server>:8090/v1 to http://<this-proxy>:8091/v1
                 and nothing else changes.) (required)
  LISTEN_HOST    Default 0.0.0.0
  LISTEN_PORT    Default 8091
"""
import json
import os
import re
import sys
import threading
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TOOL_CALL_BLOCK_RE = re.compile(
    # Handles two observed leak shapes: "<function=NAME>...</function>" and a
    # degenerate "<tool_call> NAME ...</function>" (no opening <function=> tag,
    # dangling closing one) -- both wrap the same <parameter=...> body.
    r"<tool_call>\s*(?:<function=)?([a-zA-Z_][a-zA-Z0-9_]*)>?(.*?)(?:</function>\s*)?</tool_call>",
    re.DOTALL,
)
PARAM_RE = re.compile(r"<parameter=([a-zA-Z_][a-zA-Z0-9_]*)>\s*(.*?)\s*</parameter>", re.DOTALL)
WRAPPER_RE = re.compile(r"^(?:\s*<tool_call>.*?</tool_call>\s*)+$", re.DOTALL)


def parse_leaked_tool_calls(content):
    """Return a list of OpenAI-shaped tool_calls dicts if `content` is
    entirely composed of one or more <tool_call> blocks, else None."""
    if not content or "<tool_call>" not in content:
        return None
    if not WRAPPER_RE.match(content):
        return None
    calls = []
    for name, body in TOOL_CALL_BLOCK_RE.findall(content):
        params = {k: v.strip() for k, v in PARAM_RE.findall(body)}
        calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(params)},
            }
        )
    return calls or None


def repair_message(message):
    """Mutates a chat-completion `message` dict in place if it contains a
    leaked tool call. Returns True if it was repaired."""
    if message.get("tool_calls"):
        return False
    calls = parse_leaked_tool_calls(message.get("content"))
    if not calls:
        return False
    message["content"] = None
    message["tool_calls"] = calls
    return True


def repair_nonstream_body(raw_bytes):
    try:
        payload = json.loads(raw_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        sys.stderr.write(f"[proxy] failed to parse body for repair: {e}\n")
        return raw_bytes
    changed = False
    for choice in payload.get("choices", []):
        message = choice.get("message")
        if message and repair_message(message):
            choice["finish_reason"] = "tool_calls"
            changed = True
    if not changed:
        return raw_bytes
    return json.dumps(payload).encode("utf-8")


def _sse_events(raw_text):
    for block in raw_text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        for line in block.split("\n"):
            if line.startswith("data:"):
                yield line[len("data:"):].strip()


def repair_stream_body(raw_bytes):
    """Buffer a full SSE stream, and if the assembled content is a leaked
    tool call, replay it as a single synthetic tool_calls delta sequence.
    Otherwise replay the original chunks unchanged."""
    text = raw_bytes.decode("utf-8", errors="replace")
    events = list(_sse_events(text))
    chunks = []
    content_parts = []
    template = None
    saw_tool_calls = False
    for ev in events:
        if ev == "[DONE]":
            continue
        try:
            chunk = json.loads(ev)
        except json.JSONDecodeError:
            continue
        chunks.append(chunk)
        if template is None:
            template = chunk
        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})
            if delta.get("tool_calls"):
                saw_tool_calls = True
            if delta.get("content"):
                content_parts.append(delta["content"])

    if saw_tool_calls or not chunks:
        return raw_bytes

    full_content = "".join(content_parts)
    calls = parse_leaked_tool_calls(full_content)
    if not calls:
        return raw_bytes

    base = json.loads(json.dumps(template))
    out_events = []

    role_chunk = json.loads(json.dumps(base))
    role_chunk["choices"] = [{"index": 0, "delta": {"role": "assistant", "content": None}, "finish_reason": None}]
    out_events.append(role_chunk)

    for i, call in enumerate(calls):
        tc_chunk = json.loads(json.dumps(base))
        tc_chunk["choices"] = [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": i,
                            "id": call["id"],
                            "type": "function",
                            "function": call["function"],
                        }
                    ]
                },
                "finish_reason": None,
            }
        ]
        out_events.append(tc_chunk)

    final_chunk = json.loads(json.dumps(base))
    final_chunk["choices"] = [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]
    out_events.append(final_chunk)

    body = "".join(f"data: {json.dumps(e)}\n\n" for e in out_events)
    body += "data: [DONE]\n\n"
    return body.encode("utf-8")


def _filter_headers(headers_items):
    hop_by_hop = {
        "connection",
        "keep-alive",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
    connection_header = next((v for k, v in headers_items if k.lower() == "connection"), "")
    if connection_header:
        for token in connection_header.split(","):
            token = token.strip().lower()
            if token:
                hop_by_hop.add(token)

    filtered = []
    for k, v in headers_items:
        k_lower = k.lower()
        if k_lower in hop_by_hop or k_lower.startswith("proxy-"):
            continue
        filtered.append((k, v))
    return filtered


class ProxyHandler(BaseHTTPRequestHandler):
    upstream = None  # set in main()
    timeout = 300.0  # set in main()

    def log_message(self, fmt, *args):
        sys.stderr.write("[proxy] " + (fmt % args) + "\n")

    def _proxy(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        is_stream = False
        if body:
            try:
                is_stream = bool(json.loads(body).get("stream"))
            except json.JSONDecodeError:
                pass

        url = self.upstream.rstrip("/") + self.path
        req = urllib.request.Request(url, data=body or None, method=self.command)

        filtered_req_headers = _filter_headers(list(self.headers.items()))
        for h, v in filtered_req_headers:
            h_lower = h.lower()
            if h_lower in ("host", "content-length", "accept-encoding"):
                continue
            req.add_header(h, v)
        req.add_header("Accept-Encoding", "identity")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                status = resp.status
                headers = [(h, v) for h, v in _filter_headers(resp.getheaders()) if h.lower() != "content-length"]
                raw = resp.read()
        except urllib.error.HTTPError as e:
            status = e.code
            headers = [(h, v) for h, v in _filter_headers(e.headers.items()) if h.lower() != "content-length"]
            raw = e.read()
        except urllib.error.URLError as e:
            self.send_response(502)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(f"upstream unreachable: {e}".encode())
            return

        if status == 200 and "/chat/completions" in self.path:
            raw = repair_stream_body(raw) if is_stream else repair_nonstream_body(raw)

        self.send_response(status)
        for h, v in headers:
            self.send_header(h, v)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        self._proxy()

    def do_GET(self):
        self._proxy()

    def do_PUT(self):
        self._proxy()

    def do_DELETE(self):
        self._proxy()


def main():
    upstream = os.environ.get("UPSTREAM_URL")
    if not upstream:
        sys.exit("UPSTREAM_URL is required, e.g. http://192.168.1.196:8090 (root, no /v1)")

    timeout_str = os.environ.get("UPSTREAM_TIMEOUT", "300")
    try:
        timeout = float(timeout_str)
    except ValueError:
        sys.stderr.write(f"[proxy] invalid UPSTREAM_TIMEOUT {timeout_str!r}, defaulting to 300\n")
        timeout = 300.0

    host = os.environ.get("LISTEN_HOST", "0.0.0.0")
    port = int(os.environ.get("LISTEN_PORT", "8091"))

    ProxyHandler.upstream = upstream
    ProxyHandler.timeout = timeout
    server = ThreadingHTTPServer((host, port), ProxyHandler)
    print(f"[proxy] listening on {host}:{port} -> {upstream} (timeout {timeout}s)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
