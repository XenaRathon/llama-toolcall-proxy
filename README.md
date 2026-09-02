# llama-toolcall-proxy

A tiny, (mostly) zero-dependency reverse proxy that patches a specific integration
bug in early-stage llama.cpp forks: a model correctly decides on a tool
call (right name, right arguments) but the inference server's
grammar-constrained parser fails to structure it, leaking it instead as
literal text in the response:

```
<tool_call>
<function=get_events>
<parameter=range>
week
</parameter>
</function>
</tool_call>
```

This proxy sits between your OpenAI-compatible client (Open WebUI, a
harness, anything) and the real inference server. It detects that exact
leak shape and rewrites it into a proper OpenAI `tool_calls` response
before it reaches the caller. Everything else passes through byte-for-byte
unchanged.

This fork also bakes in an **opinionated auto-escalation feature** tied to
one person's own agent setup — see [Auto-escalation](#auto-escalation-opinionated-off-label-feature)
below before you deploy it for anything else.

<details>
<summary>Why this exists instead of just fixing the inference server</summary>

Some brand-new model architectures ship ahead of mainline llama.cpp
support, via the model author's own fork. Those forks are new code and
their tool-call grammar/parser hasn't been hardened against real-world
tool schemas yet — it can work in a 1-tool smoke test and fall over at
40+ tools. This proxy is a stopgap for that specific gap, not a general
tool-calling shim: it only rewrites when the leaked text is unambiguously
a well-formed `<tool_call>` block (see design notes below), so it won't
touch a model that's genuinely just talking about tool-call syntax in
prose.

</details>

## Quick start

```bash
UPSTREAM_URL=http://192.168.1.196:8090 python3 proxy.py
```

or with the included `Dockerfile`:

```bash
docker build -t llama-toolcall-proxy .
docker run -p 8091:8091 -e UPSTREAM_URL=http://192.168.1.196:8090 llama-toolcall-proxy
```

Then point your client's `base_url` at this proxy instead of the real
server — e.g. swap `http://<server>:8090/v1` for
`http://<this-proxy>:8091/v1` — and nothing else changes. Every request
path, including `/v1/...`, is forwarded onto `UPSTREAM_URL` unchanged.

| Env var        | Required | Default   | Meaning                                                                                               |
| -------------- | -------- | --------- | ----------------------------------------------------------------------------------------------------- |
| `UPSTREAM_URL` | yes      | —         | Root URL of the real inference server, e.g. `http://192.168.1.196:8090` (no `/v1` suffix — see above) |
| `LISTEN_HOST`  | no       | `0.0.0.0` | Interface the proxy binds to                                                                          |
| `LISTEN_PORT`  | no       | `8091`    | Port the proxy binds to                                                                               |

For a walkthrough of a full deployment, see the [wiki](https://github.com/XenaRathon/llama-toolcall-proxy/wiki).

<details>
<summary>Design notes: tool-call leak repair</summary>

- Only rewrites when the message ENDS with one or more
  `<tool_call>...</tool_call>` blocks and nothing but whitespace follows
  the last one. An arbitrary leading sentence before the first block is
  allowed (e.g. `"Let me check that for you.\n\n<tool_call>..."`) — that's
  a common, natural leak shape. A model genuinely discussing `<tool_call>`
  syntax mid-prose is still left alone, since that case has real content
  trailing the example rather than ending right at `</tool_call>`.
- Also repairs two malformed leak shapes seen in production alongside the
  well-formed one: a missing `<function=NAME>` opening tag (just
  `<tool_call> NAME ...`), and missing `</parameter>`/`</function>` closing
  tags entirely — the parameter parser stops at the next
  parameter/function/tool_call boundary instead of silently losing the
  value.
- Handles both streaming and non-streaming `/v1/chat/completions`.
  Streaming responses are relayed incrementally, not buffered whole:
  headers and ordinary content stream to the client live. The proxy keeps
  watching the running text for a `<tool_call>` onset for the entire
  message (not just the first chunk), since the leading sentence above
  usually arrives as its own SSE chunk before the tag does — only text
  that's still an unresolved prefix of the tag, or the body of a block
  whose closing tag hasn't arrived yet, is held back. One consequence of
  streaming live: on the non-streaming path a leading sentence before a
  confirmed leak is discarded (`content` becomes `null`, matching a
  well-formed tool-call-only message); on the streaming path it's already
  been sent to the client by the time the leak resolves, so it's kept as
  real preceding content instead of a phantom retraction.
- Multiple `<tool_call>` blocks in one response become multiple entries
  in `tool_calls`, matching how a well-behaved server would report
  parallel tool calls. A model can chain several blocks back to back, so
  seeing one complete block close doesn't resolve the repair early — the
  proxy keeps capturing until the message actually ends (`finish_reason`
  or EOF) in case another block is chained on next.
- Zero _runtime_ dependencies — Python stdlib only (`http.server`,
  `urllib`), matching the throwaway nature of the problem it's patching
  around.
- Not model-specific. Any inference stack that leaks this exact
  `<tool_call><function=...><parameter=...>` shape benefits, not just
  Nanbeige's fork.

</details>

## License

GPL-3.0-or-later — see [LICENSE](LICENSE).

## Credit

Built with [Claude Code](https://claude.com/claude-code).
