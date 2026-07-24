# llama-toolcall-proxy

A tiny, zero-dependency reverse proxy that patches a specific integration
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

<details>
<summary>Design notes</summary>

- Only rewrites when the assistant's entire message content is composed
  of one or more `<tool_call>...</tool_call>` blocks — a stray tag
  mentioned mid-prose is left alone.
- Handles both streaming and non-streaming `/v1/chat/completions`.
  Streaming responses are buffered until the model finishes, since you
  can't know it's a leaked tool call from the first few tokens — this
  trades away token-by-token typewriter rendering for a converted
  response, but only for the final answer chunk; everything else streams
  live.
- Multiple `<tool_call>` blocks in one response become multiple entries
  in `tool_calls`, matching how a well-behaved server would report
  parallel tool calls.
- Zero dependencies — Python stdlib only (`http.server`, `urllib`),
  matching the throwaway-nature of the problem it's patching around.
- Not model-specific. Any inference stack that leaks this exact
  `<tool_call><function=...><parameter=...>` shape benefits, not just
  Nanbeige's fork.

</details>

## Setup and usage

See the [wiki](../../wiki).

## License

GPL-3.0-or-later — see [LICENSE](LICENSE).

## Credit

Built with [Claude Code](https://claude.com/claude-code).
