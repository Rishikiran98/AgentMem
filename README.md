# AgentMem

Experimental infrastructure for **"Accuracy Is Not Enough: An Operational
Characterization of Agent Memory Systems."**

A reproducible benchmark harness comparing four agent-memory systems (Mem0,
Zep/Graphiti, Letta, HIEROMEM) on LongMemEval-S accuracy *and* operational
behaviour: latency, token cost, ingestion-to-retrievability lag, concurrency
degradation, scaling, and failure semantics.

The guiding constraint is **measurement validity**. Every number in the paper
must trace back to an immutable raw event.

## Status

| Milestone | Component | State |
|---|---|---|
| 1 | Instrumented LLM proxy (`proxy/`) | **done, validated** (see below) |
| 2 | Mem0 adapter | not started |
| 3 | LongMemEval-S runner | not started |
| 4 | Mem0 reproduction | not started |
| 5–7 | Zep, Letta, HIEROMEM adapters | not started |
| 8–12 | cross-system accuracy, load driver, scale, faults, analysis | not started |

## Layout

```
proxy/          instrumented OpenAI-compatible proxy (Milestone 1)
  app.py        FastAPI app: /v1/chat/completions, /v1/embeddings, /v1/models, control plane
  tags.py       attribution tags: headers, API-key/path tokens, scope registry
  logging.py    append-only JSONL writer
  tokenization.py  optional local token estimates (tiktoken)
  client.py     harness-side helper (tagging + scopes)
  fake_upstream.py deterministic mock provider for tests/demos
  tests/        pytest suite
scripts/        demo_milestone1.py
docs/examples/  committed example trace
configs/        (system configs, later milestones)
results/raw/    gitignored raw JSONL traces
results/summaries/  committed compact summaries
```

## Setup

```bash
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Milestone 1: the instrumented proxy

### Why a custom proxy instead of LiteLLM

LiteLLM's proxy applies its own model aliasing, retries, fallbacks and
callback-based (asynchronous, batched) logging, and it recounts tokens with its
own tokenizer in several paths. Each of those is a hidden transformation between
the system under test and the provider. The harness proxy is ~600 lines of
boring FastAPI + httpx that forwards bytes unchanged, records the provider's
own `usage` block verbatim, and appends one event per request synchronously.

### Architecture

```
memory system / harness                          proxy (single process)                 provider
────────────────────────                         ──────────────────────                 ────────
OpenAI SDK / httpx  ── POST /v1/chat/completions ─▶ resolve attribution ──▶ httpx ──▶ /v1/chat/completions
  base_url  = http://proxy/v1                       (header > scope > token)             (bytes unchanged*)
  api_key   = sk-bench-<token>      ◀── response ── absorb usage / timing  ◀───────────  response
  headers   = X-Bench-Operation...                 append MODEL_CALL event (after
                                                    response is handed to client)
harness ── POST /_bench/scope/enter {client_id, tags} ─▶ scope registry
        ── POST /_bench/scope/exit                      (SCOPE_ENTER/EXIT events)
                                                          │
                                                          ▼
                                       results/raw/proxy/<run>.jsonl   (O_APPEND, one line per event)
```

`*` The only request modification the proxy ever makes is adding
`stream_options.include_usage=true` to *streaming* chat requests so the provider
reports exact usage; the extra usage-only chunk is stripped from the client
stream unless the client asked for it. The modification is recorded in the
event (`request_modifications`).

**Attribution.** Every event carries `system, configuration, seed, session_id,
operation, run_id, client_id`. Three transports exist because the systems under
test differ in what they let us set:

| Transport | Granularity | Used by |
|---|---|---|
| `X-Bench-*` headers | per request | harness-owned calls (reader, judge, canaries); adapters whose client supports default headers |
| static token in API key (`sk-bench-<b64json>`) or URL (`/b/<token>/v1/...`) | per client instance | every system: all accept an API key and a base URL |
| scope registry (`/_bench/scope/enter|exit`) | per harness operation, per `client_id` | systems that call models from their own server process (Zep, Letta) |

Precedence is header > scope > token; the event records the source of every
field (`attribution_sources`). If two scopes are active for one client and
disagree, the disagreeing fields are logged as `ambiguous` and set to null
rather than guessed. Requests with no attribution at all are logged with
`unattributed: true`; `PROXY_REQUIRE_ATTRIBUTION=1` rejects them (use during
paper campaigns).

Operations: `write | consolidate | read | embed | answer | judge` from the spec,
plus `settle` (canary writes/polls so settlement probing is separable from
benchmark cost), `meta` (`GET /v1/models`) and `warmup`. The endpoint kind
(`chat`, `embeddings`, `models`) is a separate field, so an embedding call
issued during a memory write is `operation=write, endpoint=embeddings`.

**Token accounting.** `prompt_tokens / completion_tokens / total_tokens` are the
provider's `usage` block, verbatim (`usage_source: upstream`). Local tiktoken
estimates (`*_local`) are recorded only as a cross-check and are `null` when
encodings are unavailable; nothing depends on them. Failed requests have
`usage_source: none`. Embeddings additionally record `embedding_inputs`,
`embedding_vectors`, `embedding_dimensions`.

**Timing.** `duration_ms` is proxy ingress to response-ready;
`upstream_duration_ms` is the provider round trip; `time_to_first_byte_ms` is
recorded for streams. Log I/O and local token estimation run *after* the
response is handed to the client (Starlette background task), so they never
inflate observed latency; the test suite asserts proxy overhead stays below 50
ms and the demo shows it at well under 1 ms.

**Append-only log.** The file is opened with `O_APPEND`; the writer has no
truncate/seek API; each event is one `os.write` under a lock; events carry a
contiguous per-process `seq` and a `proxy_instance_id`. The first event of every
file is `PROXY_START` with redacted settings, library versions, git commit,
host, and tokenizer availability; `PROXY_STOP` is written on graceful shutdown.

### Run it

```bash
# terminal 1: proxy in front of a real provider
export PROXY_UPSTREAM_API_KEY=sk-...
python -m proxy --port 8811 --upstream https://api.openai.com/v1 \
                --log results/raw/proxy/run-001.jsonl

# or, offline: deterministic fake provider + proxy
python -m proxy.fake_upstream --port 8899
python -m proxy --port 8811 --upstream http://127.0.0.1:8899/v1 --log results/raw/proxy/dev.jsonl

# point a client at it
python - <<'PY'
from openai import OpenAI
from proxy.client import ProxyClient
pc = ProxyClient("http://127.0.0.1:8811")
c = OpenAI(base_url=pc.base_url(), api_key=pc.api_key(system="mem0", configuration="cfg", seed=42, client_id="mem0"))
print(c.chat.completions.create(model="fake-model", messages=[{"role":"user","content":"hi"}],
      extra_headers=pc.headers(operation="write", session_id="s1")).usage)
PY
```

All settings are environment variables (`PROXY_*`, see `proxy/settings.py`);
CLI flags override them. Run a single worker: the scope registry is in-process.

### Tests

```bash
.venv/bin/python -m pytest proxy/tests -q
```

The suite (41 tests) proves: token counts are recorded from upstream usage
(non-stream, stream with and without client-requested usage, embeddings, base64
embeddings); tags propagate via headers, API-key token, path token and scopes
with the documented precedence; 300 concurrent mixed requests and concurrent
scoped clients keep isolated metadata; upstream 4xx/5xx, connect errors,
timeouts, protocol errors, invalid bodies and unsupported endpoints all produce
exactly one `status=error` event; the log only grows, `seq` is contiguous, and
threaded appends never interleave. One test is skipped offline (tiktoken
encodings cannot be downloaded).

### End-to-end demonstration

```bash
.venv/bin/python scripts/demo_milestone1.py                     # fake upstream
.venv/bin/python scripts/demo_milestone1.py --upstream https://api.openai.com/v1   # real provider
```

The script starts the proxy (and the fake upstream) as real uvicorn processes,
drives them with the official `openai` SDK and `httpx`, then verifies the trace:
a chat completion, an embedding, a streaming completion, a scope-attributed
call, 50 concurrent distinctly-tagged calls, and an injected failure; all 20
checks pass. The resulting trace lines are committed at
`docs/examples/proxy-trace-milestone1.jsonl`. Example `MODEL_CALL` event:

```json
{"seq":2,"proxy_instance_id":"…","logged_at":"2026-09-07T09:34:10.2Z","event_type":"MODEL_CALL",
 "schema_version":1,"request_id":"dc2e…","timestamp_start":"…","timestamp_end":"…",
 "duration_ms":27.98,"upstream_duration_ms":27.376,"time_to_first_byte_ms":27.37,
 "endpoint":"chat","method":"POST","path":"/v1/chat/completions",
 "system":"mem0","configuration":"demo-cfg","seed":42,"session_id":"sess-1","operation":"write",
 "run_id":"demo-m1-…","client_id":"mem0",
 "attribution_sources":{"system":"api_key_token","configuration":"api_key_token","seed":"api_key_token",
                        "run_id":"api_key_token","client_id":"api_key_token","session_id":"header","operation":"header"},
 "scope_ids":[],"scope_state":"none","ambiguous_fields":[],"unattributed":false,
 "model":"fake-model","upstream_model":"fake-model","stream":false,
 "prompt_tokens":7,"completion_tokens":5,"total_tokens":12,"usage_details":null,"usage_source":"upstream",
 "prompt_tokens_local":null,"completion_tokens_local":null,
 "embedding_inputs":null,"embedding_vectors":null,"embedding_dimensions":null,
 "status":"success","http_status":200,"upstream_http_status":200,"error_type":null,"error_message":null,
 "finish_reason":"stop","request_bytes":131,"response_bytes":272,"request_sha256":"434f…",
 "request_params":{"temperature":0,"max_tokens":5,"messages":1},"request_modifications":[],
 "upstream_request_id":"up-0668ed84","client_user_agent":"AsyncOpenAI/Python 3.8.0"}
```

### Design decisions that affect later benchmark validity

1. **Provider usage is authoritative; local counts are advisory.** Cross-system
   comparisons must use `prompt_tokens/completion_tokens` with
   `usage_source == "upstream"`. Never sum `*_local`.
2. **Streaming usage injection** is the one request modification. It is
   recorded per event; analysis can filter on `request_modifications`.
3. **Per-operation attribution for server-hosted systems relies on scopes.**
   When a system's server (Zep, Letta) issues model calls, only the scope
   registry can say which harness operation caused them. Under concurrent load,
   overlapping scopes for one `client_id` become `ambiguous` and the affected
   fields are null. Per-operation cost is therefore exact for sequential
   accuracy runs and for load workloads partitioned by client; for mixed
   concurrent workloads on a single server client, only aggregate window cost is
   exact. Adapters should use distinct `client_id`s per operation type where the
   system allows separate client instances (Milestone 2+).
4. **The proxy is a shared confounder under load.** It is one asyncio process.
   `duration_ms − upstream_duration_ms` is the proxy's own overhead and must be
   reported for load experiments; if it grows with concurrency the load results
   are proxy-limited, not system-limited.
5. **Only the OpenAI-compatible surface is proxied.** `/v1/chat/completions`,
   `/v1/embeddings`, `GET /v1/models` (Letta probes it). Anything else returns
   404 and is logged, so a system silently using another endpoint (e.g. legacy
   `/v1/completions`, Anthropic-native APIs) is detectable rather than
   unmeasured.
6. **Attribution is validated at ingress.** Unknown `operation` values, unknown
   `system` values (`PROXY_ALLOWED_SYSTEMS`) and malformed tokens are rejected
   with 400 and logged, so harness bugs surface as errors rather than as
   silently mis-attributed cost.
7. **Failed calls still count.** Provider errors, timeouts and unreachable
   upstreams are events with `status=error`; they carry no usage but are part of
   model-call counts and error rates. They must not be dropped in analysis.
8. **Log durability default is `fsync=off`.** Events are written immediately
   with `os.write` but the kernel may buffer; enable `PROXY_LOG_FSYNC=1` for
   fault experiments where the *host* might crash (the proxy is not the process
   killed in Fault C).
9. **Restart semantics.** `seq` restarts at 1 when the proxy restarts;
   `proxy_instance_id` changes. Analysis must key on `(proxy_instance_id, seq)`.
10. **Not verified against a real provider in this environment.** The sandbox
    cannot reach `api.openai.com` or tiktoken's encoding host, so the
    demonstration ran against the deterministic fake upstream. The same script
    with `--upstream https://api.openai.com/v1` must be run once with a key
    before Milestone 2 to confirm real `usage` blocks (including
    `prompt_tokens_details`) and base64 embeddings behave as tested.

## Engineering rules (from the pre-registration)

No fabricated results; no silent configuration substitution; no tuning after
seeing comparative results; no system-reported token counts for cross-system
comparison; open-loop load generation only; settlement measured by observable
retrievability, not API acknowledgement; reader latency separated from memory
retrieval latency; hosted vs self-hosted labelled explicitly; timeouts and
extremes never discarded; p99 reported with sample sizes; systems run
sequentially; system internals not patched unless documented.
