# AgentMem

Experimental infrastructure for **"Accuracy Is Not Enough: An Operational
Characterization of Agent Memory Systems."**

A reproducible benchmark harness comparing four agent-memory systems (Mem0,
Graphiti/Zep OSS, Letta, and S4 (selection pending)) on LongMemEval-S accuracy *and* operational
behaviour: latency, token cost, ingestion-to-retrievability lag, concurrency
degradation, scaling, and failure semantics.

The guiding constraint is **measurement validity**. Every number in the paper
must trace back to an immutable raw event.

## Status

| Milestone | Component | State |
|---|---|---|
| 1 | Instrumented LLM proxy (`proxy/`) | **done, validated** (see below) |
| 2 | Mem0 adapter (`adapters/`) | **done, validated** offline; real-provider run pending |
| 3 | LongMemEval-S runner (`bench/`) | **done, validated** on synthetic data; real dataset + provider run pending |
| 4 | Mem0 reproduction | **pre-registered and instrumented**; canonical campaign pending dataset hash, credentials, and budget. See `docs/milestone4_reproduction.md` |
| 5–7 | Graphiti/Zep OSS and Letta adapters; S4 selection + adapter | not started; S4 deliberately unselected (`docs/system4_selection.md`) |
| 8–12 | cross-system accuracy, load, scale, faults, analysis | open-loop/fault/validation/statistics infrastructure implemented; real campaigns pending |

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
adapters/       uniform async memory interface (Milestone 2)
  base.py       MemoryAdapter ABC, WriteResult/ReadResult/CanaryRecord/SettlementResult, event emission
  mem0.py       Mem0 (mem0ai OSS, in-process) adapter
  registry.py   config loading + adapter construction; configuration id = sha256(config bytes)
bench/          LongMemEval-S runner (Milestone 3)
  longmemeval.py dataset schema/loader, seeded selection, synthetic stand-in
  prompts.py    official LongMemEval reader + judge prompts, verbatim, with source hashes
  llm.py        proxy-routed chat client for harness-owned calls (answer/judge)
  reader.py / judge.py   fixed reader and judge
  runner.py     ingestion -> settlement -> retrieval -> reader -> judge; append-only run.jsonl
  run.py        CLI: python -m bench.run ...
  metadata.py   host/software metadata for RUN_START
analysis/       ingest.py (run + proxy JSONL -> DataFrames), metrics.py (accuracy/CI, settlement, latency, cost), statistics.py
configs/        committed system configurations (mem0.yaml, mem0-published-protocol.yaml) and benchmark configs (benchmarks/)
docs/           milestone4_reproduction.md (pre-registration), examples/
data/           gitignored datasets (scripts/fetch_longmemeval.py)
compose/        per-system persistence stacks (compose/mem0: dedicated Qdrant)
scripts/        demo_milestone1.py, demo_milestone2.py
tests/          adapter tests against live proxy + fake upstream servers
docs/examples/  committed example trace
results/raw/    gitignored raw JSONL traces
results/summaries/  committed compact summaries
```

## Setup

```bash
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[dev,mem0]"
```

## Artifact status and integrity

This checkout is **infrastructure complete only for the components marked above**;
it is not an experiment-campaign completion. No fake-provider trace is a paper
result. The repository contains no canonical numerical finding yet, so it also
contains no placeholder figure, table, or `paper_findings.md`. Graphiti (Zep OSS)
and Letta remain named targets, while S4 remains unselected pending the documented
primary-source review in `docs/system4_selection.md`.

Operational work uses the coordinated-omission-safe open-loop scheduler in
`load/`, externally controlled fault lifecycles in `faults/`, and offline,
fail-closed validation/statistics in `analysis/`. Dedicated visibility experiments
record lower and upper retrievability bounds; accuracy runs do not plant canaries.

## Reproduction commands

```bash
make test                  # offline unit + fake-upstream integration tests
make smoke                 # deterministic development slice; never paper evidence
make mem0-reproduction     # requires explicit paid gates and real dataset/provider
make accuracy              # fails closed until comparative configs are frozen
make load                  # prints safe invocation guidance; never auto-launches load
make scale
make faults
make analyze               # analysis code only; never reruns a system
make figures               # refuses missing validated canonical data
```

Real campaigns require `PROXY_REQUIRE_ATTRIBUTION=1`,
`PROXY_UPSTREAM_API_KEY`, `BENCH_ALLOW_PAID_RUN=1`, and `BENCH_MAX_USD`. Raw
append-only events belong under `results/raw/`; derived summaries, figures, and
tables belong under their corresponding `results/` directories. See
`docs/preregistration.md`, `docs/methodology.md`, and `docs/reproduction.md`.

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

## Milestone 2: the adapter layer and the Mem0 adapter

### Interface

`adapters/base.py` defines `MemoryAdapter` with the spec's `reset / write /
read / is_settled / config_fingerprint` plus the helpers the runner needs:
`search` (structured read), `plant_canary`, `wait_settled`, `reset_all`,
`health`, `close`. The base class owns timing and event emission
(`RESET_*`, `WRITE_SUBMIT`, `WRITE_ACK`, `READ_START/END`, `CANARY_SUBMIT/ACK`,
`SETTLEMENT_POLL`, `WRITE_SETTLED`, `SETTLEMENT_TIMEOUT`, `ERROR`); subclasses
implement only the system-specific `_impl` hooks, so no adapter can quietly
differ in what it measures. Context assembly is uniform: retrieved memory
texts, one per line, in the system's own order, nothing added or removed.

### Settlement protocol

1. `plant_canary(session)` writes "The verification code for this session is
   memharness-canary-XXXX." through the system's own write path and records
   submission and acknowledgement times plus whether the system reported
   storing it.
2. `is_settled(session)` issues the *normal read path* (`search`, tagged
   `operation=settle`) with the canary text as the query and returns True only
   when the token appears in the assembled context. Each poll is an event.
3. `wait_settled` polls until True or the configured timeout, then deletes the
   canary memory through the system's documented delete API so it cannot occupy
   a retrieval slot later. Timeouts are reported as `SETTLEMENT_TIMEOUT`, never
   swallowed.

Derived per canary: write acknowledgement latency, ingestion-to-retrievability
lag (ack to first retrievable), submission-to-retrievability latency, poll
count, timeout flag, cleanup outcome. Because visibility is sampled by polling,
the lag has a floor of one read latency and a resolution of one poll interval;
when `polls == 1` the true lag is only known to be at most the reported value.
Analysis must report it that way.

### Mem0 adapter (`adapters/mem0.py`)

Mem0 OSS (`mem0ai` 2.0.20, pinned) runs in-process via its documented
`AsyncMemory` API with a dedicated Qdrant server (`compose/mem0`) and its own
SQLite history DB. Model traffic reaches the proxy through Mem0's own
`openai_base_url` / `api_key` settings.

| Benchmark op | Mem0 call |
|---|---|
| `reset(s)` | `delete_all(user_id=s)`, then `get_all(filters={"user_id": s})` must be empty or the reset raises |
| `write(s, msgs)` | `add(msgs, user_id=s, infer=True)` |
| `read(s, q)` | `search(q, filters={"user_id": s}, top_k=20, threshold=0.1, rerank=False)` (library defaults) |
| canary | `add([...], user_id=s, infer=False)`; polled with the same `search`; removed with `delete(id)` |
| `reset_all()` | `AsyncMemory.reset()` on a throw-away instance plus removal of the `_entities` collection |

Attribution: three `AsyncMemory` instances (write / read / settle) share one
Qdrant client and one history DB, so they are one store, but each carries a
proxy API-key token with its own `operation` and `client_id`. Operation
attribution is therefore exact even under concurrency; the session is attached
with a proxy scope around each call. Mem0's PostHog telemetry is disabled
before import. Retries performed by the OpenAI SDK inside Mem0 appear as
separate `MODEL_CALL` events, which is correct: they are real provider calls.

The fingerprint records the mem0ai version, prompt hashes, all model and
retrieval parameters, Qdrant mode/version, and feature flags that change
retrieval behaviour: whether spaCy's `en_core_web_sm` (entity boosting) and
`fastembed` (BM25 hybrid search) are installed. Both are absent in this
sandbox; the paper deployment installs `mem0ai[nlp,extras]` so Mem0's
documented hybrid search is active, and the flags make the difference visible.

### Run it

```bash
docker compose -f compose/mem0/docker-compose.yml up -d     # Qdrant for paper runs
.venv/bin/python -m pytest tests -q                         # adapter suite (embedded Qdrant)
.venv/bin/python scripts/demo_milestone2.py --sessions 5    # repeated-session validation
.venv/bin/python scripts/demo_milestone2.py --upstream https://api.openai.com/v1 --qdrant server
```

`tests/test_mem0_adapter.py` (10 tests) runs the proxy and fake upstream as real
uvicorn servers and verifies: reset/write/read/settle over repeated sessions
with cross-session isolation; idempotent, verified reset; every model call
routed through the proxy with exact system/configuration/seed/run/session/
operation attribution and per-operation `client_id`; complete and ordered
adapter events with non-null lag fields; `is_settled` refusing to answer
without a canary; settlement timeouts surfacing as events; write failures
raised and visible as proxy error events; `reset_all` wiping and rebuilding;
fingerprint completeness; and the committed `configs/mem0.yaml` loading and
building a working adapter. The demo shows five sessions with 0 of 35 model
calls lacking full attribution.

### Design decisions that affect later benchmark validity

11. **Mem0's v3 pipeline is ADD-only.** mem0ai 2.x extracts memories with an
    additive prompt and deduplicates by hash; there are no UPDATE/DELETE
    events. Counts are still recorded per event type for other systems.
12. **The canary bypasses LLM extraction (`infer=False`).** With extraction
    on, the model may legitimately decide the canary is not memorable, which
    would make settlement inconclusive rather than false. With it off, the
    canary exercises the embedding + vector-store path, which is the only part
    of Mem0's write that can lag acknowledgement (extraction is synchronous
    and already inside the ack latency). `settlement.canary_infer` flips this.
13. **Observation dates.** mem0ai OSS rejects the `timestamp` parameter and
    grounds relative time expressions to the current date. LongMemEval session
    dates therefore have to be carried in message content, a preprocessing
    choice for Milestone 3 that must match what the published setup did.
14. **Session = LongMemEval question instance.** `session_id` maps to Mem0's
    `user_id`; all haystack sessions of one question are written under it,
    which is the retrieval scope the benchmark implies.
15. **Embedded vs server Qdrant.** Tests use embedded mode (no daemon in the
    sandbox; payload indexes are no-ops there). Paper runs use the compose
    server; the fingerprint records the mode and server version so the two
    are never confused.
16. **Not yet run against a real provider.** As with Milestone 1, the sandbox
    cannot reach OpenAI; the demo used the fake upstream's Mem0 extraction
    emulation. Run `scripts/demo_milestone2.py --upstream ... --qdrant server`
    once with a key before Milestone 3.

## Milestone 3: the LongMemEval-S runner

### Data path

```
LongMemEval-S json ─▶ bench.longmemeval (schema check, sha256, seeded order)
   ─▶ Runner: per instance  reset(qid) ─▶ write() per haystack session (dated) ─▶ wait_settled()
                            ─▶ search(question) ─▶ Reader (official facts prompt) ─▶ Judge (official prompt)
   ─▶ results/raw/runs/<run_id>/run.jsonl  +  manifest.json          (proxy log written separately by the proxy)
```

```bash
python -m proxy --port 8811 --upstream https://api.openai.com/v1 --log results/raw/proxy/campaign.jsonl
python scripts/fetch_longmemeval.py                  # data/longmemeval_s_cleaned.json + sha256
python -m bench.run --system mem0 --benchmark longmemeval_s --seed 42 --config configs/mem0.yaml
python -m bench.run ... --limit 20                   # first 20 of the seeded order (reproducible subset)
python scripts/demo_milestone3.py --limit 4          # offline vertical slice on a synthetic dataset
```

### What is fixed, and where it comes from

* **Reader**: the official `run_generation.py` template for *facts extracted
  from history chats* (the variant LongMemEval uses when retrieval returns
  facts rather than sessions), single user message, temperature 0,
  max_tokens 500, `Current Date` = the instance's raw `question_date`. The
  adapter's context (one memory per line) fills the `History Chats` slot.
* **Judge**: the official `evaluate_qa.py` prompts per question type plus the
  abstention prompt, ported verbatim; gpt-4o-2024-08-06, temperature 0,
  max_tokens 10, label = `"yes" in response.lower()`. `bench/prompts.py`
  records the upstream file hashes and fetch date; all prompt hashes go into
  RUN_START.
* **Ingestion**: one `write()` per haystack session in file order (the README
  states `_s` sessions are timestamp-sorted). Session dates reach the system
  as a leading system message (`ingest.date_mode`), because mem0ai OSS rejects
  the `timestamp` parameter. `has_answer` evidence labels are stripped and a
  test proves they never reach the adapter.
* **Settlement**: `wait_settled` once per instance after ingestion
  (`settlement.policy`), so accuracy is measured on retrievable state and the
  lag is on record for every instance.
* **Seed**: controls question ordering; `--limit N` is the first N of that
  order, so any subset is reproducible from (dataset sha256, seed, N). The
  ordered id list and its digest are in RUN_START. Haystack order is never
  shuffled: knowledge-update questions depend on it.

### Run record

`RUN_START` carries: run id, git commit + dirty flag, system config text and
its sha-derived `configuration_id`, benchmark config, dataset path/sha256/size,
selection (seed, limit, ids, digest), adapter fingerprint, reader and judge
settings with prompt hashes, proxy instance id + log path + settings, host
(CPU, RAM, GPU, OS, kernel, Docker, Python, package versions), and whether any
config overrides were applied (`overrides_present`; must be false for paper
runs). Per instance: `INSTANCE_START`, the adapter's `RESET_*`, `WRITE_SUBMIT/ACK`,
`CANARY_*`, `SETTLEMENT_POLL`, `WRITE_SETTLED`, `READ_START/END`, then
`CONTEXT` (the exact retrieved text), `ANSWER_START/END` (answer text, prompt
hash, proxy request id), `JUDGE_START/END` (raw verdict, label, gold answer),
`INSTANCE_END` (per-stage timings). Failures are `ERROR` + `INSTANCE_END` with
`error` set and the run continues (`on_error`). `RUN_END` holds a convenience
tally only; analysis recomputes from events.

`tests/test_longmemeval_runner.py` (7 tests) proves: schema validation,
seeded ordering and prefix-limit reproducibility, label stripping and date
modes, verbatim prompt ports and the official label rule, one complete
instance end to end with every stage event present and ordered, the proxy log
showing `write/read/settle/answer/judge` all attributed to the run and
question, and a three-instance run that records a mid-run reader failure and
continues.

### Design decisions that affect later benchmark validity

17. **Official prompts only.** Reader and judge prompts are the LongMemEval
    authors' text, not paraphrases. If a system's published number used a
    different reader prompt, that is a documented discrepancy for Milestone 4,
    not a reason to change the harness prompt after the fact.
18. **Context format is the one uniform choice.** Memory systems return
    facts, not sessions; the harness inserts them one per line into the
    official facts template. No timestamps or scores are added by the harness.
19. **Empty retrieval is still answered.** If the system returns nothing, the
    reader is called with an empty `History Chats` section and `CONTEXT`
    records `context_empty: true`, so abstention behaviour is measured rather
    than short-circuited.
20. **Question = session.** One instance's haystack is written under
    `session_id = question_id` and reset before ingestion, so instances never
    share memory state.
21. **The convenience tally in RUN_END is not a result.** Accuracy and its
    confidence intervals come from the analysis layer over `JUDGE_END` events.
22. **Synthetic data is fenced off.** `--allow-synthetic` is required, the
    manifest records `synthetic: true`, and the fake upstream's reader/judge
    emulation is a path test, not an evaluation. The real dataset hash goes
    into `configs/benchmarks/longmemeval_s.yaml` after download and is
    enforced at run start.
23. **Reproduction target is not yet set.** `published_reference.mem0` in the
    benchmark config is null. Milestone 4 must fill it from a citable source
    before any comparative run is inspected.

## Milestone 4: Mem0 reproduction (pre-registered)

Full text: `docs/milestone4_reproduction.md`. Summary:

**Published reference.** Mem0's README and `mem0ai/memory-benchmarks` report
LongMemEval **94.4% (472/500)** at a top-200 retrieval budget on the *managed
platform*, per type from 88.0 (multi-session) to 98.6 (single-session-user),
with a vendor-stated ±1 point judge inconsistency and the explicit statement
that OSS users "should expect directionally similar gains but not identical
numbers". An OSS table exists only with non-default models (GPT-5 extraction
and judge, Qwen embedder: 91.0%). No Mem0 number exists under the official
LongMemEval reader/judge protocol.

**Two arms, both committed before any run:**

| | Arm A: official protocol | Arm B: Mem0's published protocol |
|---|---|---|
| configs | `configs/mem0.yaml` + `configs/benchmarks/longmemeval_s.yaml` | `configs/mem0-published-protocol.yaml` + `configs/benchmarks/longmemeval_s_mem0protocol.yaml` |
| ingestion | per session, date as system message | per user/assistant pair, no date (their OSS server forwards none) |
| retrieval | top_k 20, threshold 0.1 (library defaults) | top_k 200 |
| reader / judge | official LongMemEval templates, 500 / 10 tokens | Mem0's `ANSWER_GENERATION_PROMPT` and unified `JUDGE_PROMPT` verbatim (Apache-2.0, hashed), their post-processing and verdict parser |

**Comparison rule (fixed):** reproduced iff |accuracy − 0.944| ≤ 0.03 or 0.944
lies inside the run's Wilson 95% interval; errored instances reported
separately. Arm B once over 500 instances; Arm A at three seeds.

**Discrepancies already established from source** (each is a row in the
automatic checklist): platform-only optimizations; their OSS server pins a git
branch that no longer exists; that server calls `search(limit=…, user_id=…)`,
which mem0ai 2.0.20 rejects/ignores, and never forwards session dates; Mem0's
answer prompt contains LongMemEval-item-specific rules and its judge prompt
instructs leniency; 10-token vs free-form judge output; top-200 vs default 20;
pair vs session ingestion; spaCy/fastembed prerequisites for hybrid retrieval.

**Tooling.** `analysis/` derives accuracy with Wilson intervals (overall, per
type, abstention), settlement lag, write/read latency percentiles, and
token/call cost by operation, from `run.jsonl` plus the proxy log joined on
`run_id`. `scripts/reproduce_mem0.py` turns one or more run directories into
`results/summaries/mem0-reproduction-<stamp>.{md,json}` with the verdict and
the discrepancy checklist. `scripts/demo_milestone4.py` runs both arms through
`bench.run` and the report; offline it uses the synthetic dataset and the
verdict is `not_applicable_synthetic_dataset` by construction.

```bash
python scripts/demo_milestone4.py --limit 5                                  # apparatus check (offline)
python scripts/reproduce_mem0.py --run results/raw/runs/<B> --run results/raw/runs/<A> --proxy-log results/raw/proxy/m4.jsonl
```

**Execution status: not executed.** This environment cannot reach the dataset
host, the model provider, or a Docker daemon. The protocol in the document is
the exact command sequence to run once those are available; nothing in the
configs may change after the first real verdict is seen.

### Design decisions that affect later benchmark validity

24. **Reproduction target is Mem0's own protocol, not the official one.**
    Because no official-protocol number exists, Arm B reproduces what Mem0
    published, and Arm A is a new measurement. The A–B gap is itself a
    result: the effect of prompt engineering and retrieval depth on the same
    system.
25. **Arm B mirrors their OSS code path, not their platform.** Where their
    runner and their server disagree (dates, `limit`), Arm B follows what the
    OSS server actually does, and the checklist says so.
26. **Vendor doc inconsistency on the reader/judge model** (gpt-4o in the
    README, gpt-5 in the runner default) is resolved in advance to gpt-4o; any
    gpt-5 run is a labelled follow-up.
27. **Analysis never reads system-reported counts.** Cost comes from proxy
    `MODEL_CALL` rows filtered by `run_id`; a foreign run's rows in the same
    proxy log are ignored (tested).

## Engineering rules (from the pre-registration)

No fabricated results; no silent configuration substitution; no tuning after
seeing comparative results; no system-reported token counts for cross-system
comparison; open-loop load generation only; settlement measured by observable
retrievability, not API acknowledgement; reader latency separated from memory
retrieval latency; hosted vs self-hosted labelled explicitly; timeouts and
extremes never discarded; p99 reported with sample sizes; systems run
sequentially; system internals not patched unless documented.
