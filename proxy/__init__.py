"""Instrumented OpenAI-compatible LLM/embedding proxy.

The proxy is the single source of truth for every model-related measurement in
the benchmark (token consumption, model-call counts, call latency, cost
breakdowns).  All memory systems under test are configured to reach their
model provider *only* through this proxy.

Modules
-------
settings      environment-driven configuration
tags          attribution tags: header / API-key / path-token / scope-registry
logging       append-only JSONL event writer
tokenization  optional local token estimates (tiktoken)
app           FastAPI application (endpoints + upstream forwarding)
client        harness-side helper for tagging requests and managing scopes
fake_upstream deterministic mock provider used by tests and demos
"""

PROXY_VERSION = "0.1.0"
EVENT_SCHEMA_VERSION = 1
