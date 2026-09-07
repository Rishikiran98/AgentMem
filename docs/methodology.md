# Methodology and architecture notes

The proxy event stream is the only cross-system ruler. Provider usage, not
system counters, supplies token measures. Execution only appends raw JSONL;
analysis reads it offline. Systems run sequentially with isolated persistence.

Mem0 2.0.20 runs in-process with dedicated Qdrant and SQLite history. Its write
path performs ADD-oriented fact extraction when inference is enabled; its read
path is Qdrant search. Relevant LLM and embedding clients receive proxy base
URLs and attribution tokens. Deletion is namespace-scoped. Settlement uses an
exact UUID canary returned by normal retrieval and is interval-censored.

Graphiti/Zep OSS and Letta architecture notes remain blocked until exact pinned
upstream versions can be installed and source-inspected. No architectural claim
or adapter is supplied from stale tutorials. S4 is unselected.
