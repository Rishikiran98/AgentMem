# Experimental preregistration (draft; not yet frozen)

No canonical comparative campaign has run. A literal freeze commit/tag will be
created only after Graphiti, Letta, and the S4 decision are verified. Development
fake-provider traces are never paper evidence.

* Systems: Mem0 OSS 2.0.20; Graphiti/Zep OSS and Letta versions pending upstream verification; S4 pending selection.
* Dataset: LongMemEval-S cleaned, 500 instances; exact SHA-256 **pending download**.
* Seeds: 11, 23, 37, 53, 71 (five seeds for each final cell).
* Reader/judge: `gpt-4o-2024-08-06`; prompt hashes are computed from `bench/prompts.py`.
* Mem0: `configs/mem0.yaml`; OpenAI `text-embedding-3-small`, 1536 dimensions; dense top-20, threshold 0.1, no rerank, hybrid, entity boost, or graph.
* Synchronization: no canary in accuracy runs. Dedicated visibility trials preserve polling lower/upper bounds.
* Timeout policy: API 180 s; settlement 30 s; reader/judge 180 s; provider retries 0 at the proxy; system retries 0; recovery 300 s. A timeout remains in latency data and scores incorrect.
* Load: open-loop seeded Poisson arrivals; warm-up excluded, measurement timeouts included; in-flight caps 1/10/50/100 without resetting scheduled timestamps.
* Scale: 10/100/1000 sessions. Costs retain calls and raw input/output tokens split into write/read/reader/judge; judge is excluded from operating cost.
* Statistics: median, IQR, p95, p99 and N; cluster-aware seeded bootstrap; Wilson 95% accuracy intervals; paired McNemar with Holm correction where applicable.
* Canonical validity: clean freeze commit, strict proxy attribution, exact dataset/config fingerprints, one system active, real upstream, complete sequence, and five seeds.
* Paid gate: `BENCH_ALLOW_PAID_RUN=1` and `BENCH_MAX_USD`; abort before exceeding the estimated ceiling.
