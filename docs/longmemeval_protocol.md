# LongMemEval-S protocol record

## Frozen sources

The runner consumes the official machine-readable LongMemEval schema. The
configured source is `xiaowu0162/longmemeval-cleaned`, revision
`longmemeval-cleaned-2025-09`; the downloaded bytes, size, instance count, and
SHA-256 are recorded in each manifest. Canonical execution is blocked until an
exact expected SHA-256 is filled into the benchmark configuration.

Prompts in `bench/prompts.py` are ports of the upstream `run_generation.py` and
`evaluate_qa.py` templates; their individual SHA-256 values are recorded. The
reader and judge are frozen to `gpt-4o-2024-08-06`. One question instance maps
to one memory namespace. Sessions are written chronologically in dataset order.
Evidence labels are stripped.

## Dates and ambiguity

Mem0 2.0.20 does not support the historical timestamp examples. The current
deterministic transformation prepends one system message, `This conversation
took place on <date>.`, to each session. This is a documented experimental
resolution, not claimed to be an exact recovered Mem0-paper transformation.
Its configuration and implementation source are fingerprinted. Full results
must not run until primary upstream sources are reachable and this unresolved
protocol point is re-audited.

Accuracy runs use no canaries (`settlement.policy: none`); canaries are reserved
for visibility experiments. Retrieval and fixed reader generation are separate.
Reader calls are `operation=answer`; judge calls are `operation=judge`, and the
raw answer, verdict, gold answer, prompt hash, model, and request ID are events.
Errors/timeouts count as incorrect and remain in the event stream.
