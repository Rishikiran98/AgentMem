"""Analysis layer: consumes raw JSONL (run logs + proxy logs) only.  Never runs a memory system.

Trace-level metrics (accuracy with Wilson intervals, settlement, latency, cost) take a
RunTrace from analysis.ingest; the list-level helpers (accuracy over outcomes, latency
over samples, cluster bootstrap, McNemar) serve the operational experiments.  Validation
(analysis.validation) is fail-closed for canonical runs.
"""
