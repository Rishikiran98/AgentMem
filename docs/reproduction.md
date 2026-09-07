# Reproduction

Use Python 3.11, install `.[dev,mem0]`, and run `make test`. Download and hash the
official dataset with `python scripts/fetch_longmemeval.py`; update the expected
hash in a reviewed freeze commit. Never put credentials in the repository.

A real one-instance gate requires a proxy started with
`PROXY_REQUIRE_ATTRIBUTION=1` and a provider key, then:

```bash
BENCH_ALLOW_PAID_RUN=1 BENCH_MAX_USD=2 python -m bench.run \
  --system mem0 --benchmark longmemeval_s --seed 11 \
  --config configs/mem0.yaml --limit 1
```

After validating attribution, use `make mem0-reproduction`. Other campaign
commands deliberately stop until their pinned configs exist. `make analyze` and
`make figures` read existing raw logs only and never invoke a memory system.

External resume steps: restore access to GitHub/PyPI, install the pinned extras,
run the primary-source audit in `docs/system4_selection.md`, and provide
`PROXY_UPSTREAM_API_KEY`. Large campaigns additionally require an approved
`BENCH_MAX_USD` and suitable Docker resources.
