# Reproduction

Use Python 3.11, install `.[dev,mem0]`, and run the complete suite before any
paid experiment:

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e '.[dev,mem0]'
pytest -q
```

If `python3.11` is installed by pyenv but is not selected in the current shell,
pass its interpreter path to uv (for example,
`uv venv .venv --python "$(pyenv root)/versions/3.11.15/bin/python"`). Do not
relax dependency pins or use a different interpreter merely to work around an
unavailable package index.

Download and hash the official dataset with
`python scripts/fetch_longmemeval.py`; update the expected hash in a reviewed,
dataset-only freeze commit. The download, upstream revision verification, and
schema/instance-count check must all succeed before a run is eligible for
canonical analysis. Never put credentials in the repository.

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

External resume steps, in order:

1. Restore access to GitHub, PyPI, and Hugging Face, then push and verify the
   `work` branch with `git push -u origin work` and
   `git ls-remote origin work`.
2. Create the Python 3.11 environment above and install the pinned extras.
3. Fetch and verify LongMemEval-S, freeze its SHA-256, and rerun `pytest -q`.
4. Provide `PROXY_UPSTREAM_API_KEY` without printing it, and provide a working
   Docker daemon for Qdrant. Start persistence with
   `docker compose -f compose/mem0/docker-compose.yml up -d`.
5. Run and canonically validate the one-instance Mem0 gate before implementing
   or measuring another system.

Large campaigns additionally require a user-approved `BENCH_MAX_USD` that is
sufficient for the measured campaign estimate. Never increase that ceiling
automatically.
