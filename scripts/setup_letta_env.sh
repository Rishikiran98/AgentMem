#!/usr/bin/env bash
# Build an isolated Python environment for the retired Letta V1 server (0.16.8)
# from Letta's own lockfile, for running the server outside Docker (tests, CI).
#
# The harness venv cannot host the Letta server: letta 0.16.8 needs mcp 1.12.4 /
# fastmcp 2.12.5 / openai 2.25.0, which conflict with the harness's own
# dependencies.  Letta is a separate process anyway, so it gets its own env,
# exactly as pinned by github.com/letta-ai/letta tag 0.16.8 (uv.lock).
#
#   scripts/setup_letta_env.sh            # creates ./.venv-letta
#   .venv-letta/bin/letta server --port 8283
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_DIR="${LETTA_ENV_DIR:-$ROOT/.venv-letta}"
TAG="${LETTA_TAG:-0.16.8}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
curl -sSfL "https://raw.githubusercontent.com/letta-ai/letta/$TAG/pyproject.toml" -o "$WORK/pyproject.toml"
curl -sSfL "https://raw.githubusercontent.com/letta-ai/letta/$TAG/uv.lock" -o "$WORK/uv.lock"
sha256sum "$WORK/pyproject.toml" "$WORK/uv.lock"
( cd "$WORK" && UV_PROJECT_ENVIRONMENT="$ENV_DIR" uv sync --frozen --no-install-project --no-dev --extra sqlite --extra server --python 3.11 )
VIRTUAL_ENV="$ENV_DIR" uv pip install --no-deps "letta==$TAG"
# asyncpg/pgvector are imported unconditionally even on SQLite; take the locked versions
# (the postgres extra itself needs a psycopg2 source build, which we skip).
locked() { grep -A1 "^name = \"$1\"$" "$WORK/uv.lock" | grep version | sed 's/.*"\(.*\)"/\1/'; }
VIRTUAL_ENV="$ENV_DIR" uv pip install --no-deps "asyncpg==$(locked asyncpg)" "pgvector==$(locked pgvector)" "pg8000==$(locked pg8000)"
# The wheel does not ship the Alembic migration tree that the Docker entrypoint
# runs (`alembic upgrade head`); take it from the PyPI source distribution.
SDIST_URL=$("$ENV_DIR/bin/python" - <<'PY'
import json, urllib.request
d = json.load(urllib.request.urlopen("https://pypi.org/pypi/letta/0.16.8/json"))
print(next(u["url"] for u in d["urls"] if u["packagetype"] == "sdist"))
PY
)
curl -sSfL "$SDIST_URL" -o "$WORK/letta-sdist.tar.gz"
sha256sum "$WORK/letta-sdist.tar.gz"
mkdir -p "$ENV_DIR/letta-src"
tar -xzf "$WORK/letta-sdist.tar.gz" -C "$ENV_DIR/letta-src" --strip-components=1 "letta-$TAG/alembic.ini" "letta-$TAG/alembic"
echo "migrations: $(ls "$ENV_DIR/letta-src/alembic/versions" | wc -l) revisions in $ENV_DIR/letta-src"
"$ENV_DIR/bin/python" -c "import importlib.metadata as m; print({p: m.version(p) for p in ('letta','mcp','fastmcp','openai','sqlalchemy','aiosqlite','sqlite-vec','asyncpg')})"
echo "run migrations:  (cd $ENV_DIR/letta-src && LETTA_PG_URI=postgresql://... $ENV_DIR/bin/alembic upgrade head)"
