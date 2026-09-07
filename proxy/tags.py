"""Attribution tags and how they reach the proxy.

Every model request must be attributable to::

    system, configuration, seed, session_id, operation  (+ run_id, client_id)

Three transport mechanisms are supported because the systems under test differ
in what they let us control:

1. **Per-request headers** (``X-Bench-System`` ...).  Exact, per request.  Used by
   harness-owned calls (reader, judge, canary embeddings) and by any adapter
   whose client lets us set default headers.

2. **Static token** carried in the API key (``sk-bench-<token>``) or in the URL
   prefix (``/b/<token>/v1/...``).  Both are settable on every system under
   test because every system accepts an API key and a base URL.  The token is a
   base64url-encoded JSON object of tags and is fixed for the lifetime of a
   client instance -> it carries run-level tags (system, configuration, seed,
   client_id) and, optionally, session/operation when a client is dedicated to
   one.

3. **Scope registry**.  For systems that make model calls from inside their own
   server process (no per-request headers possible), the harness tells the proxy
   "client X is now executing operation O for session S" via
   ``POST /_bench/scope/enter`` and later ``/_bench/scope/exit``.  Requests from
   that client are tagged with the active scope.  When several scopes are
   active for the same client and disagree, the affected fields are recorded as
   ambiguous rather than guessed.

Precedence per field: header > scope > static token.  The event records which
source supplied each field so that analysis can audit attribution quality.
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

TAG_FIELDS: tuple[str, ...] = (
    "system",
    "configuration",
    "seed",
    "session_id",
    "operation",
    "run_id",
    "client_id",
)

# "write | consolidate | read | embed | answer | judge" from the spec plus:
#   settle  - canary writes / polls used only for settlement detection
#   meta    - non-inference endpoints such as GET /v1/models
#   warmup  - warm-up window traffic that must be excluded from reported metrics
ALLOWED_OPERATIONS: frozenset[str] = frozenset(
    {"write", "consolidate", "read", "embed", "answer", "judge", "settle", "meta", "warmup"}
)

HEADER_PREFIX = "x-bench-"
HEADER_TO_FIELD: dict[str, str] = {
    "x-bench-system": "system",
    "x-bench-configuration": "configuration",
    "x-bench-seed": "seed",
    "x-bench-session": "session_id",
    "x-bench-operation": "operation",
    "x-bench-run": "run_id",
    "x-bench-client": "client_id",
}
FIELD_TO_HEADER: dict[str, str] = {v: k for k, v in HEADER_TO_FIELD.items()}
TAGS_HEADER = "x-bench-tags"  # compact form: a whole encoded token in one header

API_KEY_PREFIXES: tuple[str, ...] = ("sk-bench-", "bench-")
PATH_TOKEN_PREFIX = "/b/"

_MAX_TAG_LEN = 256


class TagError(ValueError):
    """Raised for malformed or disallowed tag values."""


def normalize_tags(raw: Mapping[str, Any], *, allowed_systems: Iterable[str] | None = None) -> dict[str, Any]:
    """Validate and normalise a tag mapping.  Unknown keys are rejected."""
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in TAG_FIELDS:
            raise TagError(f"unknown tag field {key!r}; allowed: {TAG_FIELDS}")
        if value is None:
            continue
        if key == "seed":
            try:
                out[key] = int(value)
            except (TypeError, ValueError) as exc:
                raise TagError(f"seed must be an integer, got {value!r}") from exc
            continue
        if not isinstance(value, str):
            value = str(value)
        if not value:
            continue
        if len(value) > _MAX_TAG_LEN:
            raise TagError(f"tag {key!r} exceeds {_MAX_TAG_LEN} characters")
        if key == "operation" and value not in ALLOWED_OPERATIONS:
            raise TagError(f"operation {value!r} not in {sorted(ALLOWED_OPERATIONS)}")
        if key == "system" and allowed_systems is not None and value not in set(allowed_systems):
            raise TagError(f"system {value!r} not in allowed systems {sorted(allowed_systems)}")
        out[key] = value
    return out


def encode_tags(tags: Mapping[str, Any]) -> str:
    """Encode tags as a URL-safe, unpadded base64 JSON token."""
    clean = normalize_tags(tags)
    payload = json.dumps(clean, separators=(",", ":"), sort_keys=True).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")


def decode_tags(token: str) -> dict[str, Any]:
    padded = token + "=" * (-len(token) % 4)
    try:
        payload = base64.urlsafe_b64decode(padded.encode())
        obj = json.loads(payload)
    except Exception as exc:  # noqa: BLE001 - any decoding problem is a TagError
        raise TagError(f"malformed tag token: {exc}") from exc
    if not isinstance(obj, dict):
        raise TagError("tag token must decode to an object")
    return normalize_tags(obj)


def api_key_for(tags: Mapping[str, Any]) -> str:
    """Build an API key that carries static tags to the proxy."""
    return "sk-bench-" + encode_tags(tags)


def token_from_api_key(authorization_header: str | None) -> str | None:
    """Extract the tag token from ``Authorization: Bearer sk-bench-<token>``."""
    if not authorization_header:
        return None
    parts = authorization_header.split(None, 1)
    key = parts[1].strip() if len(parts) == 2 and parts[0].lower() == "bearer" else authorization_header.strip()
    for prefix in API_KEY_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix):]
    return None


def split_path_token(path: str) -> tuple[str | None, str]:
    """``/b/<token>/v1/x`` -> (token, ``/v1/x``); otherwise (None, path)."""
    if path.startswith(PATH_TOKEN_PREFIX):
        rest = path[len(PATH_TOKEN_PREFIX):]
        token, sep, remainder = rest.partition("/")
        if token and sep:
            return token, "/" + remainder
    return None, path


# --------------------------------------------------------------------------- #
# Scope registry
# --------------------------------------------------------------------------- #

@dataclass
class Scope:
    scope_id: str
    tags: dict[str, Any]
    entered_at: str


class ScopeRegistry:
    """In-memory registry of active harness scopes, keyed by client_id.

    The registry is only meaningful within a single proxy process; the proxy
    must run single-worker (documented in README).
    """

    def __init__(self) -> None:
        self._scopes: dict[str, dict[str, Scope]] = {}

    def enter(self, client_id: str, scope_id: str, tags: Mapping[str, Any], entered_at: str) -> Scope:
        if not client_id or not scope_id:
            raise TagError("client_id and scope_id are required")
        per_client = self._scopes.setdefault(client_id, {})
        if scope_id in per_client:
            raise TagError(f"scope {scope_id!r} already active for client {client_id!r}")
        scope = Scope(scope_id=scope_id, tags=dict(tags), entered_at=entered_at)
        per_client[scope_id] = scope
        return scope

    def exit(self, client_id: str, scope_id: str) -> Scope:
        per_client = self._scopes.get(client_id) or {}
        scope = per_client.pop(scope_id, None)
        if scope is None:
            raise TagError(f"scope {scope_id!r} is not active for client {client_id!r}")
        if not per_client:
            self._scopes.pop(client_id, None)
        return scope

    def active(self, client_id: str | None) -> list[Scope]:
        if client_id is None:
            return []
        return list((self._scopes.get(client_id) or {}).values())

    def snapshot(self) -> dict[str, list[dict[str, Any]]]:
        return {
            cid: [{"scope_id": s.scope_id, "tags": s.tags, "entered_at": s.entered_at} for s in scopes.values()]
            for cid, scopes in self._scopes.items()
        }

    def resolve(self, client_id: str | None) -> tuple[dict[str, Any], list[str], str, list[str]]:
        """Merge the active scopes of a client.

        Returns ``(tags, scope_ids, state, ambiguous_fields)`` where state is
        ``none`` (no scope), ``single``, ``agree`` (several scopes, identical
        values on every field) or ``ambiguous``.
        """
        scopes = self.active(client_id)
        if not scopes:
            return {}, [], "none", []
        if len(scopes) == 1:
            return dict(scopes[0].tags), [scopes[0].scope_id], "single", []
        merged: dict[str, Any] = {}
        ambiguous: list[str] = []
        for fld in TAG_FIELDS:
            values = {json.dumps(s.tags.get(fld), sort_keys=True) for s in scopes if fld in s.tags}
            if len(values) == 1:
                merged[fld] = next(s.tags[fld] for s in scopes if fld in s.tags)
            elif len(values) > 1:
                ambiguous.append(fld)
        state = "ambiguous" if ambiguous else "agree"
        return merged, [s.scope_id for s in scopes], state, ambiguous


# --------------------------------------------------------------------------- #
# Attribution resolution
# --------------------------------------------------------------------------- #

@dataclass
class Attribution:
    tags: dict[str, Any] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)
    scope_ids: list[str] = field(default_factory=list)
    scope_state: str = "none"
    ambiguous_fields: list[str] = field(default_factory=list)
    static_token_present: bool = False

    @property
    def unattributed(self) -> bool:
        return not any(k in self.tags for k in ("system", "operation", "session_id", "run_id"))

    def get(self, key: str) -> Any:
        return self.tags.get(key)


def resolve_attribution(
    headers: Mapping[str, str],
    *,
    path_token: str | None,
    registry: ScopeRegistry,
    allowed_systems: Iterable[str],
) -> Attribution:
    """Combine header, scope and static-token tags with precedence header > scope > token."""
    allowed = tuple(allowed_systems)
    attr = Attribution()

    # Static token: URL path prefix or API key (path wins if both are present).
    static: dict[str, Any] = {}
    static_source = None
    key_token = token_from_api_key(headers.get("authorization"))
    if path_token:
        static = decode_tags(path_token)
        static_source = "path_token"
    elif key_token:
        static = decode_tags(key_token)
        static_source = "api_key_token"
    attr.static_token_present = static_source is not None
    for k, v in static.items():
        attr.tags[k] = v
        attr.sources[k] = static_source  # type: ignore[assignment]

    # Header tags (individual headers plus the compact X-Bench-Tags token).
    header_tags: dict[str, Any] = {}
    compact = headers.get(TAGS_HEADER)
    if compact:
        header_tags.update(decode_tags(compact))
    for hname, fld in HEADER_TO_FIELD.items():
        val = headers.get(hname)
        if val is not None and val != "":
            header_tags[fld] = val
    header_tags = normalize_tags(header_tags)

    # Client identity for scope lookup: header > token > system name.
    client_id = header_tags.get("client_id") or attr.tags.get("client_id") or header_tags.get("system") or attr.tags.get("system")
    scope_tags, scope_ids, scope_state, ambiguous = registry.resolve(client_id)
    attr.scope_ids = scope_ids
    attr.scope_state = scope_state
    attr.ambiguous_fields = ambiguous
    for k, v in scope_tags.items():
        attr.tags[k] = v
        attr.sources[k] = "scope"
    for k in ambiguous:
        # A disagreeing field must not silently keep the static-token value.
        attr.tags.pop(k, None)
        attr.sources[k] = "ambiguous"

    for k, v in header_tags.items():
        attr.tags[k] = v
        attr.sources[k] = "header"

    if client_id and "client_id" not in attr.tags:
        attr.tags["client_id"] = client_id
        attr.sources["client_id"] = "derived_from_system"

    # Final validation with the system allow-list.
    attr.tags = normalize_tags(attr.tags, allowed_systems=allowed)
    return attr
