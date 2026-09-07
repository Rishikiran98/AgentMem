"""Optional local token estimates.

The **authoritative** token counts are the ``usage`` numbers reported by the
upstream provider; those are what the provider bills and they are recorded
verbatim.  Local estimates serve two purposes only:

1. cross-checking the upstream numbers (drift detection), and
2. giving an approximate prompt size for *failed* requests, which carry no
   upstream usage but still consumed provider capacity.

Local estimates use ``tiktoken``.  Encoding files are fetched from the
internet on first use; when unavailable (offline hosts) every function returns
``None`` and the event records ``*_local: null``.  Nothing in the benchmark
depends on local estimates being present.
"""
from __future__ import annotations

import threading
from typing import Any

try:  # pragma: no cover - import guard
    import tiktoken
except Exception:  # noqa: BLE001
    tiktoken = None  # type: ignore[assignment]

_lock = threading.Lock()
_encodings: dict[str, Any] = {}
_failures: dict[str, str] = {}
# Set once the base encoding cannot be loaded (offline host).  After that no
# request ever triggers a network attempt again.
_disabled: bool = tiktoken is None
_DEFAULT_PRELOAD = ("cl100k_base", "gpt-4o-mini", "gpt-4o", "text-embedding-3-small", "text-embedding-3-large")

# Chat-format overhead per the OpenAI cookbook ("How to count tokens").
_TOKENS_PER_MESSAGE = 3
_TOKENS_PER_NAME = 1
_REPLY_PRIMING = 3


def _encoding_for(model: str | None):
    global _disabled
    if _disabled:
        return None
    key = model or "cl100k_base"
    with _lock:
        if key in _encodings:
            return _encodings[key]
        if key in _failures:
            return None
        try:
            try:
                enc = tiktoken.encoding_for_model(key)
            except KeyError:
                enc = tiktoken.get_encoding("cl100k_base")
        except Exception as exc:  # noqa: BLE001 - offline, unknown model, ...
            _failures[key] = f"{type(exc).__name__}: {exc}"[:200]
            if key == "cl100k_base" or not _encodings:
                # Base encoding unreachable -> nothing will load; stop trying.
                _disabled = True
            return None
        _encodings[key] = enc
        return enc


def preload(models: tuple[str, ...] | list[str] = _DEFAULT_PRELOAD) -> dict[str, Any]:
    """Load encodings at proxy start-up so no request pays the download cost.

    Called once from ``create_app``; returns :func:`tokenizer_status`.
    """
    _encoding_for("cl100k_base")
    if not _disabled:
        for m in models:
            _encoding_for(m)
    return tokenizer_status()


def count_text_tokens(text: str | None, model: str | None) -> int | None:
    if text is None:
        return None
    enc = _encoding_for(model)
    if enc is None:
        return None
    return len(enc.encode(text, disallowed_special=()))


def count_chat_prompt_tokens(messages: Any, model: str | None) -> int | None:
    """Approximate prompt tokens for a chat request (text content only)."""
    enc = _encoding_for(model)
    if enc is None or not isinstance(messages, list):
        return None
    total = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        total += _TOKENS_PER_MESSAGE
        for key, value in msg.items():
            if isinstance(value, str):
                total += len(enc.encode(value, disallowed_special=()))
            elif isinstance(value, list):  # multi-part content
                for part in value:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        total += len(enc.encode(part["text"], disallowed_special=()))
            if key == "name":
                total += _TOKENS_PER_NAME
    return total + _REPLY_PRIMING


def count_embedding_input_tokens(inp: Any, model: str | None) -> int | None:
    """Token estimate for an embeddings ``input`` (str, list[str], list[int], list[list[int]])."""
    if isinstance(inp, str):
        return count_text_tokens(inp, model)
    if isinstance(inp, list):
        if all(isinstance(x, int) for x in inp):
            return len(inp)
        total = 0
        for item in inp:
            if isinstance(item, str):
                n = count_text_tokens(item, model)
                if n is None:
                    return None
                total += n
            elif isinstance(item, list) and all(isinstance(x, int) for x in item):
                total += len(item)
            else:
                return None
        return total
    return None


def embedding_input_count(inp: Any) -> int | None:
    if isinstance(inp, str):
        return 1
    if isinstance(inp, list):
        if inp and all(isinstance(x, int) for x in inp):
            return 1  # a single pre-tokenised input
        return len(inp)
    return None


def tokenizer_status() -> dict[str, Any]:
    """Describe local tokenizer availability for the PROXY_START event."""
    status: dict[str, Any] = {"tiktoken_version": getattr(tiktoken, "__version__", None) if tiktoken else None}
    enc = _encoding_for("cl100k_base")
    status["local_estimates_available"] = enc is not None
    status["disabled"] = _disabled
    status["loaded_encodings"] = sorted(_encodings)
    status["failures"] = dict(_failures)
    return status
