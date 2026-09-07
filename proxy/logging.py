"""Append-only JSONL event writer.

Guarantees (tested in ``proxy/tests/test_append_only.py``):

* the file is opened with ``O_APPEND``; the writer exposes no truncate/rewrite
  API and never seeks;
* each event is exactly one line, written with a single ``os.write`` call under
  a lock, so concurrent appends never interleave;
* each event carries a monotonically increasing ``seq`` for the writer instance
  and the ``proxy_instance_id`` of the process that wrote it, so gaps, restarts
  and out-of-order delivery are detectable during analysis.
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class AppendOnlyJsonlLogger:
    def __init__(self, path: str | Path, *, fsync: bool = False, instance_id: str | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fsync = fsync
        self.instance_id = instance_id or uuid.uuid4().hex
        self._lock = threading.Lock()
        self._seq = 0
        self._closed = False
        # O_APPEND: every write lands at the current end of file regardless of
        # what other writers have done.  No O_TRUNC, ever.
        self._fd = os.open(str(self.path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)

    @property
    def seq(self) -> int:
        return self._seq

    def append(self, event: dict[str, Any]) -> dict[str, Any]:
        """Append one event.  Returns the event as written (with seq etc.)."""
        if self._closed:
            raise RuntimeError("logger is closed")
        with self._lock:
            self._seq += 1
            record = {"seq": self._seq, "proxy_instance_id": self.instance_id, "logged_at": utc_now_iso()}
            record.update(event)
            line = json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=_json_default) + "\n"
            data = line.encode("utf-8")
            view = memoryview(data)
            while view:
                written = os.write(self._fd, view)
                view = view[written:]
            if self.fsync:
                os.fsync(self._fd)
        return record

    def flush(self) -> None:
        with self._lock:
            if not self._closed:
                os.fsync(self._fd)

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                try:
                    os.fsync(self._fd)
                finally:
                    os.close(self._fd)
                    self._closed = True


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    return repr(obj)


def read_events(path: str | Path) -> Iterator[dict[str, Any]]:
    """Iterate over the events in a JSONL file (used by tests and analysis)."""
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
