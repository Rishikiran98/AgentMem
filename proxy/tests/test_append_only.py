"""The event log only ever grows; earlier bytes are never rewritten."""
from __future__ import annotations

import fcntl
import json
import os
import threading

import pytest

from proxy.logging import AppendOnlyJsonlLogger, read_events
from proxy.tests.conftest import chat_body, events


def test_file_opened_with_o_append(tmp_path):
    lg = AppendOnlyJsonlLogger(tmp_path / "a.jsonl")
    flags = fcntl.fcntl(lg._fd, fcntl.F_GETFL)
    assert flags & os.O_APPEND
    assert not hasattr(lg, "truncate") and not hasattr(lg, "rewrite") and not hasattr(lg, "seek")
    lg.close()


def test_prefix_is_immutable_and_seq_contiguous(tmp_path):
    p = tmp_path / "a.jsonl"
    lg = AppendOnlyJsonlLogger(p)
    for i in range(50):
        lg.append({"event_type": "T", "i": i})
    snapshot = p.read_bytes()
    for i in range(50, 120):
        lg.append({"event_type": "T", "i": i})
    lg.close()
    after = p.read_bytes()
    assert after[: len(snapshot)] == snapshot
    assert len(after) > len(snapshot)
    evs = list(read_events(p))
    assert [e["seq"] for e in evs] == list(range(1, 121))
    assert [e["i"] for e in evs] == list(range(120))
    assert len({e["proxy_instance_id"] for e in evs}) == 1


def test_reopen_appends_with_new_instance_id(tmp_path):
    p = tmp_path / "a.jsonl"
    lg1 = AppendOnlyJsonlLogger(p)
    lg1.append({"event_type": "T", "i": 0})
    lg1.close()
    before = p.read_bytes()
    lg2 = AppendOnlyJsonlLogger(p)
    lg2.append({"event_type": "T", "i": 1})
    lg2.close()
    after = p.read_bytes()
    assert after.startswith(before)
    evs = list(read_events(p))
    assert [e["i"] for e in evs] == [0, 1]
    assert evs[0]["proxy_instance_id"] != evs[1]["proxy_instance_id"]
    assert evs[0]["seq"] == 1 and evs[1]["seq"] == 1  # seq is per writer instance


def test_threaded_appends_never_interleave(tmp_path):
    p = tmp_path / "a.jsonl"
    lg = AppendOnlyJsonlLogger(p)
    big = "x" * 20_000

    def work(t: int) -> None:
        for i in range(50):
            lg.append({"event_type": "T", "t": t, "i": i, "pad": big})

    threads = [threading.Thread(target=work, args=(t,)) for t in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    lg.close()
    lines = p.read_bytes().split(b"\n")
    assert lines[-1] == b""
    evs = [json.loads(l) for l in lines[:-1]]  # every line must parse
    assert len(evs) == 400
    assert sorted(e["seq"] for e in evs) == list(range(1, 401))
    assert all(e["pad"] == big for e in evs)


def test_closed_logger_refuses_writes(tmp_path):
    lg = AppendOnlyJsonlLogger(tmp_path / "a.jsonl")
    lg.close()
    with pytest.raises(RuntimeError):
        lg.append({"event_type": "T"})


async def test_proxy_log_only_grows_across_requests(client, log_path):
    """End-to-end: the proxy's own log file grows monotonically and starts with PROXY_START."""
    sizes = [log_path.stat().st_size]
    contents = [log_path.read_bytes()]
    for i in range(5):
        model = "fail-500" if i % 2 else "fake-model"
        await client.post("/v1/chat/completions", json=chat_body(model=model), headers={"X-Bench-System": "mem0", "X-Bench-Operation": "write"})
        sizes.append(log_path.stat().st_size)
        contents.append(log_path.read_bytes())
    assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes)
    for earlier, later in zip(contents, contents[1:]):
        assert later.startswith(earlier)
    all_events = events(log_path, None)
    assert all_events[0]["event_type"] == "PROXY_START"
    assert [e["event_type"] for e in all_events[1:]] == ["MODEL_CALL"] * 5
