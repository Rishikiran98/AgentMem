from __future__ import annotations

from adapters.base import CanaryRecord, MemoryAdapter, ReadHit, SettlementConfig


class PollingAdapter(MemoryAdapter):
    name = "test"

    def __init__(self, visible_on: int):
        super().__init__(settlement=SettlementConfig(timeout_s=1, poll_interval_s=0))
        self.visible_on = visible_on
        self.reads = 0

    def config_fingerprint(self):
        return {"visible_on": self.visible_on}

    async def _reset_impl(self, session_id): pass
    async def _write_impl(self, session_id, messages, *, metadata): return {}
    async def _canary_write_impl(self, session_id, text): return True, ["c"]
    async def _read_impl(self, session_id, query, *, operation):
        self.reads += 1
        text = query if self.reads >= self.visible_on else "not yet"
        return [ReadHit("c", text)], None


async def test_first_poll_is_interval_censored_from_zero():
    a = PollingAdapter(1)
    result = await a.wait_settled("s")
    fields = result.to_event_fields()
    assert fields["poll_count"] == 1
    assert fields["visibility_lower_bound_ms"] == 0
    assert fields["visibility_upper_bound_ms"] >= 0
    assert len(result.canary.token.removeprefix("memharness-canary-")) == 32


async def test_failed_poll_is_next_lower_boundary():
    a = PollingAdapter(2)
    result = await a.wait_settled("s")
    fields = result.to_event_fields()
    assert fields["poll_count"] == 2
    assert 0 <= fields["visibility_lower_bound_ms"] <= fields["visibility_upper_bound_ms"]
