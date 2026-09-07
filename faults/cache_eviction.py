"""Evict a genuine configured volatile cache; otherwise report N/A."""
from faults.base import Fault
class CacheEvictionFault(Fault):
    name="cache_eviction"
    def __init__(self,emit,flush): super().__init__(emit); self.flush=flush
    async def _start(self): await self.flush(command="FLUSHDB")
    async def _end(self): pass
