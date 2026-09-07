"""Network-level stale-index fault. Unsupported systems must report N/A."""
from faults.base import Fault
class StaleIndexFault(Fault):
    name="stale_index"
    async def _start(self): raise NotImplementedError("requires a system-specific persistence network cut; never skip adapter.write")
    async def _end(self): pass
