"""SIGKILL during an observable active-write boundary."""
from faults.base import Fault
class PartialWriteFault(Fault):
    name="partial_write"
    def __init__(self,emit,kill,restart): super().__init__(emit); self.kill=kill; self.restart=restart
    async def _start(self): await self.kill(signal="SIGKILL")
    async def _end(self): await self.restart()
