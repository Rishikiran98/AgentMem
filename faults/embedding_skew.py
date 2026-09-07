"""Query-side embedding endpoint/model switch controller."""
from faults.base import Fault
class EmbeddingSkewFault(Fault):
    name="embedding_skew"
    def __init__(self,emit,switch): super().__init__(emit); self.switch=switch
    async def _start(self): await self.switch(True)
    async def _end(self): await self.switch(False)
