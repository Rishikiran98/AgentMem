"""Auditable external fault lifecycle and outcome classification."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

@dataclass(frozen=True)
class FaultOutcome:
    label:str; recovered:bool

def classify(*, surfaced_error:bool, semantic_correct:bool, degraded:bool, recovered:bool)->FaultOutcome:
    if not semantic_correct: label="FAIL_LOUD" if surfaced_error else "FAIL_SILENT"
    elif degraded: label="DEGRADE"
    else: label="RECOVERED" if recovered else "DEGRADE"
    return FaultOutcome(label,recovered)

class Fault:
    name="base"
    def __init__(self, emit:Callable[[str],None]): self.emit=emit
    async def start(self): self.emit("FAULT_START"); await self._start(); self.emit("FAULT_EFFECT")
    async def end(self): await self._end(); self.emit("FAULT_END")
    async def recovery(self): self.emit("RECOVERY")
    async def _start(self): raise NotImplementedError
    async def _end(self): raise NotImplementedError
