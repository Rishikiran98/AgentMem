"""Coordinated-omission-safe asyncio open-loop scheduler."""
from __future__ import annotations
import asyncio, inspect, time
from dataclasses import dataclass, asdict
from typing import Awaitable, Callable, Any
from load.arrivals import poisson_arrivals

@dataclass
class RequestEvent:
    request_id: int; phase: str; scheduled_send_time: float; actual_dispatch_time: float
    completion_time: float; queue_delay_ms: float; service_latency_ms: float
    end_to_end_latency_ms: float; success: bool; error: str|None

async def run_open_loop(operation: Callable[[int], Awaitable[Any]], *, rate_rps: float,
 duration_s: float, seed: int, in_flight_limit: int, phase: str="measurement",
 clock: Callable[[],float]=time.perf_counter) -> list[dict]:
    offsets=poisson_arrivals(rate_rps,duration_s,seed); origin=clock(); sem=asyncio.Semaphore(in_flight_limit)
    async def one(i, offset):
        scheduled=origin+offset
        await asyncio.sleep(max(0, scheduled-clock()))
        # Capacity waiting deliberately occurs after the immutable schedule timestamp.
        async with sem:
            dispatch=clock(); error=None
            try: await operation(i); ok=True
            except Exception as exc: ok=False; error=f"{type(exc).__name__}: {exc}"[:1000]
            done=clock()
        return asdict(RequestEvent(i,phase,scheduled,dispatch,done,(dispatch-scheduled)*1000,
          (done-dispatch)*1000,(done-scheduled)*1000,ok,error))
    return await asyncio.gather(*(one(i,o) for i,o in enumerate(offsets)))
