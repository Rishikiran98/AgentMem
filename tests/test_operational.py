from __future__ import annotations
import asyncio,json
import pytest
from load.arrivals import poisson_arrivals
from load.driver import run_open_loop
from load.histogram import summary
from faults.base import classify
from analysis.metrics import accuracy,latency
from analysis.rank import inversion
from analysis.validation import validate_run

def test_arrivals_reproducible():
 assert poisson_arrivals(10,2,4)==poisson_arrivals(10,2,4)
 assert poisson_arrivals(10,2,4)!=poisson_arrivals(10,2,5)

def test_quantile_support_and_metrics():
 assert summary(list(range(99)))['p99'] is None
 assert latency(list(range(100)))['p99_ms'] is not None
 assert accuracy([True,False])['accuracy']==.5

def test_fault_classification():
 assert classify(surfaced_error=False,semantic_correct=False,degraded=False,recovered=False).label=='FAIL_SILENT'
 assert classify(surfaced_error=True,semantic_correct=False,degraded=False,recovered=False).label=='FAIL_LOUD'

def test_rank_inversion():
 rows=[{'system':'a','accuracy':.9,'p99_ms':20,'tokens':5},{'system':'b','accuracy':.8,'p99_ms':5,'tokens':4}]
 assert inversion(rows,max_p99_ms=10)['observed']

async def test_open_loop_keeps_schedule_while_capacity_queues():
 async def op(_): await asyncio.sleep(.01)
 events=await run_open_loop(op,rate_rps=1000,duration_s=.03,seed=1,in_flight_limit=1)
 assert events and all(e['scheduled_send_time']<=e['actual_dispatch_time']<=e['completion_time'] for e in events)
 assert any(e['queue_delay_ms']>1 for e in events)

def test_validator_rejects_synthetic_and_dirty(tmp_path):
 p=tmp_path/'run.jsonl'; start={'event_type':'RUN_START','run_id':'r','seed':1,'configuration_id':'c','dataset':{'sha256':'x','synthetic':True},'adapter_fingerprint':{},'host':{'git':{'dirty':True}}}
 p.write_text(json.dumps(start)+'\n'+json.dumps({'event_type':'RUN_END','run_id':'r'})+'\n')
 report=validate_run(p); assert not report.valid and 'synthetic/fake data forbidden in canonical run' in report.errors
