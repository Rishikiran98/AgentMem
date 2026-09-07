PYTHON ?= python
.PHONY: test smoke mem0-reproduction accuracy load scale faults analyze figures

test:
	$(PYTHON) -m pytest -q
smoke:
	$(PYTHON) scripts/demo_milestone3.py
mem0-reproduction:
	@test "$$BENCH_ALLOW_PAID_RUN" = 1 || (echo 'set BENCH_ALLOW_PAID_RUN=1'; exit 2)
	@test -n "$$BENCH_MAX_USD" || (echo 'set BENCH_MAX_USD'; exit 2)
	$(PYTHON) -m bench.run --system mem0 --benchmark longmemeval_s --seed 11 --config configs/mem0.yaml
accuracy:
	@echo 'Blocked until all selected system configs are frozen'; exit 2
load:
	@echo 'Invoke load.driver with a frozen adapter workload; no campaign is auto-launched'; exit 2
scale:
	@echo 'Blocked until system configs and dataset hash are frozen'; exit 2
faults:
	@echo 'Fault mechanisms require a selected pinned system compose stack'; exit 2
analyze:
	$(PYTHON) -m compileall -q analysis
figures:
	@echo 'Figures require validated canonical rows; placeholders are forbidden'; exit 2
