"""CLI:  python -m bench.run --system mem0 --benchmark longmemeval_s --seed 42 --config configs/mem0.yaml

The proxy must already be running (``python -m proxy``); its instance id and
log path are recorded in RUN_START so the run and proxy traces can be joined.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import yaml

from adapters.registry import load_config
from bench.longmemeval import DatasetError, load_longmemeval
from bench.runner import RunConfig, Runner

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m bench.run")
    ap.add_argument("--system", required=True)
    ap.add_argument("--benchmark", required=True, choices=["longmemeval_s"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--config", required=True, help="system config YAML (configs/<system>.yaml)")
    ap.add_argument("--benchmark-config", default=None, help="default: configs/benchmarks/<benchmark>.yaml")
    ap.add_argument("--dataset", default=None, help="override dataset path from the benchmark config")
    ap.add_argument("--proxy", default="http://127.0.0.1:8811")
    ap.add_argument("--out", default=str(ROOT / "results/raw/runs"))
    ap.add_argument("--limit", type=int, default=None, help="first N instances of the seeded order")
    ap.add_argument("--question-ids", default=None, help="comma-separated question ids (then seeded ordering within that set)")
    ap.add_argument("--override-json", default=None, help="JSON merged over the system config; recorded as a deviation")
    ap.add_argument("--allow-synthetic", action="store_true", help="permit a synthetic dataset (tests/demos only)")
    ap.add_argument("--allow-hash-mismatch", action="store_true")
    ap.add_argument("--run-id", default=None, help="stable identity; required with --resume")
    ap.add_argument("--resume", action="store_true", help="resume exactly --run-id after fingerprint verification")
    args = ap.parse_args(argv)
    if args.resume and not args.run_id:
        ap.error("--resume requires --run-id")

    system_cfg = load_config(args.config)
    if system_cfg.get("system") != args.system:
        print(f"config {args.config} is for system {system_cfg.get('system')!r}, not {args.system!r}", file=sys.stderr)
        return 2
    bcfg_path = args.benchmark_config or str(ROOT / "configs/benchmarks" / f"{args.benchmark}.yaml")
    bcfg = yaml.safe_load(Path(bcfg_path).read_text())
    bcfg["_benchmark_config_path"] = bcfg_path

    dataset_path = args.dataset or bcfg["dataset"]["path"]
    try:
        dataset = load_longmemeval(dataset_path)
    except (FileNotFoundError, DatasetError) as exc:
        print(f"dataset error: {exc}\nrun scripts/fetch_longmemeval.py or pass --dataset", file=sys.stderr)
        return 2
    expected = bcfg["dataset"].get("expected_sha256")
    if expected and expected != dataset.sha256 and not args.allow_hash_mismatch:
        print(f"dataset sha256 {dataset.sha256} != expected {expected}; refusing (use --allow-hash-mismatch to record a deviation)", file=sys.stderr)
        return 2
    if dataset.synthetic and not args.allow_synthetic:
        print("dataset is synthetic; pass --allow-synthetic (never for paper runs)", file=sys.stderr)
        return 2

    cfg = RunConfig(
        system=args.system,
        benchmark=args.benchmark,
        seed=args.seed,
        config_path=args.config,
        system_config=system_cfg,
        benchmark_config=bcfg,
        proxy_url=args.proxy,
        out_dir=Path(args.out),
        limit=args.limit,
        question_ids=args.question_ids.split(",") if args.question_ids else None,
        overrides=json.loads(args.override_json) if args.override_json else None,
        run_id=args.run_id or "",
        resume=args.resume,
    )
    runner = Runner(cfg, dataset)
    print(f"run_id={cfg.run_id}\nout={runner.out_dir}")
    summary = asyncio.run(runner.run())
    print(json.dumps(summary, indent=1))
    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
