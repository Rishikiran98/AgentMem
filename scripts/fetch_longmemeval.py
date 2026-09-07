"""Download (or verify) the official LongMemEval data files and print their SHA-256.

    python scripts/fetch_longmemeval.py            # download longmemeval_s_cleaned.json into data/
    python scripts/fetch_longmemeval.py --verify   # only hash what is already there

Source: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
The hash printed here goes into configs/benchmarks/longmemeval_s.yaml (dataset.expected_sha256).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "longmemeval_s_cleaned.json": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json",
    "longmemeval_oracle.json": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json",
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--files", nargs="*", default=["longmemeval_s_cleaned.json"])
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    args = ap.parse_args()
    data = Path(args.data_dir)
    data.mkdir(parents=True, exist_ok=True)
    rc = 0
    for name in args.files:
        dest = data / name
        if not dest.exists() and not args.verify:
            print(f"downloading {FILES[name]} -> {dest}")
            try:
                with httpx.stream("GET", FILES[name], follow_redirects=True, timeout=120) as r:
                    r.raise_for_status()
                    with dest.open("wb") as f:
                        for chunk in r.iter_bytes():
                            f.write(chunk)
            except Exception as exc:  # noqa: BLE001
                print(f"  download failed: {exc}\n  fetch it manually from the HF dataset page and place it at {dest}")
                rc = 1
                continue
        if dest.exists():
            print(f"{name}: {dest.stat().st_size} bytes sha256={sha256(dest)}")
        else:
            print(f"{name}: missing")
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
