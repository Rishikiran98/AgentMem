"""Credential-gated real-provider smoke; never falls back to fake upstream."""
import os,sys
if __name__ == "__main__":
    missing=[k for k in ("PROXY_UPSTREAM_API_KEY","BENCH_ALLOW_PAID_RUN") if not os.environ.get(k)]
    if missing:
        print("PENDING: set " + ", ".join(missing)); sys.exit(2)
    print("Run the proxy with PROXY_REQUIRE_ATTRIBUTION=1, then execute the one-instance command documented in docs/reproduction.md.")
