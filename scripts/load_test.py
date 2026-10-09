"""Send realistic traffic to the API so the Grafana dashboard has data.

    python scripts/load_test.py --url http://localhost:8000 --n 300 --rps 5
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from heart.config import CLEAN_CSV, INPUT_FEATURES  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--rps", type=float, default=5.0)
    ap.add_argument("--invalid-share", type=float, default=0.05,
                    help="fraction of deliberately invalid requests (shows 4xx on dashboard)")
    args = ap.parse_args()

    rows = pd.read_csv(CLEAN_CSV)[INPUT_FEATURES].dropna().to_dict("records")
    session = requests.Session()
    session.trust_env = False  # ignore system/corporate proxy settings for local endpoints
    print(f"Sending {args.n} requests to {args.url}/predict at ~{args.rps} req/s ...", flush=True)
    ok = bad = errors = 0
    for i in range(1, args.n + 1):
        rec = {k: (int(v) if float(v).is_integer() else v) for k, v in random.choice(rows).items()}
        if random.random() < args.invalid_share:
            rec["age"] = 999
        try:
            r = session.post(f"{args.url}/predict", json=rec, timeout=5)
            ok += r.status_code == 200
            bad += r.status_code != 200
        except requests.RequestException as exc:
            errors += 1
            if errors <= 3:
                print(f"  request {i} failed: {exc}", flush=True)
            if errors == 10 and ok == 0:
                raise SystemExit("10 failures, 0 successes - is the URL reachable? "
                                 "(is minikube tunnel running?)")
        if i % 25 == 0 or i == args.n:
            print(f"  {i}/{args.n}  ok={ok} non_200={bad} errors={errors}", flush=True)
        time.sleep(1 / args.rps)
    print(f"done: sent={args.n} ok={ok} non_200={bad} errors={errors}")


if __name__ == "__main__":
    main()
