#!/usr/bin/env bash
# Smoke-test a running API instance.  Usage: scripts/smoke_test.sh [base_url]
set -euo pipefail
BASE="${1:-http://localhost:8000}"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "==> waiting for $BASE/health"
for i in $(seq 1 30); do
  if curl -fsS "$BASE/health" >/dev/null 2>&1; then break; fi
  sleep 2
  [ "$i" -eq 30 ] && { echo "API did not become healthy"; exit 1; }
done
curl -fsS "$BASE/health"; echo

echo "==> POST /predict"
RESP=$(curl -fsS -X POST "$BASE/predict" -H 'Content-Type: application/json' -d @"$HERE/sample_request.json")
echo "$RESP"
echo "$RESP" | grep -q '"confidence"' || { echo "missing confidence in response"; exit 1; }

echo "==> POST /predict/batch"
curl -fsS -X POST "$BASE/predict/batch" -H 'Content-Type: application/json' -d @"$HERE/sample_batch.json"; echo

echo "==> invalid payload must return 422"
CODE=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/predict" -H 'Content-Type: application/json' -d '{"age": 999}')
[ "$CODE" = "422" ] || { echo "expected 422, got $CODE"; exit 1; }

echo "==> /metrics"
curl -fsS "$BASE/metrics" | grep -E '^heart_api_(predictions|requests)_total' | head
echo "SMOKE TEST PASSED"
