#!/usr/bin/env bash
# End-to-end deploy to a local Minikube cluster (works on macOS/Apple Silicon and Linux).
#   ./scripts/deploy_minikube.sh
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE=heart-risk-api:latest
PF_PORT=18080

echo "==> Starting minikube (if needed)"
minikube status >/dev/null 2>&1 || minikube start --driver=docker --cpus=4 --memory=6g
minikube addons enable ingress
minikube addons enable metrics-server

echo "==> Building image and loading it into the cluster"
docker build -t "$IMAGE" .
minikube image load "$IMAGE"

echo "==> Applying manifests"
kubectl apply -k k8s/
kubectl -n heart-mlops rollout restart deploy/heart-api   # pick up a re-loaded :latest image
kubectl -n heart-mlops rollout status deploy/heart-api --timeout=180s
kubectl -n heart-mlops get pods,svc,ingress,hpa -o wide

# `minikube service --url` blocks on macOS (docker driver), so smoke-test via port-forward
echo "==> Smoke test through kubectl port-forward on :$PF_PORT"
kubectl -n heart-mlops port-forward svc/heart-api "$PF_PORT":8080 >/dev/null 2>&1 &
PF_PID=$!
trap 'kill $PF_PID 2>/dev/null || true' EXIT
sleep 3
./scripts/smoke_test.sh "http://localhost:$PF_PORT"

cat <<'MSG'

Deployed. Next steps (see README section 7):
  1. In a SECOND terminal keep running:   minikube tunnel        (asks for your Mac password)
  2. LoadBalancer : kubectl -n heart-mlops get svc heart-api  -> EXTERNAL-IP 127.0.0.1
                    curl http://127.0.0.1:8080/health
  3. Ingress      : echo "127.0.0.1 heart-api.local" | sudo tee -a /etc/hosts   (once)
                    curl http://heart-api.local/health
  4. Prometheus   : kubectl -n heart-mlops port-forward svc/prometheus 9090:9090   -> http://localhost:9090
     Grafana      : kubectl -n heart-mlops port-forward svc/grafana 3000:3000      -> http://localhost:3000 (admin/admin)
  5. Traffic      : python scripts/load_test.py --url http://heart-api.local --n 300 --rps 5
MSG
