#!/usr/bin/env bash
# End-to-end deploy to a local Minikube cluster.
#   ./scripts/deploy_minikube.sh
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE=heart-risk-api:latest

echo "==> Starting minikube (if needed)"
minikube status >/dev/null 2>&1 || minikube start --cpus=4 --memory=6g
minikube addons enable ingress
minikube addons enable metrics-server

echo "==> Building image and loading it into the cluster"
docker build -t "$IMAGE" .
minikube image load "$IMAGE"

echo "==> Applying manifests"
kubectl apply -k k8s/
kubectl -n heart-mlops rollout status deploy/heart-api --timeout=180s
kubectl -n heart-mlops get pods,svc,ingress,hpa -o wide

URL=$(minikube service heart-api -n heart-mlops --url | head -1)
echo "==> API reachable at $URL"
./scripts/smoke_test.sh "$URL"

cat <<MSG

Next steps
  LoadBalancer : run 'minikube tunnel' in another terminal, then: kubectl -n heart-mlops get svc heart-api
  Ingress      : echo "\$(minikube ip) heart-api.local" | sudo tee -a /etc/hosts ; curl http://heart-api.local/health
  Prometheus   : minikube service prometheus -n heart-mlops
  Grafana      : minikube service grafana -n heart-mlops   (admin/admin)
  Traffic      : python scripts/load_test.py --url $URL --n 300
MSG
