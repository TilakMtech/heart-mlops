#!/usr/bin/env bash
# One-shot lab setup: tools -> venv -> lint/tests -> data/EDA/train -> Docker -> Minikube
# Usage (from the repo root):   bash scripts/lab_setup.sh
set -eo pipefail
cd "$(dirname "$0")/.."

step(){ STEP="$*"; echo; echo "==================== $* ===================="; }
trap 'echo; echo "!!!! FAILED at step: $STEP  - paste the lines above to Claude"' ERR

step "1 repo"
git pull --ff-only || true
git log --oneline | head -3

step "2 tools"
command -v make >/dev/null || sudo dnf install -y make || sudo yum install -y make
if ! command -v kubectl >/dev/null; then
  curl -sLO https://dl.k8s.io/release/v1.31.1/bin/linux/amd64/kubectl
  sudo install -m 0755 kubectl /usr/local/bin/kubectl
  rm -f kubectl
fi
make --version | head -1
kubectl version --client
minikube version | head -1
docker ps >/dev/null && echo "docker OK"

step "3 python env"
[ -d .venv ] || python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python --version
pip install -q -U pip
pip install -q -r requirements-dev.txt
echo "packages installed"

step "4 lint + tests"
make lint
make test 2>&1 | tail -3

step "5 data + EDA + train"
python scripts/download_data.py --force
make data
make eda
make train 2>&1 | tee train.log | tail -8

step "6 docker"
make docker-build
docker rm -f heart-api >/dev/null 2>&1 || true
make docker-run
sleep 20
docker ps --filter name=heart-api
make smoke | tail -2

step "7 kubernetes"
./scripts/deploy_minikube.sh 2>&1 | tee deploy.log | tail -4
kubectl -n heart-mlops get pods,svc,ingress,hpa
grep -q heart-api.local /etc/hosts || echo "$(minikube ip) heart-api.local" | sudo tee -a /etc/hosts
curl -s http://heart-api.local/health; echo

echo
echo "############  SETUP DONE - now start: terminal 1 'make mlflow', terminal 3 'minikube tunnel'  ############"
