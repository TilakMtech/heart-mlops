PY ?= python
export PYTHONPATH := src:.

.PHONY: help install data eda train train-quick test lint serve mlflow docker-build docker-run smoke compose-up compose-down k8s-deploy k8s-delete notebooks all

help:            ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

install:         ## install all dependencies (training, serving, dev)
	$(PY) -m pip install -r requirements-dev.txt

data:            ## download raw data + build cleaned dataset
	$(PY) scripts/download_data.py
	$(PY) -m heart.data

eda: data        ## EDA figures -> reports/figures
	$(PY) -m heart.eda

train: data      ## full hyper-parameter search, tracked in MLflow
	$(PY) -m heart.train

train-quick: data ## tiny grids (CI smoke)
	$(PY) -m heart.train --quick

test:            ## unit tests + coverage
	$(PY) -m pytest --cov=heart --cov=api

lint:            ## flake8
	$(PY) -m flake8

serve:           ## run API locally on :8000
	uvicorn api.main:app --reload --port 8000

mlflow:          ## MLflow UI on :5000
	mlflow ui --backend-store-uri ./mlruns --port 5000

notebooks:       ## execute notebooks in place
	jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

docker-build:    ## build API image
	docker build -t heart-risk-api:latest .

docker-run:      ## run API container on :8000
	docker run --rm -d --name heart-api -p 8000:8000 heart-risk-api:latest

smoke:           ## smoke-test API on :8000
	./scripts/smoke_test.sh http://localhost:8000

compose-up:      ## API + Prometheus + Grafana
	docker compose up -d --build

compose-down:
	docker compose down

k8s-deploy:      ## deploy to minikube
	./scripts/deploy_minikube.sh

k8s-delete:
	kubectl delete -k k8s/

all: install lint test train docker-build  ## full local pipeline
