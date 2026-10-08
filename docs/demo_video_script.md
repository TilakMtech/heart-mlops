# Demo video – suggested 5-minute run-through

| Time | Show | Command / screen |
|------|------|------------------|
| 0:00 | Repo tour + architecture diagram | README, `docs/architecture.png` |
| 0:30 | Data download + cleaning | `make data` |
| 0:50 | EDA figures | `notebooks/01_eda.ipynb` or `reports/figures/` |
| 1:20 | Training with MLflow | `make train` then `make mlflow` → compare runs |
| 2:10 | Tests + lint | `make lint test` |
| 2:30 | CI/CD | GitHub → Actions: green run, artifacts, job summary; one red run |
| 3:10 | Docker | `make docker-build docker-run smoke`, Swagger `/docs` |
| 3:40 | Kubernetes | `./scripts/deploy_minikube.sh`, `kubectl get pods,svc,ingress,hpa`, curl via Ingress |
| 4:20 | Monitoring | `python scripts/load_test.py`, Grafana dashboard, Prometheus targets, `kubectl logs` JSON lines |
| 4:50 | Wrap-up | Report + results table |
