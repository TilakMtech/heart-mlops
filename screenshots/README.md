# Screenshots checklist

Capture these while running the pipeline and save them here with the given file names.
The report (`reports/REPORT.md` / `reports/Heart_Disease_MLOps_Report.docx`) has placeholders that reference them.

| # | File name | What to capture | Command |
|---|-----------|-----------------|---------|
| 1 | `01_mlflow_experiments.png` | MLflow UI – experiment list with parent + nested runs | `make mlflow` → http://localhost:5000 |
| 2 | `02_mlflow_run_compare.png` | MLflow "Compare runs" view (cv_roc_auc_mean, test_* metrics) | select the 4 child runs → Compare |
| 3 | `03_mlflow_artifacts.png` | One run's artifacts (plots, model, grid results) | open `logreg` run |
| 4 | `04_github_actions_pipeline.png` | Successful workflow run graph (lint → test → train → docker) | GitHub → Actions |
| 5 | `05_github_actions_artifacts.png` | Artifacts list + job summary of a run | bottom of the run page |
| 6 | `06_github_actions_failure.png` | A deliberately failing run (e.g. flake8 error) showing clear logs | push a lint error on a branch |
| 7 | `07_docker_build_run.png` | `docker build` + `docker run` + `scripts/smoke_test.sh` output | `make docker-build docker-run smoke` |
| 8 | `08_swagger_predict.png` | FastAPI `/docs` → POST /predict response | http://localhost:8000/docs |
| 9 | `09_k8s_resources.png` | `kubectl -n heart-mlops get pods,svc,ingress,hpa` | after `make k8s-deploy` |
| 10 | `10_k8s_predict_ingress.png` | `curl` to the Ingress / LoadBalancer URL returning a prediction | see README §7 |
| 11 | `11_prometheus_targets.png` | Prometheus → Status → Targets (heart-api UP) | `kubectl -n heart-mlops port-forward svc/prometheus 9090:9090` |
| 12 | `12_grafana_dashboard.png` | Grafana "Heart Disease API" dashboard with traffic | run `scripts/load_test.py` first |
| 13 | `13_api_json_logs.png` | `kubectl logs` showing JSON request/prediction logs | `kubectl -n heart-mlops logs deploy/heart-api` |
