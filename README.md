# Heart Disease Risk – End-to-End MLOps Pipeline

BITS Pilani · AIMLCZG523 Machine Learning Operations · Assignment 01

A reproducible ML system that predicts the presence of heart disease from the
UCI Cleveland dataset and serves it as a monitored, containerised API on Kubernetes.

![architecture](docs/architecture.png)

| Stage | Implementation | Where |
|-------|----------------|-------|
| Data acquisition | Script downloads the UCI file (falls back to a GitHub mirror if UCI is blocked) | `scripts/download_data.py` |
| Cleaning + EDA | Stateless cleaning, 6 figures + summary JSON | `src/heart/data.py`, `src/heart/eda.py`, `notebooks/01_eda.ipynb` |
| Features | Custom `HeartFeatureBuilder` + `ColumnTransformer` (impute, scale, one-hot) | `src/heart/features.py` |
| Models | Logistic Regression (± engineered features), Random Forest, Gradient Boosting, tuned with `GridSearchCV` | `src/heart/train.py` |
| Tracking | MLflow: nested run per model, params, CV + hold-out metrics, plots, models, registry | `mlruns/` (local) |
| Packaging | Whole pipeline in `models/model.joblib` + `model_meta.json`; pinned requirements | `models/`, `requirements*.txt` |
| Serving | FastAPI `/predict`, `/predict/batch`, `/health`, `/metrics` | `api/` |
| Container | `python:3.11-slim`, non-root, healthcheck | `Dockerfile` |
| CI/CD | GitHub Actions: lint → tests → train + quality gate → Docker build + smoke test → GHCR | `.github/workflows/ci.yml` |
| Deployment | Minikube: Deployment, LoadBalancer Service, Ingress, HPA (Kustomize) | `k8s/` |
| Monitoring | JSON request logs, Prometheus metrics, provisioned Grafana dashboard | `monitoring/`, `k8s/monitoring/` |

---

## 1. Setup

Requires Python 3.11, Docker, and (for deployment) Minikube + kubectl.

```bash
git clone https://github.com/TilakMtech/heart-mlops.git heart-mlops && cd heart-mlops
python3.11 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt                        # or: conda env create -f environment.yml
```

`make help` lists all shortcuts. On Windows without `make`, run the commands shown in the `Makefile`
with `PYTHONPATH=src;.` set.

## 2. Data

```bash
make data          # = python scripts/download_data.py && python -m heart.data
```

- Raw: `data/raw/heart_raw.csv` (303 rows, 14 UCI columns; `?` → empty)
- Clean: `data/processed/heart_clean.csv` (target binarised: `num > 0 → 1`)
- Manual alternative: download `processed.cleveland.data` from
  <https://archive.ics.uci.edu/dataset/45/heart+disease>, then save it as `data/raw/heart_raw.csv` with the header
  `age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,num`.

## 3. EDA

```bash
make eda           # figures -> reports/figures/
jupyter lab notebooks/01_eda.ipynb
```

## 4. Training + experiment tracking

```bash
make train         # full grid search (~1-2 min); `make train-quick` for a fast run
make mlflow        # http://localhost:5000
```

Outputs: `models/model.joblib`, `models/model_meta.json`, `reports/model_comparison.csv`,
`reports/figures/models/*`. Every model is logged as a nested MLflow run under one session run.

## 5. Tests and lint

```bash
make lint
make test          # tests: data, features, model, API
```

## 6. Serve locally / Docker

```bash
make serve                                   # uvicorn on :8000 (dev)
make docker-build && make docker-run         # container on :8000
make smoke                                   # health, predict, batch, 422 check, metrics
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d @scripts/sample_request.json
```

Example response:

```json
{"prediction": 1, "label": "heart_disease", "probability_disease": 0.8473, "confidence": 0.8473,
 "risk_band": "high", "model_version": "logreg:2026.10.08-1744", "request_id": "4e611de270f8"}
```

Swagger UI: <http://localhost:8000/docs>

### Local monitoring stack

```bash
docker compose up -d --build                 # API :8000, Prometheus :9090, Grafana :3000 (admin/admin)
python scripts/load_test.py --n 300 --rps 5  # generate traffic
```

## 7. Kubernetes deployment (Minikube)

```bash
./scripts/deploy_minikube.sh                 # build, load image, kubectl apply -k k8s/, smoke test
```

Access options (keep `minikube tunnel` running in a second terminal for the first two):

| Method | How |
|--------|-----|
| LoadBalancer | `kubectl -n heart-mlops get svc heart-api` → EXTERNAL-IP `127.0.0.1` → `curl http://127.0.0.1:8080/health` |
| Ingress | `echo "127.0.0.1 heart-api.local" \| sudo tee -a /etc/hosts` (once) → `curl http://heart-api.local/health` |
| Port-forward | `kubectl -n heart-mlops port-forward svc/heart-api 18080:8080` → `http://localhost:18080` |
| Prometheus | `kubectl -n heart-mlops port-forward svc/prometheus 9090:9090` → <http://localhost:9090> |
| Grafana | `kubectl -n heart-mlops port-forward svc/grafana 3000:3000` → <http://localhost:3000> (dashboard *MLOps → Heart Disease API*) |

On Linux with the docker driver, use `$(minikube ip)` instead of `127.0.0.1` in `/etc/hosts`.

Tear down: `make k8s-delete`.

## 8. CI/CD

`.github/workflows/ci.yml` runs on every push / PR:

1. **lint** – flake8 (report uploaded)
2. **k8s-validate** – `kustomize build` + `kubeconform` schema check
3. **test** – pytest with JUnit + coverage XML artifacts
4. **train** – download data → EDA → train (quick grid on PRs, full on `main`) → **quality gate** (CV ROC-AUC ≥ 0.85 and mean CV recall ≥ 0.80 at threshold 0.4) → model + MLflow store + reports uploaded; comparison table in the job summary
5. **docker** – build image with the freshly trained model, run it, smoke test, upload container logs, push to GHCR on `main`

Any failing lint rule, test, quality gate or smoke check fails the workflow.

## 9. Repository layout

```
├── .github/workflows/ci.yml      CI/CD pipeline
├── api/                          FastAPI app + pydantic schemas
├── data/raw, data/processed      raw download + cleaned dataset
├── docs/architecture.(dot|png)   architecture diagram
├── k8s/                          Kustomize: namespace, deployment, service, ingress, hpa, monitoring/
├── models/                       packaged pipeline + metadata
├── monitoring/                   Prometheus config + Grafana provisioning/dashboard (docker compose)
├── notebooks/                    01_eda, 02_training, 03_inference
├── reports/                      REPORT.md, .docx report, figures, model comparison
├── screenshots/                  evidence for the report (see checklist)
├── scripts/                      download, smoke test, load test, minikube deploy, samples
├── src/heart/                    config, data, features, eda, train, predict
├── tests/                        pytest suite
├── Dockerfile, docker-compose.yml, Makefile
└── requirements.txt / requirements-serve.txt / requirements-dev.txt / environment.yml
```

## 10. API input reference

| Field | Meaning | Allowed |
|-------|---------|---------|
| age | years | 1–120 |
| sex | 1 male, 0 female | 0, 1 |
| cp | chest pain type | 1 typical, 2 atypical, 3 non-anginal, 4 asymptomatic |
| trestbps | resting BP (mm Hg) | 50–250 |
| chol | serum cholesterol (mg/dl) | 80–700 |
| fbs | fasting blood sugar > 120 | 0, 1 |
| restecg | resting ECG | 0, 1, 2 |
| thalach | max heart rate | 50–250 |
| exang | exercise-induced angina | 0, 1 |
| oldpeak | ST depression | 0–10 |
| slope | peak ST slope | 1, 2, 3 |
| ca *(optional)* | vessels coloured by fluoroscopy | 0–3 |
| thal *(optional)* | thalassemia | 3 normal, 6 fixed, 7 reversible |

*Disclaimer: educational project, not a medical device.*
