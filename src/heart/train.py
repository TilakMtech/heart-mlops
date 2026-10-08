"""Model training with hyper-parameter search, cross-validation and MLflow tracking.

Every candidate model is tuned with GridSearchCV (stratified K-fold, ROC-AUC)
inside its own nested MLflow run. Params, CV metrics, hold-out metrics, plots
and the fitted pipeline are logged. The best candidate (highest mean CV ROC-AUC)
is exported to models/model.joblib together with models/model_meta.json.

    python -m heart.train                 # full search
    python -m heart.train --quick         # tiny grids (CI smoke run)
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (  # noqa: E402
    GridSearchCV, StratifiedKFold, cross_validate, train_test_split,
)

from heart import config as C  # noqa: E402
from heart.data import build_clean_dataset, split_xy  # noqa: E402
from heart.features import build_pipeline, transformed_feature_names  # noqa: E402

log = logging.getLogger("train")
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]


@dataclass
class Candidate:
    name: str
    estimator: object
    grid: dict
    quick_grid: dict
    use_engineered: bool = True
    notes: str = ""
    result: dict = field(default_factory=dict)


def candidates() -> list[Candidate]:
    rs = C.RANDOM_STATE
    return [
        Candidate(
            "logreg_baseline",
            LogisticRegression(max_iter=2000, solver="liblinear", random_state=rs),
            {"model__C": [0.01, 0.1, 1.0, 10.0], "model__penalty": ["l1", "l2"]},
            {"model__C": [1.0]},
            use_engineered=False,
            notes="Ablation: raw features only, no engineered features",
        ),
        Candidate(
            "logreg",
            LogisticRegression(max_iter=2000, solver="liblinear", random_state=rs),
            {"model__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
             "model__penalty": ["l1", "l2"],
             "model__class_weight": [None, "balanced"]},
            {"model__C": [0.1, 1.0]},
        ),
        Candidate(
            "random_forest",
            RandomForestClassifier(random_state=rs, n_jobs=-1),
            {"model__n_estimators": [200, 400],
             "model__max_depth": [None, 4, 6],
             "model__min_samples_leaf": [1, 3, 5],
             "model__max_features": ["sqrt", 0.5]},
            {"model__n_estimators": [100], "model__max_depth": [4]},
        ),
        Candidate(
            "gradient_boosting",
            GradientBoostingClassifier(random_state=rs),
            {"model__n_estimators": [100, 200],
             "model__learning_rate": [0.03, 0.1],
             "model__max_depth": [2, 3],
             "model__subsample": [0.8, 1.0]},
            {"model__n_estimators": [50]},
        ),
    ]


# --------------------------------------------------------------------------- plots
def _plot_eval(pipe, X_test, y_test, name: str, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test, ax=axes[0], cmap="Blues",
                                          display_labels=["No disease", "Disease"])
    axes[0].set_title("Confusion matrix (hold-out)")
    RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=axes[1], name=name)
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set_title("ROC curve")
    PrecisionRecallDisplay.from_estimator(pipe, X_test, y_test, ax=axes[2], name=name)
    axes[2].set_title("Precision-Recall curve")
    fig.tight_layout()
    p = out / f"{name}_evaluation.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)

    imp = _importances(pipe)
    if imp is not None:
        top = imp.head(15)[::-1]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(top.index, top.values, color="#e76f51")
        ax.set_title(f"{name}: top feature importances")
        fig.tight_layout()
        p = out / f"{name}_feature_importance.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(p)
    return paths


def _importances(pipe) -> pd.Series | None:
    model = pipe.named_steps["model"]
    names = transformed_feature_names(pipe)
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
    elif hasattr(model, "coef_"):
        vals = np.abs(model.coef_[0])
    else:
        return None
    return pd.Series(vals, index=names).sort_values(ascending=False)


def _holdout_metrics(pipe, X_test, y_test) -> dict:
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "test_accuracy": accuracy_score(y_test, pred),
        "test_precision": precision_score(y_test, pred, zero_division=0),
        "test_recall": recall_score(y_test, pred),
        "test_f1": f1_score(y_test, pred),
        "test_roc_auc": roc_auc_score(y_test, proba),
    }


# --------------------------------------------------------------------------- main
def run(quick: bool = False) -> dict:
    df = build_clean_dataset()
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=C.TEST_SIZE, stratify=y, random_state=C.RANDOM_STATE)
    cv = StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)

    mlflow.set_tracking_uri(C.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(C.MLFLOW_EXPERIMENT)
    plots_dir = C.REPORTS_DIR / "figures" / "models"
    results: list[Candidate] = []

    with mlflow.start_run(run_name=f"session-{time.strftime('%Y%m%d-%H%M%S')}") as parent:
        mlflow.set_tags({"stage": "model-selection", "quick": str(quick),
                         "python": platform.python_version(), "sklearn": sklearn.__version__})
        mlflow.log_params({"n_rows": len(df), "n_train": len(X_train), "n_test": len(X_test),
                           "cv_folds": C.CV_FOLDS, "random_state": C.RANDOM_STATE})
        mlflow.log_artifact(str(C.CLEAN_CSV), artifact_path="data")

        for cand in candidates():
            with mlflow.start_run(run_name=cand.name, nested=True):
                log.info("== %s ==", cand.name)
                pipe = build_pipeline(cand.estimator, cand.use_engineered)
                grid = cand.quick_grid if quick else cand.grid
                t0 = time.time()
                search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)
                search.fit(X_train, y_train)
                best = search.best_estimator_

                cvres = cross_validate(best, X_train, y_train, cv=cv, scoring=SCORING)
                cv_metrics = {}
                for m in SCORING:
                    cv_metrics[f"cv_{m}_mean"] = float(np.mean(cvres[f"test_{m}"]))
                    cv_metrics[f"cv_{m}_std"] = float(np.std(cvres[f"test_{m}"]))
                test_metrics = _holdout_metrics(best, X_test, y_test)

                mlflow.set_tags({"model_family": cand.name, "notes": cand.notes,
                                 "engineered_features": str(cand.use_engineered)})
                mlflow.log_params({k.replace("model__", ""): v
                                   for k, v in search.best_params_.items()})
                mlflow.log_param("grid_size", len(search.cv_results_["params"]))
                mlflow.log_metrics({**cv_metrics, **test_metrics,
                                    "search_seconds": time.time() - t0})

                cv_table = plots_dir / f"{cand.name}_grid_results.csv"
                cv_table.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(search.cv_results_).sort_values("rank_test_score").to_csv(
                    cv_table, index=False)
                mlflow.log_artifact(str(cv_table), artifact_path="search")
                for p in _plot_eval(best, X_test, y_test, cand.name, plots_dir):
                    mlflow.log_artifact(str(p), artifact_path="plots")

                signature = infer_signature(X_train, best.predict_proba(X_train)[:, 1])
                mlflow.sklearn.log_model(best, artifact_path="model", signature=signature,
                                         input_example=X_train.head(3))
                cand.result = {"best_params": search.best_params_, **cv_metrics,
                               **test_metrics, "run_id": mlflow.active_run().info.run_id}
                cand.estimator = best
                results.append(cand)
                log.info("%s cv_roc_auc=%.3f test_roc_auc=%.3f", cand.name,
                         cv_metrics["cv_roc_auc_mean"], test_metrics["test_roc_auc"])

        # ---------- selection
        ranked = sorted(results, key=lambda c: (c.result["cv_roc_auc_mean"],
                                                c.result["test_roc_auc"]), reverse=True)
        winner = ranked[0]
        rows = []
        for c in results:
            scores = {k: v for k, v in c.result.items() if k not in ("best_params", "run_id")}
            rows.append({"model": c.name, **scores})
        comparison = pd.DataFrame(rows).round(4)
        comp_path = C.REPORTS_DIR / "model_comparison.csv"
        comparison.to_csv(comp_path, index=False)
        mlflow.log_artifact(str(comp_path))
        roc_path = _roc_overlay(results, X_test, y_test, plots_dir)
        mlflow.log_artifact(str(roc_path), artifact_path="plots")
        mlflow.set_tag("selected_model", winner.name)
        mlflow.log_metric("selected_cv_roc_auc", winner.result["cv_roc_auc_mean"])

        meta = _export(winner, X_train, parent.info.run_id)
        mlflow.log_artifact(str(C.MODEL_META_PATH))
        log.info("Selected %s -> %s", winner.name, C.MODEL_PATH)
        print(comparison.to_string(index=False))
        return meta


def _roc_overlay(results, X_test, y_test, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    for c in results:
        RocCurveDisplay.from_estimator(c.estimator, X_test, y_test, ax=ax, name=c.name)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title("Hold-out ROC: all candidates")
    fig.tight_layout()
    p = out / "roc_all_models.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def _export(winner: Candidate, X_train: pd.DataFrame, parent_run_id: str) -> dict:
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(winner.estimator, C.MODELS_DIR / "model.joblib")
    meta = {
        "model_name": winner.name,
        "version": time.strftime("%Y.%m.%d-%H%M"),
        "mlflow_parent_run": parent_run_id,
        "mlflow_run_id": winner.result["run_id"],
        "best_params": {k.replace("model__", ""): v for k, v in winner.result["best_params"].items()},
        "metrics": {k: round(v, 4) for k, v in winner.result.items() if isinstance(v, float)},
        "input_features": list(X_train.columns),
        "threshold": 0.5,
        "sklearn_version": sklearn.__version__,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    C.MODEL_META_PATH.write_text(json.dumps(meta, indent=2, default=str))
    try:
        mlflow.register_model(f"runs:/{winner.result['run_id']}/model", "heart-disease-classifier")
    except Exception as exc:  # registry not available on plain file store
        log.info("Model registry skipped: %s", exc)
    return meta


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny grids for CI")
    meta = run(quick=ap.parse_args().quick)
    print(json.dumps(meta["metrics"], indent=2))
