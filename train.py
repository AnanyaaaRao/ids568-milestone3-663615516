
import argparse
import hashlib
import json
import logging
import os
import pickle
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Config (env vars allow Airflow / CI injection without code changes)
# ------------------------------------------------------------------
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
DATA_VERSION = os.getenv("DATA_VERSION", "")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "wine_quality_classification")
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _latest_data_version() -> str:
    """Return the most recently created data version directory."""
    versions = sorted(DATA_DIR.iterdir(), reverse=True) if DATA_DIR.exists() else []
    if not versions:
        raise FileNotFoundError(
            f"No data found in {DATA_DIR}. Run preprocess.py first."
        )
    return versions[0].name


def train(
    n_estimators: int = 100,
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: str = "sqrt",
    random_state: int = 42,
    data_version: str = "",
    run_name: str = "",
) -> dict:
    """Train model, log to MLflow, return run metadata."""

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # Resolve data version
    dv = data_version or DATA_VERSION or _latest_data_version()
    data_path = DATA_DIR / dv
    meta_path = data_path / "meta.json"

    with open(meta_path) as f:
        data_meta = json.load(f)

    train_df = pd.read_parquet(data_meta["train_path"])
    test_df = pd.read_parquet(data_meta["test_path"])

    X_train = train_df.drop("target", axis=1)
    y_train = train_df["target"]
    X_test = test_df.drop("target", axis=1)
    y_test = test_df["target"]

    with mlflow.start_run(run_name=run_name or f"rf_ne{n_estimators}_md{max_depth}") as run:

        # ---- Hyperparameters ------------------------------------------------
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth if max_depth is not None else "None",
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "random_state": random_state,
        }
        mlflow.log_params(params)

        # ---- Data lineage ---------------------------------------------------
        mlflow.log_param("data_version", dv)
        mlflow.log_param("n_train", data_meta["n_train"])
        mlflow.log_param("n_test", data_meta["n_test"])
        mlflow.log_param("n_features", data_meta["n_features"])
        mlflow.set_tag("train_data_hash", data_meta["train_hash"])
        mlflow.set_tag("test_data_hash", data_meta["test_hash"])

        # ---- Train ----------------------------------------------------------
        log.info("Training RandomForest with params: %s", params)
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)

        # ---- Evaluate -------------------------------------------------------
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
            "precision_macro": float(precision_score(y_test, y_pred, average="macro")),
            "recall_macro": float(recall_score(y_test, y_pred, average="macro")),
            "roc_auc_ovr": float(
                roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
            ),
        }
        mlflow.log_metrics(metrics)
        log.info("Metrics: %s", metrics)

        # ---- Persist model --------------------------------------------------
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_DIR / f"model_{run.info.run_id[:8]}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(clf, f)

        model_hash = _sha256(model_path)
        mlflow.set_tag("model_hash", model_hash)
        mlflow.set_tag("model_path", str(model_path))

        # Log model artifact (sklearn flavor — enables registry)
        mlflow.sklearn.log_model(clf, "model")

        # Also log the data meta as artifact for full lineage
        mlflow.log_artifact(str(meta_path), artifact_path="data_meta")

        run_id = run.info.run_id
        log.info("MLflow run_id: %s", run_id)

        result = {
            "run_id": run_id,
            "run_name": run.info.run_name,
            "experiment_id": run.info.experiment_id,
            "metrics": metrics,
            "params": params,
            "data_version": dv,
            "model_hash": model_hash,
        }

    return result


def parse_args():
    p = argparse.ArgumentParser(description="Train Wine Quality RandomForest")
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--max-depth", type=int, default=None)
    p.add_argument("--min-samples-split", type=int, default=2)
    p.add_argument("--min-samples-leaf", type=int, default=1)
    p.add_argument("--max-features", type=str, default="sqrt")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--data-version", type=str, default="")
    p.add_argument("--run-name", type=str, default="")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = train(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        random_state=args.random_state,
        data_version=args.data_version,
        run_name=args.run_name,
    )
    print(json.dumps(result, indent=2))
