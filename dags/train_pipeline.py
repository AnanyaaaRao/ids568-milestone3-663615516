"""
dags/train_pipeline.py

Airflow DAG: preprocess_data → train_model → register_model

Design principles:
- Idempotent: each task is keyed by DATA_VERSION (date-based default).
  Re-running on the same date produces the same artifacts.
- Retry / failure handling via default_args.
- All cross-task data passed through XCom (small metadata dicts only;
  large files stay on disk / MLflow).
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# ---- Make project root importable inside Airflow -------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

log = logging.getLogger(__name__)

# ---- Shared config (all tasks read from env so they stay idempotent) -----
DATA_VERSION = os.getenv("DATA_VERSION", datetime.utcnow().strftime("%Y%m%d"))
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", str(PROJECT_ROOT / "mlruns"))
MODEL_NAME = os.getenv("MODEL_NAME", "wine_quality_rf")

# ---- Default hyperparams (override via Airflow Variables or env) ----------
N_ESTIMATORS = int(os.getenv("N_ESTIMATORS", "150"))
MAX_DEPTH = os.getenv("MAX_DEPTH")  # None = unlimited
MAX_DEPTH = int(MAX_DEPTH) if MAX_DEPTH else None
MIN_SAMPLES_SPLIT = int(os.getenv("MIN_SAMPLES_SPLIT", "2"))
MIN_SAMPLES_LEAF = int(os.getenv("MIN_SAMPLES_LEAF", "1"))


# ==========================================================================
# Failure callback
# ==========================================================================

def on_failure_callback(context):
    """
    Called when any task fails after exhausting retries.
    In production, swap the log.error call for a Slack / PagerDuty alert.
    """
    dag_id = context["dag"].dag_id
    task_id = context["task_instance"].task_id
    execution_date = context["execution_date"]
    exception = context.get("exception", "Unknown error")

    log.error(
        "❌ PIPELINE FAILURE | DAG: %s | Task: %s | Date: %s | Error: %s",
        dag_id,
        task_id,
        execution_date,
        exception,
    )
    # Example Slack hook (uncomment and configure in production):
    # requests.post(os.getenv("SLACK_WEBHOOK_URL"), json={"text": f"..."})


# ==========================================================================
# Task functions
# ==========================================================================

def preprocess_data(**context):
    """
    Task 1 — Data cleaning and feature engineering.
    Idempotency: output is keyed by DATA_VERSION. If the versioned directory
    already exists and contains valid parquet files, the task is a no-op.
    """
    from preprocess import preprocess, DATA_DIR

    version_dir = DATA_DIR / DATA_VERSION
    train_file = version_dir / "train.parquet"
    test_file = version_dir / "test.parquet"

    if train_file.exists() and test_file.exists():
        log.info("Data version %s already exists — skipping preprocessing.", DATA_VERSION)
        meta_path = version_dir / "meta.json"
        with open(meta_path) as f:
            metadata = json.load(f)
    else:
        log.info("Running preprocessing for data version: %s", DATA_VERSION)
        metadata = preprocess(data_version=DATA_VERSION)

    # Push metadata to XCom so downstream tasks can read it
    context["ti"].xcom_push(key="data_meta", value=metadata)
    log.info("preprocess_data complete ✅  version=%s", DATA_VERSION)
    return metadata


def train_model(**context):
    """
    Task 2 — Model training with MLflow logging.
    Idempotency: MLflow run names encode the data version + hyperparameters.
    A re-run simply creates a new MLflow run (acceptable in MLflow).
    """
    import mlflow
    from train import train

    os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI

    # Pull data metadata from upstream task
    data_meta = context["ti"].xcom_pull(task_ids="preprocess_data", key="data_meta")
    dv = data_meta["data_version"] if data_meta else DATA_VERSION

    run_result = train(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features="sqrt",
        random_state=42,
        data_version=dv,
        run_name=f"airflow_{dv}_ne{N_ESTIMATORS}_md{MAX_DEPTH}",
    )

    context["ti"].xcom_push(key="train_result", value=run_result)
    log.info("train_model complete ✅  run_id=%s  accuracy=%.4f",
             run_result["run_id"], run_result["metrics"]["accuracy"])
    return run_result


def register_model(**context):
    """
    Task 3 — Validate model quality, then register to MLflow Model Registry.
    Idempotency: registers a new version each run; previous Production version
    is automatically archived.
    """
    import mlflow
    from model_validation import validate_metrics
    from register_model import register_and_promote

    os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI

    train_result = context["ti"].xcom_pull(task_ids="train_model", key="train_result")
    if not train_result:
        raise ValueError("No train_result found in XCom — did train_model run?")

    run_id = train_result["run_id"]
    metrics = train_result["metrics"]

    # Quality gate — raise if below thresholds so Airflow marks task FAILED
    passed, failures = validate_metrics(metrics)
    if not passed:
        raise ValueError(
            "Model failed quality gate — will NOT be registered.\n"
            + "\n".join(failures)
        )

    log.info("Quality gate passed ✅  — proceeding to registration")
    reg_result = register_and_promote(run_id=run_id, model_name=MODEL_NAME)

    context["ti"].xcom_push(key="registry_result", value=reg_result)
    log.info(
        "register_model complete ✅  model=%s  version=%s  stage=%s",
        reg_result["model_name"],
        reg_result["version"],
        reg_result["stage"],
    )
    return reg_result


# ==========================================================================
# DAG definition
# ==========================================================================

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "start_date": days_ago(1),
    "email_on_failure": False,   # set True and configure email in airflow.cfg
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": False,
    "on_failure_callback": on_failure_callback,
}

with DAG(
    dag_id="train_pipeline",
    default_args=default_args,
    description="Wine Quality ML pipeline: preprocess → train → register",
    schedule_interval="@daily",       # or None for manual-only triggers
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "wine", "training"],
) as dag:

    t_preprocess = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
        provide_context=True,
        doc_md="""
        **preprocess_data**
        Loads the Wine dataset, cleans it, applies StandardScaler, and saves
        versioned train/test parquet files.  Idempotent: skips if version exists.
        """,
    )

    t_train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
        provide_context=True,
        doc_md="""
        **train_model**
        Trains a RandomForestClassifier and logs params, metrics, and model
        artifact to MLflow.
        """,
    )

    t_register = PythonOperator(
        task_id="register_model",
        python_callable=register_model,
        provide_context=True,
        doc_md="""
        **register_model**
        Validates the model against quality thresholds, then registers and
        promotes to Production in the MLflow Model Registry.
        """,
    )

    # ---- Task dependencies (explicit ordering) ----------------------------
    t_preprocess >> t_train >> t_register
