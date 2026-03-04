# IDS 568 — Milestone 3: Workflow Automation & Experiment Tracking

> **Wine Quality Classification Pipeline**  
> Airflow orchestration · MLflow experiment tracking · GitHub Actions CI/CD

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Repository Structure](#repository-structure)
3. [Setup Instructions](#setup-instructions)
4. [Running the Pipeline](#running-the-pipeline)
5. [CI/CD Workflow](#cicd-workflow)
6. [Experiment Tracking Methodology](#experiment-tracking-methodology)
7. [DAG Idempotency & Lineage Guarantees](#dag-idempotency--lineage-guarantees)
8. [CI-Based Model Governance](#ci-based-model-governance)
9. [Operational Notes](#operational-notes)
10. [Rollback Procedures](#rollback-procedures)

---

## Architecture Overview

```
                    ┌─────────────────────────────────────┐
                    │           Airflow DAG                │
                    │   train_pipeline (daily schedule)    │
                    └─────────────────────────────────────┘
                              │
              ┌───────────────┼──────────────────┐
              ▼               ▼                  ▼
     preprocess_data  →  train_model  →  register_model
     (idempotent)       (MLflow run)    (quality gate +
                                         registry promotion)

                    ┌─────────────────────────────────────┐
                    │         GitHub Actions CI            │
                    │   push → preprocess → train →        │
                    │   quality gate → register (main)     │
                    └─────────────────────────────────────┘

                    ┌─────────────────────────────────────┐
                    │         MLflow Tracking              │
                    │  Experiments / Runs / Registry       │
                    │  wine_quality_classification         │
                    │  Model: wine_quality_rf              │
                    └─────────────────────────────────────┘
```

**Dataset:** Sklearn Wine (178 samples, 13 features, 3 classes)  
**Model:** `RandomForestClassifier` (scikit-learn)  
**Orchestration:** Apache Airflow (local executor)  
**Tracking:** MLflow (local `mlruns/`)  
**CI/CD:** GitHub Actions

---

## Repository Structure

```
ids568-milestone3-[netid]/
├── .github/
│   └── workflows/
│       └── train_and_validate.yml   # CI/CD pipeline
├── dags/
│   └── train_pipeline.py            # Airflow DAG
├── data/                            # Created at runtime (gitignored)
├── mlruns/                          # MLflow tracking (screenshots in docs/)
├── models/                          # Serialized models (created at runtime)
├── preprocess.py                    # Data cleaning & feature engineering
├── train.py                         # Model training + MLflow logging
├── model_validation.py              # Threshold-based quality gate
├── register_model.py                # MLflow registry promotion
├── run_experiments.py               # Run 5 experiments for lineage report
├── requirements.txt                 # Pinned dependencies
├── lineage_report.md                # Experiment analysis & production justification
└── README.md                        # This file
```

---

## Setup Instructions

### Prerequisites

- Python 3.11+
- Git

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/ids568-milestone3-<netid>.git
cd ids568-milestone3-<netid>
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install core dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Apache Airflow

Airflow has complex dependency requirements and must be installed separately:

```bash
AIRFLOW_VERSION=2.9.1
PYTHON_VERSION="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"
```

### 5. Configure Airflow

```bash
export AIRFLOW_HOME=$(pwd)/.airflow
export AIRFLOW__CORE__DAGS_FOLDER=$(pwd)/dags
export AIRFLOW__CORE__LOAD_EXAMPLES=False

airflow db init

airflow users create \
  --username admin --password admin \
  --firstname Admin --lastname User \
  --role Admin --email admin@example.com
```

### 6. (Optional) Start MLflow UI

```bash
mlflow server --host 0.0.0.0 --port 5000
# Visit http://localhost:5000
```

---

## Running the Pipeline

### Option A: Run standalone scripts (quickest)

```bash
# 1. Preprocess
python preprocess.py

# 2. Run all 5 experiments
python run_experiments.py

# 3. Validate best run (replace RUN_ID with output from step 2)
python model_validation.py --run-id <RUN_ID>

# 4. Register to MLflow registry
python register_model.py --run-id <RUN_ID>
```

### Option B: Run via Airflow DAG

```bash
# Start services (two terminals)
airflow webserver --port 8080   # Terminal 1
airflow scheduler               # Terminal 2

# Trigger DAG manually
airflow dags trigger train_pipeline

# Or visit http://localhost:8080 and trigger from the UI
```

### Option C: GitHub Actions

Push to `main` or `develop` to trigger the CI pipeline automatically.  
The workflow is defined in `.github/workflows/train_and_validate.yml`.

---

## CI/CD Workflow

The GitHub Actions pipeline runs on every push to `main`/`develop`:

```
Checkout → Install deps → Preprocess → Train → Export metrics
    → Quality gate (model_validation.py) → Register (main only)
    → Upload artifacts
```

**Quality gate thresholds** (configurable via GitHub Secrets / env vars):

| Metric | Threshold |
|--------|-----------|
| accuracy | ≥ 0.85 |
| f1_macro | ≥ 0.84 |
| roc_auc_ovr | ≥ 0.97 |

The pipeline **fails** (non-zero exit) if any metric falls below its threshold,
preventing regression models from reaching the registry.

---

## Experiment Tracking Methodology

All experiments are tracked via MLflow with:

**Parameters logged per run:**
- `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`
- `data_version` (YYYYMMDD stamp for traceability)
- `n_train`, `n_test`, `n_features`

**Metrics logged per run:**
- `accuracy`, `f1_macro`, `precision_macro`, `recall_macro`, `roc_auc_ovr`

**Artifacts logged per run:**
- Serialized sklearn model (via `mlflow.sklearn.log_model`)
- `data/meta.json` — preprocessing metadata

**Tags logged per run:**
- `model_hash` — SHA-256 of `.pkl` file
- `train_data_hash`, `test_data_hash` — SHA-256 of parquet files

### Viewing experiments

```bash
mlflow ui --port 5000
# Or if using a remote server:
export MLFLOW_TRACKING_URI=http://your-server:5000
```

---

## DAG Idempotency & Lineage Guarantees

### Idempotency

Each task is designed to be **safely re-runnable**:

- **`preprocess_data`**: Checks if `data/<DATA_VERSION>/train.parquet` exists.
  If so, skips preprocessing and reads existing metadata. Re-runs on the same
  date produce identical outputs.

- **`train_model`**: Creates a new MLflow run each time (this is intentional —
  MLflow runs are immutable records). The run name encodes version + params to
  make duplicates identifiable.

- **`register_model`**: Registers a new model version and archives the
  previous Production version. Re-runs produce a new registry version,
  not a destructive overwrite.

### Lineage Guarantees

Every registered model can be traced to:
1. **Exact code version** — MLflow logs git commit hash (via `mlflow.set_tag`)
2. **Exact data version** — `data_version` parameter + SHA-256 hash of parquet files
3. **Exact hyperparameters** — all params logged, not just tuned ones
4. **Preprocessing config** — `data/meta.json` stored as MLflow artifact

---

## CI-Based Model Governance

Model promotion to Production only happens when:

1. **CI passes** — all tests and checks in the GitHub Actions workflow succeed
2. **Quality gate passes** — `model_validation.py` confirms all metrics meet thresholds
3. **Manual review** (recommended) — team reviews MLflow run before merging to `main`

This prevents:
- Regressions from accidentally reaching Production
- Untested or undocumented models in the registry
- Manual, undocumented model promotions

---

## Operational Notes

### Retry Strategies

The Airflow DAG is configured with:
```python
'retries': 2,
'retry_delay': timedelta(minutes=5),
```

- `preprocess_data` is safe to retry (idempotent)
- `train_model` retries will create duplicate MLflow runs — acceptable; identify
  them by run name and discard earlier attempts
- `register_model` retries are safe; the quality gate re-runs each time

### Failure Handling

`on_failure_callback` in `dags/train_pipeline.py` logs a structured error.
To enable Slack / PagerDuty alerts, add your webhook URL as an Airflow
connection and uncomment the `requests.post` call in the callback.

### Monitoring Recommendations

| Signal | Alert Threshold | Action |
|--------|-----------------|--------|
| Model accuracy (7d rolling) | < 0.90 | Trigger retraining DAG |
| Prediction latency p99 | > 200ms | Profile model; consider compression |
| Feature drift (PSI) | > 0.2 on any feature | Investigate data pipeline |
| CI pipeline failure rate | > 20% over 5 runs | Review threshold calibration |

### Environment Reproducibility

All dependencies are pinned in `requirements.txt`. To reproduce the exact
environment:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Rollback Procedures

### Roll back to previous Production model

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
MODEL_NAME = "wine_quality_rf"

# List all versions
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
for v in versions:
    print(f"v{v.version} — {v.current_stage} — {v.description[:60]}")

# Promote a previous Archived version back to Production
client.transition_model_version_stage(
    name=MODEL_NAME,
    version="<PREVIOUS_VERSION>",
    stage="Production",
    archive_existing_versions=True,  # archives current Production
)
```

### Roll back via Git

1. Identify the last known good commit: `git log --oneline`
2. Create a hotfix branch: `git checkout -b hotfix/<issue> <good_commit>`
3. Push and open a PR — CI will re-validate before merge

### Emergency: disable model in serving layer

If the model is served via a REST endpoint, set the serving URL to a
fallback model or return a default prediction while the issue is
investigated. Document the incident in a post-mortem.

---

*IDS 568 MLOps — Milestone 3*