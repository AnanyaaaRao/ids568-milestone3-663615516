
import argparse
import logging
import os
import sys
import time

import mlflow
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "wine_quality_rf")


def register_and_promote(run_id: str, model_name: str = DEFAULT_MODEL_NAME) -> dict:
    """
    1. Register the model artifact from a given run.
    2. Transition: None → Staging.
    3. Archive any existing Production version.
    4. Transition: Staging → Production.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # ---- Step 1: Register ------------------------------------------------
    model_uri = f"runs:/{run_id}/model"
    log.info("Registering model from run %s as '%s'...", run_id, model_name)

    registered = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = registered.version
    log.info("Registered as version %s", version)

    # Wait for the model to be ready
    _wait_for_model_ready(client, model_name, version)

    # ---- Step 2: None → Staging ------------------------------------------
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Staging",
        archive_existing_versions=False,
    )
    log.info("Version %s → Staging", version)

    # Add description + tags
    run = client.get_run(run_id)
    accuracy = run.data.metrics.get("accuracy", 0)
    f1 = run.data.metrics.get("f1_macro", 0)

    client.update_model_version(
        name=model_name,
        version=version,
        description=(
            f"RandomForest trained on Wine dataset. "
            f"accuracy={accuracy:.4f}, f1_macro={f1:.4f}. "
            f"Source run: {run_id}"
        ),
    )
    client.set_model_version_tag(model_name, version, "validated", "true")
    client.set_model_version_tag(model_name, version, "run_id", run_id)
    client.set_model_version_tag(
        model_name, version, "model_hash", run.data.tags.get("model_hash", "unknown")
    )

    # ---- Step 3: Archive existing Production versions --------------------
    for mv in client.search_model_versions(f"name='{model_name}'"):
        if mv.current_stage == "Production" and mv.version != version:
            client.transition_model_version_stage(
                name=model_name, version=mv.version, stage="Archived"
            )
            log.info("Archived previous Production version %s", mv.version)

    # ---- Step 4: Staging → Production ------------------------------------
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=False,
    )
    log.info("Version %s → Production ✅", version)

    return {
        "model_name": model_name,
        "version": version,
        "stage": "Production",
        "run_id": run_id,
        "accuracy": accuracy,
        "f1_macro": f1,
    }


def _wait_for_model_ready(client, model_name, version, timeout=60):
    """Poll until model version reaches READY state."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        mv = client.get_model_version(model_name, version)
        if mv.status == "READY":
            return
        log.info("Waiting for model to be READY (current: %s)...", mv.status)
        time.sleep(2)
    raise TimeoutError(f"Model version {version} not READY after {timeout}s")


def main():
    parser = argparse.ArgumentParser(description="Register and promote MLflow model")
    parser.add_argument("--run-id", required=True, help="MLflow run ID")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    args = parser.parse_args()

    result = register_and_promote(args.run_id, args.model_name)
    import json
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
