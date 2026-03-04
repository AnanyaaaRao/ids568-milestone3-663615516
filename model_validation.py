
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Quality thresholds — adjust here or via env vars
# ------------------------------------------------------------------
THRESHOLDS = {
    "accuracy":         float(os.getenv("MIN_ACCURACY",        "0.85")),
    "f1_macro":         float(os.getenv("MIN_F1_MACRO",        "0.84")),
    "precision_macro":  float(os.getenv("MIN_PRECISION_MACRO", "0.84")),
    "recall_macro":     float(os.getenv("MIN_RECALL_MACRO",    "0.84")),
    "roc_auc_ovr":      float(os.getenv("MIN_ROC_AUC",         "0.97")),
}

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")


def validate_metrics(metrics: dict) -> tuple[bool, list[str]]:
    """
    Compare metrics dict against THRESHOLDS.

    Returns (passed: bool, failures: list[str]).
    """
    failures = []
    for metric, threshold in THRESHOLDS.items():
        value = metrics.get(metric)
        if value is None:
            failures.append(f"  MISSING metric '{metric}' (required threshold: {threshold})")
            continue
        if value < threshold:
            failures.append(
                f"  FAIL  '{metric}': {value:.4f} < threshold {threshold:.4f}"
            )
        else:
            log.info("  PASS  '%s': %.4f >= %.4f", metric, value, threshold)
    return len(failures) == 0, failures


def validate_run(run_id: str) -> bool:
    """Fetch metrics from MLflow and validate."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    metrics = run.data.metrics
    log.info("Validating MLflow run: %s", run_id)
    passed, failures = validate_metrics(metrics)
    _report(passed, failures)
    return passed


def validate_file(metrics_file: str) -> bool:
    """Load metrics from a JSON file and validate."""
    with open(metrics_file) as f:
        metrics = json.load(f)
    log.info("Validating metrics from file: %s", metrics_file)
    passed, failures = validate_metrics(metrics)
    _report(passed, failures)
    return passed


def _report(passed: bool, failures: list[str]) -> None:
    print("\n" + "=" * 60)
    print("MODEL VALIDATION REPORT")
    print("=" * 60)
    print(f"Thresholds: {json.dumps(THRESHOLDS, indent=2)}")
    print("-" * 60)
    if passed:
        print("✅  ALL CHECKS PASSED — model meets quality gate")
    else:
        print("❌  VALIDATION FAILED — model does NOT meet quality gate")
        for f in failures:
            print(f)
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Validate model quality thresholds")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-id", help="MLflow run ID to validate")
    group.add_argument("--metrics-file", help="JSON file with metrics dict")
    args = parser.parse_args()

    if args.run_id:
        passed = validate_run(args.run_id)
    else:
        passed = validate_file(args.metrics_file)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
