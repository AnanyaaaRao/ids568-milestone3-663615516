
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

os.environ.setdefault("MLFLOW_TRACKING_URI", "mlruns")
os.environ.setdefault("DATA_DIR", "data")

from preprocess import preprocess
from train import train

# ------------------------------------------------------------------
# Experiment grid (5 runs covering different parts of the space)
# ------------------------------------------------------------------
EXPERIMENTS = [
    {
        "run_name": "baseline_100trees",
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "notes": "Baseline configuration with unlimited depth",
    },
    {
        "run_name": "shallow_50trees",
        "n_estimators": 50,
        "max_depth": 5,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "notes": "Shallow trees — fast, less prone to overfitting",
    },
    {
        "run_name": "deep_200trees",
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "notes": "More trees with moderate depth — production candidate",
    },
    {
        "run_name": "regularized_150trees",
        "n_estimators": 150,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "log2",
        "notes": "Regularized via min_samples constraints and log2 features",
    },
    {
        "run_name": "large_300trees",
        "n_estimators": 300,
        "max_depth": 20,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "notes": "Maximum capacity model — benchmark ceiling",
    },
]


def main():
    # Step 1: Preprocess data (idempotent)
    log.info("=" * 60)
    log.info("Step 1: Preprocessing data")
    log.info("=" * 60)
    data_meta = preprocess()
    data_version = data_meta["data_version"]
    log.info("Data version: %s", data_version)

    # Step 2: Run all experiments
    results = []
    for i, exp in enumerate(EXPERIMENTS, 1):
        log.info("=" * 60)
        log.info("Experiment %d/%d: %s", i, len(EXPERIMENTS), exp["run_name"])
        log.info("=" * 60)

        notes = exp.pop("notes")
        result = train(data_version=data_version, **exp)
        result["notes"] = notes
        results.append(result)

        log.info(
            "  accuracy=%.4f  f1=%.4f  roc_auc=%.4f",
            result["metrics"]["accuracy"],
            result["metrics"]["f1_macro"],
            result["metrics"]["roc_auc_ovr"],
        )

    # Step 3: Print summary table
    print("\n" + "=" * 100)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 100)
    header = f"{'#':<3} {'Run Name':<28} {'n_est':<7} {'max_d':<7} {'Accuracy':<10} {'F1 Macro':<10} {'ROC-AUC':<10} {'Run ID':<12}"
    print(header)
    print("-" * 100)
    for i, r in enumerate(results, 1):
        p = r["params"]
        m = r["metrics"]
        print(
            f"{i:<3} {r['run_name']:<28} "
            f"{p['n_estimators']:<7} "
            f"{str(p['max_depth']):<7} "
            f"{m['accuracy']:<10.4f} "
            f"{m['f1_macro']:<10.4f} "
            f"{m['roc_auc_ovr']:<10.4f} "
            f"{r['run_id'][:10]}"
        )
    print("=" * 100)

    # Find best by accuracy
    best = max(results, key=lambda r: r["metrics"]["accuracy"])
    print(f"\n🏆 Best run: '{best['run_name']}' — accuracy={best['metrics']['accuracy']:.4f}")
    print(f"   Run ID: {best['run_id']}")

    # Save summary JSON
    summary_path = Path("experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Summary saved to %s", summary_path)

    return results


if __name__ == "__main__":
    main()
