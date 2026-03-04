
import os
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Paths (override via env vars so Airflow / CI can inject their own)
# ------------------------------------------------------------------
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
DATA_VERSION = os.getenv("DATA_VERSION", datetime.utcnow().strftime("%Y%m%d"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def preprocess(data_version: str = DATA_VERSION) -> dict:
    """
    Load, clean, scale, and split the Wine dataset.

    Returns a metadata dict with paths and hashes so callers (Airflow tasks,
    CI scripts) can log lineage to MLflow.
    """
    out_dir = DATA_DIR / data_version
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load raw data ---------------------------------------------------
    log.info("Loading Wine dataset...")
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df["target"] = wine.target

    # ---- Basic cleaning --------------------------------------------------
    before = len(df)
    df = df.drop_duplicates()
    df = df.dropna()
    log.info("Rows after dedup/dropna: %d → %d", before, len(df))

    # ---- Feature engineering ---------------------------------------------
    # Log-transform two skewed features
    for col in ["proline", "od280/od315_of_diluted_wines"]:
        if col in df.columns:
            df[f"log_{col.replace('/', '_')}"] = np.log1p(df[col])

    # ---- Train / test split ----------------------------------------------
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---- Scaling ---------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns
    )

    # ---- Persist ---------------------------------------------------------
    train_path = out_dir / "train.parquet"
    test_path = out_dir / "test.parquet"
    meta_path = out_dir / "meta.json"

    train_df = X_train_scaled.copy()
    train_df["target"] = y_train.values
    test_df = X_test_scaled.copy()
    test_df["target"] = y_test.values

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    metadata = {
        "data_version": data_version,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "n_features": len(X.columns),
        "feature_names": list(X.columns),
        "train_path": str(train_path),
        "test_path": str(test_path),
        "train_hash": _sha256(train_path),
        "test_hash": _sha256(test_path),
        "created_at": datetime.utcnow().isoformat(),
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    log.info("Preprocessing complete. Data version: %s", data_version)
    log.info("  train: %s  (%s)", train_path, metadata["train_hash"][:12])
    log.info("  test:  %s  (%s)", test_path, metadata["test_hash"][:12])
    return metadata


if __name__ == "__main__":
    meta = preprocess()
    print(json.dumps(meta, indent=2))
