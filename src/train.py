"""
Training entry-point: load config → data → preprocess → train all models → save.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml

# Allow running as script from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import load_dataset, split_if_needed
from src.preprocessing import preprocess
from src.models import (
    build_random_forest,
    build_xgboost,
    Autoencoder,
    train_autoencoder,
    compute_ae_threshold,
    save_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(config_path: str = "configs/config.yaml") -> None:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config.get("training", {}).get("random_state", 42)
    set_seed(seed)

    models_dir = Path(config["training"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    train_df, test_df = load_dataset(config)
    train_df, test_df = split_if_needed(
        train_df, test_df,
        test_size=config["data"].get("test_size", 0.2),
        random_state=seed,
    )

    # 2. Preprocess
    X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess(
        train_df, test_df, config
    )
    # Persist preprocessor and feature names for evaluate.py
    joblib.dump(preprocessor, models_dir / "preprocessor.joblib")
    joblib.dump(feature_names, models_dir / "feature_names.joblib")
    # Also save the split arrays so evaluate can reuse them without reloading raw data
    np.savez(
        models_dir / "arrays.npz",
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
    )

    input_dim = X_train.shape[1]
    logger.info("Feature matrix shape: train %s, test %s", X_train.shape, X_test.shape)

    metrics_summary = {}

    # 3a. Random Forest
    logger.info("=== Training Random Forest ===")
    rf_params = config["models"]["random_forest"]
    rf = build_random_forest(rf_params, random_state=seed)
    rf.fit(X_train, y_train)
    save_model(rf, models_dir / "random_forest.joblib")
    metrics_summary["random_forest"] = {"status": "trained"}

    # 3b. XGBoost
    logger.info("=== Training XGBoost ===")
    xgb_params = config["models"]["xgboost"]
    xgb = build_xgboost(xgb_params, random_state=seed)
    xgb.fit(X_train, y_train)
    save_model(xgb, models_dir / "xgboost.joblib")
    metrics_summary["xgboost"] = {"status": "trained"}

    # 3c. Autoencoder (train only on normal samples)
    logger.info("=== Training Autoencoder ===")
    ae_params = config["models"]["autoencoder"]
    X_normal = X_train[y_train == 0]
    logger.info("Normal samples for AE: %d", len(X_normal))
    ae = Autoencoder(
        input_dim=input_dim,
        encoding_dim=ae_params.get("encoding_dim", 32),
        hidden_dims=ae_params.get("hidden_dims", [64, 32]),
    )
    ae = train_autoencoder(
        ae,
        X_normal,
        epochs=ae_params.get("epochs", 30),
        batch_size=ae_params.get("batch_size", 256),
        lr=ae_params.get("learning_rate", 1e-3),
    )
    threshold = compute_ae_threshold(
        ae, X_normal, percentile=ae_params.get("threshold_percentile", 95)
    )
    save_model(ae, models_dir / "autoencoder.pt")
    # Save threshold and architecture info
    ae_meta = {
        "threshold": threshold,
        "input_dim": input_dim,
        "encoding_dim": ae_params.get("encoding_dim", 32),
        "hidden_dims": ae_params.get("hidden_dims", [64, 32]),
    }
    with open(models_dir / "autoencoder_meta.json", "w") as f:
        json.dump(ae_meta, f, indent=2)
    metrics_summary["autoencoder"] = {"status": "trained", "threshold": threshold}

    # 4. Write brief training log
    log_path = Path(config["training"]["metrics_path"]).parent / "train_log.json"
    with open(log_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    logger.info("Training finished. Models saved to %s", models_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train anomaly detection models")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
