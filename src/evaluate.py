"""
Evaluation: metrics, confusion matrices, ROC/PR curves, feature importance.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import load_autoencoder, autoencoder_predict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", font_scale=1.1)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> dict:
    """Compute precision, recall, F1, ROC-AUC, PR-AUC."""
    out = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_score is not None:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            out["roc_auc"] = None
        try:
            out["pr_auc"] = float(average_precision_score(y_true, y_score))
        except ValueError:
            out["pr_auc"] = None
    return out


def plot_confusion(y_true, y_pred, title: str, save_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", save_path)


def plot_roc_curves(results: dict, y_true: np.ndarray, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, data in results.items():
        if data.get("y_score") is None:
            continue
        fpr, tpr, _ = roc_curve(y_true, data["y_score"])
        auc = data["metrics"].get("roc_auc", 0)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", save_path)


def plot_pr_curves(results: dict, y_true: np.ndarray, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, data in results.items():
        if data.get("y_score") is None:
            continue
        precision, recall, _ = precision_recall_curve(y_true, data["y_score"])
        ap = data["metrics"].get("pr_auc", 0)
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", save_path)


def plot_feature_importance(model, feature_names: list, save_path: Path, top_k: int = 20) -> None:
    if not hasattr(model, "feature_importances_"):
        logger.warning("Model has no feature_importances_")
        return
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_k]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(idx)), importances[idx][::-1])
    ax.set_yticks(range(len(idx)))
    names = [feature_names[i] if i < len(feature_names) else f"f{i}" for i in idx[::-1]]
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Importance")
    ax.set_title(f"Top-{top_k} Feature Importances")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", save_path)


def plot_class_distribution(y_train: np.ndarray, y_test: np.ndarray, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, y, title in zip(axes, [y_train, y_test], ["Train", "Test"]):
        counts = np.bincount(y.astype(int))
        ax.bar(["Normal", "Attack"], counts, color=["steelblue", "coral"])
        ax.set_title(f"{title} set")
        ax.set_ylabel("Count")
        for i, v in enumerate(counts):
            ax.text(i, v + max(counts) * 0.01, str(v), ha="center")
    fig.suptitle("Class Distribution")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", save_path)


def main(config_path: str = "configs/config.yaml") -> None:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    models_dir = Path(config["training"]["models_dir"])
    figures_dir = Path(config["training"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(config["training"]["metrics_path"])

    # Load preprocessed arrays
    arrays = np.load(models_dir / "arrays.npz")
    X_train, X_test = arrays["X_train"], arrays["X_test"]
    y_train, y_test = arrays["y_train"], arrays["y_test"]
    feature_names = joblib.load(models_dir / "feature_names.joblib")

    plot_class_distribution(y_train, y_test, figures_dir / "01_class_distribution.png")

    results = {}

    # --- Random Forest ---
    rf = joblib.load(models_dir / "random_forest.joblib")
    y_pred_rf = rf.predict(X_test)
    y_score_rf = rf.predict_proba(X_test)[:, 1]
    metrics_rf = compute_metrics(y_test, y_pred_rf, y_score_rf)
    results["RandomForest"] = {"metrics": metrics_rf, "y_pred": y_pred_rf, "y_score": y_score_rf}
    plot_confusion(y_test, y_pred_rf, "Random Forest — Confusion Matrix",
                   figures_dir / "02_cm_random_forest.png")
    plot_feature_importance(rf, feature_names, figures_dir / "05_feature_importance_rf.png")

    # --- XGBoost ---
    xgb = joblib.load(models_dir / "xgboost.joblib")
    y_pred_xgb = xgb.predict(X_test)
    y_score_xgb = xgb.predict_proba(X_test)[:, 1]
    metrics_xgb = compute_metrics(y_test, y_pred_xgb, y_score_xgb)
    results["XGBoost"] = {"metrics": metrics_xgb, "y_pred": y_pred_xgb, "y_score": y_score_xgb}
    plot_confusion(y_test, y_pred_xgb, "XGBoost — Confusion Matrix",
                   figures_dir / "02_cm_xgboost.png")
    plot_feature_importance(xgb, feature_names, figures_dir / "05_feature_importance_xgb.png")

    # --- Autoencoder ---
    with open(models_dir / "autoencoder_meta.json") as f:
        ae_meta = json.load(f)
    ae = load_autoencoder(
        models_dir / "autoencoder.pt",
        input_dim=ae_meta["input_dim"],
        encoding_dim=ae_meta["encoding_dim"],
        hidden_dims=ae_meta["hidden_dims"],
    )
    # Reconstruction error as score
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ae.eval()
    with torch.no_grad():
        tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        recon = ae(tensor)
        errors = torch.mean((recon - tensor) ** 2, dim=1).cpu().numpy()
    y_pred_ae = (errors > ae_meta["threshold"]).astype(int)
    # Higher error → more anomalous → use error as score
    metrics_ae = compute_metrics(y_test, y_pred_ae, errors)
    results["Autoencoder"] = {"metrics": metrics_ae, "y_pred": y_pred_ae, "y_score": errors}
    plot_confusion(y_test, y_pred_ae, "Autoencoder — Confusion Matrix",
                   figures_dir / "02_cm_autoencoder.png")

    # Best model confusion matrix (by F1)
    best_name = max(results, key=lambda n: results[n]["metrics"]["f1"])
    plot_confusion(
        y_test, results[best_name]["y_pred"],
        f"Best model ({best_name}) — Confusion Matrix",
        figures_dir / "02_cm_best.png",
    )

    # ROC & PR
    plot_roc_curves(results, y_test, figures_dir / "03_roc_curves.png")
    plot_pr_curves(results, y_test, figures_dir / "04_pr_curves.png")

    # Save metrics JSON
    metrics_out = {name: data["metrics"] for name, data in results.items()}
    metrics_out["best_model"] = best_name
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    # Console summary
    print("\n=== Evaluation Summary ===")
    header = f"{'Model':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10} {'PR-AUC':>10}"
    print(header)
    print("-" * len(header))
    for name, data in results.items():
        m = data["metrics"]
        print(
            f"{name:<15} {m['precision']:>10.4f} {m['recall']:>10.4f} "
            f"{m['f1']:>10.4f} {m.get('roc_auc', 0) or 0:>10.4f} {m.get('pr_auc', 0) or 0:>10.4f}"
        )
    print(f"\nBest model by F1: {best_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
