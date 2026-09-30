"""
Model definitions: Random Forest, XGBoost, Autoencoder (PyTorch).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Classical models
# ---------------------------------------------------------------------------

def build_random_forest(params: dict, random_state: int = 42) -> RandomForestClassifier:
    """Create a RandomForestClassifier with given hyperparameters."""
    return RandomForestClassifier(
        n_estimators=params.get("n_estimators", 200),
        max_depth=params.get("max_depth", 20),
        min_samples_leaf=params.get("min_samples_leaf", 2),
        n_jobs=params.get("n_jobs", -1),
        class_weight=params.get("class_weight", "balanced"),
        random_state=random_state,
    )


def build_xgboost(params: dict, random_state: int = 42) -> XGBClassifier:
    """Create an XGBClassifier with given hyperparameters."""
    return XGBClassifier(
        n_estimators=params.get("n_estimators", 300),
        max_depth=params.get("max_depth", 8),
        learning_rate=params.get("learning_rate", 0.05),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        eval_metric=params.get("eval_metric", "logloss"),
        use_label_encoder=params.get("use_label_encoder", False),
        random_state=random_state,
        n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# Autoencoder for unsupervised anomaly detection
# ---------------------------------------------------------------------------

class Autoencoder(nn.Module):
    """
    Fully-connected autoencoder.
    Trained only on normal traffic; high reconstruction error → anomaly.
    """

    def __init__(self, input_dim: int, encoding_dim: int = 32, hidden_dims: list | None = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]

        # Encoder
        encoder_layers = []
        prev = input_dim
        for h in hidden_dims:
            encoder_layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        encoder_layers.append(nn.Linear(prev, encoding_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder (mirror)
        decoder_layers = []
        prev = encoding_dim
        for h in reversed(hidden_dims):
            decoder_layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        decoder_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


def train_autoencoder(
    model: Autoencoder,
    X_normal: np.ndarray,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str | None = None,
) -> Autoencoder:
    """
    Train autoencoder exclusively on normal samples (target == 0).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    tensor = torch.tensor(X_normal, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n_batches = 0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        if epoch % 5 == 0 or epoch == 1:
            logger.info("Autoencoder epoch %d/%d — loss=%.6f", epoch, epochs, total_loss / max(n_batches, 1))
    return model


def autoencoder_predict(
    model: Autoencoder,
    X: np.ndarray,
    threshold: float,
    device: str | None = None,
) -> np.ndarray:
    """
    Return binary predictions: 1 if reconstruction error > threshold.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(X, dtype=torch.float32).to(device)
        recon = model(tensor)
        errors = torch.mean((recon - tensor) ** 2, dim=1).cpu().numpy()
    return (errors > threshold).astype(int)


def compute_ae_threshold(
    model: Autoencoder,
    X_normal: np.ndarray,
    percentile: float = 95.0,
    device: str | None = None,
) -> float:
    """Compute reconstruction-error threshold on normal validation data."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(X_normal, dtype=torch.float32).to(device)
        recon = model(tensor)
        errors = torch.mean((recon - tensor) ** 2, dim=1).cpu().numpy()
    thr = float(np.percentile(errors, percentile))
    logger.info("AE threshold (p%.0f)=%.6f", percentile, thr)
    return thr


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_model(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, nn.Module):
        torch.save(obj.state_dict(), path)
    else:
        joblib.dump(obj, path)
    logger.info("Saved model → %s", path)


def load_sklearn_model(path: str | Path):
    return joblib.load(path)


def load_autoencoder(path: str | Path, input_dim: int, encoding_dim: int = 32,
                     hidden_dims: list | None = None, device: str | None = None) -> Autoencoder:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Autoencoder(input_dim, encoding_dim, hidden_dims)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model
