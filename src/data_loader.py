"""
Data loading utilities for NSL-KDD (and optionally CICIDS2017).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# NSL-KDD column names (41 features + label + difficulty)
NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"
]

# Map original attack names to binary: 0 = normal, 1 = attack
NORMAL_LABEL = "normal"


def _map_label(label: str) -> int:
    """Convert NSL-KDD label to binary: 0 normal, 1 attack."""
    return 0 if str(label).strip().lower() == NORMAL_LABEL else 1


def load_nsl_kdd(
    train_path: str | Path,
    test_path: str | Path | None = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load NSL-KDD train and test sets.

    Parameters
    ----------
    train_path : path to KDDTrain+.txt
    test_path  : path to KDDTest+.txt (optional). If None, split is done later.
    random_state : seed for any internal shuffle

    Returns
    -------
    train_df, test_df : DataFrames with columns from NSL_KDD_COLUMNS
                        and binary 'target' column (0/1).
    """
    train_path = Path(train_path)
    if not train_path.exists():
        raise FileNotFoundError(
            f"Training file not found: {train_path}\n"
            "Download NSL-KDD from https://www.unb.ca/cic/datasets/nsl.html "
            "and place KDDTrain+.txt / KDDTest+.txt into data/"
        )

    logger.info("Loading train set from %s", train_path)
    train_df = pd.read_csv(train_path, header=None, names=NSL_KDD_COLUMNS)
    train_df["target"] = train_df["label"].apply(_map_label)

    if test_path is not None and Path(test_path).exists():
        logger.info("Loading test set from %s", test_path)
        test_df = pd.read_csv(test_path, header=None, names=NSL_KDD_COLUMNS)
        test_df["target"] = test_df["label"].apply(_map_label)
    else:
        logger.warning("Test file not found; will split train later.")
        test_df = pd.DataFrame()

    # Basic integrity checks
    for name, df in [("train", train_df), ("test", test_df)]:
        if df.empty:
            continue
        if df.isnull().any().any():
            logger.warning("%s contains NaN values — they will be handled in preprocessing", name)
        n_attack = (df["target"] == 1).sum()
        n_normal = (df["target"] == 0).sum()
        logger.info(
            "%s: %d samples (normal=%d, attack=%d, attack_ratio=%.2f)",
            name, len(df), n_normal, n_attack, n_attack / max(len(df), 1)
        )

    return train_df, test_df


def load_dataset(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Dispatch loader according to config['data']['dataset'].
    Currently supports NSL-KDD.
    """
    dataset = config.get("data", {}).get("dataset", "nsl_kdd").lower()
    if dataset == "nsl_kdd":
        return load_nsl_kdd(
            train_path=config["data"]["train_path"],
            test_path=config["data"].get("test_path"),
            random_state=config["data"].get("random_state", 42),
        )
    raise ValueError(f"Unsupported dataset: {dataset}. Use 'nsl_kdd'.")


def split_if_needed(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    If test_df is empty, perform stratified split of train_df.
    """
    if test_df is not None and not test_df.empty:
        return train_df, test_df

    logger.info("Performing stratified train/test split (test_size=%.2f)", test_size)
    train_part, test_part = train_test_split(
        train_df,
        test_size=test_size,
        stratify=train_df["target"],
        random_state=random_state,
    )
    return train_part.reset_index(drop=True), test_part.reset_index(drop=True)
