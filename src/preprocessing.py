"""
Feature preprocessing: encoding, scaling, imbalance handling.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# Columns that are categorical in NSL-KDD
CATEGORICAL_COLS = ["protocol_type", "service", "flag"]
# Columns that should not be used as features
DROP_COLS = ["label", "difficulty", "target"]


def get_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return lists of numeric and categorical feature column names."""
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    cat = [c for c in CATEGORICAL_COLS if c in feature_cols]
    num = [c for c in feature_cols if c not in cat]
    return num, cat


def build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
    scaler_type: str = "standard",
) -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
    - scales numeric features
    - one-hot encodes categorical features
    """
    if scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    transformers = []
    if numeric_cols:
        transformers.append(("num", scaler, numeric_cols))
    if categorical_cols:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def preprocess(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ColumnTransformer, List[str]]:
    """
    Full preprocessing pipeline.

    Returns
    -------
    X_train, X_test, y_train, y_test, fitted_preprocessor, feature_names
    """
    numeric_cols, categorical_cols = get_feature_columns(train_df)
    logger.info(
        "Features: %d numeric, %d categorical",
        len(numeric_cols), len(categorical_cols)
    )

    y_train = train_df["target"].values.astype(int)
    y_test = test_df["target"].values.astype(int)

    scaler_type = config.get("preprocessing", {}).get("numeric_scaler", "standard")
    preprocessor = build_preprocessor(numeric_cols, categorical_cols, scaler_type)

    X_train_raw = train_df[numeric_cols + categorical_cols]
    X_test_raw = test_df[numeric_cols + categorical_cols]

    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    # Recover feature names after one-hot
    feature_names = _get_feature_names(preprocessor, numeric_cols, categorical_cols)

    # Handle class imbalance on training set only
    imbalance = config.get("preprocessing", {}).get("handle_imbalance", "smote")
    if imbalance == "smote":
        k = config.get("preprocessing", {}).get("smote_k_neighbors", 5)
        logger.info("Applying SMOTE (k_neighbors=%d)", k)
        try:
            smote = SMOTE(random_state=config.get("training", {}).get("random_state", 42),
                          k_neighbors=k)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            logger.info("After SMOTE: %d samples (attack ratio=%.2f)",
                        len(y_train), y_train.mean())
        except Exception as e:
            logger.warning("SMOTE failed (%s). Continuing without oversampling.", e)

    return X_train, X_test, y_train, y_test, preprocessor, feature_names


def _get_feature_names(
    preprocessor: ColumnTransformer,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> List[str]:
    """Extract feature names after ColumnTransformer."""
    names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "num":
            names.extend(cols)
        elif name == "cat":
            try:
                cat_names = trans.get_feature_names_out(cols)
                names.extend(list(cat_names))
            except Exception:
                names.extend([f"{c}_{i}" for c in cols for i in range(10)])
    return names
