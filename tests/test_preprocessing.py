"""
Unit tests for preprocessing helpers.
Run: pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import get_feature_columns, build_preprocessor, CATEGORICAL_COLS


def _make_dummy_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "duration": rng.integers(0, 100, n),
        "protocol_type": rng.choice(["tcp", "udp", "icmp"], n),
        "service": rng.choice(["http", "ftp", "smtp", "other"], n),
        "flag": rng.choice(["SF", "S0", "REJ"], n),
        "src_bytes": rng.integers(0, 10000, n),
        "dst_bytes": rng.integers(0, 10000, n),
        "label": rng.choice(["normal", "neptune", "satan"], n),
        "difficulty": rng.integers(0, 21, n),
        "target": rng.integers(0, 2, n),
    })


def test_get_feature_columns():
    df = _make_dummy_df()
    num, cat = get_feature_columns(df)
    assert "protocol_type" in cat
    assert "duration" in num
    assert "label" not in num and "label" not in cat
    assert "target" not in num and "target" not in cat


def test_build_preprocessor_fit_transform():
    df = _make_dummy_df()
    num, cat = get_feature_columns(df)
    pre = build_preprocessor(num, cat, scaler_type="standard")
    X = pre.fit_transform(df[num + cat])
    assert X.shape[0] == len(df)
    assert X.shape[1] > len(num)  # one-hot expands categoricals
    assert not np.isnan(X).any()


def test_preprocessor_handles_unknown_category():
    df_train = _make_dummy_df(50)
    df_test = _make_dummy_df(20)
    # inject unknown protocol
    df_test.loc[0, "protocol_type"] = "unknown_proto"
    num, cat = get_feature_columns(df_train)
    pre = build_preprocessor(num, cat)
    pre.fit(df_train[num + cat])
    X_test = pre.transform(df_test[num + cat])
    assert X_test.shape[0] == len(df_test)
