from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from attrib_regression.features.adstock import adstock_series, apply_adstock


def test_adstock_zero_alpha_is_identity():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    result = adstock_series(x, alpha=0.0, max_lag=10)
    np.testing.assert_array_equal(result, x)


def test_adstock_accumulates_carry():
    x = np.array([10.0, 0.0, 0.0])
    result = adstock_series(x, alpha=0.5, max_lag=10)
    expected = np.array([10.0, 5.0, 2.5])
    np.testing.assert_allclose(result, expected)


def test_adstock_alpha_one_is_cumsum():
    x = np.array([1.0, 1.0, 1.0])
    result = adstock_series(x, alpha=1.0, max_lag=10)
    expected = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(result, expected)


def test_adstock_empty_input():
    x = np.array([])
    result = adstock_series(x, alpha=0.5, max_lag=5)
    assert len(result) == 0


def test_apply_adstock_creates_columns():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    out = apply_adstock(df, cols=["a", "b"], alphas={"a": 0.3, "b": 0.5}, max_lag=5)
    assert "a__adstock" in out.columns
    assert "b__adstock" in out.columns
    # original columns untouched
    pd.testing.assert_series_equal(out["a"], df["a"])


def test_apply_adstock_missing_alpha_defaults_to_zero():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    out = apply_adstock(df, cols=["a"], alphas={}, max_lag=5)
    # alpha=0 means identity
    np.testing.assert_array_equal(out["a__adstock"].values, df["a"].values)
