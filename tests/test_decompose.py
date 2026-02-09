from __future__ import annotations

import numpy as np
import pandas as pd

from attrib_regression.attribution.decompose import decompose_linear


def test_contributions_sum_to_prediction():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    coef = np.array([0.5, -0.3])
    intercept = 1.0

    result = decompose_linear(X, ["a", "b"], coef, intercept)

    # Each row's contributions + intercept should equal model prediction
    predictions = X @ coef + intercept
    row_sums = result.contributions[["a", "b", "intercept"]].sum(axis=1).values
    np.testing.assert_allclose(row_sums, predictions)


def test_totals_match_column_sums():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    coef = np.array([0.5, 0.3])
    intercept = 2.0

    result = decompose_linear(X, ["a", "b"], coef, intercept)

    assert result.totals["a"] == pytest.approx(0.5 * (1 + 3 + 5))
    assert result.totals["b"] == pytest.approx(0.3 * (2 + 4 + 6))
    assert result.totals["intercept"] == pytest.approx(2.0 * 3)


def test_date_index_included():
    X = np.array([[1.0], [2.0]])
    dates = pd.Series(pd.to_datetime(["2025-01-01", "2025-01-02"]))
    result = decompose_linear(X, ["a"], np.array([1.0]), 0.0, date_index=dates)
    assert "date" in result.contributions.columns


def test_no_date_index():
    X = np.array([[1.0], [2.0]])
    result = decompose_linear(X, ["a"], np.array([1.0]), 0.0)
    assert "date" not in result.contributions.columns


import pytest
