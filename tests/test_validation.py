from __future__ import annotations

import pandas as pd
import pytest

from attrib_regression.validation import validate_dataframe


@pytest.fixture
def good_df():
    return pd.DataFrame({
        "date": ["2025-01-01", "2025-01-02", "2025-01-03"],
        "y": [10.0, 20.0, 30.0],
        "spend_a": [1.0, 2.0, 3.0],
    })


def test_valid_data_passes(good_df):
    result = validate_dataframe(good_df, date_col="date", target_col="y")
    assert result["ok"] is True
    assert result["warnings"] == []


def test_missing_column_raises(good_df):
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_dataframe(
            good_df, date_col="date", target_col="y", required_cols=["nonexistent"]
        )


def test_missing_date_col_raises(good_df):
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_dataframe(good_df, date_col="bad_date", target_col="y")


def test_non_numeric_target_raises():
    df = pd.DataFrame({"date": ["2025-01-01"], "y": ["not_a_number"]})
    with pytest.raises(ValueError, match="not numeric"):
        validate_dataframe(df, date_col="date", target_col="y")


def test_null_target_warns():
    df = pd.DataFrame({"date": ["2025-01-01", "2025-01-02"], "y": [1.0, None]})
    result = validate_dataframe(df, date_col="date", target_col="y")
    assert result["ok"] is False
    assert any("null" in w for w in result["warnings"])


def test_unparseable_dates_warn():
    df = pd.DataFrame({"date": ["2025-01-01", "not-a-date"], "y": [1.0, 2.0]})
    result = validate_dataframe(df, date_col="date", target_col="y")
    assert any("parsed" in w for w in result["warnings"])


def test_monotonic_dates_enforced():
    df = pd.DataFrame({
        "date": ["2025-01-03", "2025-01-01", "2025-01-02"],
        "y": [1.0, 2.0, 3.0],
    })
    result = validate_dataframe(
        df, date_col="date", target_col="y", enforce_monotonic_dates=True
    )
    assert any("monotonic" in w.lower() for w in result["warnings"])
