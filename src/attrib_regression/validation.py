from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


def validate_dataframe(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    required_cols: List[str] | None = None,
    enforce_monotonic_dates: bool = False,
) -> Dict[str, Any]:
    """Validate a raw input DataFrame before transforms.

    Returns a summary dict with ``ok`` (bool) and ``warnings`` (list of str).
    Raises ``ValueError`` for hard failures (missing columns).
    """
    warnings: list[str] = []

    # --- required columns ---
    all_required = [date_col, target_col] + (required_cols or [])
    missing = [c for c in all_required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # --- date parseable ---
    dates = pd.to_datetime(df[date_col], errors="coerce")
    n_bad = dates.isna().sum() - df[date_col].isna().sum()
    if n_bad > 0:
        warnings.append(f"{n_bad} date values could not be parsed")

    if enforce_monotonic_dates and not dates.dropna().is_monotonic_increasing:
        warnings.append("Dates are not monotonically increasing")

    # --- target numeric & non-null ---
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        raise ValueError(f"Target column '{target_col}' is not numeric")
    n_null = df[target_col].isna().sum()
    if n_null > 0:
        warnings.append(f"Target column has {n_null} null values")

    return {"ok": len(warnings) == 0, "warnings": warnings}
