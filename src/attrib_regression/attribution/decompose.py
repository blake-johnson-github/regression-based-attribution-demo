from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class ContributionResult:
    contributions: pd.DataFrame  # per-row contributions
    totals: pd.Series  # total contribution by feature


def decompose_linear(
    X: np.ndarray,
    feature_names: list[str],
    coef: np.ndarray,
    intercept: float,
    date_index: pd.Series | None = None,
) -> ContributionResult:
    """Contribution decomposition for linear models: contrib = X * coef."""
    contrib = X * coef.reshape(1, -1)
    df = pd.DataFrame(contrib, columns=feature_names)
    df["intercept"] = intercept

    if date_index is not None:
        df.insert(0, "date", date_index.to_numpy())

    totals = df.drop(columns=["date"], errors="ignore").sum(axis=0)
    return ContributionResult(contributions=df, totals=totals)
