from __future__ import annotations

import numpy as np
import pandas as pd


def adstock_series(x: np.ndarray, alpha: float, max_lag: int) -> np.ndarray:
    """Geometric adstock: out[t] = x[t] + alpha * out[t-1]."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    carry = 0.0
    for t in range(len(x)):
        carry = x[t] + alpha * carry
        out[t] = carry
    return out


def apply_adstock(
    df: pd.DataFrame, cols: list[str], alphas: dict[str, float], max_lag: int
) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        a = float(alphas.get(c, 0.0))
        out[f"{c}__adstock"] = adstock_series(
            out[c].to_numpy(), alpha=a, max_lag=max_lag
        )
    return out
