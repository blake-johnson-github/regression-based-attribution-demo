from __future__ import annotations

import numpy as np
import pandas as pd


def hill(x: np.ndarray, ec50: float, slope: float) -> np.ndarray:
    """Hill saturation curve.

    y = x^s / (x^s + ec50^s)
    """
    x = np.maximum(np.asarray(x, dtype=float), 0.0)
    s = float(slope)
    e = float(ec50)
    xs = np.power(x, s)
    es = np.power(e, s)
    return xs / (xs + es + 1e-12)


def apply_saturation(
    df: pd.DataFrame, cols: list[str], params: dict[str, dict], suffix: str = "__sat"
) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        p = params.get(c, {})
        ec50 = float(p.get("ec50", 1.0))
        slope = float(p.get("slope", 1.0))
        out[f"{c}{suffix}"] = hill(out[c].to_numpy(), ec50=ec50, slope=slope)
    return out
