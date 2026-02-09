from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DesignMatrix:
    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]


def build_xy(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    target_transform: str = "none",
    feature_transform: str = "none",
) -> DesignMatrix:
    d = df.copy()

    if feature_transform == "log1p":
        for c in feature_cols:
            d[c] = np.log1p(np.maximum(d[c].astype(float), 0.0))

    y = d[target_col].astype(float).to_numpy()
    if target_transform == "log1p":
        y = np.log1p(np.maximum(y, 0.0))

    X = d[feature_cols].astype(float).to_numpy()
    return DesignMatrix(X=X, y=y, feature_names=list(feature_cols))
