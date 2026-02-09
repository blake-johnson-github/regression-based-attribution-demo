from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import StandardScaler

from ..eval.tscv import TimeSeriesCV


@dataclass
class FitResult:
    model: ElasticNet
    scaler: StandardScaler | None
    coef_: np.ndarray
    intercept_: float
    metrics_by_fold: list[dict]


def fit_elasticnet_ts_cv(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    positive: bool,
    standardize: bool,
    cv: TimeSeriesCV,
    param_grid: dict,
    random_state: int = 42,
) -> tuple[FitResult, dict]:
    """Simple grid search over ElasticNet hyperparams using time-series CV.

    TODO: consider switching to sklearn GridSearchCV with custom scorer
    """
    best = None
    best_params = None

    l1_ratios = param_grid.get("l1_ratio", [0.5])
    alphas = param_grid.get("alpha", [0.1])

    for l1 in l1_ratios:
        for a in alphas:
            fold_metrics = []
            for tr, te in cv.split(len(y)):
                Xtr, Xte = X[tr], X[te]
                ytr, yte = y[tr], y[te]

                scaler = None
                if standardize:
                    scaler = StandardScaler()
                    Xtr = scaler.fit_transform(Xtr)
                    Xte = scaler.transform(Xte)

                m = ElasticNet(
                    alpha=float(a),
                    l1_ratio=float(l1),
                    fit_intercept=True,
                    positive=bool(positive),
                    max_iter=20000,
                    random_state=random_state,
                )
                m.fit(Xtr, ytr)
                pred = m.predict(Xte)

                fold_metrics.append(
                    {
                        "alpha": float(a),
                        "l1_ratio": float(l1),
                        "mape": float(mean_absolute_percentage_error(yte, pred)),
                        "r2": float(r2_score(yte, pred)),
                    }
                )

            # choose by avg MAPE (lower is better); tie-break by higher R2
            avg_mape = float(np.mean([m["mape"] for m in fold_metrics]))
            avg_r2 = float(np.mean([m["r2"] for m in fold_metrics]))
            score = (avg_mape, -avg_r2)

            if best is None or score < best[0]:
                best = (score, fold_metrics, (a, l1))

    assert best is not None
    _, best_fold_metrics, (best_alpha, best_l1) = best
    best_params = {"alpha": float(best_alpha), "l1_ratio": float(best_l1)}

    # Refit on full data with best params
    scaler = None
    Xfit = X
    if standardize:
        scaler = StandardScaler()
        Xfit = scaler.fit_transform(Xfit)

    model = ElasticNet(
        alpha=best_params["alpha"],
        l1_ratio=best_params["l1_ratio"],
        fit_intercept=True,
        positive=bool(positive),
        max_iter=20000,
        random_state=42,
    )
    model.fit(Xfit, y)

    coef_ = model.coef_.copy()
    intercept_ = float(model.intercept_)

    return (
        FitResult(
            model=model,
            scaler=scaler,
            coef_=coef_,
            intercept_=intercept_,
            metrics_by_fold=best_fold_metrics,
        ),
        best_params,
    )
