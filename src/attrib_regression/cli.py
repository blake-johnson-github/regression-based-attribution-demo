from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from attrib_regression.config import load_rba_config
from attrib_regression.io import read_table
from attrib_regression.validation import validate_dataframe
from attrib_regression.preprocess import basic_clean
from attrib_regression.features.adstock import apply_adstock
from attrib_regression.features.saturation import apply_saturation
from attrib_regression.features.build_matrix import build_xy
from attrib_regression.eval.tscv import TimeSeriesCV
from attrib_regression.models.train import fit_elasticnet_ts_cv
from attrib_regression.models.diagnostics import coef_table
from attrib_regression.attribution.decompose import decompose_linear
from attrib_regression.attribution.roi import compute_roi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    cfg = load_rba_config(args.config)

    # --- read once ---
    df = read_table(cfg.data.path)

    # --- validate raw input before transforms ---
    report = validate_dataframe(
        df,
        date_col=cfg.data.date_col,
        target_col=cfg.data.target_col,
        required_cols=cfg.variables.media_spend_cols + cfg.variables.control_cols,
        enforce_monotonic_dates=False,  # basic_clean will sort; set True if you want pre-sorted
    )
    print("Data validation passed:", report)

    # --- clean (sorts by date, drops NA dates, etc.) ---
    df = basic_clean(df, date_col=cfg.data.date_col)

    media_cols = cfg.variables.media_spend_cols
    control_cols = cfg.variables.control_cols

    # --- transforms ---
    media_work_cols = media_cols
    if cfg.transforms.adstock.enabled:
        df = apply_adstock(
            df,
            cols=media_cols,
            alphas=vars(cfg.transforms.adstock.alphas),
            max_lag=cfg.transforms.adstock.max_lag,
        )
        media_work_cols = [f"{c}__adstock" for c in media_cols]

    if cfg.transforms.saturation.enabled:
        df = apply_saturation(df, cols=media_work_cols, params=vars(cfg.transforms.saturation.params))
        media_work_cols = [f"{c}__sat" for c in media_work_cols]

    feature_cols = media_work_cols + control_cols

    dm = build_xy(
        df,
        target_col=cfg.data.target_col,
        feature_cols=feature_cols,
        target_transform=cfg.model.target_transform,
        feature_transform=cfg.model.feature_transform,
    )

    cv = TimeSeriesCV(
        n_splits=cfg.model.cv.n_splits,
        test_size=cfg.model.cv.test_size,
        gap=cfg.model.cv.gap,
    )

    fit, best_params = fit_elasticnet_ts_cv(
        dm.X,
        dm.y,
        dm.feature_names,
        positive=cfg.model.positive_media,
        standardize=cfg.model.standardize,
        cv=cv,
        param_grid=vars(cfg.model.hyperparams),
        random_state=getattr(cfg.model, "random_state", 42),
    )

    # --- contributions (in-sample; add holdout later) ---
    X_for_contrib = dm.X
    if fit.scaler is not None:
        X_for_contrib = fit.scaler.transform(X_for_contrib)

    contrib = decompose_linear(
        X=X_for_contrib,
        feature_names=dm.feature_names,
        coef=fit.coef_,
        intercept=fit.intercept_,
        date_index=df[cfg.data.date_col],
    )

    # --- "ROI" warning: keep but rename later (recommended) ---
    spend_totals = df[media_cols].sum(axis=0)
    media_totals = contrib.totals.reindex([c for c in contrib.totals.index if c.startswith(tuple(media_cols))], fill_value=0)
    roi = compute_roi(media_totals, spend_totals.reindex(media_cols))

    reports_dir = Path(cfg.outputs.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # provenance
    (reports_dir / "config_used.yml").write_text(Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")

    coef_df = coef_table(dm.feature_names, fit.coef_)
    coef_df.to_csv(reports_dir / "coef_table.csv", index=False)
    pd.DataFrame(fit.metrics_by_fold).to_csv(reports_dir / "cv_metrics.csv", index=False)
    contrib.contributions.to_csv(reports_dir / "contributions_timeseries.csv", index=False)
    contrib.totals.to_csv(reports_dir / "contribution_totals.csv")
    roi.to_csv(reports_dir / "roi_summary.csv", index=False)

    print("Best params:", best_params)
    print("Wrote reports to:", reports_dir.resolve())


if __name__ == "__main__":
    main()
