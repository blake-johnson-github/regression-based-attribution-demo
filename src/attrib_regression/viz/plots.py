import matplotlib.pyplot as plt
import pandas as pd


def plot_actual_vs_pred(
    dates: pd.Series, actual: pd.Series, pred: pd.Series, title: str = "Actual vs Pred"
) -> None:
    plt.figure()
    plt.plot(dates, actual, label="actual")
    plt.plot(dates, pred, label="pred")
    plt.title(title)
    plt.legend()
    plt.tight_layout()


def plot_contrib_share(
    contrib_totals: pd.Series, title: str = "Contribution Share"
) -> None:
    s = contrib_totals.copy()
    s = s[s.index != "intercept"]
    s = s.sort_values(ascending=False)
    plt.figure()
    (s / s.sum()).plot(kind="bar")
    plt.title(title)
    plt.tight_layout()
