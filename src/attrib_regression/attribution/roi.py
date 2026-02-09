from __future__ import annotations

import pandas as pd


def compute_roi(contrib_totals: pd.Series, spend_totals: pd.Series) -> pd.DataFrame:
    """Simple ROI: contribution / spend. Aligns by index.

    Note: this is a rough estimate -- doesn't account for baseline lift
    or interaction effects. Fine for directional reads.
    """
    df = pd.DataFrame({"contribution": contrib_totals, "spend": spend_totals}).copy()
    df["roi"] = df["contribution"] / df["spend"].replace(0, pd.NA)
    return df.reset_index(names="feature")
