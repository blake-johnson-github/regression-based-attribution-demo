import numpy as np
import pandas as pd


def coef_table(feature_names: list[str], coef: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"feature": feature_names, "coef": coef})
    return df.sort_values("coef", ascending=False).reset_index(drop=True)
