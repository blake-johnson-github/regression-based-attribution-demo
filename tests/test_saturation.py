from __future__ import annotations

import numpy as np
import pandas as pd

from attrib_regression.features.saturation import hill, apply_saturation


def test_hill_zero_input():
    result = hill(np.array([0.0]), ec50=1.0, slope=1.0)
    assert result[0] == pytest.approx(0.0, abs=1e-10)


def test_hill_at_ec50_is_half():
    """At x == ec50, Hill function should return ~0.5."""
    result = hill(np.array([5.0]), ec50=5.0, slope=1.0)
    assert result[0] == pytest.approx(0.5, abs=1e-6)


def test_hill_large_input_approaches_one():
    result = hill(np.array([1e6]), ec50=1.0, slope=2.0)
    assert result[0] == pytest.approx(1.0, abs=1e-6)


def test_hill_negative_input_clipped():
    result = hill(np.array([-5.0]), ec50=1.0, slope=1.0)
    assert result[0] == pytest.approx(0.0, abs=1e-10)


def test_hill_steeper_slope():
    """Higher slope → sharper transition around ec50."""
    x = np.array([2.0])
    gentle = hill(x, ec50=5.0, slope=1.0)[0]
    steep = hill(x, ec50=5.0, slope=5.0)[0]
    # Below ec50, steeper slope → lower value
    assert steep < gentle


def test_apply_saturation_creates_columns():
    df = pd.DataFrame({"ch__adstock": [1.0, 2.0, 3.0]})
    params = {"ch__adstock": {"ec50": 2.0, "slope": 1.5}}
    out = apply_saturation(df, cols=["ch__adstock"], params=params)
    assert "ch__adstock__sat" in out.columns
    assert len(out) == 3


import pytest
