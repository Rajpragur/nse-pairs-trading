import json
from pathlib import Path

import numpy as np
import pandas as pd

from pairs_trading.models import engle_granger
from pairs_trading.selection import _half_life


def test_python_matches_checked_cpp_numeric_fixture():
    x, y = [], []
    for i in range(80):
        walk = 0.15 * i + np.sin(i * 0.19) * 1.1
        x.append(50.0 + walk)
        y.append(3.25 + 1.27 * walk + np.sin(i * 0.73) * 0.12)
    result = engle_granger(pd.Series(y), pd.Series(x))
    expected = json.loads((Path(__file__).parent / "parity_fixture.json").read_text())
    assert abs(result.intercept - expected["intercept"]) < 1e-9
    assert abs(result.beta - expected["beta"]) < 1e-10
    assert abs(result.adf_t_stat - expected["adf_t_stat"]) < 1e-8
    assert abs(_half_life(result.residual) - expected["half_life"]) < 1e-8
