import numpy as np
import pandas as pd
import pytest

from pairs_trading.portfolio import backtest_pairs


def test_portfolio_aggregates_overlapping_pairs_and_charges_turnover_costs():
    dates = pd.date_range("2024-01-01", periods=4)
    index = pd.MultiIndex.from_product([["AB", "CD"], dates], names=["pair", "date"])
    positions = pd.Series([-1, -1, 0, 0, 1, 1, 0, 0], index=index, dtype=float)
    returns = pd.Series([0, 0.02, 0, 0, 0, 0.04, 0, 0], index=index, dtype=float)
    result = backtest_pairs(positions, returns, pair_gross=0.5, cost_bps=100)
    assert np.isclose(result.loc[dates[0], "turnover"], 1.0)
    assert np.isclose(result.loc[dates[0], "cost"], 0.01)
    assert np.isclose(result.loc[dates[0], "net_pnl"], -0.01)
    assert np.isclose(result.loc[dates[1], "gross_pnl"], 0.01)
    assert result.loc[dates[1], "cost"] == 0.0
    assert result.loc[dates[0], "gross_exposure"] == 1.0
    assert result.loc[dates[0], "net_exposure"] == 0.0


def test_portfolio_enforces_gross_limit_and_finite_inputs():
    dates = pd.date_range("2024-01-01", periods=2)
    idx = pd.MultiIndex.from_product([["A", "B"], dates], names=["pair", "date"])
    pos = pd.Series([1, 1, 1, 1], index=idx, dtype=float)
    ret = pd.Series(0.0, index=idx)
    with pytest.raises(ValueError, match="gross exposure"):
        backtest_pairs(pos, ret, pair_gross=0.75, max_gross=1.0)
    bad = ret.copy()
    bad.iloc[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        backtest_pairs(pos, bad, pair_gross=0.5)


def test_portfolio_requires_unique_sorted_pair_date_keys():
    dates = pd.date_range("2024-01-01", periods=2)
    idx = pd.MultiIndex.from_tuples([("A", dates[0]), ("A", dates[0])], names=["pair", "date"])
    with pytest.raises(ValueError, match="unique"):
        backtest_pairs(pd.Series([1.0, 1.0], index=idx), pd.Series([0.0, 0.0], index=idx), pair_gross=0.5)
