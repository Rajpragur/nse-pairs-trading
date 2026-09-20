import numpy as np
import pandas as pd

from pairs_trading.data import load_long_ohlcv, pair_close
from pairs_trading.models import engle_granger, kalman_hedge_ratio
from pairs_trading.strategy import backtest_spread, rolling_zscore
from pairs_trading.walkforward import PairWindow, split_windows


def test_engle_granger_recovers_cointegrating_beta():
    rng = np.random.default_rng(7)
    x = np.cumsum(rng.normal(size=500))
    y = 1.5 + 2.0 * x + rng.normal(scale=0.2, size=500)
    result = engle_granger(pd.Series(y), pd.Series(x))
    assert abs(result.beta - 2.0) < 0.03
    assert result.adf_t_stat < -5.0


def test_kalman_filter_tracks_time_varying_beta():
    rng = np.random.default_rng(3)
    x = np.linspace(-2, 2, 300)
    beta = np.linspace(1.0, 1.8, 300)
    y = 0.5 + beta * x + rng.normal(scale=0.03, size=300)
    result = kalman_hedge_ratio(pd.Series(y), pd.Series(x), process_variance=1e-4, observation_variance=1e-3)
    assert abs(result.beta.iloc[-1] - beta[-1]) < 0.15


def test_rolling_zscore_has_strict_warmup():
    spread = pd.Series([1.0, 2.0, 3.0, 2.0, 1.0])
    z = rolling_zscore(spread, window=3)
    assert z.iloc[:2].isna().all()
    assert np.isfinite(z.iloc[2])


def test_backtest_has_no_position_before_signal():
    dates = pd.date_range("2024-01-01", periods=6)
    z = pd.Series([np.nan, np.nan, 0.0, 2.0, 0.0, -2.0], index=dates)
    spread_returns = pd.Series([0.0, 0.0, 0.01, -0.02, 0.01, 0.02], index=dates)
    result = backtest_spread(z, spread_returns, entry_z=1.5, exit_z=0.5, cost_bps=0.0)
    assert result.position.iloc[0] == 0.0
    assert result.position.iloc[1] == 0.0
    assert result.position.iloc[3] == -1.0
    assert result.position.iloc[4] == 0.0
