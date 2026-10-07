import numpy as np
import pandas as pd
import pytest

from pairs_trading.data import load_long_ohlcv, pair_close
from pairs_trading.models import engle_granger, kalman_hedge_ratio
from pairs_trading.strategy import backtest_spread, rolling_zscore
from pairs_trading.walkforward import PairWindow, split_windows, run_frozen_holdout, run_training_validation_holdout


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
    close_a = pd.Series([100.0, 100.0, 100.0, 110.0, 110.0, 110.0], index=dates)
    close_b = pd.Series([100.0, 100.0, 100.0, 100.0, 110.0, 110.0], index=dates)
    result = backtest_spread(
        z, entry_z=1.5, exit_z=0.5, cost_bps=0.0,
        close_a=close_a, close_b=close_b, hedge_ratio=1.0,
    )
    assert result.position.iloc[3] == -1.0
    assert result.pnl.iloc[3] == 0.0  # signal-day price move is not captured
    assert np.isclose(result.pnl.iloc[4], 0.05)  # prior -0.5 A, +0.5 B
    assert result.position.iloc[4] == 0.0
    assert result.turnover.iloc[3] == 0.0  # signal is after close; fill is next session
    assert np.isclose(result.turnover.iloc[4], 1.0)  # 0.5 notional per leg at next open


def test_open_execution_uses_t_close_signal_at_next_open_and_next_open_return():
    dates = pd.date_range("2024-01-01", periods=5)
    z = pd.Series([0.0, 2.0, 0.0, 0.0, 0.0], index=dates)
    open_a = pd.Series([100.0, 500.0, 110.0, 121.0, 133.1], index=dates)
    open_b = pd.Series([100.0, 500.0, 100.0, 100.0, 100.0], index=dates)
    result = backtest_spread(
        z, entry_z=1.5, exit_z=0.5, cost_bps=0.0,
        open_a=open_a, open_b=open_b, hedge_ratio=1.0,
    )
    assert result.position.iloc[1] == -1.0
    assert result.pnl.iloc[1] == 0.0  # no return accrued before T+1 open fill
    assert np.isclose(result.pnl.iloc[2], -0.01)  # T+1 to T+2 open: -0.5 * 10% A return


def test_hedge_ratio_is_applied_from_close_signal_on_next_open_not_reestimated():
    dates = pd.date_range("2024-01-01", periods=4)
    z = pd.Series([0.0, 2.0, 0.0, 0.0], index=dates)
    opens_a = pd.Series([100.0, 100.0, 110.0, 110.0], index=dates)
    opens_b = pd.Series([100.0, 100.0, 100.0, 110.0], index=dates)
    beta = pd.Series([1.0, 1.0, 2.0, 100.0], index=dates)
    result = backtest_spread(
        z, entry_z=1.5, exit_z=0.5, cost_bps=0.0,
        open_a=opens_a, open_b=opens_b, hedge_ratio=beta,
    )
    assert result.position.iloc[1] == -1.0
    assert result.shares_a.iloc[2] == -0.5
    assert result.shares_b.iloc[2] == 0.5
    assert np.isclose(result.pnl.iloc[2], -0.05)


def test_backtest_rejects_duplicate_timestamps():
    dates = pd.DatetimeIndex(["2024-01-01", "2024-01-01"])
    z = pd.Series([0.0, 2.0], index=dates)
    prices = pd.Series([100.0, 101.0], index=dates)
    with pytest.raises(ValueError, match="unique increasing"):
        backtest_spread(z, open_a=prices, open_b=prices, hedge_ratio=1.0)


def test_holdout_runner_does_not_fit_or_trade_on_training_data():
    dates = pd.date_range("2024-01-01", periods=7)
    z = pd.Series([2.0, 0.0, 2.0, 0.0, 0.0, 2.0, 0.0], index=dates)
    open_a = pd.Series([100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0], index=dates)
    open_b = pd.Series([100.0] * 7, index=dates)
    result = run_frozen_holdout(
        z, open_a, open_b, 1.0,
        train_end=dates[2], holdout_start=dates[4], holdout_end=dates[6],
        entry_z=1.5, exit_z=0.5, cost_bps=0.0,
    )
    assert result.index.equals(dates[4:7])
    assert result.iloc[0].position == 0.0
    assert result.iloc[2].position == 0.0  # no training position leaks into holdout


def test_split_windows_reject_overlapping_named_windows():
    frame = pd.DataFrame({"value": range(5)}, index=pd.date_range("2024-01-01", periods=5))
    windows = [PairWindow("train", "2024-01-01", "2024-01-03"), PairWindow("holdout", "2024-01-03", "2024-01-05")]
    try:
        split_windows(frame, windows)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("overlapping windows must be rejected")


def test_training_validation_holdout_are_separate_real_splits():
    dates = pd.date_range("2024-01-01", periods=9)
    frame = pd.DataFrame({"value": range(9)}, index=dates)
    split = run_training_validation_holdout(
        frame, train_end=dates[2], validation_start=dates[3], validation_end=dates[5],
        holdout_start=dates[6], holdout_end=dates[8],
    )
    assert list(split) == ["train", "validation", "holdout"]
    assert [len(split[name]) for name in split] == [3, 3, 3]
    assert split["train"].index.max() < split["validation"].index.min()
    assert split["validation"].index.max() < split["holdout"].index.min()


def test_backtest_reports_only_completed_round_trip_trades():
    dates = pd.date_range("2024-01-01", periods=7)
    z = pd.Series([0.0, 2.2, 1.0, 0.2, -2.1, -1.0, -2.0], index=dates)
    opens_a = pd.Series([100.0, 101.0, 102.0, 103.0, 102.0, 101.0, 100.0], index=dates)
    opens_b = pd.Series([100.0, 100.0, 100.0, 100.0, 101.0, 102.0, 103.0], index=dates)
    result = backtest_spread(z, entry_z=2.0, exit_z=0.5, cost_bps=0.0,
                             open_a=opens_a, open_b=opens_b, hedge_ratio=1.0)
    assert len(result.trades) == 3
    assert result.trades.iloc[0].entry_signal_date == dates[1]
    assert result.trades.iloc[0].exit_signal_date == dates[3]
    assert result.trades.iloc[0].exit_reason == "threshold"
    assert result.trades.iloc[1].exit_reason == "window_end_forced_close"
    assert result.trades.iloc[2].exit_reason == "window_end_forced_close"
    assert result.position.iloc[-1] != 0.0  # signal position stays separate from liquidation
