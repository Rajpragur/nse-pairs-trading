"""Self-financing, fixed-gross-notional pair accounting with causal entry shifts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    position: pd.Series
    pnl: pd.Series
    equity: pd.Series
    turnover: pd.Series
    shares_a: pd.Series
    shares_b: pd.Series
    costs: pd.Series
    trades: pd.DataFrame


def rolling_zscore(spread: pd.Series, window: int = 60) -> pd.Series:
    if window < 2:
        raise ValueError("window must be at least two")
    rolling = spread.rolling(window=window, min_periods=window)
    return (spread - rolling.mean()) / rolling.std(ddof=0).replace(0, np.nan)


def backtest_spread(
    zscore: pd.Series,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    cost_bps: float = 2.0,
    *,
    close_a: pd.Series | None = None,
    close_b: pd.Series | None = None,
    open_a: pd.Series | None = None,
    open_b: pd.Series | None = None,
    hedge_ratio: pd.Series | float | None = None,
) -> BacktestResult:
    """Backtest a pair with one-way costs on total gross leg turnover.

    A close-based signal at T is executed at the next session's open. Returns
    accrue only between consecutive opens after execution. Prefer open prices;
    closes remain an explicit legacy approximation. `hedge_ratio[t]` must be
    known at close T; it sets that signal's next-open target, never a future
    hedge estimate. Long/short legs have unit gross notional.
    """
    if cost_bps < 0:
        raise ValueError("cost_bps must be non-negative")
    if hedge_ratio is None:
        raise ValueError("self-financing PnL requires a causal hedge_ratio")
    if not zscore.index.is_monotonic_increasing or zscore.index.has_duplicates:
        raise ValueError("signals require unique increasing timestamps")
    aligned = pd.concat([zscore.rename("z")], axis=1)
    positions = []
    position = 0.0
    opened_at = None
    trades = []
    for timestamp, z in aligned["z"].items():
        if np.isfinite(z):
            if position == 0.0 and z >= entry_z:
                position = -1.0
                opened_at = timestamp
            elif position == 0.0 and z <= -entry_z:
                position = 1.0
                opened_at = timestamp
            elif position != 0.0 and abs(z) <= exit_z:
                trades.append({"entry_signal_date": opened_at, "exit_signal_date": timestamp,
                               "direction": "short_spread" if position < 0 else "long_spread",
                               "exit_reason": "threshold"})
                position = 0.0
                opened_at = None
        positions.append(position)
    # Record a residual position as a window-end forced close, distinct from
    # a threshold exit. Accounting below charges the closing turnover.
    if position != 0.0 and len(aligned.index):
        trades.append({"entry_signal_date": opened_at, "exit_signal_date": aligned.index[-1],
                       "direction": "short_spread" if position < 0 else "long_spread",
                       "exit_reason": "window_end_forced_close"})
    position_series = pd.Series(positions, index=aligned.index, name="position")

    if (open_a is None) != (open_b is None):
        raise ValueError("open_a and open_b must be provided together")
    if open_a is None and ((close_a is None) != (close_b is None)):
        raise ValueError("close_a and close_b must be provided together")
    if open_a is None and close_a is None:
        raise ValueError("leg opens (preferred) or closes are required for self-financing PnL")
    if hedge_ratio is None:
        raise ValueError("a causal hedge_ratio is required for leg sizing")
    if open_a is not None:
        assert open_b is not None
        if close_a is not None or close_b is not None:
            raise ValueError("provide opens or closes, not both")
        prices = pd.concat([open_a.rename("a"), open_b.rename("b")], axis=1).reindex(aligned.index)
    else:
        assert close_a is not None and close_b is not None
        prices = pd.concat([close_a.rename("a"), close_b.rename("b")], axis=1).reindex(aligned.index)
    if prices.isna().any().any() or (prices <= 0).any().any() or not np.isfinite(prices.to_numpy(dtype=float)).all():
        raise ValueError("leg prices must be finite and positive")
    if np.isscalar(hedge_ratio):
        beta = pd.Series(float(hedge_ratio), index=aligned.index, dtype=float)
    elif isinstance(hedge_ratio, pd.Series):
        beta = hedge_ratio.reindex(aligned.index)
    else:
        raise TypeError("hedge_ratio must be a scalar or pandas Series")
    if beta.isna().any() or not np.isfinite(beta.to_numpy(dtype=float)).all():
        raise ValueError("hedge_ratio must be finite and available for every session")
    if (1.0 + beta.abs()).eq(0.0).any():
        raise ValueError("hedge_ratio magnitude is too large to size finite legs")

    # Each close-time target is shifted to the following open; open-to-open
    # returns then begin on that fill session's next observation.
    leg_a_weight = 1.0 / (1.0 + beta.abs())
    leg_b_weight = beta.abs() / (1.0 + beta.abs())
    target_a = position_series.mul(leg_a_weight).shift(1).fillna(0.0)
    target_b = position_series.mul(-np.sign(beta) * leg_b_weight).shift(1).fillna(0.0)
    holdings_a = target_a.copy()
    holdings_b = target_b.copy()
    asset_return_a = prices["a"].pct_change().fillna(0.0)
    asset_return_b = prices["b"].pct_change().fillna(0.0)
    raw_pnl = holdings_a.mul(asset_return_a).add(holdings_b.mul(asset_return_b))
    turnover = target_a.sub(target_a.shift(1).fillna(0.0)).abs().add(
        target_b.sub(target_b.shift(1).fillna(0.0)).abs()
    )
    costs = turnover * (cost_bps / 10000.0)
    # If strategy remained open at the last close, charge exit turnover there.
    if len(position_series) and position_series.iloc[-1] != 0.0:
        final_exit_turnover = abs(float(target_a.iloc[-1])) + abs(float(target_b.iloc[-1]))
        turnover.iloc[-1] += final_exit_turnover
        costs.iloc[-1] += final_exit_turnover * (cost_bps / 10000.0)
        trades.append({"entry_signal_date": opened_at, "exit_signal_date": aligned.index[-1],
                       "direction": "short_spread" if position < 0 else "long_spread",
                       "exit_reason": "window_end_forced_close"})
    pnl = raw_pnl - costs
    equity = (1.0 + pnl).cumprod()
    trades_frame = pd.DataFrame(trades, columns=["entry_signal_date", "exit_signal_date", "direction", "exit_reason"])
    return BacktestResult(
        position_series, pnl.rename("pnl"), equity.rename("equity"),
        turnover.rename("turnover"), target_a.rename("shares_a"),
        target_b.rename("shares_b"), costs.rename("costs"), trades_frame,
    )
