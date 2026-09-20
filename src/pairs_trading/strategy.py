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


def rolling_zscore(spread: pd.Series, window: int = 60) -> pd.Series:
    if window < 2:
        raise ValueError("window must be at least two")
    rolling = spread.rolling(window=window, min_periods=window)
    return (spread - rolling.mean()) / rolling.std(ddof=0).replace(0, np.nan)


def backtest_spread(zscore: pd.Series, spread_returns: pd.Series, entry_z: float = 2.0, exit_z: float = 0.5, cost_bps: float = 2.0) -> BacktestResult:
    aligned = pd.concat([zscore.rename("z"), spread_returns.rename("return")], axis=1)
    positions = []
    position = 0.0
    for z in aligned["z"]:
        if np.isfinite(z):
            if position == 0.0 and z >= entry_z:
                position = -1.0
            elif position == 0.0 and z <= -entry_z:
                position = 1.0
            elif position != 0.0 and abs(z) <= exit_z:
                position = 0.0
        positions.append(position)
    position_series = pd.Series(positions, index=aligned.index, name="position")
    changes = position_series.diff().abs().fillna(position_series.abs())
    costs = changes * cost_bps / 10000.0
    pnl = position_series.shift(1).fillna(0.0) * aligned["return"].fillna(0.0) - costs
    equity = (1.0 + pnl).cumprod()
    return BacktestResult(position_series, pnl.rename("pnl"), equity.rename("equity"), changes.rename("turnover"))
