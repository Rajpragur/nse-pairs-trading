from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .strategy import backtest_spread


@dataclass(frozen=True)
class PairWindow:
    name: str
    start: str | pd.Timestamp
    end: str | pd.Timestamp


def split_windows(frame: pd.DataFrame, windows: list[PairWindow]) -> dict[str, pd.DataFrame]:
    result = {}
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("split input requires a unique increasing index")
    names = [window.name for window in windows]
    if len(set(names)) != len(names):
        raise ValueError("window names must be unique")
    for window in windows:
        if pd.Timestamp(window.start) > pd.Timestamp(window.end):
            raise ValueError(f"window start must be <= end: {window.name}")
    ordered = sorted(windows, key=lambda window: pd.Timestamp(window.start))
    for previous, current in zip(ordered, ordered[1:]):
        if pd.Timestamp(current.start) <= pd.Timestamp(previous.end):
            raise ValueError(f"windows overlap: {previous.name} and {current.name}")
    for window in windows:
        start = pd.Timestamp(window.start)
        end = pd.Timestamp(window.end)
        result[window.name] = frame.loc[(frame.index >= start) & (frame.index <= end)].copy()
    return result


def run_frozen_holdout(
    zscore: pd.Series,
    open_a: pd.Series,
    open_b: pd.Series,
    hedge_ratio: pd.Series | int | float,
    *,
    train_end: str | pd.Timestamp,
    holdout_start: str | pd.Timestamp,
    holdout_end: str | pd.Timestamp,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    cost_bps: float = 2.0,
) -> pd.DataFrame:
    """Evaluate frozen parameters on an isolated, initially flat holdout.

    Inputs such as z-scores and hedge ratios must already be causal; this
    function deliberately accepts them as frozen features and never fits.
    Hedge ratios are frozen training estimates (or precomputed causal values).
    The caller must produce z-scores from training parameters and prior data.
    """

    train_end = pd.Timestamp(train_end)
    holdout_start = pd.Timestamp(holdout_start)
    holdout_end = pd.Timestamp(holdout_end)
    if not train_end < holdout_start <= holdout_end:
        raise ValueError("require train_end < holdout_start <= holdout_end")
    if not zscore.index.is_monotonic_increasing or zscore.index.has_duplicates:
        raise ValueError("holdout inputs require unique increasing timestamps")
    evaluation_index = zscore.index[(zscore.index >= holdout_start) & (zscore.index <= holdout_end)]
    if evaluation_index.empty:
        raise ValueError("holdout window has no observations")
    z = zscore.reindex(evaluation_index)
    a = open_a.reindex(evaluation_index)
    b = open_b.reindex(evaluation_index)
    if isinstance(hedge_ratio, pd.Series):
        beta: pd.Series | float = hedge_ratio.reindex(evaluation_index)
    elif isinstance(hedge_ratio, (int, float, np.integer, np.floating)):
        beta = float(hedge_ratio)
    else:
        raise TypeError("hedge_ratio must be a scalar or pandas Series")
    result = backtest_spread(
        z, entry_z=entry_z, exit_z=exit_z, cost_bps=cost_bps,
        open_a=a, open_b=b, hedge_ratio=beta,
    )
    return pd.concat(
        [result.position, result.pnl, result.equity, result.turnover,
         result.shares_a, result.shares_b, result.costs],
        axis=1,
    )


def run_training_validation_holdout(
    frame: pd.DataFrame,
    *,
    train_end: str | pd.Timestamp,
    validation_start: str | pd.Timestamp,
    validation_end: str | pd.Timestamp,
    holdout_start: str | pd.Timestamp,
    holdout_end: str | pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    """Split dated local data into disjoint train, validation and holdout sets."""
    if frame.index.empty or not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("split input requires a non-empty unique increasing index")
    start = pd.Timestamp(frame.index.min())
    windows = [
        PairWindow("train", start, pd.Timestamp(train_end)),
        PairWindow("validation", pd.Timestamp(validation_start), pd.Timestamp(validation_end)),
        PairWindow("holdout", pd.Timestamp(holdout_start), pd.Timestamp(holdout_end)),
    ]
    return split_windows(frame, windows)


def default_windows() -> list[PairWindow]:
    return [
        PairWindow("formation", "2021-01-01", "2022-12-31"),
        PairWindow("validation", "2023-01-01", "2023-12-31"),
        PairWindow("holdout", "2024-01-01", "2025-12-31"),
    ]
