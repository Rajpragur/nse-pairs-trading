from __future__ import annotations

import numpy as np
import pandas as pd


def select_monthly_top_universe(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    *,
    top_n: int = 150,
    lookback_months: int = 6,
    min_observations: int = 60,
) -> pd.DataFrame:
    """Build month-end top-N universe rows, effective from the next session.

    Liquidity score is the median daily close * volume over the six completed
    calendar months ending at selection_date. A symbol must have at least
    ``min_observations`` finite, positive observations in that window. Input
    columns define the historical security panel; this function does not use
    current index constituents or claim survivorship-free coverage.
    """
    if not isinstance(close, pd.DataFrame) or not isinstance(volume, pd.DataFrame):
        raise TypeError("close and volume must be DataFrames")
    if close.empty or volume.empty:
        raise ValueError("close and volume must be non-empty")
    if not close.index.is_unique or not volume.index.is_unique:
        raise ValueError("input indexes must be unique")
    if not close.index.is_monotonic_increasing or not volume.index.is_monotonic_increasing:
        raise ValueError("input indexes must be increasing")
    if not close.columns.equals(volume.columns):
        raise ValueError("close and volume must have identical columns")
    if not close.index.equals(volume.index):
        raise ValueError("close and volume must have identical index")
    if close.columns.has_duplicates:
        raise ValueError("symbols must be unique")
    if top_n < 1 or lookback_months < 1 or min_observations < 1:
        raise ValueError("top_n, lookback_months and min_observations must be positive")

    close_values = close.to_numpy(dtype=float)
    volume_values = volume.to_numpy(dtype=float)
    valid_prices = np.isfinite(close_values) & (close_values > 0)
    valid_volume = np.isfinite(volume_values) & (volume_values > 0)
    traded_value = pd.DataFrame(
        np.where(valid_prices & valid_volume, close_values * volume_values, np.nan),
        index=close.index,
        columns=close.columns,
    )
    rows: list[dict] = []
    month_ends = close.index.to_period("M").drop_duplicates().to_timestamp("M")
    for selection_date in month_ends:
        window_start = selection_date.to_period("M") - (lookback_months - 1)
        first_month = window_start.start_time.normalize()
        window = traded_value.loc[(traded_value.index >= first_month) & (traded_value.index <= selection_date)]
        counts = window.count()
        medians = window.median()
        eligible = medians.loc[counts >= min_observations].dropna()
        ranked = sorted(eligible.index, key=lambda symbol: (-float(eligible[symbol]), str(symbol)))[:top_n]
        future_sessions = close.index[close.index > selection_date]
        if not len(future_sessions):
            continue
        rows.append({
            "selection_date": selection_date,
            "effective_date": future_sessions[0],
            "symbols": tuple(ranked),
            "scores": {symbol: float(eligible[symbol]) for symbol in ranked},
            "observations": {symbol: int(counts[symbol]) for symbol in ranked},
        })
    return pd.DataFrame(rows, columns=["selection_date", "effective_date", "symbols", "scores", "observations"])
