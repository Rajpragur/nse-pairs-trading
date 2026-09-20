from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_long_ohlcv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    required = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"OHLCV CSV is missing columns: {sorted(missing)}")
    return frame.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def pair_close(frame: pd.DataFrame, first: str, second: str, field: str = "close") -> tuple[pd.Series, pd.Series]:
    if field not in frame.columns:
        raise ValueError(f"unknown price field: {field}")
    selected = frame[frame["symbol"].isin([first, second])]
    panel = selected.pivot(index="timestamp", columns="symbol", values=field).dropna(subset=[first, second])
    if panel.empty:
        raise ValueError("the selected pair has no overlapping observations")
    return panel[first].rename(first), panel[second].rename(second)
