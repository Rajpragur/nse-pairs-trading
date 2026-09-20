from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PairWindow:
    name: str
    start: str
    end: str


def split_windows(frame: pd.DataFrame, windows: list[PairWindow]) -> dict[str, pd.DataFrame]:
    result = {}
    for window in windows:
        start = pd.Timestamp(window.start)
        end = pd.Timestamp(window.end)
        result[window.name] = frame.loc[(frame.index >= start) & (frame.index <= end)].copy()
    return result


def default_windows() -> list[PairWindow]:
    return [
        PairWindow("formation", "2021-01-01", "2022-12-31"),
        PairWindow("validation", "2023-01-01", "2023-12-31"),
        PairWindow("holdout", "2024-01-01", "2025-12-31"),
    ]
