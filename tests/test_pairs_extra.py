import pandas as pd

from pairs_trading.data import pair_close
from pairs_trading.walkforward import PairWindow, split_windows


def test_pair_close_pivots_long_ohlcv():
    frame = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
        "symbol": ["AAA", "BBB", "AAA", "BBB"],
        "open": [10, 20, 11, 21], "high": [11, 21, 12, 22], "low": [9, 19, 10, 20],
        "close": [10.5, 20.5, 11.5, 21.5], "volume": [100, 200, 110, 210],
    })
    first, second = pair_close(frame, "AAA", "BBB")
    assert list(first) == [10.5, 11.5]
    assert list(second) == [20.5, 21.5]


def test_walkforward_windows_do_not_overlap():
    frame = pd.DataFrame({"value": range(6)}, index=pd.date_range("2024-01-01", periods=6))
    windows = [PairWindow("formation", "2024-01-01", "2024-01-02"), PairWindow("holdout", "2024-01-03", "2024-01-06")]
    split = split_windows(frame, windows)
    assert len(split["formation"]) == 2
    assert set(split["formation"].index).isdisjoint(split["holdout"].index)
