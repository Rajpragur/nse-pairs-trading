import pandas as pd
import pytest

from pairs_trading.data import load_long_ohlcv, pair_close, write_local_candle_manifest
from pairs_trading.strategy import backtest_spread
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


def test_local_candle_loader_rejects_duplicate_identity(tmp_path):
    path = tmp_path / "bars.csv"
    path.write_text(
        "timestamp,symbol,open,high,low,close,volume\n"
        "2024-01-01,AAA,10,11,9,10.5,100\n"
        "2024-01-01,AAA,10,11,9,10.5,100\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_long_ohlcv(path)


def test_costs_charge_both_leg_turnover_on_entry_and_exit():
    dates = pd.date_range("2024-01-01", periods=4)
    z = pd.Series([2.0, 2.0, 0.0, 0.0], index=dates)
    prices = pd.Series([100.0] * 4, index=dates)
    result = backtest_spread(
        z, entry_z=1.5, exit_z=0.5, cost_bps=10.0,
        open_a=prices, open_b=prices, hedge_ratio=1.0,
    )
    assert result.turnover.iloc[1] == 1.0
    assert result.turnover.iloc[2] == 0.0
    assert result.turnover.iloc[3] == 1.0
    assert result.costs.iloc[1] == 0.001
    assert result.costs.iloc[3] == 0.001
    assert result.pnl.iloc[1] == -0.001


def test_backtest_rejects_missing_open_prices_without_filling_gap():
    dates = pd.date_range("2024-01-01", periods=3)
    z = pd.Series([0.0, 2.0, 0.0], index=dates)
    open_a = pd.Series([100.0, float("nan"), 110.0], index=dates)
    open_b = pd.Series([100.0, 100.0, 100.0], index=dates)
    with pytest.raises(ValueError, match="finite and positive"):
        backtest_spread(z, open_a=open_a, open_b=open_b, hedge_ratio=1.0)


def test_backtest_rejects_missing_price_timestamp_alignment():
    dates = pd.date_range("2024-01-01", periods=3)
    z = pd.Series([0.0, 2.0, 0.0], index=dates)
    a = pd.Series([100.0, 101.0], index=dates[:2])
    b = pd.Series([100.0, 100.0, 100.0], index=dates)
    with pytest.raises(ValueError, match="finite and positive"):
        backtest_spread(z, open_a=a, open_b=b, hedge_ratio=1.0)


def test_local_manifest_records_checksum_span_and_provenance(tmp_path):
    path = tmp_path / "bars.csv"
    path.write_text(
        "timestamp,symbol,open,high,low,close,volume\n"
        "2024-01-01,AAA,10,11,9,10.5,100\n"
        "2024-01-02,AAA,11,12,10,11.5,110\n",
        encoding="utf-8",
    )
    manifest = write_local_candle_manifest(
        path, provider="authorized-local-export", product="documented entitlement",
        parser_version="1", adjustment_provenance="unadjusted", universe_provenance="user supplied",
    )
    import json
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    assert metadata["row_count"] == 2
    assert metadata["covered_start"].startswith("2024-01-01")
    assert len(metadata["sha256"]) == 64
    assert metadata["provider"] == "authorized-local-export"
