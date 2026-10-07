from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def load_long_ohlcv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    required = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"OHLCV CSV is missing columns: {sorted(missing)}")
    frame = frame.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    if frame.duplicated(["timestamp", "symbol"]).any():
        raise ValueError("OHLCV CSV contains duplicate timestamp/symbol bars")
    numeric = frame[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("OHLC prices must be finite and positive; volume must be finite")
    return frame


def pair_close(frame: pd.DataFrame, first: str, second: str, field: str = "close") -> tuple[pd.Series, pd.Series]:
    if field not in frame.columns:
        raise ValueError(f"unknown price field: {field}")
    selected = frame[frame["symbol"].isin([first, second])]
    panel = selected.pivot(index="timestamp", columns="symbol", values=field).dropna(subset=[first, second])
    if panel.empty:
        raise ValueError("the selected pair has no overlapping observations")
    return panel[first].rename(first), panel[second].rename(second)


def write_local_candle_manifest(
    path: str | Path,
    *,
    provider: str,
    product: str,
    parser_version: str,
    adjustment_provenance: str,
    universe_provenance: str,
) -> Path:
    """Write required provenance beside an existing local candle file."""
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    required = {
        "provider": provider,
        "product_entitlement": product,
        "parser_version": parser_version,
        "adjustment_provenance": adjustment_provenance,
        "universe_provenance": universe_provenance,
    }
    if any(not str(value).strip() for value in required.values()):
        raise ValueError("manifest provenance fields must be non-empty")
    suffix = data_path.suffix.lower()
    if suffix not in {".csv", ".parquet", ".pq"}:
        raise ValueError("local candle manifest supports CSV or parquet files")
    if suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(data_path)
        needed = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
        missing = needed - set(frame.columns)
        if missing:
            raise ValueError(f"OHLCV file is missing columns: {sorted(missing)}")
    else:
        frame = load_long_ohlcv(data_path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    if frame.empty:
        raise ValueError("cannot manifest an empty candle dataset")
    metadata = {
        **required,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "covered_start": frame["timestamp"].min().isoformat(),
        "covered_end": frame["timestamp"].max().isoformat(),
        "row_count": len(frame),
        "schema": list(frame.columns),
        "sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "raw_data_redistribution": "prohibited unless source entitlement explicitly permits",
    }
    manifest_path = data_path.with_suffix(data_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return manifest_path
