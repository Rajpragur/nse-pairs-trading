"""Fail-closed CLI scaffold for a manifest-pinned frozen holdout study.

This module validates inputs and emits a run manifest only; it intentionally
contains no portfolio simulation implementation and never makes up results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "frozen_study.example.json"
REQUIRED_OHLCV = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _required_text(section: dict[str, Any], key: str, label: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip() or value.strip().lower() in {"todo", "tbd", "unknown"}:
        raise ValueError(f"config requires a documented non-empty {label}.{key}")
    return value.strip()


def validate_config(config: dict[str, Any], config_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    data = config.get("data")
    if not isinstance(data, dict):
        raise ValueError("config.data must be an object")
    data_name = _required_text(data, "path", "data")
    manifest_name = _required_text(data, "manifest", "data")
    dataset_path = (config_path.parent / data_name).expanduser().resolve()
    manifest_path = (config_path.parent / manifest_name).expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"required user-provided licensed dataset is missing: {dataset_path}. "
            "Supply your own authorized OHLCV file; this repository contains no market data."
        )
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"required dataset provenance manifest is missing: {manifest_path}. "
            "Create and retain a manifest for your authorized local data before running."
        )

    meta = _load_json(manifest_path)
    for key in ("provider", "product_entitlement", "retrieved_at_utc", "parser_version",
                "adjustment_provenance", "universe_provenance", "sha256"):
        _required_text(meta, key, "manifest")
    if str(meta["sha256"]).lower() != _sha256(dataset_path):
        raise ValueError("manifest sha256 does not match dataset bytes")
    rights = _required_text(meta, "rights_status", "manifest").lower()
    if rights not in {"authorized", "permitted", "licensed"}:
        raise ValueError("manifest rights_status must explicitly record authorization")
    coverage_start = pd.Timestamp(_required_text(meta, "covered_start", "manifest"))
    coverage_end = pd.Timestamp(_required_text(meta, "covered_end", "manifest"))
    if coverage_start > coverage_end:
        raise ValueError("manifest covered_start must be <= covered_end")
    if dataset_path.suffix.lower() != ".csv":
        raise ValueError("frozen-study runner currently accepts CSV only")
    frame = pd.read_csv(dataset_path, nrows=0)
    missing = REQUIRED_OHLCV - set(frame.columns)
    if missing:
        raise ValueError(f"dataset missing OHLCV columns: {sorted(missing)}")

    study = config.get("study")
    if not isinstance(study, dict):
        raise ValueError("config.study must be an object")
    windows = study.get("windows")
    if not isinstance(windows, dict):
        raise ValueError("config.study.windows must be an object")
    ranges = []
    for name in ("formation", "validation", "holdout"):
        span = windows.get(name)
        if not isinstance(span, dict):
            raise ValueError(f"config.study.windows.{name} must be an object")
        start = pd.Timestamp(_required_text(span, "start", f"study.windows.{name}"))
        end = pd.Timestamp(_required_text(span, "end", f"study.windows.{name}"))
        if start > end:
            raise ValueError(f"{name} start must be <= end")
        ranges.append((name, start, end))
    for (previous_name, _, previous_end), (current_name, current_start, _) in zip(ranges, ranges[1:]):
        if current_start <= previous_end:
            raise ValueError(f"study windows overlap: {previous_name} and {current_name}")
    if not (ranges[0][2] < ranges[1][1] and ranges[1][2] < ranges[2][1]):
        raise ValueError("formation, validation and holdout must be strictly ordered and disjoint")
    if coverage_start > ranges[0][1] or coverage_end < ranges[2][2]:
        raise ValueError("manifest coverage does not span all frozen-study windows")

    universe = study.get("universe")
    if not isinstance(universe, dict):
        raise ValueError("config.study.universe must be an object")
    if universe.get("kind") != "historical_liquidity_proxy":
        raise ValueError("study universe kind must be historical_liquidity_proxy")
    if universe.get("top_n") != 150:
        raise ValueError("frozen study requires the top-150 historical liquidity proxy")
    if universe.get("lookback_months") != 6 or universe.get("effective_lag") != "next_session":
        raise ValueError("universe must use six completed months and next-session effectiveness")

    execution = study.get("execution")
    if not isinstance(execution, dict) or execution.get("signal") != "close" or execution.get("fill") != "next_session_open":
        raise ValueError("execution must specify close signal and next-session-open fill")
    if execution.get("return_interval") != "open_to_open_after_fill":
        raise ValueError("execution return interval must begin after the next-open fill")
    costs = study.get("costs")
    if not isinstance(costs, dict):
        raise ValueError("study.costs must explicitly specify cost assumptions")
    for key in ("aggregate_bps_per_leg_turnover", "slippage_bps_per_leg_turnover"):
        if not isinstance(costs.get(key), (int, float)) or costs[key] < 0:
            raise ValueError(f"study.costs.{key} must be non-negative numeric bps")
    if not isinstance(study.get("seed"), int) or study["seed"] < 0:
        raise ValueError("study.seed must be a fixed non-negative integer")

    outputs = config.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("config.outputs must be an object")
    run_manifest = _required_text(outputs, "run_manifest", "outputs")
    output_path = (config_path.parent / run_manifest).expanduser().resolve()
    return dataset_path, output_path, meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate frozen-study inputs and write an input manifest; does not compute performance results."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="study JSON config (default: checked-in template)")
    parser.add_argument("--validate-only", action="store_true", help="validate the supplied local data and provenance without writing output")
    args = parser.parse_args(argv)
    config_path = args.config.expanduser().resolve()
    try:
        config = _load_json(config_path)
        dataset_path, output_path, meta = validate_config(config, config_path)
        if args.validate_only:
            print(f"Inputs validated: {dataset_path}")
            return 0
        payload = {
            "status": "inputs_validated_not_run",
            "results": None,
            "note": "This scaffold validates provenance and frozen parameters; it does not execute or report backtest performance.",
            "config_sha256": _sha256(config_path),
            "dataset_sha256": _sha256(dataset_path),
            "dataset_manifest_sha256": _sha256((config_path.parent / config["data"]["manifest"]).expanduser().resolve()),
            "provider": meta["provider"],
            "product_entitlement": meta["product_entitlement"],
            "universe_provenance": meta["universe_provenance"],
            "study": config["study"],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Inputs validated; non-results manifest written: {output_path}")
        return 0
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"frozen-study: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
