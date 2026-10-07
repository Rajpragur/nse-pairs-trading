from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from pairs_trading.selection import select_pairs
from pairs_trading.strategy import backtest_spread, rolling_zscore

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT.parent / "configs" / "five_symbol_real_demo.json"
DEFAULT_DATA = Path.home() / "Documents" / "nse-factor-lab" / "local-data" / "five-symbol-full" / "combined_ohlcv.parquet"


def load_config(path: Path = CONFIG_PATH) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    study = config["study"]
    windows = study["windows"]
    ordered = [(name, pd.Timestamp(windows[name]["start"]), pd.Timestamp(windows[name]["end"]))
               for name in ("formation", "validation", "holdout")]
    for (left_name, _, left_end), (right_name, right_start, _) in zip(ordered, ordered[1:]):
        if left_end >= right_start:
            raise ValueError(f"windows overlap or are unordered: {left_name}, {right_name}")
    costs = study["costs"]
    expected = costs["brokerage_bps_per_leg_turnover"] + costs["slippage_bps_per_leg_turnover"]
    if not math.isclose(expected, costs["total_bps_per_leg_turnover"], rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("total cost must equal brokerage plus slippage")
    return config


def load_bars(data_path: Path, config: dict, exclude_reliance_december_2024: bool = False) -> pd.DataFrame:
    bars = pd.read_parquet(data_path)
    required = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
    if not required.issubset(bars.columns):
        raise ValueError(f"dataset missing columns: {sorted(required - set(bars.columns))}")
    bars = bars[bars["symbol"].isin(config["study"]["symbols"])].copy()
    if exclude_reliance_december_2024:
        sessions = pd.to_datetime(bars["timestamp"]).dt.tz_localize(None)
        bars = bars.loc[~((bars["symbol"] == "RELIANCE-EQ") & (sessions >= "2024-12-01") & (sessions < "2025-01-01"))].copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], errors="coerce")
    if bars["timestamp"].isna().any():
        raise ValueError("invalid timestamps in dataset")
    if bool(bars[["timestamp", "symbol"]].duplicated().any()):
        raise ValueError("duplicate symbol bars in dataset")
    for col in ("open", "high", "low", "close", "volume"):
        bars[col] = pd.to_numeric(bars[col], errors="coerce")
    numeric = bars[["open", "high", "low", "close", "volume"]]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("nonfinite OHLCV in dataset")
    if (bars[["open", "high", "low", "close"]] <= 0).any().any() or (bars["volume"] < 0).any():
        raise ValueError("nonpositive OHLC or negative volume")
    if (bars["high"] < bars[["open", "low", "close"]].max(axis=1)).any() or (bars["low"] > bars[["open", "high", "close"]].min(axis=1)).any():
        raise ValueError("invalid OHLC bounds")
    if set(bars["symbol"].unique()) != set(config["study"]["symbols"]):
        raise ValueError("input symbols do not exactly match frozen study symbols")
    bars = bars.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    bars["session"] = bars["timestamp"].dt.tz_localize(None).dt.normalize()
    return bars


def metric_stats(returns: pd.Series) -> dict[str, float | int]:
    values = returns.dropna().astype(float)
    if not len(values):
        return {"observations": 0, "net_pnl": float("nan"), "annualized_sharpe": float("nan"), "max_drawdown": float("nan")}
    equity = (1.0 + values).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
    sharpe = float(values.mean() / std * np.sqrt(252.0)) if np.isfinite(std) and std > 0 else float("nan")
    return {"observations": int(len(values)), "net_pnl": float(equity.iloc[-1] - 1.0),
            "annualized_sharpe": sharpe, "max_drawdown": float(drawdown.min())}


def _date_span(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return frame[(frame["session"] >= start) & (frame["session"] <= end)].copy()


def _signal_for_pair(pair_bars: pd.DataFrame, pair: tuple[str, str], beta: float, intercept: float,
                     start: pd.Timestamp, end: pd.Timestamp, window: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    a, b = pair
    panel = pair_bars.pivot(index="session", columns="symbol", values="close").sort_index().reindex(columns=[a, b])
    opens = pair_bars.pivot(index="session", columns="symbol", values="open").sort_index().reindex(columns=[a, b])
    valid = panel[[a, b]].notna().all(axis=1) & opens[[a, b]].notna().all(axis=1)
    panel = panel.loc[valid]
    opens = opens.loc[valid]
    spread = np.log(panel[a]) - (intercept + beta * np.log(panel[b]))
    z = rolling_zscore(spread, window=window)
    mask = (z.index >= start) & (z.index <= end)
    return z.loc[mask], opens.loc[mask, a], opens.loc[mask, b]


def _greedy_non_overlapping(diagnostics: pd.DataFrame, fdr: float) -> tuple[tuple[str, str], ...]:
    significant = diagnostics.loc[diagnostics["eligible"] & diagnostics["q_value"].notna() & (diagnostics["q_value"] <= fdr)].copy()
    significant = significant.sort_values(["q_value", "p_value", "pair"], kind="mergesort")
    chosen = []
    used = set()
    for label in significant["pair"]:
        first, second = label.split("~", 1)
        if first in used or second in used:
            continue
        chosen.append((first, second))
        used.update((first, second))
    return tuple(chosen)


def run_study(data_path: Path = DEFAULT_DATA, config_path: Path = CONFIG_PATH, output_dir: Path | None = None,
              fdr_override: float | None = None, cost_override: tuple[float, float] | None = None,
              exclude_reliance_december_2024: bool = False) -> dict:
    config = load_config(config_path)
    study = config["study"]
    spans = {name: (pd.Timestamp(item["start"]), pd.Timestamp(item["end"])) for name, item in study["windows"].items()}
    bars = load_bars(data_path, config, exclude_reliance_december_2024=exclude_reliance_december_2024)
    form_start, form_end = spans["formation"]
    formation_bars = _date_span(bars, form_start, form_end)
    formation_close = formation_bars.pivot(index="session", columns="symbol", values="close").sort_index()
    selection = select_pairs(
        formation_close, formation_start=form_start, formation_end=form_end,
        fdr=study["selection"]["fdr"], min_observations=study["selection"]["min_observations"],
        rolling_window=study["selection"]["rolling_window"],
        max_rolling_beta_std=study["selection"]["max_rolling_beta_std"],
        method=study["selection"]["method"], trend=study["selection"]["trend"],
        autolag=study["selection"]["autolag"],
    )
    diagnostics = selection.diagnostics.copy()
    # Explicit formation-period spread diagnostics for every candidate.
    for row_index, row in diagnostics.iterrows():
        first, second = row["pair"].split("~", 1)
        a = np.log(formation_close[first].dropna())
        b = np.log(formation_close[second].dropna())
        overlap = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
        spread = overlap["a"] - (float(row["intercept"]) + float(row["beta"]) * overlap["b"])
        diagnostics.loc[row_index, "formation_spread_mean"] = float(spread.mean())
        diagnostics.loc[row_index, "formation_spread_std"] = float(spread.std(ddof=0))
        diagnostics.loc[row_index, "formation_spread_min"] = float(spread.min())
        diagnostics.loc[row_index, "formation_spread_max"] = float(spread.max())
    # Benjamini-Hochberg correction spans all 10 pairs tested in formation.
    diagnostics["selected"] = False
    diagnostics["selection_reason"] = diagnostics["reason"]
    diagnostics["p_value"] = diagnostics["adf_pvalue"]
    # Selection diagnostics' eligibility is a hard gate; BH significance alone
    # never overrides rolling-beta stability.
    from pairs_trading.selection import benjamini_hochberg
    diagnostics["q_value"] = benjamini_hochberg(diagnostics["p_value"])
    fdr = float(study["selection"]["fdr"] if fdr_override is None else fdr_override)
    selected_pairs = _greedy_non_overlapping(diagnostics, fdr)
    diagnostics["selected"] = False
    diagnostics["selection_reason"] = diagnostics["reason"]
    diagnostics.loc[diagnostics["q_value"].notna() & (diagnostics["q_value"] <= fdr), "selection_reason"] = "bh_significant"
    diagnostics.loc[(diagnostics["q_value"] <= fdr) & (~diagnostics["eligible"]), "selection_reason"] = "bh_significant_but_ineligible_unstable_beta"
    selected_labels = {f"{a}~{b}" for a, b in selected_pairs}
    diagnostics.loc[diagnostics["pair"].isin(selected_labels), "selected"] = True
    diagnostics.loc[diagnostics["selected"], "selection_reason"] = "bh_significant_and_non_overlapping"
    reports: list[dict] = []
    if cost_override is None:
        brokerage_bps = float(study["costs"]["brokerage_bps_per_leg_turnover"])
        slippage_bps = float(study["costs"]["slippage_bps_per_leg_turnover"])
    else:
        brokerage_bps, slippage_bps = map(float, cost_override)
    cost_bps = brokerage_bps + slippage_bps
    for window_name in ("validation", "holdout"):
        start, end = spans[window_name]
        for first, second in selected_pairs:
            label = f"{first}~{second}"
            diagnostic = diagnostics.loc[diagnostics["pair"] == label].iloc[0]
            # Fit model and freeze spread normalization on formation only.
            formation_pair = formation_close[[first, second]].dropna()
            design = np.column_stack([np.ones(len(formation_pair)), np.log(formation_pair[second].to_numpy(dtype=float))])
            coefficients = np.linalg.lstsq(design, np.log(formation_pair[first].to_numpy(dtype=float)), rcond=None)[0]
            intercept, beta = float(coefficients[0]), float(coefficients[1])
            formation_spread = np.log(formation_pair[first]) - (intercept + beta * np.log(formation_pair[second]))
            spread_mean = float(formation_spread.mean())
            spread_std = float(formation_spread.std(ddof=0))
            pair_bars = bars[bars["symbol"].isin([first, second])]
            z, open_a, open_b = _signal_for_pair(pair_bars, (first, second), beta, intercept, start, end,
                                                int(study["strategy"]["lookback_sessions"]))
            z = z.sub(spread_mean).div(spread_std)
            result = backtest_spread(
                z, entry_z=float(study["strategy"]["entry_z"]), exit_z=float(study["strategy"]["exit_z"]),
                cost_bps=cost_bps, open_a=open_a, open_b=open_b, hedge_ratio=beta,
            )
            reports.append({
                "window": window_name, "pair": label, "observations": int(z.notna().sum()),
                "formation_intercept": intercept, "formation_beta": beta,
                "zscore_mean": float(z.mean()), "zscore_std": float(z.std(ddof=0)),
                "completed_round_trips": int(len(result.trades)),
                "window_end_forced_closes": int((result.trades["exit_reason"] == "window_end_forced_close").sum()) if not result.trades.empty else 0,
                "few_trades_warning": bool(len(result.trades) < 5),
                "statistical_interpretation": "statistically meaningless: fewer than 5 completed round trips" if len(result.trades) < 5 else "exploratory only; not a performance claim",
                **metric_stats(result.pnl), "brokerage_bps_per_leg_turnover": brokerage_bps,
                "slippage_bps_per_leg_turnover": slippage_bps,
                "cost_bps_per_leg_turnover": cost_bps,
            })
    output_dir = output_dir or (ROOT.parent / "local-data" / "five-symbol-real-run")
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(output_dir / "formation_pair_diagnostics.csv", index=False)
    pd.DataFrame(reports).to_csv(output_dir / "window_metrics.csv", index=False)
    run_meta = {
        "study_id": study["id"],
        "seed": study["seed"],
        "data_sha256": __import__("hashlib").sha256(data_path.read_bytes()).hexdigest(),
        "excluded_reliance_december_2024": bool(exclude_reliance_december_2024),
        "config_sha256": __import__("hashlib").sha256(config_path.read_bytes()).hexdigest(),
        "candidate_pairs": int(len(diagnostics)),
        "bh_significant_pairs_before_eligibility": int((diagnostics["q_value"] <= fdr).sum()),
        "eligible_bh_pairs": int(((diagnostics["q_value"] <= fdr) & diagnostics["eligible"]).sum()),
        "selected_pairs": [list(pair) for pair in selected_pairs],
        "fdr": fdr,
        "brokerage_bps_per_leg_turnover": brokerage_bps,
        "slippage_bps_per_leg_turnover": slippage_bps,
        "note": "Exploratory local-data pipeline demonstration, not a performance claim; raw vendor data remain local.",
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(run_meta, indent=2) + "\n", encoding="utf-8")
    return {"config": str(config_path), "data": str(data_path), "output_dir": str(output_dir),
            "candidate_pairs": int(len(diagnostics)),
            "bh_significant_pairs_before_eligibility": int((diagnostics["q_value"] <= fdr).sum()),
        "eligible_bh_pairs": int(((diagnostics["q_value"] <= fdr) & diagnostics["eligible"]).sum()),
            "eligible_bh_pairs": int(((diagnostics["q_value"] <= fdr) & diagnostics["eligible"]).sum()),
            "selected_pairs": [list(pair) for pair in selected_pairs],
            "window_metrics": reports, "note": "metrics are exploratory and are not a performance claim"}


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Run the five-symbol exploratory pairs-trading data demonstration")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--fdr", type=float, default=None)
    parser.add_argument("--brokerage-bps", type=float, default=None)
    parser.add_argument("--slippage-bps", type=float, default=None)
    parser.add_argument("--cost-sensitivity", action="store_true", help="run low/base/high cost scenarios at FDR 0.10")
    parser.add_argument("--exclude-reliance-dec-2024", action="store_true", help="exclude unresolved RELIANCE December 2024 data")
    args = parser.parse_args(argv)
    costs = None if args.brokerage_bps is None and args.slippage_bps is None else (args.brokerage_bps, args.slippage_bps)
    if costs is not None and (args.brokerage_bps is None or args.slippage_bps is None):
        parser.error("--brokerage-bps and --slippage-bps must be supplied together")
    report = run_cost_sensitivity(args.data, args.config, args.out_dir) if args.cost_sensitivity else run_study(args.data, args.config, args.out_dir, args.fdr, costs, args.exclude_reliance_dec_2024)
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def run_cost_sensitivity(data_path: Path = DEFAULT_DATA, config_path: Path = CONFIG_PATH,
                         output_dir: Path | None = None) -> dict:
    """Run the same frozen selection at three explicitly configured cost levels."""
    config = load_config(config_path)
    output_dir = output_dir or (ROOT.parent / "local-data" / "five-symbol-cost-sensitivity")
    levels = config["study"]["cost_sensitivity_bps"]
    runs = {}
    for name in ("low", "base", "high"):
        costs = levels[name]
        runs[name] = run_study(data_path, config_path, output_dir / name, fdr_override=0.10,
                               cost_override=(costs["brokerage"], costs["slippage"]))
    pairs_by_level = [runs[name]["selected_pairs"] for name in ("low", "base", "high")]
    if not (pairs_by_level[0] == pairs_by_level[1] == pairs_by_level[2]):
        raise RuntimeError("pair selection changed across cost sensitivity runs")
    return {"fdr": 0.10, "selected_pairs": pairs_by_level[0], "runs": runs}
