from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .models import engle_granger, kalman_hedge_ratio
from .strategy import backtest_spread, rolling_zscore


def make_pair_data(periods: int = 900, seed: int = 11) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=periods)
    x = pd.Series(100 + np.cumsum(rng.normal(0, 0.7, periods)), index=dates, name="asset_x")
    stationary = pd.Series(rng.normal(0, 0.7, periods), index=dates)
    y = pd.Series(1.5 + 1.25 * x.to_numpy() + stationary.to_numpy(), index=dates, name="asset_y")
    return x, y


def main(output_dir: str = "results") -> None:
    output = Path(output_dir);output.mkdir(exist_ok=True)
    x, y = make_pair_data()
    eg = engle_granger(y, x)
    kalman = kalman_hedge_ratio(y, x)
    z = rolling_zscore(kalman.spread, window=60)
    spread_returns = kalman.spread.diff().fillna(0.0) / y.shift(1).replace(0, np.nan)
    backtest = backtest_spread(z, spread_returns, entry_z=2.0, exit_z=0.5, cost_bps=2.0)
    summary = {
        "status": "synthetic_demo",
        "warning": "Synthetic data only; results are a pipeline smoke test, not a trading claim.",
        "engle_granger": {"intercept": eg.intercept, "beta": eg.beta, "adf_t_stat": eg.adf_t_stat},
        "kalman_final_beta": float(kalman.beta.iloc[-1]),
        "backtest": {"total_return": float(backtest.equity.iloc[-1] - 1.0), "turnover": float(backtest.turnover.sum()), "trades": int((backtest.turnover > 0).sum())},
    }
    (output / "demo_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
