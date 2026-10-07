from __future__ import annotations

import numpy as np
import pandas as pd


_COLUMNS = ["gross_pnl", "turnover", "cost", "net_pnl", "gross_exposure", "net_exposure", "equity"]


def backtest_pairs(
    positions: pd.Series,
    pair_returns: pd.Series,
    *,
    pair_gross: float,
    cost_bps: float = 2.0,
    max_gross: float | None = None,
) -> pd.DataFrame:
    """Aggregate already-causal, pair-level exposures into portfolio returns.

    `positions` are target pair directions in [-1, 1], indexed by (pair, date).
    Each unit is scaled to pair_gross fraction of portfolio equity; pair_returns
    are corresponding executable next-period returns. Costs apply to the
    gross-notional change of each pair. No selection, sizing fit, or signal
    generation occurs here. Equity compounds the resulting portfolio return.
    """
    if not isinstance(positions.index, pd.MultiIndex) or positions.index.nlevels != 2:
        raise ValueError("positions require a two-level (pair, date) index")
    if not positions.index.is_unique:
        raise ValueError("pair-date keys must be unique")
    if not positions.index.is_monotonic_increasing:
        raise ValueError("pair-date keys must be sorted")
    if not isinstance(pair_returns.index, pd.MultiIndex) or pair_returns.index.nlevels != 2:
        raise ValueError("pair_returns require a two-level (pair, date) index")
    if not pair_returns.index.is_unique or not pair_returns.index.is_monotonic_increasing:
        raise ValueError("pair-return keys must be unique and sorted")
    if not positions.index.equals(pair_returns.index):
        raise ValueError("positions and pair_returns must have identical pair-date keys")
    if not np.isfinite(pair_gross) or pair_gross <= 0:
        raise ValueError("pair_gross must be finite and positive")
    if not np.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError("cost_bps must be finite and non-negative")
    if max_gross is not None and (not np.isfinite(max_gross) or max_gross <= 0):
        raise ValueError("max_gross must be finite and positive")
    pos = positions.to_numpy(dtype=float)
    rets = pair_returns.to_numpy(dtype=float)
    if not np.isfinite(pos).all() or not np.isfinite(rets).all():
        raise ValueError("positions and returns must be finite")
    if (np.abs(pos) > 1.0).any():
        raise ValueError("positions must be between -1 and 1")

    frame = pd.DataFrame({
        "pair": positions.index.get_level_values(0).to_numpy(),
        "date": positions.index.get_level_values(1).to_numpy(),
        "position": pos,
        "return": rets,
    })
    frame["notional"] = frame["position"] * pair_gross
    frame["turnover"] = frame.groupby("pair", sort=False)["notional"].diff()
    first = frame.groupby("pair", sort=False).cumcount().eq(0)
    frame.loc[first, "turnover"] = frame.loc[first, "notional"]
    frame["turnover"] = frame["turnover"].abs()
    frame["gross_pnl"] = frame["notional"] * frame["return"]
    daily = frame.groupby("date", sort=True).agg(
        gross_pnl=("gross_pnl", "sum"), turnover=("turnover", "sum"),
        gross_exposure=("notional", lambda values: values.abs().sum()),
        net_exposure=("notional", "sum"),
    )
    if max_gross is not None and (daily["gross_exposure"] > max_gross + 1e-12).any():
        raise ValueError("portfolio gross exposure exceeds max gross exposure")
    daily["cost"] = daily["turnover"] * (cost_bps / 10000.0)
    daily["net_pnl"] = daily["gross_pnl"] - daily["cost"]
    daily["equity"] = (1.0 + daily["net_pnl"]).cumprod()
    return daily.loc[:, _COLUMNS]
