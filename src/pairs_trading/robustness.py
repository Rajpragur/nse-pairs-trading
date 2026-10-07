from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import backtest_spread


def _summarize(cost_bps: float, result) -> dict[str, float]:
    equity = result.equity
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return {
        "cost_bps": float(cost_bps),
        "total_return": float(equity.iloc[-1] - 1.0) if len(equity) else 0.0,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "turnover": float(result.turnover.sum()),
        "total_cost": float(result.costs.sum()),
        "trade_events": int((result.turnover > 0).sum()),
    }


def cost_sweep(
    zscore: pd.Series,
    open_a: pd.Series,
    open_b: pd.Series,
    hedge_ratio: pd.Series | float,
    *,
    cost_bps: list[float] | tuple[float, ...] | np.ndarray,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
) -> pd.DataFrame:
    """Evaluate one identical signal/price path across explicit cost levels."""
    if isinstance(cost_bps, (str, bytes)):
        raise TypeError("cost_bps must be a sequence of numeric levels")
    levels = [float(value) for value in cost_bps]
    if not levels or any(not np.isfinite(value) or value < 0 for value in levels):
        raise ValueError("cost_bps must contain finite non-negative values")
    rows = []
    for level in levels:
        result = backtest_spread(zscore, entry_z=entry_z, exit_z=exit_z, cost_bps=level,
                                 open_a=open_a, open_b=open_b, hedge_ratio=hedge_ratio)
        rows.append(_summarize(level, result))
    return pd.DataFrame(rows)


def randomized_controls(
    zscore: pd.Series,
    open_a: pd.Series,
    open_b: pd.Series,
    hedge_ratio: pd.Series | float,
    *,
    cost_bps: float,
    permutations: int = 100,
    block_length: int = 5,
    seed: int = 0,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
) -> pd.DataFrame:
    """Circularly permute contiguous blocks of signal order as a null control.

    Prices and hedge ratios stay fixed, so this is a sensitivity/control
    diagnostic, not an exchangeable statistical test or a market-result claim.
    """
    if not np.isscalar(cost_bps) or not np.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError("cost_bps must be finite and non-negative")
    if permutations < 1:
        raise ValueError("permutations must be >= 1")
    if block_length < 1:
        raise ValueError("block_length must be >= 1")
    if not zscore.index.is_monotonic_increasing or zscore.index.has_duplicates:
        raise ValueError("signals require unique increasing timestamps")
    n = len(zscore)
    if n < 2:
        raise ValueError("at least two signal observations are required")
    values = zscore.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    chunks = [np.arange(i, min(i + block_length, n)) for i in range(0, n, block_length)]
    rows = []
    for replicate in range(permutations):
        order = np.concatenate([chunks[i] for i in rng.permutation(len(chunks))])
        permuted = pd.Series(values[order], index=zscore.index, name=zscore.name)
        result = backtest_spread(permuted, entry_z=entry_z, exit_z=exit_z, cost_bps=cost_bps,
                                 open_a=open_a, open_b=open_b, hedge_ratio=hedge_ratio)
        rows.append({"replicate": replicate, **_summarize(cost_bps, result),
                     "control_type": "circular_block_permutation",
                     "null_description": "signal blocks reordered; prices and hedge path fixed; descriptive sensitivity only"})
    return pd.DataFrame(rows)
