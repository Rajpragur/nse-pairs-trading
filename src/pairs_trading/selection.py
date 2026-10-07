from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from .models import engle_granger


@dataclass(frozen=True)
class PairDiagnostic:
    pair: str
    observations: int
    intercept: float
    beta: float
    adf_t_stat: float
    adf_pvalue: float
    half_life: float
    rolling_beta_std: float
    rolling_beta_min: float
    rolling_beta_max: float
    eligible: bool
    reason: str
    p_value_kind: str = "statsmodels_mackinnon_engle_granger"


@dataclass(frozen=True)
class PairSelection:
    pairs: tuple[tuple[str, str], ...]
    diagnostics: pd.DataFrame
    formation_end: pd.Timestamp
    method: str


def benjamini_hochberg(p_values: pd.Series | list[float] | np.ndarray) -> pd.Series:
    """Benjamini-Hochberg adjusted p-values, retaining missing entries."""
    if np.isscalar(p_values):
        raise TypeError("p_values must be a Series, list, or ndarray")
    values = pd.Series(p_values, copy=True, dtype=float)
    finite = values.dropna()
    if not np.isfinite(finite.to_numpy()).all() or ((finite < 0) | (finite > 1)).any():
        raise ValueError("p-values must be finite and lie in [0, 1]")
    ordered = finite.sort_values(kind="mergesort")
    count = len(ordered)
    if count:
        ranks = np.arange(1, count + 1, dtype=float)
        adjusted = ordered.to_numpy() * count / ranks
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        result = pd.Series(np.nan, index=values.index, dtype=float)
        result.loc[ordered.index] = np.clip(adjusted, 0.0, 1.0)
    else:
        result = pd.Series(np.nan, index=values.index, dtype=float)
    result.name = "q_value"
    return result


def _engle_granger_coint(y: np.ndarray, x: np.ndarray, *, trend: str, maxlag: int | None,
                         autolag: str | None) -> tuple[float, float]:
    """Return statsmodels' Engle-Granger statistic and MacKinnon p-value."""
    try:
        from statsmodels.tsa.stattools import coint
    except ImportError as exc:
        raise ImportError("Engle-Granger selection requires optional 'statsmodels'; install nse-pairs-trading[selection]") from exc
    statistic, pvalue, _ = coint(y, x, trend=trend, maxlag=maxlag, autolag=autolag)
    return float(statistic), float(pvalue)


def _half_life(residual: pd.Series) -> float:
    lagged = residual.shift(1).dropna().to_numpy(dtype=float)
    delta = residual.diff().dropna().to_numpy(dtype=float)
    if len(lagged) < 3:
        return float("nan")
    design = np.column_stack((np.ones(len(lagged)), lagged))
    gamma = float(np.linalg.lstsq(design, delta, rcond=None)[0][1])
    if not np.isfinite(gamma) or gamma >= 0:
        return float("inf")
    return float(-np.log(2.0) / gamma)


def diagnose_pair(
    y: pd.Series,
    x: pd.Series,
    *,
    pair: str | None = None,
    min_observations: int = 60,
    rolling_window: int = 60,
    max_rolling_beta_std: float = float("inf"),
    trend: str = "c",
    maxlag: int | None = None,
    autolag: str | None = "aic",
) -> PairDiagnostic:
    if min_observations < 30 or rolling_window < 2:
        raise ValueError("min_observations must be >= 30 and rolling_window >= 2")
    aligned = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if not aligned.index.is_monotonic_increasing or aligned.index.has_duplicates:
        raise ValueError("pair observations require unique increasing timestamps")
    label = pair or f"{y.name or 'y'}~{x.name or 'x'}"
    n = len(aligned)
    if n < min_observations:
        return PairDiagnostic(label, n, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                              False, f"observations<{min_observations}")
    result = engle_granger(aligned.y, aligned.x)
    coint_t, coint_p = _engle_granger_coint(aligned.y.to_numpy(dtype=float), aligned.x.to_numpy(dtype=float),
                                            trend=trend, maxlag=maxlag, autolag=autolag)
    rolling_betas = aligned.y.rolling(rolling_window, min_periods=rolling_window).cov(aligned.x) / aligned.x.rolling(
        rolling_window, min_periods=rolling_window
    ).var()
    rolling_betas = rolling_betas.replace([np.inf, -np.inf], np.nan).dropna()
    beta_std = float(rolling_betas.std(ddof=0)) if len(rolling_betas) else float("nan")
    beta_min = float(rolling_betas.min()) if len(rolling_betas) else float("nan")
    beta_max = float(rolling_betas.max()) if len(rolling_betas) else float("nan")
    half_life = _half_life(result.residual)
    eligible = bool(np.isfinite(result.beta) and np.isfinite(coint_t) and np.isfinite(coint_p)
                    and np.isfinite(beta_std) and beta_std <= max_rolling_beta_std)
    reason = "eligible" if eligible else "unstable_or_nonfinite_diagnostic"
    return PairDiagnostic(label, n, result.intercept, result.beta, coint_t,
                          coint_p, half_life,
                          beta_std, beta_min, beta_max, eligible, reason)


def select_pairs(
    prices: pd.DataFrame,
    *,
    formation_end: str | pd.Timestamp | None = None,
    formation_start: str | pd.Timestamp | None = None,
    fdr: float = 0.05,
    min_observations: int = 60,
    rolling_window: int = 60,
    max_rolling_beta_std: float = float("inf"),
    method: str = "engle_granger",
    trend: str = "c",
    maxlag: int | None = None,
    autolag: str | None = "aic",
) -> PairSelection:
    """Rank all unique symbol pairs using formation-only diagnostics and BH.

    The input panel must have dated rows and symbol columns. Pairwise missing
    observations are dropped, never forward-filled. The output records every
    tested candidate, not only selected pairs. Features/test statistic are
    fit on the specified formation window only.
    """
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError("prices must be a non-empty wide DataFrame")
    if prices.columns.has_duplicates:
        raise ValueError("symbol columns must be unique")
    if len(prices.columns) < 2:
        raise ValueError("at least two unique symbols are required")
    if not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
        raise ValueError("price panel requires unique increasing timestamps")
    if not 0 < fdr <= 1:
        raise ValueError("fdr must lie in (0, 1]")
    if method not in {"engle_granger", "johansen"}:
        raise ValueError("method must be engle_granger or johansen")
    if method == "johansen":
        try:
            import statsmodels  # noqa: F401
        except ImportError as exc:
            raise ImportError("Johansen selection requires optional 'statsmodels'; install nse-pairs-trading[selection]") from exc
    end = pd.Timestamp(formation_end) if formation_end is not None else pd.Timestamp(prices.index[-1])
    start = pd.Timestamp(formation_start) if formation_start is not None else pd.Timestamp(prices.index[0])
    if start > end:
        raise ValueError("formation_start must be <= formation_end")
    formation = prices.loc[(prices.index >= start) & (prices.index <= end)]
    if formation.empty:
        raise ValueError("formation window contains no observations")
    rows: list[dict] = []
    for first, second in combinations(prices.columns, 2):
        label = f"{first}~{second}"
        if method == "johansen":
            item = johansen_diagnostic(formation[first], formation[second], min_observations=min_observations)
            # Trace p-values require sample-size-specific critical values; expose
            # no invented p-values and do not apply BH to Johansen trace stats.
            rows.append({**asdict(item), "p_value": np.nan, "q_value": np.nan, "selected": False,
                         "selection_reason": "johansen_trace_requires_calibrated_rank_rule"})
        else:
            item = diagnose_pair(formation[first], formation[second], pair=label,
                                 min_observations=min_observations, rolling_window=rolling_window,
                                 max_rolling_beta_std=max_rolling_beta_std,
                                 trend=trend, maxlag=maxlag, autolag=autolag)
            rows.append({**asdict(item), "p_value": item.adf_pvalue, "q_value": np.nan,
                         "selected": False, "selection_reason": item.reason})
    diagnostics = pd.DataFrame(rows)
    if method == "engle_granger":
        diagnostics.attrs["inference_warning"] = (
            "MacKinnon Engle-Granger p-values are calibrated for the specified pair test, but BH after searching "
            "a data-dependent pair family does not cure selection_bias or researcher degrees of freedom."
        )
    if method == "engle_granger" and not diagnostics.empty:
        valid_p = diagnostics["p_value"].dropna()
        if valid_p.empty:
            raise ValueError("no eligible pair has enough observations for selection")
        diagnostics["q_value"] = benjamini_hochberg(diagnostics["p_value"])
        significant = diagnostics["q_value"].notna() & (diagnostics["q_value"] <= fdr)
        eligible = diagnostics["eligible"] & significant
        diagnostics.loc[eligible, "selected"] = True
        diagnostics.loc[eligible, "selection_reason"] = "eligible_and_bh_significant"
        diagnostics.loc[significant & ~diagnostics["eligible"], "selection_reason"] = "bh_significant_but_ineligible"
        diagnostics.loc[diagnostics["eligible"] & ~significant, "selection_reason"] = "not_bh_significant"
    selected = tuple(tuple(label.split("~", 1)) for label in diagnostics.loc[diagnostics.selected, "pair"])
    return PairSelection(selected, diagnostics, end, method)
def select_monthly_top_universe(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    *,
    top_n: int = 150,
    lookback_months: int = 6,
    min_observations: int = 60,
) -> pd.DataFrame:
    """Select symbols by trailing monthly median turnover; effective next session."""
    if not close.index.equals(volume.index) or not close.columns.equals(volume.columns):
        raise ValueError("close and volume must have identical index and columns")
    if not close.index.is_monotonic_increasing or close.index.has_duplicates:
        raise ValueError("price panel requires unique increasing timestamps")
    if top_n < 1 or lookback_months < 1 or min_observations < 1:
        raise ValueError("top_n, lookback_months, and min_observations must be positive")
    turnover = close * volume
    rows = []
    month_ends = close.groupby(close.index.to_period("M")).tail(1).index
    for selection_date in month_ends:
        window_start = selection_date - pd.DateOffset(months=lookback_months) + pd.offsets.MonthBegin(1)
        window = turnover.loc[(turnover.index >= window_start) & (turnover.index <= selection_date)]
        coverage = window.notna().sum()
        scores = window.median().where(coverage >= min_observations).dropna().sort_values(ascending=False, kind="mergesort")
        symbols = tuple(scores.head(top_n).index)
        future = close.index[close.index > selection_date]
        if len(future):
            rows.append({"selection_date": selection_date, "effective_date": future[0], "symbols": symbols,
                         "scores": scores.to_dict()})
    return pd.DataFrame(rows, columns=["selection_date", "effective_date", "symbols", "scores"])


@dataclass(frozen=True)
class JohansenDiagnostic:
    method: str
    observations: int
    trace_statistics: tuple[float, ...]
    critical_values_95: tuple[float, ...]


def johansen_diagnostic(y: pd.Series, x: pd.Series, *, det_order: int = 0, k_ar_diff: int = 1,
                        min_observations: int = 60) -> JohansenDiagnostic:
    """Optional statsmodels Johansen trace test; exposes stats and criticals."""
    try:
        from statsmodels.tsa.vector_ar.vecm import coint_johansen
    except ImportError as exc:
        raise ImportError("Johansen selection requires optional 'statsmodels'") from exc
    aligned = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(aligned) < min_observations:
        raise ValueError(f"at least {min_observations} overlapping observations required")
    if not aligned.index.is_monotonic_increasing or aligned.index.has_duplicates:
        raise ValueError("pair observations require unique increasing timestamps")
    result = coint_johansen(aligned.to_numpy(dtype=float), det_order, k_ar_diff)
    return JohansenDiagnostic("johansen", len(aligned), tuple(map(float, result.lr1)),
                              tuple(map(float, result.cvt[:, 1])))
