import numpy as np
import pandas as pd
import pytest

from pairs_trading.selection import (
    benjamini_hochberg,
    diagnose_pair,
    select_pairs,
    select_monthly_top_universe,
)


def _cointegrated_fixture(seed=42, n=360):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-01", periods=n)
    common = np.cumsum(rng.normal(size=n))
    a = pd.Series(100 + common, index=index)
    b = pd.Series(5 + 1.4 * common + rng.normal(0, 0.15, n), index=index)
    c = pd.Series(80 + np.cumsum(rng.normal(size=n)), index=index)
    return pd.DataFrame({"A": a, "B": b, "C": c})


def test_bh_adjustment_is_monotone_bounded_and_order_preserving():
    p = pd.Series([0.01, 0.04, 0.03, 0.8], index=["a", "b", "c", "d"])
    adjusted = benjamini_hochberg(p)
    assert np.allclose(adjusted.loc[["a", "c", "b", "d"]], [0.04, 0.05333333333333334, 0.05333333333333334, 0.8])
    assert ((adjusted >= 0) & (adjusted <= 1)).all()
    assert np.isnan(benjamini_hochberg(pd.Series([0.1, np.nan])).iloc[1])


def test_pair_selection_reports_all_pairs_and_corrected_pvalues_without_future_data():
    prices = _cointegrated_fixture()
    chosen = select_pairs(prices, formation_end=prices.index[249], fdr=0.1, min_observations=100)
    assert len(chosen.pairs) == 1
    assert chosen.pairs[0] == ("A", "B")
    assert len(chosen.diagnostics) == 3
    assert chosen.diagnostics["q_value"].between(0, 1).all()
    assert chosen.diagnostics.loc[chosen.diagnostics.pair == "A~B", "selected"].item()
    changed = prices.copy()
    changed.iloc[250:] = changed.iloc[250:] * 1000
    selected_after_perturb = select_pairs(changed, formation_end=prices.index[249], fdr=0.1, min_observations=100)
    pd.testing.assert_frame_equal(chosen.diagnostics, selected_after_perturb.diagnostics)


def test_diagnostics_include_half_life_and_rolling_stability_flags():
    prices = _cointegrated_fixture()
    diagnostic = diagnose_pair(prices.A.iloc[:250], prices.B.iloc[:250], min_observations=100)
    assert diagnostic.observations == 250
    assert np.isfinite(diagnostic.beta)
    assert diagnostic.half_life > 0
    assert diagnostic.rolling_beta_std >= 0
    assert diagnostic.adf_pvalue >= 0 and diagnostic.adf_pvalue <= 1


def test_engle_granger_pvalue_matches_statsmodels_coint_exactly():
    statsmodels = pytest.importorskip("statsmodels")
    from statsmodels.tsa.stattools import coint

    prices = _cointegrated_fixture(seed=17, n=240)
    y, x = prices.A.iloc[:200], prices.B.iloc[:200]
    expected = coint(y.to_numpy(), x.to_numpy(), trend="c", maxlag=None, autolag="aic")
    result = diagnose_pair(y, x, min_observations=100)
    assert result.adf_t_stat == pytest.approx(expected[0], abs=0, rel=0)
    assert result.adf_pvalue == pytest.approx(expected[1], abs=0, rel=0)
    assert result.p_value_kind == "statsmodels_mackinnon_engle_granger"


def test_engle_granger_trend_lag_options_match_statsmodels():
    pytest.importorskip("statsmodels")
    from statsmodels.tsa.stattools import coint

    prices = _cointegrated_fixture(seed=101, n=240)
    y, x = prices.A.iloc[:200], prices.B.iloc[:200]
    expected = coint(y.to_numpy(), x.to_numpy(), trend="ct", maxlag=2, autolag=None)
    result = diagnose_pair(y, x, min_observations=100, trend="ct", maxlag=2, autolag=None)
    assert result.adf_t_stat == pytest.approx(expected[0], abs=0, rel=0)
    assert result.adf_pvalue == pytest.approx(expected[1], abs=0, rel=0)


def test_pair_selection_labels_calibrated_pvalues_and_documents_selection_bias():
    prices = _cointegrated_fixture()
    selected = select_pairs(prices, formation_end=prices.index[249], fdr=0.1, min_observations=100)
    assert set(selected.diagnostics.p_value_kind) == {"statsmodels_mackinnon_engle_granger"}
    assert selected.diagnostics.p_value.notna().all()
    assert "selection_bias" in selected.diagnostics.attrs.get("inference_warning", "")


def test_selection_rejects_invalid_pvalues_and_duplicate_symbols():
    with pytest.raises(ValueError, match="p-values"):
        benjamini_hochberg(pd.Series([0.1, 1.1]))
    prices = pd.DataFrame([[1, 2], [2, 3]], columns=["A", "A"])
    with pytest.raises(ValueError, match="unique"):
        select_pairs(prices)


def test_johansen_is_optional_and_returns_trace_diagnostics_when_available():
    from pairs_trading.selection import johansen_diagnostic
    pytest.importorskip("statsmodels")
    prices = _cointegrated_fixture()
    result = johansen_diagnostic(prices.A.iloc[:250], prices.B.iloc[:250])
    assert result.method == "johansen"
    assert len(result.trace_statistics) == 2


def test_pair_selection_surfaces_optional_dependency_message():
    prices = _cointegrated_fixture()
    try:
        import statsmodels  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match=r"install nse-pairs-trading\[selection\]"):
            select_pairs(prices, method="johansen")


def test_monthly_universe_uses_six_month_median_and_is_effective_next_session():
    days = pd.bdate_range("2023-01-02", "2023-09-15")

    close = pd.DataFrame({"A": 10.0, "B": 20.0}, index=days)
    volume = pd.DataFrame({"A": 10.0, "B": 10.0}, index=days)
    # B outranks A through May, but is overtaken in June. Each selection
    # uses only the six completed calendar months ending at its month-end.
    volume.loc[volume.index.to_series().dt.month <= 5, "B"] = 7.0
    volume.loc[volume.index.to_series().dt.month == 6, "B"] = 0.5
    result = select_monthly_top_universe(close, volume, top_n=1, min_observations=1)
    assert result.loc[result.effective_date == pd.Timestamp("2023-06-01"), "symbols"].item() == ("B",)
    assert result.loc[result.effective_date == pd.Timestamp("2023-07-03"), "symbols"].item() == ("B",)
    june_row = result.loc[result.selection_date == pd.Timestamp("2023-06-30")].iloc[0]
    assert june_row.effective_date == pd.Timestamp("2023-07-03")
    assert june_row.scores["B"] == 140.0


def test_monthly_universe_requires_complete_coverage_and_ignores_future_mutations():
    days = pd.bdate_range("2023-01-02", "2023-08-31")
    close = pd.DataFrame({"A": 10.0, "B": 20.0}, index=days)
    volume = pd.DataFrame({"A": 10.0, "B": 10.0}, index=days)
    original = select_monthly_top_universe(close, volume, top_n=1, min_observations=20)
    changed_close, changed_volume = close.copy(), volume.copy()
    changed_volume.loc[changed_volume.index > "2023-05-31", "B"] = 100000.0
    changed = select_monthly_top_universe(changed_close, changed_volume, top_n=1, min_observations=20)
    original_row = original.loc[original.selection_date == pd.Timestamp("2023-05-31")].iloc[0]
    changed_row = changed.loc[changed.selection_date == pd.Timestamp("2023-05-31")].iloc[0]
    assert original_row.symbols == changed_row.symbols
    assert original_row.scores == changed_row.scores
    broken = volume.drop(index=days[10])
    with pytest.raises(ValueError, match="identical index"):
        select_monthly_top_universe(close, broken, top_n=1, min_observations=20)


def test_bh_significant_pair_remains_visible_when_rolling_stability_rejects_it():
    prices = _cointegrated_fixture(n=360)
    # Verify reason semantics using one calibrated candidate: a data-dependent
    # significance threshold can flag it while separate eligibility rejects it.
    result = select_pairs(prices, formation_end=prices.index[-1], fdr=1.0,
                          min_observations=100, max_rolling_beta_std=0.0)
    rows = result.diagnostics
    rejected_significant = rows[(rows.q_value <= 1.0) & (~rows.eligible)]
    assert not rejected_significant.empty
    assert (rejected_significant.selection_reason == "bh_significant_but_ineligible").all()
