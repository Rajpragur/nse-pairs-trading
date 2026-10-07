# Five-symbol real-data proof: scope and limits

## Scope

This is an exploratory pipeline demonstration using five symbols only: RELIANCE-EQ, ABB-EQ, ABCAPITAL-EQ, ADANIENSOL-EQ, and ADANIENT-EQ. Requested observation window: 2018-01-01 through 2026-09-30. There are 105 requested calendar-month chunks per symbol (525 total). The audited local panel contains 10,836 rows; each symbol has saved observations in all 105 requested month chunks.

This is **not a performance claim**, trading recommendation, or evidence of a profitable strategy. Five names are not a representative NSE universe. Pair selection and performance statistics are exploratory.

## Data source and adjustments

**Resolved:** RELIANCE-EQ December 2024 was cleanly re-fetched from Angel One SmartAPI on 2026-10-07 into a separate local Parquet. Both files contain 21 rows (2024-12-02 through 2024-12-31); timestamps and all OHLCV values match exactly. The old bulk manifest retains its failed-attempt history; the independent refetch resolves the discrepancy. Both files remain local/ignored.

The combined study data are marked `raw_unadjusted_as_returned_by_smartapi`; dividend or corporate-action adjusted prices were not supplied or verified. Corporate-action discontinuities can distort log spreads and returns. No adjusted-price consistency check was possible without a trusted adjusted reference series or adjustment factors.

Data entitlement and permitted use (access, retention, derived-result publication) were not verified. Confirm applicable account/vendor terms independently before reuse or publication.

## Exact requested-month coverage

See `../nse-factor-lab/results/FIVE_SYMBOL_MANIFEST_AUDIT.md` for the detailed manifest/disk cross-check. Disk has nonempty Parquet data for all 525 requested symbol-months; there are no months without a saved file and no Parquet files without a manifest entry. No manifest-`ok` month is missing its data file. The RELIANCE-EQ December 2024 anomaly is **resolved**: a separate clean Angel One SmartAPI refetch returned 21 rows and exactly matched the existing Parquet on timestamp and all OHLCV values. The bulk manifest's historical error status remains in place as an audit trail; both raw files and sidecars stay local.

Monthly files verify at least one returned bar per month, not every exchange session. There is no authoritative NSE trading-calendar comparison here, so session-level completeness remains unverified. Basic checks on the combined panel found 10,836 rows, no duplicate `(timestamp, symbol)` keys, no invalid OHLC bounds, and no nonpositive prices or negative volume.

## Frozen study assumptions and actual run

Config: `configs/five_symbol_real_demo.json`; seed 20261006. Windows are disjoint:

- Formation: 2018-01-01–2021-12-31
- Validation: 2022-01-01–2023-12-31
- Holdout: 2024-01-01–2026-09-30

Formation-only log-price OLS estimates and Engle–Granger tests were computed across all ten candidate pairs. FDR is based on Benjamini–Hochberg q-values, and final selection additionally requires the configured rolling-beta stability filter (rolling beta standard deviation ≤ 0.5). At FDR 0.05 the eligible-and-diversified set is **empty**. At FDR 0.10 it is also **empty**. ADANIENSOL-EQ / ADANIENT-EQ was the only BH-significant candidate at 0.10 (q=0.057645), but failed rolling-beta stability (std=0.982493) and was correctly excluded. No pairs are forced into the final set. Because the corrected set is empty, there are no valid corrected-set performance or cost-sensitivity metrics to report.

The permissive FDR 1.0 run below is retained only as an illustration of what relaxed selection produces; it is not a verdict and allows shared-symbol concentration. The runner uses formation-fitted log-price parameters and formation-only spread normalization, a 60-session trailing z-score, entry 2.0 / exit 0.5, close signal with next-session-open fills and open-to-open returns. Pairwise missing sessions are omitted without forward-fill; holdout starts flat. The permissive run was corrected for round-trip accounting; only metrics verified from that corrected run are shown.

Costs are 10 bps assumed brokerage plus 5 bps assumed slippage, **15 bps per leg turnover**. These are illustrative, not verified Angel One tariff values. Statutory charges/taxes, financing, short-borrow availability/costs, market impact, and price limits are excluded.

## Earlier permissive run metrics (FDR 1.0 only; illustrative, not selected under the corrected protocol)

The prior run allowed overlapping pairs and therefore shared symbols. It used permissive FDR 1.0. The run was corrected for completed round-trip accounting; only verified metrics from that corrected run are shown. Counts include positions liquidated at window end, with forced closes marked in parentheses. Pair PnL, Sharpe and drawdown remain outputs of that deliberately permissive selection, not the final result. Counts are below five in each window; metrics are statistically meaningless.

| Window | Pair | Bars | Completed round trips (including forced close at window end) | Net PnL | Annualized Sharpe | Max drawdown |
|---|---|---:|---:|---:|---:|---:|
| Validation | ABCAPITAL-EQ / ADANIENT-EQ | 494 | 15 (2 forced) | -40.66% | -0.7118 | -47.96% |
| Holdout | ABCAPITAL-EQ / ADANIENT-EQ | 683 | 15 (2 forced) | -54.05% | -0.7156 | -59.14% |

This permissive illustration is not a performance claim and must not be conflated with the empty eligible set at FDR 0.05 and 0.10. The cost sensitivity previously run for a single FDR-0.10 candidate that failed beta stability is intentionally not presented as valid final-set evidence.



## Output artifacts

Local, ignored run outputs are under `local-data/five-symbol-real-run/`: `formation_pair_diagnostics.csv`, `window_metrics.csv`, and `run_manifest.json`. Raw market data are not committed.
