# Research methodology and conventions

## Data source, permissions and local files

All data is user-provided. No market dataset is bundled or committed. A student using their own broker API account (e.g. Upstox or Angel One) does not need to obtain a separate vendor permission letter as a repository prerequisite; the user must still follow their actual agreement's access, retention, research, and publication terms. Do not assume that API access automatically allows every downstream use. This repository does not scrape NSE pages. Never commit raw bars, credentials, or private filesystem paths; keep raw data in the gitignored local-data directory.

A dataset is not eligible for a study without a sidecar provenance manifest recording provider/source, product and entitlement, actual applicable rights/terms reviewed by the user, retrieval timestamp, coverage, checksum, schema/parser version, corporate-action adjustment treatment, and universe provenance. The run scaffold verifies the dataset checksum against this manifest and refuses absent data or incomplete provenance. This repository check is not a legal interpretation or guarantee of permitted use. Synthetic fixtures and the synthetic demo are for software tests only and must never be presented as market results.

Canonical local candle CSV schema:

```text
timestamp,symbol,open,high,low,close,volume
```

Do not forward-fill missing executable prices. Preserve raw observations immutably, represent adjustments separately, and state whether returns include distributions. Do not place local absolute dataset paths or provider credentials in checked-in configuration.

## Current-constituent pool and survivorship limitation

The intended study pool is the user-supplied CSV of **current Nifty 200 constituents**, manually downloaded from NSE, used as a fixed symbol pool over 2018-01-01 through 2026-09-30. Within that fixed pool, recompute the top-150 monthly screen by trailing six-completed-calendar-month median traded value and apply it next session. This is not historical Nifty 200 membership. Applying today's list backward creates survivorship/current-constituent bias: securities removed, delisted, renamed, or otherwise absent from the current snapshot are omitted. Label results as a current-constituent sample, quantify missingness/coverage, and never claim a survivorship-free historical index study.


The agreed study window is 2018-01-01–2026-09-30, not the template's current 2016–2024 dates. The frozen protocol must use the user-provided **current Nifty 200** CSV as the fixed symbol pool and calculate monthly top-150 trailing liquidity ranking within it. Historical use of today's constituents introduces survivorship/current-constituent bias; it is not historical index membership. No result values are available until a real dataset is provided and the full evaluator is implemented and run.

Invoke `pairs-frozen-study --help` (or `PYTHONPATH=src python -m pairs_trading.frozen_study --help`). The checked-in template intentionally points to nonexistent replacement filenames. The command must fail until the user supplies an authorized local CSV and matching manifest. With valid inputs the present scaffold only writes an `inputs_validated_not_run` manifest with `results: null`; it does not claim to run or report performance. Do not create `RESULTS.md` or performance values before an actual reproducible study is implemented and executed on user-authorized data.

## Signal availability and execution

A signal dated T may use only fields published and observable by the end of session T. A signal using T closing prices, volume, delivery, or other end-of-day fields is generated after the close and executes no earlier than the next valid session's open (T+1). The holding-period return begins at that execution price; no T-close-derived signal receives a T-close fill.

A signal at T is paired with the return from the T+1 open to the open at the chosen horizon. For a one-session horizon this is `open[T+2] / open[T+1] - 1`. Tests must assert this exact alignment including holidays and missing bars. Any alternate intraday convention requires explicit fields/timestamps and a separately documented execution model.

All rolling predictors, dollar-volume ranks, liquidity filters and universe selections must be computed only from observations available by signal time T. Perturbing data strictly after T must not alter the signal or selected universe at T.

The monthly liquidity proxy selects up to 150 securities by median daily `close * volume` over the six completed calendar months through month-end T. Membership is effective at the next supplied session. Eligibility requires at least 60 valid observations by default. This rule avoids direct use of future liquidity values, but is not official index membership and does not establish survivorship-free coverage. The supplied historical security panel must include historical listings and delisted securities where available; current constituents alone are not an acceptable historical universe. Missing/absent securities can bias returns upward and invalidate universe claims; report gaps and limitations rather than assuming completeness.

## Universe, corporate actions and identity

Use effective-dated membership or a point-in-time rule-based universe built only from prior data. Never apply today's index members to earlier dates. Retain securities through delisting and map symbol changes with effective dates and stable identifiers where available. Keep unadjusted observations immutable; store adjustment events/factors separately and state whether returns include cash dividends.

## Validation, costs and reporting

Fit pair selection, hedge ratios, factor weighting, neutralization parameters, and thresholds using formation/training data only. Validation may guide design choices, but must not be repeatedly queried as if it were untouched holdout. Evaluate a final frozen holdout once after choices are fixed, initially flat and with no carried position. Keep periods disjoint and disclose the exact dates.

Include trading costs and slippage in executable portfolio returns. The template's aggregate bps values are explicit sensitivity assumptions, not a complete NSE fee schedule; replace or supplement them with documented date- and instrument-specific brokerage, STT, exchange charges, stamp duty, borrow availability/cost, impact and slippage. Report data coverage/missingness, corporate-action ambiguity, universe/survivorship limitations, pair-selection multiplicity and researcher degrees of freedom, turnover, drawdown, failed/null outcomes, and all tested variants. No performance number is valid unless generated by a reproducible command from a manifest-pinned dataset and frozen configuration.
