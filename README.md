# NSE Pairs Trading

A quant-development research stack for market-neutral pairs trading, combining Python research workflows with a dependency-free C++20 numerical hot path.

This project is a research/backtesting system. It does not place orders, use exchange credentials to trade, or claim live profitability. Market data is user-provided and is not bundled or committed.

## Implemented components

- Engle–Granger diagnostics, formation-only pair selection, hedge-ratio stability and optional Johansen trace diagnostics
- Strict rolling spread z-scores; explicit close-signal, next-session-open execution convention
- Two-leg self-financing accounting and per-leg turnover costs
- Cost sensitivity and deterministic randomized-block controls (descriptive, not calibrated hypothesis tests)
- Point-in-time monthly top-150 liquidity-proxy helper; this is not official Nifty membership and does not itself eliminate survivorship bias
- Python/C++ numerical parity tests and C++20 hot path
- Data-free frozen-study config and fail-closed input/provenance CLI scaffold (not a backtest runner)

`python3 -m pip install -e '.[test]'` installs the project and test dependencies. `python3 -m pip install -e '.[selection]'` installs optional statistical-selection dependencies.

## Tests and synthetic software demo

```bash
PYTHONPATH=src pytest -q
PYTHONPATH=src python3 -m pairs_trading.demo
```

The demo uses synthetic test data and writes `results/demo_summary.json`. It is only a software smoke test, never a market result.

## Frozen-study scaffold (requires your authorized local files)

`configs/frozen_study.example.json` pins a protocol template: top-150 trailing six-month historical liquidity proxy, formation/validation/frozen-holdout dates, seed, calibrated Engle–Granger settings, close-to-next-open execution, and explicit aggregate costs. It contains no real data or secrets. This proxy is not official Nifty membership and must not be described as Nifty 50 constituents.

The runner does not download data or calculate performance. It validates the required user-provided CSV and accompanying provenance manifest; without both, it refuses to proceed. Even after validation, it only emits an `inputs_validated_not_run` manifest with `results: null` until a real study engine is implemented.

```bash
python3 -m pip install -e .
# Copy the template outside the repository and edit `data.path` and `data.manifest`
# to point to your own authorized CSV and sidecar manifest.
pairs-frozen-study --config /path/to/your/frozen-study.json --validate-only
pairs-frozen-study --config /path/to/your/frozen-study.json
```

The canonical CSV schema is:

```text
timestamp,symbol,open,high,low,close,volume
```

Read `docs/METHODOLOGY.md` and `docs/DATA_RIGHTS_CHECKLIST.md`. Users bring their own data pulled under their broker account; no separate written vendor permission letter is a repository prerequisite, but the user remains responsible for following actual account/data terms on access, local retention, analysis, and publication. Keep raw market data and local manifests in gitignored `local-data/`; never commit them. The historical top-150 liquidity proxy is not official Nifty membership and remains subject to survivorship bias if delisted/renamed securities are missing. No `RESULTS.md` or performance values exist because there is no real dataset or executed study yet.

## Local ingestion / optional fetch

The fetch command requires an explicit symbol token. `pairs-fetch-angelone --dry-run` prints a fetch plan and does not authenticate, call the API, or write output. The fetch CLI does not yet resolve tokens from the instrument-master API; the local library does provide an injectable instrument-master parser. SmartAPI candles are raw by default; `--mode adjusted` requires a local `effective_date,split_factor` CSV and adjusts split history only, not dividends. Use only local data you are authorized to retain/analyze; `.env` is ignored and must not be committed. No fetch output is automatically committed.

## Build the C++ hot path

```bash
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/pairs_benchmark
```

This reports numerical-model diagnostics and runtime, not strategy performance.

## Limitations

- The current CLI is input/provenance validation scaffolding, not an end-to-end research evaluator. Do not interpret its status manifest as results.
- The checked-in config is deliberately data-free. `pairs-frozen-study` refuses missing local dataset/provenance inputs. With authorized data it only writes an `inputs_validated_not_run` manifest (`results: null`); it does not execute a backtest or create performance claims. There is no `RESULTS.md` because no licensed local dataset has been supplied or run.
- The top-150 proxy ranks trailing six-month median traded value at month end, effective next session. It is not official Nifty membership and remains survivorship-biased if historical delisted/renamed securities are absent.
- Pair p-values use `statsmodels.tsa.stattools.coint` MacKinnon values with parity tests. They are calibrated for each specified pair test, not corrected for data-dependent pair searching or researcher degrees of freedom; BH does not eliminate that selection bias.
- Aggregate cost bps are sensitivities, not a complete NSE fee/tax, borrow, impact or slippage model. Cash-market short availability and constraints must be modeled for a real study.
