# NSE Pairs Trading

A quant-development research stack for market-neutral pairs trading, combining Python research workflows with a dependency-free C++20 numerical hot path.

This project is intentionally a research/backtesting system. It does not place orders, use exchange credentials, or claim live profitability.

## What it implements

- Engle–Granger two-step cointegration regression
- Residual ADF-style stationarity statistic
- Time-varying hedge-ratio estimation with a Kalman filter
- Strict rolling spread z-scores
- Entry/exit state machine with position limits represented by a single spread position
- One-way turnover and transaction-cost deductions
- Deterministic synthetic smoke-test data
- C++20 Engle–Granger implementation and benchmark target
- Python/C++-friendly separation for numerical parity testing

## Run the Python tests

```bash
PYTHONPATH=src pytest -q
```

## Run the synthetic demo

```bash
PYTHONPATH=src python3 -m pairs_trading.demo
```

The demo writes `results/demo_summary.json`. Its data is synthetic and is only used to verify the pipeline.

## Build the C++ hot path

```bash
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/pairs_benchmark
```

The C++ benchmark reports the estimated intercept, hedge ratio, ADF-style statistic, observation count, and elapsed runtime. It is a numerical benchmark, not a claim of trading performance.

## Real-data schema

The future ingestion layer expects timestamped pair data with at least:

```text
timestamp,symbol,open,high,low,close,volume
```

The intended research workflow is:

1. Select liquid, economically related NSE pairs.
2. Freeze formation and trading windows before evaluation.
3. Estimate OLS and Kalman hedge ratios only on the formation window.
4. Run walk-forward trading on later data.
5. Include brokerage, STT, exchange charges, stamp duty, slippage, turnover, and position limits.
6. Report spread stability, turnover, drawdown, and failure regimes.

## Limitations

- No real NSE data is bundled.
- The ADF implementation currently exposes the regression statistic; calibrated critical values and p-values should be added before making statistical claims.
- The synthetic demo is not a backtest result.
- Cash-market short-sale constraints and instrument availability must be modeled for any real NSE study.
