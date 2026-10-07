from __future__ import annotations

import argparse
from pathlib import Path

from .angelone import AngelOneClient, CandleDataMode, CandleRequest, write_candles_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Angel One historical candles into parquet")
    parser.add_argument("--symbol", required=True, help="Symbol label for data provenance")
    parser.add_argument("--token", required=True, help="Angel One instrument token")
    parser.add_argument("--exchange", default="NSE")
    parser.add_argument("--interval", default="ONE_DAY")
    parser.add_argument("--from", dest="from_date", required=True, help="YYYY-MM-DD HH:MM")
    parser.add_argument("--to", dest="to_date", required=True, help="YYYY-MM-DD HH:MM")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate request and create an empty-schema artifact without credentials or API calls")
    parser.add_argument("--mode", choices=[mode.value for mode in CandleDataMode], default=CandleDataMode.RAW.value,
                        help="Raw SmartAPI bars, or local split-adjustment using --corporate-action-file")
    parser.add_argument("--corporate-action-file", type=Path, help="CSV with effective_date,split_factor for split adjustments")
    parser.add_argument("--min-request-interval", type=float, default=1.0, help="Minimum seconds between API attempts")
    parser.add_argument("--product-entitlement", required=True, help="Documented product/entitlement for retention and analysis")
    parser.add_argument("--adjustment-provenance", required=True, help="Adjustment/corporate-action provenance")
    parser.add_argument("--universe-provenance", required=True, help="Point-in-time universe provenance")
    args = parser.parse_args()
    if args.dry_run:
        print(f"Dry run: would fetch {args.exchange}:{args.symbol} token={args.token} "
              f"interval={args.interval} from={args.from_date} to={args.to_date} mode={args.mode} "
              f"output={args.out}")
        return
    request = CandleRequest(args.exchange, args.token, args.interval, args.from_date, args.to_date)
    mode = CandleDataMode(args.mode)
    if mode is CandleDataMode.ADJUSTED and not args.corporate_action_file:
        parser.error("--mode adjusted requires --corporate-action-file; SmartAPI adjusted candles are not assumed")
    client = AngelOneClient(dry_run=args.dry_run, min_request_interval=args.min_request_interval)
    try:
        frame = client.fetch(request, mode=mode, corporate_action_file=args.corporate_action_file)
        output = write_candles_parquet(
            frame, args.out, symbol=args.symbol, symbol_token=args.token, request=request,
            product_entitlement=args.product_entitlement,
            adjustment_provenance=args.adjustment_provenance,
            universe_provenance=args.universe_provenance,
        )
        print(f"Saved {len(frame)} candle rows to {output}")
        print(f"Manifest: {output.with_suffix(output.suffix + '.manifest.txt')}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
