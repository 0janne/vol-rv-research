"""Command line entry point: volrv <command>."""

from __future__ import annotations

import argparse
import json
import sys

from .config import DATABASE_URL


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="volrv", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init-db", help="apply sql/ migrations (idempotent)")

    p_ing = sub.add_parser("ingest", help="load source data")
    p_ing.add_argument("--source", choices=["cboe", "underlying", "all"], default="all")

    sub.add_parser("build-features", help="compute realized vol and features")

    p_bt = sub.add_parser("backtest", help="run strategies and print a summary")
    p_bt.add_argument("--json", action="store_true", help="emit machine-readable output")

    sub.add_parser("report", help="run everything and regenerate reports/figures")

    p_sn = sub.add_parser("snapshot", help="record today's option chain")
    p_sn.add_argument("--underlying", default="SPY")

    args = parser.parse_args(argv)

    from .db import get_engine, init_db

    engine = get_engine()

    if args.cmd == "init-db":
        init_db(engine)
        print(f"schema applied -> {DATABASE_URL}")
        return 0

    if args.cmd == "ingest":
        from .ingest import load_cboe_indices, load_underlying

        if args.source in ("cboe", "all"):
            for sym, n in load_cboe_indices(engine=engine).items():
                print(f"  index_daily      {sym:<6} {n:>6} rows")
        if args.source in ("underlying", "all"):
            print(f"  underlying_daily SPX    {load_underlying(engine=engine):>6} rows")
        return 0

    if args.cmd == "snapshot":
        from .ingest import record_chain_snapshot

        n = record_chain_snapshot(args.underlying, engine)
        print(f"  option_quote {args.underlying} {n} rows")
        return 0

    from .pipeline import build_features, make_figures, run_backtests

    if args.cmd == "build-features":
        print(f"  features written: {len(build_features(engine))} rows")
        return 0

    results = run_backtests(engine)
    printable = {k: v for k, v in results.items() if not k.startswith("_")}

    if args.cmd == "backtest" and getattr(args, "json", False):
        json.dump(printable, sys.stdout, indent=2, default=str)
        print()
        return 0

    for name, stats in printable.items():
        print(f"\n{name}")
        for k, v in stats.items():
            if k == "selections":
                continue
            print(f"    {k:<16} {v:>12.4f}" if isinstance(v, float) else f"    {k:<16} {v:>12}")

    if args.cmd == "report":
        for p in make_figures(results, engine):
            print(f"  figure -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
