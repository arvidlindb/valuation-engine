"""Command line interface of the valuation engine.

Examples::

    python -m valuation_engine ATCO-A.ST
    python -m valuation_engine ATCO-A.ST SEB-A.ST --offline --simulations 20000
    python -m valuation_engine --all
    python -m valuation_engine --all --limit 20 --with-reports

One or more tickers: prints a summary per company and writes charts (PNG and
SVG), the Excel workbook and ``summary.json`` to ``<output-dir>/<TICKER>/``.
``--all``: values the whole universe and writes ``universe_valuation.csv`` and
``universe_summary.md`` to ``<output-dir>/``.

Exit code: 0 on success, 1 when a ticker could not be loaded (or, with
``--all``, when a company failed for a reason other than missing data), 2 for
invalid arguments.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from valuation_engine import __version__
from valuation_engine.batch import DEFAULT_BATCH_SIMULATIONS, STATUS_ERROR, run_universe
from valuation_engine.data import DataError
from valuation_engine.pipeline import run_valuation, save_report, summary_text

DEFAULT_SIMULATIONS = 10_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m valuation_engine",
        description=(
            "Value Swedish listed companies with a DCF, trading comps, an LBO and a "
            "Monte Carlo simulation."
        ),
    )
    parser.add_argument("tickers", nargs="*", metavar="TICKER", help="Yahoo ticker, e.g. ATCO-A.ST")
    parser.add_argument(
        "--all",
        action="store_true",
        dest="all_companies",
        help="value every company of the universe (writes a CSV and a summary)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="folder for the results (default: output)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="read only the local fixtures and cache (same as VALUATION_ENGINE_OFFLINE=1)",
    )
    parser.add_argument(
        "--refresh", action="store_true", help="download fresh data instead of using the cache"
    )
    parser.add_argument("--no-charts", action="store_true", help="do not write the charts")
    parser.add_argument("--no-excel", action="store_true", help="do not write the Excel workbook")
    parser.add_argument(
        "--simulations",
        type=int,
        default=None,
        metavar="N",
        help=(
            f"Monte Carlo simulations per company (default: {DEFAULT_SIMULATIONS:,}, "
            f"with --all: {DEFAULT_BATCH_SIMULATIONS:,})"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed of the simulation (default: 42); a negative value draws a fresh seed",
    )
    parser.add_argument(
        "--no-size-premium",
        action="store_true",
        help="plain CAPM: no size premium in the DCF cost of equity",
    )
    parser.add_argument(
        "--universe-peers",
        action="store_true",
        help=(
            "let the DCF take peer beta and exit multiple from the full universe "
            "(same industry, else sector) instead of the comps peer group"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help=(
            "with --all: value only the first N companies (for testing); the files get "
            "the suffix _limit<N> so that a full run is not overwritten"
        ),
    )
    parser.add_argument(
        "--with-reports",
        action="store_true",
        help="with --all: also write charts and the Excel workbook per company (slow)",
    )
    parser.add_argument("--quiet", action="store_true", help="no progress output with --all")
    parser.add_argument("--version", action="version", version=f"valuation-engine {__version__}")
    return parser


def _run_tickers(args: argparse.Namespace, seed: int | None) -> int:
    simulations = args.simulations if args.simulations is not None else DEFAULT_SIMULATIONS
    failed: list[str] = []
    for raw in args.tickers:
        ticker = raw.strip().upper()
        try:
            report = run_valuation(
                ticker,
                offline=args.offline,
                refresh=args.refresh,
                n_simulations=simulations,
                seed=seed,
                size_premium=not args.no_size_premium,
                use_comps_peers=not args.universe_peers,
            )
        except DataError as exc:
            failed.append(ticker)
            failed[-1] = shown = ticker or repr(raw)
            print(f"{shown}: no data, the company could not be loaded. {exc}", file=sys.stderr)
            continue
        folder = args.output_dir / ticker
        written = save_report(report, folder, charts=not args.no_charts, excel=not args.no_excel)
        print(summary_text(report))
        print(f"Files in {folder}:")
        if written["excel"]:
            print(f"  {written['excel'][0].name}")
        if written["charts"]:
            names = sorted({path.stem for path in written["charts"]})
            print(f"  charts/ ({len(names)} charts as PNG and SVG): {', '.join(names)}")
        print(f"  {written['summary'][0].name}")
        print(f"Done in {report.timings.get('total', float('nan')):.1f} s (valuation only).")
        print()
    if failed:
        print(f"Could not be loaded: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


def _run_all(args: argparse.Namespace, seed: int | None) -> int:
    simulations = args.simulations if args.simulations is not None else DEFAULT_BATCH_SIMULATIONS
    result = run_universe(
        offline=args.offline,
        refresh=args.refresh,
        output_dir=args.output_dir,
        n_simulations=simulations,
        seed=seed,
        size_premium=not args.no_size_premium,
        use_comps_peers=not args.universe_peers,
        limit=args.limit,
        with_reports=args.with_reports,
        charts=not args.no_charts,
        excel=not args.no_excel,
        progress=not args.quiet,
    )
    counts = result.status_counts
    shown = ", ".join(f"{status} {count}" for status, count in sorted(counts.items()))
    print(f"{len(result.frame)} companies in {result.seconds:.0f} s: {shown}")
    print(f"Wrote {result.csv_path}")
    print(f"Wrote {result.summary_path}")
    return 1 if counts.get(STATUS_ERROR, 0) else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.all_companies and args.tickers:
        parser.error("give either tickers or --all, not both")
    if not args.all_companies and not args.tickers:
        parser.error("give at least one ticker, or --all")
    if args.simulations is not None and args.simulations < 1:
        parser.error("--simulations must be at least 1")
    if not args.all_companies and (args.limit is not None or args.with_reports):
        parser.error("--limit and --with-reports require --all")
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")
    seed = args.seed if args.seed >= 0 else None
    if args.all_companies:
        return _run_all(args, seed)
    return _run_tickers(args, seed)


if __name__ == "__main__":
    sys.exit(main())
