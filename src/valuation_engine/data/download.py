"""Resumable batch download of raw Yahoo data into the local cache.

Usage::

    python -m valuation_engine.data.download --all            # whole universe
    python -m valuation_engine.data.download ATCO-A.ST SAND.ST
    python -m valuation_engine.data.download --all --refresh  # re-download
    python -m valuation_engine.data.download --peers-only     # rebuild table
    python -m valuation_engine.data.download --freeze ATCO-A.ST SEB-A.ST

Tickers that are already cached are skipped, so an interrupted run is simply
started again. Failures are appended to ``download_failures.csv`` in the
cache directory; one broken ticker never aborts the batch. After the
download the peer table (``peer_snapshots.csv``) and a coverage summary
(``download_summary.json``) are rebuilt from the cache.
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from . import yahoo
from .errors import DataUnavailableError, RateLimitError
from .loader import (
    _fx_lookup,
    build_peer_snapshots,
    load_universe_entries,
    write_peer_table,
)
from .store import Store, cache_store, fixture_store, normalize

logger = logging.getLogger("valuation_engine.data.download")

#: Pause after this many consecutive rate-limit failures.
RATE_LIMIT_STREAK = 3
COOL_DOWN_SECONDS = 300.0


def _log_failure(store: Store, ticker: str, stage: str, error: BaseException | str) -> None:
    path = store.root / "download_failures.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.is_file()
    kind = error if isinstance(error, str) else type(error).__name__
    message = "" if isinstance(error, str) else str(error)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        if new:
            writer.writerow(["timestamp", "ticker", "stage", "error", "message"])
        stamp = datetime.now(UTC).replace(microsecond=0).isoformat()
        writer.writerow([stamp, ticker, stage, kind, message[:500]])


def download_tickers(
    tickers: Sequence[str], *, refresh: bool = False, cool_down: float = COOL_DOWN_SECONDS
) -> dict[str, str]:
    """Download raw payloads into the cache. Returns ticker -> error for failures."""
    store = cache_store()
    failures: dict[str, str] = {}
    try:
        store.write_benchmark(normalize(yahoo.fetch_benchmark()))
    except Exception as exc:  # noqa: BLE001 - keep going with a cached benchmark
        logger.warning("benchmark download failed: %s", exc)
        _log_failure(store, "(benchmark)", "download", exc)

    streak = 0
    started = time.monotonic()
    done = 0
    for index, ticker in enumerate(tickers, start=1):
        if not refresh and store.has_company(ticker):
            continue
        try:
            payload = normalize(yahoo.fetch_company(ticker, fx_lookup=_fx_lookup))
            store.write_company(ticker, payload)
            streak = 0
            done += 1
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:  # noqa: BLE001 - one ticker must not abort the batch
            failures[ticker] = f"{type(exc).__name__}: {exc}"
            _log_failure(store, ticker, "download", exc)
            logger.warning("[%d/%d] %s FAILED: %s", index, len(tickers), ticker, failures[ticker])
            if isinstance(exc, RateLimitError):
                streak += 1
                if streak >= RATE_LIMIT_STREAK:
                    logger.warning(
                        "rate limited %d times in a row; pausing %.0fs", streak, cool_down
                    )
                    time.sleep(cool_down)
                    streak = 0
            continue
        if done % 25 == 0:
            rate = (time.monotonic() - started) / done
            logger.info("[%d/%d] %s ok (%.1fs per ticker)", index, len(tickers), ticker, rate)
    return failures


def rebuild_peer_table(tickers: Sequence[str]) -> dict[str, Any]:
    """Rebuild ``peer_snapshots.csv`` from the cache; return a coverage summary."""
    store = cache_store()
    rows, failures = build_peer_snapshots(tickers, cache_only=True)
    write_peer_table(store, rows)
    for ticker, reason in failures.items():
        if "not in the local cache" not in reason:
            _log_failure(store, ticker, "build", reason)
    snapshots = [snapshot for snapshot, _ in rows]
    extras = [extra for _, extra in rows]
    summary: dict[str, Any] = {
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "universe_companies": len(tickers),
        "loaded": len(rows),
        "not_downloaded": sorted(t for t, r in failures.items() if "not in the local cache" in r),
        "unusable": {
            t: r for t, r in sorted(failures.items()) if "not in the local cache" not in r
        },
        "financial_or_real_estate": sum(1 for s in snapshots if s.is_financial),
        "missing_ebitda": sum(1 for s in snapshots if s.ebitda is None),
        "negative_ebitda": sum(1 for s in snapshots if s.ebitda is not None and s.ebitda <= 0),
        "missing_revenue": sum(1 for s in snapshots if s.revenue is None),
        "missing_sector": sum(1 for s in snapshots if not s.sector),
        "missing_beta": sum(1 for s in snapshots if s.beta is None),
        "non_sek_reporting": sum(1 for e in extras if e["reporting_currency"] != "SEK"),
        "non_sek_trading": sum(1 for e in extras if e["currency"] != "SEK"),
        "fx_riksbank_per_year": sum(1 for e in extras if e["fx_source"].startswith("Riksbank")),
        "fx_spot_fallback": sum(
            1
            for e in extras
            if e["reporting_currency"] != e["currency"]
            and not e["fx_source"].startswith("Riksbank")
        ),
    }
    store.write_json(store.root / "download_summary.json", summary)
    return summary


def freeze_fixtures(tickers: Sequence[str]) -> None:
    """Copy selected raw files from the cache into the fixtures directory."""
    cache, fixtures = cache_store(), fixture_store()
    (fixtures.root / "companies").mkdir(parents=True, exist_ok=True)
    for ticker in tickers:
        source = cache.company_path(ticker)
        if not source.is_file():
            raise DataUnavailableError(f"{ticker}: not in the cache, cannot freeze")
        shutil.copyfile(source, fixtures.company_path(ticker))
    for source, target in (
        (cache.universe_path, fixtures.universe_path),
        (cache.benchmark_path, fixtures.benchmark_path),
        (cache.peers_path, fixtures.peers_path),
    ):
        if not source.is_file():
            raise DataUnavailableError(f"{source.name}: not in the cache, cannot freeze")
        shutil.copyfile(source, target)
    riksbank_file = cache.root / "riksbank.json"
    if riksbank_file.is_file():  # exchange rates and the government bond yield
        shutil.copyfile(riksbank_file, fixtures.root / "riksbank.json")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m valuation_engine.data.download", description=__doc__.split("\n\n")[0]
    )
    parser.add_argument("tickers", nargs="*", help="Yahoo tickers, e.g. ATCO-A.ST")
    parser.add_argument("--all", action="store_true", help="download the whole universe")
    parser.add_argument("--refresh", action="store_true", help="re-download cached tickers")
    parser.add_argument("--refresh-universe", action="store_true", help="re-run the screener")
    parser.add_argument("--limit", type=int, default=None, help="only the first N tickers")
    parser.add_argument("--peers-only", action="store_true", help="only rebuild the peer table")
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="copy the given tickers plus universe, benchmark and peer table to the fixtures",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.freeze:
        freeze_fixtures([t.upper() for t in args.tickers])
        print(f"fixtures written to {fixture_store().root}")
        return 0

    entries, stats = load_universe_entries(refresh=args.refresh_universe)
    universe = [entry.ticker for entry in entries]
    targets = universe if (args.all or args.peers_only) else [t.upper() for t in args.tickers]
    if args.limit is not None:
        targets = targets[: args.limit]
    if not targets:
        parser.error("give tickers, --all or --peers-only")

    failures: dict[str, str] = {}
    if not args.peers_only:
        failures = download_tickers(targets, refresh=args.refresh)
    print(
        f"screener symbols: {stats.screener_symbols}, quoteType EQUITY: {stats.equity_quote_type}, "
        f"ordinary shares: {stats.ordinary_share_symbols}, companies: {stats.companies}"
    )
    print(f"download failures this run: {len(failures)}")
    if args.all or args.peers_only:
        summary = rebuild_peer_table(universe)
        for key, value in summary.items():
            shown = len(value) if isinstance(value, (list, dict)) else value
            print(f"{key}: {shown}")
    print(f"cache: {cache_store().root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
