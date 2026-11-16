"""Public loaders of the data layer.

Resolution order
----------------
* Offline (``offline=True`` or env ``VALUATION_ENGINE_OFFLINE=1``):
  fixtures -> local cache -> :class:`DataUnavailableError`. Never any network.
* Online: local cache (skipped when ``refresh=True``) -> network (result is
  written to the cache) -> on network failure: cache -> fixtures.

All objects are built from the raw JSON payloads (see ``store``), so a
company loaded offline is identical to the online load that produced the
payload.

Exchange rates: companies that report in another currency than they trade in
are converted with Riksbank rates per fiscal year (see ``fx``), resolved
through the same fixtures/cache/network order (file ``riksbank.json``), with
the Yahoo spot rate of the raw payload as fallback.
"""

from __future__ import annotations

import csv
import logging
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, fields
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd

from ..core.types import (
    CompanyData,
    CompanyProfile,
    MarketData,
    PeerSnapshot,
    PriceHistory,
)
from . import yahoo
from .errors import DataError, DataUnavailableError
from .fx import FxPlan, legacy_plan, plan_fx
from .mapping import (
    fiscal_period_ends,
    raw_statement_frame,
    standardize,
    standardize_balance_periods,
)
from .store import Store, cache_store, fixture_store, is_offline, normalize
from .universe import UniverseEntry, UniverseStats, build_universe

logger = logging.getLogger(__name__)

#: Minimum number of weekly observations for a usable price history.
MIN_PRICE_OBSERVATIONS = 52
#: Minimum number of weekly returns for a regression beta (one year); the DCF
#: (``dcf.MIN_BETA_OBSERVATIONS``) and the comps fallback beta use the same rule.
MIN_BETA_RETURNS = 52
PEER_CURRENCY = "SEK"

#: Extra, informational columns of ``peer_snapshots.csv`` (not in PeerSnapshot).
PEER_EXTRA_COLUMNS: tuple[str, ...] = (
    "beta_source",
    "fiscal_year",
    "currency",
    "reporting_currency",
    "fx_to_sek",
    "as_of",
    "fx_to_sek_average",
    "fx_source",
)

#: Master switch for Riksbank per-year FX in the loaders. ``build_company``
#: called without a plan always uses the single Yahoo spot rate.
USE_RIKSBANK_FX = True

_fx_memo: dict[str, dict[str, Any]] = {}


@dataclass
class CompanyBundle:
    """A loaded company together with provenance needed by the peer table."""

    company: CompanyData
    origin: str  # "fixtures", "cache" or "network"
    fx_to_sek: float  # trading currency -> SEK, spot (market values)
    fetched_at: str
    #: trading currency -> SEK, average of the latest fiscal year (P&L items).
    fx_to_sek_average: float | None = None
    fx_source: str = ""  # how statements / SEK amounts were converted


# --------------------------------------------------------------------------
# Raw payload resolution
# --------------------------------------------------------------------------


def _today_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def _fx_lookup(source: str, target: str) -> dict[str, Any]:
    """FX rate with an in-process memo and a same-day on-disk cache."""
    pair = f"{source}{target}"
    today = _today_utc()
    memo = _fx_memo.get(pair)
    if memo and memo.get("fetched_on") == today:
        return {k: memo[k] for k in ("symbol", "rate", "as_of")}
    store = cache_store()
    table = store.read_json(store.fx_path) or {}
    cached = table.get(pair)
    if isinstance(cached, dict) and cached.get("fetched_on") == today and cached.get("rate"):
        _fx_memo[pair] = cached
        return {k: cached[k] for k in ("symbol", "rate", "as_of")}
    fresh = yahoo.fetch_fx_rate(source, target)
    entry = {**fresh, "fetched_on": today}
    _fx_memo[pair] = entry
    table[pair] = entry
    try:
        store.write_json(store.fx_path, table)
    except OSError as exc:  # cache is best effort
        logger.warning("could not write FX cache: %s", exc)
    return fresh


def _offline_lookup(
    reader: Callable[[Store], dict[str, Any] | None],
) -> tuple[dict[str, Any], str] | None:
    """Fixtures first, then cache."""
    for store in (fixture_store(), cache_store()):
        payload = reader(store)
        if payload is not None:
            return payload, store.name
    return None


def _resolve(
    what: str,
    reader: Callable[[Store], dict[str, Any] | None],
    fetcher: Callable[[], dict[str, Any]],
    writer: Callable[[Store, dict[str, Any]], None],
    *,
    offline: bool,
    refresh: bool,
) -> tuple[dict[str, Any], str]:
    """Generic fixtures/cache/network resolution (see module docstring)."""
    if is_offline(offline):
        found = _offline_lookup(reader)
        if found is None:
            raise DataUnavailableError(
                f"{what}: not available offline (looked in {fixture_store().root} and "
                f"{cache_store().root}). Run once online or add it to the fixtures."
            )
        return found
    cache = cache_store()
    if not refresh:
        cached = reader(cache)
        if cached is not None:
            return cached, cache.name
    try:
        payload = normalize(fetcher())
    except DataUnavailableError:
        raise
    except DataError as exc:
        for store in (cache, fixture_store()):
            fallback = reader(store)
            if fallback is not None:
                logger.warning("%s: network failed (%s); using %s", what, exc, store.name)
                return fallback, store.name
        raise DataUnavailableError(
            f"{what}: network failed and no local copy exists ({exc})"
        ) from exc
    try:
        writer(cache, payload)
    except OSError as exc:
        logger.warning("%s: could not write cache (%s)", what, exc)
    return payload, "network"


def _raw_company(ticker: str, *, offline: bool, refresh: bool) -> tuple[dict[str, Any], str]:
    return _resolve(
        ticker,
        lambda store: store.read_company(ticker),
        lambda: yahoo.fetch_company(ticker, fx_lookup=_fx_lookup),
        lambda store, payload: store.write_company(ticker, payload),
        offline=offline,
        refresh=refresh,
    )


def _raw_universe(*, offline: bool, refresh: bool) -> tuple[dict[str, Any], str]:
    return _resolve(
        "universe",
        lambda store: store.read_universe(),
        yahoo.fetch_screen,
        lambda store, payload: store.write_universe(payload),
        offline=offline,
        refresh=refresh,
    )


def _raw_benchmark(
    *, offline: bool, refresh: bool, prefer: str | None = None
) -> dict[str, Any] | None:
    """Benchmark payload or None if it cannot be obtained (never raises)."""
    if is_offline(offline):
        stores = [fixture_store(), cache_store()]
        if prefer == "cache":
            stores.reverse()
        for store in stores:
            payload = store.read_benchmark()
            if payload is not None:
                return payload
        return None
    try:
        payload, _ = _resolve(
            "benchmark",
            lambda store: store.read_benchmark(),
            yahoo.fetch_benchmark,
            lambda store, data: store.write_benchmark(data),
            offline=False,
            refresh=refresh,
        )
    except DataError as exc:
        logger.warning("benchmark unavailable: %s", exc)
        return None
    return payload


# --------------------------------------------------------------------------
# Building CompanyData from a raw payload
# --------------------------------------------------------------------------


def _number(value: Any) -> float | None:
    """Finite float or None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _weekly_series(raw_prices: dict[str, Any] | None) -> pd.Series:
    """Raw weekly closes -> Series indexed by the Monday of each week."""
    if not raw_prices or not raw_prices.get("dates"):
        return pd.Series(dtype=float)
    index = pd.DatetimeIndex(pd.to_datetime(raw_prices["dates"]))
    series = pd.Series(raw_prices["close"], index=index, dtype=float).dropna()
    series = series[series > 0]
    mondays = series.index.normalize() - pd.to_timedelta(series.index.day_of_week, unit="D")
    series.index = pd.DatetimeIndex(mondays)
    return series[~series.index.duplicated(keep="last")].sort_index()


def _monday_of(iso_timestamp: str | None) -> pd.Timestamp | None:
    if not iso_timestamp:
        return None
    day = pd.Timestamp(iso_timestamp[:10])
    return day - timedelta(days=day.weekday())


def _build_price_history(
    raw: dict[str, Any], benchmark_raw: dict[str, Any] | None
) -> PriceHistory | None:
    if benchmark_raw is None:
        return None
    stock = _weekly_series(raw.get("prices"))
    bench = _weekly_series(benchmark_raw)
    if stock.empty or bench.empty:
        return None
    prices = pd.concat({"stock": stock, "benchmark": bench}, axis=1, join="inner").dropna()
    # Drop the week(s) that were still in progress when either side was fetched.
    cutoffs = [
        m
        for m in (_monday_of(raw.get("fetched_at")), _monday_of(benchmark_raw.get("fetched_at")))
        if m is not None
    ]
    if cutoffs:
        prices = prices[prices.index < min(cutoffs)]
    if len(prices) < MIN_PRICE_OBSERVATIONS:
        return None
    prices.index.name = "date"
    return PriceHistory(
        ticker=str(raw["ticker"]),
        benchmark=str(benchmark_raw.get("symbol") or "unknown"),
        frequency="W",
        prices=prices.astype(float),
    )


_BORROWING_ROWS: tuple[str, ...] = ("Long Term Debt", "Current Debt")


def _borrowings_missing(raw: dict[str, Any], quarter: pd.Timestamp, year_end: pd.Timestamp) -> bool:
    """True if the quarterly "Total Debt" consists of lease liabilities only
    although the annual balance sheet reports loans.

    Observed for e.g. ARJO-B.ST and HOLM-B.ST: Yahoo's quarterly balance sheet
    lacks the loan rows, so its "Total Debt" equals "Capital Lease
    Obligations" and understates debt by several billion SEK.
    """
    annual = raw_statement_frame((raw.get("statements") or {}).get("balance_sheet"))
    quarterly = raw_statement_frame(raw.get("quarterly_balance_sheet"))

    def cell(frame: pd.DataFrame, period: pd.Timestamp, row: str) -> float | None:
        if period not in frame.index or row not in frame.columns:
            return None
        value = frame.at[period, row]
        return float(value) if pd.notna(value) else None

    annual_loans = [cell(annual, year_end, row) for row in _BORROWING_ROWS]
    if not any(value is not None and value > 0 for value in annual_loans):
        return False
    quarter_loans = [cell(quarterly, quarter, row) for row in _BORROWING_ROWS]
    if any(value is not None and value > 0 for value in quarter_loans):
        return False
    total = cell(quarterly, quarter, "Total Debt")
    leases = cell(quarterly, quarter, "Capital Lease Obligations")
    return total is not None and leases is not None and total <= leases * 1.001


def _latest_balance(raw: dict[str, Any]) -> tuple[pd.Series | None, dict[str, str], str]:
    """Most recent standardized balance sheet (quarterly if newer than annual)."""
    annual, annual_tags = standardize_balance_periods(
        (raw.get("statements") or {}).get("balance_sheet")
    )
    quarterly, quarterly_tags = standardize_balance_periods(raw.get("quarterly_balance_sheet"))

    def usable(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        return frame[frame[["cash", "total_debt", "total_assets"]].notna().any(axis=1)]

    annual, quarterly = usable(annual), usable(quarterly)
    if not quarterly.empty and (annual.empty or quarterly.index[-1] > annual.index[-1]):
        period = quarterly.index[-1]
        latest = quarterly.iloc[-1].copy()
        if not annual.empty and _borrowings_missing(raw, period, annual.index[-1]):
            # Sparse quarter: Yahoo dropped the loan rows, so "Total Debt" is
            # leases only. Treat it as missing (the caller falls back).
            latest["total_debt"] = float("nan")
        return latest, quarterly_tags, f"quarterly balance sheet {period:%Y-%m-%d}"
    if not annual.empty:
        period = annual.index[-1]
        return annual.iloc[-1], annual_tags, f"annual balance sheet {period:%Y-%m-%d}"
    return None, {}, "no balance sheet"


def company_currencies(raw: dict[str, Any]) -> tuple[str, str]:
    """``(reporting currency, trading currency)`` of a raw payload."""
    info: dict[str, Any] = raw.get("info") or {}
    trading = str(info.get("currency") or "SEK")
    return str(info.get("financialCurrency") or trading), trading


def build_company(
    raw: dict[str, Any],
    benchmark_raw: dict[str, Any] | None = None,
    *,
    fx_plan: FxPlan | None = None,
) -> CompanyData:
    """Build :class:`CompanyData` from a raw company payload.

    ``fx_plan`` (see ``fx.plan_fx``) supplies per-year exchange rates. Without
    it every amount is converted at the single Yahoo spot rate stored in the
    payload. If the plan does not cover a fiscal year, the spot rate of the
    plan is used for all years and the reason is recorded in ``source_map``.

    Raises :class:`DataUnavailableError` when the payload has no usable price,
    market capitalization or income statement.
    """
    ticker = str(raw.get("ticker") or "")
    info: dict[str, Any] = raw.get("info") or {}
    reporting, currency = company_currencies(raw)
    plan = fx_plan if fx_plan is not None else legacy_plan(raw, reporting, currency)
    fx = plan.spot
    fx_notes = dict(plan.notes)

    statements_raw = raw.get("statements") or {}
    try:
        statements = standardize(ticker, currency, statements_raw, fx, plan.by_year)
    except ValueError as exc:
        statements = standardize(ticker, currency, statements_raw, fx)
        fx_notes = {k: v for k, v in fx_notes.items() if k == "fx.spot"}
        fx_notes["fx.method"] = f"spot rate for all fiscal years (fallback: {exc})"
    statements.source_map.update(fx_notes)
    if statements.income.empty:
        raise DataUnavailableError(
            f"{ticker}: Yahoo has no annual income statement for this ticker"
        )
    sources = statements.source_map

    # --- price and market capitalization ---------------------------------
    price: float | None = None
    for key in ("currentPrice", "regularMarketPrice", "previousClose"):
        candidate = _number(info.get(key))
        if candidate is not None and candidate > 0:
            price, sources["market.price"] = candidate, f"info:{key}"
            break
    if price is None:
        closes = (raw.get("prices") or {}).get("close") or []
        if closes and _number(closes[-1]) and closes[-1] > 0:
            price, sources["market.price"] = float(closes[-1]), "last weekly close"
    if price is None:
        raise DataUnavailableError(f"{ticker}: no usable price")

    market_cap = _number(info.get("marketCap"))
    sources["market.market_cap"] = "info:marketCap"
    if market_cap is None or market_cap <= 0:
        # Yahoo sometimes has no market cap (a few depositary receipts and
        # small caps). Fall back to the latest reported share count.
        shares = statements.income["diluted_shares"].dropna()
        latest_shares = float(shares.iloc[-1]) if not shares.empty else None
        if latest_shares is None or latest_shares <= 0:
            raise DataUnavailableError(f"{ticker}: no market capitalization and no share count")
        market_cap = latest_shares * price
        sources["market.market_cap"] = (
            f"income.diluted_shares x price ({sources.get('income.diluted_shares')})"
        )

    # --- balance-sheet items of the snapshot ------------------------------
    # Per item: latest balance sheet (quarterly if newer than the annual one)
    # -> Yahoo's most-recent-quarter figure in ``info`` -> latest annual
    # statement -> 0. Quarterly balance sheets from Yahoo are sometimes
    # sparse (no debt rows although the company has debt), hence the chain.
    balance, tags, balance_label = _latest_balance(raw)
    info_keys = {"total_debt": "totalDebt", "cash": "totalCash"}
    annual = statements.balance.iloc[-1] if not statements.balance.empty else None
    values: dict[str, float] = {}
    for item in ("total_debt", "cash", "minority_interest", "preferred_equity"):
        key = f"market.{item}"
        from_info = _number(info.get(info_keys.get(item, "")))
        if balance is not None and pd.notna(balance[item]):
            values[item] = float(balance[item]) * fx
            sources[key] = f"{balance_label}: {tags[item]}"
        elif from_info is not None and from_info >= 0:
            values[item] = from_info * fx
            sources[key] = f"info:{info_keys[item]} (most recent quarter)"
        elif annual is not None and item in annual.index and pd.notna(annual[item]):
            values[item] = float(annual[item])  # already in the trading currency
            sources[key] = (
                f"annual statement {statements.latest_year}: {sources[f'balance.{item}']}"
            )
        else:
            values[item] = 0.0
            sources[key] = "assumed 0 (not reported)"

    timestamp = _number(info.get("regularMarketTime"))
    if timestamp:
        as_of = datetime.fromtimestamp(timestamp, tz=UTC).date()
    else:
        as_of = date.fromisoformat(str(raw.get("fetched_at"))[:10])

    market = MarketData(
        ticker=ticker,
        as_of=as_of,
        currency=currency,
        price=price,
        shares_outstanding=market_cap / price,
        market_cap=market_cap,
        total_debt=values["total_debt"],
        cash=values["cash"],
        minority_interest=max(values["minority_interest"], 0.0),
        preferred_equity=max(values["preferred_equity"], 0.0),
        beta=_number(info.get("beta")),
    )
    profile = CompanyProfile(
        ticker=ticker,
        name=str(info.get("longName") or info.get("shortName") or ticker),
        sector=info.get("sector") or None,
        industry=info.get("industry") or None,
        country=info.get("country") or None,
        currency=currency,
        reporting_currency=reporting,
        fx_rate=fx,
    )
    return CompanyData(
        profile=profile,
        statements=statements,
        market=market,
        price_history=_build_price_history(raw, benchmark_raw),
    )


def load_company_bundle(
    ticker: str, *, offline: bool = False, refresh: bool = False, cache_only: bool = False
) -> CompanyBundle:
    """Like :func:`load_company` but also returns provenance information.

    ``cache_only=True`` reads the local cache and nothing else (used by the
    batch downloader to build the peer table from what was downloaded).
    """
    ticker = str(ticker).strip().upper()
    if not ticker or ticker.startswith("."):
        raise DataUnavailableError(f"{ticker!r}: not a valid ticker (expected e.g. 'ATCO-A.ST')")
    if cache_only:
        cached = cache_store().read_company(ticker)
        if cached is None:
            raise DataUnavailableError(f"{ticker}: not in the local cache")
        raw, origin = cached, "cache"
    else:
        raw, origin = _raw_company(ticker, offline=offline, refresh=refresh)
    if cache_only:
        benchmark = cache_store().read_benchmark()
    elif origin == "network":
        # Keep the benchmark as fresh as the company that was just downloaded.
        cached = cache_store().read_benchmark()
        stale = cached is None or str(cached.get("fetched_at", ""))[:10] < _today_utc()
        benchmark = _raw_benchmark(offline=False, refresh=stale)
    else:
        benchmark = _raw_benchmark(offline=offline, refresh=False, prefer=origin)
    reporting, currency = company_currencies(raw)
    if USE_RIKSBANK_FX:
        period_ends = fiscal_period_ends(raw.get("statements") or {})
        # The batch builder (cache_only) may still fetch Riksbank rates online.
        plan = plan_fx(raw, reporting, currency, period_ends, offline=offline, refresh=refresh)
    else:
        plan = legacy_plan(raw, reporting, currency)
    company = build_company(raw, benchmark, fx_plan=plan)
    methods = [company.statements.source_map.get("fx.method", "")]
    if currency != PEER_CURRENCY:
        methods.append(f"to SEK: {plan.to_sek_method}")
    return CompanyBundle(
        company=company,
        origin=origin,
        fx_to_sek=plan.to_sek_spot,
        fetched_at=str(raw.get("fetched_at") or ""),
        fx_to_sek_average=plan.to_sek_average,
        fx_source="; ".join(m for m in methods if m),
    )


def load_company(ticker: str, *, offline: bool = False, refresh: bool = False) -> CompanyData:
    """Load one company (statements, market snapshot, price history).

    All amounts are in the trading currency of the listing. See the module
    docstring for the fixtures/cache/network resolution order.
    """
    return load_company_bundle(ticker, offline=offline, refresh=refresh).company


# --------------------------------------------------------------------------
# Universe
# --------------------------------------------------------------------------


def load_universe_entries(
    *, offline: bool = False, refresh: bool = False
) -> tuple[list[UniverseEntry], UniverseStats]:
    """Universe with share-class details and the filter statistics."""
    payload, _ = _raw_universe(offline=offline, refresh=refresh)
    return build_universe(payload)


def list_universe(*, offline: bool = False, refresh: bool = False) -> list[str]:
    """Yahoo tickers of all Swedish listed companies, one share class each.

    ETFs, funds, warrants, certificates, subscription rights, preference
    shares and symbols without a price are excluded (rules in ``universe``).
    """
    entries, _ = load_universe_entries(offline=offline, refresh=refresh)
    return [entry.ticker for entry in entries]


# --------------------------------------------------------------------------
# Peer snapshots
# --------------------------------------------------------------------------


def regression_beta(history: PriceHistory | None) -> float | None:
    """Raw OLS beta of weekly simple returns against the benchmark (levered,
    no Blume adjustment). None if there is too little data."""
    if history is None:
        return None
    returns = history.prices[["stock", "benchmark"]].pct_change().dropna()
    if len(returns) < MIN_BETA_RETURNS:
        return None
    variance = float(returns["benchmark"].var())
    if not variance > 0:
        return None
    beta = float(returns["stock"].cov(returns["benchmark"])) / variance
    return beta if math.isfinite(beta) else None


def _opt(value: float) -> float | None:
    return float(value) if pd.notna(value) and math.isfinite(float(value)) else None


def snapshot_from_bundle(bundle: CompanyBundle) -> tuple[PeerSnapshot, dict[str, Any]]:
    """Build the peer-table row (amounts in SEK) and its informational extras."""
    company, rate = bundle.company, bundle.fx_to_sek
    income, balance = company.statements.income, company.statements.balance

    def latest(item: str) -> float | None:
        return _opt(income[item].iloc[-1])

    # Flow items of the latest fiscal year use that year's average rate.
    average_rate = bundle.fx_to_sek_average if bundle.fx_to_sek_average is not None else rate

    def sek(value: float | None) -> float | None:
        return None if value is None else value * average_rate

    revenue, ebitda, net_income = latest("revenue"), latest("ebitda"), latest("net_income")
    growth: float | None = None
    if len(income) >= 2:
        previous = _opt(income["revenue"].iloc[-2])
        consecutive = int(income.index[-1]) - int(income.index[-2]) == 1
        if revenue is not None and previous is not None and previous > 0 and consecutive:
            growth = revenue / previous - 1.0
    margin = (
        ebitda / revenue if ebitda is not None and revenue is not None and revenue > 0 else None
    )

    roe: float | None = None
    equity = balance["total_equity"].dropna()
    if (
        net_income is not None
        and not equity.empty
        and int(equity.index[-1]) == int(income.index[-1])
    ):
        base = float(equity.iloc[-2:].mean())  # average of opening and closing equity
        if base > 0:
            roe = net_income / base

    beta = regression_beta(company.price_history)
    beta_source = (
        f"regression vs {company.price_history.benchmark}"
        if beta is not None and company.price_history
        else ""
    )
    if beta is None and company.market.beta is not None:
        beta, beta_source = company.market.beta, "yahoo"

    snapshot = PeerSnapshot(
        ticker=company.ticker,
        name=company.profile.name,
        sector=company.profile.sector,
        industry=company.profile.industry,
        market_cap=company.market.market_cap * rate,
        enterprise_value=company.market.enterprise_value * rate,
        revenue=sek(revenue),
        ebitda=sek(ebitda),
        ebit=sek(latest("ebit")),
        net_income=sek(net_income),
        total_debt=company.market.total_debt * rate,
        revenue_growth=growth,
        ebitda_margin=margin,
        return_on_equity=roe,
        beta=beta,
    )
    extras = {
        "beta_source": beta_source,
        "fiscal_year": company.statements.latest_year,
        "currency": company.profile.currency,
        "reporting_currency": company.profile.reporting_currency,
        "fx_to_sek": rate,
        "as_of": company.market.as_of.isoformat(),
        "fx_to_sek_average": average_rate,
        "fx_source": bundle.fx_source,
    }
    return snapshot, extras


_PEER_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(PeerSnapshot))
_PEER_TEXT_FIELDS = frozenset({"ticker", "name", "sector", "industry"})


def write_peer_table(store: Store, rows: Iterable[tuple[PeerSnapshot, dict[str, Any]]]) -> int:
    """Write ``peer_snapshots.csv``; floats use ``repr`` so reads are lossless."""
    path = store.peers_path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    count = 0
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([*_PEER_FIELDS, *PEER_EXTRA_COLUMNS])
        for snapshot, extras in sorted(rows, key=lambda row: row[0].ticker):
            record: list[str] = []
            for name in _PEER_FIELDS:
                value = getattr(snapshot, name)
                record.append(
                    ""
                    if value is None
                    else (value if isinstance(value, str) else repr(float(value)))
                )
            record.extend(
                "" if extras.get(k) is None else str(extras[k]) for k in PEER_EXTRA_COLUMNS
            )
            writer.writerow(record)
            count += 1
    tmp.replace(path)
    return count


def read_peer_table(store: Store) -> list[PeerSnapshot] | None:
    """Read ``peer_snapshots.csv`` from a store; None if it does not exist."""
    path = store.peers_path
    if not path.is_file():
        return None
    snapshots: list[PeerSnapshot] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            kwargs: dict[str, Any] = {}
            for name in _PEER_FIELDS:
                text = record.get(name, "")
                if name in _PEER_TEXT_FIELDS:
                    kwargs[name] = text if text or name in ("ticker", "name") else None
                else:
                    kwargs[name] = float(text) if text else None
            if kwargs["market_cap"] is None or kwargs["enterprise_value"] is None:
                continue
            snapshots.append(PeerSnapshot(**kwargs))
    return snapshots


def build_peer_snapshots(
    tickers: Iterable[str] | None = None,
    *,
    offline: bool = False,
    refresh: bool = False,
    cache_only: bool = False,
    progress: Callable[[int, int, str, str | None], None] | None = None,
) -> tuple[list[tuple[PeerSnapshot, dict[str, Any]]], dict[str, str]]:
    """Load every company and build the peer rows.

    Returns ``(rows, failures)`` where ``failures`` maps ticker -> reason.
    One broken ticker never aborts the run.
    """
    names = list(tickers) if tickers is not None else list_universe(offline=offline, refresh=False)
    rows: list[tuple[PeerSnapshot, dict[str, Any]]] = []
    failures: dict[str, str] = {}
    for index, ticker in enumerate(names, start=1):
        error: str | None = None
        try:
            bundle = load_company_bundle(
                ticker, offline=offline, refresh=refresh, cache_only=cache_only
            )
            rows.append(snapshot_from_bundle(bundle))
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:  # noqa: BLE001 - one ticker must not abort the batch
            error = f"{type(exc).__name__}: {exc}"
            failures[ticker] = error
        if progress is not None:
            progress(index, len(names), ticker, error)
    return rows, failures


def load_peer_snapshots(*, offline: bool = False, refresh: bool = False) -> list[PeerSnapshot]:
    """One :class:`PeerSnapshot` per universe company, amounts in SEK.

    Offline: the table from the fixtures, else from the cache, else built
    from cached company files. Online: the cached table (unless ``refresh``),
    else it is built by loading every universe company (cache first, network
    for the rest - this is the slow full-universe download; prefer
    ``python -m valuation_engine.data.download --all``), with the fixtures as
    last resort.
    """
    if is_offline(offline):
        for store in (fixture_store(), cache_store()):
            table = read_peer_table(store)
            if table:
                return table
        rows, _ = _build_offline_rows()
        if rows:
            return [snapshot for snapshot, _ in rows]
        raise DataUnavailableError(
            f"peer snapshots: not available offline (looked in {fixture_store().root} and "
            f"{cache_store().root}). Run `python -m valuation_engine.data.download --all` "
            "online first."
        )
    cache = cache_store()
    if not refresh:
        table = read_peer_table(cache)
        if table:
            return table
    try:
        rows, failures = build_peer_snapshots(offline=False, refresh=refresh)
    except DataError as exc:
        logger.warning("peer snapshots: build failed (%s)", exc)
        rows, failures = [], {}
    if failures:
        logger.warning("peer snapshots: %d tickers could not be loaded", len(failures))
    if rows:
        try:
            write_peer_table(cache, rows)
        except OSError as exc:
            logger.warning("could not write peer table: %s", exc)
        return [snapshot for snapshot, _ in sorted(rows, key=lambda row: row[0].ticker)]
    for store in (cache, fixture_store()):
        table = read_peer_table(store)
        if table:
            logger.warning("peer snapshots: using the table from %s", store.name)
            return table
    raise DataUnavailableError("peer snapshots: network failed and no local table exists")


def _build_offline_rows() -> tuple[list[tuple[PeerSnapshot, dict[str, Any]]], dict[str, str]]:
    try:
        return build_peer_snapshots(offline=True)
    except DataError:
        return [], {}
