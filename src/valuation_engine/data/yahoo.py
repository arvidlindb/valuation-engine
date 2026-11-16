"""Network access to Yahoo Finance (via ``yfinance``).

This is the ONLY module that talks to the network. Every function returns a
plain, JSON-serializable ``dict`` (the *raw payload*) that is written to the
cache unchanged; the standardization happens later in ``mapping``/``loader``.

Politeness and robustness:

* a global throttle keeps a minimum interval between requests,
* every request is retried with exponential backoff,
* rate-limit responses trigger a much longer backoff and are surfaced as
  :class:`RateLimitError` so that batch jobs can cool down.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, TypeVar

import pandas as pd

from .errors import DataUnavailableError, FetchError, NetworkDisabledError, RateLimitError
from .store import SCHEMA_VERSION, is_offline

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: Benchmarks in order of preference (all-share index, then OMXS30).
BENCHMARK_CANDIDATES: tuple[str, ...] = ("^OMXSPI", "^OMX")

PRICE_PERIOD = "5y"
PRICE_INTERVAL = "1wk"

#: Minimum number of seconds between two requests.
THROTTLE_SECONDS = float(os.environ.get("VALUATION_ENGINE_THROTTLE_SECONDS", "0.25"))
MAX_RETRIES = 4
BACKOFF_SECONDS = 2.0
RATE_LIMIT_BACKOFF_SECONDS = 30.0

#: Fields of ``Ticker.info`` that are kept in the raw payload.
INFO_KEYS: tuple[str, ...] = (
    "symbol",
    "shortName",
    "longName",
    "quoteType",
    "exchange",
    "country",
    "sector",
    "industry",
    "currency",
    "financialCurrency",
    "marketCap",
    "currentPrice",
    "regularMarketPrice",
    "previousClose",
    "regularMarketTime",
    "sharesOutstanding",
    "impliedSharesOutstanding",
    "floatShares",
    "beta",
    "totalDebt",
    "totalCash",
    "enterpriseValue",
    "totalRevenue",
    "ebitda",
    "netIncomeToCommon",
    "operatingCashflow",
    "freeCashflow",
    "trailingEps",
    "trailingPE",
    "bookValue",
    "dividendRate",
    "fiftyTwoWeekLow",
    "fiftyTwoWeekHigh",
    "averageVolume",
    "lastFiscalYearEnd",
    "mostRecentQuarter",
    "fullTimeEmployees",
)

#: Screener fields kept in the universe file.
SCREEN_KEYS: tuple[str, ...] = (
    "symbol",
    "quoteType",
    "longName",
    "shortName",
    "exchange",
    "currency",
    "financialCurrency",
    "regularMarketPrice",
    "marketCap",
    "averageDailyVolume3Month",
    "sharesOutstanding",
    "impliedSharesOutstanding",
)

_last_request: float = 0.0


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _yf() -> Any:
    """Import yfinance lazily so offline use never needs it."""
    if is_offline():
        raise NetworkDisabledError("offline mode is active; refusing to contact Yahoo Finance")
    import yfinance as yf

    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    return yf


def _throttle() -> None:
    global _last_request
    wait = THROTTLE_SECONDS - (time.monotonic() - _last_request)
    if wait > 0:
        time.sleep(wait)
    _last_request = time.monotonic()


def _is_rate_limit(exc: BaseException) -> bool:
    name = type(exc).__name__
    text = str(exc).lower()
    return name == "YFRateLimitError" or "too many requests" in text or "rate limit" in text


def _call(func: Callable[[], T], what: str, *, retries: int = MAX_RETRIES) -> T:
    """Run ``func`` with throttling, retries and exponential backoff."""
    if is_offline():
        raise NetworkDisabledError(f"offline mode is active; refusing to fetch {what}")
    last: BaseException | None = None
    rate_limited = False
    for attempt in range(retries):
        _throttle()
        try:
            return func()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:  # noqa: BLE001 - yfinance raises many types
            last = exc
            rate_limited = _is_rate_limit(exc)
            base = RATE_LIMIT_BACKOFF_SECONDS if rate_limited else BACKOFF_SECONDS
            delay = base * (2**attempt)
            logger.warning(
                "%s failed (attempt %d/%d): %s: %s",
                what,
                attempt + 1,
                retries,
                type(exc).__name__,
                exc,
            )
            if attempt < retries - 1:
                time.sleep(delay)
    message = f"{what} failed after {retries} attempts: {type(last).__name__}: {last}"
    if rate_limited:
        raise RateLimitError(message) from last
    raise FetchError(message) from last


# --------------------------------------------------------------------------
# Serialization helpers
# --------------------------------------------------------------------------


def _clean_number(value: Any) -> Any:
    """Convert numpy/pandas scalars to plain Python; NaN -> None."""
    if value is None:
        return None
    if isinstance(value, (str, bool)):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (int, float)):
        return value
    return str(value)


def _statement_to_raw(frame: pd.DataFrame | None, max_periods: int | None = None) -> dict[str, Any]:
    """Serialize a yfinance statement (rows = line items, columns = period ends)."""
    if frame is None or frame.empty:
        return {"periods": [], "rows": {}}
    frame = frame.loc[:, ~frame.columns.duplicated()]
    frame = frame.loc[~frame.index.duplicated()]
    columns = sorted(frame.columns, reverse=True)
    if max_periods is not None:
        columns = columns[:max_periods]
    frame = frame[columns]
    periods = [pd.Timestamp(c).strftime("%Y-%m-%d") for c in columns]
    rows: dict[str, list[Any]] = {}
    for label, values in frame.iterrows():
        cleaned = [_clean_number(v) for v in values.tolist()]
        if any(v is not None for v in cleaned):
            rows[str(label)] = cleaned
    return {"periods": periods, "rows": rows}


def _history_to_raw(history: pd.DataFrame | None) -> dict[str, Any]:
    """Serialize weekly adjusted closes."""
    if history is None or history.empty or "Close" not in history.columns:
        return {"interval": PRICE_INTERVAL, "adjusted": True, "dates": [], "close": []}
    close = history["Close"].dropna()
    dates = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in close.index]
    return {
        "interval": PRICE_INTERVAL,
        "adjusted": True,
        "dates": dates,
        "close": [float(v) for v in close.tolist()],
    }


# --------------------------------------------------------------------------
# Fetchers
# --------------------------------------------------------------------------


def fetch_screen(page_size: int = 250) -> dict[str, Any]:
    """Download all symbols of the Swedish market from Yahoo's screener."""
    yf = _yf()
    query = yf.EquityQuery("eq", ["region", "se"])
    quotes: dict[str, dict[str, Any]] = {}
    total = 0
    offset = 0
    while True:

        def page(off: int = offset) -> dict[str, Any]:
            result = yf.screen(query, size=page_size, offset=off, sortField="ticker", sortAsc=True)
            if not isinstance(result, dict) or "quotes" not in result:
                raise FetchError(f"unexpected screener response at offset {off}")
            return result

        result = _call(page, f"screener page at offset {offset}")
        batch = result.get("quotes") or []
        total = int(result.get("total") or total)
        for quote in batch:
            symbol = quote.get("symbol")
            if symbol:
                quotes[str(symbol)] = quote
        offset += len(batch)
        if not batch or offset >= total:
            break
    if len(quotes) < total:
        logger.warning("screener returned %d unique symbols, expected %d", len(quotes), total)
    rows = [[_clean_number(quotes[s].get(k)) for k in SCREEN_KEYS] for s in sorted(quotes)]
    return {
        "schema_version": SCHEMA_VERSION,
        "fetched_at": _now_iso(),
        "source": "yfinance.screen(EquityQuery('eq', ['region', 'se']))",
        "total": total,
        "columns": list(SCREEN_KEYS),
        "rows": rows,
    }


def _fx_quote(symbol: str, retries: int) -> dict[str, Any]:
    """Latest daily close of a Yahoo currency symbol such as ``USDSEK=X``."""
    yf = _yf()

    def get() -> dict[str, Any]:
        history = yf.Ticker(symbol).history(period="10d", interval="1d", auto_adjust=False)
        close = history["Close"].dropna() if "Close" in history.columns else pd.Series(dtype=float)
        if close.empty:
            raise FetchError(f"no FX quotes for {symbol}")
        rate = float(close.iloc[-1])
        if not rate > 0:
            raise FetchError(f"non-positive FX rate for {symbol}: {rate}")
        return {
            "symbol": symbol,
            "rate": rate,
            "as_of": pd.Timestamp(close.index[-1]).strftime("%Y-%m-%d"),
        }

    return _call(get, f"FX rate {symbol}", retries=retries)


def fetch_fx_rate(from_currency: str, to_currency: str) -> dict[str, Any]:
    """Latest spot rate: 1 unit of ``from_currency`` in ``to_currency``.

    Uses the direct Yahoo pair (``EURSEK=X``). Pairs that Yahoo does not quote
    (observed: ``ISKSEK=X``, ``PLNSEK=X``) are computed as a cross rate via USD.
    """
    if from_currency == to_currency:
        return {"symbol": None, "rate": 1.0, "as_of": _now_iso()[:10]}
    try:
        return _fx_quote(f"{from_currency}{to_currency}=X", retries=2)
    except RateLimitError:
        raise
    except FetchError as direct_error:
        if "USD" in (from_currency, to_currency):
            raise
        logger.info("no direct FX pair (%s); using a cross rate via USD", direct_error)
    usd_to_target = _fx_quote(f"USD{to_currency}=X", retries=3)
    usd_to_source = _fx_quote(f"USD{from_currency}=X", retries=3)
    return {
        "symbol": f"{usd_to_target['symbol']} / {usd_to_source['symbol']}",
        "rate": usd_to_target["rate"] / usd_to_source["rate"],
        "as_of": min(usd_to_target["as_of"], usd_to_source["as_of"]),
    }


def fetch_benchmark() -> dict[str, Any]:
    """Weekly closes of the first available benchmark index."""
    yf = _yf()
    errors: list[str] = []
    for symbol in BENCHMARK_CANDIDATES:

        def get(sym: str = symbol) -> pd.DataFrame:
            history = yf.Ticker(sym).history(
                period=PRICE_PERIOD, interval=PRICE_INTERVAL, auto_adjust=True
            )
            if history is None or history.empty or history["Close"].dropna().shape[0] < 52:
                raise FetchError(f"benchmark {sym} has too little history")
            return history

        try:
            history = _call(get, f"benchmark {symbol}", retries=3)
        except RateLimitError:
            raise
        except FetchError as exc:
            errors.append(str(exc))
            continue
        payload = _history_to_raw(history)
        payload.update(
            {"schema_version": SCHEMA_VERSION, "symbol": symbol, "fetched_at": _now_iso()}
        )
        return payload
    raise FetchError("no benchmark index available: " + "; ".join(errors))


def fetch_company(
    ticker: str,
    fx_lookup: Callable[[str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Download everything needed for one ticker and return the raw payload.

    ``fx_lookup(from, to)`` supplies FX rates (lets batch jobs reuse rates);
    it defaults to :func:`fetch_fx_rate`.
    """
    yf = _yf()
    handle = yf.Ticker(ticker)

    def get_info() -> dict[str, Any]:
        info = handle.info
        if not isinstance(info, dict):
            raise FetchError(f"{ticker}: info is not a dict")
        return info

    info_full = _call(get_info, f"{ticker} info")
    info = {k: _clean_number(info_full.get(k)) for k in INFO_KEYS if info_full.get(k) is not None}
    has_price = any(info.get(k) for k in ("currentPrice", "regularMarketPrice", "previousClose"))
    if not info.get("symbol") and not has_price:
        raise DataUnavailableError(f"{ticker}: Yahoo returned no quote data (unknown or delisted)")

    def get_statements() -> dict[str, dict[str, Any]]:
        return {
            "income_stmt": _statement_to_raw(handle.income_stmt),
            "balance_sheet": _statement_to_raw(handle.balance_sheet),
            "cashflow": _statement_to_raw(handle.cashflow),
        }

    statements = _call(get_statements, f"{ticker} statements")
    if not any(statements[name]["periods"] for name in statements) and info.get("totalRevenue"):
        # yfinance swallows some transport errors and returns empty frames.
        # The quote says the company has revenue, so try once more.
        time.sleep(BACKOFF_SECONDS)
        handle = yf.Ticker(ticker)
        statements = _call(get_statements, f"{ticker} statements (second try)")

    def get_quarterly_balance() -> dict[str, Any]:
        return _statement_to_raw(handle.quarterly_balance_sheet, max_periods=2)

    try:
        quarterly_balance = _call(get_quarterly_balance, f"{ticker} quarterly balance", retries=2)
    except RateLimitError:
        raise
    except FetchError as exc:
        logger.warning("%s: quarterly balance sheet unavailable (%s)", ticker, exc)
        quarterly_balance = {"periods": [], "rows": {}}

    def get_history() -> pd.DataFrame:
        return handle.history(period=PRICE_PERIOD, interval=PRICE_INTERVAL, auto_adjust=True)

    try:
        prices = _history_to_raw(_call(get_history, f"{ticker} price history", retries=3))
    except RateLimitError:
        raise
    except FetchError as exc:
        logger.warning("%s: price history unavailable (%s)", ticker, exc)
        prices = _history_to_raw(None)

    lookup = fx_lookup or fetch_fx_rate
    trading = str(info.get("currency") or "SEK")
    reporting = str(info.get("financialCurrency") or trading)
    fx: dict[str, Any] = {}
    # Yahoo spot rates are only the fallback (the Riksbank is the primary FX
    # source, see ``fx``), so a missing pair must not fail the download.
    for source, target in ((reporting, trading), (trading, "SEK")):
        if source == target:
            continue
        try:
            fx[f"{source}{target}"] = lookup(source, target)
        except RateLimitError:
            raise
        except FetchError as exc:
            logger.warning("%s: no Yahoo FX rate %s->%s (%s)", ticker, source, target, exc)

    return {
        "schema_version": SCHEMA_VERSION,
        "ticker": ticker,
        "fetched_at": _now_iso(),
        "source": f"yfinance {getattr(yf, '__version__', '?')}",
        "info": info,
        "statements": statements,
        "quarterly_balance_sheet": quarterly_balance,
        "prices": prices,
        "fx": fx,
    }
