"""Client for the Sveriges Riksbank SWEA API (interest and exchange rates).

Used for two things:

* the risk-free rate: latest 10-year Swedish government bond yield
  (series ``SEGVB10YC``),
* exchange rates against SEK (series ``SEK<CCY>PMI``, SEK per one unit of the
  foreign currency): the latest fixing plus monthly averages and month-end
  values, from which fiscal-year average and period-end rates are derived.

Storage follows the conventions of ``store``: one human-inspectable file,
``riksbank.json``, in the cache directory and (frozen) in the fixtures.

    {"series": {"SEKUSDPMI": {"fetched_at": ..., "latest": {"date", "value"},
                              "monthly": {"columns": [...], "rows": [...]}}}}

Resolution order: offline -> fixtures, then cache. Online -> cache if fresh
enough, else network (written to the cache), else stale cache, else fixtures.
Nothing here ever raises for missing data: callers get ``None`` and fall back.

The API is rate limited for anonymous clients (HTTP 429 after about five
quick calls), hence the long throttle interval and the backoff on 429.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

from .errors import FetchError, NetworkDisabledError, RateLimitError
from .store import SCHEMA_VERSION, Store, cache_store, fixture_store, is_offline

logger = logging.getLogger(__name__)

BASE_URL = "https://api.riksbank.se/swea/v1"
USER_AGENT = "valuation-engine (python-urllib)"

#: 10-year Swedish government bond yield, per cent.
RISK_FREE_SERIES = "SEGVB10YC"
#: First month requested for exchange-rate history.
FX_HISTORY_START = "2019-01-01"

#: Seconds between two requests (anonymous clients get HTTP 429 quickly).
THROTTLE_SECONDS = float(os.environ.get("VALUATION_ENGINE_RIKSBANK_THROTTLE_SECONDS", "13"))
TIMEOUT_SECONDS = 30.0
MAX_RETRIES = 4
RATE_LIMIT_BACKOFF_SECONDS = 20.0

MONTHLY_COLUMNS: tuple[str, ...] = ("month", "average", "ultimo", "observations", "last_date")

_last_request: float = 0.0
_memo: dict[str, tuple[float, dict[str, Any]]] = {}


def fx_series_id(currency: str) -> str:
    """Riksbank series for SEK per one unit of ``currency`` (e.g. SEKUSDPMI)."""
    return f"SEK{currency.upper()}PMI"


# --------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------


def _get(path: str) -> Any:
    """GET ``BASE_URL/path`` as JSON with throttling and retries."""
    global _last_request
    if is_offline():
        raise NetworkDisabledError(f"offline mode is active; refusing to fetch Riksbank {path}")
    url = f"{BASE_URL}/{path}"
    request = urllib.request.Request(  # noqa: S310 - fixed https URL
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
    )
    last: BaseException | None = None
    rate_limited = False
    for attempt in range(MAX_RETRIES):
        wait = THROTTLE_SECONDS - (time.monotonic() - _last_request)
        if _last_request and wait > 0:
            time.sleep(wait)
        _last_request = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:  # noqa: S310
                body = response.read().decode("utf-8")
            if not body.strip():
                raise FetchError(f"Riksbank {path}: empty response (unknown series or no data)")
            return json.loads(body)
        except urllib.error.HTTPError as exc:
            last = exc
            rate_limited = exc.code == 429
            if exc.code in (400, 404):
                raise FetchError(f"Riksbank {path}: HTTP {exc.code} (unknown series?)") from exc
            delay = (
                RATE_LIMIT_BACKOFF_SECONDS * (2**attempt) if rate_limited else 3.0 * (attempt + 1)
            )
        except FetchError:
            raise
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            last = exc
            rate_limited = False
            delay = 3.0 * (attempt + 1)
        logger.warning(
            "Riksbank %s failed (attempt %d/%d): %s", path, attempt + 1, MAX_RETRIES, last
        )
        if attempt < MAX_RETRIES - 1:
            time.sleep(delay)
    message = f"Riksbank {path} failed after {MAX_RETRIES} attempts: {last}"
    raise RateLimitError(message) if rate_limited else FetchError(message)


def fetch_latest(series: str) -> dict[str, Any]:
    """Latest observation of a series: ``{"date": "YYYY-MM-DD", "value": float}``."""
    data = _get(f"Observations/Latest/{series}")
    if not isinstance(data, dict) or "value" not in data or "date" not in data:
        raise FetchError(f"Riksbank {series}: unexpected latest-observation response")
    return {"date": str(data["date"])[:10], "value": float(data["value"])}


def fetch_aggregates(series: str, frequency: str, start: str, end: str) -> list[dict[str, Any]]:
    """Aggregated observations (``frequency``: ``"M"`` monthly, ``"Y"`` yearly).

    Each entry has ``from``, ``to``, ``average``, ``ultimo`` (last value of
    the period) and ``observationCount``. Only completed periods are returned.
    """
    data = _get(f"ObservationAggregates/{series}/{frequency}/{start}/{end}")
    if not isinstance(data, list):
        raise FetchError(f"Riksbank {series}: unexpected aggregate response")
    return data


def fetch_series(series: str, *, monthly: bool) -> dict[str, Any]:
    """Download one series in the storage format of ``riksbank.json``."""
    entry: dict[str, Any] = {
        "fetched_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "latest": fetch_latest(series),
    }
    if monthly:
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        rows = [
            [
                str(item["to"])[:7],
                float(item["average"]),
                float(item["ultimo"]),
                int(item["observationCount"]),
                str(item["to"])[:10],
            ]
            for item in fetch_aggregates(series, "M", FX_HISTORY_START, today)
            if item.get("average") is not None and item.get("ultimo") is not None
        ]
        entry["monthly"] = {"columns": list(MONTHLY_COLUMNS), "rows": rows}
    return entry


# --------------------------------------------------------------------------
# Storage and resolution
# --------------------------------------------------------------------------


def _path(store: Store) -> Any:
    return store.root / "riksbank.json"


def _read(store: Store) -> dict[str, Any]:
    """All series of a store (memoized on the file's modification time)."""
    path = _path(store)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
    key = str(path)
    cached = _memo.get(key)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    payload = store.read_json(path) or {}
    series = payload.get("series")
    result = series if isinstance(series, dict) else {}
    _memo[key] = (mtime, result)
    return result


def _write(store: Store, series_id: str, entry: dict[str, Any]) -> None:
    path = _path(store)
    payload = store.read_json(path) or {}
    series = payload.get("series") if isinstance(payload.get("series"), dict) else {}
    series[series_id] = entry
    store.write_json(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "source": BASE_URL,
            "series": dict(sorted(series.items())),
        },
    )
    _memo.pop(str(path), None)


def _usable(entry: Any, monthly: bool) -> bool:
    if not isinstance(entry, dict) or not isinstance(entry.get("latest"), dict):
        return False
    return not monthly or bool((entry.get("monthly") or {}).get("rows"))


def load_series(
    series: str,
    *,
    monthly: bool,
    offline: bool = False,
    refresh: bool = False,
    fresh_as_of: date | None = None,
) -> tuple[dict[str, Any], str] | None:
    """Resolve one series; returns ``(entry, origin)`` or None if unavailable.

    ``fresh_as_of``: online, a cached entry is reused only if it was fetched
    on or after this date (default: today, UTC).
    """
    cache, fixtures = cache_store(), fixture_store()
    if is_offline(offline):
        for store in (fixtures, cache):
            entry = _read(store).get(series)
            if _usable(entry, monthly):
                return entry, store.name
        return None
    cached = _read(cache).get(series)
    needed = (fresh_as_of or datetime.now(UTC).date()).isoformat()
    if (
        not refresh
        and _usable(cached, monthly)
        and str(cached.get("fetched_at", ""))[:10] >= needed
    ):
        return cached, cache.name
    try:
        entry = fetch_series(series, monthly=monthly)
    except (FetchError, NetworkDisabledError) as exc:
        logger.warning("Riksbank %s unavailable (%s); using local data if any", series, exc)
        for store, candidate in ((cache, cached), (fixtures, _read(fixtures).get(series))):
            if _usable(candidate, monthly):
                return candidate, f"{store.name} (stale)"
        return None
    try:
        _write(cache, series, entry)
    except OSError as exc:
        logger.warning("could not write Riksbank cache: %s", exc)
    return entry, "network"


# --------------------------------------------------------------------------
# Exchange rates
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FxSeries:
    """SEK per one unit of ``currency``: latest fixing and monthly history."""

    currency: str
    series: str
    latest_date: date
    latest_value: float
    #: month "YYYY-MM" -> (average, ultimo, observation count)
    months: dict[str, tuple[float, float, int]]

    @classmethod
    def sek(cls) -> FxSeries:
        """The trivial series for SEK itself (always 1)."""
        return cls("SEK", "", date.min, 1.0, {})

    @classmethod
    def from_entry(cls, currency: str, entry: dict[str, Any]) -> FxSeries:
        table = entry.get("monthly") or {}
        columns = list(table.get("columns") or MONTHLY_COLUMNS)
        months: dict[str, tuple[float, float, int]] = {}
        for row in table.get("rows") or []:
            record = dict(zip(columns, row, strict=False))
            months[str(record["month"])] = (
                float(record["average"]),
                float(record["ultimo"]),
                int(record["observations"]),
            )
        latest = entry["latest"]
        return cls(
            currency=currency,
            series=fx_series_id(currency),
            latest_date=date.fromisoformat(str(latest["date"])[:10]),
            latest_value=float(latest["value"]),
            months=months,
        )

    def period_end_rate(self, period_end: date) -> float | None:
        """Last fixing of the month in which the period ends."""
        if self.currency == "SEK":
            return 1.0
        month = self.months.get(f"{period_end:%Y-%m}")
        return month[1] if month else None

    def average_rate(self, period_end: date, months: int = 12) -> float | None:
        """Average daily fixing over the ``months`` calendar months that end
        with the month of ``period_end`` (None unless all months are known)."""
        if self.currency == "SEK":
            return 1.0
        total, count = 0.0, 0
        year, month = period_end.year, period_end.month
        for _ in range(months):
            item = self.months.get(f"{year:04d}-{month:02d}")
            if item is None:
                return None
            total += item[0] * item[2]
            count += item[2]
            month -= 1
            if month == 0:
                year, month = year - 1, 12
        return total / count if count else None


def load_fx_series(
    currency: str,
    *,
    offline: bool = False,
    refresh: bool = False,
    fresh_as_of: date | None = None,
) -> FxSeries | None:
    """SEK rates of ``currency`` from the Riksbank, or None if unavailable."""
    currency = currency.upper()
    if currency == "SEK":
        return FxSeries.sek()
    found = load_series(
        fx_series_id(currency),
        monthly=True,
        offline=offline,
        refresh=refresh,
        fresh_as_of=fresh_as_of,
    )
    if found is None:
        return None
    try:
        return FxSeries.from_entry(currency, found[0])
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("Riksbank data for %s is malformed: %s", currency, exc)
        return None


# --------------------------------------------------------------------------
# Risk-free rate
# --------------------------------------------------------------------------


def load_risk_free_rate(
    *, offline: bool = False, refresh: bool = False
) -> tuple[float, str, str] | None:
    """Latest 10-year government bond yield as ``(decimal, as_of, origin)``.

    Online, a cached value is reused only if it was fetched today (UTC).
    Returns None if neither the network, the cache nor the fixtures have it.
    """
    found = load_series(RISK_FREE_SERIES, monthly=False, offline=offline, refresh=refresh)
    if found is None:
        return None
    entry, origin = found
    try:
        value = round(float(entry["latest"]["value"]) / 100.0, 10)
        as_of = str(entry["latest"]["date"])[:10]
    except (KeyError, TypeError, ValueError):
        return None
    return value, as_of, origin
