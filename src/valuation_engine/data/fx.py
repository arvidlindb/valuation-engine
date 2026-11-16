"""Currency conversion plan for one company.

Statements of companies that report in another currency than they trade in
are converted the way consolidated accounts are translated:

* income-statement and cash-flow items at the AVERAGE rate of the fiscal year
  (the twelve calendar months ending with the month of the period end),
* balance-sheet items at the rate of the PERIOD END (last fixing of that
  month),
* the market snapshot (latest balance sheet) at the latest SPOT rate.

Rates come from Sveriges Riksbank (``riksbank``). If a currency has no
Riksbank series, the data is unavailable, or a fiscal year is not covered,
the plan falls back to the single Yahoo spot rate stored in the raw payload
(the behaviour before the Riksbank source existed). The path taken is
recorded in ``FxPlan.notes`` and ends up in ``source_map``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from . import riksbank
from .errors import DataUnavailableError

#: The Riksbank latest fixing is used as spot only if it is at most this many
#: days away from the date the Yahoo payload (price, market cap) was fetched.
MAX_SPOT_GAP_DAYS = 10

METHOD_NONE = "none"
METHOD_RIKSBANK = "riksbank"
METHOD_YAHOO_SPOT = "yahoo-spot"


@dataclass(frozen=True)
class FxPlan:
    """All rates needed to build one company."""

    reporting: str
    trading: str
    method: str  # METHOD_* for the reporting -> trading conversion
    spot: float  # reporting -> trading, applied to the market snapshot
    #: fiscal year -> (average rate, period-end rate), reporting -> trading;
    #: None means "use ``spot`` for every year".
    by_year: dict[int, tuple[float, float]] | None
    to_sek_spot: float  # trading -> SEK, for market values in the peer table
    to_sek_average: float  # trading -> SEK, average of the latest fiscal year
    to_sek_method: str
    notes: dict[str, str] = field(default_factory=dict)


def _yahoo_rate(raw: dict[str, Any], source: str, target: str) -> float | None:
    if source == target:
        return 1.0
    entry = (raw.get("fx") or {}).get(f"{source}{target}")
    rate = entry.get("rate") if isinstance(entry, dict) else None
    if isinstance(rate, (int, float)) and not isinstance(rate, bool) and math.isfinite(rate):
        return float(rate) if rate > 0 else None
    return None


def _payload_date(raw: dict[str, Any]) -> date | None:
    try:
        return date.fromisoformat(str(raw.get("fetched_at"))[:10])
    except ValueError:
        return None


def legacy_plan(raw: dict[str, Any], reporting: str, trading: str, reason: str = "") -> FxPlan:
    """Single Yahoo spot rate for everything (the pre-Riksbank behaviour)."""
    ticker = str(raw.get("ticker") or "")
    spot = _yahoo_rate(raw, reporting, trading)
    to_sek = _yahoo_rate(raw, trading, "SEK")
    for rate, pair in ((spot, f"{reporting}->{trading}"), (to_sek, f"{trading}->SEK")):
        if rate is None:
            raise DataUnavailableError(f"{ticker}: no FX rate {pair} in the raw payload")
    assert spot is not None and to_sek is not None
    notes: dict[str, str] = {}
    if reporting != trading:
        suffix = f" (fallback: {reason})" if reason else ""
        notes["fx.method"] = f"Yahoo spot {reporting}{trading}=X for all fiscal years{suffix}"
        notes["fx.spot"] = f"{spot!r} (Yahoo, stored in the raw payload)"
    return FxPlan(
        reporting=reporting,
        trading=trading,
        method=METHOD_NONE if reporting == trading else METHOD_YAHOO_SPOT,
        spot=spot,
        by_year=None,
        to_sek_spot=to_sek,
        to_sek_average=to_sek,
        to_sek_method=METHOD_NONE if trading == "SEK" else METHOD_YAHOO_SPOT,
        notes=notes,
    )


def _spot(
    numerator: riksbank.FxSeries,
    denominator: riksbank.FxSeries,
    as_of: date | None,
    yahoo: float | None,
) -> tuple[float, str]:
    """Cross spot rate numerator/denominator with its description."""
    dated = [s for s in (numerator, denominator) if s.currency != "SEK"]
    latest = min(s.latest_date for s in dated)
    close = as_of is None or all(
        abs((s.latest_date - as_of).days) <= MAX_SPOT_GAP_DAYS for s in dated
    )
    if close or yahoo is None:
        rate = numerator.latest_value / denominator.latest_value
        series = " / ".join(s.series for s in dated)
        return rate, f"{rate!r} (Riksbank {series}, fixing {latest.isoformat()})"
    return yahoo, (
        f"{yahoo!r} (Yahoo; the Riksbank fixing of {latest.isoformat()} is more than "
        f"{MAX_SPOT_GAP_DAYS} days from the price date {as_of.isoformat()})"
    )


def plan_fx(
    raw: dict[str, Any],
    reporting: str,
    trading: str,
    period_ends: dict[int, date],
    *,
    offline: bool = False,
    refresh: bool = False,
) -> FxPlan:
    """Build the conversion plan, preferring Riksbank rates.

    ``period_ends`` maps each fiscal year of the raw statements to its period
    end. Never touches the network for a company that reports and trades in
    SEK.
    """
    if reporting == trading == "SEK":
        return legacy_plan(raw, reporting, trading)

    as_of = _payload_date(raw)
    series: dict[str, riksbank.FxSeries] = {}
    for currency in {reporting, trading}:
        loaded = riksbank.load_fx_series(
            currency, offline=offline, refresh=refresh, fresh_as_of=as_of
        )
        if loaded is None:
            return legacy_plan(raw, reporting, trading, f"no Riksbank data for {currency}")
        series[currency] = loaded
    source, target, sek = series[reporting], series[trading], riksbank.FxSeries.sek()

    # --- trading currency -> SEK (peer table) ----------------------------
    to_sek_spot, to_sek_average, to_sek_method = 1.0, 1.0, METHOD_NONE
    if trading != "SEK":
        to_sek_spot, _ = _spot(target, sek, as_of, _yahoo_rate(raw, trading, "SEK"))
        latest_end = period_ends[max(period_ends)] if period_ends else None
        average = target.average_rate(latest_end) if latest_end else None
        to_sek_average = average if average is not None else to_sek_spot
        to_sek_method = METHOD_RIKSBANK

    if reporting == trading:
        return FxPlan(
            reporting, trading, METHOD_NONE, 1.0, None, to_sek_spot, to_sek_average, to_sek_method
        )

    # --- reporting -> trading currency -----------------------------------
    spot, spot_note = _spot(source, target, as_of, _yahoo_rate(raw, reporting, trading))
    by_year: dict[int, tuple[float, float]] = {}
    notes = {
        "fx.method": (
            f"Riksbank {source.series}"
            + (f" / {target.series}" if trading != "SEK" else "")
            + ": income and cash flow at the fiscal-year average, balance sheet at the "
            "period-end rate, market snapshot at spot"
        ),
        "fx.spot": spot_note,
    }
    for year, period_end in sorted(period_ends.items()):
        averages = (source.average_rate(period_end), target.average_rate(period_end))
        ends = (source.period_end_rate(period_end), target.period_end_rate(period_end))
        if None in averages or None in ends:
            continue  # not covered; ``standardize`` rejects the plan if the year is needed
        assert averages[0] and averages[1] and ends[0] and ends[1]
        by_year[year] = (averages[0] / averages[1], ends[0] / ends[1])
        notes[f"fx.average.{year}"] = repr(by_year[year][0])
        notes[f"fx.year_end.{year}"] = f"{by_year[year][1]!r} ({period_end:%Y-%m})"
    return FxPlan(
        reporting=reporting,
        trading=trading,
        method=METHOD_RIKSBANK,
        spot=spot,
        by_year=by_year,
        to_sek_spot=to_sek_spot,
        to_sek_average=to_sek_average,
        to_sek_method=to_sek_method,
        notes=notes,
    )


def describe_statement_fx(company: Any) -> str | None:
    """One truthful sentence on how a foreign reporter's figures were converted.

    Returns None when the company reports in its trading currency. Reads the
    conversion record the loader left in ``statements.source_map`` (``fx.method``,
    ``fx.average.<year>``, ``fx.year_end.<year>``). ``CompanyProfile.fx_rate`` is
    the SPOT rate of the market snapshot; the statements are converted per fiscal
    year (income and cash flow at the year average, balance sheet at year-end)
    unless the single-spot fallback was used, and the text says which.
    """
    profile = company.profile
    reporting, trading = profile.reporting_currency, profile.currency
    if reporting == trading:
        return None
    source_map = getattr(company.statements, "source_map", None) or {}
    method = str(source_map.get("fx.method") or "")
    spot = float(profile.fx_rate)
    year = company.statements.latest_year
    average = _leading_number(source_map.get(f"fx.average.{year}"))
    year_end = _leading_number(source_map.get(f"fx.year_end.{year}"))
    if average is not None and year_end is not None:
        provider = method.split(":", 1)[0].strip() or "per-year rates"
        return (
            f"Statements reported in {reporting}; converted to {trading} per fiscal year "
            f"({provider}): income statement and cash flow at the year average "
            f"(FY{year}: {average:.4f}), balance sheet at the year-end rate "
            f"(FY{year}: {year_end:.4f}). Market snapshot at the spot rate {spot:.4f}."
        )
    if method:
        return (
            f"Statements reported in {reporting}; every fiscal year and the market snapshot "
            f"are converted to {trading} with one spot rate, {spot:.4f} (per-year rates were "
            f"not available: {method})."
        )
    return (
        f"Statements reported in {reporting} and converted to {trading}; the conversion "
        f"method was not recorded. Market snapshot at the spot rate {spot:.4f}."
    )


def _leading_number(text: Any) -> float | None:
    """First number of a ``source_map`` entry such as ``'9.20133 (2025-12)'``."""
    if text is None:
        return None
    try:
        value = float(str(text).split()[0])
    except (ValueError, IndexError):
        return None
    return value if math.isfinite(value) else None
