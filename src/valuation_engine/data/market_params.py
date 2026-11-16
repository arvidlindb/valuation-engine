"""Sourced market parameters (risk-free rate, equity risk premium, tax rate).

* ``risk_free_rate``: latest 10-year Swedish government bond yield from the
  Riksbank (series ``SEGVB10YC``); falls back to the cache, the fixtures and
  finally the default of :class:`MarketParameters`.
* ``equity_risk_premium`` and ``tax_rate``: Damodaran's figures for Sweden,
  kept in the committed file ``reference/market_reference.json`` (never
  downloaded at runtime; the file explains how to refresh it).
* ``benchmark`` and ``long_term_growth``: defaults of the dataclass.

:func:`market_parameter_sources` returns value, source and as-of date for
every parameter so that reports can cite them.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from ..core.types import MarketParameters
from . import riksbank

logger = logging.getLogger(__name__)

REFERENCE_PATH = Path(__file__).resolve().parent / "reference" / "market_reference.json"
_RIKSBANK_SERIES_URL = f"{riksbank.BASE_URL}/Observations/Latest/{riksbank.RISK_FREE_SERIES}"


@dataclass(frozen=True)
class ParameterSource:
    """Provenance of one market parameter."""

    name: str
    value: float | str
    source: str
    as_of: str | None = None  # ISO date the value refers to
    url: str | None = None
    origin: str = "default"  # "network", "cache", "fixtures", "reference file" or "default"


def _reference() -> dict[str, dict[str, Any]]:
    """Parameters of the committed reference file ({} if unreadable)."""
    try:
        with REFERENCE_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        logger.warning("market reference file unreadable (%s); using defaults", exc)
        return {}
    parameters = payload.get("parameters") if isinstance(payload, dict) else None
    return parameters if isinstance(parameters, dict) else {}


def _from_reference(
    name: str, default: float, reference: dict[str, dict[str, Any]]
) -> ParameterSource:
    entry = reference.get(name)
    value = entry.get("value") if isinstance(entry, dict) else None
    if isinstance(entry, dict) and isinstance(value, (int, float)) and not isinstance(value, bool):
        return ParameterSource(
            name=name,
            value=float(value),
            source=str(entry.get("source") or "reference file"),
            as_of=entry.get("as_of"),
            url=entry.get("url"),
            origin="reference file",
        )
    return ParameterSource(name, default, "default of MarketParameters (reference file missing)")


def market_parameter_sources(
    *, offline: bool = False, refresh: bool = False
) -> dict[str, ParameterSource]:
    """Value and provenance of every field of :class:`MarketParameters`.

    Also contains the informational entry ``country_risk_premium`` (already
    included in ``equity_risk_premium``; not a field of the dataclass).
    """
    defaults = MarketParameters()
    reference = _reference()

    risk_free = riksbank.load_risk_free_rate(offline=offline, refresh=refresh)
    if risk_free is not None:
        value, as_of, origin = risk_free
        rate = ParameterSource(
            name="risk_free_rate",
            value=value,
            source=(
                "Sveriges Riksbank, Swedish 10-year government bond yield "
                f"(series {riksbank.RISK_FREE_SERIES})"
            ),
            as_of=as_of,
            url=_RIKSBANK_SERIES_URL,
            origin=origin,
        )
    else:
        rate = ParameterSource(
            name="risk_free_rate",
            value=defaults.risk_free_rate,
            source="default of MarketParameters (Riksbank data unavailable)",
        )

    sources = {
        "risk_free_rate": rate,
        "equity_risk_premium": _from_reference(
            "equity_risk_premium", defaults.equity_risk_premium, reference
        ),
        "tax_rate": _from_reference("tax_rate", defaults.tax_rate, reference),
        "benchmark": ParameterSource(
            "benchmark", defaults.benchmark, "OMX Stockholm PI on Yahoo Finance (project default)"
        ),
        "long_term_growth": ParameterSource(
            "long_term_growth",
            defaults.long_term_growth,
            "Riksbank inflation target of 2 % as nominal anchor (project default)",
        ),
        "country_risk_premium": _from_reference("country_risk_premium", 0.0, reference),
    }
    return sources


def load_market_parameters(*, offline: bool = False, refresh: bool = False) -> MarketParameters:
    """Market parameters with a live risk-free rate and sourced ERP/tax rate.

    Never raises for missing data: every parameter falls back to the default
    of :class:`MarketParameters`. Use :func:`market_parameter_sources` for
    the provenance of the values.
    """
    sources = market_parameter_sources(offline=offline, refresh=refresh)
    names = {field.name for field in fields(MarketParameters)}
    return MarketParameters(**{name: sources[name].value for name in names if name in sources})
