"""Data layer: Yahoo Finance -> standardized company data for Swedish equities.

Public API (signatures fixed by PLAN.md)::

    load_company(ticker, *, offline=False, refresh=False) -> CompanyData
    list_universe(*, offline=False, refresh=False) -> list[str]
    load_peer_snapshots(*, offline=False, refresh=False) -> list[PeerSnapshot]
    load_market_parameters(*, offline=False, refresh=False) -> MarketParameters
    market_parameter_sources(*, offline=False, refresh=False) -> dict[str, ParameterSource]
    describe_statement_fx(company) -> str | None      # how a foreign reporter was converted

Offline mode (``offline=True`` or env ``VALUATION_ENGINE_OFFLINE=1``) reads the
fixtures and the local cache only. Batch download of the whole universe::

    python -m valuation_engine.data.download --all

See ``docs/data_mapping.md`` for the mapping and its caveats.
"""

from __future__ import annotations

from .errors import (
    DataError,
    DataUnavailableError,
    FetchError,
    NetworkDisabledError,
    RateLimitError,
)
from .fx import describe_statement_fx
from .loader import (
    build_company,
    list_universe,
    load_company,
    load_peer_snapshots,
    load_universe_entries,
    regression_beta,
)
from .market_params import ParameterSource, load_market_parameters, market_parameter_sources
from .store import cache_dir, fixtures_dir, is_offline
from .universe import UniverseEntry, UniverseStats

__all__ = [
    "DataError",
    "DataUnavailableError",
    "FetchError",
    "NetworkDisabledError",
    "ParameterSource",
    "RateLimitError",
    "UniverseEntry",
    "UniverseStats",
    "build_company",
    "cache_dir",
    "describe_statement_fx",
    "fixtures_dir",
    "is_offline",
    "list_universe",
    "load_company",
    "load_market_parameters",
    "load_peer_snapshots",
    "load_universe_entries",
    "market_parameter_sources",
    "regression_beta",
]
