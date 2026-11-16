"""Structural types the presentation layer (charts, Excel, app) is written against.

The Monte Carlo model and the pipeline are developed in parallel with this
package, so nothing here imports them. Any object with the listed attributes
works:

* :class:`MonteCarloLike` - the fields of ``MonteCarloResult`` (see PLAN.md).
* :class:`ValuationReportLike` - the bundle ``pipeline.run_valuation`` returns.
* :class:`ReportBundle` - a plain dataclass that satisfies
  :class:`ValuationReportLike`; handy for tests, notebooks and the app.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from valuation_engine.core.types import CompanyData, MarketParameters, ValuationResult
from valuation_engine.models.comps import CompsResult
from valuation_engine.models.dcf import DCFResult
from valuation_engine.models.lbo import LBOResult

__all__ = [
    "METHOD_KEYS",
    "MonteCarloLike",
    "ReportBundle",
    "ValuationReportLike",
    "percentile",
]

#: Keys of ``ValuationReportLike.not_applicable`` (one per model).
METHOD_KEYS: tuple[str, ...] = ("dcf", "comps", "lbo", "monte_carlo")


@runtime_checkable
class MonteCarloLike(Protocol):
    """Fields of the Monte Carlo result that charts and the export read."""

    ticker: str
    currency: str
    price: float
    base_value: float
    n_simulations: int
    #: Value per share per simulation (Gordon growth terminal value).
    values: np.ndarray
    #: Value per share per simulation (exit multiple terminal value).
    values_exit: np.ndarray
    #: One row per simulation; columns are the simulated drivers.
    inputs: pd.DataFrame
    #: Percentile (5, 10, 25, 50, 75, 90, 95) -> value per share.
    percentiles: dict[Any, float]
    mean: float
    std: float
    standard_error: float
    prob_above_price: float
    #: Index = driver; columns input_low, input_high, value_low, value_high, swing.
    tornado: pd.DataFrame
    #: Share of output variance per driver.
    variance_contribution: pd.Series
    #: Columns n, running_mean, running_median, standard_error.
    convergence: pd.DataFrame
    config: Any


@runtime_checkable
class ValuationReportLike(Protocol):
    """Everything the charts, the Excel export and the app need for one company.

    A model that could not be applied is ``None`` and its reason is stored in
    ``not_applicable`` under the model's key (see ``METHOD_KEYS``).
    ``valuations`` holds the football-field rows of all applicable methods; a
    52-week trading range is added by the presentation layer when missing.

    Optional attributes, read with ``getattr`` when present: ``params``
    (:class:`MarketParameters`) and ``warnings`` (``list[str]``).
    """

    company: CompanyData
    dcf: DCFResult | None
    comps: CompsResult | None
    lbo: LBOResult | None
    monte_carlo: MonteCarloLike | None
    valuations: list[ValuationResult]
    not_applicable: dict[str, str]


@dataclass
class ReportBundle:
    """Concrete :class:`ValuationReportLike`."""

    company: CompanyData
    dcf: DCFResult | None = None
    comps: CompsResult | None = None
    lbo: LBOResult | None = None
    monte_carlo: MonteCarloLike | None = None
    valuations: list[ValuationResult] = field(default_factory=list)
    not_applicable: dict[str, str] = field(default_factory=dict)
    params: MarketParameters = field(default_factory=MarketParameters)
    warnings: list[str] = field(default_factory=list)


def percentile(monte_carlo: MonteCarloLike, level: float) -> float:
    """Percentile ``level`` (0-100) of the simulated value per share.

    Reads ``monte_carlo.percentiles`` (keys may be ints, floats or strings) and
    falls back to computing it from ``values``.
    """
    table = getattr(monte_carlo, "percentiles", None) or {}
    for key in (level, int(level), float(level), str(int(level)), f"p{int(level)}"):
        if key in table:
            return float(table[key])
    values = np.asarray(monte_carlo.values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, level)) if values.size else float("nan")
