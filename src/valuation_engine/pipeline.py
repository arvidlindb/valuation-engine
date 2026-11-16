"""Valuation pipeline: one company in, one :class:`ValuationReport` out.

Public API::

    run_valuation(ticker, ...) -> ValuationReport
    value_company(company, universe, params, ...) -> ValuationReport
    report_summary(report) -> dict        # JSON-ready
    save_report(report, output_dir)       # charts, Excel workbook, summary.json
    summary_text(report) -> str           # plain-text table for the terminal

Orchestration (design decisions in PLAN.md)
---------------------------------------------
1. **Market parameters.** ``params=None`` loads the sourced parameters
   (Riksbank risk-free rate, Damodaran equity risk premium and tax rate). The
   provenance of every parameter is kept in ``ValuationReport.parameter_sources``.
2. **Comps first.** The comparable company analysis selects the peer group.
3. **DCF on the same peers.** The peers selected by comps are handed to
   ``run_dcf(..., peers=..., peers_are_peer_group=True)``, which uses all of
   them for the relevered peer beta and for the default exit multiple (without
   the flag ``run_dcf`` would narrow the list to the target's industry or
   sector). When comps is not applicable, or its peers carry fewer than
   ``dcf.MIN_PEERS`` usable betas or EV/EBITDA multiples, the full universe is
   used instead (the DCF's own industry -> sector logic) and a warning says so.
   **One peer multiple:** with the comps peer group, the DCF's default exit
   multiple is the EV/EBITDA median comps itself reports (after its outlier
   fence), clipped to the DCF's band. It is adopted only when comps applied
   EV/EBITDA to the target and the median rests on at least ``dcf.MIN_PEERS``
   peer multiples. Otherwise a warning says why and the DCF takes the median of
   the comps peers' raw multiples when at least ``dcf.MIN_PEERS`` of them are
   usable, else it falls back to the full universe as described above.
4. **LBO shares the DCF's view.** Without caller-supplied ``lbo_assumptions``
   and with a DCF base case, the LBO's operating forecast (growth, margin,
   D&A, capex, NWC, tax rate) is the DCF's (:func:`lbo_assumptions_from_dcf`);
   without a DCF the LBO keeps its own history-based defaults. The LBO exits
   at the DCF's exit multiple (instead of exit = entry) unless
   ``lbo_assumptions`` carry an explicit ``exit_multiple``. The LBO financing
   defaults are always derived with the loaded market parameters.
5. **Monte Carlo** simulates around the base DCF (``dcf_result=``).

Failure handling
----------------
* ``NotApplicableError`` of a model -> its reason goes to
  ``ValuationReport.not_applicable`` under ``dcf`` / ``comps`` / ``lbo`` /
  ``monte_carlo``.
* A model that runs but yields no football-field bar (LBO without a solvable
  base price, comps without any applicable multiple) -> ``warnings``.
* Any other exception inside one model -> ``warnings`` (with the exception
  type and message) and ``errors``; the remaining models still run.
* A company that cannot be loaded raises ``DataUnavailableError`` from
  :func:`run_valuation`.
"""

from __future__ import annotations

import json
import logging
import math
import textwrap
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from valuation_engine import data as data_layer
from valuation_engine.core.types import (
    CompanyData,
    DCFAssumptions,
    MarketParameters,
    NotApplicableError,
    PeerSnapshot,
    ValuationResult,
)
from valuation_engine.data import DataError, ParameterSource
from valuation_engine.models import comps as comps_model
from valuation_engine.models import dcf as dcf_model
from valuation_engine.models import lbo as lbo_model
from valuation_engine.models import monte_carlo as mc_model
from valuation_engine.models.comps import CompsResult
from valuation_engine.models.dcf import DCFResult
from valuation_engine.models.lbo import LBOAssumptions
from valuation_engine.viz.protocols import METHOD_KEYS, ReportBundle

__all__ = [
    "METHOD_NAMES",
    "PEER_SOURCE_COMPS",
    "PEER_SOURCE_NONE",
    "PEER_SOURCE_UNIVERSE",
    "ValuationReport",
    "lbo_assumptions_from_dcf",
    "report_summary",
    "run_valuation",
    "save_report",
    "summary_text",
    "value_company",
]

logger = logging.getLogger(__name__)

#: Display name per key of ``ValuationReport.not_applicable``.
METHOD_NAMES: dict[str, str] = {
    "dcf": "DCF",
    "comps": "Comps",
    "lbo": "LBO",
    "monte_carlo": "Monte Carlo",
}

#: Values of ``ValuationReport.dcf_peer_source``.
PEER_SOURCE_COMPS = "comps"
PEER_SOURCE_UNIVERSE = "universe"
PEER_SOURCE_NONE = "none"


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


@dataclass
class ValuationReport(ReportBundle):
    """Result of :func:`run_valuation`; satisfies ``viz.protocols.ValuationReportLike``.

    Inherited from :class:`ReportBundle`: ``company``, ``dcf``, ``comps``,
    ``lbo``, ``monte_carlo`` (``None`` when a model produced no result),
    ``valuations`` (football-field rows of all applicable methods, without the
    52-week market row, which the presentation layer adds), ``not_applicable``
    (model key -> reason), ``params`` and ``warnings``.
    """

    #: Value, source and as-of date of every market parameter.
    parameter_sources: dict[str, ParameterSource] = field(default_factory=dict)
    #: Peers selected by comps, nearest first (empty when comps did not run).
    peers: list[PeerSnapshot] = field(default_factory=list)
    #: Peer set behind the DCF's peer beta and default exit multiple:
    #: ``"comps"``, ``"universe"`` or ``"none"``.
    dcf_peer_source: str = PEER_SOURCE_NONE
    #: Where the LBO exit multiple comes from (plain text).
    lbo_exit_multiple_source: str = ""
    #: False when the size premium in the cost of equity was switched off.
    size_premium: bool = True
    #: Informational remarks about the orchestration (not problems).
    notes: list[str] = field(default_factory=list)
    #: Model key -> "ExceptionType: message" for models that crashed.
    errors: dict[str, str] = field(default_factory=dict)
    #: Seconds per stage (``load``, ``comps``, ``dcf``, ``lbo``, ``monte_carlo``, ``total``).
    timings: dict[str, float] = field(default_factory=dict)
    #: Local time the report was generated (ISO, seconds).
    generated_at: str = ""
    n_simulations: int = 0
    seed: int | None = None

    @property
    def ticker(self) -> str:
        return self.company.ticker

    @property
    def price(self) -> float:
        return float(self.company.market.price)

    @property
    def currency(self) -> str:
        return self.company.market.currency

    @property
    def peer_tickers(self) -> list[str]:
        """Tickers of the comps peers, nearest first."""
        return [peer.ticker for peer in self.peers]

    @property
    def status(self) -> str:
        """``ok`` (every model gave a result), ``partial`` or ``none``."""
        results = [self.dcf, self.comps, self.lbo, self.monte_carlo]
        available = sum(result is not None for result in results)
        if available == len(results) and self.valuations:
            return "ok"
        return "partial" if self.valuations else "none"

    def summary(self) -> dict[str, Any]:
        """JSON-ready summary, see :func:`report_summary`."""
        return report_summary(self)


# --------------------------------------------------------------------------
# Peer hand-off
# --------------------------------------------------------------------------


def _usable_peer_counts(peers: Sequence[PeerSnapshot]) -> tuple[int, int]:
    """Number of peers the DCF can use for (peer beta, exit multiple).

    Mirrors the filters of ``dcf._peer_unlevered_beta`` and
    ``dcf.resolve_exit_multiple``.
    """
    low, high = dcf_model.RAW_BETA_BAND
    betas = 0
    multiples = 0
    for peer in peers:
        beta = peer.beta
        if (
            beta is not None
            and math.isfinite(beta)
            and low < beta <= high
            and peer.debt_to_equity is not None
            and math.isfinite(peer.debt_to_equity)
        ):
            betas += 1
        multiple = peer.ev_ebitda
        if multiple is not None and 0 < multiple <= dcf_model.MAX_PEER_MULTIPLE:
            multiples += 1
    return betas, multiples


def _comps_exit_multiple(
    comps: CompsResult | None,
) -> tuple[tuple[float, str] | None, str]:
    """EV/EBITDA median of the comps peer group, if the DCF may adopt it.

    Returns ``((median, source label), "")`` or ``(None, reason)``. The median is
    the one comps reports and applies (``multiples_summary``, i.e. after comps'
    bounds and outlier fence), so comps and the DCF exit multiple rest on one
    peer multiple. It is adopted only when comps itself applied EV/EBITDA to the
    target AND the median rests on at least ``dcf.MIN_PEERS`` peer multiples,
    the same minimum the DCF demands of any peer statistic.
    """
    if comps is None:
        return None, "comps did not run"
    if "ev_ebitda" not in comps.implied_values.index:
        why = comps.skipped.get("ev_ebitda", "no implied value")
        return None, f"comps did not apply EV/EBITDA ({why})"
    try:
        row = comps.multiples_summary.loc["ev_ebitda"]
        median, count = float(row["median"]), float(row["n"])
    except (KeyError, TypeError, ValueError):
        return None, "comps reports no EV/EBITDA median"
    if not (math.isfinite(median) and median > 0 and math.isfinite(count)):
        return None, "comps reports no usable EV/EBITDA median"
    if count < dcf_model.MIN_PEERS:
        return None, (
            f"the comps EV/EBITDA median rests on only {int(count)} peer multiples, "
            f"{dcf_model.MIN_PEERS} needed"
        )
    return (median, f"comps median EV/EBITDA ({int(count)} peers)"), ""


def lbo_assumptions_from_dcf(
    assumptions: LBOAssumptions, dcf_assumptions: DCFAssumptions
) -> LBOAssumptions:
    """``assumptions`` with the operating forecast of the DCF.

    One company, one operating forecast: revenue growth and EBITDA margin per
    year, D&A, capex and operating NWC as a share of revenue and the tax rate are
    taken from ``dcf_assumptions``. The holding period is kept; if it is longer
    than the DCF forecast the last DCF year is repeated. Transaction, financing
    and exit terms are untouched.
    """
    years = assumptions.holding_period

    def path(values: tuple[float, ...]) -> tuple[float, ...]:
        return tuple(float(values[min(t, len(values) - 1)]) for t in range(years))

    return replace(
        assumptions,
        revenue_growth=path(dcf_assumptions.revenue_growth),
        ebitda_margin=path(dcf_assumptions.ebitda_margin),
        da_pct_revenue=float(dcf_assumptions.da_pct_revenue),
        capex_pct_revenue=float(dcf_assumptions.capex_pct_revenue),
        nwc_pct_revenue=float(dcf_assumptions.nwc_pct_revenue),
        tax_rate=float(dcf_assumptions.tax_rate),
    )


def _run_dcf(
    company: CompanyData,
    assumptions: DCFAssumptions | None,
    params: MarketParameters,
    comps_peers: list[PeerSnapshot],
    universe: list[PeerSnapshot],
    size_premium: bool,
    use_comps_peers: bool,
    comps: CompsResult | None = None,
) -> tuple[DCFResult, str, list[str], list[str]]:
    """Run the DCF on the comps peers when they suffice, else on the universe.

    Returns (result, peer source, notes, warnings).
    """
    notes: list[str] = []
    warnings: list[str] = []
    premium = None if size_premium else 0.0
    minimum = dcf_model.MIN_PEERS
    needs_beta = assumptions is None or assumptions.wacc is None
    needs_multiple = assumptions is None or assumptions.exit_multiple is None

    if use_comps_peers and comps_peers:
        betas, multiples = _usable_peer_counts(comps_peers)
        peer_multiple, rejected = _comps_exit_multiple(comps) if needs_multiple else (None, "")
        has_multiple = peer_multiple is not None or multiples >= minimum or not needs_multiple
        if (betas >= minimum or not needs_beta) and has_multiple:
            # The target's own snapshot rides along: the DCF never uses it as a peer
            # but reads its SEK market cap for the size premium.
            own = next((p for p in universe if p.ticker == company.ticker), None)
            group = [*comps_peers, own] if own is not None else list(comps_peers)
            result = dcf_model.run_dcf(
                company,
                assumptions,
                params,
                group,
                size_premium=premium,
                peers_are_peer_group=True,
                peer_multiple=peer_multiple,
            )
            if peer_multiple is not None:
                multiple_basis = "the exit multiple is the comps EV/EBITDA median"
            elif needs_multiple:
                multiple_basis = (
                    f"the exit multiple is the median of their {multiples} usable EV/EBITDA "
                    "multiples"
                )
                warnings.append(
                    f"DCF: the comps EV/EBITDA median is not used as exit multiple ({rejected}); "
                    f"the median of the {multiples} usable EV/EBITDA multiples of the comps "
                    "peers is used instead."
                )
            else:
                multiple_basis = "the exit multiple is given in the assumptions"
            notes.append(
                f"DCF: the peer beta is based on the {len(comps_peers)} peers selected by comps "
                f"({betas} usable betas); {multiple_basis}."
            )
            return result, PEER_SOURCE_COMPS, notes, warnings
        detail = f"; {rejected}" if rejected else ""
        warnings.append(
            f"DCF: the comps peer group is too thin for the DCF ({betas} usable betas and "
            f"{multiples} usable EV/EBITDA multiples among {len(comps_peers)} peers, {minimum} "
            f"needed{detail}); the full universe (same industry, else sector) is used for the "
            "peer beta and the exit multiple."
        )
    elif use_comps_peers and universe:
        warnings.append(
            "DCF: no comps peer group available; the full universe (same industry, else sector) "
            "is used for the peer beta and the exit multiple."
        )

    if universe:
        result = dcf_model.run_dcf(company, assumptions, params, universe, size_premium=premium)
        if not use_comps_peers:
            notes.append(
                "DCF: peer beta and exit multiple are based on the full universe "
                "(same industry, else sector); the comps peer hand-off is switched off."
            )
        return result, PEER_SOURCE_UNIVERSE, notes, warnings

    result = dcf_model.run_dcf(company, assumptions, params, None, size_premium=premium)
    warnings.append(
        "DCF: no peer universe available; own regression beta and own EV/EBITDA multiple are used."
    )
    return result, PEER_SOURCE_NONE, notes, warnings


# --------------------------------------------------------------------------
# Market parameters
# --------------------------------------------------------------------------


def _resolve_sources(
    params: MarketParameters, sources: dict[str, ParameterSource] | None
) -> dict[str, ParameterSource]:
    """Provenance per field of ``params``; values that differ from the sourced
    ones (caller overrides) are labelled as supplied by the caller."""
    resolved: dict[str, ParameterSource] = {}
    for item in fields(MarketParameters):
        value = getattr(params, item.name)
        known = (sources or {}).get(item.name)
        if known is not None and _same_value(known.value, value):
            resolved[item.name] = known
        else:
            resolved[item.name] = ParameterSource(
                name=item.name, value=value, source="supplied by the caller", origin="caller"
            )
    for name, known in (sources or {}).items():
        resolved.setdefault(name, known)
    return resolved


def _same_value(left: Any, right: Any) -> bool:
    if isinstance(left, str) or isinstance(right, str):
        return left == right
    try:
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def _timed(timings: dict[str, float], key: str, started: float) -> None:
    timings[key] = round(time.perf_counter() - started, 4)


def _guard(
    report: ValuationReport, key: str, step: Callable[[], None], timings: dict[str, float]
) -> None:
    """Run one model step; route its failure into the report."""
    started = time.perf_counter()
    try:
        step()
    except NotApplicableError as exc:
        report.not_applicable[key] = str(exc)
    except Exception as exc:  # noqa: BLE001 - one failing model must not kill the report
        message = f"{type(exc).__name__}: {exc}"
        report.errors[key] = message
        report.warnings.append(
            f"{METHOD_NAMES[key]}: the model failed and was skipped ({message})."
        )
        logger.warning("%s: %s failed: %s", report.company.ticker, key, message, exc_info=True)
    finally:
        _timed(timings, key, started)


def value_company(
    company: CompanyData,
    universe: Sequence[PeerSnapshot] | None,
    params: MarketParameters,
    *,
    parameter_sources: dict[str, ParameterSource] | None = None,
    dcf_assumptions: DCFAssumptions | None = None,
    lbo_assumptions: LBOAssumptions | None = None,
    n_simulations: int = 10_000,
    seed: int | None = 42,
    run_monte_carlo: bool = True,
    size_premium: bool = True,
    use_comps_peers: bool = True,
) -> ValuationReport:
    """Run all models on data that is already loaded (no I/O).

    This is the core of :func:`run_valuation`; the batch run calls it directly
    with the universe table and the market parameters loaded once. See the
    module docstring for the orchestration and :func:`run_valuation` for the
    arguments.
    """
    total_started = time.perf_counter()
    peer_universe = list(universe or [])
    by_ticker = {peer.ticker: peer for peer in peer_universe}
    timings: dict[str, float] = {}
    report = ValuationReport(
        company=company,
        params=params,
        parameter_sources=_resolve_sources(params, parameter_sources),
        size_premium=size_premium,
        n_simulations=int(n_simulations),
        seed=seed,
        generated_at=datetime.now().isoformat(timespec="seconds"),
    )
    if not size_premium:
        report.notes.append(
            "DCF: the size premium in the cost of equity is switched off (plain CAPM)."
        )

    # ---- comps ------------------------------------------------------------
    def comps_step() -> None:
        result = comps_model.run_comps(company, peer_universe)
        report.comps = result
        report.peers = [by_ticker[t] for t in result.peer_tickers if t in by_ticker]
        rows = result.to_valuation_results()
        report.valuations.extend(rows)
        if not rows:
            reasons = "; ".join(f"{name}: {why}" for name, why in result.skipped.items())
            report.warnings.append(
                "Comps: no multiple could be applied" + (f" ({reasons})." if reasons else ".")
            )

    _guard(report, "comps", comps_step, timings)

    # ---- DCF --------------------------------------------------------------
    def dcf_step() -> None:
        result, source, notes, warnings = _run_dcf(
            company,
            dcf_assumptions,
            params,
            report.peers,
            peer_universe,
            size_premium,
            use_comps_peers,
            report.comps,
        )
        report.dcf = result
        report.dcf_peer_source = source
        report.notes.extend(notes)
        report.warnings.extend(warnings)
        report.valuations.extend(result.to_valuation_results())

    _guard(report, "dcf", dcf_step, timings)

    # ---- LBO --------------------------------------------------------------
    def lbo_step() -> None:
        default_notes: list[str] = []
        if lbo_assumptions is None:
            # Same derivation as ``lbo.default_assumptions(company, params)``, plus the
            # notes ``run_lbo`` would have reported had it built the defaults itself.
            assumptions, default_notes = lbo_model._default_assumptions_with_notes(company, params)
            if report.dcf is not None:
                # One company, one operating forecast (the notes about the LBO's own
                # history-based drivers no longer apply).
                assumptions = lbo_assumptions_from_dcf(assumptions, report.dcf.assumptions)
                default_notes = []
                report.notes.append(
                    "LBO: the operating forecast (revenue growth, EBITDA margin, D&A, capex, "
                    "NWC and tax rate) is the one of the DCF."
                )
            else:
                report.notes.append(
                    "LBO: no DCF base case, so the LBO uses its own history-based operating "
                    "forecast."
                )
        else:
            assumptions = lbo_assumptions
        if assumptions.exit_multiple is not None:
            report.lbo_exit_multiple_source = "given in the LBO assumptions"
        elif report.dcf is not None and report.dcf.exit_multiple > 0:
            assumptions = replace(assumptions, exit_multiple=float(report.dcf.exit_multiple))
            report.lbo_exit_multiple_source = (
                f"the exit multiple of the DCF ({report.dcf.exit_multiple_source})"
            )
        else:
            report.lbo_exit_multiple_source = "equal to the entry multiple (no DCF available)"
        report.notes.append(f"LBO: the exit multiple is {report.lbo_exit_multiple_source}.")
        result = lbo_model.run_lbo(company, assumptions)
        result.warnings[:0] = default_notes
        report.lbo = result
        rows = result.to_valuation_results()
        report.valuations.extend(rows)
        if not rows:
            report.warnings.append(
                "LBO: no ability-to-pay range (no positive price delivers the base target IRR "
                f"of {assumptions.target_irr_base:.0%})."
            )

    _guard(report, "lbo", lbo_step, timings)

    # ---- Monte Carlo ------------------------------------------------------
    def monte_carlo_step() -> None:
        if not run_monte_carlo:
            raise NotApplicableError("skipped: the Monte Carlo simulation was switched off")
        if report.dcf is None:
            reason = report.not_applicable.get("dcf") or report.errors.get("dcf") or "no DCF"
            raise NotApplicableError(f"requires a DCF base case: {reason}")
        config = mc_model.MonteCarloConfig(n_simulations=int(n_simulations), seed=seed)
        result = mc_model.run_monte_carlo(company, config=config, dcf_result=report.dcf)
        report.monte_carlo = result
        report.valuations.extend(result.to_valuation_results())

    _guard(report, "monte_carlo", monte_carlo_step, timings)

    _timed(timings, "total", total_started)
    report.timings = timings
    return report


def run_valuation(
    ticker: str,
    *,
    offline: bool = False,
    refresh: bool = False,
    params: MarketParameters | None = None,
    dcf_assumptions: DCFAssumptions | None = None,
    lbo_assumptions: LBOAssumptions | None = None,
    n_simulations: int = 10_000,
    seed: int | None = 42,
    universe: list[PeerSnapshot] | None = None,
    run_monte_carlo: bool = True,
    size_premium: bool = True,
    use_comps_peers: bool = True,
) -> ValuationReport:
    """Full valuation of one company: comps, DCF, LBO and Monte Carlo.

    Parameters
    ----------
    ticker:
        Yahoo ticker, e.g. ``"ATCO-A.ST"``.
    offline, refresh:
        Passed to the data layer (``offline`` reads fixtures and cache only;
        the environment variable ``VALUATION_ENGINE_OFFLINE=1`` does the same).
    params:
        Market parameters; ``None`` loads the sourced ones
        (``data.load_market_parameters``).
    dcf_assumptions, lbo_assumptions:
        Override the history-driven defaults. An LBO ``exit_multiple`` of
        ``None`` is filled with the DCF's exit multiple.
    n_simulations, seed:
        Monte Carlo settings (a fixed seed makes the run reproducible).
    universe:
        Peer snapshots; ``None`` loads them (``data.load_peer_snapshots``).
    run_monte_carlo:
        ``False`` skips the simulation.
    size_premium:
        ``False`` removes the size premium from the DCF cost of equity.
    use_comps_peers:
        ``False`` lets the DCF pick its peers from the full universe instead of
        using the comps peer group.

    Raises
    ------
    DataUnavailableError
        When the company itself cannot be loaded. Problems inside single
        models never raise; see ``not_applicable``, ``warnings`` and ``errors``
        of the report.
    """
    started = time.perf_counter()
    company = data_layer.load_company(ticker, offline=offline, refresh=refresh)

    early_warnings: list[str] = []
    try:
        sources = data_layer.market_parameter_sources(offline=offline, refresh=refresh)
    except Exception as exc:  # noqa: BLE001 - provenance must never block a valuation
        sources = {}
        early_warnings.append(
            f"Market parameter sources unavailable ({type(exc).__name__}: {exc}); "
            "defaults of MarketParameters are used where needed."
        )
    if params is None:
        names = {item.name for item in fields(MarketParameters)}
        params = MarketParameters(
            **{name: source.value for name, source in sources.items() if name in names}
        )
    if universe is None:
        try:
            universe = data_layer.load_peer_snapshots(offline=offline)
        except DataError as exc:
            universe = []
            early_warnings.append(f"Peer universe unavailable ({exc}); comps cannot be run.")
    load_seconds = round(time.perf_counter() - started, 4)

    report = value_company(
        company,
        universe,
        params,
        parameter_sources=sources,
        dcf_assumptions=dcf_assumptions,
        lbo_assumptions=lbo_assumptions,
        n_simulations=n_simulations,
        seed=seed,
        run_monte_carlo=run_monte_carlo,
        size_premium=size_premium,
        use_comps_peers=use_comps_peers,
    )
    report.warnings[:0] = early_warnings
    report.timings = {
        "load": load_seconds,
        **report.timings,
        "total": round(time.perf_counter() - started, 4),
    }
    return report


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------


def _clean(value: Any) -> Any:
    """Make ``value`` JSON-safe: NaN/inf -> None, numpy scalars -> Python."""
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if value is None or isinstance(value, str):
        return value
    return str(value)


def _ratio(value: float | None, price: float) -> float | None:
    if value is None or not math.isfinite(value) or not price > 0:
        return None
    return value / price


def _valuation_row(row: ValuationResult, price: float) -> dict[str, Any]:
    return {
        "method": row.method.value,
        "label": row.label,
        "low": row.low,
        "base": row.base,
        "high": row.high,
        "base_to_price": _ratio(row.base, price),
        "enterprise_value": row.enterprise_value,
        "equity_value": row.equity_value,
    }


def _comps_block(comps: CompsResult | None) -> dict[str, Any] | None:
    if comps is None:
        return None
    multiples: dict[str, Any] = {}
    for name, row in comps.implied_values.iterrows():
        multiples[str(name)] = {
            "n_peers": int(row["n_peers"]),
            "multiple_low": float(row["multiple_low"]),
            "multiple_base": float(row["multiple_base"]),
            "multiple_high": float(row["multiple_high"]),
        }
    return {
        "candidate_scope": comps.candidate_scope,
        "chosen_k": comps.chosen_k,
        "n_candidates": int(len(comps.candidates)),
        "n_peers": int(len(comps.peers)),
        "multiples": multiples,
        "skipped": dict(comps.skipped),
        "fallbacks": list(comps.fallbacks),
    }


def report_summary(report: ValuationReport) -> dict[str, Any]:
    """Machine-readable summary of a report (plain dicts, lists and scalars).

    Everything that varies between two identical runs (wall-clock times, the
    generation timestamp) sits under the key ``timing``.
    """
    company = report.company
    profile, market = company.profile, company.market
    price = float(market.price)
    dcf, lbo, mc = report.dcf, report.lbo, report.monte_carlo

    dcf_block: dict[str, Any] | None = None
    if dcf is not None:
        wacc = dcf.wacc_result
        dcf_block = {
            "value_per_share_gordon": dcf.value_per_share_gordon,
            "value_per_share_exit": dcf.value_per_share_exit,
            "wacc": dcf.wacc,
            "wacc_source": "CAPM" if wacc is not None else "assumption",
            "terminal_growth": dcf.assumptions.terminal_growth,
            "exit_multiple": dcf.exit_multiple,
            "exit_multiple_source": dcf.exit_multiple_source,
            "implied_exit_multiple_gordon": dcf.implied_exit_multiple,
            "implied_perpetual_growth_exit": dcf.implied_perpetual_growth,
            "tv_share_gordon": dcf.tv_share_gordon,
            "tv_share_exit": dcf.tv_share_exit,
            "peer_source": report.dcf_peer_source,
            "beta": wacc.beta if wacc else None,
            "beta_method": wacc.beta_method if wacc else None,
            "own_blume_beta": wacc.blume_beta if wacc else None,
            "peer_beta": wacc.peer_beta if wacc else None,
            "peer_group": wacc.peer_group if wacc else None,
            "peer_count_beta": wacc.peer_count if wacc else None,
            "size_premium": wacc.size_premium if wacc else None,
            "cost_of_equity": wacc.cost_of_equity if wacc else None,
            "pre_tax_cost_of_debt": wacc.pre_tax_cost_of_debt if wacc else None,
            "after_tax_cost_of_debt": wacc.after_tax_cost_of_debt if wacc else None,
            "equity_weight": wacc.equity_weight if wacc else None,
            "debt_weight": wacc.debt_weight if wacc else None,
            "notes": list(dcf.notes),
        }

    lbo_block: dict[str, Any] | None = None
    if lbo is not None:
        lbo_block = {
            "entry_price_per_share": lbo.entry_price_per_share,
            "entry_multiple": lbo.entry_multiple,
            "exit_multiple": lbo.exit_multiple,
            "exit_multiple_source": report.lbo_exit_multiple_source,
            "leverage_x_ebitda": lbo.leverage,
            "equity_share": lbo.equity_share,
            "irr": lbo.irr,
            "moic": lbo.moic,
            "risk_free_rate": lbo.assumptions.risk_free_rate,
            "ability_to_pay": {f"{k:.2f}": v for k, v in lbo.ability_to_pay.items()},
            "ability_to_pay_status": {f"{k:.2f}": v for k, v in lbo.ability_to_pay_status.items()},
            "warnings": list(lbo.warnings),
        }

    mc_block: dict[str, Any] | None = None
    if mc is not None:
        mc_block = {
            "n_simulations": mc.n_simulations,
            "seed": mc.config.seed,
            "base_value": mc.base_value,
            "mean": mc.mean,
            "std": mc.std,
            "standard_error": mc.standard_error,
            "percentiles": {f"p{level}": value for level, value in mc.percentiles.items()},
            "prob_above_price": mc.prob_above_price,
            "n_non_finite": mc.n_non_finite,
            "notes": list(mc.notes),
        }

    summary = {
        "ticker": company.ticker,
        "name": profile.name,
        "sector": profile.sector,
        "industry": profile.industry,
        "currency": market.currency,
        "reporting_currency": profile.reporting_currency,
        "fx_rate": profile.fx_rate,
        "as_of": str(market.as_of),
        "latest_fiscal_year": company.statements.latest_year,
        "price": price,
        "market_cap": market.market_cap,
        "enterprise_value": market.enterprise_value,
        "net_debt": market.net_debt,
        "status": report.status,
        "valuations": [_valuation_row(row, price) for row in report.valuations],
        "not_applicable": dict(report.not_applicable),
        "errors": dict(report.errors),
        "warnings": list(report.warnings),
        "notes": list(report.notes),
        "dcf": dcf_block,
        "comps": _comps_block(report.comps),
        "lbo": lbo_block,
        "monte_carlo": mc_block,
        "peers": [
            {"ticker": p.ticker, "name": p.name, "industry": p.industry} for p in report.peers
        ],
        "market_parameters": {
            name: asdict(source) for name, source in report.parameter_sources.items()
        },
        "settings": {
            "n_simulations": report.n_simulations,
            "seed": report.seed,
            "size_premium": report.size_premium,
            "dcf_peer_source": report.dcf_peer_source,
        },
        "timing": {"generated_at": report.generated_at, "seconds": dict(report.timings)},
    }
    return _clean(summary)


def save_report(
    report: ValuationReport,
    output_dir: str | Path,
    *,
    charts: bool = True,
    excel: bool = True,
) -> dict[str, list[Path]]:
    """Write the deliverables of one report into ``output_dir``.

    * ``charts/<name>.png`` and ``.svg`` for every applicable chart,
    * ``<TICKER>_valuation.xlsx`` (the football field is embedded),
    * ``summary.json`` (:func:`report_summary` plus the list of files written).

    Returns the written paths under the keys ``charts``, ``excel`` and
    ``summary``. A failing chart set or workbook is recorded in
    ``report.warnings`` and does not prevent the other files.
    """
    # Imported here so that valuing a company never needs the chart stack.
    from valuation_engine.export import export_to_excel
    from valuation_engine.viz import save_charts

    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    written: dict[str, list[Path]] = {"charts": [], "excel": [], "summary": []}

    football_field: Path | None = None
    if charts:
        try:
            saved = save_charts(report, folder / "charts")
            written["charts"] = [path for paths in saved.values() for path in paths]
            football_field = next(
                (p for p in saved.get("football_field", []) if p.suffix == ".png"), None
            )
        except Exception as exc:  # noqa: BLE001 - charts must not block the numbers
            report.warnings.append(f"Charts could not be saved ({type(exc).__name__}: {exc}).")
    if excel:
        try:
            workbook = export_to_excel(
                report,
                folder / f"{report.ticker}_valuation.xlsx",
                football_field=football_field,
            )
            written["excel"] = [workbook]
        except Exception as exc:  # noqa: BLE001 - the workbook must not block the summary
            report.warnings.append(
                f"The Excel workbook could not be written ({type(exc).__name__}: {exc})."
            )

    summary = report_summary(report)
    summary["files"] = sorted(
        path.relative_to(folder).as_posix() for paths in written.values() for path in paths
    )
    target = folder / "summary.json"
    target.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    written["summary"] = [target]
    return written


def _num(value: Any, decimals: int = 2) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:,.{decimals}f}" if math.isfinite(number) else "n/a"


def _pct(value: Any, decimals: int = 1, *, sign: bool = False) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value) * 100.0
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return f"{number:+,.{decimals}f}%" if sign else f"{number:,.{decimals}f}%"


def _money(value: float, currency: str) -> str:
    if not math.isfinite(value):
        return "n/a"
    for divisor, unit in ((1e9, "bn"), (1e6, "m")):
        if abs(value) >= divisor:
            return f"{currency} {value / divisor:,.1f} {unit}"
    return f"{currency} {value:,.0f}"


def summary_text(report: ValuationReport, *, width: int = 100) -> str:
    """Compact plain-text summary for the terminal (ASCII only)."""
    company = report.company
    profile, market = company.profile, company.market
    currency, price = market.currency, float(market.price)
    rule = "=" * width
    thin = "-" * width
    lines = [rule, f"{profile.name} ({company.ticker})"]
    lines.append(
        ", ".join(
            str(part)
            for part in (profile.sector, profile.industry, f"market data as of {market.as_of}")
            if part
        )
    )
    lines.append(
        f"Price {currency} {_num(price)} | market cap {_money(market.market_cap, currency)} | "
        f"EV {_money(market.enterprise_value, currency)} | "
        f"net debt {_money(market.net_debt, currency)} | FY{company.statements.latest_year}"
    )
    fx_text = data_layer.describe_statement_fx(company)
    if fx_text:
        lines.extend(textwrap.wrap(fx_text, width=width))
    lines.append(thin)

    header = f"{'Method':<30}{'Low':>12}{'Base':>12}{'High':>12}{'Base/price':>12}{'vs. price':>12}"
    lines.append(f"Value per share ({currency})")
    lines.append(header)
    if report.valuations:
        for row in report.valuations:
            ratio = _ratio(row.base, price)
            lines.append(
                f"{row.label:<30}{_num(row.low):>12}{_num(row.base):>12}{_num(row.high):>12}"
                f"{_num(ratio) + 'x' if ratio is not None else 'n/a':>12}"
                f"{_pct(ratio - 1.0, sign=True) if ratio is not None else 'n/a':>12}"
            )
    else:
        lines.append("(no method could be applied)")
    lines.append(f"{'Current price':<30}{'':>12}{_num(price):>12}")

    if report.not_applicable:
        lines.append(thin)
        lines.append("Not applicable")
        for key in METHOD_KEYS:
            if key in report.not_applicable:
                lines.append(f"  {METHOD_NAMES[key]}: {report.not_applicable[key]}")

    dcf, lbo, mc = report.dcf, report.lbo, report.monte_carlo
    if dcf is not None or lbo is not None or mc is not None or report.comps is not None:
        lines.append(thin)
    if dcf is not None:
        wacc = dcf.wacc_result
        if wacc is not None:
            lines.append(
                f"WACC {_pct(dcf.wacc, 2)}: cost of equity {_pct(wacc.cost_of_equity, 2)} "
                f"(risk-free {_pct(wacc.risk_free_rate, 2)} + beta {_num(wacc.beta)} x ERP "
                f"{_pct(wacc.equity_risk_premium, 2)} + size premium "
                f"{_pct(wacc.size_premium, 2)}), after-tax cost of debt "
                f"{_pct(wacc.after_tax_cost_of_debt, 2)}, weights E/D "
                f"{_pct(wacc.equity_weight, 0)}/{_pct(wacc.debt_weight, 0)}"
            )
            lines.append(f"Beta {_num(wacc.beta)}: {wacc.beta_method}")
            if wacc.peer_group:
                lines.append(
                    f"Peer beta {_num(wacc.peer_beta)} from {wacc.peer_count} peers "
                    f"({wacc.peer_group}); own Blume-adjusted beta {_num(wacc.blume_beta)}"
                )
        else:
            lines.append(f"WACC {_pct(dcf.wacc, 2)}: given as an assumption (CAPM not run)")
        lines.append(
            f"Terminal growth {_pct(dcf.assumptions.terminal_growth, 2)}, exit multiple "
            f"{_num(dcf.exit_multiple, 1)}x: {dcf.exit_multiple_source}"
        )
    if report.comps is not None:
        lines.append(
            f"Comps peers ({len(report.comps.peers)}): {', '.join(report.comps.peer_tickers)}"
        )
    if lbo is not None:
        lines.append(
            f"LBO at {currency} {_num(lbo.entry_price_per_share)} per share "
            f"({_num(lbo.entry_multiple, 1)}x EBITDA): IRR {_pct(lbo.irr)}, MOIC "
            f"{_num(lbo.moic)}x, debt {_num(lbo.leverage, 1)}x EBITDA, exit at "
            f"{_num(lbo.exit_multiple, 1)}x"
        )
    if mc is not None:
        p = mc.percentiles
        lines.append(
            f"Monte Carlo ({mc.n_simulations:,} simulations, seed {mc.config.seed}): "
            f"P10 {_num(p.get(10))}, P50 {_num(p.get(50))}, P90 {_num(p.get(90))}, "
            f"P(value > price) {_pct(mc.prob_above_price)}"
        )

    if report.parameter_sources:
        lines.append(thin)
        lines.append("Market parameters")
        for name, source in report.parameter_sources.items():
            value = source.value
            if isinstance(value, str):
                shown = value
            elif name == "country_risk_premium" or name in {
                "risk_free_rate",
                "equity_risk_premium",
                "tax_rate",
                "long_term_growth",
            }:
                shown = _pct(value, 3 if name == "risk_free_rate" else 2)
            else:
                shown = _num(value, 4)
            origin = (
                f"[{source.origin}, as of {source.as_of}]" if source.as_of else f"[{source.origin}]"
            )
            lines.append(f"  {name:<22}{shown:>9}  {origin}")
            lines.extend(
                f"  {'':<33}{part}"
                for part in textwrap.wrap(str(source.source), width=max(width - 35, 30))
            )

    for title, items in (("Warnings", report.warnings), ("Notes", report.notes)):
        if items:
            lines.append(thin)
            lines.append(title)
            lines.extend(f"  - {item}" for item in items)
    lines.append(rule)
    return "\n".join(line.replace("\u2013", "-").replace("\u2014", "-") for line in lines)
