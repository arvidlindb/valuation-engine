"""Discounted cash flow (FCFF) valuation.

Public API
----------
* :func:`default_assumptions` - history-driven :class:`DCFAssumptions`.
* :func:`default_assumptions_with_notes` - same, plus the list of fallbacks that fired.
* :func:`compute_wacc` - CAPM / peer-beta WACC with every component (:class:`WACCResult`).
* :func:`resolve_exit_multiple` - exit EV/EBITDA multiple and where it came from.
* :func:`dcf_core` - the pure, vectorized arithmetic (single source of truth; the
  Monte Carlo model calls it with arrays of scenarios).
* :func:`run_dcf` - full valuation of one company (:class:`DCFResult`).

Model in short
--------------
Forecast (per year ``t = 1..N``, driven by :class:`DCFAssumptions`)::

    revenue_t = revenue_{t-1} * (1 + growth_t)
    EBITDA_t  = revenue_t * margin_t
    D&A_t     = revenue_t * da_pct
    EBIT_t    = EBITDA_t - D&A_t
    NOPAT_t   = EBIT_t - tax_rate * max(EBIT_t, 0)        (no tax credit on losses)
    capex_t   = revenue_t * capex_pct
    dNWC_t    = nwc_pct * (revenue_t - revenue_{t-1})     (incremental operating NWC)
    FCFF_t    = NOPAT_t + D&A_t - capex_t - dNWC_t

Year 1 is the fiscal year after the latest reported one, and the valuation date is
taken to be the start of year 1 (no stub period, see "Known simplifications").

Terminal value and the mid-year convention
------------------------------------------
With ``mid_year=True`` the cash flow of year ``t`` is discounted ``t - 0.5`` years
(it arrives on average in the middle of the year); otherwise ``t`` years.

* **Gordon growth.** The terminal cash flow is a normalized year ``N+1`` flow
  (revenue grows with ``g``, margin of year ``N``, NWC investment consistent with
  ``g``, and capex floored at D&A so that the asset base is not run down forever).
  ``FCFF_{N+1} / (WACC - g)`` is the value at the END of year ``N`` of a perpetuity
  whose flows arrive at year-ENDS. Under the mid-year convention the perpetuity
  flows also arrive mid-year, i.e. half a year earlier, so the year-end-``N`` value
  is ``FCFF_{N+1} / (WACC - g) * (1 + WACC)**0.5``. It is then discounted ``N``
  full years. (Equivalent to discounting the unadjusted TV by ``N - 0.5`` years.)
* **Exit multiple.** ``multiple * EBITDA_N`` is a transaction value at the END of
  year ``N`` (a sale at the year-end balance sheet date). It is discounted ``N``
  full years and gets NO mid-year adjustment, with or without the convention.

Cross-checks reported: the exit multiple implied by the Gordon case
(``TV_gordon at year-end N / EBITDA_N``) and the perpetual growth implied by the
exit case (the ``g`` that makes the Gordon terminal value equal the exit value,
solved in closed form on the same normalized terminal flow).

Equity bridge
-------------
``equity = EV - net debt - minority interest - preferred equity`` and value per
share uses ``MarketData.shares_outstanding`` (never the statements' diluted share
count, which is unreliable for small caps).

Applicability
-------------
:func:`run_dcf` either returns a result or raises :class:`NotApplicableError` with a
reason that starts with a stable category (text before the first colon). Negative
latest EBITDA is treated as not applicable: the forecast extrapolates the recent
margin, and a perpetuity on a negative margin has no meaning.

Known simplifications
---------------------
* No stub period. Prices are current, the balance sheet is the latest quarter and
  year 1 is treated as entirely in the future. Discounting year 1 in full
  understates value, while cash already generated during year 1 and sitting in net
  debt is counted twice; the two effects work in opposite directions.
* IFRS 16: EBITDA excludes lease costs and ``total_debt`` includes lease
  liabilities, but reported capex excludes lease payments. Explicit-period FCFF of
  lease-heavy companies is therefore somewhat overstated; the terminal capex floor
  (capex >= D&A) removes the effect from the terminal value.
* The cost of equity includes a size premium (``SIZE_PREMIUM_SCHEDULE``), which is a
  practitioner convention and an assumption of this module, not part of CAPM. Set
  the schedule to ``()`` or pass ``size_premium=0.0`` to :func:`compute_wacc` to
  get plain CAPM.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from valuation_engine.core.types import (
    CompanyData,
    DCFAssumptions,
    MarketParameters,
    NotApplicableError,
    PeerSnapshot,
    PriceHistory,
    ValuationMethod,
    ValuationResult,
)

__all__ = [
    "DCFCoreOutput",
    "DCFResult",
    "WACCResult",
    "check_applicability",
    "compute_wacc",
    "dcf_core",
    "default_assumptions",
    "default_assumptions_with_notes",
    "regression_beta_stats",
    "resolve_exit_multiple",
    "run_dcf",
]

# --------------------------------------------------------------------------
# Documented default choices
# --------------------------------------------------------------------------

_DEFAULT_PARAMS = MarketParameters()

#: Number of explicit forecast years.
FORECAST_YEARS = 5
#: Number of most recent fiscal years averaged for margins and ratios.
HISTORY_WINDOW = 3
#: Band for the starting revenue growth (historical CAGR is clipped to it).
GROWTH_BAND = (-0.05, 0.15)
#: Upper bound for the forecast EBITDA margin.
MAX_EBITDA_MARGIN = 0.70
#: Upper bound for D&A and capex as a share of revenue.
MAX_DA_CAPEX_PCT = 0.50
#: Band for operating NWC as a share of revenue.
NWC_BAND = (-0.30, 0.60)

#: Blume adjustment: adjusted = 2/3 * raw + 1/3.
BLUME_WEIGHT = 2.0 / 3.0
#: Minimum number of weekly RETURN observations for the own-beta regression (one
#: year). The data layer (``data.loader.MIN_BETA_RETURNS``) and the comps fallback
#: beta use the same rule.
MIN_BETA_OBSERVATIONS = 52
#: Raw betas (own and peers) outside this band are treated as noise and dropped.
RAW_BETA_BAND = (0.0, 3.0)
#: The beta finally used in CAPM is clipped to this band.
FINAL_BETA_BAND = (0.5, 2.0)
#: D/E used for (un)levering and for the WACC weights is clipped to this band.
DEBT_TO_EQUITY_BAND = (0.0, 2.0)
#: Minimum number of usable peers for a peer statistic (industry, then sector).
MIN_PEERS = 5
#: Label of a ready-made peer group handed in with ``peers_are_peer_group=True``.
PEER_GROUP_LABEL = "comps peer group"
#: Pre-tax cost of debt band and fallback, expressed as spreads over the risk-free rate.
COST_OF_DEBT_SPREAD_BAND = (0.005, 0.06)
COST_OF_DEBT_FALLBACK_SPREAD = 0.02
#: Size premium added to the CAPM cost of equity: (market cap floor in SEK, premium),
#: first matching row wins; below the last floor ``SIZE_PREMIUM_SMALLEST`` applies.
#: An assumption in the spirit of practitioner surveys, not an estimate from data.
#: Regression betas of thinly traded small caps are biased downward, so plain CAPM
#: gives micro caps a lower cost of equity than large caps, which is not credible.
SIZE_PREMIUM_SCHEDULE: tuple[tuple[float, float], ...] = (
    (20e9, 0.0),
    (5e9, 0.005),
    (1e9, 0.010),
    (200e6, 0.020),
)
SIZE_PREMIUM_SMALLEST = 0.030

#: Band the default exit EV/EBITDA multiple is clipped to (user input is not clipped).
EXIT_MULTIPLE_BAND = (4.0, 20.0)
#: Peer EV/EBITDA multiples above this are ignored when taking the median.
MAX_PEER_MULTIPLE = 50.0

#: Minimum WACC - g spread; below it the Gordon value explodes -> not applicable.
MIN_WACC_GROWTH_SPREAD = 0.005

#: Sensitivity grids (offsets from the base case) and the cells used for low/high.
WACC_STEPS = (-0.010, -0.005, 0.0, 0.005, 0.010)
GROWTH_STEPS = (-0.010, -0.005, 0.0, 0.005, 0.010)
MULTIPLE_STEPS = (-2.0, -1.0, 0.0, 1.0, 2.0)
RANGE_WACC_STEP = 0.010
RANGE_GROWTH_STEP = 0.005
RANGE_MULTIPLE_STEP = 1.0


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _num(value: Any) -> float | None:
    """Return ``value`` as a finite float, or None."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _clip(value: float, band: tuple[float, float]) -> float:
    return min(max(value, band[0]), band[1])


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    """Numeric column of a statement; all-NaN if the column is absent."""
    if name in frame.columns:
        return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _recent_ratio(numerator: pd.Series, denominator: pd.Series) -> tuple[float | None, int]:
    """Mean of ``numerator / denominator`` over the last ``HISTORY_WINDOW`` fiscal
    years, using only years where both are finite and the denominator is positive.
    Returns (mean or None, number of years used)."""
    num = numerator.iloc[-HISTORY_WINDOW:]
    den = denominator.reindex(num.index)
    mask = np.isfinite(num.to_numpy(float)) & np.isfinite(den.to_numpy(float))
    mask &= den.to_numpy(float) > 0
    if not mask.any():
        return None, 0
    ratios = num.to_numpy(float)[mask] / den.to_numpy(float)[mask]
    return float(np.mean(ratios)), int(mask.sum())


def _reason_category(reason: str) -> str:
    return reason.split(":", 1)[0]


# --------------------------------------------------------------------------
# Applicability
# --------------------------------------------------------------------------


def check_applicability(company: CompanyData) -> None:
    """Raise :class:`NotApplicableError` if an FCFF DCF cannot be built.

    Categories (text before the colon): ``financial sector``, ``insufficient
    history``, ``missing revenue``, ``missing EBITDA``, ``negative EBITDA``,
    ``invalid market data``.
    """
    profile = company.profile
    if profile.is_financial:
        raise NotApplicableError(
            f"financial sector: FCFF-based DCF is not meaningful for {profile.sector} companies"
        )
    income = company.statements.income
    if len(income.index) < 2:
        raise NotApplicableError(
            f"insufficient history: {len(income.index)} fiscal year(s), at least 2 required"
        )
    revenue = _num(_column(income, "revenue").iloc[-1])
    if revenue is None or revenue <= 0:
        raise NotApplicableError("missing revenue: latest revenue is missing or non-positive")
    ebitda = _num(_column(income, "ebitda").iloc[-1])
    if ebitda is None:
        raise NotApplicableError("missing EBITDA: latest EBITDA is not available")
    if ebitda <= 0:
        raise NotApplicableError(
            "negative EBITDA: latest EBITDA is non-positive; a DCF extrapolating a "
            "negative margin is not meaningful"
        )
    market = company.market
    price = _num(market.price)
    shares = _num(market.shares_outstanding)
    market_cap = _num(market.market_cap)
    if price is None or shares is None or market_cap is None or min(price, shares, market_cap) <= 0:
        raise NotApplicableError(
            "invalid market data: price, shares outstanding or market cap missing or non-positive"
        )


# --------------------------------------------------------------------------
# Default assumptions
# --------------------------------------------------------------------------


def default_assumptions_with_notes(
    company: CompanyData, params: MarketParameters = _DEFAULT_PARAMS
) -> tuple[DCFAssumptions, list[str]]:
    """Build history-driven default assumptions and list the fallbacks that fired.

    Choices (constants at the top of the module):

    * ``FORECAST_YEARS`` (5) explicit years.
    * Revenue growth: starts at the historical revenue CAGR over the available
      window (at most the last 4 fiscal years, i.e. 3 intervals), clipped to
      ``GROWTH_BAND`` (-5 % .. +15 %), and fades linearly so that year 5 grows at
      ``params.long_term_growth``. Fallbacks: latest year-over-year growth, then
      the long-term growth.
    * EBITDA margin: mean of the last ``HISTORY_WINDOW`` (3) years, held flat (no
      fade: there is no company-specific evidence for a direction), capped at
      ``MAX_EBITDA_MARGIN``. If that mean is non-positive although the latest
      year is profitable, the latest margin is used.
    * D&A % of revenue: 3-year mean from the income statement; fallbacks: cash
      flow statement D&A, then EBITDA - EBIT, then 0.
    * Capex % of revenue: 3-year mean over the years where capex is reported;
      fallback: equal to D&A (maintenance-capex assumption).
    * Operating NWC % of revenue: 3-year mean of balance sheet
      ``net_working_capital / revenue`` (receivables + inventory - payables), NOT
      Yahoo's broader ``change_in_nwc``; fallback 0 (no NWC investment).
    * Tax rate and terminal growth from ``params``; ``wacc`` and ``exit_multiple``
      left as None (resolved in :func:`run_dcf`); mid-year convention on.

    Raises :class:`NotApplicableError` via :func:`check_applicability`.
    """
    check_applicability(company)
    notes: list[str] = []
    income = company.statements.income
    balance = company.statements.balance
    cash_flow = company.statements.cash_flow
    revenue = _column(income, "revenue")
    long_term = float(params.long_term_growth)

    # --- revenue growth ----------------------------------------------------
    window = revenue.iloc[-(HISTORY_WINDOW + 1) :]
    valid = window[np.isfinite(window.to_numpy(float)) & (window.to_numpy(float) > 0)]
    start_growth: float | None = None
    if len(valid) >= 2:
        span = int(valid.index[-1]) - int(valid.index[0])
        if span > 0:
            start_growth = float((valid.iloc[-1] / valid.iloc[0]) ** (1.0 / span) - 1.0)
            if len(valid) < len(window) or span < HISTORY_WINDOW:
                notes.append(
                    f"revenue growth: CAGR based on {span} year(s) "
                    f"({int(valid.index[0])}-{int(valid.index[-1])})"
                )
    if start_growth is None or not math.isfinite(start_growth):
        start_growth = long_term
        notes.append("revenue growth: no usable history, long-term growth used from year 1")
    clipped = _clip(start_growth, GROWTH_BAND)
    if clipped != start_growth:
        notes.append(
            f"revenue growth: historical CAGR {start_growth:.1%} clipped to {clipped:.1%} "
            f"(band {GROWTH_BAND[0]:.0%}..{GROWTH_BAND[1]:.0%})"
        )
    n = FORECAST_YEARS
    growth = tuple(float(clipped + (long_term - clipped) * t / (n - 1)) for t in range(n))

    # --- EBITDA margin -----------------------------------------------------
    ebitda = _column(income, "ebitda")
    margin, used = _recent_ratio(ebitda, revenue)
    latest_margin = float(ebitda.iloc[-1] / revenue.iloc[-1])  # both checked above
    if margin is None or margin <= 0:
        margin = latest_margin
        notes.append(
            "EBITDA margin: recent average is non-positive or missing, latest year's margin used"
        )
    elif used < HISTORY_WINDOW:
        notes.append(f"EBITDA margin: average of {used} year(s) only")
    if margin > MAX_EBITDA_MARGIN:
        notes.append(f"EBITDA margin: {margin:.1%} capped at {MAX_EBITDA_MARGIN:.0%}")
        margin = MAX_EBITDA_MARGIN

    # --- D&A ---------------------------------------------------------------
    da_pct, _ = _recent_ratio(_column(income, "depreciation_amortization"), revenue)
    if da_pct is None:
        da_pct, _ = _recent_ratio(_column(cash_flow, "depreciation_amortization"), revenue)
        if da_pct is not None:
            notes.append("D&A: income statement D&A missing, cash flow statement D&A used")
    if da_pct is None:
        da_pct, _ = _recent_ratio(ebitda - _column(income, "ebit"), revenue)
        if da_pct is not None:
            notes.append("D&A: missing, derived as EBITDA - EBIT")
    if da_pct is None:
        da_pct = 0.0
        notes.append("D&A: not available, set to 0 % of revenue")
    if not 0.0 <= da_pct <= MAX_DA_CAPEX_PCT:
        notes.append(f"D&A: {da_pct:.1%} of revenue clipped to 0..{MAX_DA_CAPEX_PCT:.0%}")
        da_pct = _clip(da_pct, (0.0, MAX_DA_CAPEX_PCT))

    # --- capex -------------------------------------------------------------
    capex_pct, capex_years = _recent_ratio(_column(cash_flow, "capex"), revenue)
    if capex_pct is None:
        capex_pct = da_pct
        notes.append("capex: not reported, set equal to D&A")
    else:
        if capex_years < min(HISTORY_WINDOW, len(revenue)):
            notes.append(f"capex: average of {capex_years} reported year(s) only")
        if not 0.0 <= capex_pct <= MAX_DA_CAPEX_PCT:
            notes.append(f"capex: {capex_pct:.1%} of revenue clipped to 0..{MAX_DA_CAPEX_PCT:.0%}")
            capex_pct = _clip(capex_pct, (0.0, MAX_DA_CAPEX_PCT))

    # --- operating NWC -----------------------------------------------------
    nwc_pct, _ = _recent_ratio(_column(balance, "net_working_capital"), revenue)
    if nwc_pct is None:
        nwc_pct = 0.0
        notes.append("NWC: balance sheet operating NWC not available, set to 0 % of revenue")
    elif not NWC_BAND[0] <= nwc_pct <= NWC_BAND[1]:
        notes.append(
            f"NWC: {nwc_pct:.1%} of revenue clipped to {NWC_BAND[0]:.0%}..{NWC_BAND[1]:.0%}"
        )
        nwc_pct = _clip(nwc_pct, NWC_BAND)

    assumptions = DCFAssumptions(
        revenue_growth=growth,
        ebitda_margin=tuple(float(margin) for _ in range(n)),
        da_pct_revenue=float(da_pct),
        capex_pct_revenue=float(capex_pct),
        nwc_pct_revenue=float(nwc_pct),
        tax_rate=float(params.tax_rate),
        wacc=None,
        terminal_growth=long_term,
        exit_multiple=None,
        mid_year=True,
    )
    return assumptions, notes


def default_assumptions(
    company: CompanyData, params: MarketParameters = _DEFAULT_PARAMS
) -> DCFAssumptions:
    """History-driven default assumptions.

    See :func:`default_assumptions_with_notes` for the choices and fallbacks.
    """
    return default_assumptions_with_notes(company, params)[0]


# --------------------------------------------------------------------------
# WACC
# --------------------------------------------------------------------------


@dataclass
class WACCResult:
    """Every component of the WACC, ready to be shown in Excel or the app."""

    risk_free_rate: float
    equity_risk_premium: float
    tax_rate: float
    # own regression beta
    raw_beta: float | None  # OLS slope, weekly simple returns vs. benchmark
    raw_beta_r_squared: float | None
    beta_observations: int
    benchmark: str | None
    blume_beta: float | None  # 2/3 * raw + 1/3; None if the raw beta is unusable
    # peer beta
    peer_group: str | None  # e.g. "industry: Specialty Industrial Machinery"
    peer_count: int
    peer_unlevered_beta: float | None  # median of Hamada-unlevered raw peer betas
    peer_relevered_beta: float | None  # relevered at the target's D/E (raw)
    peer_beta: float | None  # relevered and Blume-adjusted
    # selection
    beta: float  # beta used in CAPM (after clipping to FINAL_BETA_BAND)
    beta_method: str
    size_premium: float
    size_premium_basis: str
    cost_of_equity: float  # rf + beta * ERP + size premium
    # debt
    pre_tax_cost_of_debt: float
    cost_of_debt_source: str
    after_tax_cost_of_debt: float
    # capital structure (market values)
    market_cap: float
    total_debt: float
    debt_to_equity: float  # as used (clipped to DEBT_TO_EQUITY_BAND)
    equity_weight: float
    debt_weight: float
    wacc: float
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Flat mapping of all components (for tables and export)."""
        return dict(self.__dict__)


def regression_beta_stats(
    history: PriceHistory | None,
) -> tuple[float | None, float | None, int]:
    """OLS beta of the stock's simple returns on the benchmark's.

    Uses the series in ``history.prices`` at their native (weekly) frequency.
    Returns ``(raw beta, R squared, observations)``; beta is None when there are
    fewer than ``MIN_BETA_OBSERVATIONS`` returns or the benchmark has no variance.
    """
    if history is None:
        return None, None, 0
    prices = history.prices
    if not {"stock", "benchmark"}.issubset(prices.columns):
        return None, None, 0
    returns = prices[["stock", "benchmark"]].astype(float).pct_change()
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    n_obs = len(returns)
    if n_obs < MIN_BETA_OBSERVATIONS:
        return None, None, n_obs
    stock = returns["stock"].to_numpy(float)
    bench = returns["benchmark"].to_numpy(float)
    variance = float(np.var(bench, ddof=1))
    stock_variance = float(np.var(stock, ddof=1))
    if not variance > 0:
        return None, None, n_obs
    covariance = float(np.cov(stock, bench, ddof=1)[0, 1])
    beta = covariance / variance
    if not math.isfinite(beta):
        return None, None, n_obs
    r_squared = covariance**2 / (variance * stock_variance) if stock_variance > 0 else None
    return beta, r_squared, n_obs


def _same_group_peers(
    company: CompanyData,
    peers: list[PeerSnapshot] | None,
    peers_are_peer_group: bool = False,
) -> list[tuple[str, list[PeerSnapshot]]]:
    """Candidate peer groups in order of preference: same industry, same sector.
    Financial companies and the company itself are excluded.

    With ``peers_are_peer_group=True`` the given peers are one ready-made peer
    group (e.g. the peers selected by comps): they are used as they are, without
    any industry, sector or financial filter; only the company itself is excluded.
    """
    if not peers:
        return []
    profile = company.profile
    if peers_are_peer_group:
        return [(PEER_GROUP_LABEL, [p for p in peers if p.ticker != profile.ticker])]
    others = [p for p in peers if p.ticker != profile.ticker and not p.is_financial]
    groups: list[tuple[str, list[PeerSnapshot]]] = []
    if profile.industry:
        groups.append(
            (f"industry: {profile.industry}", [p for p in others if p.industry == profile.industry])
        )
    if profile.sector:
        groups.append(
            (f"sector: {profile.sector}", [p for p in others if p.sector == profile.sector])
        )
    return groups


def _peer_unlevered_beta(
    company: CompanyData,
    peers: list[PeerSnapshot] | None,
    tax_rate: float,
    peers_are_peer_group: bool = False,
) -> tuple[float | None, int, str | None]:
    """Median Hamada-unlevered beta of the first peer group with enough members.

    A peer is usable when its raw beta lies inside ``RAW_BETA_BAND`` (exclusive
    lower bound; negative and extreme betas are regression noise from illiquid
    shares) and its D/E is known. Peer D/E is clipped to ``DEBT_TO_EQUITY_BAND``.
    """
    for label, group in _same_group_peers(company, peers, peers_are_peer_group):
        unlevered: list[float] = []
        for peer in group:
            beta = _num(peer.beta)
            leverage = _num(peer.debt_to_equity)
            if beta is None or leverage is None:
                continue
            if not RAW_BETA_BAND[0] < beta <= RAW_BETA_BAND[1]:
                continue
            leverage = _clip(leverage, DEBT_TO_EQUITY_BAND)
            unlevered.append(beta / (1.0 + (1.0 - tax_rate) * leverage))
        if len(unlevered) >= MIN_PEERS:
            return float(np.median(unlevered)), len(unlevered), label
    return None, 0, None


def compute_wacc(
    company: CompanyData,
    params: MarketParameters = _DEFAULT_PARAMS,
    peers: list[PeerSnapshot] | None = None,
    tax_rate: float | None = None,
    size_premium: float | None = None,
    *,
    peers_are_peer_group: bool = False,
) -> WACCResult:
    """Weighted average cost of capital with all components.

    Cost of equity: CAPM plus a size premium, ``rf + beta * ERP + size premium``.

    Size premium
        ``size_premium=None`` -> looked up in ``SIZE_PREMIUM_SCHEDULE`` by market
        cap in SEK (taken from the company's own row in ``peers`` when present,
        because peer snapshots are always in SEK; otherwise the market cap in the
        trading currency). Pass a number (e.g. 0.0) to override.

    Beta
        * Own beta: OLS on weekly returns vs. the benchmark, Blume-adjusted
          (``2/3 * raw + 1/3``). Discarded if the raw beta is outside
          ``RAW_BETA_BAND`` or there are too few observations.
        * Peer beta: raw peer betas (same industry, else same sector; at least
          ``MIN_PEERS`` usable peers) are unlevered with Hamada
          ``beta_u = beta / (1 + (1 - t) * D/E)``, the median is relevered at the
          target's market D/E and then Blume-adjusted (the peer betas in the data
          layer are raw, with a universe median well below 1 because of thin
          trading, so the same shrinkage toward 1 is applied).
          ``peers_are_peer_group=True`` uses the given peers as one ready-made
          group instead of filtering them by industry and sector (the usability
          filters and ``MIN_PEERS`` still apply).
        * Default: the average of the two when both exist, otherwise the one that
          exists, otherwise 1.0. The result is clipped to ``FINAL_BETA_BAND``.

    Cost of debt
        Latest interest expense / average of the last two fiscal year-end debt
        balances, clipped to risk-free + ``COST_OF_DEBT_SPREAD_BAND``; fallback
        risk-free + ``COST_OF_DEBT_FALLBACK_SPREAD``. After tax at ``tax_rate``.

    Weights
        Market cap and ``MarketData.total_debt`` (book debt as proxy for market
        value). D/E is clipped to ``DEBT_TO_EQUITY_BAND`` so that distressed
        companies (tiny market cap against large debt) do not get a WACC that is
        almost entirely cheap debt.

    Raises :class:`NotApplicableError` when the market cap is missing or non-positive.
    """
    notes: list[str] = []
    tax = float(params.tax_rate if tax_rate is None else tax_rate)
    rf = float(params.risk_free_rate)
    erp = float(params.equity_risk_premium)

    market_cap = _num(company.market.market_cap)
    if market_cap is None or market_cap <= 0:
        raise NotApplicableError("invalid market data: market cap missing or non-positive")
    total_debt = max(_num(company.market.total_debt) or 0.0, 0.0)
    raw_leverage = total_debt / market_cap
    leverage = _clip(raw_leverage, DEBT_TO_EQUITY_BAND)
    if leverage != raw_leverage:
        notes.append(
            f"capital structure: market D/E {raw_leverage:.2f} clipped to {leverage:.2f} "
            "for relevering and weights"
        )

    # --- own beta ------------------------------------------------------------
    raw_beta, r_squared, n_obs = regression_beta_stats(company.price_history)
    blume_beta: float | None = None
    if raw_beta is None:
        notes.append("beta: no usable price history for an own regression beta")
    elif not RAW_BETA_BAND[0] < raw_beta <= RAW_BETA_BAND[1]:
        notes.append(
            f"beta: own raw beta {raw_beta:.2f} outside {RAW_BETA_BAND} treated as noise and "
            "not used"
        )
    else:
        blume_beta = BLUME_WEIGHT * raw_beta + (1.0 - BLUME_WEIGHT)

    # --- peer beta -----------------------------------------------------------
    unlevered, peer_count, peer_group = _peer_unlevered_beta(
        company, peers, tax, peers_are_peer_group
    )
    peer_relevered: float | None = None
    peer_beta: float | None = None
    if unlevered is not None:
        peer_relevered = unlevered * (1.0 + (1.0 - tax) * leverage)
        peer_beta = BLUME_WEIGHT * peer_relevered + (1.0 - BLUME_WEIGHT)
    elif peers:
        scope = "the given peer group" if peers_are_peer_group else "industry and sector"
        notes.append(f"beta: fewer than {MIN_PEERS} usable peers in {scope}")

    # --- selection -----------------------------------------------------------
    if blume_beta is not None and peer_beta is not None:
        beta = 0.5 * (blume_beta + peer_beta)
        method = "average of own Blume-adjusted beta and relevered peer beta"
    elif blume_beta is not None:
        beta = blume_beta
        method = "own Blume-adjusted regression beta"
    elif peer_beta is not None:
        beta = peer_beta
        method = "relevered peer beta (Blume-adjusted)"
    else:
        beta = 1.0
        method = "market beta 1.0 (no own or peer beta available)"
    clipped_beta = _clip(beta, FINAL_BETA_BAND)
    if clipped_beta != beta:
        notes.append(f"beta: {beta:.2f} clipped to {clipped_beta:.2f} (band {FINAL_BETA_BAND})")
    if size_premium is None:
        own_snapshot = next((p for p in peers or [] if p.ticker == company.ticker), None)
        cap_sek = _num(own_snapshot.market_cap) if own_snapshot is not None else None
        if cap_sek is None or cap_sek <= 0:
            cap_sek = market_cap
            if company.market.currency != "SEK":
                notes.append(
                    f"size premium: market cap in {company.market.currency} compared with "
                    "SEK thresholds (no SEK peer snapshot for the company)"
                )
        premium = next(
            (prem for floor, prem in SIZE_PREMIUM_SCHEDULE if cap_sek >= floor),
            SIZE_PREMIUM_SMALLEST if SIZE_PREMIUM_SCHEDULE else 0.0,
        )
        premium_basis = f"schedule, market cap {cap_sek / 1e9:.2f} bn SEK"
    else:
        premium = float(size_premium)
        premium_basis = "given"
    cost_of_equity = rf + clipped_beta * erp + premium

    # --- cost of debt --------------------------------------------------------
    interest = _num(_column(company.statements.income, "interest_expense").iloc[-1])
    debt_history = _column(company.statements.balance, "total_debt").iloc[-2:].to_numpy(float)
    debt_history = debt_history[np.isfinite(debt_history) & (debt_history > 0)]
    band = (rf + COST_OF_DEBT_SPREAD_BAND[0], rf + COST_OF_DEBT_SPREAD_BAND[1])
    if interest is not None and interest > 0 and debt_history.size > 0:
        implied = interest / float(np.mean(debt_history))
        pre_tax_kd = _clip(implied, band)
        source = "interest expense / average total debt"
        if pre_tax_kd != implied:
            source += f" ({implied:.1%}, clipped to {band[0]:.1%}..{band[1]:.1%})"
    else:
        pre_tax_kd = rf + COST_OF_DEBT_FALLBACK_SPREAD
        source = f"fallback: risk-free + {COST_OF_DEBT_FALLBACK_SPREAD:.1%} spread"
    after_tax_kd = pre_tax_kd * (1.0 - tax)

    # --- weights -------------------------------------------------------------
    equity_weight = 1.0 / (1.0 + leverage)
    debt_weight = 1.0 - equity_weight
    wacc = equity_weight * cost_of_equity + debt_weight * after_tax_kd

    return WACCResult(
        risk_free_rate=rf,
        equity_risk_premium=erp,
        tax_rate=tax,
        raw_beta=raw_beta,
        raw_beta_r_squared=r_squared,
        beta_observations=n_obs,
        benchmark=company.price_history.benchmark if company.price_history else None,
        blume_beta=blume_beta,
        peer_group=peer_group,
        peer_count=peer_count,
        peer_unlevered_beta=unlevered,
        peer_relevered_beta=peer_relevered,
        peer_beta=peer_beta,
        beta=clipped_beta,
        beta_method=method,
        size_premium=premium,
        size_premium_basis=premium_basis,
        cost_of_equity=cost_of_equity,
        pre_tax_cost_of_debt=pre_tax_kd,
        cost_of_debt_source=source,
        after_tax_cost_of_debt=after_tax_kd,
        market_cap=market_cap,
        total_debt=total_debt,
        debt_to_equity=leverage,
        equity_weight=equity_weight,
        debt_weight=debt_weight,
        wacc=wacc,
        notes=notes,
    )


# --------------------------------------------------------------------------
# Exit multiple
# --------------------------------------------------------------------------


def resolve_exit_multiple(
    company: CompanyData,
    assumptions: DCFAssumptions,
    peers: list[PeerSnapshot] | None = None,
    *,
    peers_are_peer_group: bool = False,
    peer_multiple: tuple[float, str] | None = None,
) -> tuple[float, str]:
    """Return the exit EV/EBITDA multiple and a description of its source.

    Order: ``assumptions.exit_multiple`` (used as given) -> median EV/EBITDA of
    non-financial peers in the same industry, else the same sector (at least
    ``MIN_PEERS`` peers with a multiple in ``(0, MAX_PEER_MULTIPLE]``) -> the
    company's own current EV / latest EBITDA. Peer and own multiples are clipped
    to ``EXIT_MULTIPLE_BAND``: a terminal-year multiple describes a mature
    business, so neither distressed nor growth-stock multiples are carried over.
    Peer multiples are "current EV / latest fiscal year EBITDA", consistent with
    applying the multiple to year-N EBITDA. With ``peers_are_peer_group=True``
    the given peers are used as one ready-made group (no industry or sector
    filter; the multiple bounds and ``MIN_PEERS`` still apply).

    ``peer_multiple=(multiple, source label)`` supplies a ready-made peer multiple
    (e.g. the comps EV/EBITDA median, so that comps and the DCF use one number).
    It takes the place of the peer median, i.e. it ranks after an explicit
    ``assumptions.exit_multiple`` and is clipped to ``EXIT_MULTIPLE_BAND``; a
    non-positive or non-finite value is ignored.
    """
    given = _num(assumptions.exit_multiple)
    if assumptions.exit_multiple is not None:
        if given is None or given <= 0:
            raise NotApplicableError("invalid assumptions: exit multiple must be positive")
        return given, "assumption"

    if peer_multiple is not None:
        supplied = _num(peer_multiple[0])
        if supplied is not None and supplied > 0:
            clipped = _clip(supplied, EXIT_MULTIPLE_BAND)
            source = str(peer_multiple[1])
            if clipped != supplied:
                source += f", {supplied:.1f}x clipped to {clipped:.1f}x"
            return clipped, source

    for label, group in _same_group_peers(company, peers, peers_are_peer_group):
        multiples = [
            m for p in group if (m := p.ev_ebitda) is not None and 0 < m <= MAX_PEER_MULTIPLE
        ]
        if len(multiples) >= MIN_PEERS:
            median = float(np.median(multiples))
            clipped = _clip(median, EXIT_MULTIPLE_BAND)
            source = f"peer median EV/EBITDA, {label} ({len(multiples)} peers)"
            if clipped != median:
                source += f", {median:.1f}x clipped to {clipped:.1f}x"
            return clipped, source

    ebitda = _num(_column(company.statements.income, "ebitda").iloc[-1])
    enterprise_value = _num(company.market.enterprise_value)
    if ebitda is not None and ebitda > 0 and enterprise_value is not None and enterprise_value > 0:
        own = enterprise_value / ebitda
        clipped = _clip(own, EXIT_MULTIPLE_BAND)
        source = "own current EV / latest EBITDA"
        if clipped != own:
            source += f", {own:.1f}x clipped to {clipped:.1f}x"
        return clipped, source
    midpoint = 0.5 * (EXIT_MULTIPLE_BAND[0] + EXIT_MULTIPLE_BAND[1])
    return midpoint, "fallback: midpoint of the allowed band (no peer or own multiple)"


# --------------------------------------------------------------------------
# Vectorized core
# --------------------------------------------------------------------------


@dataclass
class DCFCoreOutput:
    """Arrays produced by :func:`dcf_core`.

    Per-year arrays have shape ``(n_sims, n_years)``, all others ``(n_sims,)``.
    Gordon quantities are NaN where ``wacc <= terminal_growth``; exit quantities
    are NaN where the exit multiple is NaN.
    """

    revenue: np.ndarray
    ebitda: np.ndarray
    depreciation_amortization: np.ndarray
    ebit: np.ndarray
    taxes: np.ndarray
    nopat: np.ndarray
    capex: np.ndarray
    delta_nwc: np.ndarray
    fcff: np.ndarray
    discount_periods: np.ndarray  # shape (n_years,)
    discount_factors: np.ndarray
    pv_fcff: np.ndarray  # sum of discounted explicit-period FCFF
    terminal_fcff: np.ndarray  # normalized FCFF of year N+1
    tv_gordon: np.ndarray  # value at END of year N (incl. mid-year factor)
    pv_tv_gordon: np.ndarray
    ev_gordon: np.ndarray
    equity_gordon: np.ndarray
    value_per_share_gordon: np.ndarray
    tv_exit: np.ndarray  # value at END of year N
    pv_tv_exit: np.ndarray
    ev_exit: np.ndarray
    equity_exit: np.ndarray
    value_per_share_exit: np.ndarray
    implied_exit_multiple: np.ndarray  # of the Gordon case
    implied_perpetual_growth: np.ndarray  # of the exit case


def dcf_core(
    *,
    base_revenue: float | np.ndarray,
    revenue_growth: np.ndarray | tuple[float, ...],
    ebitda_margin: np.ndarray | tuple[float, ...],
    da_pct_revenue: float | np.ndarray,
    capex_pct_revenue: float | np.ndarray,
    nwc_pct_revenue: float | np.ndarray,
    tax_rate: float | np.ndarray,
    wacc: float | np.ndarray,
    terminal_growth: float | np.ndarray,
    exit_multiple: float | np.ndarray,
    net_debt: float | np.ndarray,
    shares_outstanding: float | np.ndarray,
    minority_interest: float | np.ndarray = 0.0,
    preferred_equity: float | np.ndarray = 0.0,
    mid_year: bool = True,
    terminal_capex_floor: bool = True,
) -> DCFCoreOutput:
    """Pure DCF arithmetic, vectorized over scenarios.

    ``revenue_growth`` and ``ebitda_margin`` have shape ``(n_years,)`` or
    ``(n_sims, n_years)``; every other numeric argument is a scalar or an array of
    shape ``(n_sims,)``. Everything broadcasts, there is no Python loop over
    scenarios, and no input validation or exception: invalid combinations
    (``wacc <= terminal_growth``) yield NaN. :func:`run_dcf` calls this function
    with one scenario, so identical inputs give bit-identical results.

    ``terminal_capex_floor``: in the normalized terminal year capex is
    ``max(capex_pct, da_pct) * revenue``. In perpetuity capex below D&A would
    liquidate the asset base while revenue keeps growing.

    See the module docstring for the formulas and the mid-year treatment.
    """
    growth = np.atleast_2d(np.asarray(revenue_growth, dtype=float))
    margin = np.atleast_2d(np.asarray(ebitda_margin, dtype=float))

    def column(value: float | np.ndarray) -> np.ndarray:
        return np.asarray(value, dtype=float).reshape(-1, 1)

    base = column(base_revenue)
    da_pct = column(da_pct_revenue)
    capex_pct = column(capex_pct_revenue)
    nwc_pct = column(nwc_pct_revenue)
    tax = column(tax_rate)
    rate = column(wacc)
    g = column(terminal_growth)
    multiple = column(exit_multiple)
    debt = column(net_debt)
    minority = column(minority_interest)
    preferred = column(preferred_equity)
    shares = column(shares_outstanding)

    n_years = max(growth.shape[1], margin.shape[1])
    n_sims = np.broadcast_shapes(
        growth.shape[:1],
        margin.shape[:1],
        *(
            a.shape[:1]
            for a in (
                base,
                da_pct,
                capex_pct,
                nwc_pct,
                tax,
                rate,
                g,
                multiple,
                debt,
                minority,
                preferred,
                shares,
            )
        ),
    )[0]
    shape_2d = (n_sims, n_years)

    # --- explicit forecast ---------------------------------------------------
    revenue = np.broadcast_to(base * np.cumprod(1.0 + growth, axis=1), shape_2d)
    previous = np.concatenate([np.broadcast_to(base, (n_sims, 1)), revenue[:, :-1]], axis=1)
    ebitda = revenue * margin
    da = revenue * da_pct
    ebit = ebitda - da
    taxes = tax * np.maximum(ebit, 0.0)
    nopat = ebit - taxes
    capex = revenue * capex_pct
    delta_nwc = nwc_pct * (revenue - previous)
    fcff = nopat + da - capex - delta_nwc

    # --- discounting ---------------------------------------------------------
    periods = np.arange(1, n_years + 1, dtype=float) - (0.5 if mid_year else 0.0)
    factors = (1.0 + rate) ** (-periods)  # (n_sims or 1, n_years)
    pv_fcff = np.sum(fcff * factors, axis=1, keepdims=True)
    end_factor = (1.0 + rate) ** (-float(n_years))  # discount from END of year N

    # --- normalized terminal year (N+1) ---------------------------------------
    last_revenue = revenue[:, -1:]
    last_margin = np.broadcast_to(margin, shape_2d)[:, -1:]
    last_ebitda = last_revenue * last_margin
    terminal_capex_pct = np.maximum(capex_pct, da_pct) if terminal_capex_floor else capex_pct
    # Operating cash flow before NWC per unit of revenue at the terminal margin.
    unit_ebit = last_margin - da_pct
    unit_flow = unit_ebit - tax * np.maximum(unit_ebit, 0.0) + da_pct - terminal_capex_pct
    terminal_fcff = last_revenue * ((1.0 + g) * unit_flow - nwc_pct * g)

    # --- Gordon growth ---------------------------------------------------------
    mid_year_factor = (1.0 + rate) ** 0.5 if mid_year else np.ones_like(rate)
    spread = rate - g
    with np.errstate(divide="ignore", invalid="ignore"):
        tv_gordon = np.where(spread > 0, terminal_fcff / spread * mid_year_factor, np.nan)
    pv_tv_gordon = tv_gordon * end_factor
    ev_gordon = pv_fcff + pv_tv_gordon

    # --- exit multiple ---------------------------------------------------------
    tv_exit = multiple * last_ebitda
    pv_tv_exit = tv_exit * end_factor
    ev_exit = pv_fcff + pv_tv_exit

    # --- bridge ----------------------------------------------------------------
    claims = debt + minority + preferred
    equity_gordon = ev_gordon - claims
    equity_exit = ev_exit - claims

    # --- cross-checks ------------------------------------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        implied_multiple = np.where(last_ebitda > 0, tv_gordon / last_ebitda, np.nan)
        # Solve tv_exit = k * R * ((1 + g) * u - nwc * g) / (wacc - g) for g.
        scaled = mid_year_factor * last_revenue
        denominator = tv_exit + scaled * (unit_flow - nwc_pct)
        implied_growth = np.where(
            denominator > 0, (tv_exit * rate - scaled * unit_flow) / denominator, np.nan
        )

    def flat(values: np.ndarray) -> np.ndarray:
        return np.array(np.broadcast_to(values, (n_sims, 1))[:, 0])

    def full(values: np.ndarray) -> np.ndarray:
        return np.array(np.broadcast_to(values, shape_2d))

    return DCFCoreOutput(
        revenue=full(revenue),
        ebitda=full(ebitda),
        depreciation_amortization=full(da),
        ebit=full(ebit),
        taxes=full(taxes),
        nopat=full(nopat),
        capex=full(capex),
        delta_nwc=full(delta_nwc),
        fcff=full(fcff),
        discount_periods=periods,
        discount_factors=full(factors),
        pv_fcff=flat(pv_fcff),
        terminal_fcff=flat(terminal_fcff),
        tv_gordon=flat(tv_gordon),
        pv_tv_gordon=flat(pv_tv_gordon),
        ev_gordon=flat(ev_gordon),
        equity_gordon=flat(equity_gordon),
        value_per_share_gordon=flat(equity_gordon / shares),
        tv_exit=flat(tv_exit),
        pv_tv_exit=flat(pv_tv_exit),
        ev_exit=flat(ev_exit),
        equity_exit=flat(equity_exit),
        value_per_share_exit=flat(equity_exit / shares),
        implied_exit_multiple=flat(implied_multiple),
        implied_perpetual_growth=flat(implied_growth),
    )


# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------


#: Appended to ``details["range_definition"]`` of both DCF football-field rows.
_RANGE_FALLBACK_RULE = (
    f". Cells with WACC - g below {MIN_WACC_GROWTH_SPREAD:.1%} are invalid, as in the base "
    "case; an invalid low or high cell is replaced by the nearest valid cell between it and "
    "the base case (ties: the more extreme value), see range_fallback_to_base"
)


@dataclass
class DCFResult:
    """Full DCF output for one company. Monetary values in the trading currency."""

    ticker: str
    currency: str
    price: float
    shares_outstanding: float
    #: Assumptions as resolved: ``wacc`` and ``exit_multiple`` are always filled in.
    assumptions: DCFAssumptions
    wacc: float
    #: CAPM breakdown; None when the WACC was given in the assumptions.
    wacc_result: WACCResult | None
    exit_multiple: float
    exit_multiple_source: str
    #: One row per forecast year (index = fiscal year).
    forecast: pd.DataFrame
    terminal_fcff: float  # normalized FCFF of year N+1 (Gordon)
    pv_fcff: float  # sum of discounted explicit-period FCFF
    # Gordon growth
    terminal_value_gordon: float  # at end of year N
    pv_terminal_value_gordon: float
    enterprise_value_gordon: float
    equity_value_gordon: float
    value_per_share_gordon: float
    tv_share_gordon: float  # PV(TV) / EV
    implied_exit_multiple: float  # TV_gordon / EBITDA_N
    # Exit multiple
    terminal_value_exit: float  # at end of year N
    pv_terminal_value_exit: float
    enterprise_value_exit: float
    equity_value_exit: float
    value_per_share_exit: float
    tv_share_exit: float
    implied_perpetual_growth: float  # g that equates the Gordon TV to the exit TV
    # Bridge items (subtracted from EV)
    net_debt: float
    minority_interest: float
    preferred_equity: float
    #: Value per share; index = WACC, columns = terminal growth.
    sensitivity_gordon: pd.DataFrame
    #: Value per share; index = WACC, columns = exit multiple.
    sensitivity_exit: pd.DataFrame
    #: Exact keyword arguments of the base-case :func:`dcf_core` call. Monte Carlo
    #: overrides entries with arrays: ``dcf_core(**{**core_inputs, "wacc": draws})``.
    core_inputs: dict[str, Any] = field(default_factory=dict)
    #: Fallbacks and adjustments that fired (assumptions, WACC, exit multiple).
    notes: list[str] = field(default_factory=list)

    @property
    def upside_gordon(self) -> float:
        """Gordon value per share relative to the current price, minus one."""
        return self.value_per_share_gordon / self.price - 1.0

    @property
    def upside_exit(self) -> float:
        """Exit-multiple value per share relative to the current price, minus one."""
        return self.value_per_share_exit / self.price - 1.0

    def bridge(self) -> pd.DataFrame:
        """EV-to-equity bridge for both terminal value methods."""
        rows = {
            "pv_fcff": (self.pv_fcff, self.pv_fcff),
            "pv_terminal_value": (self.pv_terminal_value_gordon, self.pv_terminal_value_exit),
            "enterprise_value": (self.enterprise_value_gordon, self.enterprise_value_exit),
            "less_net_debt": (-self.net_debt, -self.net_debt),
            "less_minority_interest": (-self.minority_interest, -self.minority_interest),
            "less_preferred_equity": (-self.preferred_equity, -self.preferred_equity),
            "equity_value": (self.equity_value_gordon, self.equity_value_exit),
            "shares_outstanding": (self.shares_outstanding, self.shares_outstanding),
            "value_per_share": (self.value_per_share_gordon, self.value_per_share_exit),
        }
        return pd.DataFrame.from_dict(rows, orient="index", columns=["gordon", "exit_multiple"])

    def _range(
        self, grid: pd.DataFrame, base: float, low_cell: tuple[int, int], high_cell: tuple[int, int]
    ) -> tuple[float, float, bool]:
        """Low/high from two grid cells, enforcing ``low <= base <= high``.

        When an intended cell is NaN (invalid under the rules of the base case)
        the nearest valid cell is used instead, see :meth:`_nearest_valid`.
        Returns (low, high, fallback_used).
        """
        low, low_fallback = self._nearest_valid(grid, low_cell, prefer_high=False)
        high, high_fallback = self._nearest_valid(grid, high_cell, prefer_high=True)
        low = base if low is None else low
        high = base if high is None else high
        return min(low, base), max(high, base), low_fallback or high_fallback

    @staticmethod
    def _nearest_valid(
        grid: pd.DataFrame, cell: tuple[int, int], *, prefer_high: bool
    ) -> tuple[float | None, bool]:
        """Value of ``cell``, or of the nearest valid cell when it is NaN.

        Candidates are the cells of the rectangle spanned by the base cell (the
        centre of the grid) and the intended cell, so the substitute never lies
        beyond the intended cell. The nearest one in grid steps (Euclidean) wins;
        ties go to the more extreme value (highest for the high end, lowest for
        the low end). Returns (value or None when nothing is valid, fallback used).
        """
        value = _num(grid.iloc[cell])
        if value is not None:
            return value, False
        centre = (grid.shape[0] // 2, grid.shape[1] // 2)
        rows = range(min(centre[0], cell[0]), max(centre[0], cell[0]) + 1)
        cols = range(min(centre[1], cell[1]), max(centre[1], cell[1]) + 1)
        best: tuple[float, float] | None = None  # (distance, signed value)
        for r in rows:
            for c in cols:
                candidate = _num(grid.iloc[r, c])
                if candidate is None:
                    continue
                distance = math.hypot(r - cell[0], c - cell[1])
                key = (distance, -candidate if prefer_high else candidate)
                if best is None or key < best:
                    best = key
        if best is None:
            return None, True
        return (-best[1] if prefer_high else best[1]), True

    def to_valuation_results(self) -> list[ValuationResult]:
        """Two per-share ranges: Gordon growth and exit multiple.

        Low/high are cells of the sensitivity grids (definitions in ``details``).
        A cell that is invalid (WACC - g below ``MIN_WACC_GROWTH_SPREAD``) is
        replaced by the nearest valid cell between it and the base case, and
        ``details["range_fallback_to_base"]`` is then True:

        * Gordon: low = WACC + 1.0 pp and g - 0.5 pp; high = WACC - 1.0 pp and
          g + 0.5 pp.
        * Exit: low = WACC + 1.0 pp and multiple - 1.0x; high = WACC - 1.0 pp
          and multiple + 1.0x.

        A value per share can be negative (net debt above enterprise value); it is
        reported as computed and flagged in ``details``.
        """
        wacc_low_row = WACC_STEPS.index(RANGE_WACC_STEP)
        wacc_high_row = WACC_STEPS.index(-RANGE_WACC_STEP)
        common: dict[str, Any] = {
            "wacc": self.wacc,
            "terminal_growth": self.assumptions.terminal_growth,
            "exit_multiple": self.exit_multiple,
            "exit_multiple_source": self.exit_multiple_source,
            "mid_year": self.assumptions.mid_year,
            "forecast_years": self.assumptions.forecast_years,
            "price": self.price,
            "net_debt": self.net_debt,
            "notes": list(self.notes),
        }

        g_low, g_high, g_fallback = self._range(
            self.sensitivity_gordon,
            self.value_per_share_gordon,
            (wacc_low_row, GROWTH_STEPS.index(-RANGE_GROWTH_STEP)),
            (wacc_high_row, GROWTH_STEPS.index(RANGE_GROWTH_STEP)),
        )
        gordon = ValuationResult(
            method=ValuationMethod.DCF_GORDON,
            label="DCF - Gordon growth",
            low=g_low,
            base=self.value_per_share_gordon,
            high=g_high,
            currency=self.currency,
            enterprise_value=self.enterprise_value_gordon,
            equity_value=self.equity_value_gordon,
            details={
                **common,
                "range_definition": (
                    f"low: WACC +{RANGE_WACC_STEP:.1%} and g -{RANGE_GROWTH_STEP:.1%}; "
                    f"high: WACC -{RANGE_WACC_STEP:.1%} and g +{RANGE_GROWTH_STEP:.1%} "
                    "(cells of the WACC x g sensitivity grid)" + _RANGE_FALLBACK_RULE
                ),
                "range_fallback_to_base": g_fallback,
                "tv_share_of_ev": self.tv_share_gordon,
                "implied_exit_multiple": self.implied_exit_multiple,
                "upside_vs_price": self.upside_gordon,
                "negative_equity_value": self.equity_value_gordon < 0,
            },
        )

        e_low, e_high, e_fallback = self._range(
            self.sensitivity_exit,
            self.value_per_share_exit,
            (wacc_low_row, MULTIPLE_STEPS.index(-RANGE_MULTIPLE_STEP)),
            (wacc_high_row, MULTIPLE_STEPS.index(RANGE_MULTIPLE_STEP)),
        )
        exit_result = ValuationResult(
            method=ValuationMethod.DCF_EXIT_MULTIPLE,
            label="DCF - Exit multiple",
            low=e_low,
            base=self.value_per_share_exit,
            high=e_high,
            currency=self.currency,
            enterprise_value=self.enterprise_value_exit,
            equity_value=self.equity_value_exit,
            details={
                **common,
                "range_definition": (
                    f"low: WACC +{RANGE_WACC_STEP:.1%} and exit multiple "
                    f"-{RANGE_MULTIPLE_STEP:.1f}x; high: WACC -{RANGE_WACC_STEP:.1%} and exit "
                    f"multiple +{RANGE_MULTIPLE_STEP:.1f}x (cells of the WACC x multiple grid)"
                    + _RANGE_FALLBACK_RULE
                ),
                "range_fallback_to_base": e_fallback,
                "tv_share_of_ev": self.tv_share_exit,
                "implied_perpetual_growth": self.implied_perpetual_growth,
                "upside_vs_price": self.upside_exit,
                "negative_equity_value": self.equity_value_exit < 0,
            },
        )
        return [gordon, exit_result]


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def _validate_assumptions(assumptions: DCFAssumptions) -> None:
    """Raise NotApplicableError ('invalid assumptions: ...') on unusable input."""
    scalars = {
        "da_pct_revenue": assumptions.da_pct_revenue,
        "capex_pct_revenue": assumptions.capex_pct_revenue,
        "nwc_pct_revenue": assumptions.nwc_pct_revenue,
        "tax_rate": assumptions.tax_rate,
        "terminal_growth": assumptions.terminal_growth,
    }
    for name, value in scalars.items():
        if _num(value) is None:
            raise NotApplicableError(f"invalid assumptions: {name} is not a finite number")
    for name, values in (
        ("revenue_growth", assumptions.revenue_growth),
        ("ebitda_margin", assumptions.ebitda_margin),
    ):
        if any(_num(v) is None for v in values):
            raise NotApplicableError(f"invalid assumptions: {name} contains non-finite values")
    if any(g <= -1.0 for g in assumptions.revenue_growth):
        raise NotApplicableError("invalid assumptions: revenue growth must be above -100 %")
    if not 0.0 <= assumptions.tax_rate < 1.0:
        raise NotApplicableError("invalid assumptions: tax rate must be in [0, 1)")
    if assumptions.wacc is not None and (_num(assumptions.wacc) is None or assumptions.wacc <= 0):
        raise NotApplicableError("invalid assumptions: WACC must be a positive number")


#: Floating-point slack when a grid cell's spread is compared with the minimum.
_SPREAD_TOLERANCE = 1e-12


def _sensitivity(
    core_inputs: dict[str, Any], wacc_values: np.ndarray, name: str, other_values: np.ndarray
) -> pd.DataFrame:
    """Value per share on a WACC x ``name`` grid, evaluated in one core call.

    Cells the base case would reject are NaN: non-positive WACC or exit multiple,
    and ``WACC - g`` below ``MIN_WACC_GROWTH_SPREAD`` (in the exit-multiple grid
    ``g`` is the base terminal growth, so whole WACC rows can be NaN).
    """
    rows, cols = len(wacc_values), len(other_values)
    inputs = dict(core_inputs)
    inputs["wacc"] = np.repeat(wacc_values, cols)
    inputs[name] = np.tile(other_values, rows)
    output = dcf_core(**inputs)
    values = (
        output.value_per_share_gordon if name == "terminal_growth" else output.value_per_share_exit
    )
    if name == "exit_multiple":
        values = np.where(inputs[name] > 0, values, np.nan)
    values = np.where(inputs["wacc"] > 0, values, np.nan)
    # Same rule as the base case (and Monte Carlo): a cell whose WACC - g spread is
    # below the minimum is not a valuation but an exploding perpetuity -> NaN.
    spread = inputs["wacc"] - np.asarray(inputs["terminal_growth"], dtype=float)
    values = np.where(spread >= MIN_WACC_GROWTH_SPREAD - _SPREAD_TOLERANCE, values, np.nan)
    frame = pd.DataFrame(
        values.reshape(rows, cols),
        index=pd.Index(wacc_values, name="wacc"),
        columns=pd.Index(other_values, name=name),
    )
    return frame


def run_dcf(
    company: CompanyData,
    assumptions: DCFAssumptions | None = None,
    params: MarketParameters = _DEFAULT_PARAMS,
    peers: list[PeerSnapshot] | None = None,
    *,
    size_premium: float | None = None,
    peers_are_peer_group: bool = False,
    peer_multiple: tuple[float, str] | None = None,
) -> DCFResult:
    """Value one company with an FCFF DCF (Gordon growth and exit multiple).

    * ``assumptions=None`` -> :func:`default_assumptions` (fallbacks end up in
      ``DCFResult.notes``).
    * ``assumptions.wacc`` set -> used as is, CAPM skipped (``wacc_result=None``);
      otherwise :func:`compute_wacc` with ``assumptions.tax_rate``.
    * ``assumptions.exit_multiple`` set -> used as is; otherwise
      :func:`resolve_exit_multiple`.
    * ``peers`` (the peer universe) feeds the peer beta and the default exit
      multiple; without it the own beta and own EV/EBITDA are used.
    * ``size_premium`` is passed through to :func:`compute_wacc`: ``None`` (the
      default) looks the premium up in ``SIZE_PREMIUM_SCHEDULE``, ``0.0`` gives
      plain CAPM. Ignored when ``assumptions.wacc`` is set.
    * ``peers_are_peer_group=True``: ``peers`` is a ready-made peer group (e.g.
      the peers selected by comps) and is used as it is for the relevered beta
      and the exit multiple, without the industry -> sector filter. Minimum
      counts and outlier filters still apply, the company itself is still
      excluded (its own row is only read for the SEK market cap of the size
      premium), and the source label reads "comps peer group (N peers)".
    * ``peer_multiple=(multiple, source label)``: ready-made peer EV/EBITDA
      multiple used as the default exit multiple instead of the peer median
      (see :func:`resolve_exit_multiple`).

    All arithmetic is done by :func:`dcf_core`; the base case, both sensitivity
    grids and the Monte Carlo model share that one implementation.

    Raises :class:`NotApplicableError` (never another exception for bad data).
    Reason categories, i.e. the text before the first colon: ``financial sector``,
    ``insufficient history``, ``missing revenue``, ``missing EBITDA``,
    ``negative EBITDA``, ``invalid market data``, ``invalid assumptions``,
    ``WACC not above terminal growth``, ``non-positive terminal cash flow``,
    ``non-finite result``.
    """
    check_applicability(company)
    notes: list[str] = []
    if assumptions is None:
        assumptions, notes = default_assumptions_with_notes(company, params)
    _validate_assumptions(assumptions)

    # --- WACC ------------------------------------------------------------------
    wacc_result: WACCResult | None = None
    if assumptions.wacc is not None:
        wacc = float(assumptions.wacc)
    else:
        wacc_result = compute_wacc(
            company,
            params,
            peers,
            tax_rate=assumptions.tax_rate,
            size_premium=size_premium,
            peers_are_peer_group=peers_are_peer_group,
        )
        wacc = wacc_result.wacc
        notes.extend(wacc_result.notes)
    growth_terminal = float(assumptions.terminal_growth)
    if not math.isfinite(wacc) or wacc - growth_terminal < MIN_WACC_GROWTH_SPREAD:
        raise NotApplicableError(
            f"WACC not above terminal growth: WACC {wacc:.2%} vs g {growth_terminal:.2%} "
            f"(minimum spread {MIN_WACC_GROWTH_SPREAD:.1%})"
        )

    # --- exit multiple ---------------------------------------------------------
    exit_multiple, exit_source = resolve_exit_multiple(
        company,
        assumptions,
        peers,
        peers_are_peer_group=peers_are_peer_group,
        peer_multiple=peer_multiple,
    )
    if "clipped" in exit_source or exit_source.startswith("fallback"):
        notes.append(f"exit multiple: {exit_source}")

    # --- base case -------------------------------------------------------------
    market = company.market
    statements = company.statements
    total_debt = max(_num(market.total_debt) or 0.0, 0.0)
    cash = _num(market.cash) or 0.0
    core_inputs: dict[str, Any] = {
        "base_revenue": float(_column(statements.income, "revenue").iloc[-1]),
        "revenue_growth": np.asarray(assumptions.revenue_growth, dtype=float),
        "ebitda_margin": np.asarray(assumptions.ebitda_margin, dtype=float),
        "da_pct_revenue": float(assumptions.da_pct_revenue),
        "capex_pct_revenue": float(assumptions.capex_pct_revenue),
        "nwc_pct_revenue": float(assumptions.nwc_pct_revenue),
        "tax_rate": float(assumptions.tax_rate),
        "wacc": wacc,
        "terminal_growth": growth_terminal,
        "exit_multiple": exit_multiple,
        "net_debt": total_debt - cash,
        "shares_outstanding": float(market.shares_outstanding),
        "minority_interest": max(_num(market.minority_interest) or 0.0, 0.0),
        "preferred_equity": max(_num(market.preferred_equity) or 0.0, 0.0),
        "mid_year": bool(assumptions.mid_year),
        "terminal_capex_floor": True,
    }
    out = dcf_core(**core_inputs)

    terminal_fcff = float(out.terminal_fcff[0])
    if not terminal_fcff > 0:
        raise NotApplicableError(
            "non-positive terminal cash flow: normalized terminal FCFF is not positive under "
            "the assumptions (margin too thin for D&A, capex and NWC investment)"
        )
    headline = (
        out.ev_gordon[0],
        out.ev_exit[0],
        out.value_per_share_gordon[0],
        out.value_per_share_exit[0],
    )
    if not all(math.isfinite(float(v)) for v in headline) or min(headline[:2]) <= 0:
        raise NotApplicableError(
            "non-finite result: enterprise value is not a positive finite number under the "
            "assumptions"
        )

    # --- forecast table ----------------------------------------------------------
    n = assumptions.forecast_years
    first_year = statements.latest_year + 1
    forecast = pd.DataFrame(
        {
            "revenue": out.revenue[0],
            "revenue_growth": np.asarray(assumptions.revenue_growth, dtype=float),
            "ebitda": out.ebitda[0],
            "ebitda_margin": np.asarray(assumptions.ebitda_margin, dtype=float),
            "depreciation_amortization": out.depreciation_amortization[0],
            "ebit": out.ebit[0],
            "taxes": out.taxes[0],
            "nopat": out.nopat[0],
            "capex": out.capex[0],
            "delta_nwc": out.delta_nwc[0],
            "fcff": out.fcff[0],
            "discount_period": out.discount_periods,
            "discount_factor": out.discount_factors[0],
            "pv_fcff": out.fcff[0] * out.discount_factors[0],
        },
        index=pd.Index(range(first_year, first_year + n), name="fiscal_year"),
    )

    # --- sensitivities -----------------------------------------------------------
    wacc_values = wacc + np.asarray(WACC_STEPS, dtype=float)
    sensitivity_gordon = _sensitivity(
        core_inputs,
        wacc_values,
        "terminal_growth",
        growth_terminal + np.asarray(GROWTH_STEPS, dtype=float),
    )
    sensitivity_exit = _sensitivity(
        core_inputs,
        wacc_values,
        "exit_multiple",
        exit_multiple + np.asarray(MULTIPLE_STEPS, dtype=float),
    )

    ev_gordon = float(out.ev_gordon[0])
    ev_exit = float(out.ev_exit[0])
    return DCFResult(
        ticker=company.ticker,
        currency=market.currency,
        price=float(market.price),
        shares_outstanding=float(market.shares_outstanding),
        assumptions=replace(assumptions, wacc=wacc, exit_multiple=exit_multiple),
        wacc=wacc,
        wacc_result=wacc_result,
        exit_multiple=exit_multiple,
        exit_multiple_source=exit_source,
        forecast=forecast,
        terminal_fcff=terminal_fcff,
        pv_fcff=float(out.pv_fcff[0]),
        terminal_value_gordon=float(out.tv_gordon[0]),
        pv_terminal_value_gordon=float(out.pv_tv_gordon[0]),
        enterprise_value_gordon=ev_gordon,
        equity_value_gordon=float(out.equity_gordon[0]),
        value_per_share_gordon=float(out.value_per_share_gordon[0]),
        tv_share_gordon=float(out.pv_tv_gordon[0]) / ev_gordon,
        implied_exit_multiple=float(out.implied_exit_multiple[0]),
        terminal_value_exit=float(out.tv_exit[0]),
        pv_terminal_value_exit=float(out.pv_tv_exit[0]),
        enterprise_value_exit=ev_exit,
        equity_value_exit=float(out.equity_exit[0]),
        value_per_share_exit=float(out.value_per_share_exit[0]),
        tv_share_exit=float(out.pv_tv_exit[0]) / ev_exit,
        implied_perpetual_growth=float(out.implied_perpetual_growth[0]),
        net_debt=core_inputs["net_debt"],
        minority_interest=core_inputs["minority_interest"],
        preferred_equity=core_inputs["preferred_equity"],
        sensitivity_gordon=sensitivity_gordon,
        sensitivity_exit=sensitivity_exit,
        core_inputs=core_inputs,
        notes=notes,
    )
