"""Leveraged buyout (LBO) model and "ability to pay" analysis.

The model answers two questions for a listed, non-financial company:

1. *Returns at a given price.* A financial sponsor buys all shares at the
   market price plus a takeover premium, refinances the existing debt, funds
   the deal with a senior term loan, a subordinated tranche and its own
   equity, runs the company for the holding period and sells it at an exit
   EV/EBITDA multiple. What are IRR and MOIC, and where do they come from?
2. *Ability to pay.* Which price per share could the sponsor pay today and
   still reach a target IRR? This is the LBO bar in the football field:
   base = price giving 20 % IRR, low = 25 % IRR, high = 15 % IRR.

Structure and conventions
-------------------------
* **Entry.** Equity purchase price = price per share x
  ``MarketData.shares_outstanding``. Entry EV = equity purchase price
  + total debt - cash + minority interest + preferred equity. Entry multiple =
  entry EV / EBITDA of the latest fiscal year (no LTM data is available).
* **Sources & uses (gross presentation).** Uses: purchase of equity,
  refinancing of ALL existing interest-bearing debt (gross, including lease
  liabilities as delivered by the data layer), redemption of preferred equity,
  transaction fees (% of entry EV), financing fees (% of new debt, expensed at
  closing) and the minimum cash left on the balance sheet. Sources: senior
  debt, subordinated debt, the target's entire existing cash balance, and
  sponsor equity as the plug. Sources equal uses by construction.
* **Minority interest is NOT bought out.** The sponsor acquires the shares of
  the parent; minorities sit in subsidiaries and stay. They are part of the
  entry EV, are no use of funds, and are deducted again from the exit EV at
  the same (book) value.
* **Debt sizing.** Each tranche is requested as a multiple of entry EBITDA and
  then capped, subordinated debt first and senior debt second, so that
  (a) entry EBITDA / first-year interest >= ``min_interest_coverage``,
  (b) (entry EBITDA - capex) / (first-year cash interest + mandatory
  amortization) >= ``min_debt_service_coverage`` and
  (c) sponsor equity >= ``min_equity_share`` of the new capital raised
  (senior + subordinated + sponsor equity, i.e. total sources excluding the
  target's own cash).
* **Interest** is charged on the OPENING balance of each year, so there is no
  circularity between interest and the cash sweep.
* **Cash sweep.** After interest, mandatory amortization and repayment of any
  revolver drawing, a configurable share of all cash above the minimum balance
  prepays the senior loan. The
  subordinated tranche is a bullet (optionally PIK) and is never prepaid.
* **Liquidity.** Interest and mandatory amortization are contractual and are
  always paid. A resulting cash deficit against the minimum balance is drawn
  on a revolving credit facility (undrawn at closing, senior interest rate,
  repaid before any sweep). If the revolver limit is exhausted, cash falls
  below the minimum and the year is flagged as a liquidity breach instead of
  being hidden; a negative cash balance is an unfunded shortfall that carries
  the senior interest rate.
* **Tax.** Tax = rate x max(EBIT - deductible interest, 0). Losses create no
  tax asset (no loss carry-forward). Deductible interest is capped at 30 % of
  EBITDA by default (Swedish interest deduction limitation rule).
* **Exit.** Exit EV = exit multiple x EBITDA of the exit year. When
  ``exit_multiple`` is ``None`` the exit multiple equals the entry multiple of
  the transaction being evaluated (multiple-neutral case). Sponsor proceeds =
  exit EV - debt + cash - minority interest, floored at zero.

The operating forecast is self-contained and driven by ``LBOAssumptions``; this
module does not import the DCF or comps modules. Note that the pipeline
(``valuation_engine.pipeline``) by default replaces the standalone defaults
described here: it passes the DCF's operating forecast and the DCF's exit
multiple, so that all methods describe the same company.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from valuation_engine.core.types import (
    CompanyData,
    MarketParameters,
    NotApplicableError,
    ValuationMethod,
    ValuationResult,
)

__all__ = [
    "LBOAssumptions",
    "LBOResult",
    "compute_irr",
    "default_assumptions",
    "run_lbo",
    "with_entry_price",
]

# Bounds applied to history-derived default drivers (documented in
# ``default_assumptions``).
_GROWTH_BOUNDS: tuple[float, float] = (-0.05, 0.10)
_MARGIN_REVERSION_BAND: tuple[float, float] = (0.75, 1.25)
_FALLBACK_DA_PCT: float = 0.03
_SENSITIVITY_STEPS: tuple[float, ...] = (0.8, 0.9, 1.0, 1.1, 1.2)
_LEVERAGE_STEPS: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)
_DEFAULT_PARAMS = MarketParameters()

#: Status codes of the ability-to-pay solver.
STATUS_SOLVED = "solved"
STATUS_UNREACHABLE = "unreachable"  # target IRR not reached even at a zero price
STATUS_UNBOUNDED = "unbounded"  # target IRR reached at any price tested


# --------------------------------------------------------------------------
# Assumptions
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LBOAssumptions:
    """Drivers of the LBO. Tuples have one entry per year of the holding period.

    Rates, margins and shares are decimals (0.20 means 20 %).
    """

    # --- operating forecast -------------------------------------------------
    revenue_growth: tuple[float, ...]
    ebitda_margin: tuple[float, ...]
    da_pct_revenue: float
    capex_pct_revenue: float
    nwc_pct_revenue: float  # operating NWC as share of revenue
    tax_rate: float = 0.206
    #: Deductible interest as max share of EBITDA; ``None`` = fully deductible.
    interest_deduction_cap_pct_ebitda: float | None = 0.30

    # --- transaction --------------------------------------------------------
    takeover_premium: float = 0.30
    #: Explicit offer price per share; overrides ``takeover_premium``.
    entry_price_per_share: float | None = None
    transaction_fee_pct: float = 0.02  # of entry EV
    financing_fee_pct: float = 0.02  # of new debt raised

    # --- financing ----------------------------------------------------------
    senior_debt_multiple: float = 4.0  # x entry EBITDA, before caps
    subordinated_debt_multiple: float = 1.5  # x entry EBITDA, before caps
    #: Default: the risk-free rate of ``MarketParameters()`` (one project default).
    risk_free_rate: float = _DEFAULT_PARAMS.risk_free_rate
    senior_spread: float = 0.040
    subordinated_spread: float = 0.080
    senior_amortization_pct: float = 0.05  # of original principal, per year
    cash_sweep_pct: float = 1.0  # share of cash above the minimum that prepays senior
    subordinated_pik: bool = False  # True: interest accrues to principal
    min_equity_share: float = 0.30  # of debt + sponsor equity
    min_interest_coverage: float = 2.0  # entry EBITDA / first-year interest
    #: (entry EBITDA - capex) / (first-year cash interest + mandatory amortization).
    min_debt_service_coverage: float = 1.25
    min_cash_pct_revenue: float = 0.02
    #: Revolving credit facility, undrawn at closing, as a multiple of entry EBITDA.
    revolver_limit_x_ebitda: float = 0.5

    # --- exit and returns ---------------------------------------------------
    #: EV/EBITDA at exit; ``None`` = entry multiple of the evaluated deal.
    exit_multiple: float | None = None
    target_irr_base: float = 0.20  # -> base price
    target_irr_conservative: float = 0.25  # -> low price
    target_irr_aggressive: float = 0.15  # -> high price

    def __post_init__(self) -> None:
        if len(self.revenue_growth) != len(self.ebitda_margin):
            raise ValueError("revenue_growth and ebitda_margin must have equal length")
        if not self.revenue_growth:
            raise ValueError("the holding period must be at least one year")
        if any(g <= -1.0 for g in self.revenue_growth):
            raise ValueError("revenue_growth must be greater than -100 %")
        non_negative = {
            "transaction_fee_pct": self.transaction_fee_pct,
            "financing_fee_pct": self.financing_fee_pct,
            "senior_debt_multiple": self.senior_debt_multiple,
            "subordinated_debt_multiple": self.subordinated_debt_multiple,
            "senior_amortization_pct": self.senior_amortization_pct,
            "min_cash_pct_revenue": self.min_cash_pct_revenue,
            "revolver_limit_x_ebitda": self.revolver_limit_x_ebitda,
            "min_interest_coverage": self.min_interest_coverage,
            "min_debt_service_coverage": self.min_debt_service_coverage,
            "tax_rate": self.tax_rate,
        }
        for name, value in non_negative.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a non-negative number, got {value}")
        if not 0.0 <= self.cash_sweep_pct <= 1.0:
            raise ValueError("cash_sweep_pct must be between 0 and 1")
        if not 0.0 <= self.min_equity_share < 1.0:
            raise ValueError("min_equity_share must be in [0, 1)")
        if self.financing_fee_pct >= 1.0 or self.transaction_fee_pct >= 1.0:
            raise ValueError("fee percentages must be below 100 %")
        if self.senior_rate < 0 or self.subordinated_rate < 0:
            raise ValueError("interest rates must not be negative")
        if self.exit_multiple is not None and not self.exit_multiple > 0:
            raise ValueError("exit_multiple must be positive")
        if self.entry_price_per_share is not None and not self.entry_price_per_share > 0:
            raise ValueError("entry_price_per_share must be positive")
        if not (
            -1.0
            < self.target_irr_aggressive
            <= self.target_irr_base
            <= self.target_irr_conservative
        ):
            raise ValueError(
                "expected target_irr_aggressive <= target_irr_base <= target_irr_conservative"
            )

    @property
    def holding_period(self) -> int:
        """Years from closing to exit (exit at the end of the last year)."""
        return len(self.revenue_growth)

    @property
    def senior_rate(self) -> float:
        """All-in interest rate of the senior term loan."""
        return self.risk_free_rate + self.senior_spread

    @property
    def subordinated_rate(self) -> float:
        """All-in interest rate of the subordinated tranche."""
        return self.risk_free_rate + self.subordinated_spread


# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------


@dataclass
class LBOResult:
    """Output of :func:`run_lbo`. Amounts in the trading currency."""

    ticker: str
    currency: str
    assumptions: LBOAssumptions

    # Entry
    market_price: float
    shares_outstanding: float
    entry_price_per_share: float
    entry_equity_value: float  # purchase price of 100 % of the shares
    entry_enterprise_value: float
    entry_ebitda: float  # latest fiscal year
    entry_multiple: float  # entry EV / entry EBITDA

    # Financing
    #: Columns ``side`` ("sources"/"uses"), ``item``, ``amount``, ``pct`` of side total.
    sources_and_uses: pd.DataFrame
    senior_debt: float
    subordinated_debt: float
    sponsor_equity: float
    leverage: float  # (senior + subordinated) / entry EBITDA
    equity_share: float  # sponsor equity / (debt + sponsor equity)
    sizing_notes: list[str]

    # Projection (index = fiscal year)
    forecast: pd.DataFrame
    debt_schedule: pd.DataFrame

    # Exit and returns
    exit_year: int
    exit_multiple: float
    exit_ebitda: float
    exit_enterprise_value: float
    exit_net_debt: float  # senior + subordinated + revolver - cash at exit
    exit_minority_interest: float
    exit_equity_value: float  # sponsor proceeds, floored at zero
    equity_cash_flows: tuple[float, ...]  # t = 0 .. holding period
    irr: float  # NaN when not defined; -1.0 for a total loss
    moic: float

    #: Index = component, columns ``amount`` and ``share`` (amount / |total value
    #: creation|: a share has the sign of its amount; the total row is +1 or -1).
    attribution: pd.DataFrame

    # Sensitivities (IRR / MOIC; NaN where the deal is not feasible)
    sensitivity_irr_entry_exit: pd.DataFrame  # index entry multiple, columns exit multiple
    sensitivity_moic_entry_exit: pd.DataFrame
    sensitivity_irr_leverage_exit: pd.DataFrame  # index requested debt / EBITDA

    # Ability to pay: target IRR -> price per share and solver status
    ability_to_pay: dict[float, float]
    ability_to_pay_status: dict[float, str]

    liquidity_breach_years: list[int] = field(default_factory=list)
    equity_wiped_out: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_valuation_results(self) -> list[ValuationResult]:
        """Return the LBO bar of the football field (at most one result).

        base = price per share at which the sponsor earns the base target IRR
        (default 20 %), low = price for the conservative target (25 %), high =
        price for the aggressive target (15 %). A target that cannot be reached
        at any positive price maps to a price of zero. The list is empty when
        the base price is not a positive, solved value (the reason is in
        ``warnings``).
        """
        a = self.assumptions
        low = self.ability_to_pay.get(a.target_irr_conservative, math.nan)
        base = self.ability_to_pay.get(a.target_irr_base, math.nan)
        high = self.ability_to_pay.get(a.target_irr_aggressive, math.nan)
        base_status = self.ability_to_pay_status.get(a.target_irr_base)
        values_ok = all(math.isfinite(v) for v in (low, base, high))
        if base_status != STATUS_SOLVED or not values_ok or base <= 0 or not low <= base <= high:
            return []
        equity_value = base * self.shares_outstanding
        net_claims = self.entry_enterprise_value - self.entry_equity_value
        details: dict[str, Any] = {
            "definition": (
                "Ability to pay: price per share a financial sponsor could pay today and "
                "still reach the target IRR at exit. base = "
                f"{a.target_irr_base:.0%} IRR, low = {a.target_irr_conservative:.0%} IRR, "
                f"high = {a.target_irr_aggressive:.0%} IRR. Debt is sized as a multiple of "
                "EBITDA, sponsor equity is the plug."
            ),
            "exit_multiple_convention": (
                "fixed at the assumption" if a.exit_multiple is not None else "equal to entry"
            ),
            "target_irr": {
                "low": a.target_irr_conservative,
                "base": a.target_irr_base,
                "high": a.target_irr_aggressive,
            },
            "solver_status": {f"{k:.4f}": v for k, v in self.ability_to_pay_status.items()},
            "market_price": self.market_price,
            "base_price_to_market": base / self.market_price,
            "implied_ev_ebitda_at_base": (equity_value + net_claims) / self.entry_ebitda,
            "holding_period_years": a.holding_period,
            "default_entry_price": self.entry_price_per_share,
            "default_entry_multiple": self.entry_multiple,
            "default_entry_irr": self.irr,
            "default_entry_moic": self.moic,
            "leverage_x_ebitda": self.leverage,
            "liquidity_breach_years": list(self.liquidity_breach_years),
            "warnings": list(self.warnings),
        }
        return [
            ValuationResult(
                method=ValuationMethod.LBO,
                label="LBO - ability to pay",
                low=low,
                base=base,
                high=high,
                currency=self.currency,
                enterprise_value=equity_value + net_claims,
                equity_value=equity_value,
                details=details,
            )
        ]


# --------------------------------------------------------------------------
# IRR
# --------------------------------------------------------------------------


def compute_irr(cash_flows: Sequence[float]) -> float:
    """Internal rate of return of annual cash flows (index = year).

    Solved by root finding on the NPV. Returns ``-1.0`` for a total loss (an
    initial outflow followed by nothing) and ``NaN`` when no rate in
    (-100 %, 1 000 000 %) sets the NPV to zero, e.g. when all flows have the
    same sign.

    Flows with at most one sign change (every LBO: one outflow followed by
    inflows) have a unique root. With several sign changes there can be several
    roots; the SMALLEST root above -100 % is returned, found by scanning the
    NPV for its first change of sign on a grid of rates (roots closer together
    than the grid spacing, about 1 % of ``1 + rate``, can be missed).
    """
    flows = [float(c) for c in cash_flows]
    if len(flows) < 2 or not all(math.isfinite(c) for c in flows):
        return math.nan
    has_outflow = any(c < 0 for c in flows)
    has_inflow = any(c > 0 for c in flows)
    if has_outflow and not has_inflow:
        return -1.0
    if not has_outflow:
        return math.nan

    def npv(rate: float) -> float:
        return sum(c / (1.0 + rate) ** t for t, c in enumerate(flows))

    signs = [c > 0 for c in flows if c != 0]
    if sum(a != b for a, b in zip(signs, signs[1:], strict=False)) > 1:
        return _smallest_root(npv)

    low, high = -0.9999, 1.0
    f_low = npv(low)
    f_high = npv(high)
    while f_low * f_high > 0 and high < 1e6:
        high *= 10.0
        f_high = npv(high)
    if f_low * f_high > 0:
        return math.nan
    if f_low == 0:
        return low
    if f_high == 0:
        return high
    return float(brentq(npv, low, high, xtol=1e-12, rtol=1e-12, maxiter=500))


def _smallest_root(npv: Any) -> float:
    """Smallest rate in (-100 %, 1 000 000 %) with ``npv(rate) == 0``, else NaN.

    Scans ``1 + rate`` on a logarithmic grid from 1e-4 to 1e6 and refines the
    first bracket with a change of sign.
    """
    grid = np.logspace(-4.0, 6.0, 2301) - 1.0
    previous_rate = float(grid[0])
    previous = npv(previous_rate)
    if previous == 0:
        return previous_rate
    for rate in grid[1:]:
        rate = float(rate)
        value = npv(rate)
        if value == 0:
            return rate
        if math.isfinite(previous) and math.isfinite(value) and previous * value < 0:
            return float(brentq(npv, previous_rate, rate, xtol=1e-12, rtol=1e-12, maxiter=500))
        previous_rate, previous = rate, value
    return math.nan


# --------------------------------------------------------------------------
# Internal building blocks
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Target:
    """Scalars extracted from ``CompanyData`` (validated, finite)."""

    ticker: str
    currency: str
    latest_year: int
    price: float
    shares: float
    total_debt: float
    cash: float
    minority_interest: float
    preferred_equity: float
    revenue: float
    ebitda: float

    @property
    def net_claims(self) -> float:
        """Entry EV minus equity purchase price."""
        return self.total_debt - self.cash + self.minority_interest + self.preferred_equity


@dataclass(frozen=True)
class _Deal:
    """A fully funded transaction at one entry price."""

    price: float
    equity_purchase: float
    enterprise_value: float
    entry_multiple: float
    transaction_fees: float
    financing_fees: float
    minimum_cash: float
    senior: float
    subordinated: float
    sponsor_equity: float  # plug; may be <= 0 for infeasible trial prices
    notes: tuple[str, ...]


@dataclass(frozen=True)
class _Projection:
    """Year-by-year projection of one deal (lists have one entry per year)."""

    rows: tuple[dict[str, float], ...]
    exit_multiple: float
    exit_ebitda: float
    exit_enterprise_value: float
    exit_net_debt: float  # senior + subordinated + revolver - cash
    exit_equity_raw: float  # before the zero floor

    @property
    def exit_equity(self) -> float:
        return max(self.exit_equity_raw, 0.0)


def _finite(value: Any, default: float = math.nan) -> float:
    """Return ``value`` as a finite float, else ``default``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    """Numeric column of a statement; all-NaN when the column is absent."""
    if name not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").astype(float)


def _median_ratio(numerator: pd.Series, revenue: pd.Series) -> float:
    """Median of ``numerator / revenue`` over years with positive revenue."""
    ratio = (numerator / revenue.where(revenue > 0)).replace([np.inf, -np.inf], np.nan).dropna()
    return float(ratio.median()) if len(ratio) else math.nan


def _clip(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _extract_target(company: CompanyData) -> _Target:
    """Validate applicability and pull the scalars the model needs."""
    profile = company.profile
    ticker = profile.ticker
    if profile.is_financial:
        raise NotApplicableError(
            f"{ticker}: LBO not applicable to financial or real estate companies "
            f"(sector: {profile.sector})"
        )
    income = company.statements.income
    if income is None or len(income.index) < 2:
        years = 0 if income is None else len(income.index)
        raise NotApplicableError(
            f"{ticker}: LBO requires at least 2 fiscal years of statements, found {years}"
        )
    revenue = _finite(_column(income, "revenue").iloc[-1])
    if not math.isfinite(revenue):
        raise NotApplicableError(f"{ticker}: revenue of the latest fiscal year is missing")
    if revenue <= 0:
        raise NotApplicableError(f"{ticker}: revenue of the latest fiscal year is not positive")
    ebitda = _finite(_column(income, "ebitda").iloc[-1])
    if not math.isfinite(ebitda):
        raise NotApplicableError(f"{ticker}: EBITDA of the latest fiscal year is missing")
    if ebitda <= 0:
        raise NotApplicableError(
            f"{ticker}: EBITDA of the latest fiscal year is not positive; "
            "debt cannot be sized on EBITDA"
        )
    market = company.market
    price = _finite(market.price)
    shares = _finite(market.shares_outstanding)
    if not (price > 0 and shares > 0):
        raise NotApplicableError(f"{ticker}: market price or shares outstanding missing")
    try:
        latest_year = int(income.index[-1])
    except (TypeError, ValueError) as exc:
        raise NotApplicableError(f"{ticker}: fiscal year index is not numeric") from exc
    return _Target(
        ticker=ticker,
        currency=profile.currency,
        latest_year=latest_year,
        price=price,
        shares=shares,
        total_debt=max(_finite(market.total_debt, 0.0), 0.0),
        cash=max(_finite(market.cash, 0.0), 0.0),
        minority_interest=max(_finite(market.minority_interest, 0.0), 0.0),
        preferred_equity=max(_finite(market.preferred_equity, 0.0), 0.0),
        revenue=revenue,
        ebitda=ebitda,
    )


def _cut_to_budget(
    senior: float,
    subordinated: float,
    senior_cost: float,
    subordinated_cost: float,
    budget: float,
) -> tuple[float, float, bool]:
    """Reduce the tranches until ``senior_cost * senior + subordinated_cost *
    subordinated <= budget``; subordinated debt is cut first. Returns the new
    amounts and whether anything was cut."""
    budget = max(budget, 0.0)
    if senior_cost * senior + subordinated_cost * subordinated <= budget + 1e-9:
        return senior, subordinated, False
    room = budget - senior_cost * senior
    if room >= 0:
        # Only reachable with a positive subordinated cost.
        return senior, min(subordinated, room / subordinated_cost), True
    if subordinated_cost > 0:
        subordinated = 0.0
    if senior_cost > 0:
        senior = min(senior, budget / senior_cost)
    return senior, subordinated, True


def _size_debt(
    a: LBOAssumptions,
    ebitda: float,
    ebitda_less_capex: float,
    funding_need_before_financing_fees: float,
    senior_multiple: float,
    subordinated_multiple: float,
) -> tuple[float, float, tuple[str, ...]]:
    """Size the two tranches: multiples of EBITDA, then three caps.

    All caps cut the subordinated tranche first (the most expensive layer)
    and the senior loan second.
    """
    senior = senior_multiple * ebitda
    subordinated = subordinated_multiple * ebitda
    notes: list[str] = []

    # (a) Interest coverage: entry EBITDA / first-year interest >= minimum.
    if a.min_interest_coverage > 0:
        senior, subordinated, cut = _cut_to_budget(
            senior,
            subordinated,
            a.senior_rate,
            a.subordinated_rate,
            ebitda / a.min_interest_coverage,
        )
        if cut:
            notes.append(
                f"debt cut to keep EBITDA / interest at {a.min_interest_coverage:.1f}x or above"
            )

    # (b) Debt service coverage: (entry EBITDA - capex) / (first-year cash
    # interest + mandatory amortization) >= minimum. PIK interest is not cash.
    if a.min_debt_service_coverage > 0:
        senior, subordinated, cut = _cut_to_budget(
            senior,
            subordinated,
            a.senior_rate + a.senior_amortization_pct,
            0.0 if a.subordinated_pik else a.subordinated_rate,
            ebitda_less_capex / a.min_debt_service_coverage,
        )
        if cut:
            notes.append(
                "debt cut to keep (EBITDA - capex) / (cash interest + mandatory "
                f"amortization) at {a.min_debt_service_coverage:.2f}x or above"
            )

    # (c) Minimum equity share of the new capital (debt + sponsor equity).
    # Capital K = need + f * D and equity = K - D >= m * K  =>
    # D <= (1 - m) * need / (1 - (1 - m) * f).
    debt_share = 1.0 - a.min_equity_share
    max_debt = max(
        debt_share * funding_need_before_financing_fees / (1.0 - debt_share * a.financing_fee_pct),
        0.0,
    )
    if senior + subordinated > max_debt + 1e-9:
        subordinated = max(max_debt - senior, 0.0)
        senior = min(senior, max_debt)
        notes.append(
            f"debt cut so that sponsor equity is at least {a.min_equity_share:.0%} "
            "of debt plus equity"
        )
    return senior, subordinated, tuple(notes)


def _build_deal(
    target: _Target,
    a: LBOAssumptions,
    price: float,
    senior_multiple: float | None = None,
    subordinated_multiple: float | None = None,
) -> _Deal:
    """Fund the acquisition at ``price`` per share. Never raises; the caller
    checks feasibility (``sponsor_equity`` > 0, ``enterprise_value`` > 0)."""
    equity_purchase = price * target.shares
    enterprise_value = equity_purchase + target.net_claims
    transaction_fees = a.transaction_fee_pct * max(enterprise_value, 0.0)
    minimum_cash = a.min_cash_pct_revenue * target.revenue
    uses_before_financing_fees = (
        equity_purchase
        + target.total_debt
        + target.preferred_equity
        + transaction_fees
        + minimum_cash
    )
    need = uses_before_financing_fees - target.cash
    senior, subordinated, notes = _size_debt(
        a,
        target.ebitda,
        target.ebitda - a.capex_pct_revenue * target.revenue,
        need,
        a.senior_debt_multiple if senior_multiple is None else senior_multiple,
        a.subordinated_debt_multiple if subordinated_multiple is None else subordinated_multiple,
    )
    financing_fees = a.financing_fee_pct * (senior + subordinated)
    sponsor_equity = need + financing_fees - senior - subordinated
    return _Deal(
        price=price,
        equity_purchase=equity_purchase,
        enterprise_value=enterprise_value,
        entry_multiple=enterprise_value / target.ebitda,
        transaction_fees=transaction_fees,
        financing_fees=financing_fees,
        minimum_cash=minimum_cash,
        senior=senior,
        subordinated=subordinated,
        sponsor_equity=sponsor_equity,
        notes=notes,
    )


def _project(
    target: _Target,
    a: LBOAssumptions,
    deal: _Deal,
    exit_multiple: float | None = None,
) -> _Projection:
    """Operating forecast, debt schedule and exit for one deal.

    Invariants by construction: mandatory amortization and sweep are each
    bounded by the outstanding senior balance, the sweep is bounded by cash
    above the minimum balance (after the revolver has been repaid), the
    revolver stays within [0, limit] and no debt balance can become negative.
    """
    multiple = exit_multiple
    if multiple is None:
        multiple = a.exit_multiple if a.exit_multiple is not None else deal.entry_multiple

    revenue = target.revenue
    nwc = a.nwc_pct_revenue * revenue  # opening NWC at the normalized ratio
    senior = deal.senior
    subordinated = deal.subordinated
    revolver = 0.0  # undrawn at closing
    revolver_limit = a.revolver_limit_x_ebitda * target.ebitda
    cash = deal.minimum_cash
    senior_rate = a.senior_rate
    sub_rate = a.subordinated_rate
    scheduled = a.senior_amortization_pct * deal.senior
    rows: list[dict[str, float]] = []
    ebitda = target.ebitda

    for growth, margin in zip(a.revenue_growth, a.ebitda_margin, strict=True):
        revenue *= 1.0 + growth
        ebitda = margin * revenue
        da = a.da_pct_revenue * revenue
        ebit = ebitda - da
        capex = a.capex_pct_revenue * revenue
        nwc_close = a.nwc_pct_revenue * revenue
        change_in_nwc = nwc_close - nwc
        nwc = nwc_close

        # Interest on opening balances (no circularity with the sweep).
        senior_interest = senior_rate * senior
        sub_interest = sub_rate * subordinated
        revolver_interest = senior_rate * revolver
        shortfall_interest = senior_rate * max(-cash, 0.0)
        total_interest = senior_interest + sub_interest + revolver_interest + shortfall_interest
        deductible = total_interest
        if a.interest_deduction_cap_pct_ebitda is not None:
            deductible = min(total_interest, a.interest_deduction_cap_pct_ebitda * max(ebitda, 0.0))
        tax = a.tax_rate * max(ebit - deductible, 0.0)

        cfads = ebitda - tax - capex - change_in_nwc
        sub_pik = sub_interest if a.subordinated_pik else 0.0
        cash_interest = total_interest - sub_pik
        mandatory = min(scheduled, senior)
        excess_cash_flow = cfads - cash_interest - mandatory
        minimum_cash = a.min_cash_pct_revenue * revenue
        # Cash above (+) or below (-) the minimum after contractual debt service.
        surplus = cash + excess_cash_flow - minimum_cash
        # A deficit is drawn on the revolver up to its limit; a surplus first
        # repays the revolver, then the sweep share prepays the senior loan.
        revolver_draw = min(max(-surplus, 0.0), max(revolver_limit - revolver, 0.0))
        revolver_repayment = min(max(surplus, 0.0), revolver)
        sweep = min(
            a.cash_sweep_pct * max(surplus - revolver_repayment, 0.0),
            senior - mandatory,
        )
        sweep = max(sweep, 0.0)
        cash_close = cash + excess_cash_flow + revolver_draw - revolver_repayment - sweep

        row = {
            "revenue": revenue,
            "revenue_growth": growth,
            "ebitda": ebitda,
            "ebitda_margin": margin,
            "depreciation_amortization": da,
            "ebit": ebit,
            "interest_expense": total_interest,
            "deductible_interest": deductible,
            "pretax_income": ebit - total_interest,
            "tax": tax,
            "net_income": ebit - total_interest - tax,
            "capex": capex,
            "net_working_capital": nwc_close,
            "change_in_nwc": change_in_nwc,
            "cash_flow_available_for_debt_service": cfads,
            "senior_opening": senior,
            "senior_interest": senior_interest,
            "senior_mandatory_amortization": mandatory,
            "senior_cash_sweep": sweep,
            "senior_closing": senior - mandatory - sweep,
            "subordinated_opening": subordinated,
            "subordinated_interest": sub_interest,
            "subordinated_pik_accrual": sub_pik,
            "subordinated_closing": subordinated + sub_pik,
            "revolver_opening": revolver,
            "revolver_interest": revolver_interest,
            "revolver_draw": revolver_draw,
            "revolver_repayment": revolver_repayment,
            "revolver_closing": revolver + revolver_draw - revolver_repayment,
            "shortfall_interest": shortfall_interest,
            "cash_interest": cash_interest,
            "total_interest": total_interest,
            "excess_cash_flow": excess_cash_flow,
            "cash_opening": cash,
            "minimum_cash": minimum_cash,
            "cash_closing": cash_close,
        }
        senior = row["senior_closing"]
        subordinated = row["subordinated_closing"]
        revolver = row["revolver_closing"]
        cash = cash_close
        tolerance = 1e-9 * max(abs(minimum_cash), 1.0)
        row["liquidity_breach"] = float(cash < minimum_cash - tolerance)
        row["total_debt_closing"] = senior + subordinated + revolver
        row["net_debt_closing"] = senior + subordinated + revolver - cash
        row["net_debt_to_ebitda"] = row["net_debt_closing"] / ebitda if ebitda > 0 else math.nan
        row["interest_coverage"] = ebitda / total_interest if total_interest > 0 else math.inf
        rows.append(row)

    exit_ev = multiple * ebitda
    exit_net_debt = senior + subordinated + revolver - cash
    exit_equity_raw = exit_ev - exit_net_debt - target.minority_interest
    return _Projection(
        rows=tuple(rows),
        exit_multiple=multiple,
        exit_ebitda=ebitda,
        exit_enterprise_value=exit_ev,
        exit_net_debt=exit_net_debt,
        exit_equity_raw=exit_equity_raw,
    )


def _returns(
    target: _Target,
    a: LBOAssumptions,
    price: float,
    exit_multiple: float | None = None,
    senior_multiple: float | None = None,
    subordinated_multiple: float | None = None,
) -> tuple[float, float]:
    """(IRR, MOIC) of a deal at ``price``; NaN when the deal is not feasible."""
    if not price > 0:
        return math.nan, math.nan
    deal = _build_deal(target, a, price, senior_multiple, subordinated_multiple)
    if deal.sponsor_equity <= 0 or deal.enterprise_value <= 0:
        return math.nan, math.nan
    projection = _project(target, a, deal, exit_multiple)
    flows = _equity_cash_flows(a, deal, projection)
    return compute_irr(flows), projection.exit_equity / deal.sponsor_equity


def _equity_cash_flows(
    a: LBOAssumptions, deal: _Deal, projection: _Projection
) -> tuple[float, ...]:
    """Sponsor cash flows t = 0..N. The model pays no interim dividends, so
    the vector is the equity cheque, zeros, and the exit proceeds."""
    flows = [-deal.sponsor_equity] + [0.0] * a.holding_period
    flows[-1] += projection.exit_equity
    return tuple(flows)


def _solve_price(target: _Target, a: LBOAssumptions, target_irr: float) -> tuple[float, str]:
    """Price per share at which the sponsor earns exactly ``target_irr``.

    Root of NPV(price) = -sponsor equity + exit equity / (1 + IRR)^N, which is
    zero exactly when the deal IRR equals the target. The NPV form avoids a
    nested root search and is continuous in the price (also where the sizing
    caps start to bind). Returns (price, status).
    """
    discount = (1.0 + target_irr) ** a.holding_period

    def npv(price: float) -> float:
        deal = _build_deal(target, a, price)
        projection = _project(target, a, deal)
        return -deal.sponsor_equity + projection.exit_equity_raw / discount

    if npv(0.0) <= 0:
        return 0.0, STATUS_UNREACHABLE
    high = target.price
    f_high = npv(high)
    doublings = 0
    while f_high > 0 and doublings < 40:
        high *= 2.0
        f_high = npv(high)
        doublings += 1
    if f_high > 0 or not math.isfinite(f_high):
        return math.nan, STATUS_UNBOUNDED
    price = float(brentq(npv, 0.0, high, xtol=1e-12 * high, rtol=1e-14, maxiter=500))
    return price, STATUS_SOLVED


# --------------------------------------------------------------------------
# Default assumptions
# --------------------------------------------------------------------------


def _default_assumptions_with_notes(
    company: CompanyData, params: MarketParameters
) -> tuple[LBOAssumptions, list[str]]:
    target = _extract_target(company)
    statements = company.statements
    income, balance, cash_flow = statements.income, statements.balance, statements.cash_flow
    revenue = _column(income, "revenue")
    notes: list[str] = []
    years = 5

    # Revenue growth: historical CAGR, bounded, fading linearly to long-term growth.
    positive = revenue[revenue > 0].dropna()
    start_growth = params.long_term_growth
    if len(positive) >= 2:
        span = int(positive.index[-1]) - int(positive.index[0])
        if span >= 1:
            cagr = (positive.iloc[-1] / positive.iloc[0]) ** (1.0 / span) - 1.0
            start_growth = _clip(float(cagr), *_GROWTH_BOUNDS)
            if not math.isclose(start_growth, float(cagr)):
                notes.append(
                    f"historical revenue CAGR {cagr:.1%} bounded to {start_growth:.1%} "
                    "as starting growth"
                )
    else:
        notes.append("revenue history too short for a CAGR; long-term growth used")
    growth = tuple(
        start_growth + (params.long_term_growth - start_growth) * t / years
        for t in range(1, years + 1)
    )

    # EBITDA margin: latest year, reverting to the historical median within a band.
    latest_margin = target.ebitda / target.revenue
    median_margin = _median_ratio(_column(income, "ebitda"), revenue)
    low, high = (band * latest_margin for band in _MARGIN_REVERSION_BAND)
    end_margin = latest_margin
    if math.isfinite(median_margin):
        end_margin = _clip(median_margin, low, high)
    margin = tuple(
        latest_margin + (end_margin - latest_margin) * t / years for t in range(1, years + 1)
    )

    # D&A: income statement, then cash flow statement, then a flat fallback.
    da_pct = _median_ratio(_column(income, "depreciation_amortization"), revenue)
    if not math.isfinite(da_pct):
        da_pct = _median_ratio(_column(cash_flow, "depreciation_amortization"), revenue)
    if not math.isfinite(da_pct):
        da_pct = _FALLBACK_DA_PCT
        notes.append(f"D&A missing; assumed {_FALLBACK_DA_PCT:.0%} of revenue")
    da_pct = _clip(da_pct, 0.0, 0.5)

    # Capex: historical median; missing -> equal to D&A (steady state).
    capex_pct = _median_ratio(_column(cash_flow, "capex"), revenue)
    if not math.isfinite(capex_pct):
        capex_pct = da_pct
        notes.append("capex missing; assumed equal to D&A")
    capex_pct = _clip(capex_pct, 0.0, 0.5)

    # Operating NWC from the balance sheet (never Yahoo's change_in_nwc).
    nwc_pct = _median_ratio(_column(balance, "net_working_capital"), revenue)
    if not math.isfinite(nwc_pct):
        nwc_pct = 0.0
        notes.append("net working capital missing; assumed zero (no NWC investment)")
    nwc_pct = _clip(nwc_pct, -0.5, 1.0)

    assumptions = LBOAssumptions(
        revenue_growth=growth,
        ebitda_margin=margin,
        da_pct_revenue=da_pct,
        capex_pct_revenue=capex_pct,
        nwc_pct_revenue=nwc_pct,
        tax_rate=params.tax_rate,
        risk_free_rate=params.risk_free_rate,
    )
    return assumptions, notes


def default_assumptions(
    company: CompanyData, params: MarketParameters = _DEFAULT_PARAMS
) -> LBOAssumptions:
    """History-based default assumptions for a five-year LBO.

    * Revenue growth: historical revenue CAGR over the available years,
      bounded to [-5 %, +10 %], fading linearly to ``params.long_term_growth``
      in year 5. Too little history -> long-term growth throughout.
    * EBITDA margin: latest-year margin (the basis of the entry multiple),
      moving linearly to the historical median margin by year 5; the median is
      bounded to 75-125 % of the latest margin so that one weak or strong year
      cannot dominate the case.
    * D&A, capex and operating NWC: historical medians as a share of revenue.
      Fallbacks: D&A missing -> 3 % of revenue; capex missing -> equal to
      D&A; NWC missing -> zero.
    * Tax rate and risk-free rate from ``params``; financing terms as in
      :class:`LBOAssumptions`.

    Raises ``NotApplicableError`` for companies an LBO cannot be run on.
    """
    assumptions, _ = _default_assumptions_with_notes(company, params)
    return assumptions


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def _sources_and_uses(target: _Target, deal: _Deal) -> pd.DataFrame:
    sources = [
        ("Senior term loan", deal.senior),
        ("Subordinated debt", deal.subordinated),
        ("Existing cash on balance sheet", target.cash),
        ("Sponsor equity", deal.sponsor_equity),
    ]
    uses = [
        ("Purchase of equity", deal.equity_purchase),
        ("Refinancing of existing debt", target.total_debt),
        ("Redemption of preferred equity", target.preferred_equity),
        ("Transaction fees", deal.transaction_fees),
        ("Financing fees", deal.financing_fees),
        ("Minimum cash to balance sheet", deal.minimum_cash),
    ]
    records: list[dict[str, Any]] = []
    for side, items in (("sources", sources), ("uses", uses)):
        total = sum(amount for _, amount in items)
        for item, amount in items:
            records.append(
                {
                    "side": side,
                    "item": item,
                    "amount": amount,
                    "pct": amount / total if total else math.nan,
                }
            )
    return pd.DataFrame.from_records(records, columns=["side", "item", "amount", "pct"])


def _attribution(target: _Target, deal: _Deal, projection: _Projection) -> pd.DataFrame:
    """Standard PE value bridge; the components sum to total value creation.

    sponsor equity = entry EV + fees - net debt after closing, and
    exit equity = exit EV - net debt at exit (net debt including minorities),
    so exit equity - sponsor equity = EBITDA growth at the entry multiple
    + multiple change on exit EBITDA + reduction of net debt - fees.

    ``share`` is the amount divided by the ABSOLUTE total value creation, so a
    share keeps the sign of its amount: a component that destroys value is
    negative also when the deal as a whole loses money. The shares sum to +1
    for a gain and to -1 for a loss (the total row shows that); NaN when the
    total is exactly zero.
    """
    entry_net_debt = deal.senior + deal.subordinated - deal.minimum_cash
    components = {
        "ebitda_growth": (projection.exit_ebitda - target.ebitda) * deal.entry_multiple,
        "multiple_expansion": (projection.exit_multiple - deal.entry_multiple)
        * projection.exit_ebitda,
        "debt_paydown": entry_net_debt - projection.exit_net_debt,
        "fees": -(deal.transaction_fees + deal.financing_fees),
        "equity_floor": projection.exit_equity - projection.exit_equity_raw,
    }
    total = projection.exit_equity - deal.sponsor_equity
    components["total_value_creation"] = total
    frame = pd.DataFrame({"amount": pd.Series(components, dtype=float)})
    # Relative to the absolute total: with a negative total, dividing by the signed
    # total would show value-destroying components as positive shares.
    frame["share"] = frame["amount"] / abs(total) if total != 0 else math.nan
    frame.index.name = "component"
    return frame


def _sensitivities(
    target: _Target, a: LBOAssumptions, deal: _Deal, exit_multiple: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    entry_multiples = [deal.entry_multiple * step for step in _SENSITIVITY_STEPS]
    exit_multiples = [exit_multiple * step for step in _SENSITIVITY_STEPS]
    columns = pd.Index([round(m, 2) for m in exit_multiples], name="exit_ev_ebitda")

    irr = np.full((len(entry_multiples), len(exit_multiples)), np.nan)
    moic = np.full_like(irr, np.nan)
    for i, entry in enumerate(entry_multiples):
        price = (entry * target.ebitda - target.net_claims) / target.shares
        for j, exit_ in enumerate(exit_multiples):
            irr[i, j], moic[i, j] = _returns(target, a, price, exit_)
    index = pd.Index([round(m, 2) for m in entry_multiples], name="entry_ev_ebitda")
    irr_frame = pd.DataFrame(irr, index=index, columns=columns)
    moic_frame = pd.DataFrame(moic, index=index, columns=columns)

    requested = a.senior_debt_multiple + a.subordinated_debt_multiple
    senior_weight = a.senior_debt_multiple / requested if requested > 0 else 1.0
    levels = sorted({max(requested + step, 0.0) for step in _LEVERAGE_STEPS})
    lev = np.full((len(levels), len(exit_multiples)), np.nan)
    for i, level in enumerate(levels):
        for j, exit_ in enumerate(exit_multiples):
            lev[i, j], _ = _returns(
                target,
                a,
                deal.price,
                exit_,
                senior_multiple=level * senior_weight,
                subordinated_multiple=level * (1.0 - senior_weight),
            )
    lev_frame = pd.DataFrame(
        lev,
        index=pd.Index([round(x, 2) for x in levels], name="requested_debt_to_ebitda"),
        columns=columns,
    )
    return irr_frame, moic_frame, lev_frame


_FORECAST_COLUMNS: tuple[str, ...] = (
    "revenue",
    "revenue_growth",
    "ebitda",
    "ebitda_margin",
    "depreciation_amortization",
    "ebit",
    "interest_expense",
    "deductible_interest",
    "pretax_income",
    "tax",
    "net_income",
    "capex",
    "net_working_capital",
    "change_in_nwc",
    "cash_flow_available_for_debt_service",
)

_DEBT_COLUMNS: tuple[str, ...] = (
    "cash_flow_available_for_debt_service",
    "senior_opening",
    "senior_interest",
    "senior_mandatory_amortization",
    "senior_cash_sweep",
    "senior_closing",
    "subordinated_opening",
    "subordinated_interest",
    "subordinated_pik_accrual",
    "subordinated_closing",
    "revolver_opening",
    "revolver_interest",
    "revolver_draw",
    "revolver_repayment",
    "revolver_closing",
    "shortfall_interest",
    "total_interest",
    "cash_interest",
    "excess_cash_flow",
    "cash_opening",
    "minimum_cash",
    "cash_closing",
    "total_debt_closing",
    "net_debt_closing",
    "net_debt_to_ebitda",
    "interest_coverage",
    "liquidity_breach",
)


def run_lbo(
    company: CompanyData,
    assumptions: LBOAssumptions | None = None,
    params: MarketParameters | None = None,
) -> LBOResult:
    """Run the LBO for ``company``.

    ``assumptions=None`` uses :func:`default_assumptions` with ``params`` (tax
    rate, risk-free rate, long-term growth; ``None`` means ``MarketParameters()``).
    ``params`` is ignored when ``assumptions`` are given. Raises
    ``NotApplicableError`` (with the reason) when an LBO is not meaningful:
    financial or real estate company, fewer than two fiscal years, missing or
    non-positive revenue or EBITDA, missing market data, a non-positive
    entry enterprise value / equity cheque (the target's own cash would pay
    for the acquisition), or no debt capacity at all under the sizing rule.
    """
    target = _extract_target(company)
    warnings: list[str] = []
    if assumptions is None:
        a, notes = _default_assumptions_with_notes(
            company, params if params is not None else _DEFAULT_PARAMS
        )
        warnings.extend(notes)
    else:
        a = assumptions

    price = (
        a.entry_price_per_share
        if a.entry_price_per_share is not None
        else target.price * (1.0 + a.takeover_premium)
    )
    if not (math.isfinite(price) and price > 0):
        raise NotApplicableError(f"{target.ticker}: entry price per share is not positive")
    deal = _build_deal(target, a, price)
    if deal.enterprise_value <= 0:
        raise NotApplicableError(
            f"{target.ticker}: entry enterprise value is not positive "
            "(net cash exceeds the purchase price of the equity)"
        )
    if deal.sponsor_equity <= 0:
        raise NotApplicableError(
            f"{target.ticker}: no sponsor equity required at the entry price "
            "(the target's own cash funds the acquisition)"
        )

    requested_debt = a.senior_debt_multiple + a.subordinated_debt_multiple
    if requested_debt > 0 and deal.senior + deal.subordinated <= 0:
        raise NotApplicableError(
            f"{target.ticker}: no debt capacity (EBITDA does not cover capex, or the "
            "coverage and equity constraints leave no room for debt)"
        )

    projection = _project(target, a, deal)
    years = [target.latest_year + t for t in range(1, a.holding_period + 1)]
    table = pd.DataFrame.from_records(projection.rows, index=pd.Index(years, name="fiscal_year"))
    forecast = table[list(_FORECAST_COLUMNS)].copy()
    debt_schedule = table[list(_DEBT_COLUMNS)].copy()
    debt_schedule["liquidity_breach"] = debt_schedule["liquidity_breach"].astype(bool)
    breach_years = [int(y) for y in debt_schedule.index[debt_schedule["liquidity_breach"]]]
    if breach_years:
        warnings.append(
            "liquidity breach: cash below the minimum balance after contractual debt "
            f"service in {breach_years}"
        )

    flows = _equity_cash_flows(a, deal, projection)
    irr = compute_irr(flows)
    moic = projection.exit_equity / deal.sponsor_equity
    wiped_out = projection.exit_equity_raw <= 0
    if wiped_out:
        warnings.append("sponsor equity is wiped out at exit (exit EV below net debt)")
    warnings.extend(deal.notes)

    irr_entry_exit, moic_entry_exit, irr_leverage_exit = _sensitivities(
        target, a, deal, projection.exit_multiple
    )

    ability: dict[float, float] = {}
    status: dict[float, str] = {}
    for target_irr in (a.target_irr_conservative, a.target_irr_base, a.target_irr_aggressive):
        solved_price, solved_status = _solve_price(target, a, target_irr)
        if solved_status == STATUS_SOLVED:
            check = _build_deal(target, a, solved_price)
            if solved_price <= 0 or check.sponsor_equity <= 0 or check.enterprise_value <= 0:
                solved_price, solved_status = 0.0, STATUS_UNREACHABLE
        ability[target_irr] = solved_price
        status[target_irr] = solved_status
        if solved_status == STATUS_UNREACHABLE:
            warnings.append(
                f"ability to pay: no positive price delivers a {target_irr:.0%} IRR "
                "(price set to zero)"
            )
        elif solved_status == STATUS_UNBOUNDED:
            warnings.append(
                f"ability to pay: a {target_irr:.0%} IRR is reached at every price tested; "
                "no finite price (EBITDA growth at or above the target IRR with the exit "
                "multiple tied to the entry multiple)"
            )

    total_debt = deal.senior + deal.subordinated
    return LBOResult(
        ticker=target.ticker,
        currency=target.currency,
        assumptions=a,
        market_price=target.price,
        shares_outstanding=target.shares,
        entry_price_per_share=price,
        entry_equity_value=deal.equity_purchase,
        entry_enterprise_value=deal.enterprise_value,
        entry_ebitda=target.ebitda,
        entry_multiple=deal.entry_multiple,
        sources_and_uses=_sources_and_uses(target, deal),
        senior_debt=deal.senior,
        subordinated_debt=deal.subordinated,
        sponsor_equity=deal.sponsor_equity,
        leverage=total_debt / target.ebitda,
        equity_share=deal.sponsor_equity / (total_debt + deal.sponsor_equity),
        sizing_notes=list(deal.notes),
        forecast=forecast,
        debt_schedule=debt_schedule,
        exit_year=years[-1],
        exit_multiple=projection.exit_multiple,
        exit_ebitda=projection.exit_ebitda,
        exit_enterprise_value=projection.exit_enterprise_value,
        exit_net_debt=projection.exit_net_debt,
        exit_minority_interest=target.minority_interest,
        exit_equity_value=projection.exit_equity,
        equity_cash_flows=flows,
        irr=irr,
        moic=moic,
        attribution=_attribution(target, deal, projection),
        sensitivity_irr_entry_exit=irr_entry_exit,
        sensitivity_moic_entry_exit=moic_entry_exit,
        sensitivity_irr_leverage_exit=irr_leverage_exit,
        ability_to_pay=ability,
        ability_to_pay_status=status,
        liquidity_breach_years=breach_years,
        equity_wiped_out=wiped_out,
        warnings=warnings,
    )


def with_entry_price(assumptions: LBOAssumptions, price_per_share: float) -> LBOAssumptions:
    """Copy of ``assumptions`` with an explicit offer price per share."""
    return replace(assumptions, entry_price_per_share=price_per_share)
