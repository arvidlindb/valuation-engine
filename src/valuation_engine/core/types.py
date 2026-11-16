"""Shared data contract for the valuation engine.

Every module (data, models, viz, export, app) builds against the types in this
file, so change it with care.

Conventions
-----------
* All monetary values are absolute amounts (not thousands or millions).
* All monetary values of one company are expressed in ONE currency: the
  trading currency of the listing (normally SEK). The data layer converts
  statements reported in another currency (e.g. USD or EUR) before handing
  them to the models, and records the original currency and the FX rate.
* Statement DataFrames are indexed by fiscal year (int, ascending) and use the
  standardized column names listed in ``INCOME_COLUMNS``, ``BALANCE_COLUMNS``
  and ``CASH_FLOW_COLUMNS``. A missing item is ``NaN``, never a missing column.
* Sign convention: expenses, capex, D&A, interest and taxes are stored as
  POSITIVE numbers. ``change_in_nwc`` is positive when working capital grows
  (a cash outflow).
* Rates and margins are decimals (0.08 means 8 %).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from typing import Any

import pandas as pd

# --------------------------------------------------------------------------
# Standardized line items
# --------------------------------------------------------------------------

INCOME_COLUMNS: tuple[str, ...] = (
    "revenue",
    "cogs",
    "gross_profit",
    "ebitda",
    "depreciation_amortization",
    "ebit",
    "interest_expense",
    "pretax_income",
    "income_tax",
    "net_income",  # attributable to common shareholders
    "diluted_shares",  # weighted average, all share classes
)

BALANCE_COLUMNS: tuple[str, ...] = (
    "cash",  # cash, cash equivalents and short-term investments
    "accounts_receivable",
    "inventory",
    "accounts_payable",
    "total_current_assets",
    "total_current_liabilities",
    "net_ppe",
    "goodwill_intangibles",
    "total_assets",
    "total_debt",  # interest-bearing debt including lease liabilities
    "total_liabilities",
    "minority_interest",
    "total_equity",  # equity attributable to shareholders of the parent
    "net_working_capital",  # operating: receivables + inventory - payables
)

CASH_FLOW_COLUMNS: tuple[str, ...] = (
    "cash_from_operations",
    "depreciation_amortization",
    "change_in_nwc",
    "capex",  # tangible + intangible investments, positive number
    "dividends_paid",
    "free_cash_flow",  # cash_from_operations - capex
)

#: Yahoo sectors for which an FCFF-based DCF and an LBO are not meaningful.
FINANCIAL_SECTORS: frozenset[str] = frozenset({"Financial Services", "Real Estate"})


# --------------------------------------------------------------------------
# Company data
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CompanyProfile:
    """Static description of a listed company."""

    ticker: str  # Yahoo ticker of the chosen primary share class, e.g. "ATCO-A.ST"
    name: str
    sector: str | None = None
    industry: str | None = None
    country: str | None = None
    currency: str = "SEK"  # trading currency; all amounts are in this currency
    reporting_currency: str = "SEK"  # currency of the original statements
    fx_rate: float = 1.0  # reporting currency -> trading currency, as applied

    @property
    def is_financial(self) -> bool:
        """True for banks, insurers, investment and real estate companies."""
        return self.sector in FINANCIAL_SECTORS


@dataclass
class FinancialStatements:
    """Annual statements with standardized line items (see module docstring)."""

    ticker: str
    currency: str
    income: pd.DataFrame
    balance: pd.DataFrame
    cash_flow: pd.DataFrame
    #: Standardized item -> source tag actually used (documents fallbacks).
    source_map: dict[str, str] = field(default_factory=dict)

    @property
    def fiscal_years(self) -> list[int]:
        """Fiscal years covered, ascending."""
        return [int(y) for y in self.income.index]

    @property
    def latest_year(self) -> int:
        """Most recent fiscal year."""
        return self.fiscal_years[-1]

    def latest(self, statement: str, item: str) -> float:
        """Return ``item`` of ``statement`` ('income', 'balance', 'cash_flow')
        for the most recent fiscal year (may be NaN)."""
        frame: pd.DataFrame = getattr(self, statement)
        return float(frame[item].iloc[-1])


@dataclass
class MarketData:
    """Market snapshot. Company-level figures cover ALL share classes."""

    ticker: str
    as_of: date
    currency: str
    price: float  # price of the primary share class
    shares_outstanding: float  # market_cap / price (all classes, primary-class equivalents)
    market_cap: float
    total_debt: float
    cash: float
    minority_interest: float = 0.0
    preferred_equity: float = 0.0
    beta: float | None = None  # provider beta, informational only

    @property
    def net_debt(self) -> float:
        """Total debt less cash."""
        return self.total_debt - self.cash

    @property
    def enterprise_value(self) -> float:
        """Market cap + net debt + minority interest + preferred equity."""
        return self.market_cap + self.net_debt + self.minority_interest + self.preferred_equity


@dataclass
class PriceHistory:
    """Aligned price series used for the beta regression.

    ``prices`` has a DatetimeIndex and the columns ``stock`` and ``benchmark``
    (adjusted closes, same frequency, no missing values).
    """

    ticker: str
    benchmark: str  # e.g. "^OMXSPI"
    frequency: str  # "W" or "M"
    prices: pd.DataFrame


@dataclass
class CompanyData:
    """Everything the models need for one company."""

    profile: CompanyProfile
    statements: FinancialStatements
    market: MarketData
    price_history: PriceHistory | None = None

    @property
    def ticker(self) -> str:
        return self.profile.ticker


@dataclass(frozen=True)
class PeerSnapshot:
    """One row of the peer universe used by comps and by beta relevering.

    Amounts are in SEK so that companies are comparable for clustering.
    Ratios that cannot be computed are ``None``.
    """

    ticker: str
    name: str
    sector: str | None
    industry: str | None
    market_cap: float
    enterprise_value: float
    revenue: float | None
    ebitda: float | None
    ebit: float | None
    net_income: float | None
    total_debt: float | None = None
    revenue_growth: float | None = None  # latest year over year
    ebitda_margin: float | None = None
    return_on_equity: float | None = None
    beta: float | None = None

    @property
    def is_financial(self) -> bool:
        return self.sector in FINANCIAL_SECTORS

    @property
    def ev_ebitda(self) -> float | None:
        return _ratio(self.enterprise_value, self.ebitda)

    @property
    def ev_sales(self) -> float | None:
        return _ratio(self.enterprise_value, self.revenue)

    @property
    def pe(self) -> float | None:
        return _ratio(self.market_cap, self.net_income)

    @property
    def debt_to_equity(self) -> float | None:
        if self.total_debt is None or self.market_cap <= 0:
            return None
        return self.total_debt / self.market_cap


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    """Return numerator / denominator, or None when not meaningful (<= 0)."""
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


# --------------------------------------------------------------------------
# Assumptions
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketParameters:
    """Market-wide inputs for the Swedish market. Documented in PLAN.md."""

    # Riksbank SEGVB10YC (10-year government bond) was 3.075 % on 2026-11-13.
    risk_free_rate: float = 0.03075
    # Damodaran country ERP for Sweden (Aaa, no country premium), January 2026.
    equity_risk_premium: float = 0.0423
    tax_rate: float = 0.206  # Swedish corporate tax rate
    benchmark: str = "^OMXSPI"
    long_term_growth: float = 0.02  # Riksbank inflation target as nominal anchor


@dataclass(frozen=True)
class DCFAssumptions:
    """Drivers of the FCFF forecast. Tuples have one entry per forecast year."""

    revenue_growth: tuple[float, ...]
    ebitda_margin: tuple[float, ...]
    da_pct_revenue: float
    capex_pct_revenue: float
    nwc_pct_revenue: float  # operating NWC as share of revenue
    tax_rate: float = 0.206
    wacc: float | None = None  # None -> computed via CAPM
    terminal_growth: float = 0.02
    exit_multiple: float | None = None  # EV/EBITDA; None -> peer/own multiple
    mid_year: bool = True

    def __post_init__(self) -> None:
        if len(self.revenue_growth) != len(self.ebitda_margin):
            raise ValueError("revenue_growth and ebitda_margin must have equal length")
        if not self.revenue_growth:
            raise ValueError("at least one forecast year is required")

    @property
    def forecast_years(self) -> int:
        return len(self.revenue_growth)


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


class ValuationMethod(StrEnum):
    """Identifier of a valuation method (one bar in the football field)."""

    DCF_GORDON = "dcf_gordon"
    DCF_EXIT_MULTIPLE = "dcf_exit_multiple"
    COMPS_EV_EBITDA = "comps_ev_ebitda"
    COMPS_EV_SALES = "comps_ev_sales"
    COMPS_PE = "comps_pe"
    LBO = "lbo"
    MONTE_CARLO = "monte_carlo"
    MARKET = "market"  # e.g. 52-week trading range


@dataclass
class ValuationResult:
    """Value range per share from one method, in the trading currency."""

    method: ValuationMethod
    label: str  # human-readable, e.g. "DCF - Gordon growth"
    low: float
    base: float
    high: float
    currency: str = "SEK"
    enterprise_value: float | None = None  # base case
    equity_value: float | None = None  # base case
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (self.low <= self.base <= self.high):
            raise ValueError(
                f"{self.label}: expected low <= base <= high, "
                f"got {self.low}, {self.base}, {self.high}"
            )


class NotApplicableError(ValueError):
    """Raised when a method cannot be applied to a company (e.g. DCF on a
    bank, or negative EBITDA for an LBO). The message states the reason."""
