"""Standardization: Yahoo row labels -> the line items of ``core.types``.

Each standardized item has an ordered chain of :class:`Candidate` formulas.
The first candidate that yields a value for a fiscal year wins *for that
year*; later candidates only fill the remaining gaps. The tags that were
actually used are recorded in ``FinancialStatements.source_map``.

Tag syntax (also used in ``source_map``): ``is:`` income statement, ``bs:``
balance sheet, ``cf:`` cash flow statement, ``std:`` an already standardized
item. ``-`` in front of a term flips Yahoo's sign, ``?`` after a term marks it
optional (missing -> 0), ``abs(...)`` forces a positive number.

Sign conventions follow ``core.types``: expenses, D&A, interest, taxes, capex
and dividends are positive; ``change_in_nwc`` is positive when working
capital grows (the opposite of Yahoo's cash-flow sign).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from ..core.types import BALANCE_COLUMNS, CASH_FLOW_COLUMNS, INCOME_COLUMNS, FinancialStatements

#: Standardized columns that are NOT monetary (never FX-converted).
NON_MONETARY: frozenset[str] = frozenset({"diluted_shares"})

_STATEMENT_KEYS: dict[str, str] = {"is": "income_stmt", "bs": "balance_sheet", "cf": "cashflow"}


@dataclass(frozen=True)
class Term:
    """One addend of a candidate formula."""

    source: str  # "is", "bs", "cf" or "std:<statement>" for standardized items
    label: str
    sign: float = 1.0
    optional: bool = False  # missing/NaN counts as zero

    @property
    def tag(self) -> str:
        prefix = "-" if self.sign < 0 else ""
        suffix = "?" if self.optional else ""
        source = "std" if self.source.startswith("std") else self.source
        return f"{prefix}{source}:{self.label}{suffix}"


@dataclass(frozen=True)
class Candidate:
    """A formula: the signed sum of its terms (optionally in absolute value)."""

    terms: tuple[Term, ...]
    absolute: bool = False

    @property
    def tag(self) -> str:
        parts: list[str] = []
        for index, term in enumerate(self.terms):
            tag = term.tag
            if index and not tag.startswith("-"):
                tag = "+ " + tag
            elif index:
                tag = "- " + tag[1:]
            parts.append(tag)
        text = " ".join(parts)
        return f"abs({text})" if self.absolute else text


def _c(*terms: Term, absolute: bool = False) -> Candidate:
    if all(term.optional for term in terms):
        raise ValueError("a candidate needs at least one required term")
    return Candidate(tuple(terms), absolute)


def _is(label: str, sign: float = 1.0, optional: bool = False) -> Term:
    return Term("is", label, sign, optional)


def _bs(label: str, sign: float = 1.0, optional: bool = False) -> Term:
    return Term("bs", label, sign, optional)


def _cf(label: str, sign: float = 1.0, optional: bool = False) -> Term:
    return Term("cf", label, sign, optional)


def _std(statement: str, item: str, sign: float = 1.0, optional: bool = False) -> Term:
    return Term(f"std:{statement}", item, sign, optional)


# --------------------------------------------------------------------------
# Fallback chains. ORDER MATTERS, both within a chain and between items
# (``std:`` terms need the referenced item to be computed first).
# --------------------------------------------------------------------------

_DA_CHAIN: tuple[Candidate, ...] = (
    _c(_is("Reconciled Depreciation"), absolute=True),
    _c(_cf("Depreciation And Amortization"), absolute=True),
    _c(_cf("Depreciation Amortization Depletion"), absolute=True),
    _c(_is("Depreciation And Amortization In Income Statement"), absolute=True),
    _c(_is("Depreciation Amortization Depletion Income Statement"), absolute=True),
    _c(_is("EBITDA"), _is("EBIT", -1.0)),
)

INCOME_CHAINS: dict[str, tuple[Candidate, ...]] = {
    "revenue": (
        _c(_is("Total Revenue")),
        _c(_is("Operating Revenue")),
    ),
    "cogs": (
        _c(_is("Cost Of Revenue")),
        _c(_is("Reconciled Cost Of Revenue")),
        _c(_std("income", "revenue"), _is("Gross Profit", -1.0)),
    ),
    "gross_profit": (
        _c(_is("Gross Profit")),
        _c(_std("income", "revenue"), _std("income", "cogs", -1.0)),
    ),
    "depreciation_amortization": _DA_CHAIN,
    "ebit": (
        _c(_is("EBIT")),
        _c(_is("Operating Income")),
        _c(_is("Total Operating Income As Reported")),
        # Deliberately no "Pretax Income + Interest Expense" fallback: the only
        # companies without an EBIT row are banks and investment companies,
        # for which that sum is meaningless.
    ),
    "ebitda": (
        _c(_is("EBITDA")),
        _c(_is("Normalized EBITDA")),
        _c(_std("income", "ebit"), _std("income", "depreciation_amortization")),
        _c(_is("Operating Income"), _std("income", "depreciation_amortization")),
    ),
    "interest_expense": (
        _c(_is("Interest Expense"), absolute=True),
        _c(_is("Interest Expense Non Operating"), absolute=True),
    ),
    "pretax_income": (
        _c(_is("Pretax Income")),
        _c(_is("Net Income Including Noncontrolling Interests"), _is("Tax Provision")),
    ),
    "income_tax": (_c(_is("Tax Provision")),),
    "net_income": (
        _c(_is("Net Income Common Stockholders")),
        _c(_is("Net Income")),
        _c(_is("Net Income From Continuing Operation Net Minority Interest")),
        _c(
            _is("Net Income Including Noncontrolling Interests"),
            _is("Minority Interests", optional=True),
        ),
    ),
    "diluted_shares": (
        _c(_is("Diluted Average Shares")),
        _c(_is("Basic Average Shares")),
        _c(_bs("Ordinary Shares Number")),
        _c(_bs("Share Issued")),
    ),
}

BALANCE_CHAINS: dict[str, tuple[Candidate, ...]] = {
    "cash": (
        _c(_bs("Cash Cash Equivalents And Short Term Investments")),
        _c(_bs("Cash And Cash Equivalents"), _bs("Other Short Term Investments", optional=True)),
        _c(_bs("Cash Financial"), _bs("Cash Equivalents", optional=True)),
        _c(_bs("Cash Cash Equivalents And Federal Funds Sold")),
        _c(_bs("Cash And Due From Banks")),
    ),
    "accounts_receivable": (
        _c(_bs("Accounts Receivable")),
        _c(
            _bs("Gross Accounts Receivable"),
            _bs("Allowance For Doubtful Accounts Receivable", optional=True),
        ),
        _c(_bs("Receivables")),
    ),
    "inventory": (_c(_bs("Inventory")),),
    "accounts_payable": (
        _c(_bs("Accounts Payable")),
        _c(_bs("Payables")),
        _c(_bs("Payables And Accrued Expenses")),
    ),
    "total_current_assets": (_c(_bs("Current Assets")),),
    "total_current_liabilities": (_c(_bs("Current Liabilities")),),
    "net_ppe": (
        _c(_bs("Net PPE")),
        _c(_bs("Gross PPE"), _bs("Accumulated Depreciation")),
    ),
    "goodwill_intangibles": (
        _c(_bs("Goodwill And Other Intangible Assets")),
        _c(_bs("Goodwill"), _bs("Other Intangible Assets", optional=True)),
        _c(_bs("Other Intangible Assets")),
    ),
    "total_assets": (_c(_bs("Total Assets")),),
    "total_debt": (
        # Yahoo's "Total Debt" includes lease liabilities (verified, see docs).
        _c(_bs("Total Debt")),
        _c(
            _bs("Long Term Debt And Capital Lease Obligation"),
            _bs("Current Debt And Capital Lease Obligation", optional=True),
        ),
        _c(_bs("Current Debt And Capital Lease Obligation")),
        _c(
            _bs("Long Term Debt"),
            _bs("Current Debt", optional=True),
            _bs("Capital Lease Obligations", optional=True),
        ),
        _c(_bs("Current Debt"), _bs("Capital Lease Obligations", optional=True)),
        _c(_bs("Capital Lease Obligations")),
    ),
    "total_liabilities": (
        _c(_bs("Total Liabilities Net Minority Interest")),
        _c(_bs("Total Assets"), _bs("Total Equity Gross Minority Interest", -1.0)),
    ),
    "minority_interest": (
        _c(_bs("Minority Interest")),
        _c(_bs("Total Equity Gross Minority Interest"), _bs("Stockholders Equity", -1.0)),
    ),
    "total_equity": (
        _c(_bs("Stockholders Equity")),
        _c(_bs("Common Stock Equity")),
        _c(
            _bs("Total Equity Gross Minority Interest"),
            _bs("Minority Interest", -1.0, optional=True),
        ),
    ),
    "net_working_capital": (
        _c(
            _std("balance", "accounts_receivable"),
            _std("balance", "inventory", optional=True),
            _std("balance", "accounts_payable", -1.0),
        ),
    ),
}

CASH_FLOW_CHAINS: dict[str, tuple[Candidate, ...]] = {
    "cash_from_operations": (
        _c(_cf("Operating Cash Flow")),
        _c(_cf("Cash Flow From Continuing Operating Activities")),
    ),
    "depreciation_amortization": (
        _c(_cf("Depreciation And Amortization"), absolute=True),
        _c(_cf("Depreciation Amortization Depletion"), absolute=True),
        _c(_std("income", "depreciation_amortization")),
    ),
    "change_in_nwc": (
        # Yahoo: positive = cash inflow = working capital released. We flip it.
        _c(_cf("Change In Working Capital", -1.0)),
        _c(
            _cf("Change In Receivables", -1.0),
            _cf("Change In Inventory", -1.0, optional=True),
            _cf("Change In Payables And Accrued Expense", -1.0, optional=True),
            _cf("Change In Other Working Capital", -1.0, optional=True),
        ),
        # Last resort: year-over-year change of the balance-sheet NWC
        # (handled specially in ``_evaluate``; first year stays NaN).
        _c(_std("balance", "net_working_capital:diff")),
    ),
    "capex": (
        # Yahoo's "Capital Expenditure" already contains purchases of
        # intangibles (verified, see docs) -> never add them on top.
        _c(_cf("Capital Expenditure"), absolute=True),
        _c(_cf("Purchase Of PPE"), _cf("Purchase Of Intangibles", optional=True), absolute=True),
        _c(_cf("Capital Expenditure Reported"), absolute=True),
        _c(
            _cf("Net PPE Purchase And Sale"),
            _cf("Net Intangibles Purchase And Sale", optional=True),
            absolute=True,
        ),
        _c(_cf("Purchase Of Intangibles"), absolute=True),
    ),
    "dividends_paid": (
        _c(_cf("Cash Dividends Paid"), absolute=True),
        _c(
            _cf("Common Stock Dividend Paid"),
            _cf("Preferred Stock Dividend Paid", optional=True),
            absolute=True,
        ),
    ),
    "free_cash_flow": (
        _c(_std("cash_flow", "cash_from_operations"), _std("cash_flow", "capex", -1.0)),
        _c(_cf("Free Cash Flow")),
    ),
}


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def raw_statement_frame(raw: dict[str, Any] | None) -> pd.DataFrame:
    """Raw statement -> DataFrame indexed by period end (ascending), one
    column per Yahoo label."""
    if not raw or not raw.get("periods"):
        return pd.DataFrame(index=pd.DatetimeIndex([], name="period"))
    index = pd.DatetimeIndex(pd.to_datetime(raw["periods"]), name="period")
    frame = pd.DataFrame(raw.get("rows") or {}, index=index, dtype=float)
    return frame.sort_index()


def _by_fiscal_year(frame: pd.DataFrame) -> pd.DataFrame:
    """Index a period-end frame by fiscal year (calendar year of the period
    end). If two periods end in the same calendar year (changed fiscal year
    end) the later one wins."""
    if frame.empty:
        return pd.DataFrame(index=pd.Index([], dtype=int, name="fiscal_year"))
    out = frame.copy()
    out.index = pd.Index([int(ts.year) for ts in frame.index], dtype=int, name="fiscal_year")
    return out[~out.index.duplicated(keep="last")].sort_index()


class _Context:
    """Raw frames by fiscal year plus the standardized items computed so far."""

    def __init__(self, raw: dict[str, pd.DataFrame], years: pd.Index) -> None:
        self.raw = raw
        self.years = years
        self.std: dict[str, dict[str, pd.Series]] = {"income": {}, "balance": {}, "cash_flow": {}}

    def nan(self) -> pd.Series:
        return pd.Series(np.nan, index=self.years, dtype=float)

    def series(self, term: Term) -> pd.Series:
        if term.source.startswith("std:"):
            statement = term.source.split(":", 1)[1]
            item = term.label
            if item.endswith(":diff"):
                base = self.std[statement].get(item[: -len(":diff")], self.nan())
                return base.diff()
            return self.std[statement].get(item, self.nan())
        frame = self.raw[term.source]
        if term.label not in frame.columns:
            return self.nan()
        return frame[term.label].reindex(self.years).astype(float)


def _period_label(period: Any) -> str:
    return period.strftime("%Y-%m-%d") if isinstance(period, pd.Timestamp) else str(int(period))


def _evaluate(chain: tuple[Candidate, ...], ctx: _Context) -> tuple[pd.Series, str]:
    """Evaluate a fallback chain year by year; return values and source tag."""
    result = ctx.nan()
    used: list[str] = []
    for candidate in chain:
        missing = result.isna()
        if not missing.any():
            break
        total = pd.Series(0.0, index=ctx.years, dtype=float)
        for term in candidate.terms:
            values = ctx.series(term)
            if term.optional:
                values = values.fillna(0.0)
            total = total + term.sign * values
        if candidate.absolute:
            total = total.abs()
        fill = missing & total.notna()
        if fill.any():
            result = result.where(~fill, total)
            if fill.sum() == len(ctx.years):
                used.append(candidate.tag)
            else:
                labels = [_period_label(p) for p in ctx.years[fill.to_numpy()]]
                used.append(f"{candidate.tag} [{', '.join(labels)}]")
    return result, " | ".join(used) if used else "missing"


def _repair_ebitda(values: pd.Series, tag: str, ctx: _Context) -> tuple[pd.Series, str]:
    """Fix years where Yahoo's EBITDA equals EBIT although D&A is reported.

    Observed for a handful of companies (one year each): the ``EBITDA`` row
    simply repeats ``EBIT``. Those years are replaced by ``ebit + D&A``.
    """
    ebit = ctx.std["income"].get("ebit", ctx.nan())
    da = ctx.std["income"].get("depreciation_amortization", ctx.nan())
    broken = values.notna() & ebit.notna() & (values == ebit) & (da > 0)
    if not broken.any():
        return values, tag
    labels = ", ".join(_period_label(p) for p in ctx.years[broken.to_numpy()])
    note = f"std:ebit + std:depreciation_amortization [{labels}; Yahoo EBITDA equalled EBIT]"
    return values.where(~broken, ebit + da), f"{tag} | {note}"


def fiscal_period_ends(statements_raw: dict[str, Any]) -> dict[int, date]:
    """Fiscal year -> period-end date, over all three raw statements (the
    latest period end wins when two periods end in the same calendar year)."""
    ends: dict[int, date] = {}
    for key in _STATEMENT_KEYS.values():
        for period in (statements_raw.get(key) or {}).get("periods") or []:
            day = pd.Timestamp(period).date()
            if day.year not in ends or day > ends[day.year]:
                ends[day.year] = day
    return ends


def standardize(
    ticker: str,
    currency: str,
    statements_raw: dict[str, Any],
    fx_rate: float = 1.0,
    fx_by_year: Mapping[int, tuple[float, float]] | None = None,
) -> FinancialStatements:
    """Build standardized statements from the raw payload of one company.

    ``fx_rate`` converts the reporting currency into ``currency`` (the trading
    currency) and is applied to every monetary item of every year.

    ``fx_by_year`` (fiscal year -> ``(average rate, period-end rate)``)
    replaces ``fx_rate``: income-statement and cash-flow items are converted
    at the average rate of their year, balance-sheet items at the period-end
    rate. Raises :class:`ValueError` if a kept fiscal year has no rates.
    """
    raw = {
        short: _by_fiscal_year(raw_statement_frame(statements_raw.get(key)))
        for short, key in _STATEMENT_KEYS.items()
    }
    all_years: set[int] = set()
    for frame in raw.values():
        all_years.update(int(y) for y in frame.index)
    years = pd.Index(sorted(all_years), dtype=int, name="fiscal_year")
    ctx = _Context(raw, years)
    source_map: dict[str, str] = {}

    plan: tuple[tuple[str, dict[str, tuple[Candidate, ...]], tuple[str, ...]], ...] = (
        ("income", INCOME_CHAINS, INCOME_COLUMNS),
        ("balance", BALANCE_CHAINS, BALANCE_COLUMNS),
        ("cash_flow", CASH_FLOW_CHAINS, CASH_FLOW_COLUMNS),
    )
    frames: dict[str, pd.DataFrame] = {}
    for statement, chains, columns in plan:
        # Evaluate in chain order (dependencies first), output in contract order.
        for item, chain in chains.items():
            values, tag = _evaluate(chain, ctx)
            if statement == "income" and item == "ebitda":
                values, tag = _repair_ebitda(values, tag, ctx)
            ctx.std[statement][item] = values
            source_map[f"{statement}.{item}"] = tag
        frame = pd.DataFrame(
            {c: ctx.std[statement].get(c, ctx.nan()) for c in columns}, index=years
        )
        frames[statement] = frame.astype(float)

    # A fiscal year is kept if the income statement has at least two of
    # revenue, pretax income and net income. Yahoo often returns an extra,
    # nearly empty column (a single stray value) that is dropped here.
    core = frames["income"][["revenue", "pretax_income", "net_income"]]
    keep = core.notna().sum(axis=1) >= 2
    for statement in frames:
        frame = frames[statement].loc[keep.to_numpy()]
        monetary = [c for c in frame.columns if c not in NON_MONETARY]
        if fx_by_year is not None:
            missing = [int(y) for y in frame.index if int(y) not in fx_by_year]
            if missing:
                raise ValueError(f"{ticker}: no FX rates for fiscal year(s) {missing}")
            position = 1 if statement == "balance" else 0  # (average, period end)
            rates = pd.Series(
                [fx_by_year[int(y)][position] for y in frame.index], index=frame.index, dtype=float
            )
            frame = frame.copy()
            frame[monetary] = frame[monetary].mul(rates, axis=0)
        elif fx_rate != 1.0:
            frame = frame.copy()
            frame[monetary] = frame[monetary] * fx_rate
        frame.index = pd.Index([int(y) for y in frame.index], dtype=int, name="fiscal_year")
        frames[statement] = frame

    return FinancialStatements(
        ticker=ticker,
        currency=currency,
        income=frames["income"],
        balance=frames["balance"],
        cash_flow=frames["cash_flow"],
        source_map=source_map,
    )


def standardize_balance_periods(
    raw_statement: dict[str, Any] | None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Standardize a raw balance sheet WITHOUT collapsing to fiscal years.

    Returns a frame indexed by period end (ascending) with ``BALANCE_COLUMNS``
    plus ``preferred_equity``, and the source tags. Used for the market
    snapshot, which reads the most recent (often quarterly) balance sheet.
    """
    frame = raw_statement_frame(raw_statement)
    ctx = _Context({"is": frame.iloc[:, :0], "bs": frame, "cf": frame.iloc[:, :0]}, frame.index)
    tags: dict[str, str] = {}
    chains = dict(BALANCE_CHAINS)
    chains["preferred_equity"] = (
        _c(_bs("Preferred Stock Equity")),
        _c(_bs("Preferred Stock")),
    )
    for item, chain in chains.items():
        values, tag = _evaluate(chain, ctx)
        ctx.std["balance"][item] = values
        tags[item] = tag
    out = pd.DataFrame(ctx.std["balance"], index=frame.index).astype(float)
    return out, tags
