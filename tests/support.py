"""Shared builders for the test suite (synthetic companies, peers, price histories).

Everything here is independent of the code under test: it only assembles the
dataclasses of ``core.types`` from plain numbers chosen by the tests.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from valuation_engine.core.types import (
    BALANCE_COLUMNS,
    CASH_FLOW_COLUMNS,
    INCOME_COLUMNS,
    CompanyData,
    CompanyProfile,
    FinancialStatements,
    MarketData,
    PeerSnapshot,
    PriceHistory,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures"
FIXTURE_TICKERS: tuple[str, ...] = tuple(
    sorted(path.stem for path in (FIXTURES / "companies").glob("*.json"))
)
NON_FINANCIAL_TICKERS: tuple[str, ...] = tuple(t for t in FIXTURE_TICKERS if t != "SEB-A.ST")

INDUSTRY = "Specialty Industrial Machinery"
SECTOR = "Industrials"


def make_company(
    ticker: str = "TEST.ST",
    *,
    years: Sequence[int] = (2022, 2023, 2024, 2025),
    revenue: Sequence[float] = (800.0, 900.0, 1000.0, 1100.0),
    margin: float | Sequence[float] = 0.20,
    da: float = 0.04,
    capex: float | None = 0.05,
    nwc: float | None = 0.10,
    price: float = 10.0,
    shares: float = 100.0,
    debt: float = 300.0,
    cash: float = 50.0,
    minority: float = 0.0,
    preferred: float = 0.0,
    interest: float = 15.0,
    net_income: float | None = None,
    sector: str | None = SECTOR,
    industry: str | None = INDUSTRY,
    name: str | None = None,
    currency: str = "SEK",
    history: PriceHistory | None = None,
) -> CompanyData:
    """A small, fully specified company. Ratios are shares of revenue."""
    index = pd.Index([int(y) for y in years], name="fiscal_year")
    rev = np.asarray(revenue, dtype=float)
    margins = np.broadcast_to(np.asarray(margin, dtype=float), rev.shape)

    income = pd.DataFrame(np.nan, index=index, columns=list(INCOME_COLUMNS))
    income["revenue"] = rev
    income["cogs"] = rev * 0.6
    income["gross_profit"] = rev * 0.4
    income["ebitda"] = rev * margins
    income["depreciation_amortization"] = rev * da
    income["ebit"] = income["ebitda"] - income["depreciation_amortization"]
    income["interest_expense"] = interest
    income["pretax_income"] = income["ebit"] - interest
    income["income_tax"] = income["pretax_income"] * 0.2
    income["net_income"] = (
        income["pretax_income"] - income["income_tax"] if net_income is None else net_income
    )
    income["diluted_shares"] = shares

    balance = pd.DataFrame(np.nan, index=index, columns=list(BALANCE_COLUMNS))
    balance["cash"] = cash
    balance["total_debt"] = debt
    balance["total_equity"] = rev * 0.5
    balance["minority_interest"] = minority
    if nwc is not None:
        balance["net_working_capital"] = rev * nwc

    cash_flow = pd.DataFrame(np.nan, index=index, columns=list(CASH_FLOW_COLUMNS))
    cash_flow["depreciation_amortization"] = rev * da
    if capex is not None:
        cash_flow["capex"] = rev * capex

    statements = FinancialStatements(ticker, currency, income, balance, cash_flow)
    market = MarketData(
        ticker=ticker,
        as_of=date(2026, 11, 13),
        currency=currency,
        price=price,
        shares_outstanding=shares,
        market_cap=price * shares,
        total_debt=debt,
        cash=cash,
        minority_interest=minority,
        preferred_equity=preferred,
    )
    profile = CompanyProfile(
        ticker=ticker,
        name=name or f"{ticker} AB",
        sector=sector,
        industry=industry,
        country="Sweden",
        currency=currency,
        reporting_currency=currency,
    )
    return CompanyData(profile, statements, market, history)


def make_peer(
    ticker: str,
    *,
    market_cap: float = 1000.0,
    enterprise_value: float = 1200.0,
    revenue: float | None = 600.0,
    ebitda: float | None = 100.0,
    net_income: float | None = 50.0,
    total_debt: float | None = 200.0,
    revenue_growth: float | None = 0.05,
    return_on_equity: float | None = 0.15,
    beta: float | None = 1.0,
    sector: str | None = SECTOR,
    industry: str | None = INDUSTRY,
    name: str | None = None,
) -> PeerSnapshot:
    margin = ebitda / revenue if ebitda is not None and revenue else None
    return PeerSnapshot(
        ticker=ticker,
        name=name or f"{ticker} AB",
        sector=sector,
        industry=industry,
        market_cap=market_cap,
        enterprise_value=enterprise_value,
        revenue=revenue,
        ebitda=ebitda,
        ebit=None if ebitda is None else 0.8 * ebitda,
        net_income=net_income,
        total_debt=total_debt,
        revenue_growth=revenue_growth,
        ebitda_margin=margin,
        return_on_equity=return_on_equity,
        beta=beta,
    )


def make_history(
    beta: float,
    *,
    n_returns: int = 120,
    alpha: float = 0.001,
    noise: float = 0.0,
    seed: int = 7,
    ticker: str = "TEST.ST",
) -> PriceHistory:
    """Weekly prices whose simple returns satisfy ``stock = alpha + beta * bench + e``.

    The residual ``e`` is made exactly orthogonal to the benchmark returns (and to
    a constant), so the OLS slope is exactly ``beta`` whatever ``noise`` is.
    """
    rng = np.random.default_rng(seed)
    bench = rng.normal(0.002, 0.02, n_returns)
    residual = np.zeros(n_returns)
    if noise:
        raw = rng.normal(0.0, noise, n_returns)
        design = np.column_stack([np.ones(n_returns), bench])
        residual = raw - design @ np.linalg.lstsq(design, raw, rcond=None)[0]
    stock = alpha + beta * bench + residual
    dates = pd.date_range("2022-01-03", periods=n_returns + 1, freq="W-MON")
    prices = pd.DataFrame(
        {
            "stock": 100.0 * np.concatenate([[1.0], np.cumprod(1.0 + stock)]),
            "benchmark": 1000.0 * np.concatenate([[1.0], np.cumprod(1.0 + bench)]),
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )
    return PriceHistory(ticker=ticker, benchmark="^TEST", frequency="W", prices=prices)


def linear_percentile(values: Sequence[float], q: float) -> float:
    """Percentile with linear interpolation between order statistics (the
    "inclusive" / Excel PERCENTILE definition), written out by hand."""
    data = sorted(float(v) for v in values)
    position = (len(data) - 1) * q / 100.0
    lower = int(position)
    upper = min(lower + 1, len(data) - 1)
    return data[lower] + (data[upper] - data[lower]) * (position - lower)


def fenced_median(
    multiples: Sequence[float | None],
    bounds: tuple[float, float],
    *,
    iqr_factor: float = 1.5,
    min_observations: int = 5,
) -> tuple[float, int]:
    """Median of peer multiples the way comps documents it, written out by hand:
    drop values outside the hard ``bounds``, then (from five observations) drop
    values outside a Tukey fence on the log multiple. Returns (median, n used)."""
    import math

    kept = [float(m) for m in multiples if m is not None and bounds[0] <= m <= bounds[1]]
    if len(kept) >= min_observations:
        logs = [math.log(m) for m in kept]
        q1, q3 = linear_percentile(logs, 25), linear_percentile(logs, 75)
        low, high = q1 - iqr_factor * (q3 - q1), q3 + iqr_factor * (q3 - q1)
        kept = [m for m in kept if low - 1e-12 <= math.log(m) <= high + 1e-12]
    return linear_percentile(kept, 50), len(kept)
