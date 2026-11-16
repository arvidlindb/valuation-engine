"""Data layer: offline loading from fixtures, the standardized contract, sign
conventions, accounting identities, currency conversion and the universe."""

from __future__ import annotations

import json
import math
import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from support import FIXTURE_TICKERS, FIXTURES
from valuation_engine import data
from valuation_engine.core.types import (
    BALANCE_COLUMNS,
    CASH_FLOW_COLUMNS,
    INCOME_COLUMNS,
    CompanyData,
    MarketParameters,
)
from valuation_engine.data import riksbank, yahoo

SEK_REPORTERS = (
    "ALFA.ST",
    "ASSA-B.ST",
    "ATCO-A.ST",
    "EPI-A.ST",
    "INDT.ST",
    "NIBE-B.ST",
    "SAND.ST",
    "SEB-A.ST",
    "SKF-B.ST",
    "TREL-B.ST",
    "VOLV-B.ST",
)
FOREIGN_REPORTERS = {"ABB.ST": "USD", "ALIV-SDB.ST": "USD", "HEXA-B.ST": "EUR"}


def raw_company(ticker: str) -> dict:
    return json.loads((FIXTURES / "companies" / f"{ticker}.json").read_text(encoding="utf-8"))


def raw_value(raw: dict, statement: str, label: str, year: int) -> float:
    """One cell of a raw annual statement (reporting currency, Yahoo's sign)."""
    table = raw["statements"][statement]
    column = next(i for i, period in enumerate(table["periods"]) if period.startswith(str(year)))
    return float(table["rows"][label][column])


def riksbank_series(currency: str) -> dict:
    payload = json.loads((FIXTURES / "riksbank.json").read_text(encoding="utf-8"))
    return payload["series"][f"SEK{currency}PMI"]


def riksbank_year_average(currency: str, year: int) -> float:
    """Observation-weighted mean of the twelve monthly averages of a calendar year."""
    rows = {row[0]: row for row in riksbank_series(currency)["monthly"]["rows"]}
    months = [rows[f"{year}-{month:02d}"] for month in range(1, 13)]
    return sum(row[1] * row[3] for row in months) / sum(row[3] for row in months)


def riksbank_december_ultimo(currency: str, year: int) -> float:
    rows = {row[0]: row for row in riksbank_series(currency)["monthly"]["rows"]}
    return float(rows[f"{year}-12"][2])


# ------------------------------------------------------------ offline loading


def test_fixture_set_is_the_documented_fifteen_companies() -> None:
    assert len(FIXTURE_TICKERS) == 15
    assert {"ATCO-A.ST", "SEB-A.ST", "ABB.ST", "HEXA-B.ST", "VSURE.ST"} <= set(FIXTURE_TICKERS)


@pytest.mark.parametrize("ticker", FIXTURE_TICKERS)
def test_company_loads_offline_with_the_standardized_contract(
    ticker: str, companies: dict[str, CompanyData]
) -> None:
    company = companies[ticker]
    statements = company.statements
    assert company.ticker == company.profile.ticker == statements.ticker == ticker
    assert company.market.ticker == ticker

    # exactly the standardized columns, in contract order; a missing item is NaN
    assert tuple(statements.income.columns) == INCOME_COLUMNS
    assert tuple(statements.balance.columns) == BALANCE_COLUMNS
    assert tuple(statements.cash_flow.columns) == CASH_FLOW_COLUMNS
    for frame in (statements.income, statements.balance, statements.cash_flow):
        assert frame.index.name == "fiscal_year"
        assert pd.api.types.is_integer_dtype(frame.index)
        assert frame.index.is_monotonic_increasing and frame.index.is_unique
        assert list(frame.index) == statements.fiscal_years
        assert all(pd.api.types.is_float_dtype(dtype) for dtype in frame.dtypes)
    assert statements.latest_year in (2025, 2026)
    assert statements.income["revenue"].notna().all()

    # one currency for everything
    profile, market = company.profile, company.market
    assert statements.currency == market.currency == profile.currency
    assert profile.name and profile.sector and profile.industry

    # market snapshot
    assert market.price > 0 and market.market_cap > 0
    assert market.shares_outstanding == market.market_cap / market.price
    assert market.total_debt >= 0 and market.cash >= 0
    assert market.minority_interest >= 0 and market.preferred_equity >= 0
    assert market.net_debt == market.total_debt - market.cash
    assert market.enterprise_value == pytest.approx(
        market.market_cap
        + market.total_debt
        - market.cash
        + market.minority_interest
        + market.preferred_equity
    )
    assert isinstance(market.as_of, date)

    # price history: aligned weekly series without gaps in the data
    history = company.price_history
    assert history is not None
    assert history.benchmark == "^OMXSPI" and history.frequency == "W"
    assert list(history.prices.columns) == ["stock", "benchmark"]
    assert len(history.prices) >= 52
    assert history.prices.notna().all().all() and (history.prices > 0).all().all()
    assert history.prices.index.is_monotonic_increasing and history.prices.index.is_unique
    assert set(history.prices.index.day_of_week) == {0}  # keyed on Mondays


@pytest.mark.parametrize("ticker", FIXTURE_TICKERS)
def test_sign_conventions_and_identities(ticker: str, companies: dict[str, CompanyData]) -> None:
    statements = companies[ticker].statements
    income, balance, cash_flow = statements.income, statements.balance, statements.cash_flow

    # expenses, capex, D&A, interest, dividends are stored as positive numbers
    for frame, item in (
        (cash_flow, "capex"),
        (cash_flow, "depreciation_amortization"),
        (cash_flow, "dividends_paid"),
        (income, "depreciation_amortization"),
        (income, "interest_expense"),
        (income, "cogs"),
    ):
        values = frame[item].dropna()
        assert (values >= 0).all(), item
    assert income["income_tax"].iloc[-1] > 0  # all fixture companies paid tax in the latest year

    def both(left: pd.Series, right: pd.Series) -> pd.Series:
        return left.notna() & right.notna()

    scale = float(income["revenue"].abs().max())
    # gross profit = revenue - COGS (Yahoo's own rows; Autoliv 2023 is off by USD 1 m)
    mask = both(income["gross_profit"], income["cogs"])
    gap = (income["gross_profit"] - (income["revenue"] - income["cogs"]))[mask]
    assert (gap.abs() <= 2e-4 * scale).all()
    # EBITDA = EBIT + D&A
    mask = both(income["ebitda"], income["ebit"]) & income["depreciation_amortization"].notna()
    gap = (income["ebitda"] - income["ebit"] - income["depreciation_amortization"])[mask]
    assert (gap.abs() <= 0.02 * scale).all()
    # free cash flow = cash from operations - capex, by construction
    mask = both(cash_flow["cash_from_operations"], cash_flow["capex"])
    gap = (cash_flow["free_cash_flow"] - cash_flow["cash_from_operations"] + cash_flow["capex"])[
        mask
    ]
    assert mask.any() and (gap.abs() <= 1e-9 * scale).all()
    # operating NWC = receivables + inventory - payables (missing inventory counts as zero)
    expected = (
        balance["accounts_receivable"]
        + balance["inventory"].fillna(0.0)
        - balance["accounts_payable"]
    )
    mask = both(balance["net_working_capital"], expected)
    assert ((balance["net_working_capital"] - expected)[mask].abs() <= 1e-9 * scale).all()
    # pretax income = EBIT - interest, where all three exist (Yahoo's EBIT definition)
    mask = both(income["ebit"], income["pretax_income"]) & income["interest_expense"].notna()
    gap = (income["ebit"] - income["interest_expense"] - income["pretax_income"])[mask]
    assert (gap.abs() <= 0.02 * scale).all()


def test_atlas_copco_matches_the_raw_payload_and_the_documented_check() -> None:
    company = data.load_company("ATCO-A.ST", offline=True)
    raw = raw_company("ATCO-A.ST")
    income = company.statements.income.loc[2025]
    cash_flow = company.statements.cash_flow.loc[2025]
    balance = company.statements.balance.loc[2025]

    # the table in docs/data_mapping.md, section 10 (billions of SEK)
    assert income["revenue"] / 1e9 == pytest.approx(168.34, abs=0.005)
    assert income["cogs"] / 1e9 == pytest.approx(96.13, abs=0.005)
    assert income["gross_profit"] == income["revenue"] - income["cogs"]
    assert income["gross_profit"] / 1e9 == pytest.approx(72.21, abs=0.005)
    assert income["ebitda"] == income["ebit"] + income["depreciation_amortization"]
    assert income["ebitda"] / 1e9 == pytest.approx(44.08, abs=0.005)
    assert income["ebit"] / 1e9 == pytest.approx(34.55, abs=0.005)
    assert income["depreciation_amortization"] / 1e9 == pytest.approx(9.53, abs=0.005)
    assert cash_flow["free_cash_flow"] == cash_flow["cash_from_operations"] - cash_flow["capex"]
    assert cash_flow["free_cash_flow"] / 1e9 == pytest.approx(26.38, abs=0.005)
    assert cash_flow["capex"] / 1e9 == pytest.approx(6.187, abs=0.0005)
    assert balance["total_debt"] / 1e9 == pytest.approx(34.83, abs=0.005)

    # every figure is the raw Yahoo cell, with the documented sign changes only
    assert income["revenue"] == raw_value(raw, "income_stmt", "Total Revenue", 2025)
    assert income["net_income"] == raw_value(
        raw, "income_stmt", "Net Income Common Stockholders", 2025
    )
    assert income["income_tax"] == raw_value(raw, "income_stmt", "Tax Provision", 2025)
    raw_capex = raw_value(raw, "cashflow", "Capital Expenditure", 2025)
    assert raw_capex < 0 and cash_flow["capex"] == -raw_capex
    raw_dividends = raw_value(raw, "cashflow", "Cash Dividends Paid", 2025)
    assert raw_dividends < 0 and cash_flow["dividends_paid"] == -raw_dividends
    raw_nwc_change = raw_value(raw, "cashflow", "Change In Working Capital", 2025)
    assert cash_flow["change_in_nwc"] == -raw_nwc_change
    assert cash_flow["free_cash_flow"] == raw_value(raw, "cashflow", "Free Cash Flow", 2025)
    assert company.statements.fiscal_years == [2022, 2023, 2024, 2025]

    # market snapshot: price and market cap from the quote, debt from the latest quarter
    market, info = company.market, raw["info"]
    assert market.price == info["currentPrice"] == 192.25
    assert market.market_cap == info["marketCap"]
    assert market.shares_outstanding == pytest.approx(info["impliedSharesOutstanding"], rel=1e-6)
    assert market.shares_outstanding > 1.4 * info["sharesOutstanding"]  # all classes, not only A
    quarter = raw["quarterly_balance_sheet"]
    assert quarter["periods"][0] == "2026-09-30"
    assert market.total_debt == quarter["rows"]["Total Debt"][0]
    assert market.enterprise_value / 1e9 == pytest.approx(950.9, abs=0.05)
    assert market.enterprise_value == pytest.approx(info["enterpriseValue"], rel=0.002)
    assert market.as_of == date(2026, 11, 13)
    assert company.profile.fx_rate == 1.0
    assert company.statements.source_map["income.revenue"].startswith("is:Total Revenue")
    assert "market.total_debt" in company.statements.source_map


def test_loading_is_deterministic(companies: dict[str, CompanyData]) -> None:
    again = data.load_company("ATCO-A.ST", offline=True)
    first = companies["ATCO-A.ST"]
    pd.testing.assert_frame_equal(again.statements.income, first.statements.income)
    pd.testing.assert_frame_equal(again.statements.balance, first.statements.balance)
    pd.testing.assert_frame_equal(again.statements.cash_flow, first.statements.cash_flow)
    pd.testing.assert_frame_equal(again.price_history.prices, first.price_history.prices)
    assert again.market == first.market and again.profile == first.profile
    assert data.load_company(" atco-a.st ", offline=True).market == first.market


# ------------------------------------------------------------------- currency


@pytest.mark.parametrize("ticker", SEK_REPORTERS)
def test_sek_reporters_are_not_converted(ticker: str, companies: dict[str, CompanyData]) -> None:
    company = companies[ticker]
    assert company.profile.currency == company.profile.reporting_currency == "SEK"
    assert company.profile.fx_rate == 1.0
    assert not [key for key in company.statements.source_map if key.startswith("fx.")]
    raw = raw_company(ticker)
    year = company.statements.latest_year
    assert company.statements.income.loc[year, "revenue"] == raw_value(
        raw, "income_stmt", "Total Revenue", year
    )


@pytest.mark.parametrize(("ticker", "currency"), sorted(FOREIGN_REPORTERS.items()))
def test_foreign_reporter_is_converted_with_riksbank_rates_per_year(
    ticker: str, currency: str, companies: dict[str, CompanyData]
) -> None:
    company = companies[ticker]
    raw = raw_company(ticker)
    sources = company.statements.source_map
    assert company.profile.currency == "SEK"
    assert company.profile.reporting_currency == currency
    assert sources["fx.method"].startswith(f"Riksbank SEK{currency}PMI")

    for year in company.statements.fiscal_years:
        # the rates recorded in source_map are the ones recomputed from the Riksbank fixture
        average = riksbank_year_average(currency, year)
        ultimo = riksbank_december_ultimo(currency, year)
        assert float(sources[f"fx.average.{year}"]) == pytest.approx(average, rel=1e-12)
        assert float(sources[f"fx.year_end.{year}"].split()[0]) == ultimo
        # income statement and cash flow at the average rate of the year
        revenue = raw_value(raw, "income_stmt", "Total Revenue", year)
        assert company.statements.income.loc[year, "revenue"] == pytest.approx(
            revenue * average, rel=1e-12
        )
        capex = raw_value(raw, "cashflow", "Capital Expenditure", year)
        assert company.statements.cash_flow.loc[year, "capex"] == pytest.approx(
            -capex * average, rel=1e-12
        )
        # balance sheet at the period-end rate
        debt = raw_value(raw, "balance_sheet", "Total Debt", year)
        assert company.statements.balance.loc[year, "total_debt"] == pytest.approx(
            debt * ultimo, rel=1e-12
        )
    # the average is a different number from the year-end and the spot rate
    assert float(sources["fx.average.2025"]) != float(sources["fx.year_end.2025"].split()[0])

    # market snapshot at the latest fixing; share counts are not converted
    spot = riksbank_series(currency)["latest"]["value"]
    assert company.profile.fx_rate == spot
    assert sources["fx.spot"].startswith(repr(spot))
    quarter = raw["quarterly_balance_sheet"]
    assert company.market.total_debt == pytest.approx(quarter["rows"]["Total Debt"][0] * spot)
    assert company.market.market_cap == raw["info"]["marketCap"]  # already in SEK
    assert company.statements.income.loc[2025, "diluted_shares"] == raw_value(
        raw, "income_stmt", "Diluted Average Shares", 2025
    )


def test_documented_fx_effect_on_revenue(companies: dict[str, CompanyData]) -> None:
    """The 'new' revenue figures of the table in docs/data_mapping.md, section 5."""
    documented = {
        "ALIV-SDB.ST": [89.5, 111.2, 109.7, 106.2],
        "ABB.ST": [298.1, 342.1, 323.0, 326.2],
        "HEXA-B.ST": [54.9, 62.4, 61.8, 60.0],
    }
    for ticker, expected in documented.items():
        revenue = companies[ticker].statements.income["revenue"] / 1e9
        assert revenue.tolist() == pytest.approx(expected, abs=0.06), ticker


def test_company_trading_in_euro_stays_in_euro(companies, universe) -> None:
    company = companies["VSURE.ST"]
    assert company.profile.currency == company.market.currency == "EUR"
    assert company.profile.reporting_currency == "EUR" and company.profile.fx_rate == 1.0
    assert company.statements.fiscal_years == [2024, 2025]
    raw = raw_company("VSURE.ST")
    assert company.statements.income.loc[2025, "revenue"] == raw_value(
        raw, "income_stmt", "Total Revenue", 2025
    )
    # the peer table is in SEK: market values at spot, P&L items at the year average
    snapshot = next(peer for peer in universe if peer.ticker == "VSURE.ST")
    spot = riksbank_series("EUR")["latest"]["value"]
    assert snapshot.market_cap == pytest.approx(company.market.market_cap * spot, rel=1e-9)
    assert snapshot.enterprise_value == pytest.approx(
        company.market.enterprise_value * spot, rel=1e-9
    )
    assert snapshot.revenue == pytest.approx(
        company.statements.income.loc[2025, "revenue"] * riksbank_year_average("EUR", 2025),
        rel=1e-9,
    )


def test_riksbank_series_arithmetic() -> None:
    series = riksbank.load_fx_series("USD", offline=True)
    assert series is not None and series.series == "SEKUSDPMI"
    assert series.average_rate(date(2025, 12, 31)) == pytest.approx(
        riksbank_year_average("USD", 2025), rel=1e-12
    )
    assert series.period_end_rate(date(2025, 12, 31)) == riksbank_december_ultimo("USD", 2025)
    # broken fiscal year: April 2025 .. March 2026
    rows = {row[0]: row for row in riksbank_series("USD")["monthly"]["rows"]}
    months = [f"2025-{m:02d}" for m in range(4, 13)] + [f"2026-{m:02d}" for m in range(1, 4)]
    expected = sum(rows[m][1] * rows[m][3] for m in months) / sum(rows[m][3] for m in months)
    assert series.average_rate(date(2026, 3, 31)) == pytest.approx(expected, rel=1e-12)
    assert series.period_end_rate(date(2026, 3, 15)) == rows["2026-03"][2]
    assert series.average_rate(date(2031, 12, 31)) is None  # months not covered
    assert series.period_end_rate(date(1999, 12, 31)) is None
    sek = riksbank.load_fx_series("SEK", offline=True)
    assert (
        sek.average_rate(date(2025, 12, 31)) == 1.0 and sek.period_end_rate(date(2025, 1, 1)) == 1.0
    )
    assert riksbank.load_fx_series("XXX", offline=True) is None


def test_missing_riksbank_data_falls_back_to_the_yahoo_spot_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = tmp_path / "fixtures"
    (store / "companies").mkdir(parents=True)
    shutil.copy(FIXTURES / "companies" / "ABB.ST.json", store / "companies" / "ABB.ST.json")
    shutil.copy(FIXTURES / "benchmark.json", store / "benchmark.json")
    monkeypatch.setenv("VALUATION_ENGINE_FIXTURES_DIR", str(store))

    company = data.load_company("ABB.ST", offline=True)
    raw = raw_company("ABB.ST")
    yahoo_spot = raw["fx"]["USDSEK"]["rate"]
    sources = company.statements.source_map
    assert sources["fx.method"].startswith("Yahoo spot USDSEK=X for all fiscal years")
    assert "no Riksbank data for USD" in sources["fx.method"]
    assert company.profile.fx_rate == yahoo_spot
    for year in company.statements.fiscal_years:
        revenue = raw_value(raw, "income_stmt", "Total Revenue", year)
        assert company.statements.income.loc[year, "revenue"] == pytest.approx(revenue * yahoo_spot)
    assert company.statements.income["revenue"].tolist() == pytest.approx(
        [x * 1e9 for x in (296.6, 324.7, 308.1, 334.6)], rel=1e-3
    )  # the 'old' column of the documented table


# -------------------------------------------------------------- error handling


@pytest.mark.parametrize("ticker", ["NOPE.ST", "ARENIT-SDB.ST", "AAK.ST"])
def test_unknown_or_uncached_ticker_raises_data_unavailable(ticker: str) -> None:
    with pytest.raises(data.DataUnavailableError, match="not available offline") as error:
        data.load_company(ticker, offline=True)
    assert isinstance(error.value, data.DataError)
    assert ticker in str(error.value)


@pytest.mark.parametrize("ticker", ["", ".ST", "   ", "../x"])
def test_malformed_ticker_raises_the_documented_data_error(ticker: str) -> None:
    with pytest.raises(data.DataUnavailableError, match="not a valid ticker"):
        data.load_company(ticker, offline=True)
    from valuation_engine import pipeline

    with pytest.raises(data.DataError):
        pipeline.run_valuation(ticker, offline=True)


def test_error_hierarchy() -> None:
    for error in (
        data.DataUnavailableError,
        data.NetworkDisabledError,
        data.FetchError,
        data.RateLimitError,
    ):
        assert issubclass(error, data.DataError)
    assert issubclass(data.RateLimitError, data.FetchError)
    assert not issubclass(data.DataError, ValueError)


# --------------------------------------------------------------- no network


def test_offline_mode_is_forced_by_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    assert data.is_offline() is True
    assert data.is_offline(False) is True
    # without the argument the environment still forbids the network
    company = data.load_company("ATCO-A.ST")
    assert company.market.price == 192.25
    assert len(data.list_universe()) == 687
    monkeypatch.setenv("VALUATION_ENGINE_OFFLINE", "0")
    assert data.is_offline() is False and data.is_offline(True) is True
    for value in ("1", "true", "YES", " on "):
        monkeypatch.setenv("VALUATION_ENGINE_OFFLINE", value)
        assert data.is_offline() is True


def test_network_layer_refuses_to_work_offline() -> None:
    with pytest.raises(data.NetworkDisabledError):
        yahoo.fetch_company("ATCO-A.ST")
    with pytest.raises(data.NetworkDisabledError):
        yahoo.fetch_screen()
    with pytest.raises(data.NetworkDisabledError):
        yahoo.fetch_benchmark()
    with pytest.raises(data.NetworkDisabledError):
        riksbank.fetch_latest("SEGVB10YC")


def test_offline_loading_never_imports_yfinance_and_never_writes_the_cache(
    cache_dir: Path, companies, universe, params
) -> None:
    data.load_universe_entries(offline=True)
    data.market_parameter_sources(offline=True)
    assert "yfinance" not in sys.modules
    assert data.cache_dir() == cache_dir
    assert list(cache_dir.rglob("*")) == []
    assert data.fixtures_dir() == FIXTURES


def test_the_socket_guard_of_the_suite_is_active() -> None:
    import socket

    import conftest

    before = len(conftest.NETWORK_ATTEMPTS)
    with pytest.raises(conftest.NetworkAccessError):
        socket.create_connection(("192.0.2.1", 80), timeout=0.1)
    with pytest.raises(conftest.NetworkAccessError):
        socket.getaddrinfo("example.com", 443)
    assert len(conftest.NETWORK_ATTEMPTS) == before + 2
    del conftest.NETWORK_ATTEMPTS[before:]  # these two were deliberate


# ----------------------------------------------------------- market parameters


def test_market_parameters_offline() -> None:
    params = data.load_market_parameters(offline=True)
    assert isinstance(params, MarketParameters)
    payload = json.loads((FIXTURES / "riksbank.json").read_text(encoding="utf-8"))
    latest = payload["series"]["SEGVB10YC"]["latest"]
    assert params.risk_free_rate == pytest.approx(latest["value"] / 100.0, abs=1e-12)
    assert params.risk_free_rate == 0.03075
    assert params.equity_risk_premium == 0.0423
    assert params.tax_rate == 0.206
    assert params.benchmark == "^OMXSPI"
    assert params.long_term_growth == 0.02

    sources = data.market_parameter_sources(offline=True)
    assert set(sources) == {
        "risk_free_rate",
        "equity_risk_premium",
        "tax_rate",
        "benchmark",
        "long_term_growth",
        "country_risk_premium",
    }
    rate = sources["risk_free_rate"]
    assert (rate.value, rate.as_of, rate.origin) == (0.03075, latest["date"], "fixtures")
    assert "Riksbank" in rate.source and "SEGVB10YC" in rate.url
    assert sources["equity_risk_premium"].origin == "reference file"
    assert "Damodaran" in sources["equity_risk_premium"].source
    assert sources["country_risk_premium"].value == 0.0
    for name in ("risk_free_rate", "equity_risk_premium", "tax_rate", "long_term_growth"):
        assert sources[name].value == getattr(params, name)


def test_market_parameters_fall_back_to_defaults_without_riksbank_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VALUATION_ENGINE_FIXTURES_DIR", str(tmp_path / "empty"))
    params = data.load_market_parameters(offline=True)
    assert params.risk_free_rate == MarketParameters().risk_free_rate
    assert params.equity_risk_premium == 0.0423
    source = data.market_parameter_sources(offline=True)["risk_free_rate"]
    assert source.origin == "default" and "unavailable" in source.source


# -------------------------------------------------------------------- universe


def test_universe_lists_one_share_class_per_company() -> None:
    tickers = data.list_universe(offline=True)
    assert len(tickers) == len(set(tickers)) == 687
    assert tickers == sorted(tickers)
    assert all(ticker.endswith(".ST") for ticker in tickers)
    assert set(FIXTURE_TICKERS) <= set(tickers)
    # the liquid class is the primary one
    for chosen, other in (
        ("ATCO-A.ST", "ATCO-B.ST"),
        ("VOLV-B.ST", "VOLV-A.ST"),
        ("SHB-A.ST", "SHB-B.ST"),
        ("ERIC-B.ST", "ERIC-A.ST"),
        ("INVE-B.ST", "INVE-A.ST"),
    ):
        assert chosen in tickers and other not in tickers
    assert not [t for t in tickers if "-PREF" in t or t.split(".")[0].endswith("-D")]
    roots = [t.split(".")[0].split("-")[0] for t in tickers]
    assert len(roots) == len(set(roots))


def test_universe_statistics_match_the_documented_funnel() -> None:
    entries, stats = data.load_universe_entries(offline=True)
    assert (stats.screener_symbols, stats.equity_quote_type) == (984, 921)
    assert (stats.ordinary_share_symbols, stats.companies) == (737, 687)
    assert stats.preference_only_companies == ("HEIM-PREF.ST",)
    atlas = next(entry for entry in entries if entry.ticker == "ATCO-A.ST")
    assert atlas.share_classes == ("ATCO-A.ST", "ATCO-B.ST")
    assert atlas.name == "Atlas Copco AB (publ)"
    assert sum(len(entry.share_classes) for entry in entries) == 737 - 1  # minus HEIM-PREF
    names = [entry.name for entry in entries]
    assert len(names) == len(set(names))


def test_peer_snapshots_cover_the_loadable_universe(universe) -> None:
    tickers = [peer.ticker for peer in universe]
    assert len(tickers) == len(set(tickers)) == 680
    assert set(tickers) <= set(data.list_universe(offline=True))
    assert set(FIXTURE_TICKERS) <= set(tickers)
    assert sum(peer.is_financial for peer in universe) == 103
    assert all(peer.market_cap > 0 for peer in universe)
    for peer in universe:
        for value in (peer.revenue, peer.ebitda, peer.net_income, peer.beta, peer.total_debt):
            assert value is None or math.isfinite(value)


@pytest.mark.parametrize("ticker", ["ATCO-A.ST", "SAND.ST", "ABB.ST", "SEB-A.ST"])
def test_peer_snapshot_is_derived_from_the_company_data(ticker: str, companies, universe) -> None:
    company = companies[ticker]
    peer = next(p for p in universe if p.ticker == ticker)
    income = company.statements.income
    assert peer.name == company.profile.name
    assert (peer.sector, peer.industry) == (company.profile.sector, company.profile.industry)
    assert peer.market_cap == pytest.approx(company.market.market_cap, rel=1e-12)
    assert peer.enterprise_value == pytest.approx(company.market.enterprise_value, rel=1e-12)
    assert peer.total_debt == pytest.approx(company.market.total_debt, rel=1e-12)
    assert peer.revenue == pytest.approx(income["revenue"].iloc[-1], rel=1e-12)
    assert peer.net_income == pytest.approx(income["net_income"].iloc[-1], rel=1e-12)
    assert peer.revenue_growth == pytest.approx(
        income["revenue"].iloc[-1] / income["revenue"].iloc[-2] - 1, rel=1e-9
    )
    if math.isnan(income["ebitda"].iloc[-1]):
        assert peer.ebitda is None and peer.ebitda_margin is None and peer.ev_ebitda is None
    else:
        assert peer.ebitda == pytest.approx(income["ebitda"].iloc[-1], rel=1e-12)
        assert peer.ebitda_margin == pytest.approx(peer.ebitda / peer.revenue, rel=1e-9)
    equity = company.statements.balance["total_equity"]
    assert peer.return_on_equity == pytest.approx(
        income["net_income"].iloc[-1] / equity.iloc[-2:].mean(), rel=1e-9
    )
    # raw OLS beta of weekly returns, recomputed with a least-squares fit
    returns = company.price_history.prices.pct_change().dropna()
    slope = np.polyfit(returns["benchmark"], returns["stock"], 1)[0]
    assert peer.beta == pytest.approx(slope, rel=1e-9)
    assert data.regression_beta(company.price_history) == pytest.approx(slope, rel=1e-9)


# ------------------------------------------------- building from a raw payload


def payload(**changes: object) -> dict:
    """A minimal raw Yahoo payload with three fiscal years (newest first)."""
    periods = ["2025-12-31", "2024-12-31", "2023-12-31"]
    raw: dict = {
        "schema_version": 1,
        "ticker": "RAW.ST",
        "fetched_at": "2026-11-16T07:00:00+00:00",
        "info": {
            "longName": "Raw AB",
            "sector": "Industrials",
            "industry": "Tools & Accessories",
            "country": "Sweden",
            "currency": "SEK",
            "financialCurrency": "SEK",
            "currentPrice": 50.0,
            "marketCap": 5000.0,
            "beta": 1.1,
        },
        "statements": {
            "income_stmt": {
                "periods": periods,
                "rows": {
                    "Total Revenue": [1000.0, 900.0, 800.0],
                    "Cost Of Revenue": [600.0, 540.0, 480.0],
                    "EBIT": [150.0, 130.0, 110.0],
                    "EBITDA": [200.0, 130.0, 150.0],  # 2024 repeats EBIT
                    "Reconciled Depreciation": [50.0, 45.0, 40.0],
                    "Interest Expense": [-10.0, 9.0, 8.0],
                    "Pretax Income": [140.0, 121.0, 102.0],
                    "Tax Provision": [30.0, -5.0, 20.0],
                    "Net Income Common Stockholders": [110.0, 126.0, 82.0],
                    "Diluted Average Shares": [100.0, 100.0, 100.0],
                },
            },
            "balance_sheet": {
                "periods": periods,
                "rows": {
                    "Cash Cash Equivalents And Short Term Investments": [80.0, 70.0, 60.0],
                    "Accounts Receivable": [200.0, 180.0, 160.0],
                    "Inventory": [100.0, None, 80.0],
                    "Accounts Payable": [120.0, 110.0, 100.0],
                    "Total Assets": [2000.0, 1900.0, 1800.0],
                    "Total Debt": [400.0, 380.0, 360.0],
                    "Stockholders Equity": [900.0, 850.0, 800.0],
                    "Minority Interest": [-15.0, 5.0, 5.0],
                },
            },
            "cashflow": {
                "periods": periods,
                "rows": {
                    "Operating Cash Flow": [170.0, 150.0, 130.0],
                    "Capital Expenditure": [-60.0, -55.0, -50.0],
                    "Change In Working Capital": [-12.0, 7.0, -3.0],
                    "Cash Dividends Paid": [-40.0, -35.0, -30.0],
                },
            },
        },
        "quarterly_balance_sheet": {"periods": [], "rows": {}},
        "prices": {"interval": "1wk", "adjusted": True, "dates": [], "close": []},
        "fx": {},
    }
    raw.update(changes)
    return raw


def test_build_company_applies_the_mapping_rules() -> None:
    company = data.build_company(payload())
    income, balance, cash_flow = (
        company.statements.income,
        company.statements.balance,
        company.statements.cash_flow,
    )
    assert company.statements.fiscal_years == [2023, 2024, 2025]
    assert income["revenue"].tolist() == [800.0, 900.0, 1000.0]
    assert income["gross_profit"].tolist() == [320.0, 360.0, 400.0]  # revenue - COGS
    assert income["interest_expense"].tolist() == [8.0, 9.0, 10.0]  # absolute value
    assert income["income_tax"].tolist() == [20.0, -5.0, 30.0]  # a tax credit stays negative
    # Yahoo's EBITDA repeated EBIT in 2024 although D&A was reported: repaired
    assert income["ebitda"].tolist() == [150.0, 175.0, 200.0]
    assert "Yahoo EBITDA equalled EBIT" in company.statements.source_map["income.ebitda"]
    assert cash_flow["capex"].tolist() == [50.0, 55.0, 60.0]
    assert cash_flow["dividends_paid"].tolist() == [30.0, 35.0, 40.0]
    assert cash_flow["change_in_nwc"].tolist() == [3.0, -7.0, 12.0]  # sign flipped
    assert cash_flow["free_cash_flow"].tolist() == [80.0, 95.0, 110.0]
    assert cash_flow["depreciation_amortization"].tolist() == [40.0, 45.0, 50.0]
    assert balance["net_working_capital"].tolist() == [140.0, 70.0, 180.0]  # 2024: no inventory
    assert math.isnan(balance["goodwill_intangibles"].iloc[-1])

    market = company.market
    assert (market.price, market.market_cap, market.shares_outstanding) == (50.0, 5000.0, 100.0)
    assert (market.total_debt, market.cash) == (400.0, 80.0)
    assert market.minority_interest == 0.0  # a negative minority is floored at zero
    assert market.preferred_equity == 0.0
    assert market.beta == 1.1
    assert market.as_of == date(2026, 11, 16)
    assert company.price_history is None
    assert (company.profile.name, company.profile.currency) == ("Raw AB", "SEK")
    assert "annual balance sheet 2025-12-31" in company.statements.source_map["market.total_debt"]


def test_latest_quarterly_balance_sheet_feeds_the_market_snapshot() -> None:
    raw = payload(
        quarterly_balance_sheet={
            "periods": ["2026-09-30", "2026-06-30"],
            "rows": {
                "Total Debt": [450.0, 430.0],
                "Cash Cash Equivalents And Short Term Investments": [95.0, 90.0],
                "Total Assets": [2100.0, 2050.0],
                "Preferred Stock Equity": [25.0, 25.0],
            },
        }
    )
    company = data.build_company(raw)
    assert (company.market.total_debt, company.market.cash) == (450.0, 95.0)
    assert company.market.preferred_equity == 25.0
    assert company.market.minority_interest == 0.0  # not in the quarter, annual value negative
    assert company.statements.balance.loc[2025, "total_debt"] == 400.0  # statements unchanged
    assert "quarterly balance sheet 2026-09-30" in company.statements.source_map["market.cash"]


def test_fiscal_year_rules() -> None:
    raw = payload()
    income = raw["statements"]["income_stmt"]
    # a broken year ending in March 2026, and a stray, nearly empty fifth column
    income["periods"] = ["2026-03-31", "2025-03-31", "2024-03-31", "2023-03-31"]
    for label, values in income["rows"].items():
        values.append(42.0 if label == "Total Revenue" else None)
    for name in ("balance_sheet", "cashflow"):
        raw["statements"][name]["periods"] = ["2026-03-31", "2025-03-31", "2024-03-31"]
    company = data.build_company(raw)
    assert company.statements.fiscal_years == [2024, 2025, 2026]  # 2023 has one stray value
    assert company.statements.income.loc[2026, "revenue"] == 1000.0
    assert company.statements.balance.index.tolist() == [2024, 2025, 2026]


def test_single_fiscal_year_loads_but_models_decline() -> None:
    from valuation_engine.core.types import NotApplicableError
    from valuation_engine.models import dcf, lbo

    raw = payload()
    for statement in raw["statements"].values():
        statement["periods"] = statement["periods"][:1]
        statement["rows"] = {label: values[:1] for label, values in statement["rows"].items()}
    company = data.build_company(raw)
    assert company.statements.fiscal_years == [2025]
    with pytest.raises(NotApplicableError, match="insufficient history"):
        dcf.run_dcf(company)
    with pytest.raises(NotApplicableError, match="at least 2 fiscal years"):
        lbo.run_lbo(company)


def test_market_cap_falls_back_to_shares_times_price() -> None:
    raw = payload()
    raw["info"]["marketCap"] = None
    company = data.build_company(raw)
    assert company.market.market_cap == 100.0 * 50.0
    assert "diluted_shares x price" in company.statements.source_map["market.market_cap"]


def test_unusable_payloads_raise_data_unavailable() -> None:
    no_price = payload()
    no_price["info"]["currentPrice"] = None
    with pytest.raises(data.DataUnavailableError, match="no usable price"):
        data.build_company(no_price)
    with pytest.raises(data.DataUnavailableError, match="no annual income statement"):
        data.build_company(payload(statements={}))
    no_cap = payload()
    no_cap["info"]["marketCap"] = None
    no_cap["statements"]["income_stmt"]["rows"].pop("Diluted Average Shares")
    with pytest.raises(data.DataUnavailableError, match="no market capitalization"):
        data.build_company(no_cap)
    foreign = payload()
    foreign["info"]["financialCurrency"] = "USD"
    with pytest.raises(data.DataUnavailableError, match="no FX rate USD->SEK"):
        data.build_company(foreign)


def test_build_company_without_a_plan_uses_the_stored_spot_rate() -> None:
    raw = payload(fx={"USDSEK": {"symbol": "USDSEK=X", "rate": 10.0, "as_of": "2026-11-16"}})
    raw["info"]["financialCurrency"] = "USD"
    company = data.build_company(raw)
    assert company.profile.reporting_currency == "USD" and company.profile.fx_rate == 10.0
    assert company.statements.income["revenue"].tolist() == [8000.0, 9000.0, 10000.0]
    assert company.statements.balance.loc[2025, "total_debt"] == 4000.0
    assert company.statements.income["diluted_shares"].tolist() == [100.0] * 3  # not converted
    assert company.market.total_debt == 4000.0
    assert company.market.market_cap == 5000.0  # the quote is already in the trading currency


def test_price_history_needs_52_common_weeks_and_drops_the_running_week() -> None:
    weeks = pd.date_range("2025-09-01", periods=63, freq="W-MON")  # last: 2026-11-09
    closes = [100.0 + i for i in range(63)]
    dates = [d.strftime("%Y-%m-%d") for d in weeks]
    benchmark = {
        "symbol": "^OMXSPI",
        "fetched_at": "2026-11-11T08:00:00+00:00",
        "dates": dates,
        "close": closes,
    }
    raw = payload(prices={"dates": dates, "close": closes})
    raw["fetched_at"] = "2026-11-11T07:00:00+00:00"  # a Wednesday: that week is in progress
    company = data.build_company(raw, benchmark)
    history = company.price_history
    assert history is not None and len(history.prices) == 62
    assert history.prices.index[-1] == pd.Timestamp("2026-11-02")
    short = payload(prices={"dates": dates[-40:], "close": closes[-40:]})
    assert data.build_company(short, benchmark).price_history is None
    assert data.build_company(payload(), benchmark).price_history is None
    assert data.regression_beta(None) is None


def test_sparse_quarter_with_leases_only_falls_back_to_the_quoted_total_debt() -> None:
    """docs/data_mapping.md 10.7: Yahoo's quarter sometimes lacks the loan rows."""
    raw = payload(
        quarterly_balance_sheet={
            "periods": ["2026-09-30"],
            "rows": {
                "Total Debt": [35.0],  # leases only
                "Capital Lease Obligations": [35.0],
                "Cash Cash Equivalents And Short Term Investments": [95.0],
                "Total Assets": [2100.0],
            },
        }
    )
    raw["statements"]["balance_sheet"]["rows"]["Long Term Debt"] = [300.0, 290.0, 280.0]
    raw["info"]["totalDebt"] = 410.0
    company = data.build_company(raw)
    assert company.market.total_debt == 410.0
    assert company.statements.source_map["market.total_debt"].startswith("info:totalDebt")
    assert company.market.cash == 95.0  # the rest of the quarter is still used

    without_quote = payload(quarterly_balance_sheet=raw["quarterly_balance_sheet"])
    without_quote["statements"]["balance_sheet"]["rows"]["Long Term Debt"] = [300.0, 290.0, 280.0]
    company = data.build_company(without_quote)
    assert company.market.total_debt == 400.0  # the latest annual balance sheet
    assert "annual statement 2025" in company.statements.source_map["market.total_debt"]

    with_loans = payload(quarterly_balance_sheet=raw["quarterly_balance_sheet"])
    with_loans["quarterly_balance_sheet"] = {
        "periods": ["2026-09-30"],
        "rows": {
            "Total Debt": [335.0],
            "Capital Lease Obligations": [35.0],
            "Long Term Debt": [300.0],
            "Total Assets": [2100.0],
        },
    }
    with_loans["statements"]["balance_sheet"]["rows"]["Long Term Debt"] = [300.0, 290.0, 280.0]
    assert data.build_company(with_loans).market.total_debt == 335.0


def test_missing_cash_and_debt_rows_use_the_quote_then_zero() -> None:
    raw = payload()
    rows = raw["statements"]["balance_sheet"]["rows"]
    del rows["Cash Cash Equivalents And Short Term Investments"], rows["Total Debt"]
    raw["info"]["totalCash"] = 77.0
    company = data.build_company(raw)
    assert company.market.cash == 77.0
    assert company.market.total_debt == 0.0  # no row anywhere: assumed zero, and said so
    assert company.statements.source_map["market.total_debt"] == "assumed 0 (not reported)"
    assert math.isnan(company.statements.balance.loc[2025, "total_debt"])  # NaN, not 0, in history


# -------------------------------------------- describing the FX conversion


def test_describe_statement_fx_is_none_without_a_conversion(companies) -> None:
    assert data.describe_statement_fx(companies["ATCO-A.ST"]) is None
    assert data.describe_statement_fx(companies["VSURE.ST"]) is None  # EUR reporter traded in EUR
    assert data.describe_statement_fx(data.build_company(payload())) is None


@pytest.mark.parametrize(("ticker", "currency"), sorted(FOREIGN_REPORTERS.items()))
def test_describe_statement_fx_names_the_rates_that_were_applied(
    ticker: str, currency: str, companies
) -> None:
    company = companies[ticker]
    text = data.describe_statement_fx(company)
    year = company.statements.latest_year
    average = riksbank_year_average(currency, year)
    year_end = riksbank_december_ultimo(currency, year)
    spot = riksbank_series(currency)["latest"]["value"]
    assert text.startswith(f"Statements reported in {currency}; converted to SEK per fiscal year")
    assert f"Riksbank SEK{currency}PMI" in text
    assert f"year average (FY{year}: {average:.4f})" in text
    assert f"year-end rate (FY{year}: {year_end:.4f})" in text
    assert text.endswith(f"Market snapshot at the spot rate {spot:.4f}.")
    # and the numbers quoted are the ones the statements carry
    raw = raw_company(ticker)
    revenue = raw_value(raw, "income_stmt", "Total Revenue", year)
    assert company.statements.income.loc[year, "revenue"] / revenue == pytest.approx(average)
    debt = raw_value(raw, "balance_sheet", "Total Debt", year)
    assert company.statements.balance.loc[year, "total_debt"] / debt == pytest.approx(year_end)


def test_describe_statement_fx_says_so_when_one_spot_rate_was_used() -> None:
    raw = payload(fx={"USDSEK": {"symbol": "USDSEK=X", "rate": 10.0, "as_of": "2026-11-16"}})
    raw["info"]["financialCurrency"] = "USD"
    company = data.build_company(raw)  # no plan: the legacy single-spot conversion
    text = data.describe_statement_fx(company)
    assert (
        "every fiscal year and the market snapshot are converted to SEK with one spot rate" in text
    )
    assert "10.0000" in text and "Yahoo spot USDSEK=X" in text
    assert "year average" not in text
    assert company.statements.income.loc[2025, "revenue"] == 1000.0 * 10.0  # it is true

    company.statements.source_map.pop("fx.method")
    unknown = data.describe_statement_fx(company)
    assert "conversion method was not recorded" in unknown and "10.0000" in unknown


def test_describe_statement_fx_for_the_riksbank_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = tmp_path / "fixtures"
    (store / "companies").mkdir(parents=True)
    shutil.copy(FIXTURES / "companies" / "ABB.ST.json", store / "companies" / "ABB.ST.json")
    monkeypatch.setenv("VALUATION_ENGINE_FIXTURES_DIR", str(store))
    company = data.load_company("ABB.ST", offline=True)
    text = data.describe_statement_fx(company)
    yahoo_spot = raw_company("ABB.ST")["fx"]["USDSEK"]["rate"]
    assert f"with one spot rate, {yahoo_spot:.4f}" in text
    assert "no Riksbank data for USD" in text


def test_one_year_of_weekly_returns_is_the_beta_threshold_everywhere() -> None:
    from support import make_company, make_history, make_peer
    from valuation_engine.models import comps, dcf

    short, enough = make_history(1.3, n_returns=51), make_history(1.3, n_returns=52)
    assert data.regression_beta(short) is None
    assert data.regression_beta(enough) == pytest.approx(1.3)
    assert dcf.regression_beta_stats(short)[0] is None
    assert dcf.regression_beta_stats(enough)[0] == pytest.approx(1.3)
    universe = [make_peer(f"P{i}.ST", enterprise_value=(8.0 + i) * 100.0) for i in range(6)]
    raw_beta = {
        n: comps.run_comps(
            make_company(history=make_history(1.3, n_returns=n)), universe
        ).target_features.loc["beta", "raw"]
        for n in (51, 52)
    }
    assert math.isnan(raw_beta[51]) and raw_beta[52] == pytest.approx(1.3)


# ------------------------------------------------------- optional live check


@pytest.mark.network
def test_live_download_still_matches_the_contract(live_network: Path) -> None:
    """Run with ``pytest -m network``: Yahoo's payload still maps onto the contract."""
    company = data.load_company("ATCO-A.ST", refresh=True)
    assert (live_network / "companies" / "ATCO-A.ST.json").is_file()
    assert tuple(company.statements.income.columns) == INCOME_COLUMNS
    assert tuple(company.statements.balance.columns) == BALANCE_COLUMNS
    assert tuple(company.statements.cash_flow.columns) == CASH_FLOW_COLUMNS
    assert len(company.statements.fiscal_years) >= 3
    assert company.market.price > 0 and company.market.market_cap > 1e11
    assert company.market.shares_outstanding == company.market.market_cap / company.market.price
    latest = company.statements.income.iloc[-1]
    assert latest["ebitda"] == pytest.approx(
        latest["ebit"] + latest["depreciation_amortization"], rel=0.02
    )
    assert company.price_history is not None and len(company.price_history.prices) >= 52
