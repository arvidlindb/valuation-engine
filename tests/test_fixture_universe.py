"""Every model on every fixture company (and on a seeded swarm of synthetic ones):
a model either returns a result that satisfies all invariants or raises
``NotApplicableError``; nothing else is acceptable."""

from __future__ import annotations

import math

import numpy as np
import pytest

from invariants import (
    assert_comps_invariants,
    assert_dcf_invariants,
    assert_lbo_invariants,
    assert_monte_carlo_invariants,
)
from support import FIXTURE_TICKERS, NON_FINANCIAL_TICKERS, make_company
from valuation_engine import pipeline
from valuation_engine.core.types import MarketParameters, NotApplicableError, ValuationMethod
from valuation_engine.models import comps, dcf, lbo, monte_carlo


@pytest.mark.parametrize("ticker", FIXTURE_TICKERS)
def test_dcf_on_every_fixture_company(ticker: str, companies, universe, params) -> None:
    company = companies[ticker]
    try:
        result = dcf.run_dcf(company, None, params, universe)
    except NotApplicableError as error:
        assert ticker == "SEB-A.ST", f"unexpected: {error}"
        assert str(error).startswith("financial sector")
        return
    assert ticker in NON_FINANCIAL_TICKERS
    assert_dcf_invariants(result, company)
    assert 0.04 < result.wacc < 0.12
    assert 4.0 <= result.exit_multiple <= 20.0
    assert result.currency == company.market.currency
    # without peers the model still runs, on the company's own beta and multiple
    alone = dcf.run_dcf(company, None, params)
    assert_dcf_invariants(alone, company)
    assert alone.wacc_result.peer_beta is None


@pytest.mark.parametrize("ticker", FIXTURE_TICKERS)
def test_comps_on_every_fixture_company(ticker: str, companies, universe) -> None:
    company = companies[ticker]
    result = comps.run_comps(company, universe)  # applicable to all fifteen
    assert_comps_invariants(result, company)
    assert result.is_financial is (ticker == "SEB-A.ST")
    assert len(result.peers) >= 5
    assert result.applicable_multiples, ticker
    assert result.currency == company.market.currency


@pytest.mark.parametrize("ticker", FIXTURE_TICKERS)
def test_lbo_on_every_fixture_company(ticker: str, companies, params) -> None:
    company = companies[ticker]
    try:
        standalone = lbo.run_lbo(company)
        with_params = lbo.run_lbo(company, lbo.default_assumptions(company, params))
    except NotApplicableError as error:
        assert ticker == "SEB-A.ST", f"unexpected: {error}"
        assert "financial or real estate" in str(error)
        return
    assert ticker in NON_FINANCIAL_TICKERS
    for result in (standalone, with_params):
        assert_lbo_invariants(result, company)
        assert result.liquidity_breach_years == []
        assert result.equity_share >= 0.30 - 1e-9
        assert math.isfinite(result.irr)
        # every solved ability-to-pay price reproduces its target IRR
        for target_irr, price in result.ability_to_pay.items():
            if result.ability_to_pay_status[target_irr] != lbo.STATUS_SOLVED:
                continue
            fed_back = lbo.run_lbo(company, lbo.with_entry_price(result.assumptions, price))
            assert fed_back.irr == pytest.approx(target_irr, abs=1e-8)
            assert_lbo_invariants(fed_back, company)


@pytest.mark.parametrize("ticker", FIXTURE_TICKERS)
def test_monte_carlo_on_every_fixture_company(ticker: str, companies, universe, params) -> None:
    company = companies[ticker]
    config = monte_carlo.MonteCarloConfig(n_simulations=3_000, seed=1)
    try:
        result = monte_carlo.run_monte_carlo(company, config=config, params=params, peers=universe)
    except NotApplicableError as error:
        assert ticker == "SEB-A.ST", f"unexpected: {error}"
        return
    base = dcf.run_dcf(company, None, params, universe)
    assert_monte_carlo_invariants(result, base)
    assert result.n_non_finite == 0  # the default configuration cannot produce invalid draws
    assert (
        abs(result.median / base.value_per_share_gordon - 1.0) < 0.10
        or abs(result.median - base.value_per_share_gordon) < 0.02 * company.market.price
    )


@pytest.mark.parametrize("ticker", FIXTURE_TICKERS)
def test_pipeline_report_on_every_fixture_company(ticker: str, reports, companies) -> None:
    report = reports[ticker]
    company = companies[ticker]
    assert report.errors == {}, report.errors  # no model crashed
    assert set(report.not_applicable) <= {"dcf", "lbo", "monte_carlo"}
    if ticker == "SEB-A.ST":
        assert report.status == "partial"
        assert [row.method for row in report.valuations] == [ValuationMethod.COMPS_PE]
    else:
        assert report.status == "ok" and report.not_applicable == {}
        assert_dcf_invariants(report.dcf, company)
        assert_lbo_invariants(report.lbo, company)
        assert_monte_carlo_invariants(report.monte_carlo, report.dcf)
        assert report.lbo.exit_multiple == report.dcf.exit_multiple
        assert report.dcf_peer_source == "comps"
        methods = {row.method for row in report.valuations}
        assert {
            ValuationMethod.DCF_GORDON,
            ValuationMethod.DCF_EXIT_MULTIPLE,
            ValuationMethod.LBO,
            ValuationMethod.MONTE_CARLO,
            ValuationMethod.COMPS_EV_EBITDA,
            ValuationMethod.COMPS_EV_SALES,
        } <= methods
    assert_comps_invariants(report.comps, company)
    assert company.ticker not in report.peer_tickers
    for row in report.valuations:
        assert row.low <= row.base <= row.high
        assert all(math.isfinite(v) for v in (row.low, row.base, row.high))
        assert row.currency == company.market.currency
    methods = [row.method for row in report.valuations]
    assert len(methods) == len(set(methods))
    import json

    json.dumps(report.summary(), allow_nan=False)
    assert pipeline.summary_text(report).isascii()


# ------------------------------------------------------------ synthetic swarm


def swarm(count: int = 60):
    rng = np.random.default_rng(2026)
    for index in range(count):
        n = int(rng.integers(2, 6))
        growth = rng.normal(0.05, 0.25, n).clip(-0.8, 2.0)
        revenue = np.cumprod(1 + growth) * float(10 ** rng.uniform(5, 11))
        last = float(revenue[-1])
        company = make_company(
            f"SYN{index}.ST",
            years=tuple(range(2026 - n, 2026)),
            revenue=tuple(revenue),
            margin=float(rng.normal(0.15, 0.2)),
            da=float(abs(rng.normal(0.04, 0.04))),
            capex=float(abs(rng.normal(0.05, 0.05))),
            nwc=float(rng.normal(0.1, 0.3)),
            price=float(10 ** rng.uniform(-1, 3)),
            shares=float(10 ** rng.uniform(5, 10)),
            debt=last * float(abs(rng.normal(0.3, 0.5))),
            cash=last * float(abs(rng.normal(0.1, 0.3))),
            minority=last * float(max(rng.normal(0.0, 0.05), 0.0)),
            preferred=last * float(max(rng.normal(-0.05, 0.05), 0.0)),
            interest=last * float(abs(rng.normal(0.01, 0.02))),
        )
        roll = rng.random(5)
        if roll[0] < 0.3:  # a NaN somewhere in the history
            column = str(rng.choice(["ebitda", "depreciation_amortization", "interest_expense"]))
            company.statements.income.iloc[int(rng.integers(0, n - 1)), :] = np.nan
            company.statements.income.loc[company.statements.income.index[0], column] = np.nan
        if roll[1] < 0.2:
            company.statements.cash_flow["capex"] = np.nan
        if roll[2] < 0.2:
            company.statements.balance["net_working_capital"] = np.nan
        if roll[3] < 0.2:
            company.statements.balance["total_debt"] = np.nan
            company.market.total_debt = 0.0
        if roll[4] < 0.15:
            company.market.cash = 0.0
        yield company


def test_models_never_fail_unexpectedly_on_a_swarm_of_synthetic_companies(universe) -> None:
    outcomes = {"dcf": 0, "lbo": 0, "comps": 0, "monte_carlo": 0, "not_applicable": 0}
    config = monte_carlo.MonteCarloConfig(n_simulations=300)
    for company in swarm():
        try:
            base = dcf.run_dcf(company, peers=universe)
            assert_dcf_invariants(base, company)
            outcomes["dcf"] += 1
            simulated = monte_carlo.run_monte_carlo(company, config=config, dcf_result=base)
            assert_monte_carlo_invariants(simulated, base)
            outcomes["monte_carlo"] += 1
        except NotApplicableError:
            outcomes["not_applicable"] += 1
        try:
            deal = lbo.run_lbo(company)
            assert_lbo_invariants(deal, company)
            outcomes["lbo"] += 1
        except NotApplicableError:
            outcomes["not_applicable"] += 1
        try:
            result = comps.run_comps(company, universe)
            assert_comps_invariants(result, company)
            outcomes["comps"] += 1
        except NotApplicableError:
            outcomes["not_applicable"] += 1
        report = pipeline.value_company(company, universe, MarketParameters(), n_simulations=200)
        assert report.errors == {}, (company.ticker, report.errors)
    # the swarm really exercises both the results and the not-applicable paths
    assert outcomes["dcf"] >= 20 and outcomes["lbo"] >= 20 and outcomes["comps"] == 60
    assert outcomes["monte_carlo"] == outcomes["dcf"]
    assert outcomes["not_applicable"] >= 10


@pytest.mark.parametrize(
    "changes",
    [
        {"debt": 0.0, "cash": 0.0, "interest": 0.0},  # no debt, no cash
        {"cash": 0.0},
        {"debt": 0.0},
        {"minority": 500.0, "preferred": 300.0},
        {"nwc": -0.25},
        {"da": 0.0},
        {"capex": 0.0},
        {"price": 1e-3},
        {"shares": 1e12},
        {"revenue": (1.0, 10.0, 100.0, 1000.0)},
        {"revenue": (1000.0, 500.0, 250.0, 125.0)},
        {"margin": 0.65},
        {"margin": 0.01},
        {"debt": 1e7},
        {"years": (2024, 2025), "revenue": (900.0, 1000.0)},
        {"years": (2019, 2021, 2024, 2025)},  # gaps in the fiscal years
    ],
)
def test_edge_cases_give_results_or_not_applicable(changes: dict, universe) -> None:
    company = make_company(**changes)
    ran = 0
    try:
        base = dcf.run_dcf(company, peers=universe)
        assert_dcf_invariants(base, company)
        simulated = monte_carlo.run_monte_carlo(
            company, config=monte_carlo.MonteCarloConfig(n_simulations=500), dcf_result=base
        )
        assert_monte_carlo_invariants(simulated, base)
        ran += 1
    except NotApplicableError:
        pass
    try:
        assert_lbo_invariants(lbo.run_lbo(company), company)
        ran += 1
    except NotApplicableError:
        pass
    assert_comps_invariants(comps.run_comps(company, universe), company)
    report = pipeline.value_company(company, universe, MarketParameters(), n_simulations=200)
    assert report.errors == {}
    assert ran >= 1 or changes in ({"margin": 0.01}, {"debt": 1e7})
