"""End to end on the fixtures: the pipeline, its wiring between models, the report."""

from __future__ import annotations

import json
import math
import statistics
import subprocess
import sys
from dataclasses import replace

import pytest

from invariants import (
    assert_comps_invariants,
    assert_dcf_invariants,
    assert_lbo_invariants,
    assert_monte_carlo_invariants,
)
from support import FIXTURES, fenced_median, make_company, make_peer
from valuation_engine import data, pipeline
from valuation_engine.core.types import (
    DCFAssumptions,
    MarketParameters,
    ValuationMethod,
)
from valuation_engine.models import dcf, lbo
from valuation_engine.viz.protocols import METHOD_KEYS, MonteCarloLike, ValuationReportLike

ALL_METHODS = [
    ValuationMethod.COMPS_EV_EBITDA,
    ValuationMethod.COMPS_EV_SALES,
    ValuationMethod.COMPS_PE,
    ValuationMethod.DCF_GORDON,
    ValuationMethod.DCF_EXIT_MULTIPLE,
    ValuationMethod.LBO,
    ValuationMethod.MONTE_CARLO,
]


# ---------------------------------------------------------------- Atlas Copco


def test_atlas_copco_report_has_every_method_with_finite_values(atco_report) -> None:
    report = atco_report
    assert isinstance(report, ValuationReportLike)
    assert isinstance(report.monte_carlo, MonteCarloLike)
    assert report.status == "ok"
    assert report.not_applicable == {} and report.errors == {} and report.warnings == []
    assert (report.ticker, report.currency, report.price) == ("ATCO-A.ST", "SEK", 192.25)
    assert [row.method for row in report.valuations] == ALL_METHODS
    for row in report.valuations:
        assert all(math.isfinite(v) for v in (row.low, row.base, row.high))
        assert 0 < row.low <= row.base <= row.high
        assert row.currency == "SEK"
        assert 0.3 < row.base / report.price < 1.5  # a sane order of magnitude
    for model in (report.dcf, report.comps, report.lbo, report.monte_carlo):
        assert model is not None
    assert report.n_simulations == report.monte_carlo.n_simulations == 10_000
    assert report.seed == 42
    assert set(report.timings) == {"load", "comps", "dcf", "lbo", "monte_carlo", "total"}
    assert report.generated_at


def test_atlas_copco_matches_the_documented_figures(atco_report) -> None:
    """PLAN.md result table (price 192.25). The DCF exit-multiple and LBO rows depend
    on the shared peer multiple and the shared operating forecast and are
    re-derived independently below."""
    documented = {
        ValuationMethod.DCF_GORDON: (93.7, 115.9, 153.7),
        ValuationMethod.DCF_EXIT_MULTIPLE: (160.55, 175.23, 191.04),
        ValuationMethod.COMPS_EV_EBITDA: (152.8, 172.9, 190.7),
        ValuationMethod.COMPS_EV_SALES: (111.7, 138.6, 168.8),
        ValuationMethod.COMPS_PE: (162.8, 178.3, 196.5),
        ValuationMethod.LBO: (104.38, 117.93, 135.34),
        ValuationMethod.MONTE_CARLO: (84.3, 116.7, 164.9),
    }
    by_method = {row.method: row for row in atco_report.valuations}
    for method, expected in documented.items():
        row = by_method[method]
        assert (row.low, row.base, row.high) == pytest.approx(expected, abs=0.06), method
    assert atco_report.dcf.wacc == pytest.approx(0.0765, abs=5e-5)
    assert atco_report.dcf.wacc_result.beta == pytest.approx(1.13, abs=5e-3)
    assert atco_report.dcf.exit_multiple == pytest.approx(19.43, abs=0.005)


def test_new_exit_and_lbo_figures_follow_from_independently_built_inputs(
    atco_report, atco, params
) -> None:
    """The two rows that changed, recomputed from hand-assembled assumptions."""
    report = atco_report
    multiple, used = fenced_median([p.ev_ebitda for p in report.peers], (1.0, 60.0))
    assert used == 11 and multiple == pytest.approx(19.43, abs=0.005)

    # DCF: default drivers, the report's WACC, the hand-computed peer multiple
    drivers = dcf.default_assumptions(atco, params)
    by_hand = dcf.run_dcf(atco, replace(drivers, wacc=report.dcf.wacc, exit_multiple=multiple))
    assert report.dcf.value_per_share_exit == pytest.approx(by_hand.value_per_share_exit, rel=1e-12)
    assert by_hand.value_per_share_exit == pytest.approx(175.23, abs=0.005)

    # LBO: default financing terms, the DCF's operating drivers, exit at that multiple
    financing = lbo.default_assumptions(atco, params)
    deal = lbo.run_lbo(
        atco,
        replace(
            financing,
            revenue_growth=drivers.revenue_growth,
            ebitda_margin=drivers.ebitda_margin,
            da_pct_revenue=drivers.da_pct_revenue,
            capex_pct_revenue=drivers.capex_pct_revenue,
            nwc_pct_revenue=drivers.nwc_pct_revenue,
            tax_rate=drivers.tax_rate,
            exit_multiple=multiple,
        ),
    )
    assert report.lbo.ability_to_pay == pytest.approx(deal.ability_to_pay, rel=1e-12)
    assert [deal.ability_to_pay[t] for t in (0.25, 0.20, 0.15)] == pytest.approx(
        [104.38, 117.93, 135.34], abs=0.005
    )
    assert report.lbo.irr == pytest.approx(deal.irr, abs=1e-12)


def test_models_inside_the_report_satisfy_their_invariants(atco_report, atco) -> None:
    assert_dcf_invariants(atco_report.dcf, atco)
    assert_comps_invariants(atco_report.comps, atco)
    assert_lbo_invariants(atco_report.lbo, atco)
    assert_monte_carlo_invariants(atco_report.monte_carlo, atco_report.dcf)


def test_market_parameters_are_the_sourced_ones(atco_report) -> None:
    params = atco_report.params
    assert params == data.load_market_parameters(offline=True)
    assert params.risk_free_rate == 0.03075
    wacc = atco_report.dcf.wacc_result
    assert (wacc.risk_free_rate, wacc.equity_risk_premium, wacc.tax_rate) == (
        0.03075,
        0.0423,
        0.206,
    )
    assert atco_report.dcf.assumptions.tax_rate == 0.206
    # the LBO defaults are built with the same parameters
    assert atco_report.lbo.assumptions.risk_free_rate == 0.03075
    assert atco_report.lbo.assumptions.tax_rate == 0.206
    sources = atco_report.parameter_sources
    assert sources["risk_free_rate"].origin == "fixtures"
    assert sources["equity_risk_premium"].origin == "reference file"


# ------------------------------------------------------ wiring between models


def test_comps_peers_are_handed_to_the_dcf(atco_report, universe) -> None:
    report = atco_report
    assert report.dcf_peer_source == pipeline.PEER_SOURCE_COMPS
    assert report.peer_tickers == report.comps.peer_tickers
    assert "ATCO-A.ST" not in report.peer_tickers
    by_ticker = {peer.ticker: peer for peer in universe}
    assert report.peers == [by_ticker[t] for t in report.comps.peer_tickers]

    # one peer multiple: the exit multiple is the EV/EBITDA median comps itself applies,
    # i.e. the median of these peers after comps' bounds and its Tukey fence on logs ...
    multiple, used = fenced_median([p.ev_ebitda for p in report.peers], (1.0, 60.0))
    assert used == 11  # twelve peers, Electrolux Professional (10.5x) is outside the fence
    assert report.dcf.exit_multiple == pytest.approx(multiple, rel=1e-12)
    assert report.dcf.exit_multiple == report.comps.multiples_summary.loc["ev_ebitda", "median"]
    assert report.dcf.exit_multiple == report.comps.implied_values.loc["ev_ebitda", "multiple_base"]
    assert report.dcf.exit_multiple_source == "comps median EV/EBITDA (11 peers)"
    raw_median = statistics.median(p.ev_ebitda for p in report.peers)
    assert report.dcf.exit_multiple != pytest.approx(raw_median, rel=1e-3)  # 18.6x before
    # ... and the peer beta is their Hamada-unlevered median, relevered for Atlas Copco
    tax = report.dcf.assumptions.tax_rate
    unlevered = [
        p.beta / (1 + (1 - tax) * min(p.debt_to_equity, 2.0))
        for p in report.peers
        if p.beta is not None and 0 < p.beta <= 3 and p.debt_to_equity is not None
    ]
    wacc = report.dcf.wacc_result
    assert wacc.peer_count == len(unlevered) == 12
    assert wacc.peer_unlevered_beta == pytest.approx(statistics.median(unlevered), rel=1e-12)
    leverage = report.company.market.total_debt / report.company.market.market_cap
    relevered = statistics.median(unlevered) * (1 + (1 - tax) * leverage)
    assert wacc.peer_beta == pytest.approx(2 / 3 * relevered + 1 / 3, rel=1e-12)
    assert wacc.beta == pytest.approx((wacc.blume_beta + wacc.peer_beta) / 2, rel=1e-12)
    assert any(
        "12 peers selected by comps" in note and "comps EV/EBITDA median" in note
        for note in report.notes
    )


def test_lbo_exits_at_the_dcf_exit_multiple(atco_report) -> None:
    report = atco_report
    assert report.lbo.exit_multiple == report.dcf.exit_multiple
    assert report.lbo.assumptions.exit_multiple == report.dcf.exit_multiple
    assert report.lbo.exit_multiple != pytest.approx(report.lbo.entry_multiple, rel=0.05)
    assert "exit multiple of the DCF" in report.lbo_exit_multiple_source
    assert report.lbo.exit_enterprise_value == pytest.approx(
        report.dcf.exit_multiple * report.lbo.exit_ebitda, rel=1e-12
    )


def test_monte_carlo_simulates_around_the_dcf_of_the_report(atco_report) -> None:
    mc, base = atco_report.monte_carlo, atco_report.dcf
    assert mc.base_value == base.value_per_share_gordon
    assert mc.base_value_exit == base.value_per_share_exit
    assert mc.base_inputs["wacc"] == base.wacc
    assert mc.base_inputs["revenue_growth"] == base.assumptions.revenue_growth[0]
    assert mc.config.seed == 42 and mc.n_non_finite == 0
    assert abs(mc.median / base.value_per_share_gordon - 1) < 0.02


def test_dcf_can_fall_back_to_the_full_universe(atco, universe, params, atco_report) -> None:
    report = pipeline.value_company(
        atco, universe, params, use_comps_peers=False, n_simulations=500
    )
    assert report.dcf_peer_source == pipeline.PEER_SOURCE_UNIVERSE
    assert "industry: Specialty Industrial Machinery" in report.dcf.exit_multiple_source
    direct = dcf.run_dcf(atco, None, params, universe)
    assert report.dcf.value_per_share_gordon == direct.value_per_share_gordon
    assert report.dcf.exit_multiple == direct.exit_multiple != atco_report.dcf.exit_multiple
    assert report.lbo.exit_multiple == direct.exit_multiple
    assert report.comps.peer_tickers == atco_report.comps.peer_tickers  # comps unaffected
    assert any("switched off" in note for note in report.notes)


def test_thin_comps_peer_group_falls_back_to_the_universe_with_a_warning(universe, params) -> None:
    # a bank-free universe of four usable peers: enough for comps, too thin for the DCF
    target = make_company("TGT.ST", shares=1e7)
    small = [p for p in universe if p.industry == "Specialty Industrial Machinery"][:4]
    report = pipeline.value_company(target, small, params, n_simulations=200)
    assert report.comps is not None and len(report.peers) == 4
    assert report.dcf_peer_source == pipeline.PEER_SOURCE_UNIVERSE
    assert any("too thin for the DCF" in warning for warning in report.warnings)
    assert report.dcf.exit_multiple_source.startswith("own current EV")


def test_without_a_universe_only_comps_is_missing(atco, params) -> None:
    report = pipeline.value_company(atco, [], params, n_simulations=500)
    assert report.status == "partial"
    assert set(report.not_applicable) == {"comps"}
    assert report.dcf_peer_source == pipeline.PEER_SOURCE_NONE
    assert report.peers == []
    assert [row.method for row in report.valuations] == ALL_METHODS[3:]
    assert report.dcf.wacc_result.peer_beta is None
    assert report.dcf.exit_multiple_source.startswith("own current EV")
    assert any("no peer universe" in warning for warning in report.warnings)


def test_size_premium_switch(universe, params) -> None:
    small = make_company("SMALL.ST", shares=3e7)  # market cap 300 m: 2 % premium
    with_premium = pipeline.value_company(small, universe, params, n_simulations=200)
    without = pipeline.value_company(small, universe, params, n_simulations=200, size_premium=False)
    assert with_premium.dcf.wacc_result.size_premium == 0.020
    assert without.dcf.wacc_result.size_premium == 0.0
    assert with_premium.size_premium is True and without.size_premium is False
    gap = with_premium.dcf.wacc_result.cost_of_equity - without.dcf.wacc_result.cost_of_equity
    assert gap == pytest.approx(0.020, abs=1e-12)
    assert without.dcf.value_per_share_gordon > with_premium.dcf.value_per_share_gordon
    assert any("plain CAPM" in note for note in without.notes)


def test_assumption_overrides_reach_the_models(atco, universe, params) -> None:
    dcf_assumptions = replace(
        dcf.default_assumptions(atco, params), wacc=0.08, exit_multiple=11.0, terminal_growth=0.025
    )
    lbo_assumptions = replace(lbo.default_assumptions(atco, params), exit_multiple=14.0)
    report = pipeline.value_company(
        atco,
        universe,
        params,
        dcf_assumptions=dcf_assumptions,
        lbo_assumptions=lbo_assumptions,
        n_simulations=300,
        seed=9,
    )
    assert report.dcf.wacc == 0.08 and report.dcf.wacc_result is None
    assert report.dcf.exit_multiple == 11.0
    assert report.dcf.assumptions.terminal_growth == 0.025
    assert report.lbo.exit_multiple == 14.0  # an explicit LBO exit multiple is kept
    assert report.lbo_exit_multiple_source == "given in the LBO assumptions"
    assert report.monte_carlo.n_simulations == 300 and report.monte_carlo.config.seed == 9
    assert report.monte_carlo.base_inputs["wacc"] == 0.08
    # an LBO exit multiple of None is filled with the DCF's
    open_exit = pipeline.value_company(
        atco,
        universe,
        params,
        dcf_assumptions=dcf_assumptions,
        lbo_assumptions=lbo.default_assumptions(atco, params),
        run_monte_carlo=False,
    )
    assert open_exit.lbo.exit_multiple == 11.0
    assert "skipped" in open_exit.not_applicable["monte_carlo"]
    assert open_exit.monte_carlo is None and open_exit.status == "partial"


def test_lbo_keeps_exit_equal_to_entry_without_a_dcf(atco, universe, params) -> None:
    impossible = DCFAssumptions(
        revenue_growth=(0.03,) * 5,
        ebitda_margin=(0.25,) * 5,
        da_pct_revenue=0.05,
        capex_pct_revenue=0.04,
        nwc_pct_revenue=0.2,
        wacc=0.02,  # not above terminal growth
        terminal_growth=0.02,
    )
    report = pipeline.value_company(atco, universe, params, dcf_assumptions=impossible)
    assert report.dcf is None and report.monte_carlo is None
    assert report.not_applicable["dcf"].startswith("WACC not above terminal growth")
    assert report.not_applicable["monte_carlo"].startswith("requires a DCF base case")
    assert report.lbo is not None
    assert report.lbo.exit_multiple == pytest.approx(report.lbo.entry_multiple)
    assert "equal to the entry multiple" in report.lbo_exit_multiple_source
    assert report.status == "partial"


def test_a_crashing_model_does_not_take_the_report_down(
    atco, universe, params, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(pipeline.lbo_model, "run_lbo", boom)
    report = pipeline.value_company(atco, universe, params, n_simulations=200)
    assert report.lbo is None
    assert report.errors == {"lbo": "RuntimeError: synthetic failure"}
    assert any("LBO: the model failed" in warning for warning in report.warnings)
    assert "lbo" not in report.not_applicable
    assert report.dcf is not None and report.comps is not None and report.monte_carlo is not None
    assert report.status == "partial"
    assert ValuationMethod.LBO not in [row.method for row in report.valuations]


def test_caller_supplied_parameters_are_labelled(atco, universe) -> None:
    params = MarketParameters(risk_free_rate=0.04)
    report = pipeline.run_valuation(
        "ATCO-A.ST", offline=True, params=params, n_simulations=200, universe=universe
    )
    assert report.params is params
    assert report.dcf.wacc_result.risk_free_rate == 0.04
    assert report.lbo.assumptions.risk_free_rate == 0.04
    assert report.parameter_sources["risk_free_rate"].origin == "caller"
    assert report.parameter_sources["equity_risk_premium"].origin == "reference file"


def test_unknown_ticker_raises_data_unavailable() -> None:
    with pytest.raises(data.DataUnavailableError):
        pipeline.run_valuation("NOPE.ST", offline=True)


# ------------------------------- C1: one operating forecast, C2: one peer multiple


def test_default_lbo_uses_the_operating_forecast_of_the_dcf(atco_report, atco, params) -> None:
    report = atco_report
    a, d = report.lbo.assumptions, report.dcf.assumptions
    assert a.revenue_growth == d.revenue_growth and a.ebitda_margin == d.ebitda_margin
    assert (a.da_pct_revenue, a.capex_pct_revenue, a.nwc_pct_revenue, a.tax_rate) == (
        d.da_pct_revenue,
        d.capex_pct_revenue,
        d.nwc_pct_revenue,
        d.tax_rate,
    )
    # the two forecast tables agree year by year over the holding period
    deal, cash_flows = report.lbo.forecast, report.dcf.forecast
    assert list(deal.index) == list(cash_flows.index) == [2026, 2027, 2028, 2029, 2030]
    for lbo_column, dcf_column in (
        ("revenue", "revenue"),
        ("ebitda", "ebitda"),
        ("depreciation_amortization", "depreciation_amortization"),
        ("ebit", "ebit"),
        ("capex", "capex"),
        ("change_in_nwc", "delta_nwc"),
    ):
        assert deal[lbo_column].tolist() == pytest.approx(
            cash_flows[dcf_column].tolist(), rel=1e-12
        ), lbo_column
    # this really replaced the LBO's own history-based drivers ...
    own = lbo.default_assumptions(atco, params)
    assert own.revenue_growth[0] == pytest.approx(0.0520, abs=5e-5)
    assert a.revenue_growth[0] == pytest.approx(0.0600, abs=5e-5)
    assert own.ebitda_margin != a.ebitda_margin
    # ... while transaction and financing terms are the LBO defaults with sourced parameters
    for name in (
        "takeover_premium",
        "senior_debt_multiple",
        "subordinated_debt_multiple",
        "risk_free_rate",
        "senior_spread",
        "min_equity_share",
        "interest_deduction_cap_pct_ebitda",
        "target_irr_base",
    ):
        assert getattr(a, name) == getattr(own, name), name
    assert a.risk_free_rate == 0.03075
    assert any("operating forecast" in n and "the one of the DCF" in n for n in report.notes)


def test_default_lbo_follows_dcf_assumption_overrides(atco, universe, params) -> None:
    custom = replace(
        dcf.default_assumptions(atco, params),
        revenue_growth=(0.10, 0.08),
        ebitda_margin=(0.30, 0.28),
        da_pct_revenue=0.06,
        capex_pct_revenue=0.07,
        nwc_pct_revenue=0.15,
        tax_rate=0.25,
    )
    report = pipeline.value_company(
        atco, universe, params, dcf_assumptions=custom, run_monte_carlo=False
    )
    a = report.lbo.assumptions
    # a five-year holding period on a two-year DCF: the last DCF year is repeated
    assert a.revenue_growth == (0.10, 0.08, 0.08, 0.08, 0.08)
    assert a.ebitda_margin == (0.30, 0.28, 0.28, 0.28, 0.28)
    assert (a.da_pct_revenue, a.capex_pct_revenue, a.nwc_pct_revenue, a.tax_rate) == (
        0.06,
        0.07,
        0.15,
        0.25,
    )


def test_lbo_assumptions_from_dcf_keeps_the_holding_period_and_the_deal_terms(atco) -> None:
    base = replace(
        lbo.default_assumptions(atco),
        revenue_growth=(0.01, 0.01, 0.01),
        ebitda_margin=(0.2, 0.2, 0.2),
        takeover_premium=0.15,
        exit_multiple=9.0,
    )
    drivers = dcf.default_assumptions(atco)
    aligned = pipeline.lbo_assumptions_from_dcf(base, drivers)
    assert aligned.holding_period == 3
    assert aligned.revenue_growth == drivers.revenue_growth[:3]
    assert aligned.ebitda_margin == drivers.ebitda_margin[:3]
    assert aligned.tax_rate == drivers.tax_rate
    assert (aligned.takeover_premium, aligned.exit_multiple) == (0.15, 9.0)
    expected = replace(
        base,
        revenue_growth=drivers.revenue_growth[:3],
        ebitda_margin=drivers.ebitda_margin[:3],
        da_pct_revenue=drivers.da_pct_revenue,
        capex_pct_revenue=drivers.capex_pct_revenue,
        nwc_pct_revenue=drivers.nwc_pct_revenue,
        tax_rate=drivers.tax_rate,
    )
    assert aligned == expected


def test_caller_supplied_lbo_assumptions_are_respected_as_given(atco, universe, params) -> None:
    own = replace(
        lbo.default_assumptions(atco, params),
        revenue_growth=(0.01, 0.02, 0.03),
        ebitda_margin=(0.22, 0.23, 0.24),
        da_pct_revenue=0.031,
        capex_pct_revenue=0.029,
        nwc_pct_revenue=0.11,
        tax_rate=0.19,
        takeover_premium=0.10,
    )
    report = pipeline.value_company(
        atco, universe, params, lbo_assumptions=own, run_monte_carlo=False
    )
    # only the open exit multiple is filled in, with the DCF's
    assert report.lbo.assumptions == replace(own, exit_multiple=report.dcf.exit_multiple)
    assert report.lbo.forecast.index.tolist() == [2026, 2027, 2028]
    assert report.lbo.forecast["revenue_growth"].tolist() == [0.01, 0.02, 0.03]
    assert report.dcf.assumptions.revenue_growth != own.revenue_growth
    assert not any("the one of the DCF" in note for note in report.notes)
    closed = pipeline.value_company(
        atco,
        universe,
        params,
        lbo_assumptions=replace(own, exit_multiple=10.0),
        run_monte_carlo=False,
    )
    assert closed.lbo.assumptions == replace(own, exit_multiple=10.0)


def test_lbo_without_a_dcf_keeps_its_own_history_based_forecast(atco, universe, params) -> None:
    impossible = replace(dcf.default_assumptions(atco, params), wacc=0.02)
    report = pipeline.value_company(atco, universe, params, dcf_assumptions=impossible)
    assert report.dcf is None
    assert report.lbo.assumptions == lbo.default_assumptions(atco, params)
    assert any("own history-based operating forecast" in note for note in report.notes)


def synthetic_sector(multiples: list[float]) -> list:
    return [
        make_peer(
            f"P{i}.ST",
            enterprise_value=m * 100.0,
            ebitda=100.0,
            market_cap=1000.0 + 10.0 * i,
            beta=0.9 + 0.02 * i,
        )
        for i, m in enumerate(multiples)
    ]


def test_dcf_exit_multiple_is_the_fenced_comps_median(params) -> None:
    target = make_company("TGT.ST", shares=1e7)
    # seven ordinary multiples and one outlier that comps' fence removes
    multiples = [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 55.0]
    report = pipeline.value_company(target, synthetic_sector(multiples), params, n_simulations=200)
    assert report.comps.multiples_summary.loc["ev_ebitda", "n"] == 7
    expected, used = fenced_median(multiples, (1.0, 60.0))
    assert (expected, used) == (11.0, 7)
    assert report.dcf.exit_multiple == pytest.approx(11.0)
    assert report.dcf.exit_multiple != pytest.approx(statistics.median(multiples))  # 11.5 raw
    assert report.dcf.exit_multiple_source == "comps median EV/EBITDA (7 peers)"
    assert report.lbo.exit_multiple == report.dcf.exit_multiple
    assert report.dcf.wacc_result.peer_count == 8  # the beta still uses every peer


def test_comps_median_is_still_clipped_to_the_dcf_band(params) -> None:
    target = make_company("TGT.ST", shares=1e7)
    for multiples, clipped in (([22.0, 24.0, 25.0, 26.0, 28.0, 30.0], 20.0), ([2.0] * 6, 4.0)):
        report = pipeline.value_company(
            target, synthetic_sector(multiples), params, run_monte_carlo=False
        )
        median = statistics.median(multiples)
        assert report.comps.multiples_summary.loc["ev_ebitda", "median"] == pytest.approx(median)
        assert report.dcf.exit_multiple == clipped
        assert f"{median:.1f}x clipped to {clipped:.1f}x" in report.dcf.exit_multiple_source
        assert any("exit multiple" in note and "clipped" in note for note in report.dcf.notes)
        assert report.lbo.exit_multiple == clipped


def test_explicit_dcf_exit_multiple_beats_the_comps_median(atco, universe, params) -> None:
    given = replace(dcf.default_assumptions(atco, params), exit_multiple=13.0)
    report = pipeline.value_company(
        atco, universe, params, dcf_assumptions=given, run_monte_carlo=False
    )
    assert report.dcf.exit_multiple == 13.0 and report.dcf.exit_multiple_source == "assumption"
    assert report.lbo.exit_multiple == 13.0
    assert any("given in the assumptions" in note for note in report.notes)


def test_dcf_does_not_adopt_a_comps_multiple_that_comps_itself_rejects(params) -> None:
    target = make_company("TGT.ST", shares=1e7)
    peers = [
        make_peer(
            f"P{i}.ST",
            enterprise_value=(8.0 + i) * 100.0,
            ebitda=100.0 if i < 2 else -10.0,
            market_cap=1000.0 + i,
            beta=1.0 + 0.01 * i,
        )
        for i in range(6)
    ]
    report = pipeline.value_company(target, peers, params, run_monte_carlo=False)
    assert "ev_ebitda" in report.comps.skipped  # comps: too few observations
    assert report.comps.multiples_summary.loc["ev_ebitda", "n"] == 2
    assert "comps median" not in report.dcf.exit_multiple_source


# --------------------------------------------------------- bank, USD reporter


def test_bank_gets_pe_comps_only(seb_report) -> None:
    report = seb_report
    assert isinstance(report, ValuationReportLike)
    assert report.status == "partial"
    assert set(report.not_applicable) == {"dcf", "lbo", "monte_carlo"}
    assert report.not_applicable["dcf"].startswith("financial sector")
    assert "financial or real estate" in report.not_applicable["lbo"]
    assert "requires a DCF base case" in report.not_applicable["monte_carlo"]
    assert report.errors == {}
    assert (report.dcf, report.lbo, report.monte_carlo) == (None, None, None)
    assert [row.method for row in report.valuations] == [ValuationMethod.COMPS_PE]
    (row,) = report.valuations
    assert 0 < row.low <= row.base <= row.high
    assert 0.5 < row.base / report.price < 2.0
    assert report.comps.candidate_scope == "industry_family"
    assert report.dcf_peer_source == pipeline.PEER_SOURCE_NONE
    assert all(peer.is_financial for peer in report.peers)
    summary = report.summary()
    assert summary["dcf"] is None and summary["lbo"] is None and summary["monte_carlo"] is None
    assert summary["status"] == "partial"


@pytest.mark.parametrize(("ticker", "currency"), [("ABB.ST", "USD"), ("HEXA-B.ST", "EUR")])
def test_foreign_currency_reporter_runs_end_to_end(reports, ticker: str, currency: str) -> None:
    report = reports[ticker]
    assert report.status == "ok" and report.errors == {}
    assert report.currency == "SEK"
    assert report.company.profile.reporting_currency == currency
    assert [row.method for row in report.valuations] == ALL_METHODS
    for row in report.valuations:
        assert row.currency == "SEK"
        assert all(math.isfinite(v) for v in (row.low, row.base, row.high))
        assert 0.2 < row.base / report.price < 2.0  # values are in SEK, like the price
    summary = report.summary()
    assert summary["reporting_currency"] == currency
    assert summary["fx_rate"] == report.company.profile.fx_rate > 5
    assert f"Statements reported in {currency}" in pipeline.summary_text(report)


def test_company_trading_in_euro_is_valued_in_euro(reports) -> None:
    report = reports["VSURE.ST"]
    assert report.currency == "EUR" and report.errors == {}
    assert {row.currency for row in report.valuations} == {"EUR"}
    assert report.comps.skipped.keys() == {"pe"}  # a loss-making year
    # the size premium is looked up with the SEK market cap (124 bn), not 11 bn EUR
    assert report.dcf.wacc_result.size_premium == 0.0
    assert "124.17 bn SEK" in report.dcf.wacc_result.size_premium_basis
    exit_row = next(r for r in report.valuations if r.method is ValuationMethod.DCF_EXIT_MULTIPLE)
    assert 0.5 < exit_row.base / report.price < 2.0


# ------------------------------------------------------------------ summaries


def test_summary_is_json_ready_and_mirrors_the_models(atco_report) -> None:
    summary = atco_report.summary()
    text = json.dumps(summary, allow_nan=False)  # raises on NaN / inf / non-JSON types
    assert json.loads(text) == summary

    report = atco_report
    assert (summary["ticker"], summary["currency"], summary["status"]) == ("ATCO-A.ST", "SEK", "ok")
    assert summary["price"] == 192.25
    assert summary["market_cap"] == report.company.market.market_cap
    assert summary["enterprise_value"] == report.company.market.enterprise_value
    assert summary["net_debt"] == report.company.market.net_debt
    assert summary["latest_fiscal_year"] == 2025 and summary["as_of"] == "2026-11-13"
    assert [row["method"] for row in summary["valuations"]] == [m.value for m in ALL_METHODS]
    for row, result in zip(summary["valuations"], report.valuations, strict=True):
        assert (row["low"], row["base"], row["high"]) == (result.low, result.base, result.high)
        assert row["base_to_price"] == pytest.approx(result.base / 192.25)
        assert row["label"] == result.label
    assert summary["dcf"]["value_per_share_gordon"] == report.dcf.value_per_share_gordon
    assert summary["dcf"]["wacc"] == report.dcf.wacc
    assert summary["dcf"]["beta"] == report.dcf.wacc_result.beta
    assert summary["dcf"]["peer_source"] == "comps"
    assert summary["dcf"]["wacc_source"] == "CAPM"
    assert summary["lbo"]["irr"] == report.lbo.irr and summary["lbo"]["moic"] == report.lbo.moic
    assert summary["lbo"]["exit_multiple"] == report.dcf.exit_multiple
    assert summary["lbo"]["ability_to_pay"] == {
        "0.25": report.lbo.ability_to_pay[0.25],
        "0.20": report.lbo.ability_to_pay[0.20],
        "0.15": report.lbo.ability_to_pay[0.15],
    }
    assert summary["monte_carlo"]["percentiles"]["p50"] == report.monte_carlo.percentiles[50]
    assert summary["monte_carlo"]["mean"] == report.monte_carlo.mean
    assert summary["comps"]["chosen_k"] == report.comps.chosen_k
    assert summary["comps"]["n_peers"] == len(report.comps.peers) == len(summary["peers"])
    assert summary["comps"]["multiples"]["ev_ebitda"]["multiple_base"] == pytest.approx(
        report.comps.multiples_summary.loc["ev_ebitda", "median"]
    )
    assert summary["market_parameters"]["risk_free_rate"]["value"] == 0.03075
    assert summary["settings"] == {
        "n_simulations": 10_000,
        "seed": 42,
        "size_premium": True,
        "dcf_peer_source": "comps",
    }
    assert set(summary["timing"]) == {"generated_at", "seconds"}
    assert set(METHOD_KEYS) == {"dcf", "comps", "lbo", "monte_carlo"}


def test_two_runs_differ_only_in_the_timing_block(atco, universe, params) -> None:
    first = pipeline.value_company(atco, universe, params, n_simulations=1_000).summary()
    second = pipeline.value_company(atco, universe, params, n_simulations=1_000).summary()
    first.pop("timing")
    second.pop("timing")
    assert first == second


def test_summary_text_is_ascii_and_shows_the_headline_numbers(atco_report, seb_report) -> None:
    text = pipeline.summary_text(atco_report)
    assert text.isascii()
    assert "Atlas Copco AB (publ) (ATCO-A.ST)" in text
    assert "192.25" in text and "WACC 7.65%" in text
    for row in atco_report.valuations:
        assert row.label in text
        assert f"{row.base:,.2f}" in text
    assert "Comps peers (12)" in text and "Market parameters" in text
    bank = pipeline.summary_text(seb_report)
    assert "Not applicable" in bank and "DCF: financial sector" in bank
    assert "WACC" not in bank


# ------------------------------------------------------------- import hygiene


def _python(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-W", "error", "-c", code], capture_output=True, text=True, timeout=120
    )


def test_package_imports_cleanly_with_warnings_as_errors() -> None:
    done = _python(
        "import valuation_engine.pipeline, valuation_engine.viz, valuation_engine.export"
    )
    assert done.returncode == 0, done.stderr[-400:]


def test_valuing_a_company_does_not_need_the_chart_stack() -> None:
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, valuation_engine.pipeline; sys.exit('matplotlib' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0


# ------------------------------------------------------ statements about FX


def applied_statement_rate(report) -> float:
    """SEK per USD actually applied to ABB's 2025 income statement."""
    raw = json.loads((FIXTURES / "companies" / "ABB.ST.json").read_text(encoding="utf-8"))
    table = raw["statements"]["income_stmt"]
    column = table["periods"].index("2025-12-31")
    revenue_usd = table["rows"]["Total Revenue"][column]
    return report.company.statements.income.loc[2025, "revenue"] / revenue_usd


def test_summary_text_does_not_present_the_spot_rate_as_the_statement_conversion_rate(
    reports,
) -> None:
    report = reports["ABB.ST"]
    spot = report.company.profile.fx_rate
    applied = applied_statement_rate(report)
    assert abs(applied - spot) > 0.2  # 9.82 applied, 10.05 is the spot rate
    text = " ".join(pipeline.summary_text(report).split())
    assert f"converted to SEK at {spot:.4f}" not in text
    # the summary now names the rates that were applied, and what the spot rate is for
    assert f"income statement and cash flow at the year average (FY2025: {applied:.4f})" in text
    assert "balance sheet at the year-end rate (FY2025: 9.2013)" in text
    assert f"Market snapshot at the spot rate {spot:.4f}" in text


def test_reports_can_be_pickled(atco_report, seb_report) -> None:
    import pickle

    for report in (atco_report, seb_report):
        clone = pickle.loads(pickle.dumps(report))
        assert type(clone) is pipeline.ValuationReport
        first, second = clone.summary(), report.summary()
        first.pop("timing"), second.pop("timing")
        assert first == second
