"""WACC: beta regression, Blume, Hamada, CAPM arithmetic, cost of debt and weights."""

from __future__ import annotations

import statistics

import numpy as np
import pytest

from support import make_company, make_history, make_peer
from valuation_engine.core.types import MarketParameters, NotApplicableError
from valuation_engine.models import dcf

PARAMS = MarketParameters(risk_free_rate=0.03, equity_risk_premium=0.05, tax_rate=0.20)
REL = 1e-10


# ------------------------------------------------------------- beta regression


@pytest.mark.parametrize("beta", [0.4, 1.0, 1.37, 2.5])
def test_regression_recovers_an_exact_beta(beta: float) -> None:
    raw, r_squared, n_obs = dcf.regression_beta_stats(make_history(beta, n_returns=120))
    assert raw == pytest.approx(beta, rel=1e-9)
    assert r_squared == pytest.approx(1.0, rel=1e-9)
    assert n_obs == 120


def test_regression_beta_is_exact_with_orthogonal_noise_and_r_squared_matches() -> None:
    history = make_history(1.25, n_returns=200, noise=0.03, seed=3)
    raw, r_squared, n_obs = dcf.regression_beta_stats(history)
    assert raw == pytest.approx(1.25, rel=1e-9)
    # independent R squared: squared correlation of the two return series
    returns = history.prices.pct_change().dropna()
    expected = float(np.corrcoef(returns["stock"], returns["benchmark"])[0, 1] ** 2)
    assert 0.0 < expected < 1.0
    assert r_squared == pytest.approx(expected, rel=1e-9)
    assert n_obs == 200


def test_regression_needs_52_weekly_returns_and_a_moving_benchmark() -> None:
    assert dcf.regression_beta_stats(None) == (None, None, 0)
    short = dcf.regression_beta_stats(make_history(1.0, n_returns=51))
    assert short[0] is None and short[2] == 51
    assert dcf.regression_beta_stats(make_history(1.0, n_returns=52))[0] == pytest.approx(1.0)
    flat = make_history(1.0, n_returns=60)
    flat.prices["benchmark"] = 100.0
    assert dcf.regression_beta_stats(flat)[0] is None


# ------------------------------------------------------------ CAPM arithmetic


def test_capm_blume_weights_and_wacc_by_hand() -> None:
    company = make_company(history=make_history(1.3), interest=15.0, debt=300.0)
    wacc = dcf.compute_wacc(company, PARAMS, size_premium=0.0)

    blume = 2 / 3 * 1.3 + 1 / 3
    cost_of_equity = 0.03 + blume * 0.05
    pre_tax_kd = 15.0 / 300.0  # inside the band rf + 0.5 % .. rf + 6 %
    equity_weight = 1000.0 / 1300.0
    expected = equity_weight * cost_of_equity + (1 - equity_weight) * pre_tax_kd * (1 - 0.20)

    assert wacc.raw_beta == pytest.approx(1.3, rel=1e-9)
    assert wacc.blume_beta == pytest.approx(blume, rel=1e-9)
    assert wacc.beta == pytest.approx(blume, rel=1e-9)
    assert wacc.beta_method == "own Blume-adjusted regression beta"
    assert wacc.size_premium == 0.0
    assert wacc.cost_of_equity == pytest.approx(cost_of_equity, rel=1e-9)
    assert wacc.pre_tax_cost_of_debt == pytest.approx(pre_tax_kd, rel=REL)
    assert wacc.after_tax_cost_of_debt == pytest.approx(pre_tax_kd * 0.8, rel=REL)
    assert wacc.debt_to_equity == pytest.approx(0.3, rel=REL)
    assert wacc.equity_weight == pytest.approx(equity_weight, rel=REL)
    assert wacc.equity_weight + wacc.debt_weight == pytest.approx(1.0, abs=1e-15)
    assert wacc.wacc == pytest.approx(expected, rel=1e-9)
    assert (wacc.risk_free_rate, wacc.equity_risk_premium, wacc.tax_rate) == (0.03, 0.05, 0.20)
    assert wacc.benchmark == "^TEST"
    assert wacc.as_dict()["wacc"] == wacc.wacc


def test_run_dcf_uses_the_computed_wacc_and_the_assumed_tax_rate() -> None:
    company = make_company(history=make_history(1.1))
    assumptions = dcf.default_assumptions(company, PARAMS)
    result = dcf.run_dcf(company, assumptions, PARAMS, size_premium=0.0)
    direct = dcf.compute_wacc(company, PARAMS, tax_rate=assumptions.tax_rate, size_premium=0.0)
    assert result.wacc == direct.wacc
    assert result.wacc_result is not None
    assert result.wacc_result.tax_rate == assumptions.tax_rate


def test_market_beta_of_one_without_history_or_peers() -> None:
    wacc = dcf.compute_wacc(make_company(), PARAMS, size_premium=0.0)
    assert wacc.raw_beta is None and wacc.blume_beta is None and wacc.peer_beta is None
    assert wacc.beta == 1.0
    assert wacc.cost_of_equity == pytest.approx(0.03 + 0.05)
    assert any("no usable price history" in note for note in wacc.notes)


@pytest.mark.parametrize(("raw", "used"), [(0.1, 0.5), (2.9, 2.0)])
def test_final_beta_is_clipped_to_the_band(raw: float, used: float) -> None:
    wacc = dcf.compute_wacc(make_company(history=make_history(raw)), PARAMS, size_premium=0.0)
    assert wacc.blume_beta == pytest.approx(2 / 3 * raw + 1 / 3, rel=1e-9)
    assert wacc.beta == used
    assert any("clipped" in note for note in wacc.notes)


@pytest.mark.parametrize("raw", [-0.4, 3.5])
def test_noisy_own_beta_outside_the_raw_band_is_discarded(raw: float) -> None:
    wacc = dcf.compute_wacc(make_company(history=make_history(raw)), PARAMS, size_premium=0.0)
    assert wacc.raw_beta == pytest.approx(raw, rel=1e-9)
    assert wacc.blume_beta is None
    assert wacc.beta == 1.0


# ------------------------------------------------------------------- Hamada


def hamada_peers() -> list:
    betas = [0.8, 1.0, 1.2, 1.4, 1.6]
    debts = [0.0, 250.0, 500.0, 1000.0, 3000.0]  # D/E 0, 0.25, 0.5, 1, 3 (clipped to 2)
    return [
        make_peer(f"P{i}.ST", beta=b, total_debt=d, market_cap=1000.0)
        for i, (b, d) in enumerate(zip(betas, debts, strict=True))
    ]


def test_peer_beta_is_unlevered_relevered_and_blume_adjusted_by_hand() -> None:
    company = make_company()  # D/E = 300 / 1000
    wacc = dcf.compute_wacc(company, PARAMS, hamada_peers(), size_premium=0.0)
    tax = 0.20
    leverage = [0.0, 0.25, 0.5, 1.0, 2.0]
    unlevered = [
        b / (1 + (1 - tax) * d) for b, d in zip([0.8, 1.0, 1.2, 1.4, 1.6], leverage, strict=True)
    ]
    median = statistics.median(unlevered)
    relevered = median * (1 + (1 - tax) * 0.3)
    assert wacc.peer_count == 5
    assert wacc.peer_group == "industry: Specialty Industrial Machinery"
    assert wacc.peer_unlevered_beta == pytest.approx(median, rel=REL)
    assert wacc.peer_relevered_beta == pytest.approx(relevered, rel=REL)
    assert wacc.peer_beta == pytest.approx(2 / 3 * relevered + 1 / 3, rel=REL)
    assert wacc.beta == pytest.approx(2 / 3 * relevered + 1 / 3, rel=REL)
    assert wacc.beta_method == "relevered peer beta (Blume-adjusted)"


@pytest.mark.parametrize("leverage", [0.0, 0.3, 1.0, 1.9])
def test_hamada_unlever_relever_round_trip(leverage: float) -> None:
    """Peers with the target's own D/E: relevering returns the levered peer beta."""
    company = make_company(debt=leverage * 1000.0, cash=0.0)
    peers = [
        make_peer(f"P{i}.ST", beta=1.15, total_debt=leverage * 800.0, market_cap=800.0)
        for i in range(5)
    ]
    wacc = dcf.compute_wacc(company, PARAMS, peers, size_premium=0.0)
    assert wacc.peer_unlevered_beta == pytest.approx(1.15 / (1 + 0.8 * leverage), rel=REL)
    assert wacc.peer_relevered_beta == pytest.approx(1.15, rel=REL)


def test_own_and_peer_beta_are_averaged() -> None:
    company = make_company(history=make_history(1.3))
    peers = [make_peer(f"P{i}.ST", beta=0.9, total_debt=300.0, market_cap=1000.0) for i in range(6)]
    wacc = dcf.compute_wacc(company, PARAMS, peers, size_premium=0.0)
    own = 2 / 3 * 1.3 + 1 / 3
    peer = 2 / 3 * 0.9 + 1 / 3
    assert wacc.peer_relevered_beta == pytest.approx(0.9, rel=REL)
    assert wacc.beta == pytest.approx((own + peer) / 2, rel=1e-9)
    assert wacc.beta_method.startswith("average")


def test_unusable_peers_do_not_count() -> None:
    company = make_company()
    peers = [make_peer(f"P{i}.ST", beta=1.0) for i in range(4)]
    peers += [
        make_peer("NOBETA.ST", beta=None),
        make_peer("NEG.ST", beta=-0.2),
        make_peer("WILD.ST", beta=9.0),
        make_peer("NODEBT.ST", total_debt=None),
        make_peer("BANK.ST", sector="Financial Services", industry="Banks - Regional"),
        make_peer("TEST.ST", beta=1.0),  # the company itself
    ]
    wacc = dcf.compute_wacc(company, PARAMS, peers, size_premium=0.0)
    assert wacc.peer_count == 0 and wacc.peer_beta is None
    assert wacc.beta == 1.0
    assert any("fewer than 5 usable peers" in note for note in wacc.notes)
    with_fifth = dcf.compute_wacc(
        company, PARAMS, [*peers, make_peer("P4.ST", beta=1.0)], size_premium=0.0
    )
    assert with_fifth.peer_count == 5


def test_sector_peers_are_the_fallback_for_a_thin_industry() -> None:
    company = make_company()
    peers = [make_peer(f"P{i}.ST", beta=1.2, industry="Tools") for i in range(5)]
    wacc = dcf.compute_wacc(company, PARAMS, peers, size_premium=0.0)
    assert wacc.peer_group == "sector: Industrials"
    assert wacc.peer_count == 5


# ---------------------------------------------------------------- cost of debt


@pytest.mark.parametrize(
    ("interest", "expected", "clipped"),
    [(15.0, 0.05, False), (3.0, 0.035, True), (60.0, 0.09, True), (27.0, 0.09, False)],
)
def test_cost_of_debt_is_interest_over_debt_clipped_to_the_spread_band(
    interest: float, expected: float, clipped: bool
) -> None:
    wacc = dcf.compute_wacc(make_company(interest=interest, debt=300.0), PARAMS, size_premium=0.0)
    assert wacc.pre_tax_cost_of_debt == pytest.approx(expected, rel=REL)
    assert ("clipped" in wacc.cost_of_debt_source) is clipped
    assert wacc.after_tax_cost_of_debt == pytest.approx(expected * 0.8, rel=REL)


def test_cost_of_debt_uses_the_average_of_the_last_two_year_end_balances() -> None:
    company = make_company(interest=16.0)
    company.statements.balance["total_debt"] = [900.0, 900.0, 300.0, 500.0]
    wacc = dcf.compute_wacc(company, PARAMS, size_premium=0.0)
    assert wacc.pre_tax_cost_of_debt == pytest.approx(16.0 / 400.0, rel=REL)


def test_cost_of_debt_fallback_without_interest_or_debt() -> None:
    for company in (make_company(interest=0.0), make_company(debt=0.0, cash=0.0)):
        wacc = dcf.compute_wacc(company, PARAMS, size_premium=0.0)
        assert wacc.pre_tax_cost_of_debt == pytest.approx(0.03 + 0.02)
        assert wacc.cost_of_debt_source.startswith("fallback")


# -------------------------------------------------------------------- weights


def test_all_equity_company_has_wacc_equal_to_cost_of_equity() -> None:
    wacc = dcf.compute_wacc(make_company(debt=0.0, cash=0.0), PARAMS, size_premium=0.0)
    assert wacc.debt_weight == 0.0 and wacc.equity_weight == 1.0
    assert wacc.wacc == pytest.approx(wacc.cost_of_equity, rel=REL)


def test_distressed_leverage_is_clipped_for_the_weights() -> None:
    wacc = dcf.compute_wacc(make_company(debt=9000.0), PARAMS, size_premium=0.0)
    assert wacc.debt_to_equity == 2.0
    assert wacc.equity_weight == pytest.approx(1 / 3, rel=REL)
    assert wacc.debt_weight == pytest.approx(2 / 3, rel=REL)
    assert any("clipped" in note for note in wacc.notes)


def test_negative_reported_debt_counts_as_zero() -> None:
    company = make_company()
    company.market.total_debt = -50.0
    wacc = dcf.compute_wacc(company, PARAMS, size_premium=0.0)
    assert wacc.total_debt == 0.0 and wacc.debt_weight == 0.0


def test_wacc_requires_a_positive_market_cap() -> None:
    company = make_company()
    company.market.market_cap = 0.0
    with pytest.raises(NotApplicableError, match="market cap"):
        dcf.compute_wacc(company, PARAMS)


# --------------------------------------------------------------- size premium


@pytest.mark.parametrize(
    ("market_cap", "premium"),
    [
        (25e9, 0.0),
        (20e9, 0.0),
        (19.9e9, 0.005),
        (5e9, 0.005),
        (4.9e9, 0.010),
        (1e9, 0.010),
        (0.9e9, 0.020),
        (200e6, 0.020),
        (199e6, 0.030),
        (1e6, 0.030),
    ],
)
def test_size_premium_schedule(market_cap: float, premium: float) -> None:
    company = make_company(shares=market_cap / 10.0, debt=0.0, cash=0.0)
    wacc = dcf.compute_wacc(company, PARAMS)
    assert wacc.size_premium == premium
    assert wacc.cost_of_equity == pytest.approx(0.03 + 1.0 * 0.05 + premium, rel=REL)
    plain = dcf.compute_wacc(company, PARAMS, size_premium=0.0)
    assert wacc.cost_of_equity - plain.cost_of_equity == pytest.approx(premium, abs=1e-15)


def test_size_premium_uses_the_sek_market_cap_of_the_own_peer_snapshot() -> None:
    # 300 m EUR is "small" against SEK thresholds, 3.3 bn SEK is not.
    company = make_company("EUR.ST", shares=30e6, debt=0.0, cash=0.0, currency="EUR")
    alone = dcf.compute_wacc(company, PARAMS)
    assert alone.size_premium == 0.020
    assert any("size premium" in note for note in alone.notes)
    own_row = make_peer("EUR.ST", market_cap=3.3e9)
    with_row = dcf.compute_wacc(company, PARAMS, [own_row])
    assert with_row.size_premium == 0.010


def test_size_premium_can_be_switched_off_by_emptying_the_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    company = make_company(shares=1e6, debt=0.0, cash=0.0)  # market cap 10 m
    assert dcf.compute_wacc(company, PARAMS).size_premium == 0.030
    monkeypatch.setattr(dcf, "SIZE_PREMIUM_SCHEDULE", ())
    plain = dcf.compute_wacc(company, PARAMS)
    assert plain.size_premium == 0.0
    assert plain.cost_of_equity == pytest.approx(0.03 + 0.05)
