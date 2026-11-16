"""LBO: funding, debt schedule, returns, attribution and ability to pay."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from invariants import assert_lbo_invariants
from support import make_company
from valuation_engine.core.types import MarketParameters, NotApplicableError, ValuationMethod
from valuation_engine.models import lbo
from valuation_engine.models.lbo import LBOAssumptions, compute_irr

REL = 1e-10


def assumptions(**changes: object) -> LBOAssumptions:
    """Three years, flat 20 % margin, no sizing caps unless a test switches them on."""
    base: dict[str, object] = {
        "revenue_growth": (0.05, 0.05, 0.05),
        "ebitda_margin": (0.20, 0.20, 0.20),
        "da_pct_revenue": 0.04,
        "capex_pct_revenue": 0.05,
        "nwc_pct_revenue": 0.10,
        "tax_rate": 0.25,
        "interest_deduction_cap_pct_ebitda": None,
        "takeover_premium": 0.20,
        "senior_debt_multiple": 3.0,
        "subordinated_debt_multiple": 1.0,
        "risk_free_rate": 0.03,
        "min_equity_share": 0.0,
        "min_interest_coverage": 0.0,
        "min_debt_service_coverage": 0.0,
        "exit_multiple": 7.0,
    }
    base.update(changes)
    return LBOAssumptions(**base)  # type: ignore[arg-type]


def company(**kwargs: object):
    """Latest revenue 1,000, EBITDA 200, 100 shares at 10, debt 300, cash 50."""
    kwargs.setdefault("years", (2024, 2025))
    kwargs.setdefault("revenue", (900.0, 1000.0))
    return make_company(**kwargs)  # type: ignore[arg-type]


# ------------------------------------------------------------------------ IRR


@pytest.mark.parametrize(
    ("flows", "expected"),
    [
        ([-100.0, 110.0], 0.10),
        ([-100.0, 0.0, 121.0], 0.10),
        ([-100.0, 20.0, 20.0, 120.0], 0.20),  # par bond with a 20 % coupon
        ([-100.0, 50.0, 50.0], 0.0),
        ([-100.0, 50.0], -0.50),
        ([-1.0, 1000.0], 999.0),
        ([-100.0, 0.0, 0.0, 0.0, 0.0, 248.832], 0.20),  # 1.2 ** 5
        ([100.0, -110.0], 0.10),  # a loan seen from the borrower
    ],
)
def test_irr_textbook_vectors(flows: list[float], expected: float) -> None:
    assert compute_irr(flows) == pytest.approx(expected, abs=1e-9)


def test_irr_solves_the_npv_equation() -> None:
    flows = [-100.0, 60.0, 60.0]
    # 100 x^2 - 60 x - 60 = 0 with x = 1 + r
    expected = (60.0 + math.sqrt(60.0**2 + 4 * 100.0 * 60.0)) / 200.0 - 1.0
    irr = compute_irr(flows)
    assert irr == pytest.approx(expected, abs=1e-10)
    assert sum(c / (1 + irr) ** t for t, c in enumerate(flows)) == pytest.approx(0.0, abs=1e-8)


@pytest.mark.parametrize(
    ("flows", "expected"),
    [
        ([-100.0, 0.0, 0.0], -1.0),  # total loss
        ([-100.0, -5.0], -1.0),
    ],
)
def test_irr_total_loss(flows: list[float], expected: float) -> None:
    assert compute_irr(flows) == expected


@pytest.mark.parametrize(
    "flows",
    [[100.0, 10.0], [0.0, 0.0], [-100.0], [], [-100.0, math.nan], [-100.0, math.inf]],
)
def test_irr_is_nan_when_undefined(flows: list[float]) -> None:
    assert math.isnan(compute_irr(flows))


def test_irr_with_two_sign_changes_returns_one_of_the_roots() -> None:
    irr = compute_irr([-100.0, 230.0, -132.0])
    assert any(irr == pytest.approx(root, abs=1e-9) for root in (0.10, 0.20))


# ------------------------------------------------- hand-computed deal economics


def test_all_equity_zero_growth_case_by_hand() -> None:
    a = LBOAssumptions(
        revenue_growth=(0.0,) * 5,
        ebitda_margin=(0.20,) * 5,
        da_pct_revenue=0.04,
        capex_pct_revenue=0.05,
        nwc_pct_revenue=0.10,
        tax_rate=0.20,
        takeover_premium=0.0,
        transaction_fee_pct=0.0,
        financing_fee_pct=0.0,
        senior_debt_multiple=0.0,
        subordinated_debt_multiple=0.0,
        min_cash_pct_revenue=0.0,
    )
    target = company()
    result = lbo.run_lbo(target, a)

    # entry: buy the equity (1,000), repay the debt (300), use the cash (50)
    sponsor_equity = 1000.0 + 300.0 - 50.0
    entry_ev = 1000.0 + 300.0 - 50.0
    assert result.sponsor_equity == pytest.approx(sponsor_equity, rel=REL)
    assert result.entry_enterprise_value == pytest.approx(entry_ev, rel=REL)
    assert result.entry_multiple == pytest.approx(entry_ev / 200.0, rel=REL)
    assert result.exit_multiple == pytest.approx(result.entry_multiple, rel=REL)  # exit = entry
    assert (result.senior_debt, result.subordinated_debt, result.leverage) == (0.0, 0.0, 0.0)
    assert result.equity_share == 1.0

    # each year: EBITDA 200, D&A 40, EBIT 160, tax 32, capex 50, no NWC change
    yearly_cash = 200.0 - 0.20 * (200.0 - 40.0) - 50.0
    assert yearly_cash == pytest.approx(118.0)
    exit_equity = entry_ev + 5 * yearly_cash  # same EBITDA and multiple, plus the cash pile
    assert result.exit_equity_value == pytest.approx(exit_equity, rel=REL)
    assert result.debt_schedule["cash_closing"].tolist() == pytest.approx(
        [yearly_cash * t for t in range(1, 6)], rel=REL
    )
    assert result.moic == pytest.approx(exit_equity / sponsor_equity, rel=REL)
    assert result.irr == pytest.approx((exit_equity / sponsor_equity) ** 0.2 - 1, abs=1e-10)
    assert result.equity_cash_flows == pytest.approx(
        (-sponsor_equity, 0.0, 0.0, 0.0, 0.0, exit_equity), rel=REL
    )

    # all value comes from cash generation
    amounts = result.attribution["amount"]
    assert amounts["ebitda_growth"] == pytest.approx(0.0, abs=1e-9)
    assert amounts["multiple_expansion"] == pytest.approx(0.0, abs=1e-9)
    assert amounts["debt_paydown"] == pytest.approx(5 * yearly_cash, rel=REL)
    assert amounts["fees"] == 0.0
    assert amounts["total_value_creation"] == pytest.approx(5 * yearly_cash, rel=REL)
    assert_lbo_invariants(result, target)


def independent_levered_case() -> dict[str, float]:
    """A three-year deal re-implemented from the description of the model."""
    revenue, shares = 1000.0, 100.0
    price = 10.0 * 1.20
    equity_purchase = price * shares  # 1,200
    entry_ev = equity_purchase + 300.0 - 50.0  # 1,450
    senior, sub = 3.0 * 200.0, 1.0 * 200.0  # 600 and 200
    transaction_fees = 0.02 * entry_ev  # 29
    financing_fees = 0.02 * (senior + sub)  # 16
    cash = 0.02 * revenue  # 20 left on the balance sheet
    uses = equity_purchase + 300.0 + transaction_fees + financing_fees + cash
    sponsor_equity = uses - senior - sub - 50.0
    scheduled = 0.05 * senior
    ebitda = 0.0
    for _ in range(3):
        previous = revenue
        revenue *= 1.05
        ebitda = 0.20 * revenue
        ebit = ebitda - 0.04 * revenue
        interest = 0.07 * senior + 0.11 * sub
        tax = 0.25 * max(ebit - interest, 0.0)
        cash_flow = ebitda - tax - 0.05 * revenue - 0.10 * (revenue - previous)
        amortization = min(scheduled, senior)
        cash += cash_flow - interest - amortization
        surplus = cash - 0.02 * revenue
        assert surplus > 0  # no revolver needed in this case
        sweep = min(surplus, senior - amortization)
        senior -= amortization + sweep
        cash -= sweep
    exit_equity = 7.0 * ebitda - senior - sub + cash
    return {
        "sponsor_equity": sponsor_equity,
        "exit_equity": exit_equity,
        "senior_closing": senior,
        "cash_closing": cash,
        "exit_ebitda": ebitda,
        "irr": (exit_equity / sponsor_equity) ** (1 / 3) - 1,
    }


def test_levered_case_matches_an_independent_reimplementation() -> None:
    expected = independent_levered_case()
    assert expected["sponsor_equity"] == pytest.approx(715.0)  # the hand number
    target = company()
    result = lbo.run_lbo(target, assumptions())
    assert result.sponsor_equity == pytest.approx(expected["sponsor_equity"], rel=REL)
    assert result.exit_ebitda == pytest.approx(expected["exit_ebitda"], rel=REL)
    assert result.debt_schedule["senior_closing"].iloc[-1] == pytest.approx(
        expected["senior_closing"], rel=REL
    )
    assert result.debt_schedule["cash_closing"].iloc[-1] == pytest.approx(
        expected["cash_closing"], rel=REL
    )
    assert result.exit_equity_value == pytest.approx(expected["exit_equity"], rel=REL)
    assert result.irr == pytest.approx(expected["irr"], abs=1e-10)
    assert result.moic == pytest.approx(expected["exit_equity"] / expected["sponsor_equity"])
    assert result.sizing_notes == []
    assert_lbo_invariants(result, target)


def test_entry_price_and_enterprise_value_include_all_claims() -> None:
    target = company(minority=40.0, preferred=25.0)
    result = lbo.run_lbo(target, assumptions())
    assert result.market_price == 10.0
    assert result.entry_price_per_share == pytest.approx(12.0)
    assert result.entry_equity_value == pytest.approx(1200.0)
    assert result.entry_enterprise_value == pytest.approx(1200.0 + 300.0 - 50.0 + 40.0 + 25.0)
    uses = result.sources_and_uses.set_index("item")["amount"]
    assert uses["Redemption of preferred equity"] == 25.0
    # minorities are not bought out: no use of funds, deducted again at exit
    assert "minority" not in " ".join(uses.index).lower()
    assert result.exit_minority_interest == 40.0
    assert result.exit_equity_value == pytest.approx(
        result.exit_enterprise_value - result.exit_net_debt - 40.0
    )
    assert_lbo_invariants(result, target)


def test_explicit_entry_price_overrides_the_premium() -> None:
    target = company()
    a = lbo.with_entry_price(assumptions(), 15.0)
    assert a.entry_price_per_share == 15.0
    result = lbo.run_lbo(target, a)
    assert result.entry_price_per_share == 15.0
    assert result.entry_equity_value == pytest.approx(1500.0)


# ------------------------------------------------------------------ invariants


SCENARIOS = {
    "base": {},
    "default_caps": {
        "min_equity_share": 0.30,
        "min_interest_coverage": 2.0,
        "min_debt_service_coverage": 1.25,
        "interest_deduction_cap_pct_ebitda": 0.30,
    },
    "pik": {"subordinated_pik": True},
    "half_sweep": {"cash_sweep_pct": 0.5},
    "no_sweep": {"cash_sweep_pct": 0.0},
    "no_amortization": {"senior_amortization_pct": 0.0},
    "fast_amortization": {"senior_amortization_pct": 0.45},
    "margin_dip_draws_revolver": {
        "revenue_growth": (0.05, 0.05, 0.05, 0.05),
        "ebitda_margin": (0.20, 0.10, 0.20, 0.20),
        "senior_debt_multiple": 4.0,
    },
    "collapse_breaches_liquidity": {
        "ebitda_margin": (0.20, 0.03, 0.02),
        "senior_debt_multiple": 6.0,
        "subordinated_pik": True,
    },
    "no_revolver": {"ebitda_margin": (0.20, 0.05, 0.20), "revolver_limit_x_ebitda": 0.0},
    "shrinking": {"revenue_growth": (-0.10, -0.10, -0.10)},
    "negative_nwc": {"nwc_pct_revenue": -0.15},
    "exit_equals_entry": {"exit_multiple": None},
    "one_year": {"revenue_growth": (0.05,), "ebitda_margin": (0.2,)},
    "senior_only": {"subordinated_debt_multiple": 0.0},
    "wiped_out": {"exit_multiple": 1.0, "senior_debt_multiple": 5.0},
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_schedule_invariants_hold_in_every_scenario(name: str) -> None:
    target = company()
    result = lbo.run_lbo(target, assumptions(**SCENARIOS[name]))
    assert_lbo_invariants(result, target)


def test_scenarios_exercise_the_revolver_the_breach_and_the_floor() -> None:
    target = company()
    dip = lbo.run_lbo(target, assumptions(**SCENARIOS["margin_dip_draws_revolver"]))
    assert dip.debt_schedule["revolver_draw"].sum() > 0
    assert dip.debt_schedule["revolver_repayment"].sum() > 0
    assert dip.liquidity_breach_years == []

    collapse = lbo.run_lbo(target, assumptions(**SCENARIOS["collapse_breaches_liquidity"]))
    assert collapse.liquidity_breach_years
    assert any("liquidity breach" in w for w in collapse.warnings)
    assert (collapse.debt_schedule["cash_closing"] < 0).any()
    assert (collapse.debt_schedule["shortfall_interest"] > 0).any()
    pik = collapse.debt_schedule["subordinated_closing"]
    assert pik.iloc[-1] == pytest.approx(collapse.subordinated_debt * 1.11**3, rel=REL)

    wiped = lbo.run_lbo(target, assumptions(**SCENARIOS["wiped_out"]))
    assert wiped.equity_wiped_out is True
    assert wiped.exit_equity_value == 0.0
    assert wiped.moic == 0.0 and wiped.irr == -1.0
    assert wiped.attribution.loc["equity_floor", "amount"] > 0
    assert any("wiped out" in w for w in wiped.warnings)


def test_interest_deduction_cap_raises_tax_by_the_disallowed_interest() -> None:
    target = company()
    heavy = {
        "senior_debt_multiple": 6.0,
        "subordinated_debt_multiple": 2.0,
        "entry_price_per_share": 20.0,
    }
    uncapped = lbo.run_lbo(target, assumptions(**heavy))
    capped = lbo.run_lbo(target, assumptions(**heavy, interest_deduction_cap_pct_ebitda=0.30))
    year = uncapped.forecast.index[0]
    interest = 0.07 * 1200.0 + 0.11 * 400.0  # 128 on EBITDA of 210
    ebitda, ebit = 0.20 * 1050.0, (0.20 - 0.04) * 1050.0
    assert uncapped.forecast.loc[year, "interest_expense"] == pytest.approx(interest)
    assert uncapped.forecast.loc[year, "tax"] == pytest.approx(0.25 * (ebit - interest))
    assert capped.forecast.loc[year, "deductible_interest"] == pytest.approx(0.30 * ebitda)
    assert capped.forecast.loc[year, "tax"] == pytest.approx(0.25 * (ebit - 0.30 * ebitda))
    extra_tax = capped.forecast.loc[year, "tax"] - uncapped.forecast.loc[year, "tax"]
    assert extra_tax == pytest.approx(0.25 * (interest - 0.30 * ebitda))


def test_no_tax_credit_when_interest_exceeds_ebit() -> None:
    target = company()
    result = lbo.run_lbo(
        target, assumptions(ebitda_margin=(0.06, 0.06, 0.06), senior_debt_multiple=5.0)
    )
    assert (result.forecast["pretax_income"] < 0).all()
    assert (result.forecast["tax"] == 0.0).all()
    assert_lbo_invariants(result, target)


# ------------------------------------------------------------------ debt sizing


def test_interest_coverage_cap_cuts_subordinated_debt_first() -> None:
    target = company()
    result = lbo.run_lbo(target, assumptions(min_interest_coverage=4.0))
    # budget 200 / 4 = 50 of first-year interest; senior alone costs 0.07 * 600 = 42
    assert result.senior_debt == pytest.approx(600.0)
    assert result.subordinated_debt == pytest.approx((50.0 - 42.0) / 0.11, rel=REL)
    first_year_interest = 0.07 * result.senior_debt + 0.11 * result.subordinated_debt
    assert 200.0 / first_year_interest == pytest.approx(4.0, rel=REL)
    assert any("EBITDA / interest" in note for note in result.sizing_notes)
    tighter = lbo.run_lbo(target, assumptions(min_interest_coverage=8.0))
    assert tighter.subordinated_debt == 0.0
    assert tighter.senior_debt == pytest.approx(200.0 / 8.0 / 0.07, rel=REL)


def test_debt_service_coverage_cap() -> None:
    target = company()
    result = lbo.run_lbo(target, assumptions(min_debt_service_coverage=2.0))
    ebitda_less_capex = 200.0 - 0.05 * 1000.0
    service = (0.07 + 0.05) * result.senior_debt + 0.11 * result.subordinated_debt
    assert ebitda_less_capex / service == pytest.approx(2.0, rel=REL)
    assert result.senior_debt == pytest.approx(600.0)
    assert result.subordinated_debt < 200.0
    assert any("mandatory" in note for note in result.sizing_notes)
    # PIK interest is not cash: the subordinated tranche escapes this cap
    pik = lbo.run_lbo(target, assumptions(min_debt_service_coverage=2.0, subordinated_pik=True))
    assert pik.subordinated_debt == pytest.approx(200.0)


def test_minimum_equity_share_cap_binds_exactly() -> None:
    target = company()
    result = lbo.run_lbo(target, assumptions(min_equity_share=0.60))
    assert result.equity_share == pytest.approx(0.60, rel=1e-9)
    assert result.senior_debt + result.subordinated_debt < 800.0
    assert any("sponsor equity is at least 60%" in note for note in result.sizing_notes)
    assert_lbo_invariants(result, target)
    loose = lbo.run_lbo(target, assumptions(min_equity_share=0.30))
    assert loose.equity_share > 0.30
    assert loose.senior_debt + loose.subordinated_debt == pytest.approx(800.0)


# ------------------------------------------------------------- ability to pay


def test_ability_to_pay_prices_reproduce_their_target_irrs() -> None:
    target = company()
    a = assumptions()
    result = lbo.run_lbo(target, a)
    assert set(result.ability_to_pay) == {0.25, 0.20, 0.15}
    for target_irr, price in result.ability_to_pay.items():
        assert result.ability_to_pay_status[target_irr] == lbo.STATUS_SOLVED
        fed_back = lbo.run_lbo(target, lbo.with_entry_price(a, price))
        assert fed_back.irr == pytest.approx(target_irr, abs=1e-9)
    low, base, high = (result.ability_to_pay[t] for t in (0.25, 0.20, 0.15))
    assert 0 < low < base < high
    (bar,) = result.to_valuation_results()
    assert bar.method is ValuationMethod.LBO
    assert (bar.low, bar.base, bar.high) == (low, base, high)
    assert bar.equity_value == pytest.approx(base * 100.0)
    assert bar.enterprise_value == pytest.approx(base * 100.0 + 250.0)
    assert bar.details["target_irr"] == {"low": 0.25, "base": 0.20, "high": 0.15}
    assert bar.details["default_entry_irr"] == result.irr


def test_ability_to_pay_does_not_depend_on_the_evaluated_entry_price() -> None:
    target = company()
    cheap = lbo.run_lbo(target, lbo.with_entry_price(assumptions(), 8.0))
    dear = lbo.run_lbo(target, lbo.with_entry_price(assumptions(), 16.0))
    for target_irr in (0.25, 0.20, 0.15):
        assert cheap.ability_to_pay[target_irr] == pytest.approx(
            dear.ability_to_pay[target_irr], rel=1e-9
        )


def test_custom_target_irrs_are_solved_too() -> None:
    target = company()
    a = assumptions(target_irr_conservative=0.30, target_irr_base=0.18, target_irr_aggressive=0.10)
    result = lbo.run_lbo(target, a)
    for target_irr, price in result.ability_to_pay.items():
        assert lbo.run_lbo(target, lbo.with_entry_price(a, price)).irr == pytest.approx(
            target_irr, abs=1e-9
        )


def test_higher_entry_price_means_lower_irr() -> None:
    target = company()
    irrs = [
        lbo.run_lbo(target, lbo.with_entry_price(assumptions(), price)).irr
        for price in (6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0)
    ]
    assert all(later < earlier for earlier, later in zip(irrs, irrs[1:], strict=False))


def test_unreachable_target_gives_no_football_field_bar() -> None:
    # the debt to be refinanced alone is worth more than the business
    target = company(debt=5000.0, cash=0.0)
    result = lbo.run_lbo(target, assumptions())
    assert result.ability_to_pay_status[0.20] == lbo.STATUS_UNREACHABLE
    assert result.ability_to_pay[0.20] == 0.0
    assert result.to_valuation_results() == []
    assert any("no positive price delivers a 20% IRR" in w for w in result.warnings)


# ---------------------------------------------------------------- sensitivities


def test_sensitivity_tables() -> None:
    target = company()
    a = assumptions()
    result = lbo.run_lbo(target, a)
    irr, moic = result.sensitivity_irr_entry_exit, result.sensitivity_moic_entry_exit
    assert irr.shape == moic.shape == (5, 5)
    steps = (0.8, 0.9, 1.0, 1.1, 1.2)
    assert irr.index.tolist() == [round(result.entry_multiple * s, 2) for s in steps]
    assert irr.columns.tolist() == [round(7.0 * s, 2) for s in steps]
    assert irr.iloc[2, 2] == pytest.approx(result.irr, abs=1e-12)
    assert moic.iloc[2, 2] == pytest.approx(result.moic, rel=1e-12)
    values = irr.to_numpy()
    assert (np.diff(values, axis=1) > 0).all()  # higher exit multiple, higher IRR
    assert (np.diff(values, axis=0) < 0).all()  # higher entry multiple, lower IRR
    # a cell is the deal at that entry multiple and exit multiple
    entry_multiple = result.entry_multiple * 0.8
    price = (entry_multiple * 200.0 - 250.0) / 100.0
    cell = lbo.run_lbo(target, replace(a, entry_price_per_share=price, exit_multiple=7.0 * 1.2))
    assert irr.iloc[0, 4] == pytest.approx(cell.irr, abs=1e-12)

    leverage = result.sensitivity_irr_leverage_exit
    assert leverage.index.tolist() == [2.0, 3.0, 4.0, 5.0, 6.0]
    assert leverage.loc[4.0].iloc[2] == pytest.approx(result.irr, abs=1e-12)
    lower = lbo.run_lbo(
        target, replace(a, senior_debt_multiple=2.0 * 0.75, subordinated_debt_multiple=2.0 * 0.25)
    )
    assert leverage.loc[2.0].iloc[2] == pytest.approx(lower.irr, abs=1e-12)


# --------------------------------------------------------------- applicability


def test_not_applicable_paths() -> None:
    a = assumptions()
    with pytest.raises(NotApplicableError, match="financial or real estate"):
        lbo.run_lbo(company(sector="Financial Services"), a)
    with pytest.raises(NotApplicableError, match="financial or real estate"):
        lbo.run_lbo(company(sector="Real Estate"), a)
    with pytest.raises(NotApplicableError, match="at least 2 fiscal years"):
        lbo.run_lbo(make_company(years=(2025,), revenue=(1000.0,)), a)
    with pytest.raises(
        NotApplicableError, match="EBITDA of the latest fiscal year is not positive"
    ):
        lbo.run_lbo(company(margin=-0.05), a)
    with pytest.raises(NotApplicableError, match="market price or shares"):
        lbo.run_lbo(company(price=math.nan), a)
    with pytest.raises(NotApplicableError, match="enterprise value is not positive"):
        lbo.run_lbo(company(cash=1e6), a)
    with pytest.raises(NotApplicableError, match="no debt capacity"):
        lbo.run_lbo(company(), assumptions(capex_pct_revenue=0.30, min_debt_service_coverage=1.25))
    with pytest.raises(NotApplicableError, match="no sponsor equity required"):
        lbo.run_lbo(company(), assumptions(senior_debt_multiple=30.0))
    missing = company()
    missing.statements.income.loc[2025, "revenue"] = np.nan
    with pytest.raises(NotApplicableError, match="revenue of the latest fiscal year is missing"):
        lbo.run_lbo(missing, a)
    with pytest.raises(NotApplicableError):
        lbo.default_assumptions(company(sector="Financial Services"))


@pytest.mark.parametrize(
    "changes",
    [
        {"revenue_growth": (0.05, 0.05), "ebitda_margin": (0.2, 0.2, 0.2)},
        {"revenue_growth": (), "ebitda_margin": ()},
        {"revenue_growth": (0.05, -1.0, 0.05)},
        {"transaction_fee_pct": -0.01},
        {"financing_fee_pct": 1.0},
        {"senior_debt_multiple": -1.0},
        {"cash_sweep_pct": 1.5},
        {"min_equity_share": 1.0},
        {"tax_rate": math.nan},
        {"exit_multiple": 0.0},
        {"entry_price_per_share": 0.0},
        {"risk_free_rate": -0.10},
        {"target_irr_base": 0.30},  # above the conservative target
        {"target_irr_aggressive": 0.22},  # above the base target
    ],
)
def test_assumption_validation(changes: dict) -> None:
    with pytest.raises(ValueError):
        assumptions(**changes)


def test_assumption_properties() -> None:
    a = assumptions()
    assert a.holding_period == 3
    assert a.senior_rate == pytest.approx(0.07)
    assert a.subordinated_rate == pytest.approx(0.11)


# ---------------------------------------------------------- default assumptions


def test_default_assumptions_follow_the_documented_recipe() -> None:
    target = make_company(
        years=(2022, 2023, 2024, 2025),
        revenue=(900.0, 950.0, 1000.0, 1050.0),
        margin=(0.10, 0.30, 0.30, 0.20),
    )
    params = MarketParameters(risk_free_rate=0.04, tax_rate=0.22, long_term_growth=0.03)
    a = lbo.default_assumptions(target, params)

    start = (1050.0 / 900.0) ** (1 / 3) - 1  # 5.3 %, inside the -5 % .. +10 % band
    assert a.revenue_growth == pytest.approx(
        [start + (0.03 - start) * t / 5 for t in range(1, 6)], rel=REL
    )
    assert a.revenue_growth[-1] == pytest.approx(0.03)
    # margin: from the latest 20 % towards the median 25 % (inside the 75-125 % band)
    assert a.ebitda_margin == pytest.approx([0.20 + 0.05 * t / 5 for t in range(1, 6)], rel=REL)
    assert a.da_pct_revenue == pytest.approx(0.04)
    assert a.capex_pct_revenue == pytest.approx(0.05)
    assert a.nwc_pct_revenue == pytest.approx(0.10)
    assert (a.tax_rate, a.risk_free_rate) == (0.22, 0.04)
    assert a.holding_period == 5
    assert a.exit_multiple is None and a.entry_price_per_share is None
    assert (a.takeover_premium, a.senior_debt_multiple, a.subordinated_debt_multiple) == (
        0.30,
        4.0,
        1.5,
    )


def test_default_growth_and_margin_are_bounded() -> None:
    fast = make_company(revenue=(100.0, 200.0, 400.0, 800.0), margin=(0.9, 0.9, 0.9, 0.2))
    a = lbo.default_assumptions(fast)
    assert a.revenue_growth[0] == pytest.approx(0.10 + (0.02 - 0.10) / 5)
    assert a.ebitda_margin[-1] == pytest.approx(0.20 * 1.25)  # median 90 % capped at 125 %
    slow = make_company(revenue=(800.0, 400.0, 200.0, 100.0), margin=(0.01, 0.01, 0.01, 0.2))
    a = lbo.default_assumptions(slow)
    assert a.revenue_growth[0] == pytest.approx(-0.05 + (0.02 + 0.05) / 5)
    assert a.ebitda_margin[-1] == pytest.approx(0.20 * 0.75)


def test_default_fallbacks_for_missing_items() -> None:
    target = make_company(capex=None, nwc=None)
    a = lbo.default_assumptions(target)
    assert a.capex_pct_revenue == pytest.approx(a.da_pct_revenue)
    assert a.nwc_pct_revenue == 0.0
    result = lbo.run_lbo(target)
    assert any("capex missing" in w for w in result.warnings)
    assert any("net working capital missing" in w for w in result.warnings)
    assert_lbo_invariants(result, target)


def test_run_lbo_without_assumptions_uses_the_defaults() -> None:
    target = make_company()
    result = lbo.run_lbo(target)
    assert result.assumptions == lbo.default_assumptions(target)
    assert result.exit_multiple == pytest.approx(result.entry_multiple)
    assert result.entry_price_per_share == pytest.approx(13.0)
    assert_lbo_invariants(result, target)


# ------------------------------------------------------------ remaining paths


def test_target_irr_reached_at_any_price_is_reported_as_unbounded() -> None:
    # EBITDA grows 40 % a year and the exit multiple follows the entry multiple:
    # paying more always buys more exit value than the 25 % hurdle costs
    target = company()
    a = assumptions(revenue_growth=(0.40,) * 5, ebitda_margin=(0.20,) * 5, exit_multiple=None)
    result = lbo.run_lbo(target, a)
    assert set(result.ability_to_pay_status.values()) == {lbo.STATUS_UNBOUNDED}
    assert all(math.isnan(price) for price in result.ability_to_pay.values())
    assert result.to_valuation_results() == []
    assert sum("reached at every price tested" in w for w in result.warnings) == 3
    assert result.irr > 0.25
    assert_lbo_invariants(result, target)


def test_default_depreciation_fallbacks() -> None:
    target = make_company(da=0.04)
    target.statements.income["depreciation_amortization"] = np.nan
    assert lbo.default_assumptions(target).da_pct_revenue == pytest.approx(0.04)  # cash flow
    target.statements.cash_flow["depreciation_amortization"] = np.nan
    assert lbo.default_assumptions(target).da_pct_revenue == pytest.approx(0.03)  # flat fallback
    assert any("D&A missing" in w for w in lbo.run_lbo(target).warnings)


def test_default_growth_without_history_is_the_long_term_rate() -> None:
    target = make_company()
    target.statements.income.loc[[2022, 2023, 2024], "revenue"] = np.nan
    a = lbo.default_assumptions(target)
    assert a.revenue_growth == pytest.approx([0.02] * 5)
    assert any("too short for a CAGR" in w for w in lbo.run_lbo(target).warnings)


def test_a_premium_of_minus_100_percent_is_not_applicable() -> None:
    with pytest.raises(NotApplicableError, match="entry price per share is not positive"):
        lbo.run_lbo(company(), assumptions(takeover_premium=-1.0))


def test_ability_to_pay_is_an_enterprise_value_net_of_existing_claims() -> None:
    """One more unit of existing net debt lowers the affordable equity price one for one."""
    a = assumptions()
    base = lbo.run_lbo(company(debt=1000.0, cash=0.0), a)
    more_debt = lbo.run_lbo(company(debt=1100.0, cash=0.0), a)
    more_cash = lbo.run_lbo(company(debt=1000.0, cash=100.0), a)
    for target_irr in (0.25, 0.20, 0.15):
        price = base.ability_to_pay[target_irr]
        assert more_debt.ability_to_pay[target_irr] == pytest.approx(price - 1.0, abs=1e-8)
        assert more_cash.ability_to_pay[target_irr] == pytest.approx(price + 1.0, abs=1e-8)
    (bar,), (bar_debt,) = base.to_valuation_results(), more_debt.to_valuation_results()
    assert bar.enterprise_value == pytest.approx(bar_debt.enterprise_value, rel=1e-9)


def test_unreachable_conservative_target_gives_a_bar_that_starts_at_zero() -> None:
    result = lbo.run_lbo(company(debt=1300.0, cash=0.0), assumptions())
    assert result.ability_to_pay_status == {
        0.25: lbo.STATUS_UNREACHABLE,
        0.20: lbo.STATUS_SOLVED,
        0.15: lbo.STATUS_SOLVED,
    }
    (bar,) = result.to_valuation_results()
    assert bar.low == 0.0 < bar.base < bar.high
    assert sum("no positive price delivers a 25% IRR" in w for w in result.warnings) == 1
    no_base = lbo.run_lbo(company(debt=1400.0, cash=0.0), assumptions())
    assert no_base.ability_to_pay_status[0.20] == lbo.STATUS_UNREACHABLE
    assert no_base.ability_to_pay_status[0.15] == lbo.STATUS_SOLVED
    assert no_base.to_valuation_results() == []  # no base price, no bar


# ------------------------------------------------------------ fix-round changes


def test_attribution_shares_keep_the_sign_of_their_amounts() -> None:
    target = company()
    gain = lbo.run_lbo(target, assumptions()).attribution
    loss = lbo.run_lbo(target, assumptions(exit_multiple=4.0)).attribution
    assert gain.loc["total_value_creation", "amount"] > 0
    assert loss.loc["total_value_creation", "amount"] < 0
    for table, total_share in ((gain, 1.0), (loss, -1.0)):
        total = table.loc["total_value_creation", "amount"]
        assert table["share"].tolist() == pytest.approx((table["amount"] / abs(total)).tolist())
        assert table.loc["total_value_creation", "share"] == pytest.approx(total_share)
        parts = table.drop("total_value_creation")
        assert parts["share"].sum() == pytest.approx(total_share)
        assert (np.sign(parts["share"]) == np.sign(parts["amount"])).all()
    # in a losing deal the value destroyers are negative, the contributors positive
    assert loss.loc["multiple_expansion", "share"] < 0 and loss.loc["fees", "share"] < 0
    assert loss.loc["ebitda_growth", "share"] > 0 and loss.loc["debt_paydown", "share"] > 0


def test_run_lbo_builds_its_defaults_with_the_given_market_parameters() -> None:
    target = make_company()
    params = MarketParameters(risk_free_rate=0.05, tax_rate=0.25, long_term_growth=0.03)
    result = lbo.run_lbo(target, params=params)
    assert result.assumptions == lbo.default_assumptions(target, params)
    assert (result.assumptions.risk_free_rate, result.assumptions.tax_rate) == (0.05, 0.25)
    assert result.assumptions.revenue_growth[-1] == pytest.approx(0.03)
    first = result.debt_schedule.iloc[0]
    assert first["senior_interest"] == pytest.approx((0.05 + 0.04) * result.senior_debt)
    assert lbo.run_lbo(target).assumptions == lbo.default_assumptions(target, MarketParameters())
    assert lbo.run_lbo(target, None, None).assumptions.risk_free_rate == 0.03075
    # explicit assumptions win; params are then ignored
    explicit = assumptions()
    assert lbo.run_lbo(company(), explicit, params).assumptions is explicit
    assert_lbo_invariants(result, target)


def test_one_default_risk_free_rate() -> None:
    plain = LBOAssumptions(
        revenue_growth=(0.02,),
        ebitda_margin=(0.2,),
        da_pct_revenue=0.04,
        capex_pct_revenue=0.04,
        nwc_pct_revenue=0.1,
    )
    assert plain.risk_free_rate == MarketParameters().risk_free_rate == 0.03075
    assert plain.tax_rate == MarketParameters().tax_rate


@pytest.mark.parametrize(
    ("flows", "roots"),
    [
        ([-100.0, 230.0, -132.0], (0.10, 0.20)),
        ([-1000.0, 3600.0, -4310.0, 1716.0], (0.10, 0.20, 0.30)),
    ],
)
def test_irr_with_several_sign_changes_is_the_smallest_root(
    flows: list[float], roots: tuple[float, ...]
) -> None:
    for root in roots:  # they really are roots
        assert sum(c / (1 + root) ** t for t, c in enumerate(flows)) == pytest.approx(0, abs=1e-6)
    assert compute_irr(flows) == pytest.approx(min(roots), abs=1e-7)


def test_irr_with_several_sign_changes_and_no_root_is_nan() -> None:
    assert math.isnan(compute_irr([-100.0, 100.0, -100.0]))  # NPV < 0 for every rate
