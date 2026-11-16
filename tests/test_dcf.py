"""DCF arithmetic checked against closed forms and a hand-built forecast.

Every expected number is derived here with plain Python arithmetic; the model is
never used to produce its own expected value.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from support import make_company, make_peer
from valuation_engine.core.types import (
    DCFAssumptions,
    MarketParameters,
    NotApplicableError,
    ValuationMethod,
)
from valuation_engine.models import dcf

REL = 1e-10


def core(**changes: object) -> dcf.DCFCoreOutput:
    """``dcf_core`` on a small base case with keyword overrides."""
    inputs: dict[str, object] = {
        "base_revenue": 1000.0,
        "revenue_growth": (0.03,) * 5,
        "ebitda_margin": (0.25,) * 5,
        "da_pct_revenue": 0.05,
        "capex_pct_revenue": 0.06,
        "nwc_pct_revenue": 0.12,
        "tax_rate": 0.22,
        "wacc": 0.09,
        "terminal_growth": 0.03,
        "exit_multiple": 9.0,
        "net_debt": 400.0,
        "shares_outstanding": 50.0,
        "minority_interest": 0.0,
        "preferred_equity": 0.0,
        "mid_year": False,
    }
    inputs.update(changes)
    return dcf.dcf_core(**inputs)  # type: ignore[arg-type]


def first_year_fcff(
    revenue0: float, g: float, margin: float, da: float, capex: float, nwc: float, tax: float
) -> float:
    revenue1 = revenue0 * (1 + g)
    ebit = (margin - da) * revenue1
    nopat = ebit * (1 - tax)
    return nopat + da * revenue1 - capex * revenue1 - nwc * (revenue1 - revenue0)


# ------------------------------------------------------------ analytic truths


@pytest.mark.parametrize("years", [1, 2, 5, 10, 25])
def test_constant_growth_without_mid_year_equals_growing_perpetuity(years: int) -> None:
    """Growth g from year 1 on, capex >= D&A: EV = FCFF_1 / (WACC - g), for any N."""
    g, wacc = 0.03, 0.09
    out = core(revenue_growth=(g,) * years, ebitda_margin=(0.25,) * years, terminal_growth=g)
    fcff1 = first_year_fcff(1000.0, g, 0.25, 0.05, 0.06, 0.12, 0.22)
    assert out.fcff[0, 0] == pytest.approx(fcff1, rel=REL)
    assert out.ev_gordon[0] == pytest.approx(fcff1 / (wacc - g), rel=REL)
    # the explicit flows themselves grow at g
    assert out.fcff[0] == pytest.approx(fcff1 * (1 + g) ** np.arange(years), rel=REL)
    assert out.terminal_fcff[0] == pytest.approx(fcff1 * (1 + g) ** years, rel=REL)


@pytest.mark.parametrize("wacc", [0.06, 0.09, 0.15])
def test_mid_year_convention_scales_the_gordon_value_by_sqrt_one_plus_wacc(wacc: float) -> None:
    end = core(wacc=wacc, mid_year=False)
    mid = core(wacc=wacc, mid_year=True)
    factor = (1 + wacc) ** 0.5
    assert mid.discount_periods.tolist() == [0.5, 1.5, 2.5, 3.5, 4.5]
    assert end.discount_periods.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert mid.pv_fcff[0] == pytest.approx(end.pv_fcff[0] * factor, rel=REL)
    assert mid.tv_gordon[0] == pytest.approx(end.tv_gordon[0] * factor, rel=REL)
    assert mid.ev_gordon[0] == pytest.approx(end.ev_gordon[0] * factor, rel=REL)
    # undiscounted flows are untouched
    assert mid.fcff[0] == pytest.approx(end.fcff[0], rel=REL)


def test_exit_multiple_terminal_value_gets_no_mid_year_adjustment() -> None:
    end = core(mid_year=False)
    mid = core(mid_year=True)
    assert mid.tv_exit[0] == end.tv_exit[0]
    assert mid.pv_tv_exit[0] == pytest.approx(end.pv_tv_exit[0], rel=REL)
    last_ebitda = 1000.0 * 1.03**5 * 0.25
    assert mid.tv_exit[0] == pytest.approx(9.0 * last_ebitda, rel=REL)
    assert mid.pv_tv_exit[0] == pytest.approx(9.0 * last_ebitda / 1.09**5, rel=REL)


def test_mid_year_gordon_tv_equals_discounting_the_plain_perpetuity_n_minus_half_years() -> None:
    wacc, g, years = 0.09, 0.03, 5
    out = core(mid_year=True)
    plain_tv = out.terminal_fcff[0] / (wacc - g)
    assert out.pv_tv_gordon[0] == pytest.approx(plain_tv / (1 + wacc) ** (years - 0.5), rel=REL)


def test_no_tax_credit_on_negative_ebit() -> None:
    out = core(ebitda_margin=(0.02,) * 5)  # D&A 5 % > margin 2 %
    assert np.all(out.ebit < 0)
    assert np.all(out.taxes == 0.0)
    assert out.nopat[0] == pytest.approx(out.ebit[0], rel=REL)


def test_terminal_capex_is_floored_at_depreciation() -> None:
    low_capex = core(capex_pct_revenue=0.01)  # below D&A of 5 %
    at_da = core(capex_pct_revenue=0.05)
    unfloored = core(capex_pct_revenue=0.01, terminal_capex_floor=False)
    assert low_capex.terminal_fcff[0] == pytest.approx(at_da.terminal_fcff[0], rel=REL)
    assert unfloored.terminal_fcff[0] > low_capex.terminal_fcff[0]
    # the explicit period still uses the assumption as given
    assert low_capex.capex[0] == pytest.approx(0.01 * low_capex.revenue[0], rel=REL)


def test_nwc_investment_reduces_cash_flow_when_revenue_grows_and_releases_when_it_falls() -> None:
    growing = core(revenue_growth=(0.10,) * 5)
    shrinking = core(revenue_growth=(-0.10,) * 5)
    assert np.all(growing.delta_nwc > 0)
    assert np.all(shrinking.delta_nwc < 0)
    no_nwc = core(revenue_growth=(0.10,) * 5, nwc_pct_revenue=0.0)
    assert no_nwc.fcff[0] - growing.fcff[0] == pytest.approx(growing.delta_nwc[0], rel=REL)
    assert growing.delta_nwc[0, 0] == pytest.approx(0.12 * 100.0, rel=REL)


def test_wacc_at_or_below_growth_gives_nan_gordon_but_a_finite_exit_value() -> None:
    out = core(wacc=np.array([0.03, 0.02, 0.031]), terminal_growth=0.03)
    assert np.isnan(out.ev_gordon[:2]).all()
    assert np.isfinite(out.ev_gordon[2])
    assert np.isfinite(out.ev_exit).all()


def test_core_is_vectorised_and_matches_scalar_calls() -> None:
    waccs = np.array([0.07, 0.09, 0.11, 0.13])
    growth = np.array([0.01, 0.02, 0.03, 0.035])
    margins = np.array([[0.2] * 5, [0.22] * 5, [0.25] * 5, [0.3] * 5])
    batch = core(wacc=waccs, terminal_growth=growth, ebitda_margin=margins)
    assert batch.ev_gordon.shape == (4,)
    assert batch.fcff.shape == (4, 5)
    for i in range(4):
        single = core(wacc=waccs[i], terminal_growth=growth[i], ebitda_margin=margins[i])
        assert batch.ev_gordon[i] == single.ev_gordon[0]
        assert batch.value_per_share_exit[i] == single.value_per_share_exit[0]


# ------------------------------------------------- hand-built two-year forecast

HAND_ASSUMPTIONS = DCFAssumptions(
    revenue_growth=(0.10, 0.05),
    ebitda_margin=(0.20, 0.25),
    da_pct_revenue=0.06,
    capex_pct_revenue=0.04,  # below D&A: the terminal floor is active
    nwc_pct_revenue=0.10,
    tax_rate=0.25,
    wacc=0.10,
    terminal_growth=0.02,
    exit_multiple=8.0,
    mid_year=True,
)


def hand_company():
    return make_company(
        years=(2024, 2025),
        revenue=(900.0, 1000.0),
        price=10.0,
        shares=100.0,
        debt=300.0,
        cash=50.0,
        minority=20.0,
        preferred=10.0,
    )


def test_hand_built_two_year_case_every_line() -> None:
    result = dcf.run_dcf(hand_company(), HAND_ASSUMPTIONS)
    wacc, g, tax = 0.10, 0.02, 0.25

    # --- explicit years, line by line -----------------------------------------
    rev = [1000.0 * 1.10, 1000.0 * 1.10 * 1.05]  # 1100, 1155
    ebitda = [0.20 * rev[0], 0.25 * rev[1]]  # 220, 288.75
    da = [0.06 * rev[0], 0.06 * rev[1]]  # 66, 69.3
    ebit = [ebitda[0] - da[0], ebitda[1] - da[1]]  # 154, 219.45
    taxes = [tax * ebit[0], tax * ebit[1]]  # 38.5, 54.8625
    nopat = [ebit[0] - taxes[0], ebit[1] - taxes[1]]  # 115.5, 164.5875
    capex = [0.04 * rev[0], 0.04 * rev[1]]  # 44, 46.2
    d_nwc = [0.10 * (rev[0] - 1000.0), 0.10 * (rev[1] - rev[0])]  # 10, 5.5
    fcff = [nopat[i] + da[i] - capex[i] - d_nwc[i] for i in range(2)]
    assert fcff == pytest.approx([127.5, 182.1875], rel=1e-12)  # the hand numbers
    factors = [1 / (1 + wacc) ** 0.5, 1 / (1 + wacc) ** 1.5]
    pv_fcff = fcff[0] * factors[0] + fcff[1] * factors[1]

    table = result.forecast
    assert list(table.index) == [2026, 2027]
    assert table.index.name == "fiscal_year"
    expected = {
        "revenue": rev,
        "revenue_growth": [0.10, 0.05],
        "ebitda": ebitda,
        "ebitda_margin": [0.20, 0.25],
        "depreciation_amortization": da,
        "ebit": ebit,
        "taxes": taxes,
        "nopat": nopat,
        "capex": capex,
        "delta_nwc": d_nwc,
        "fcff": fcff,
        "discount_period": [0.5, 1.5],
        "discount_factor": factors,
        "pv_fcff": [fcff[0] * factors[0], fcff[1] * factors[1]],
    }
    assert set(expected) == set(table.columns)
    for column, values in expected.items():
        assert table[column].tolist() == pytest.approx(values, rel=REL), column
    assert result.pv_fcff == pytest.approx(pv_fcff, rel=REL)

    # --- normalized terminal year (year 3) --------------------------------------
    rev3 = rev[1] * (1 + g)  # 1178.1
    ebit3 = 0.25 * rev3 - 0.06 * rev3
    fcff3 = (
        ebit3 * (1 - tax)
        + 0.06 * rev3  # D&A
        - 0.06 * rev3  # capex floored at D&A (assumption 4 % < 6 %)
        - 0.10 * (rev3 - rev[1])
    )
    assert fcff3 == pytest.approx(165.56925, rel=1e-12)  # the hand number
    assert result.terminal_fcff == pytest.approx(fcff3, rel=REL)

    # --- Gordon growth ----------------------------------------------------------
    tv_gordon = fcff3 / (wacc - g) * (1 + wacc) ** 0.5  # at the end of year 2
    pv_tv_gordon = tv_gordon / (1 + wacc) ** 2
    ev_gordon = pv_fcff + pv_tv_gordon
    assert result.terminal_value_gordon == pytest.approx(tv_gordon, rel=REL)
    assert result.pv_terminal_value_gordon == pytest.approx(pv_tv_gordon, rel=REL)
    assert result.enterprise_value_gordon == pytest.approx(ev_gordon, rel=REL)
    assert result.tv_share_gordon == pytest.approx(pv_tv_gordon / ev_gordon, rel=REL)
    assert result.implied_exit_multiple == pytest.approx(tv_gordon / ebitda[1], rel=REL)

    # --- exit multiple ------------------------------------------------------------
    tv_exit = 8.0 * ebitda[1]  # 2310
    pv_tv_exit = tv_exit / (1 + wacc) ** 2
    ev_exit = pv_fcff + pv_tv_exit
    assert result.terminal_value_exit == pytest.approx(2310.0, rel=REL)
    assert result.pv_terminal_value_exit == pytest.approx(pv_tv_exit, rel=REL)
    assert result.enterprise_value_exit == pytest.approx(ev_exit, rel=REL)
    assert result.tv_share_exit == pytest.approx(pv_tv_exit / ev_exit, rel=REL)

    # --- equity bridge and per share ---------------------------------------------
    net_debt, minority, preferred, shares = 300.0 - 50.0, 20.0, 10.0, 100.0
    assert result.net_debt == net_debt
    assert result.minority_interest == minority
    assert result.preferred_equity == preferred
    equity_gordon = ev_gordon - net_debt - minority - preferred
    equity_exit = ev_exit - net_debt - minority - preferred
    assert result.equity_value_gordon == pytest.approx(equity_gordon, rel=REL)
    assert result.equity_value_exit == pytest.approx(equity_exit, rel=REL)
    assert result.value_per_share_gordon == pytest.approx(equity_gordon / shares, rel=REL)
    assert result.value_per_share_exit == pytest.approx(equity_exit / shares, rel=REL)
    assert result.upside_gordon == pytest.approx(equity_gordon / shares / 10.0 - 1, rel=REL)
    assert result.upside_exit == pytest.approx(equity_exit / shares / 10.0 - 1, rel=REL)

    # --- bookkeeping ----------------------------------------------------------------
    assert result.wacc == 0.10
    assert result.wacc_result is None  # WACC was given
    assert result.exit_multiple == 8.0
    assert result.exit_multiple_source == "assumption"
    assert result.assumptions == HAND_ASSUMPTIONS
    assert (result.ticker, result.currency, result.price) == ("TEST.ST", "SEK", 10.0)

    bridge = result.bridge()
    assert list(bridge.columns) == ["gordon", "exit_multiple"]
    assert bridge.loc["enterprise_value", "gordon"] == pytest.approx(
        bridge.loc["pv_fcff", "gordon"] + bridge.loc["pv_terminal_value", "gordon"], rel=REL
    )
    for column in bridge.columns:
        walked = bridge.loc[
            [
                "enterprise_value",
                "less_net_debt",
                "less_minority_interest",
                "less_preferred_equity",
            ],
            column,
        ].sum()
        assert bridge.loc["equity_value", column] == pytest.approx(walked, rel=REL)
        assert bridge.loc["value_per_share", column] == pytest.approx(walked / shares, rel=REL)


def test_implied_perpetual_growth_solves_the_gordon_formula_for_the_exit_value() -> None:
    result = dcf.run_dcf(hand_company(), HAND_ASSUMPTIONS)
    g_star = result.implied_perpetual_growth
    # plug g* into the hand formula of the normalized terminal flow
    rev2, wacc, tax = 1155.0, 0.10, 0.25
    rev3 = rev2 * (1 + g_star)
    fcff3 = (0.25 - 0.06) * rev3 * (1 - tax) - 0.10 * (rev3 - rev2)
    tv = fcff3 / (wacc - g_star) * (1 + wacc) ** 0.5
    assert tv == pytest.approx(result.terminal_value_exit, rel=1e-9)


def test_gordon_and_exit_multiple_round_trip() -> None:
    company = hand_company()
    base = dcf.run_dcf(company, HAND_ASSUMPTIONS)
    # 1) exit at the multiple implied by Gordon -> the exit case reproduces Gordon
    at_implied_multiple = dcf.run_dcf(
        company, replace(HAND_ASSUMPTIONS, exit_multiple=base.implied_exit_multiple)
    )
    assert at_implied_multiple.enterprise_value_exit == pytest.approx(
        base.enterprise_value_gordon, rel=REL
    )
    assert at_implied_multiple.implied_perpetual_growth == pytest.approx(0.02, abs=1e-12)
    # 2) grow at the rate implied by the exit multiple -> Gordon reproduces the exit case
    at_implied_growth = dcf.run_dcf(
        company, replace(HAND_ASSUMPTIONS, terminal_growth=base.implied_perpetual_growth)
    )
    assert at_implied_growth.enterprise_value_gordon == pytest.approx(
        base.enterprise_value_exit, rel=1e-9
    )
    assert at_implied_growth.implied_exit_multiple == pytest.approx(8.0, rel=1e-9)


def test_core_inputs_reproduce_the_base_case_bit_for_bit() -> None:
    result = dcf.run_dcf(hand_company(), HAND_ASSUMPTIONS)
    again = dcf.dcf_core(**result.core_inputs)
    assert again.value_per_share_gordon[0] == result.value_per_share_gordon
    assert again.value_per_share_exit[0] == result.value_per_share_exit
    assert again.ev_gordon[0] == result.enterprise_value_gordon


# -------------------------------------------------------------- monotonicity


def values(**changes: object) -> tuple[float, float]:
    result = dcf.run_dcf(hand_company(), replace(HAND_ASSUMPTIONS, **changes))  # type: ignore[arg-type]
    return result.value_per_share_gordon, result.value_per_share_exit


def test_value_falls_when_wacc_rises() -> None:
    series = [values(wacc=w) for w in (0.06, 0.08, 0.10, 0.12, 0.15)]
    for (g_low, e_low), (g_high, e_high) in zip(series, series[1:], strict=False):
        assert g_high < g_low
        assert e_high < e_low


def test_gordon_value_rises_with_terminal_growth_and_exit_value_ignores_it() -> None:
    series = [values(terminal_growth=g) for g in (0.0, 0.01, 0.02, 0.03, 0.04)]
    gordon = [v[0] for v in series]
    assert gordon == sorted(gordon) and len(set(gordon)) == 5
    assert len({v[1] for v in series}) == 1


def test_value_rises_with_margin_and_with_explicit_growth() -> None:
    by_margin = [values(ebitda_margin=(m, m)) for m in (0.15, 0.20, 0.25, 0.30)]
    assert [v[0] for v in by_margin] == sorted(v[0] for v in by_margin)
    assert [v[1] for v in by_margin] == sorted(v[1] for v in by_margin)
    assert by_margin[0][0] < by_margin[-1][0]
    by_growth = [values(revenue_growth=(x, x)) for x in (0.0, 0.05, 0.10)]
    assert [v[0] for v in by_growth] == sorted(v[0] for v in by_growth)
    assert [v[1] for v in by_growth] == sorted(v[1] for v in by_growth)


def test_exit_value_rises_with_the_multiple_and_gordon_ignores_it() -> None:
    series = [values(exit_multiple=m) for m in (4.0, 8.0, 12.0)]
    assert series[0][1] < series[1][1] < series[2][1]
    assert len({v[0] for v in series}) == 1


def test_one_unit_of_net_debt_costs_one_unit_of_equity() -> None:
    lean = dcf.run_dcf(
        make_company(years=(2024, 2025), revenue=(900.0, 1000.0), debt=100.0, cash=0.0),
        HAND_ASSUMPTIONS,
    )
    geared = dcf.run_dcf(
        make_company(years=(2024, 2025), revenue=(900.0, 1000.0), debt=350.0, cash=0.0),
        HAND_ASSUMPTIONS,
    )
    assert lean.enterprise_value_gordon == pytest.approx(geared.enterprise_value_gordon, rel=REL)
    assert lean.equity_value_gordon - geared.equity_value_gordon == pytest.approx(250.0, rel=REL)
    assert lean.equity_value_exit - geared.equity_value_exit == pytest.approx(250.0, rel=REL)


# ---------------------------------------------------------------- sensitivities


def test_sensitivity_grids_axes_centre_and_cells() -> None:
    company = hand_company()
    result = dcf.run_dcf(company, HAND_ASSUMPTIONS)
    gordon, exit_grid = result.sensitivity_gordon, result.sensitivity_exit
    assert gordon.shape == exit_grid.shape == (5, 5)
    assert gordon.index.tolist() == pytest.approx([0.09, 0.095, 0.10, 0.105, 0.11])
    assert gordon.columns.tolist() == pytest.approx([0.01, 0.015, 0.02, 0.025, 0.03])
    assert exit_grid.columns.tolist() == pytest.approx([6.0, 7.0, 8.0, 9.0, 10.0])
    # centre cell is the base case, exactly
    assert gordon.iloc[2, 2] == result.value_per_share_gordon
    assert exit_grid.iloc[2, 2] == result.value_per_share_exit
    # every cell is an ordinary DCF at that WACC / g / multiple
    for row, col in [(0, 0), (0, 4), (4, 0), (4, 4), (1, 3)]:
        wacc, g = gordon.index[row], gordon.columns[col]
        cell = dcf.run_dcf(company, replace(HAND_ASSUMPTIONS, wacc=wacc, terminal_growth=g))
        assert gordon.iloc[row, col] == pytest.approx(cell.value_per_share_gordon, rel=REL)
        multiple = exit_grid.columns[col]
        cell = dcf.run_dcf(company, replace(HAND_ASSUMPTIONS, wacc=wacc, exit_multiple=multiple))
        assert exit_grid.iloc[row, col] == pytest.approx(cell.value_per_share_exit, rel=REL)
    # monotone along both axes
    assert (np.diff(gordon.to_numpy(), axis=0) < 0).all()
    assert (np.diff(gordon.to_numpy(), axis=1) > 0).all()
    assert (np.diff(exit_grid.to_numpy(), axis=0) < 0).all()
    assert (np.diff(exit_grid.to_numpy(), axis=1) > 0).all()


def test_valuation_results_use_the_documented_grid_cells() -> None:
    company = hand_company()
    result = dcf.run_dcf(company, HAND_ASSUMPTIONS)
    gordon, exit_row = result.to_valuation_results()
    assert gordon.method is ValuationMethod.DCF_GORDON
    assert exit_row.method is ValuationMethod.DCF_EXIT_MULTIPLE
    assert gordon.low <= gordon.base <= gordon.high
    assert exit_row.low <= exit_row.base <= exit_row.high
    assert gordon.base == result.value_per_share_gordon
    assert exit_row.base == result.value_per_share_exit
    # Gordon: low = WACC + 1 pp, g - 0.5 pp; high = WACC - 1 pp, g + 0.5 pp
    low = dcf.run_dcf(company, replace(HAND_ASSUMPTIONS, wacc=0.11, terminal_growth=0.015))
    high = dcf.run_dcf(company, replace(HAND_ASSUMPTIONS, wacc=0.09, terminal_growth=0.025))
    assert gordon.low == pytest.approx(low.value_per_share_gordon, rel=REL)
    assert gordon.high == pytest.approx(high.value_per_share_gordon, rel=REL)
    # Exit: low = WACC + 1 pp, multiple - 1x; high = WACC - 1 pp, multiple + 1x
    low = dcf.run_dcf(company, replace(HAND_ASSUMPTIONS, wacc=0.11, exit_multiple=7.0))
    high = dcf.run_dcf(company, replace(HAND_ASSUMPTIONS, wacc=0.09, exit_multiple=9.0))
    assert exit_row.low == pytest.approx(low.value_per_share_exit, rel=REL)
    assert exit_row.high == pytest.approx(high.value_per_share_exit, rel=REL)
    assert gordon.enterprise_value == result.enterprise_value_gordon
    assert exit_row.equity_value == result.equity_value_exit
    assert gordon.details["range_fallback_to_base"] is False
    assert gordon.details["wacc"] == 0.10
    assert exit_row.details["exit_multiple"] == 8.0


def test_range_falls_back_to_base_when_a_grid_cell_is_undefined() -> None:
    # WACC 2.6 %, g 2 %: the high cell (WACC - 1 pp, g + 0.5 pp) has WACC < g.
    tight = replace(HAND_ASSUMPTIONS, wacc=0.026, terminal_growth=0.02)
    result = dcf.run_dcf(hand_company(), tight)
    gordon = result.to_valuation_results()[0]
    assert math.isnan(result.sensitivity_gordon.iloc[0, 3])
    assert gordon.high == gordon.base
    assert gordon.low < gordon.base
    assert gordon.details["range_fallback_to_base"] is True


def test_gordon_range_never_uses_a_spread_the_base_case_would_reject() -> None:
    assumptions = replace(HAND_ASSUMPTIONS, wacc=0.0361, terminal_growth=0.02)
    result = dcf.run_dcf(hand_company(), assumptions)
    grid = result.sensitivity_gordon
    spread = np.subtract.outer(grid.index.to_numpy(), grid.columns.to_numpy())
    too_tight = spread < dcf.MIN_WACC_GROWTH_SPREAD
    assert too_tight.any()
    assert np.isnan(grid.to_numpy()[too_tight]).all()


# --------------------------------------------------------------- applicability


def category(error: pytest.ExceptionInfo[NotApplicableError]) -> str:
    return str(error.value).split(":", 1)[0]


def test_not_applicable_for_a_financial_company() -> None:
    bank = make_company(sector="Financial Services", industry="Banks - Regional")
    with pytest.raises(NotApplicableError) as error:
        dcf.run_dcf(bank)
    assert category(error) == "financial sector"
    with pytest.raises(NotApplicableError):
        dcf.default_assumptions(bank)
    with pytest.raises(NotApplicableError):
        dcf.run_dcf(make_company(sector="Real Estate"))


def test_not_applicable_for_negative_or_missing_ebitda() -> None:
    with pytest.raises(NotApplicableError) as error:
        dcf.run_dcf(make_company(margin=-0.05))
    assert category(error) == "negative EBITDA"
    with pytest.raises(NotApplicableError) as error:
        dcf.run_dcf(make_company(margin=0.0))
    assert category(error) == "negative EBITDA"
    company = make_company()
    company.statements.income.loc[2025, "ebitda"] = np.nan
    with pytest.raises(NotApplicableError) as error:
        dcf.run_dcf(company)
    assert category(error) == "missing EBITDA"


@pytest.mark.parametrize("wacc", [0.02, 0.01, 0.0249, 0.0249999])
def test_not_applicable_when_wacc_is_not_above_growth(wacc: float) -> None:
    with pytest.raises(NotApplicableError) as error:
        dcf.run_dcf(hand_company(), replace(HAND_ASSUMPTIONS, wacc=wacc, terminal_growth=0.02))
    assert category(error) == "WACC not above terminal growth"


def test_minimum_spread_boundary_is_accepted() -> None:
    result = dcf.run_dcf(hand_company(), replace(HAND_ASSUMPTIONS, wacc=0.0251))
    assert math.isfinite(result.value_per_share_gordon)


def test_not_applicable_with_too_little_history() -> None:
    with pytest.raises(NotApplicableError) as error:
        dcf.run_dcf(make_company(years=(2025,), revenue=(1000.0,)))
    assert category(error) == "insufficient history"


def test_not_applicable_for_missing_revenue_or_market_data() -> None:
    company = make_company()
    company.statements.income.loc[2025, "revenue"] = np.nan
    with pytest.raises(NotApplicableError) as error:
        dcf.run_dcf(company)
    assert category(error) == "missing revenue"
    for bad in ({"price": 0.0}, {"shares": 0.0}, {"price": math.nan}):
        with pytest.raises(NotApplicableError) as error:
            dcf.run_dcf(make_company(**bad))
        assert category(error) == "invalid market data"


@pytest.mark.parametrize(
    "changes",
    [
        {"revenue_growth": (-1.0, 0.0)},
        {"revenue_growth": (math.nan, 0.0)},
        {"ebitda_margin": (0.2, math.inf)},
        {"tax_rate": 1.0},
        {"tax_rate": -0.1},
        {"wacc": 0.0},
        {"wacc": -0.05},
        {"da_pct_revenue": math.nan},
        {"exit_multiple": 0.0},
        {"exit_multiple": -3.0},
    ],
)
def test_invalid_assumptions_are_reported_as_not_applicable(changes: dict) -> None:
    with pytest.raises(NotApplicableError) as error:
        dcf.run_dcf(hand_company(), replace(HAND_ASSUMPTIONS, **changes))
    assert category(error) == "invalid assumptions"


def test_not_applicable_when_the_terminal_cash_flow_is_not_positive() -> None:
    with pytest.raises(NotApplicableError) as error:
        dcf.run_dcf(hand_company(), replace(HAND_ASSUMPTIONS, ebitda_margin=(0.20, 0.05)))
    assert category(error) == "non-positive terminal cash flow"


def test_negative_equity_value_is_reported_not_hidden() -> None:
    company = make_company(years=(2024, 2025), revenue=(900.0, 1000.0), debt=50_000.0, cash=0.0)
    result = dcf.run_dcf(company, HAND_ASSUMPTIONS)
    assert result.value_per_share_gordon < 0
    for row in result.to_valuation_results():
        assert row.low <= row.base <= row.high < 0
        assert row.details["negative_equity_value"] is True


# ---------------------------------------------------------- default assumptions


def test_default_assumptions_follow_the_documented_recipe() -> None:
    revenue = (700.0, 800.0, 900.0, 1000.0, 1100.0)
    margins = (0.50, 0.10, 0.20, 0.30, 0.25)
    company = make_company(
        years=(2021, 2022, 2023, 2024, 2025), revenue=revenue, margin=margins, da=0.04, capex=0.07
    )
    params = MarketParameters(tax_rate=0.30, long_term_growth=0.025)
    assumptions, notes = dcf.default_assumptions_with_notes(company, params)

    # revenue CAGR over the last four fiscal years (three intervals), fading to 2.5 %
    start = (1100.0 / 800.0) ** (1 / 3) - 1
    expected_growth = [start + (0.025 - start) * t / 4 for t in range(5)]
    assert assumptions.revenue_growth == pytest.approx(expected_growth, rel=REL)
    assert assumptions.revenue_growth[-1] == pytest.approx(0.025, rel=REL)
    # margin: mean of the last three years, flat
    assert assumptions.ebitda_margin == pytest.approx([(0.20 + 0.30 + 0.25) / 3] * 5, rel=REL)
    assert assumptions.da_pct_revenue == pytest.approx(0.04, rel=REL)
    assert assumptions.capex_pct_revenue == pytest.approx(0.07, rel=REL)
    assert assumptions.nwc_pct_revenue == pytest.approx(0.10, rel=REL)
    assert assumptions.tax_rate == 0.30
    assert assumptions.terminal_growth == 0.025
    assert assumptions.wacc is None and assumptions.exit_multiple is None
    assert assumptions.mid_year is True
    assert assumptions.forecast_years == dcf.FORECAST_YEARS == 5
    assert notes == []
    assert dcf.default_assumptions(company, params) == assumptions


def test_default_growth_is_clipped_to_the_band() -> None:
    fast = make_company(revenue=(100.0, 200.0, 400.0, 800.0))
    slow = make_company(revenue=(800.0, 400.0, 200.0, 100.0))
    fast_assumptions, fast_notes = dcf.default_assumptions_with_notes(fast)
    slow_assumptions, slow_notes = dcf.default_assumptions_with_notes(slow)
    assert fast_assumptions.revenue_growth[0] == pytest.approx(0.15)
    assert slow_assumptions.revenue_growth[0] == pytest.approx(-0.05)
    assert any("clipped" in note for note in fast_notes)
    assert any("clipped" in note for note in slow_notes)


def test_default_fallbacks_for_missing_capex_and_working_capital() -> None:
    company = make_company(capex=None, nwc=None)
    assumptions, notes = dcf.default_assumptions_with_notes(company)
    assert assumptions.capex_pct_revenue == pytest.approx(assumptions.da_pct_revenue)
    assert assumptions.nwc_pct_revenue == 0.0
    assert any(note.startswith("capex") for note in notes)
    assert any(note.startswith("NWC") for note in notes)


def test_default_assumptions_ignore_nan_years() -> None:
    company = make_company()
    company.statements.income.loc[2023, ["revenue", "ebitda"]] = np.nan
    assumptions = dcf.default_assumptions(company)
    # CAGR 2022 -> 2025 over three calendar years, although one year is missing
    assert assumptions.revenue_growth[0] == pytest.approx((1100 / 800) ** (1 / 3) - 1, rel=REL)
    assert all(math.isfinite(m) for m in assumptions.ebitda_margin)
    assert math.isfinite(dcf.run_dcf(company).value_per_share_gordon)


def test_run_dcf_without_assumptions_uses_the_defaults() -> None:
    company = make_company()
    default = dcf.run_dcf(company)
    explicit = dcf.run_dcf(company, dcf.default_assumptions(company))
    assert default.value_per_share_gordon == explicit.value_per_share_gordon
    assert default.value_per_share_exit == explicit.value_per_share_exit
    assert default.wacc_result is not None
    assert default.assumptions.wacc == default.wacc
    assert default.assumptions.exit_multiple == default.exit_multiple
    assert default.forecast.index.tolist() == [2026, 2027, 2028, 2029, 2030]


# --------------------------------------------------------------- exit multiple


def peers_with_multiples(multiples: list[float], **kwargs: object) -> list:
    return [
        make_peer(f"P{i}.ST", enterprise_value=m * 100.0, ebitda=100.0, **kwargs)  # type: ignore[arg-type]
        for i, m in enumerate(multiples)
    ]


def test_exit_multiple_given_in_the_assumptions_wins() -> None:
    company = make_company()
    assumptions = replace(dcf.default_assumptions(company), exit_multiple=33.0)
    peers = peers_with_multiples([8, 9, 10, 11, 12])
    assert dcf.resolve_exit_multiple(company, assumptions, peers) == (33.0, "assumption")


def test_exit_multiple_is_the_industry_peer_median() -> None:
    company = make_company()
    assumptions = dcf.default_assumptions(company)
    peers = peers_with_multiples([7.0, 9.0, 10.0, 12.0, 30.0, 14.0])
    peers.append(make_peer("TEST.ST", enterprise_value=5000.0, ebitda=100.0))  # the target
    peers.append(make_peer("BANK.ST", sector="Financial Services", enterprise_value=100.0))
    peers.append(make_peer("FAR.ST", enterprise_value=7000.0, ebitda=100.0))  # 70x > 50x
    peers.append(make_peer("LOSS.ST", ebitda=-10.0))
    multiple, source = dcf.resolve_exit_multiple(company, assumptions, peers)
    assert multiple == pytest.approx((10.0 + 12.0) / 2)  # median of 7, 9, 10, 12, 14, 30
    assert "industry" in source and "6 peers" in source


def test_exit_multiple_falls_back_to_the_sector_then_to_the_own_multiple() -> None:
    company = make_company()
    assumptions = dcf.default_assumptions(company)
    other_industry = peers_with_multiples([5.0, 6.0, 7.0, 8.0, 9.0], industry="Tools")
    multiple, source = dcf.resolve_exit_multiple(company, assumptions, other_industry)
    assert multiple == pytest.approx(7.0)
    assert "sector" in source
    too_few = peers_with_multiples([5.0, 6.0, 7.0, 8.0])
    multiple, source = dcf.resolve_exit_multiple(company, assumptions, too_few)
    own = (10.0 * 100.0 + 300.0 - 50.0) / (0.2 * 1100.0)
    assert multiple == pytest.approx(own)
    assert source.startswith("own current EV / latest EBITDA")
    assert dcf.resolve_exit_multiple(company, assumptions, None)[0] == pytest.approx(own)


@pytest.mark.parametrize(
    ("multiples", "expected"), [([30.0] * 5, 20.0), ([2.0] * 5, 4.0), ([11.0] * 5, 11.0)]
)
def test_default_exit_multiple_is_clipped_to_the_band(
    multiples: list[float], expected: float
) -> None:
    company = make_company()
    result = dcf.run_dcf(company, peers=peers_with_multiples(multiples))
    assert result.exit_multiple == pytest.approx(expected)
    assert ("clipped" in result.exit_multiple_source) is (expected != multiples[0])
    assert any("exit multiple" in n for n in result.notes) is (expected != multiples[0])


def test_ready_made_peer_group_is_used_without_industry_or_sector_filter() -> None:
    company = make_company()
    group = peers_with_multiples([6.0, 8.0, 10.0, 12.0, 14.0], sector="Technology", industry="X")
    filtered = dcf.run_dcf(company, peers=group)
    assert filtered.exit_multiple_source.startswith("own current EV")
    handed_over = dcf.run_dcf(company, peers=group, peers_are_peer_group=True)
    assert handed_over.exit_multiple == pytest.approx(10.0)
    assert handed_over.wacc_result is not None
    assert handed_over.wacc_result.peer_count == 5


def test_forecast_table_is_internally_consistent_for_default_assumptions() -> None:
    result = dcf.run_dcf(make_company())
    table = result.forecast
    pd.testing.assert_series_equal(
        table["ebit"], table["ebitda"] - table["depreciation_amortization"], check_names=False
    )
    pd.testing.assert_series_equal(
        table["nopat"], table["ebit"] - table["taxes"], check_names=False
    )
    pd.testing.assert_series_equal(
        table["fcff"],
        table["nopat"] + table["depreciation_amortization"] - table["capex"] - table["delta_nwc"],
        check_names=False,
    )
    assert table["pv_fcff"].sum() == pytest.approx(result.pv_fcff, rel=REL)
    assert table["discount_factor"].tolist() == pytest.approx(
        [(1 + result.wacc) ** -(t + 0.5) for t in range(5)], rel=REL
    )
    assert result.enterprise_value_gordon == pytest.approx(
        result.pv_fcff + result.pv_terminal_value_gordon, rel=REL
    )


# ------------------------------------------------- default fallbacks and clipping


def test_default_margin_falls_back_to_the_latest_year_and_is_capped() -> None:
    recovering = make_company(margin=(-0.60, -0.50, -0.40, 0.10))
    assumptions, notes = dcf.default_assumptions_with_notes(recovering)
    assert assumptions.ebitda_margin == pytest.approx([0.10] * 5)
    assert any("latest year's margin used" in note for note in notes)
    rich = make_company(margin=0.90)
    assumptions, notes = dcf.default_assumptions_with_notes(rich)
    assert assumptions.ebitda_margin == pytest.approx([0.70] * 5)
    assert any("capped at 70%" in note for note in notes)


def test_default_depreciation_fallback_chain() -> None:
    from_cash_flow = make_company(da=0.04)
    from_cash_flow.statements.income["depreciation_amortization"] = np.nan
    assumptions, notes = dcf.default_assumptions_with_notes(from_cash_flow)
    assert assumptions.da_pct_revenue == pytest.approx(0.04)
    assert any("cash flow statement D&A used" in note for note in notes)

    from_ebit = make_company(da=0.04)
    from_ebit.statements.income["depreciation_amortization"] = np.nan
    from_ebit.statements.cash_flow["depreciation_amortization"] = np.nan
    assumptions, notes = dcf.default_assumptions_with_notes(from_ebit)
    assert assumptions.da_pct_revenue == pytest.approx(0.04)  # EBITDA - EBIT
    assert any("derived as EBITDA - EBIT" in note for note in notes)

    nothing = make_company(da=0.04)
    for frame in (nothing.statements.income, nothing.statements.cash_flow):
        frame["depreciation_amortization"] = np.nan
    nothing.statements.income["ebit"] = np.nan
    assumptions, notes = dcf.default_assumptions_with_notes(nothing)
    assert assumptions.da_pct_revenue == 0.0
    assert any("set to 0 % of revenue" in note for note in notes)


def test_default_ratios_are_clipped_to_their_bands() -> None:
    company = make_company(margin=0.69, da=0.60, capex=0.80, nwc=5.0)
    assumptions, notes = dcf.default_assumptions_with_notes(company)
    assert assumptions.da_pct_revenue == 0.50
    assert assumptions.capex_pct_revenue == 0.50
    assert assumptions.nwc_pct_revenue == 0.60
    assert sum("clipped" in note for note in notes) == 3
    negative = make_company(nwc=-0.9)
    assert dcf.default_assumptions(negative).nwc_pct_revenue == -0.30


def test_default_growth_without_usable_history_is_the_long_term_rate() -> None:
    company = make_company()
    company.statements.income.loc[[2022, 2023, 2024], "revenue"] = np.nan
    assumptions, notes = dcf.default_assumptions_with_notes(company)
    assert assumptions.revenue_growth == pytest.approx([0.02] * 5)
    assert any("no usable history" in note for note in notes)
    short = make_company(years=(2024, 2025), revenue=(1000.0, 1100.0))
    assumptions, notes = dcf.default_assumptions_with_notes(short)
    assert assumptions.revenue_growth[0] == pytest.approx(0.10)
    assert any("CAGR based on 1 year(s)" in note for note in notes)


def test_exit_multiple_midpoint_fallback_for_a_net_cash_company() -> None:
    # more cash than market cap: the own EV / EBITDA is negative and cannot be used
    company = make_company(cash=5000.0)
    result = dcf.run_dcf(company)
    assert result.exit_multiple == pytest.approx(12.0)  # midpoint of the 4x .. 20x band
    assert result.exit_multiple_source.startswith("fallback")
    assert any("exit multiple: fallback" in note for note in result.notes)
    assert result.net_debt == 300.0 - 5000.0
    assert result.equity_value_gordon == pytest.approx(result.enterprise_value_gordon + 4700.0)


def test_negative_enterprise_value_is_not_applicable() -> None:
    burning = replace(
        HAND_ASSUMPTIONS,
        revenue_growth=(0.0,) * 5,
        ebitda_margin=(-2.0, -2.0, -2.0, -2.0, 0.25),
        wacc=0.30,
    )
    with pytest.raises(NotApplicableError) as error:
        dcf.run_dcf(hand_company(), burning)
    assert category(error) == "non-finite result"


# ------------------------------------------ range fallback rule, peer multiple


def test_invalid_range_cell_is_replaced_by_the_nearest_valid_cell() -> None:
    """WACC 3.8 %, g 2 %: the intended high cell (2.8 %, 2.5 %) has a 0.3 pp spread."""
    company = hand_company()
    tight = replace(HAND_ASSUMPTIONS, wacc=0.038, terminal_growth=0.02)
    result = dcf.run_dcf(company, tight)
    grid = result.sensitivity_gordon
    spread = np.subtract.outer(grid.index.to_numpy(), grid.columns.to_numpy())
    valid = spread >= dcf.MIN_WACC_GROWTH_SPREAD - 1e-12
    assert np.array_equal(np.isnan(grid.to_numpy()), ~valid)  # exactly the base-case rule
    assert math.isnan(grid.iloc[0, 3])

    # the two valid neighbours between the intended cell and the base case are one
    # grid step away each: (WACC 2.8 %, g 2.0 %) and (WACC 3.3 %, g 2.5 %); the more
    # extreme (higher) of the two is the high end
    neighbours = [
        dcf.run_dcf(
            company, replace(HAND_ASSUMPTIONS, wacc=w, terminal_growth=g)
        ).value_per_share_gordon
        for w, g in ((0.028, 0.02), (0.033, 0.025))
    ]
    gordon, exit_row = result.to_valuation_results()
    assert gordon.high == pytest.approx(max(neighbours), rel=REL)
    assert gordon.high > gordon.base
    assert gordon.details["range_fallback_to_base"] is True
    # the low end (WACC 4.8 %, g 1.5 %) is valid and untouched
    low = dcf.run_dcf(company, replace(HAND_ASSUMPTIONS, wacc=0.048, terminal_growth=0.015))
    assert gordon.low == pytest.approx(low.value_per_share_gordon, rel=REL)
    assert gordon.high < 5 * gordon.base  # no exploding perpetuity in the football field
    for row in (gordon, exit_row):
        definition = row.details["range_definition"]
        assert "below 0.5%" in definition and "nearest valid cell" in definition

    # exit-multiple grid: the WACC rows the base case would reject (2.8 % - 2 % is
    # fine, so nothing is cut here) stay valid
    assert not result.sensitivity_exit.isna().any().any()
    assert exit_row.details["range_fallback_to_base"] is False


def test_exit_grid_rows_below_the_minimum_spread_are_cut_as_well() -> None:
    tight = replace(HAND_ASSUMPTIONS, wacc=0.032, terminal_growth=0.02)
    result = dcf.run_dcf(hand_company(), tight)
    exit_grid = result.sensitivity_exit  # WACC rows 2.2 %, 2.7 %, 3.2 %, 3.7 %, 4.2 %
    assert exit_grid.iloc[0].isna().all()  # 0.2 pp spread
    assert exit_grid.iloc[1:].notna().all().all()
    exit_row = result.to_valuation_results()[1]
    # intended high cell (WACC - 1 pp, multiple + 1x) is invalid: nearest valid is one row down
    assert exit_row.high == pytest.approx(exit_grid.iloc[1, 3], rel=REL)
    assert exit_row.details["range_fallback_to_base"] is True
    assert exit_row.low <= exit_row.base <= exit_row.high


def test_ready_made_peer_multiple_replaces_the_peer_median() -> None:
    company = make_company()
    peers = peers_with_multiples([8.0, 9.0, 10.0, 11.0, 12.0])
    assumptions = dcf.default_assumptions(company)
    plain = dcf.run_dcf(company, peers=peers)
    assert plain.exit_multiple == pytest.approx(10.0)
    given = dcf.run_dcf(company, peers=peers, peer_multiple=(13.5, "my label"))
    assert (given.exit_multiple, given.exit_multiple_source) == (13.5, "my label")
    assert given.wacc == plain.wacc  # the peers still feed the beta
    clipped = dcf.resolve_exit_multiple(company, assumptions, peers, peer_multiple=(31.0, "x"))
    assert clipped == (20.0, "x, 31.0x clipped to 20.0x")
    # an explicit assumption wins; an unusable value is ignored
    explicit = replace(assumptions, exit_multiple=7.0)
    assert dcf.resolve_exit_multiple(company, explicit, peers, peer_multiple=(13.5, "x")) == (
        7.0,
        "assumption",
    )
    for bad in (0.0, -4.0, math.nan):
        multiple, _ = dcf.resolve_exit_multiple(
            company, assumptions, peers, peer_multiple=(bad, "x")
        )
        assert multiple == pytest.approx(10.0)
