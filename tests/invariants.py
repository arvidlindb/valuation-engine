"""Invariant checkers shared by the unit tests and the all-fixtures sweep.

Each checker recomputes identities from the tables a result REPORTS (roll-forwards,
bridges, quartiles) with plain arithmetic, so it does not rely on the model's
internals and holds for any company.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from support import linear_percentile
from valuation_engine.core.types import CompanyData, ValuationMethod
from valuation_engine.models import comps, dcf, lbo, monte_carlo

REL = 1e-9


def close(actual: float, expected: float, scale: float = 0.0) -> bool:
    return abs(actual - expected) <= REL * max(abs(expected), abs(scale), 1e-12)


# --------------------------------------------------------------------------- DCF


def assert_dcf_invariants(result: dcf.DCFResult, company: CompanyData) -> None:
    a = result.assumptions
    table = result.forecast
    n = a.forecast_years
    assert len(table) == n
    first_year = company.statements.latest_year + 1
    assert table.index.tolist() == list(range(first_year, first_year + n))

    # forecast identities, line by line
    revenue = float(company.statements.income["revenue"].iloc[-1])
    scale = revenue
    for t, (_, row) in enumerate(table.iterrows()):
        previous = revenue
        revenue = revenue * (1 + a.revenue_growth[t])
        assert close(row["revenue"], revenue)
        assert close(row["ebitda"], revenue * a.ebitda_margin[t], scale)
        assert close(row["depreciation_amortization"], revenue * a.da_pct_revenue, scale)
        ebit = revenue * a.ebitda_margin[t] - revenue * a.da_pct_revenue
        assert close(row["ebit"], ebit, scale)
        assert close(row["taxes"], a.tax_rate * max(ebit, 0.0), scale)
        assert close(row["nopat"], ebit - a.tax_rate * max(ebit, 0.0), scale)
        assert close(row["capex"], revenue * a.capex_pct_revenue, scale)
        assert close(row["delta_nwc"], a.nwc_pct_revenue * (revenue - previous), scale)
        fcff = row["nopat"] + row["depreciation_amortization"] - row["capex"] - row["delta_nwc"]
        assert close(row["fcff"], fcff, scale)
        period = t + (0.5 if a.mid_year else 1.0)
        assert row["discount_period"] == period
        assert close(row["discount_factor"], (1 + result.wacc) ** -period)
        assert close(row["pv_fcff"], row["fcff"] * row["discount_factor"], scale)
    assert close(result.pv_fcff, float(table["pv_fcff"].sum()), scale)

    # terminal values
    wacc, g = result.wacc, a.terminal_growth
    assert wacc - g >= dcf.MIN_WACC_GROWTH_SPREAD - 1e-12
    assert result.terminal_fcff > 0
    mid = (1 + wacc) ** 0.5 if a.mid_year else 1.0
    tv_gordon = result.terminal_fcff / (wacc - g) * mid
    assert close(result.terminal_value_gordon, tv_gordon)
    assert close(result.pv_terminal_value_gordon, tv_gordon / (1 + wacc) ** n)
    last_ebitda = float(table["ebitda"].iloc[-1])
    assert close(result.terminal_value_exit, result.exit_multiple * last_ebitda)
    assert close(result.pv_terminal_value_exit, result.terminal_value_exit / (1 + wacc) ** n)
    assert close(result.implied_exit_multiple, tv_gordon / last_ebitda)

    # EV, bridge, per share
    market = company.market
    claims = result.net_debt + result.minority_interest + result.preferred_equity
    assert close(result.net_debt, max(market.total_debt, 0.0) - market.cash, scale)
    assert result.shares_outstanding == market.shares_outstanding
    for ev, pv_tv, equity, per_share, tv_share in (
        (
            result.enterprise_value_gordon,
            result.pv_terminal_value_gordon,
            result.equity_value_gordon,
            result.value_per_share_gordon,
            result.tv_share_gordon,
        ),
        (
            result.enterprise_value_exit,
            result.pv_terminal_value_exit,
            result.equity_value_exit,
            result.value_per_share_exit,
            result.tv_share_exit,
        ),
    ):
        assert ev > 0
        assert close(ev, result.pv_fcff + pv_tv, scale)
        assert close(equity, ev - claims, ev)
        assert close(per_share, equity / market.shares_outstanding, ev / market.shares_outstanding)
        assert close(tv_share, pv_tv / ev)

    # WACC build-up
    w = result.wacc_result
    if w is not None:
        assert close(
            w.cost_of_equity, w.risk_free_rate + w.beta * w.equity_risk_premium + w.size_premium
        )
        assert close(w.after_tax_cost_of_debt, w.pre_tax_cost_of_debt * (1 - w.tax_rate))
        assert close(w.equity_weight + w.debt_weight, 1.0)
        assert close(w.equity_weight, 1 / (1 + w.debt_to_equity))
        assert close(
            w.wacc, w.equity_weight * w.cost_of_equity + w.debt_weight * w.after_tax_cost_of_debt
        )
        assert w.wacc == result.wacc
        assert 0.5 <= w.beta <= 2.0
        assert 0.0 <= w.debt_to_equity <= 2.0
        assert w.risk_free_rate + 0.005 - 1e-12 <= w.pre_tax_cost_of_debt
        assert w.pre_tax_cost_of_debt <= w.risk_free_rate + 0.06 + 1e-12
        assert w.size_premium in {0.0, 0.005, 0.010, 0.020, 0.030}
        if w.raw_beta is not None and w.blume_beta is not None:
            assert close(w.blume_beta, 2 / 3 * w.raw_beta + 1 / 3)
        if w.peer_unlevered_beta is not None:
            relevered = w.peer_unlevered_beta * (1 + (1 - w.tax_rate) * w.debt_to_equity)
            assert close(w.peer_relevered_beta, relevered)
            assert close(w.peer_beta, 2 / 3 * relevered + 1 / 3)

    # sensitivities and ranges
    assert result.sensitivity_gordon.iloc[2, 2] == result.value_per_share_gordon
    assert result.sensitivity_exit.iloc[2, 2] == result.value_per_share_exit
    rows = result.to_valuation_results()
    assert [r.method for r in rows] == [
        ValuationMethod.DCF_GORDON,
        ValuationMethod.DCF_EXIT_MULTIPLE,
    ]
    for row, base in zip(
        rows, (result.value_per_share_gordon, result.value_per_share_exit), strict=True
    ):
        assert row.low <= row.base <= row.high
        assert row.base == base
        assert all(math.isfinite(v) for v in (row.low, row.base, row.high))
    again = dcf.dcf_core(**result.core_inputs)
    assert again.value_per_share_gordon[0] == result.value_per_share_gordon


# --------------------------------------------------------------------------- LBO


def assert_lbo_invariants(result: lbo.LBOResult, company: CompanyData) -> None:
    a = result.assumptions
    market = company.market
    money = max(abs(result.entry_enterprise_value), abs(result.sponsor_equity), 1.0)
    tol = 1e-9 * money

    # entry
    debt, cash = max(market.total_debt, 0.0), max(market.cash, 0.0)
    claims = debt - cash + market.minority_interest + market.preferred_equity
    equity_price = result.entry_price_per_share * market.shares_outstanding
    assert abs(result.entry_equity_value - equity_price) <= tol
    assert abs(result.entry_enterprise_value - (equity_price + claims)) <= tol
    entry_ebitda = float(company.statements.income["ebitda"].iloc[-1])
    assert result.entry_ebitda == entry_ebitda
    assert close(result.entry_multiple, result.entry_enterprise_value / entry_ebitda)
    if a.entry_price_per_share is None:
        assert close(result.entry_price_per_share, market.price * (1 + a.takeover_premium))

    # sources and uses
    table = result.sources_and_uses
    sources = table[table["side"] == "sources"].set_index("item")["amount"]
    uses = table[table["side"] == "uses"].set_index("item")["amount"]
    assert abs(sources.sum() - uses.sum()) <= tol
    assert sources["Senior term loan"] == result.senior_debt >= 0
    assert sources["Subordinated debt"] == result.subordinated_debt >= 0
    assert sources["Sponsor equity"] == result.sponsor_equity > 0
    assert abs(sources["Existing cash on balance sheet"] - cash) <= tol
    assert abs(uses["Purchase of equity"] - equity_price) <= tol
    assert abs(uses["Refinancing of existing debt"] - debt) <= tol
    new_debt = result.senior_debt + result.subordinated_debt
    assert abs(uses["Financing fees"] - a.financing_fee_pct * new_debt) <= tol
    assert (
        abs(uses["Transaction fees"] - a.transaction_fee_pct * result.entry_enterprise_value) <= tol
    )
    for side in ("sources", "uses"):
        assert table.loc[table["side"] == side, "pct"].sum() == pytest.approx(1.0)
    assert close(result.leverage, new_debt / entry_ebitda)
    assert close(result.equity_share, result.sponsor_equity / (new_debt + result.sponsor_equity))
    assert result.equity_share >= a.min_equity_share - 1e-9

    # year-by-year roll-forward
    forecast, schedule = result.forecast, result.debt_schedule
    years = a.holding_period
    assert len(forecast) == len(schedule) == years
    first = company.statements.latest_year + 1
    assert forecast.index.tolist() == schedule.index.tolist() == list(range(first, first + years))
    assert result.exit_year == first + years - 1

    revenue = float(company.statements.income["revenue"].iloc[-1])
    nwc = a.nwc_pct_revenue * revenue
    senior, sub, revolver = result.senior_debt, result.subordinated_debt, 0.0
    cash_balance = a.min_cash_pct_revenue * revenue
    revolver_limit = a.revolver_limit_x_ebitda * entry_ebitda
    scheduled = a.senior_amortization_pct * result.senior_debt
    senior_rate = a.risk_free_rate + a.senior_spread
    sub_rate = a.risk_free_rate + a.subordinated_spread
    breach_years = []
    for t, year in enumerate(forecast.index):
        f, d = forecast.loc[year], schedule.loc[year]
        revenue *= 1 + a.revenue_growth[t]
        assert abs(f["revenue"] - revenue) <= tol
        assert abs(f["ebitda"] - a.ebitda_margin[t] * revenue) <= tol
        assert abs(f["ebit"] - (f["ebitda"] - f["depreciation_amortization"])) <= tol
        assert abs(f["depreciation_amortization"] - a.da_pct_revenue * revenue) <= tol
        assert abs(f["capex"] - a.capex_pct_revenue * revenue) <= tol
        assert abs(f["net_working_capital"] - a.nwc_pct_revenue * revenue) <= tol
        assert abs(f["change_in_nwc"] - (a.nwc_pct_revenue * revenue - nwc)) <= tol
        nwc = a.nwc_pct_revenue * revenue

        # opening balances chain from the previous closing balances
        assert abs(d["senior_opening"] - senior) <= tol
        assert abs(d["subordinated_opening"] - sub) <= tol
        assert abs(d["revolver_opening"] - revolver) <= tol
        assert abs(d["cash_opening"] - cash_balance) <= tol

        # interest on opening balances
        assert abs(d["senior_interest"] - senior_rate * senior) <= tol
        assert abs(d["subordinated_interest"] - sub_rate * sub) <= tol
        assert abs(d["revolver_interest"] - senior_rate * revolver) <= tol
        assert abs(d["shortfall_interest"] - senior_rate * max(-cash_balance, 0.0)) <= tol
        total_interest = (
            d["senior_interest"]
            + d["subordinated_interest"]
            + d["revolver_interest"]
            + d["shortfall_interest"]
        )
        assert abs(d["total_interest"] - total_interest) <= tol
        assert abs(f["interest_expense"] - total_interest) <= tol
        pik = d["subordinated_interest"] if a.subordinated_pik else 0.0
        assert abs(d["subordinated_pik_accrual"] - pik) <= tol
        assert abs(d["cash_interest"] - (total_interest - pik)) <= tol

        # tax with the interest deduction cap, no credit for losses
        deductible = total_interest
        if a.interest_deduction_cap_pct_ebitda is not None:
            cap = a.interest_deduction_cap_pct_ebitda * max(f["ebitda"], 0.0)
            deductible = min(total_interest, cap)
        assert abs(f["deductible_interest"] - deductible) <= tol
        tax = a.tax_rate * max(f["ebit"] - deductible, 0.0)
        assert abs(f["tax"] - tax) <= tol
        assert f["tax"] >= 0
        assert abs(f["pretax_income"] - (f["ebit"] - total_interest)) <= tol
        assert abs(f["net_income"] - (f["ebit"] - total_interest - tax)) <= tol

        # cash flow waterfall
        cfads = f["ebitda"] - tax - f["capex"] - f["change_in_nwc"]
        assert abs(f["cash_flow_available_for_debt_service"] - cfads) <= tol
        assert abs(d["cash_flow_available_for_debt_service"] - cfads) <= tol
        mandatory = d["senior_mandatory_amortization"]
        assert abs(mandatory - min(scheduled, senior)) <= tol
        excess = cfads - d["cash_interest"] - mandatory
        assert abs(d["excess_cash_flow"] - excess) <= tol
        minimum_cash = a.min_cash_pct_revenue * revenue
        assert abs(d["minimum_cash"] - minimum_cash) <= tol
        surplus = cash_balance + excess - minimum_cash
        sweep, draw, repay = d["senior_cash_sweep"], d["revolver_draw"], d["revolver_repayment"]
        assert min(sweep, draw, repay, mandatory) >= -tol
        assert repay <= revolver + tol and repay <= max(surplus, 0.0) + tol
        assert draw <= max(-surplus, 0.0) + tol
        assert draw <= max(revolver_limit - revolver, 0.0) + tol
        assert draw * repay <= tol * money  # never both in one year
        # the sweep is limited by the cash above the minimum and by the balance
        assert sweep <= a.cash_sweep_pct * max(surplus - repay, 0.0) + tol
        assert sweep <= senior - mandatory + tol
        if surplus < 0:
            assert sweep <= tol  # no prepayment out of borrowed money

        # closing balances
        assert abs(d["senior_closing"] - (senior - mandatory - sweep)) <= tol
        assert abs(d["subordinated_closing"] - (sub + pik)) <= tol
        assert abs(d["revolver_closing"] - (revolver + draw - repay)) <= tol
        cash_close = cash_balance + excess + draw - repay - sweep
        assert abs(d["cash_closing"] - cash_close) <= tol
        senior, sub, revolver = (
            d["senior_closing"],
            d["subordinated_closing"],
            d["revolver_closing"],
        )
        cash_balance = d["cash_closing"]
        assert senior >= -tol and sub >= -tol
        assert -tol <= revolver <= revolver_limit + tol
        assert abs(d["total_debt_closing"] - (senior + sub + revolver)) <= tol
        assert abs(d["net_debt_closing"] - (senior + sub + revolver - cash_balance)) <= tol
        breached = cash_balance < minimum_cash - tol
        assert bool(d["liquidity_breach"]) is bool(breached)
        if breached:
            breach_years.append(int(year))
            assert revolver >= revolver_limit - tol  # the revolver is exhausted first
    assert result.liquidity_breach_years == breach_years

    # exit and returns
    assert abs(result.exit_ebitda - forecast["ebitda"].iloc[-1]) <= tol
    if a.exit_multiple is not None:
        assert result.exit_multiple == a.exit_multiple
    else:
        assert close(result.exit_multiple, result.entry_multiple)
    assert close(result.exit_enterprise_value, result.exit_multiple * result.exit_ebitda, money)
    assert abs(result.exit_net_debt - (senior + sub + revolver - cash_balance)) <= tol
    raw_equity = result.exit_enterprise_value - result.exit_net_debt - result.exit_minority_interest
    assert abs(result.exit_equity_value - max(raw_equity, 0.0)) <= tol
    assert result.equity_wiped_out is bool(raw_equity <= 0)
    assert result.exit_minority_interest == market.minority_interest

    flows = result.equity_cash_flows
    assert len(flows) == years + 1
    assert flows[0] == -result.sponsor_equity
    assert all(flow == 0.0 for flow in flows[1:-1])
    assert flows[-1] == result.exit_equity_value
    assert result.moic == result.exit_equity_value / result.sponsor_equity
    if result.exit_equity_value > 0:
        by_hand = result.moic ** (1 / years) - 1
        assert result.irr == pytest.approx(by_hand, abs=1e-9)
    else:
        assert result.irr == -1.0

    # attribution adds up to the value created
    amounts = result.attribution["amount"]
    total = result.exit_equity_value - result.sponsor_equity
    assert abs(amounts["total_value_creation"] - total) <= tol
    assert abs(amounts.drop("total_value_creation").sum() - total) <= 1e-7 * money

    # ability to pay
    targets = (a.target_irr_conservative, a.target_irr_base, a.target_irr_aggressive)
    assert set(result.ability_to_pay) == set(targets)
    solved = [result.ability_to_pay_status[t] == lbo.STATUS_SOLVED for t in targets]
    prices = [result.ability_to_pay[t] for t in targets]
    if all(solved):
        assert 0 < prices[0] <= prices[1] <= prices[2]
    rows = result.to_valuation_results()
    assert len(rows) <= 1
    if rows:
        assert rows[0].method is ValuationMethod.LBO
        assert (rows[0].low, rows[0].base, rows[0].high) == tuple(prices)
        assert rows[0].low <= rows[0].base <= rows[0].high


# -------------------------------------------------------------------- Monte Carlo


def assert_monte_carlo_invariants(
    result: monte_carlo.MonteCarloResult, base: dcf.DCFResult
) -> None:
    n = result.n_simulations
    assert result.values.shape == result.values_exit.shape == (n,)
    assert len(result.inputs) == n
    assert list(result.inputs.columns) == list(monte_carlo.DRIVERS)
    assert result.base_value == base.value_per_share_gordon
    assert result.base_value_exit == base.value_per_share_exit
    finite = np.isfinite(result.values)
    assert result.n_valid == int(finite.sum())
    assert result.n_valid + result.n_non_finite == n
    assert np.array_equal(finite, np.isfinite(result.values_exit))
    valid = result.values[finite]

    levels = [result.percentiles[p] for p in (5, 10, 25, 50, 75, 90, 95)]
    assert levels == sorted(levels)
    assert levels == pytest.approx(np.percentile(valid, [5, 10, 25, 50, 75, 90, 95]), rel=1e-12)
    assert valid.min() <= result.mean <= valid.max()
    assert result.mean == pytest.approx(valid.mean(), rel=1e-12)
    assert result.std == pytest.approx(valid.std(ddof=1), rel=1e-9, abs=1e-12)
    assert result.standard_error == pytest.approx(result.std / math.sqrt(len(valid)), rel=1e-12)
    assert result.prob_above_price == pytest.approx(float((valid > result.price).mean()))
    assert 0.0 <= result.prob_above_price <= 1.0

    # sampled inputs stay inside the resolved bounds, centred on the base case
    for name in monte_carlo.DRIVERS:
        spec = result.config.spec(name)
        offsets = result.inputs[name].to_numpy() - result.base_inputs[name]
        assert offsets.min() >= spec.low - 1e-12
        assert offsets.max() <= spec.high + 1e-12
    spread = result.inputs["wacc"] - result.inputs["terminal_growth"]
    if result.n_non_finite == 0:
        assert spread.min() >= dcf.MIN_WACC_GROWTH_SPREAD

    contribution = result.variance_contribution
    assert list(contribution.index) == list(monte_carlo.DRIVERS)
    assert (contribution >= 0).all()
    if result.std > 0 and len(valid) >= 3:
        assert contribution.sum() == pytest.approx(1.0)

    tornado = result.tornado
    assert set(tornado.index) == set(monte_carlo.DRIVERS)
    assert tornado["swing"].tolist() == pytest.approx(
        (tornado["value_high"] - tornado["value_low"]).abs().tolist()
    )
    assert tornado["swing"].is_monotonic_decreasing
    assert (tornado["input_low"] <= tornado["input_high"]).all()

    convergence = result.convergence
    assert list(convergence.columns) == ["n", "running_mean", "running_median", "standard_error"]
    assert convergence["n"].iloc[-1] == len(valid)
    assert convergence["running_mean"].iloc[-1] == pytest.approx(result.mean, rel=1e-9)
    assert convergence["standard_error"].iloc[-1] == pytest.approx(
        result.standard_error, rel=1e-9, abs=1e-12
    )

    (row,) = result.to_valuation_results()
    assert row.method is ValuationMethod.MONTE_CARLO
    assert (row.low, row.base, row.high) == (
        result.percentiles[10],
        result.percentiles[50],
        result.percentiles[90],
    )


# ------------------------------------------------------------------------- comps


def assert_comps_invariants(result: comps.CompsResult, company: CompanyData) -> None:
    market = company.market
    peers = result.peers
    assert company.ticker not in peers.index
    assert company.ticker not in result.candidates.index
    assert peers.index.is_unique
    assert set(peers.index) <= set(result.candidates.index)
    assert result.candidates.loc[peers.index, "is_peer"].all()

    if result.silhouette_scores:
        best = max(result.silhouette_scores.values())
        assert result.silhouette_scores[result.chosen_k] == best
        assert min(result.silhouette_scores) == 2
        assert sorted(result.silhouette_scores) == list(range(2, max(result.silhouette_scores) + 1))
    else:
        assert result.chosen_k == 1

    applicable = result.applicable_multiples
    if company.profile.is_financial:
        assert set(applicable) <= {"pe"}
        assert {"ev_ebitda", "ev_sales"} <= set(result.skipped)
    assert set(applicable).isdisjoint(result.skipped)
    assert set(applicable) | set(result.skipped) == {"ev_ebitda", "ev_sales", "pe"}

    claims = market.total_debt - market.cash + market.minority_interest + market.preferred_equity
    metrics = {"ev_ebitda": "ebitda", "ev_sales": "revenue", "pe": "net_income"}
    for name in applicable:
        row = result.implied_values.loc[name]
        used = peers.loc[peers[f"used_{name}"], name].tolist()
        assert len(used) == row["n_peers"] >= 3
        assert all(v > 0 for v in used)
        quartiles = [linear_percentile(used, q) for q in (25, 50, 75)]
        assert [row["multiple_low"], row["multiple_base"], row["multiple_high"]] == pytest.approx(
            quartiles, rel=1e-9
        )
        metric = float(company.statements.income[metrics[name]].iloc[-1])
        assert metric > 0 and row["metric_value"] == metric
        for suffix, q in zip(("low", "base", "high"), quartiles, strict=True):
            if name == "pe":
                equity, ev = q * metric, q * metric + claims
            else:
                ev, equity = q * metric, q * metric - claims
            assert row[f"ev_{suffix}"] == pytest.approx(ev, rel=1e-9)
            assert row[f"equity_{suffix}"] == pytest.approx(max(equity, 0.0), rel=1e-9, abs=1e-6)
            assert row[f"per_share_{suffix}"] == pytest.approx(
                max(equity, 0.0) / market.shares_outstanding, rel=1e-9, abs=1e-12
            )
        assert row["equity_high"] > 0
    for row in result.to_valuation_results():
        assert 0.0 <= row.low <= row.base <= row.high
