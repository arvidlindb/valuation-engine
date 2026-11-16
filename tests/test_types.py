"""The shared contract in ``core.types``: validation rules and derived quantities."""

from __future__ import annotations

import dataclasses
import math
from datetime import date

import pytest

from support import make_company, make_peer
from valuation_engine.core import types
from valuation_engine.core.types import (
    DCFAssumptions,
    MarketData,
    MarketParameters,
    NotApplicableError,
    ValuationMethod,
    ValuationResult,
)

# ---------------------------------------------------------------- ValuationResult


@pytest.mark.parametrize(
    ("low", "base", "high"),
    [(1.0, 2.0, 3.0), (2.0, 2.0, 2.0), (-5.0, -1.0, 0.0), (0.0, 0.0, 10.0)],
)
def test_valuation_result_accepts_ordered_ranges(low: float, base: float, high: float) -> None:
    result = ValuationResult(ValuationMethod.DCF_GORDON, "x", low, base, high)
    assert (result.low, result.base, result.high) == (low, base, high)
    assert result.currency == "SEK"
    assert result.details == {}


@pytest.mark.parametrize(
    ("low", "base", "high"),
    [
        (3.0, 2.0, 1.0),
        (2.0, 1.0, 3.0),
        (1.0, 3.0, 2.0),
        (1.0, 1.0, 0.999999),
        (math.nan, 1.0, 2.0),
        (1.0, math.nan, 2.0),
        (1.0, 2.0, math.nan),
    ],
)
def test_valuation_result_rejects_unordered_or_nan_ranges(
    low: float, base: float, high: float
) -> None:
    with pytest.raises(ValueError, match="low <= base <= high"):
        ValuationResult(ValuationMethod.LBO, "LBO", low, base, high)


def test_valuation_result_details_are_not_shared_between_instances() -> None:
    first = ValuationResult(ValuationMethod.MARKET, "a", 1, 2, 3)
    second = ValuationResult(ValuationMethod.MARKET, "b", 1, 2, 3)
    first.details["k"] = 1
    assert second.details == {}


def test_not_applicable_error_is_a_value_error() -> None:
    assert issubclass(NotApplicableError, ValueError)


def test_valuation_method_identifiers_are_stable() -> None:
    assert {m.value for m in ValuationMethod} == {
        "dcf_gordon",
        "dcf_exit_multiple",
        "comps_ev_ebitda",
        "comps_ev_sales",
        "comps_pe",
        "lbo",
        "monte_carlo",
        "market",
    }
    assert ValuationMethod("lbo") is ValuationMethod.LBO
    assert f"{ValuationMethod.COMPS_PE}" == "comps_pe"


# ------------------------------------------------------------------ PeerSnapshot


def test_peer_ratios_are_plain_quotients_of_positive_metrics() -> None:
    peer = make_peer(
        "A.ST",
        market_cap=900.0,
        enterprise_value=1200.0,
        revenue=600.0,
        ebitda=150.0,
        net_income=60.0,
        total_debt=450.0,
    )
    assert peer.ev_ebitda == pytest.approx(8.0)
    assert peer.ev_sales == pytest.approx(2.0)
    assert peer.pe == pytest.approx(15.0)
    assert peer.debt_to_equity == pytest.approx(0.5)


@pytest.mark.parametrize("bad", [None, 0.0, -10.0])
def test_peer_ratios_are_none_for_missing_or_non_positive_denominators(bad: float | None) -> None:
    peer = make_peer("A.ST", revenue=bad, ebitda=bad, net_income=bad)
    assert peer.ev_ebitda is None
    assert peer.ev_sales is None
    assert peer.pe is None


def test_peer_debt_to_equity_rules() -> None:
    assert make_peer("A.ST", total_debt=None).debt_to_equity is None
    assert make_peer("A.ST", market_cap=0.0).debt_to_equity is None
    assert make_peer("A.ST", market_cap=-1.0).debt_to_equity is None
    assert make_peer("A.ST", total_debt=0.0).debt_to_equity == 0.0


@pytest.mark.parametrize(
    ("sector", "expected"),
    [
        ("Financial Services", True),
        ("Real Estate", True),
        ("Industrials", False),
        ("Technology", False),
        (None, False),
    ],
)
def test_financial_flag_follows_the_sector(sector: str | None, expected: bool) -> None:
    assert make_peer("A.ST", sector=sector).is_financial is expected
    assert make_company(sector=sector).profile.is_financial is expected
    assert frozenset({"Financial Services", "Real Estate"}) == types.FINANCIAL_SECTORS


def test_peer_snapshot_and_profile_are_immutable() -> None:
    peer = make_peer("A.ST")
    with pytest.raises(dataclasses.FrozenInstanceError):
        peer.market_cap = 1.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        make_company().profile.ticker = "X"  # type: ignore[misc]


# ---------------------------------------------------------------- DCFAssumptions


def _assumptions(**changes: object) -> DCFAssumptions:
    base: dict[str, object] = {
        "revenue_growth": (0.05, 0.04, 0.03),
        "ebitda_margin": (0.2, 0.2, 0.2),
        "da_pct_revenue": 0.04,
        "capex_pct_revenue": 0.05,
        "nwc_pct_revenue": 0.1,
    }
    base.update(changes)
    return DCFAssumptions(**base)  # type: ignore[arg-type]


def test_dcf_assumptions_defaults_and_forecast_years() -> None:
    assumptions = _assumptions()
    assert assumptions.forecast_years == 3
    assert assumptions.tax_rate == 0.206
    assert assumptions.terminal_growth == 0.02
    assert assumptions.wacc is None
    assert assumptions.exit_multiple is None
    assert assumptions.mid_year is True


def test_dcf_assumptions_require_equal_lengths() -> None:
    with pytest.raises(ValueError, match="equal length"):
        _assumptions(revenue_growth=(0.05, 0.04), ebitda_margin=(0.2, 0.2, 0.2))


def test_dcf_assumptions_require_at_least_one_year() -> None:
    with pytest.raises(ValueError, match="at least one forecast year"):
        _assumptions(revenue_growth=(), ebitda_margin=())


# ------------------------------------------------------------ market data, misc.


def test_market_data_net_debt_and_enterprise_value() -> None:
    market = MarketData(
        ticker="A.ST",
        as_of=date(2026, 11, 13),
        currency="SEK",
        price=50.0,
        shares_outstanding=20.0,
        market_cap=1000.0,
        total_debt=400.0,
        cash=150.0,
        minority_interest=30.0,
        preferred_equity=20.0,
    )
    assert market.net_debt == 250.0
    assert market.enterprise_value == 1000.0 + 250.0 + 30.0 + 20.0


def test_statements_helpers() -> None:
    company = make_company(years=(2023, 2024, 2025), revenue=(10.0, 20.0, 30.0))
    statements = company.statements
    assert statements.fiscal_years == [2023, 2024, 2025]
    assert statements.latest_year == 2025
    assert statements.latest("income", "revenue") == 30.0
    assert math.isnan(statements.latest("balance", "inventory"))
    assert company.ticker == "TEST.ST"


def test_standard_columns_have_no_duplicates() -> None:
    for columns in (types.INCOME_COLUMNS, types.BALANCE_COLUMNS, types.CASH_FLOW_COLUMNS):
        assert len(columns) == len(set(columns))
    assert "free_cash_flow" in types.CASH_FLOW_COLUMNS
    assert "net_working_capital" in types.BALANCE_COLUMNS


def test_market_parameter_defaults_match_the_plan() -> None:
    defaults = MarketParameters()
    assert defaults.risk_free_rate == 0.03075
    assert defaults.equity_risk_premium == 0.0423
    assert defaults.tax_rate == 0.206
    assert defaults.benchmark == "^OMXSPI"
    assert defaults.long_term_growth == 0.02


def test_package_exposes_a_version() -> None:
    import valuation_engine

    assert isinstance(valuation_engine.__version__, str)
    assert valuation_engine.__version__


def test_public_api_signatures_match_the_plan() -> None:
    """PLAN.md, section 'Gemensamt kontrakt'."""
    import inspect

    from valuation_engine import data, pipeline
    from valuation_engine.models import comps, dcf, lbo, monte_carlo

    def parameters(function: object) -> dict[str, inspect.Parameter]:
        return dict(inspect.signature(function).parameters)  # type: ignore[arg-type]

    for loader in (data.load_company, data.list_universe, data.load_peer_snapshots):
        params = parameters(loader)
        for name in ("offline", "refresh"):
            assert params[name].kind is inspect.Parameter.KEYWORD_ONLY
            assert params[name].default is False
    assert next(iter(parameters(data.load_company))) == "ticker"
    assert set(parameters(data.load_market_parameters)) == {"offline", "refresh"}

    assert list(parameters(dcf.default_assumptions)) == ["company", "params"]
    run_dcf = parameters(dcf.run_dcf)
    assert list(run_dcf)[:4] == ["company", "assumptions", "params", "peers"]
    assert run_dcf["assumptions"].default is None and run_dcf["peers"].default is None
    assert run_dcf["params"].default == MarketParameters()
    assert list(parameters(comps.run_comps))[:2] == ["company", "universe"]
    run_lbo = parameters(lbo.run_lbo)
    assert list(run_lbo) == ["company", "assumptions", "params"]
    assert run_lbo["assumptions"].default is None and run_lbo["params"].default is None
    assert {"peers_are_peer_group", "peer_multiple", "size_premium"} <= set(run_dcf)
    run_mc = parameters(monte_carlo.run_monte_carlo)
    assert list(run_mc)[:3] == ["company", "assumptions", "config"]
    assert run_mc["config"].default is None
    assert next(iter(parameters(pipeline.run_valuation))) == "ticker"
    for result in (dcf.DCFResult, comps.CompsResult, lbo.LBOResult, monte_carlo.MonteCarloResult):
        assert callable(result.to_valuation_results)
