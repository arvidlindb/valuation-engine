"""The Streamlit app under ``streamlit.testing.v1.AppTest`` (offline, no server)."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace

import pytest
from streamlit.testing.v1 import AppTest

from support import REPO_ROOT, make_company
from valuation_engine import pipeline
from valuation_engine.core.types import MarketParameters, NotApplicableError
from valuation_engine.models import dcf, lbo

APP = REPO_ROOT / "app" / "streamlit_app.py"
TIMEOUT = 180


def metrics(app: AppTest) -> dict[str, str]:
    return {metric.label: str(metric.value) for metric in app.metric}


def titles(app: AppTest) -> list[str]:
    return [str(title.value) for title in app.title]


def problems(app: AppTest) -> list[str]:
    return [str(e.value) for e in app.exception] + [str(e.value) for e in app.error]


@pytest.fixture(scope="module")
def app() -> AppTest:
    return AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()


def test_app_runs_for_atlas_copco_and_shows_the_pipeline_numbers(app: AppTest, atco_report) -> None:
    assert problems(app) == []
    assert [w.value for w in app.warning] == []
    assert app.selectbox(key="ticker").value == "ATCO-A.ST"  # the default company
    assert "Atlas Copco AB (publ)" in titles(app)
    assert len(app.tabs) == 6
    shown = metrics(app)
    assert shown["Share price"] == "SEK 192.25"
    assert shown["Latest fiscal year"] == "2025"
    # an untouched sidebar is exactly the default pipeline run
    dcf_result = atco_report.dcf
    assert shown["Value per share, Gordon"] == f"SEK {dcf_result.value_per_share_gordon:,.2f}"
    assert shown["Value per share, exit multiple"] == f"SEK {dcf_result.value_per_share_exit:,.2f}"
    assert shown["WACC"] == f"{dcf_result.wacc:.2%}"
    assert shown["Exit multiple"] == f"{dcf_result.exit_multiple:.1f}x"
    assert shown["Peers"] == str(len(atco_report.comps.peers))
    captions = " ".join(c.value for c in app.caption)
    assert "offline data" in captions and "market data as of 2026-11-13" in captions
    # offline is forced by the environment: the switch is on and locked
    assert app.toggle[0].value is True and app.toggle[0].disabled is True


def test_app_runs_for_a_bank(app: AppTest) -> None:
    app.selectbox(key="ticker").set_value("SEB-A.ST").run()
    assert problems(app) == []
    assert any("Skandinaviska Enskilda Banken" in title for title in titles(app))
    assert "Atlas Copco AB (publ)" not in titles(app)
    captions = " ".join(c.value for c in app.caption)
    assert "DCF, LBO and Monte Carlo are not applicable to this company" in captions
    shown = metrics(app)
    assert shown["Share price"] == "SEK 222.70"
    assert "Value per share, Gordon" not in shown
    assert not [n for n in app.number_input if n.key and n.key.startswith("growth_")]
    texts = " ".join(str(e.value) for e in [*app.info, *app.warning, *app.markdown, *app.caption])
    assert "financial sector" in texts


def test_app_does_not_claim_a_silhouette_winner_when_nothing_was_clustered(
    app: AppTest, seb_report
) -> None:
    assert seb_report.comps.chosen_k == 1 and seb_report.comps.silhouette_scores == {}
    app.selectbox(key="ticker").set_value("SEB-A.ST").run()
    assert problems(app) == []
    texts = " ".join(str(element.value) for element in app.markdown)
    assert "Peer selection" in texts
    assert "has the highest silhouette score" not in texts


def test_app_applies_sidebar_overrides(app: AppTest, atco, params, universe) -> None:
    app.selectbox(key="ticker").set_value("ATCO-A.ST").run()
    app.number_input(key="growth_ATCO-A.ST").set_value(8.0)
    app.checkbox(key="wacc_on_ATCO-A.ST").check()
    app.run()
    app.number_input(key="wacc_ATCO-A.ST").set_value(9.0).run()
    assert problems(app) == []
    shown = metrics(app)
    assert shown["WACC"] == "9.00%"

    # the same overrides, written out by hand, through the pipeline
    base = dcf.default_assumptions(atco, params)
    growth = tuple(0.08 + (0.02 - 0.08) * t / 4 for t in range(5))
    expected = pipeline.value_company(
        atco,
        universe,
        params,
        dcf_assumptions=replace(base, revenue_growth=growth, wacc=0.09),
        n_simulations=500,
    ).dcf
    assert shown["Value per share, Gordon"] == f"SEK {expected.value_per_share_gordon:,.2f}"
    assert shown["Value per share, exit multiple"] == f"SEK {expected.value_per_share_exit:,.2f}"


def test_app_reports_a_company_without_local_data(app: AppTest) -> None:
    app.selectbox(key="ticker").set_value("AAK.ST").run()
    assert [str(e.value) for e in app.exception] == []
    errors = [str(e.value) for e in app.error]
    assert len(errors) == 1 and "No data for AAK.ST" in errors[0]


# ------------------------------------------------- the app's assumption helpers


@pytest.fixture(scope="module")
def app_module():
    spec = importlib.util.spec_from_file_location("valuation_engine_streamlit_app", APP)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look the module up while it is executed
    try:
        spec.loader.exec_module(module)  # __name__ is not "__main__": main() does not run
        yield module
    finally:
        sys.modules.pop(spec.name, None)


def test_untouched_overrides_mean_model_defaults(app_module) -> None:
    company, params = make_company(), MarketParameters()
    overrides = app_module.Overrides()
    assert app_module.resolve_dcf_assumptions(company, overrides, params) is None
    assert app_module.resolve_lbo_assumptions(company, overrides, params, None) == (
        lbo.default_assumptions(company, params)
    )


def test_overrides_are_applied_to_the_default_assumptions(app_module) -> None:
    company, params = make_company(), MarketParameters()
    overrides = app_module.Overrides(
        revenue_growth_start=0.10,
        ebitda_margin=0.30,
        wacc=0.085,
        terminal_growth=0.03,
        exit_multiple=12.0,
        takeover_premium=0.4,
        senior_debt_multiple=3.0,
        subordinated_debt_multiple=0.5,
    )
    base = dcf.default_assumptions(company, params)
    resolved = app_module.resolve_dcf_assumptions(company, overrides, params)
    fade = tuple(0.10 + (0.02 - 0.10) * t / 4 for t in range(5))
    assert resolved.revenue_growth == pytest.approx(fade)
    assert resolved.ebitda_margin == (0.30,) * 5
    assert (resolved.wacc, resolved.terminal_growth, resolved.exit_multiple) == (0.085, 0.03, 12.0)
    assert resolved.da_pct_revenue == base.da_pct_revenue  # untouched drivers stay
    assert resolved.tax_rate == base.tax_rate

    deal = app_module.resolve_lbo_assumptions(company, overrides, params, 12.0)
    assert deal.revenue_growth == pytest.approx(fade)
    assert deal.ebitda_margin == (0.30,) * 5
    assert (deal.takeover_premium, deal.senior_debt_multiple, deal.subordinated_debt_multiple) == (
        0.4,
        3.0,
        0.5,
    )
    assert deal.exit_multiple == 12.0
    with pytest.raises(NotApplicableError):
        app_module.resolve_dcf_assumptions(
            make_company(sector="Financial Services"), overrides, params
        )
