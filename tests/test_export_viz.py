"""Excel workbook and charts: everything reopens / renders and shows the model's numbers."""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import openpyxl
import pytest
from matplotlib.figure import Figure

from support import make_company
from valuation_engine import pipeline
from valuation_engine.core.types import ValuationMethod, ValuationResult
from valuation_engine.export import SHEET_NAMES, export_to_excel
from valuation_engine.viz import (
    CHART_NAMES,
    ReportBundle,
    build_charts,
    football_field,
    message_figure,
    order_valuations,
    report_valuations,
    save_charts,
    save_figure,
    trading_range_result,
)

MILLION = 1e6


# ----------------------------------------------------------- workbook helpers


def find_row(ws, label: str, start: int = 1, *, exact: bool = False) -> int:
    """Row whose label cell (column B) starts with (or equals) ``label``."""
    for row in range(start, ws.max_row + 1):
        value = ws.cell(row=row, column=2).value
        if isinstance(value, str) and (value == label if exact else value.startswith(label)):
            return row
    raise AssertionError(f"label {label!r} not found on sheet {ws.title!r}")


def evaluate(ws, ref: str) -> object:
    """Value of a cell; the few simple formulas of the workbook are computed here
    (+ - * /, parentheses, SUM over a range, IF(x=0,0,y), absolute references)."""
    value = ws[ref.replace("$", "")].value
    if not (isinstance(value, str) and value.startswith("=")):
        return value
    expression = value[1:].replace("$", "")

    def expand(match: re.Match[str]) -> str:
        cells = [cell.coordinate for row in ws[f"{match[1]}:{match[2]}"] for cell in row]
        return "(" + "+".join(f"cell('{c}')" for c in cells) + ")"

    expression = re.sub(r"SUM\(([A-Z]+\d+):([A-Z]+\d+)\)", expand, expression)
    expression = re.sub(r"IF\(([^,]+)=0,0,([^)]+)\)", r"(0 if (\1)==0 else (\2))", expression)
    expression = re.sub(r"(?<!')\b([A-Z]{1,2}\d+)\b", r"cell('\1')", expression)
    return eval(expression, {"__builtins__": {}}, {"cell": lambda r: evaluate(ws, r)})  # noqa: S307


def value_at(ws, label: str, column: str = "C", start: int = 1) -> object:
    return evaluate(ws, f"{column}{find_row(ws, label, start)}")


@pytest.fixture(scope="module")
def atco_workbook(atco_report, tmp_path_factory: pytest.TempPathFactory):
    path = tmp_path_factory.mktemp("xlsx") / "atco.xlsx"
    written = export_to_excel(atco_report, path)
    assert written == path and path.is_file()
    return openpyxl.load_workbook(path)


# -------------------------------------------------------------------- workbook


def test_workbook_reopens_with_all_sheets(atco_workbook) -> None:
    assert tuple(atco_workbook.sheetnames) == SHEET_NAMES
    assert SHEET_NAMES == ("Summary", "Assumptions", "DCF", "Comps", "LBO", "Monte Carlo", "Data")
    for sheet in atco_workbook:
        assert sheet.max_row > 10
        assert "Atlas Copco" in sheet["B1"].value
    assert len(atco_workbook["Summary"]._images) == 1  # the embedded football field
    assert atco_workbook.properties.creator == "valuation-engine"


def test_every_formula_of_the_workbook_can_be_evaluated(atco_workbook) -> None:
    formulas = 0
    for sheet in atco_workbook:
        for row in sheet.iter_rows():
            for cell in row:
                if isinstance(cell.value, str) and cell.value.startswith("="):
                    result = evaluate(sheet, cell.coordinate)
                    assert isinstance(result, (int, float)) and math.isfinite(result), (
                        sheet.title,
                        cell.coordinate,
                        cell.value,
                    )
                    formulas += 1
    assert formulas > 30


def test_summary_sheet_equals_the_report(atco_workbook, atco_report) -> None:
    ws = atco_workbook["Summary"]
    market = atco_report.company.market
    assert value_at(ws, "Ticker") == "ATCO-A.ST"
    assert value_at(ws, "Share price") == 192.25
    assert value_at(ws, "Shares outstanding") == pytest.approx(market.shares_outstanding / MILLION)
    assert value_at(ws, "Market capitalization") == pytest.approx(market.market_cap / MILLION)
    assert value_at(ws, "Total debt") == pytest.approx(market.total_debt / MILLION)
    assert value_at(ws, "Net debt") == pytest.approx(market.net_debt / MILLION)
    assert value_at(ws, "Enterprise value") == pytest.approx(market.enterprise_value / MILLION)
    assert value_at(ws, "Latest fiscal year") == 2025

    rows = {row.label: row for row in report_valuations(atco_report)}
    assert len(rows) == 8  # seven methods plus the 52-week range
    for label, row in rows.items():
        r = find_row(ws, label, start=find_row(ws, "Valuation summary"))
        assert ws.cell(row=r, column=3).value == pytest.approx(row.low)
        assert ws.cell(row=r, column=4).value == pytest.approx(row.base)
        assert ws.cell(row=r, column=5).value == pytest.approx(row.high)
        assert evaluate(ws, f"F{r}") == pytest.approx(row.base / 192.25 - 1)
        if row.enterprise_value is not None:
            assert ws.cell(row=r, column=7).value == pytest.approx(row.enterprise_value / MILLION)
            assert ws.cell(row=r, column=8).value == pytest.approx(row.equity_value / MILLION)
    assert value_at(ws, "All methods are applicable") is None


def test_assumptions_sheet_shows_values_and_sources(atco_workbook, atco_report) -> None:
    ws = atco_workbook["Assumptions"]
    row = find_row(ws, "Risk-free rate")
    assert ws.cell(row=row, column=3).value == 0.03075
    assert "Riksbank" in ws.cell(row=row, column=4).value
    assert ws.cell(row=row, column=5).value == "2026-11-13"
    assert ws.cell(row=row, column=6).value == "fixtures"
    row = find_row(ws, "Equity risk premium")
    assert ws.cell(row=row, column=3).value == 0.0423
    assert "Damodaran" in ws.cell(row=row, column=4).value
    assert ws.cell(row=row, column=6).value == "reference file"
    assert value_at(ws, "Corporate tax rate") == 0.206

    dcf, lbo = atco_report.dcf, atco_report.lbo
    start = find_row(ws, "DCF assumptions")
    growth_row = find_row(ws, "Revenue growth", start)
    assert [ws.cell(row=growth_row, column=c).value for c in range(3, 8)] == pytest.approx(
        list(dcf.assumptions.revenue_growth)
    )
    assert value_at(ws, "WACC", start=start) == pytest.approx(dcf.wacc)
    assert value_at(ws, "Terminal growth", start=start) == dcf.assumptions.terminal_growth
    assert value_at(ws, "Exit EV/EBITDA multiple", start=start) == pytest.approx(dcf.exit_multiple)
    start = find_row(ws, "LBO assumptions")
    margin_row = find_row(ws, "EBITDA margin", start)
    assert [ws.cell(row=margin_row, column=c).value for c in range(3, 8)] == pytest.approx(
        list(lbo.assumptions.ebitda_margin)
    )
    assert value_at(ws, "Senior debt multiple", start=start) == 4
    assert value_at(ws, "Exit multiple", start=start) == pytest.approx(dcf.exit_multiple)
    assert value_at(ws, "Simulations") == 10_000 and value_at(ws, "Seed") == 42


def test_dcf_sheet_equals_the_model_including_its_live_formulas(atco_workbook, atco_report) -> None:
    ws = atco_workbook["DCF"]
    dcf = atco_report.dcf
    labels = {
        "Revenue": "revenue",
        "EBITDA": "ebitda",
        "Depreciation & amortization": "depreciation_amortization",
        "EBIT": "ebit",
        "Taxes": "taxes",
        "NOPAT": "nopat",
        "Capex": "capex",
        "Increase in NWC": "delta_nwc",
        "FCFF": "fcff",
        "PV of FCFF": "pv_fcff",
    }
    header = find_row(ws, "SEK m", exact=True)
    assert [ws.cell(row=header, column=c).value for c in range(3, 8)] == [
        f"{year}E" for year in dcf.forecast.index
    ]
    for label, column in labels.items():
        row = next(r for r in range(header, header + 20) if ws.cell(row=r, column=2).value == label)
        cells = [ws.cell(row=row, column=c).value for c in range(3, 8)]
        assert cells == pytest.approx((dcf.forecast[column] / MILLION).tolist()), label
    factor_row = find_row(ws, "Discount factor")
    assert [ws.cell(row=factor_row, column=c).value for c in range(3, 8)] == pytest.approx(
        dcf.forecast["discount_factor"].tolist()
    )

    # the WACC build-up is a chain of formulas that must land on the model's WACC
    wacc = dcf.wacc_result
    assert value_at(ws, "Beta used") == pytest.approx(wacc.beta)
    assert value_at(ws, "Cost of equity") == pytest.approx(wacc.cost_of_equity)
    assert value_at(ws, "After-tax cost of debt") == pytest.approx(wacc.after_tax_cost_of_debt)
    assert value_at(ws, "WACC", start=find_row(ws, "Debt weight")) == pytest.approx(dcf.wacc)

    # so is the bridge from enterprise value to value per share
    start = find_row(ws, "Enterprise value to equity value")
    for column, ev, equity, per_share in (
        ("C", dcf.enterprise_value_gordon, dcf.equity_value_gordon, dcf.value_per_share_gordon),
        ("D", dcf.enterprise_value_exit, dcf.equity_value_exit, dcf.value_per_share_exit),
    ):
        assert value_at(ws, "PV of explicit FCFF", column, start) == pytest.approx(
            dcf.pv_fcff / MILLION
        )
        assert value_at(ws, "Enterprise value", column, start + 1) == pytest.approx(ev / MILLION)
        assert value_at(ws, "Equity value", column, start) == pytest.approx(equity / MILLION)
        assert value_at(ws, "Value per share", column, start) == pytest.approx(per_share)
        assert value_at(ws, "Upside / (downside)", column, start) == pytest.approx(
            per_share / 192.25 - 1
        )
    assert value_at(ws, "Normalized FCFF") == pytest.approx(dcf.terminal_fcff / MILLION)
    assert value_at(ws, "EV/EBITDA multiple at exit", "C") == pytest.approx(
        dcf.implied_exit_multiple
    )
    assert value_at(ws, "EV/EBITDA multiple at exit", "D") == pytest.approx(dcf.exit_multiple)

    # sensitivity grid: axes, centre and corners
    top = find_row(ws, "WACC \\ terminal growth")
    grid = dcf.sensitivity_gordon
    assert [ws.cell(row=top, column=c).value for c in range(3, 8)] == pytest.approx(
        grid.columns.tolist()
    )
    for i in range(5):
        assert ws.cell(row=top + 1 + i, column=2).value == pytest.approx(grid.index[i])
        cells = [ws.cell(row=top + 1 + i, column=c).value for c in range(3, 8)]
        assert cells == pytest.approx(grid.iloc[i].tolist())
    assert ws.cell(row=top + 3, column=5).value == pytest.approx(dcf.value_per_share_gordon)


def test_comps_lbo_and_monte_carlo_sheets_equal_the_models(atco_workbook, atco_report) -> None:
    comps, lbo, mc = atco_report.comps, atco_report.lbo, atco_report.monte_carlo

    ws = atco_workbook["Comps"]
    assert value_at(ws, "Clusters chosen (k)") == comps.chosen_k
    assert value_at(ws, "Peers selected") == len(comps.peers)
    first_peer = find_row(ws, "Ticker") + 1
    listed = [ws.cell(row=first_peer + i, column=2).value for i in range(len(comps.peers))]
    assert listed == comps.peer_tickers
    start = find_row(ws, "Peer multiples")
    for label, name in (("EV/EBITDA", "ev_ebitda"), ("EV/Sales", "ev_sales"), ("P/E", "pe")):
        row = find_row(ws, label, start)
        summary = comps.multiples_summary.loc[name]
        cells = [ws.cell(row=row, column=c).value for c in range(3, 10)]
        assert cells == pytest.approx(
            summary[["n", "min", "p25", "median", "mean", "p75", "max"]].tolist()
        )
    start = find_row(ws, "Implied values")
    row = find_row(ws, "EV/EBITDA", start)
    implied = comps.implied_values.loc["ev_ebitda"]
    assert ws.cell(row=row, column=4).value == pytest.approx(implied["metric_value"] / MILLION)
    assert ws.cell(row=row, column=6).value == pytest.approx(implied["multiple_base"])
    assert ws.cell(row=row, column=8).value == pytest.approx(implied["ev_base"] / MILLION)
    assert ws.cell(row=row, column=9).value == pytest.approx(implied["equity_base"] / MILLION)
    assert ws.cell(row=row, column=10).value == pytest.approx(implied["per_share_low"])

    ws = atco_workbook["LBO"]
    assert value_at(ws, "Offer price per share") == pytest.approx(lbo.entry_price_per_share)
    assert value_at(ws, "Entry enterprise value") == pytest.approx(
        lbo.entry_enterprise_value / MILLION
    )
    assert value_at(ws, "Entry EV/EBITDA") == pytest.approx(lbo.entry_multiple)
    assert value_at(ws, "Senior term loan") == pytest.approx(lbo.senior_debt / MILLION)
    assert value_at(ws, "Sponsor equity", start=find_row(ws, "Sources and uses")) == (
        pytest.approx(lbo.sponsor_equity / MILLION)
    )
    total_sources = value_at(ws, "Total sources")
    total_uses = value_at(ws, "Total uses")
    assert total_sources == pytest.approx(total_uses, rel=1e-12)
    assert value_at(ws, "Check: sources less uses") == pytest.approx(0.0, abs=1e-6)
    assert value_at(ws, "Total sources", "D") == pytest.approx(1.0)
    start = find_row(ws, "Debt schedule")
    closing = find_row(ws, "Senior closing", start)
    assert [ws.cell(row=closing, column=c).value for c in range(3, 8)] == pytest.approx(
        (lbo.debt_schedule["senior_closing"] / MILLION).tolist()
    )
    start = find_row(ws, "Exit and returns")
    assert value_at(ws, "Exit EV/EBITDA", start=start) == pytest.approx(lbo.exit_multiple)
    assert value_at(ws, "Sponsor proceeds at exit") == pytest.approx(
        lbo.exit_equity_value / MILLION
    )
    assert value_at(ws, "IRR", start=start) == pytest.approx(lbo.irr)
    assert value_at(ws, "MOIC", start=start) == pytest.approx(lbo.moic)
    start = find_row(ws, "Ability to pay")
    prices = {
        ws.cell(row=r, column=2).value: ws.cell(row=r, column=3).value
        for r in range(start + 2, start + 5)
    }
    assert prices == pytest.approx(lbo.ability_to_pay)

    ws = atco_workbook["Monte Carlo"]
    assert value_at(ws, "Simulations") == 10_000
    assert value_at(ws, "Base case value per share") == pytest.approx(mc.base_value)
    assert value_at(ws, "Mean") == pytest.approx(mc.mean)
    assert value_at(ws, "Standard deviation") == pytest.approx(mc.std)
    assert value_at(ws, "Standard error of the mean") == pytest.approx(mc.standard_error)
    assert value_at(ws, "Probability of value above price") == pytest.approx(mc.prob_above_price)
    for level in (5, 10, 25, 50, 75, 90, 95):
        row = find_row(ws, f"P{level}", find_row(ws, "Percentiles"))
        assert ws.cell(row=row, column=3).value == pytest.approx(mc.percentiles[level])
        assert ws.cell(row=row, column=4).value == pytest.approx(mc.percentiles_exit[level])


def test_data_sheet_is_the_standardized_history_in_millions(atco_workbook, atco) -> None:
    ws = atco_workbook["Data"]
    income = atco.statements.income
    header = find_row(ws, "SEK m", exact=True)
    assert [ws.cell(row=header, column=c).value for c in range(3, 7)] == [
        "2022A",
        "2023A",
        "2024A",
        "2025A",
    ]
    for label, column in (
        ("Revenue", "revenue"),
        ("EBITDA", "ebitda"),
        ("Net income", "net_income"),
    ):
        row = find_row(ws, label)
        cells = [ws.cell(row=row, column=c).value for c in range(3, 7)]
        assert cells == pytest.approx((income[column] / MILLION).tolist())
    capex = find_row(ws, "Capex")
    assert ws.cell(row=capex, column=6).value == pytest.approx(6187.0)
    assert value_at(ws, "income.revenue").startswith("is:Total Revenue")


def test_bank_workbook_keeps_all_sheets_and_states_why(seb_report, tmp_path: Path) -> None:
    path = export_to_excel(seb_report, tmp_path / "nested" / "seb", embed_football_field=False)
    assert path.name == "seb.xlsx"  # the suffix is appended, the folder created
    workbook = openpyxl.load_workbook(path)
    assert tuple(workbook.sheetnames) == SHEET_NAMES
    assert workbook["Summary"]._images == []
    for sheet, key in (("DCF", "dcf"), ("LBO", "lbo"), ("Monte Carlo", "monte_carlo")):
        text = " ".join(
            str(cell.value) for row in workbook[sheet].iter_rows() for cell in row if cell.value
        )
        assert "Not applicable" in text or "not applicable" in text
        assert seb_report.not_applicable[key] in text
    summary = workbook["Summary"]
    row = find_row(summary, "Comps - P/E")
    (bar,) = seb_report.valuations
    assert summary.cell(row=row, column=4).value == pytest.approx(bar.base)
    find_row(summary, "Methods not applicable")
    income = workbook["Data"]
    assert income.cell(row=find_row(income, "EBITDA"), column=3).value in (None, "n/a", "-")


def test_workbook_for_a_euro_company_and_a_report_without_sources(reports, tmp_path: Path) -> None:
    report = reports["VSURE.ST"]
    workbook = openpyxl.load_workbook(export_to_excel(report, tmp_path / "vsure.xlsx"))
    ws = workbook["Summary"]
    assert value_at(ws, "Trading currency") == "EUR"
    assert "EUR" in ws.cell(row=find_row(ws, "Share price"), column=2).value
    gordon = find_row(ws, "DCF - Gordon growth")
    assert ws.cell(row=gordon, column=3).value < 0  # negative values are shown as computed

    bundle = ReportBundle(
        company=report.company,
        dcf=report.dcf,
        valuations=report.dcf.to_valuation_results(),
        not_applicable={"comps": "not run", "lbo": "not run", "monte_carlo": "not run"},
    )
    workbook = openpyxl.load_workbook(export_to_excel(bundle, tmp_path / "bundle.xlsx"))
    assert tuple(workbook.sheetnames) == SHEET_NAMES
    text = " ".join(
        str(c.value) for row in workbook["Assumptions"].iter_rows() for c in row if c.value
    )
    assert "no parameter sources" in text


# ---------------------------------------------------------------------- charts


def test_build_charts_returns_every_chart_for_a_full_report(atco_report) -> None:
    charts = build_charts(atco_report)
    assert tuple(charts) == CHART_NAMES
    assert len(CHART_NAMES) == 16
    for name, figure in charts.items():
        assert isinstance(figure, Figure), name
        assert figure.axes, name
        width, height = figure.get_size_inches()
        assert width > 3 and height > 1.5
        figure.canvas.draw()  # renders without an exception


def test_build_charts_leaves_out_models_that_are_not_applicable(seb_report) -> None:
    charts = build_charts(seb_report)
    assert list(charts) == ["football_field", "comps_peer_multiples", "comps_silhouette"]
    for figure in charts.values():
        figure.canvas.draw()
    texts = " ".join(t.get_text() for t in charts["football_field"].texts)
    assert "not applicable" in texts


def test_save_charts_writes_png_and_svg(reports, tmp_path: Path) -> None:
    written = save_charts(reports["ABB.ST"], tmp_path / "charts")
    assert tuple(written) == CHART_NAMES
    for name, paths in written.items():
        assert [p.name for p in paths] == [f"{name}.png", f"{name}.svg"]
        assert paths[0].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert "<svg" in paths[1].read_text(encoding="utf-8")[:400]
    only_png = save_charts(reports["SEB-A.ST"], tmp_path / "png", formats=("png",))
    assert all(len(paths) == 1 for paths in only_png.values())


@pytest.mark.parametrize("ticker", ["VSURE.ST", "NIBE-B.ST", "ALIV-SDB.ST"])
def test_charts_render_for_awkward_companies(reports, ticker: str) -> None:
    """Negative values per share, thin margins, a USD reporter."""
    charts = build_charts(reports[ticker])
    assert tuple(charts) == CHART_NAMES
    for figure in charts.values():
        figure.canvas.draw()


def test_football_field_rows_and_the_trading_range(atco_report, atco) -> None:
    rows = report_valuations(atco_report)
    assert [row.method for row in rows] == [
        ValuationMethod.DCF_GORDON,
        ValuationMethod.DCF_EXIT_MULTIPLE,
        ValuationMethod.COMPS_EV_EBITDA,
        ValuationMethod.COMPS_EV_SALES,
        ValuationMethod.COMPS_PE,
        ValuationMethod.LBO,
        ValuationMethod.MONTE_CARLO,
        ValuationMethod.MARKET,
    ]
    market = rows[-1]
    closes = atco.price_history.prices["stock"].iloc[-52:]
    assert market.base == 192.25
    assert market.low == pytest.approx(min(closes.min(), 192.25))
    assert market.high == pytest.approx(max(closes.max(), 192.25))
    assert market.details["observations"] == 52
    assert len(atco_report.valuations) == 7  # the report itself is not modified
    # a report that already carries a market row is left alone
    bundle = ReportBundle(company=atco, valuations=[market])
    assert report_valuations(bundle) == [market]


def test_trading_range_edge_cases(atco) -> None:
    assert trading_range_result(None, 100.0) is None
    assert trading_range_result(atco.price_history, math.nan) is None
    assert trading_range_result(atco.price_history, 0.0) is None
    wide = trading_range_result(atco.price_history, 1_000.0, "SEK", weeks=10)
    assert wide.high == 1_000.0 and wide.label == "10-week trading range"
    assert wide.low == pytest.approx(atco.price_history.prices["stock"].iloc[-10:].min())


def test_football_field_handles_negative_empty_and_unordered_input() -> None:
    rows = [
        ValuationResult(ValuationMethod.MONTE_CARLO, "MC", -5.0, 2.0, 9.0),
        ValuationResult(ValuationMethod.DCF_GORDON, "DCF", -3.0, -1.0, 4.0),
        ValuationResult(ValuationMethod.LBO, "LBO", 0.0, 0.0, 0.0),
    ]
    assert [r.method for r in order_valuations(rows)] == [
        ValuationMethod.DCF_GORDON,
        ValuationMethod.LBO,
        ValuationMethod.MONTE_CARLO,
    ]
    figure = football_field(rows, 10.0, currency="SEK")
    figure.canvas.draw()
    assert "shown at 0" in " ".join(t.get_text() for t in figure.texts)
    for price in (math.nan, 0.0):
        football_field(rows, price).canvas.draw()
    empty = football_field([], 10.0)
    empty.canvas.draw()
    assert "No valuation method is applicable" in " ".join(
        t.get_text() for ax in empty.axes for t in ax.texts
    ) + " ".join(t.get_text() for t in empty.texts)


def test_report_without_any_model_still_gives_a_football_field() -> None:
    company = make_company()  # no price history: no trading range either
    bundle = ReportBundle(company=company, not_applicable={"dcf": "nothing to show"})
    charts = build_charts(bundle)
    assert list(charts) == ["football_field"]
    charts["football_field"].canvas.draw()


def test_save_figure_and_message_figure(tmp_path: Path) -> None:
    figure = message_figure("Title", "Nothing to show")
    paths = save_figure(figure, tmp_path / "deep" / "note", ("png",))
    assert [p.name for p in paths] == ["note.png"]
    assert paths[0].stat().st_size > 500


def test_monte_carlo_charts_accept_runs_with_invalid_simulations(atco) -> None:
    from valuation_engine.models import dcf, monte_carlo
    from valuation_engine.viz import mc_convergence, mc_histogram, mc_tornado

    base = dcf.run_dcf(atco)
    config = monte_carlo.MonteCarloConfig(
        n_simulations=2_000, wacc=monte_carlo.DistributionSpec("normal", std=0.04)
    )
    result = monte_carlo.run_monte_carlo(atco, config=config, dcf_result=base)
    assert result.n_non_finite > 0 and np.isnan(result.values).any()
    for chart in (mc_histogram, mc_tornado, mc_convergence):
        chart(result).canvas.draw()
    report = pipeline.ValuationReport(company=atco, dcf=base, monte_carlo=result)
    assert {"mc_histogram", "dcf_bridge_gordon"} <= set(build_charts(report))


def test_data_sheet_does_not_present_the_spot_rate_as_the_statement_conversion_rate(
    reports, tmp_path: Path
) -> None:
    report = reports["ABB.ST"]
    workbook = openpyxl.load_workbook(export_to_excel(report, tmp_path / "abb.xlsx"))
    ws = workbook["Data"]
    spot = report.company.profile.fx_rate
    revenue_sek_m = ws.cell(row=find_row(ws, "Revenue"), column=6).value  # 2025A
    assert revenue_sek_m == pytest.approx(33_220.0 * 9.8191, rel=1e-4)
    assert abs(revenue_sek_m / 33_220.0 - spot) > 0.2
    assert f"converted from USD at {spot:.4f}" not in ws["B2"].value
    everything = " ".join(str(c.value) for row in ws.iter_rows() for c in row if c.value)
    assert f"converted from USD at {spot:.4f}" not in everything
    assert "year average (FY2025: 9.8191)" in everything
    assert "year-end rate (FY2025: 9.2013)" in everything
    # the per-year rates are tabulated and reproduce the statement figures
    sources = report.company.statements.source_map
    header = find_row(ws, "FX rates applied (USD to SEK)")
    assert [ws.cell(row=header, column=c).value for c in range(3, 7)] == [
        "2022A",
        "2023A",
        "2024A",
        "2025A",
    ]
    averages = [ws.cell(row=header + 1, column=c).value for c in range(3, 7)]
    year_ends = [ws.cell(row=header + 2, column=c).value for c in range(3, 7)]
    assert averages == pytest.approx(
        [float(sources[f"fx.average.{y}"]) for y in range(2022, 2026)], rel=1e-12
    )
    assert year_ends == pytest.approx(
        [float(sources[f"fx.year_end.{y}"].split()[0]) for y in range(2022, 2026)], rel=1e-12
    )
    assert revenue_sek_m == pytest.approx(33_220.0 * averages[-1])
    summary = workbook["Summary"]
    assert value_at(summary, "Spot FX rate, market snapshot") == spot
    # a SEK reporter gets no FX note at all


def test_fix_round_cosmetics_in_the_workbook(atco_workbook, atco_report) -> None:
    text = {
        sheet.title: " ".join(str(c.value) for row in sheet.iter_rows() for c in row if c.value)
        for sheet in atco_workbook
    }
    assert "FX rates applied" not in text["Data"] and "Spot FX rate" in text["Summary"]
    # Monte Carlo distributions are a table, not a repr
    ws = atco_workbook["Assumptions"]
    assert "DistributionSpec(" not in text["Assumptions"]
    header = find_row(ws, "Simulated driver")
    assert [ws.cell(row=header, column=c).value for c in range(3, 8)] == [
        "Distribution",
        "Mean / mode",
        "Std",
        "Low",
        "High",
    ]
    spec = atco_report.monte_carlo.config.wacc
    row = find_row(ws, "WACC", header)
    assert [ws.cell(row=row, column=c).value for c in range(3, 8)] == [
        "truncnormal",
        spec.mean,
        spec.std,
        spec.low,
        spec.high,
    ]
    # the LBO sheet says where its exit multiple comes from
    lbo_sheet = atco_workbook["LBO"]
    row = find_row(lbo_sheet, "Exit EV/EBITDA", find_row(lbo_sheet, "Exit and returns"))
    assert "exit multiple of the DCF" in lbo_sheet.cell(row=row, column=4).value
    assert "comps median EV/EBITDA (11 peers)" in lbo_sheet.cell(row=row, column=4).value
    # attribution shares: amount / |total|
    start = find_row(lbo_sheet, "Returns attribution")
    assert lbo_sheet.cell(row=start + 1, column=4).value == "Share of |total|"
    attribution = atco_report.lbo.attribution
    row = find_row(lbo_sheet, "Multiple expansion", start)
    assert lbo_sheet.cell(row=row, column=4).value == pytest.approx(
        attribution.loc["multiple_expansion", "share"]
    )
    assert attribution.loc["multiple_expansion", "share"] < 0  # exit below entry multiple


def test_lazy_viz_package_behaves_like_an_ordinary_one() -> None:
    import subprocess
    import sys

    code = """
import sys
import valuation_engine.viz as viz
import valuation_engine.viz.protocols
import valuation_engine.pipeline, valuation_engine.batch, valuation_engine.models.dcf
assert EAGER or "matplotlib" not in sys.modules, "numbers only: no chart stack"
assert viz.METHOD_KEYS == ("dcf", "comps", "lbo", "monte_carlo")
assert EAGER or "matplotlib" not in sys.modules
assert set(viz.__all__) <= set(dir(viz))
try:
    viz.no_such_name
except AttributeError as error:
    assert "no_such_name" in str(error)
else:
    raise AssertionError("unknown attribute did not raise")
import valuation_engine.viz.football_field  # the submodule shares the function's name
assert callable(viz.football_field) and viz.football_field.__name__ == "football_field"
from valuation_engine.viz import football_field, build_charts
assert football_field is viz.football_field and callable(build_charts)
assert viz.build_charts is build_charts
namespace = {}
exec("from valuation_engine.viz import *", namespace)
missing = [name for name in viz.__all__ if name not in namespace]
assert not missing, missing
assert namespace["football_field"] is viz.football_field
assert "matplotlib" in sys.modules
import pickle
bundle = pickle.loads(pickle.dumps(viz.ReportBundle(company=None)))
assert type(bundle) is viz.ReportBundle
"""
    # second pass: the submodule named like the function is imported before anything else
    eager = "EAGER = True\nimport valuation_engine.viz.football_field\n"
    for order in ("EAGER = False\n", eager):
        done = subprocess.run(
            [sys.executable, "-W", "error", "-c", order + code],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert done.returncode == 0, done.stderr[-600:]
