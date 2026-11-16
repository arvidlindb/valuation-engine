"""Excel export of a valuation report, formatted like a bank model.

``export_to_excel(report, path)`` writes one workbook with the sheets Summary,
Assumptions, DCF, Comps, LBO, Monte Carlo and Data. ``report`` is any object
with the attributes of :class:`valuation_engine.viz.protocols.ValuationReportLike`.

Conventions of the workbook
---------------------------
* It is an export of the Python model, not a live formula model: almost every
  number is a static value. A few cheap and robust relations are live formulas
  (market cap, net debt and EV on Summary, upside vs. price, the WACC build-up,
  the DCF bridge, totals and shares in sources and uses). Cells with formulas
  have no cached value until the file is opened in Excel.
* Blue font = hard-coded input or source data; black = calculated by the model.
* Amounts are in millions of the trading currency unless the label says
  otherwise; per-share values are in the currency; percentages and multiples
  are formatted as ``0.0%`` and ``0.0x``.
* A model that is not applicable keeps its sheet, with the reason on it.
"""

from __future__ import annotations

import dataclasses
import io
import math
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

from valuation_engine.core.types import MarketParameters, ValuationMethod
from valuation_engine.data.fx import describe_statement_fx
from valuation_engine.viz.protocols import ValuationReportLike, percentile
from valuation_engine.viz.report import report_football_field, report_valuations

__all__ = ["SHEET_NAMES", "export_to_excel"]

SHEET_NAMES: tuple[str, ...] = (
    "Summary",
    "Assumptions",
    "DCF",
    "Comps",
    "LBO",
    "Monte Carlo",
    "Data",
)

# --------------------------------------------------------------------------
# Styles
# --------------------------------------------------------------------------

_FONT_NAME = "Arial"
_FONT_SIZE = 10
_BLUE = "0000FF"  # hard-coded inputs
_BLACK = "000000"
_GREY = "595959"
_NAVY = "1F3864"
_LIGHT = "D9E1F2"
_HIGHLIGHT = "FFF2CC"

_FILL_SECTION = PatternFill("solid", fgColor=_NAVY)
_FILL_HEADER = PatternFill("solid", fgColor=_LIGHT)
_FILL_HIGHLIGHT = PatternFill("solid", fgColor=_HIGHLIGHT)
_THIN = Side(style="thin", color="7F7F7F")
_BORDER_BOTTOM = Border(bottom=_THIN)
_BORDER_TOP = Border(top=_THIN)
_BORDER_BOX = Border(left=_THIN, right=_THIN, top=_THIN, bottom=_THIN)

FMT_MILLIONS = '#,##0;(#,##0);"-"'
FMT_MILLIONS_1 = '#,##0.0;(#,##0.0);"-"'
FMT_PER_SHARE = "#,##0.00;(#,##0.00)"
FMT_PCT = "0.0%;(0.0%)"
FMT_PCT_2 = "0.00%;(0.00%)"
FMT_MULTIPLE = '0.0"x";(0.0"x")'
FMT_MULTIPLE_2 = '0.00"x";(0.00"x")'
FMT_INT = "#,##0"
FMT_FACTOR = "0.000"
FMT_DECIMAL = "0.00"
FMT_DATE = "yyyy-mm-dd"

_MILLION = 1e6

_ACRONYMS: dict[str, str] = {
    "ebitda": "EBITDA",
    "ebit": "EBIT",
    "nwc": "NWC",
    "fcff": "FCFF",
    "pv": "PV",
    "nopat": "NOPAT",
    "ppe": "PP&E",
    "pik": "PIK",
    "ev": "EV",
    "pe": "P/E",
    "irr": "IRR",
    "moic": "MOIC",
    "wacc": "WACC",
    "cogs": "COGS",
    "pct": "%",
    "roe": "ROE",
}
_LABEL_OVERRIDES: dict[str, str] = {
    "depreciation_amortization": "Depreciation & amortization",
    "delta_nwc": "Increase in NWC",
    "change_in_nwc": "Increase in NWC",
    "pv_fcff": "PV of FCFF",
    "goodwill_intangibles": "Goodwill & intangibles",
    "ev_ebitda": "EV/EBITDA",
    "ev_sales": "EV/Sales",
    "pe": "P/E",
    "da_pct_revenue": "D&A, % of revenue",
    "capex_pct_revenue": "Capex, % of revenue",
    "nwc_pct_revenue": "Operating NWC, % of revenue",
    "min_cash_pct_revenue": "Minimum cash, % of revenue",
    "interest_deduction_cap_pct_ebitda": "Interest deduction cap, % of EBITDA",
    "revolver_limit_x_ebitda": "Revolver limit, x EBITDA",
    "net_debt_to_ebitda": "Net debt / EBITDA",
    "cash_flow_available_for_debt_service": "Cash flow available for debt service",
    "diluted_shares": "Diluted shares (millions)",
}


def _label(name: Any) -> str:
    """``net_working_capital`` -> ``Net working capital``; acronyms upper-cased."""
    key = str(name)
    if key in _LABEL_OVERRIDES:
        return _LABEL_OVERRIDES[key]
    words = [_ACRONYMS.get(word, word) for word in key.split("_")]
    text = " ".join(words)
    return text[:1].upper() + text[1:]


def _clean(value: Any) -> Any:
    """Convert ``value`` to something openpyxl can store; missing -> None."""
    if value is None:
        return None
    if isinstance(value, bool | np.bool_):
        return "Yes" if bool(value) else "No"
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, datetime | date | str):
        return value
    if isinstance(value, list | tuple | set | frozenset):
        return ", ".join(str(v) for v in value)
    return str(value)


def _scaled(value: Any, divisor: float) -> Any:
    cleaned = _clean(value)
    if isinstance(cleaned, float | int) and not isinstance(cleaned, bool):
        return cleaned / divisor
    return cleaned


def _font(
    *, colour: str = _BLACK, bold: bool = False, italic: bool = False, size: float = _FONT_SIZE
) -> Font:
    return Font(name=_FONT_NAME, size=size, bold=bold, italic=italic, color=colour)


# --------------------------------------------------------------------------
# Sheet writer
# --------------------------------------------------------------------------


class _Sheet:
    """Cursor-based writer that keeps the formatting of all sheets identical.

    Column A is a narrow margin, column B holds labels, values start in C.
    """

    LABEL_COL = 2
    VALUE_COL = 3

    def __init__(self, ws: Worksheet, title: str, subtitle: str, *, span: int = 9) -> None:
        self.ws = ws
        self.span = span
        self.row = 5
        self.max_col = self.LABEL_COL + span - 1
        ws.sheet_view.showGridLines = False
        ws.sheet_view.zoomScale = 90
        ws.column_dimensions["A"].width = 2
        ws.column_dimensions["B"].width = 40
        ws["B1"] = title
        ws["B1"].font = _font(bold=True, size=14, colour=_NAVY)
        ws["B2"] = subtitle
        ws["B2"].font = _font(italic=True, colour=_GREY)
        ws.freeze_panes = "C4"
        ws.page_setup.orientation = "landscape"
        ws.page_setup.fitToWidth = 1
        ws.page_setup.fitToHeight = 0
        ws.sheet_properties.pageSetUpPr.fitToPage = True

    # -- primitives ---------------------------------------------------------

    def cell(
        self,
        row: int,
        col: int,
        value: Any,
        *,
        fmt: str | None = None,
        kind: str = "calc",
        bold: bool = False,
        italic: bool = False,
        align: str | None = None,
        fill: PatternFill | None = None,
        border: Border | None = None,
    ) -> Any:
        """Write one cell. ``kind``: ``input`` (blue), ``calc`` or ``text`` (black)."""
        cell = self.ws.cell(row=row, column=col, value=value)
        cell.font = _font(colour=_BLUE if kind == "input" else _BLACK, bold=bold, italic=italic)
        if fmt is not None and (not isinstance(value, str) or value.startswith("=")):
            cell.number_format = fmt
        if align is None and isinstance(value, str) and not value.startswith("=") and col > 2:
            align = "right" if fmt is not None else "left"
        if align is not None:
            cell.alignment = Alignment(horizontal=align, vertical="center")
        if fill is not None:
            cell.fill = fill
        if border is not None:
            cell.border = border
        self.max_col = max(self.max_col, col)
        return cell

    def blank(self, count: int = 1) -> None:
        self.row += count

    def section(self, text: str, span: int | None = None) -> None:
        """Dark band with the section name."""
        width = span or self.span
        for col in range(self.LABEL_COL, self.LABEL_COL + width):
            cell = self.ws.cell(row=self.row, column=col)
            cell.fill = _FILL_SECTION
        head = self.ws.cell(row=self.row, column=self.LABEL_COL, value=text)
        head.font = _font(colour="FFFFFF", bold=True)
        self.max_col = max(self.max_col, self.LABEL_COL + width - 1)
        self.row += 1

    def header(
        self, labels: Sequence[Any], *, start_col: int | None = None, fmt: str | None = None
    ) -> None:
        """Column header row (bold, light fill, bottom border)."""
        col0 = self.LABEL_COL if start_col is None else start_col
        for offset, label in enumerate(labels):
            value = _clean(label)
            cell = self.cell(
                self.row,
                col0 + offset,
                value,
                fmt=fmt,
                bold=True,
                fill=_FILL_HEADER,
                border=_BORDER_BOTTOM,
                align="left" if col0 + offset == self.LABEL_COL else "right",
            )
            if fmt is not None:
                cell.number_format = fmt
        self.row += 1

    def note(self, text: str, *, bold: bool = False) -> None:
        cell = self.cell(self.row, self.LABEL_COL, text, kind="text")
        cell.font = _font(colour=_BLACK if bold else _GREY, italic=not bold, bold=bold)
        self.row += 1

    def notes(self, title: str, lines: Sequence[str]) -> None:
        if not lines:
            return
        self.section(title)
        for line in lines:
            self.note(f"• {line}")
        self.blank()

    def kv(
        self,
        label: str,
        value: Any,
        fmt: str | None = None,
        *,
        kind: str = "calc",
        note: str | None = None,
        bold: bool = False,
    ) -> str:
        """Label in B, value in C (optional remark in D). Returns the value cell's
        coordinate so that formulas can refer to it."""
        self.cell(self.row, self.LABEL_COL, label, kind="text", bold=bold)
        value = value if isinstance(value, str) and value.startswith("=") else _clean(value)
        is_text = isinstance(value, str) and not value.startswith("=")
        cell = self.cell(
            self.row,
            self.VALUE_COL,
            value,
            fmt=fmt,
            kind=kind,
            bold=bold,
            align="left" if is_text else "right",
        )
        if note:
            remark = self.cell(self.row, self.VALUE_COL + 1, note, kind="text", italic=True)
            remark.font = _font(colour=_GREY, italic=True)
            remark.alignment = Alignment(horizontal="left")
        self.row += 1
        return cell.coordinate

    def line(
        self,
        label: str,
        values: Sequence[Any],
        fmt: str | None = None,
        *,
        kind: str = "calc",
        bold: bool = False,
        top_border: bool = False,
    ) -> int:
        """Label in B and one value per column from C. Returns the row number."""
        border = _BORDER_TOP if top_border else None
        self.cell(self.row, self.LABEL_COL, label, kind="text", bold=bold, border=border)
        for offset, value in enumerate(values):
            stored = value if isinstance(value, str) and value.startswith("=") else _clean(value)
            self.cell(
                self.row,
                self.VALUE_COL + offset,
                stored,
                fmt=fmt,
                kind=kind,
                bold=bold,
                border=border,
                align="right",
            )
        self.row += 1
        return self.row - 1

    # -- composite blocks ---------------------------------------------------

    def statement(
        self,
        frame: pd.DataFrame,
        *,
        kind: str = "calc",
        column_suffix: str = "",
        input_rows: Sequence[str] = (),
        bold_rows: Sequence[str] = (),
        corner: str = "",
    ) -> dict[str, int]:
        """Write ``frame`` transposed: one row per line item, one column per
        fiscal year. Returns line item -> row number."""
        self.header([corner, *[f"{column}{column_suffix}" for column in frame.index]])
        rows: dict[str, int] = {}
        for name in frame.columns:
            fmt, divisor = _row_format(str(name))
            values = [_scaled(v, divisor) for v in frame[name].tolist()]
            rows[str(name)] = self.line(
                _label(name),
                values,
                fmt,
                kind="input" if (kind == "input" or name in input_rows) else "calc",
                bold=name in bold_rows,
            )
        return rows

    def table(
        self,
        frame: pd.DataFrame,
        columns: Sequence[tuple[str, str, str | None, float]],
        *,
        index_header: str | None = None,
        index_format: Callable[[Any], Any] = str,
        kind: str = "calc",
    ) -> None:
        """Write ``frame`` row by row. ``columns`` holds (column name, header,
        number format, divisor); a missing column is written blank."""
        headers = [index_header] if index_header is not None else []
        headers += [header for _, header, _, _ in columns]
        self.header(headers)
        for index, record in frame.iterrows():
            col = self.LABEL_COL
            if index_header is not None:
                self.cell(self.row, col, _clean(index_format(index)), kind="text")
                col += 1
            for name, _, fmt, divisor in columns:
                value = _scaled(record.get(name), divisor) if name in frame.columns else None
                self.cell(
                    self.row,
                    col,
                    value,
                    fmt=fmt,
                    kind=kind,
                    align="left" if fmt is None and col == self.LABEL_COL else None,
                )
                col += 1
            self.row += 1

    def grid(
        self,
        frame: pd.DataFrame,
        *,
        corner: str,
        row_fmt: str,
        column_fmt: str,
        cell_fmt: str,
        base: tuple[int | None, int | None] | None = None,
    ) -> None:
        """Two-way sensitivity table; the base cell is highlighted."""
        self.cell(
            self.row,
            self.LABEL_COL,
            corner,
            bold=True,
            fill=_FILL_HEADER,
            border=_BORDER_BOTTOM,
            kind="text",
        )
        for offset, column in enumerate(frame.columns):
            self.cell(
                self.row,
                self.VALUE_COL + offset,
                _clean(column),
                fmt=column_fmt,
                bold=True,
                fill=_FILL_HEADER,
                border=_BORDER_BOTTOM,
                kind="input",
                align="right",
            )
        self.row += 1
        for i, (index, record) in enumerate(frame.iterrows()):
            self.cell(
                self.row,
                self.LABEL_COL,
                _clean(index),
                fmt=row_fmt,
                bold=True,
                kind="input",
                align="right",
            )
            for j, value in enumerate(record.tolist()):
                cleaned = _clean(value)
                is_base = base is not None and base == (i, j)
                self.cell(
                    self.row,
                    self.VALUE_COL + j,
                    "n/a" if cleaned is None else cleaned,
                    fmt=cell_fmt,
                    bold=is_base,
                    fill=_FILL_HIGHLIGHT if is_base else None,
                    border=_BORDER_BOX if is_base else None,
                    align="right",
                )
            self.row += 1

    def finish(
        self, widths: Mapping[int, float] | None = None, default_width: float = 14.0
    ) -> None:
        """Set column widths (values columns share ``default_width``)."""
        for col in range(self.VALUE_COL, self.max_col + 1):
            self.ws.column_dimensions[get_column_letter(col)].width = default_width
        for col, width in (widths or {}).items():
            self.ws.column_dimensions[get_column_letter(col)].width = width


_PCT_ROWS = frozenset({"revenue_growth", "ebitda_margin"})
_MULTIPLE_ROWS = frozenset({"net_debt_to_ebitda", "interest_coverage"})


def _row_format(name: str) -> tuple[str | None, float]:
    """Number format and divisor of a statement / forecast line item."""
    if name in _PCT_ROWS:
        return FMT_PCT, 1.0
    if name in _MULTIPLE_ROWS:
        return FMT_MULTIPLE, 1.0
    if name == "discount_period":
        return FMT_DECIMAL, 1.0
    if name == "discount_factor":
        return FMT_FACTOR, 1.0
    if name == "liquidity_breach":
        return None, 1.0
    return FMT_MILLIONS, _MILLION


def _nearest(values: Any, target: float) -> int | None:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not math.isfinite(target):
        return None
    return int(np.nanargmin(np.abs(array - target)))


def _not_applicable(sheet: _Sheet, reason: str | None) -> None:
    sheet.note(f"Not applicable: {reason or 'no result available for this company'}", bold=True)
    sheet.finish()


# --------------------------------------------------------------------------
# Sheets
# --------------------------------------------------------------------------


def _summary_sheet(
    ws: Worksheet, report: ValuationReportLike, image: Figure | str | Path | None, generated: str
) -> None:
    company = report.company
    profile, market = company.profile, company.market
    currency = market.currency
    sheet = _Sheet(
        ws,
        f"{profile.name} ({profile.ticker}): valuation summary",
        f"{currency} millions unless stated otherwise · market data as of {market.as_of} "
        f"· generated {generated}",
        span=7,
    )
    sheet.note(
        "This workbook is an export of the Python valuation-engine model, not a live formula "
        "model: values are static unless a cell holds a formula."
    )
    sheet.note("Blue font = hard-coded input or source data. Black font = calculated value.")
    sheet.blank()

    sheet.section("Company")
    sheet.kv("Name", profile.name, kind="input")
    sheet.kv("Ticker (primary share class)", profile.ticker, kind="input")
    sheet.kv("Sector", profile.sector or "n/a", kind="input")
    sheet.kv("Industry", profile.industry or "n/a", kind="input")
    sheet.kv("Country", profile.country or "n/a", kind="input")
    sheet.kv("Trading currency", profile.currency, kind="input")
    sheet.kv("Reporting currency", profile.reporting_currency, kind="input")
    sheet.kv(
        "Spot FX rate, market snapshot (reporting to trading)",
        profile.fx_rate,
        "0.0000",
        kind="input",
    )
    fx_text = describe_statement_fx(company)
    if fx_text:
        sheet.note(fx_text)
    sheet.kv("Latest fiscal year", company.statements.latest_year, "0", kind="input")
    sheet.blank()

    sheet.section("Market data")
    price = sheet.kv(f"Share price ({currency})", market.price, FMT_PER_SHARE, kind="input")
    shares = sheet.kv(
        "Shares outstanding (millions, all classes)",
        market.shares_outstanding / _MILLION,
        FMT_MILLIONS_1,
        kind="input",
        note="market cap / price",
    )
    cap = sheet.kv("Market capitalization", f"={price}*{shares}", FMT_MILLIONS)
    debt = sheet.kv("Total debt", market.total_debt / _MILLION, FMT_MILLIONS, kind="input")
    cash = sheet.kv("Cash and equivalents", market.cash / _MILLION, FMT_MILLIONS, kind="input")
    net_debt = sheet.kv("Net debt", f"={debt}-{cash}", FMT_MILLIONS)
    minority = sheet.kv(
        "Minority interest", market.minority_interest / _MILLION, FMT_MILLIONS, kind="input"
    )
    preferred = sheet.kv(
        "Preferred equity", market.preferred_equity / _MILLION, FMT_MILLIONS, kind="input"
    )
    sheet.kv(
        "Enterprise value", f"={cap}+{net_debt}+{minority}+{preferred}", FMT_MILLIONS, bold=True
    )
    if market.beta is not None:
        sheet.kv("Beta (data provider, informational)", market.beta, FMT_DECIMAL, kind="input")
    sheet.blank()

    sheet.section("Valuation summary")
    sheet.header(
        [
            "Method",
            f"Low ({currency}/share)",
            f"Base ({currency}/share)",
            f"High ({currency}/share)",
            "Base vs. price",
            "Implied EV",
            "Implied equity value",
        ]
    )
    rows = report_valuations(report)
    if not rows:
        sheet.note("No valuation method is applicable to this company.")
    for result in rows:
        row = sheet.row
        sheet.cell(row, 2, result.label, kind="text")
        is_market = result.method == ValuationMethod.MARKET
        for offset, value in enumerate((result.low, result.base, result.high)):
            sheet.cell(
                row,
                3 + offset,
                _clean(value),
                fmt=FMT_PER_SHARE,
                kind="input" if is_market else "calc",
            )
        base_cell = ws.cell(row=row, column=4).coordinate
        absolute_price = f"${price[0]}${price[1:]}"
        sheet.cell(row, 6, f"={base_cell}/{absolute_price}-1", fmt=FMT_PCT)
        sheet.cell(row, 7, _scaled(result.enterprise_value, _MILLION), fmt=FMT_MILLIONS)
        sheet.cell(row, 8, _scaled(result.equity_value, _MILLION), fmt=FMT_MILLIONS)
        sheet.row += 1
    sheet.note(
        "Ranges: DCF = sensitivity cells (WACC ±1.0pp with growth ±0.5pp or exit multiple "
        "±1.0x); comps = 25th percentile, median, 75th percentile peer multiple; LBO = price "
        "giving the sponsor 25% / 20% / 15% IRR; Monte Carlo = simulated distribution; trading "
        "range = weekly closes."
    )
    sheet.blank()

    sheet.section("Methods not applicable")
    if report.not_applicable:
        sheet.header(["Method", "Reason"])
        for name, reason in report.not_applicable.items():
            sheet.cell(sheet.row, 2, _method_name(name), kind="text")
            sheet.cell(sheet.row, 3, str(reason), kind="text", align="left")
            sheet.row += 1
    else:
        sheet.note("All methods are applicable.")
    warnings = [str(w) for w in (getattr(report, "warnings", None) or [])]
    if warnings:
        sheet.blank()
        sheet.notes("Warnings", warnings)

    sheet.finish(default_width=17.0)
    if image is not None:
        _embed_image(ws, image, "J4")


def _method_name(key: str) -> str:
    return {"dcf": "DCF", "comps": "Trading comps", "lbo": "LBO", "monte_carlo": "Monte Carlo"}.get(
        key, _label(key)
    )


def _embed_image(
    ws: Worksheet, image: Figure | str | Path, anchor: str, width_px: int = 760
) -> None:
    """Place a PNG (path or matplotlib figure) on the sheet at ``anchor``."""
    if isinstance(image, Figure):
        buffer = io.BytesIO()
        image.savefig(buffer, format="png", dpi=200, facecolor="white")
        buffer.seek(0)
        picture = XLImage(buffer)
    else:
        picture = XLImage(str(image))
    ratio = picture.height / picture.width if picture.width else 0.5
    picture.width = width_px
    picture.height = int(width_px * ratio)
    ws.add_image(picture, anchor)


def _dataclass_rows(obj: Any) -> list[tuple[str, Any]]:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return [(f.name, getattr(obj, f.name)) for f in dataclasses.fields(obj)]
    if isinstance(obj, Mapping):
        return [(str(k), v) for k, v in obj.items()]
    return []


_ASSUMPTION_FORMATS: dict[str, str] = {
    "senior_debt_multiple": FMT_MULTIPLE,
    "subordinated_debt_multiple": FMT_MULTIPLE,
    "revolver_limit_x_ebitda": FMT_MULTIPLE,
    "min_interest_coverage": FMT_MULTIPLE,
    "min_debt_service_coverage": FMT_MULTIPLE_2,
    "exit_multiple": FMT_MULTIPLE,
    "entry_price_per_share": FMT_PER_SHARE,
}


def _assumption_kv(sheet: _Sheet, name: str, value: Any, note: str | None = None) -> None:
    """Write a scalar assumption with a format guessed from its name and type."""
    if value is None:
        sheet.kv(_label(name), "n/a (model default)", kind="input", note=note)
    elif isinstance(value, bool | str):
        sheet.kv(_label(name), value, kind="input", note=note)
    elif isinstance(value, int | np.integer):
        sheet.kv(_label(name), int(value), FMT_INT, kind="input", note=note)
    elif isinstance(value, float | np.floating):
        fmt = _ASSUMPTION_FORMATS.get(name, FMT_PCT_2 if abs(float(value)) < 5 else FMT_DECIMAL)
        sheet.kv(_label(name), float(value), fmt, kind="input", note=note)
    else:
        sheet.kv(_label(name), _clean(value), kind="input", note=note)


#: Market parameter -> (label, number format); order of the Assumptions sheet.
_MARKET_PARAMETERS: tuple[tuple[str, str, str | None], ...] = (
    ("risk_free_rate", "Risk-free rate", FMT_PCT_2),
    ("equity_risk_premium", "Equity risk premium", FMT_PCT_2),
    ("country_risk_premium", "Country risk premium (included in the ERP)", FMT_PCT_2),
    ("tax_rate", "Corporate tax rate", FMT_PCT_2),
    ("long_term_growth", "Long-term growth", FMT_PCT_2),
    ("benchmark", "Benchmark index", None),
)


def _market_parameters_block(sheet: _Sheet, params: MarketParameters, sources: Any) -> None:
    """Market parameters with value, source, as-of date and origin.

    ``sources`` is the report's ``parameter_sources`` (name -> object with
    ``value``, ``source``, ``as_of`` and ``origin``); any object with those
    attributes works. Without it the values of ``params`` are listed and marked
    as carrying no provenance.
    """
    sources = sources if isinstance(sources, Mapping) else {}
    sheet.section("Market parameters (Sweden)")
    sheet.header(["Parameter", "Value", "Source", "As of", "Origin"])
    for name, label, fmt in _MARKET_PARAMETERS:
        source = sources.get(name)
        if hasattr(params, name):
            value = getattr(params, name)
        elif source is not None:
            value = getattr(source, "value", None)
        else:
            continue  # informational entry without a value in this report
        sheet.cell(sheet.row, sheet.LABEL_COL, label, kind="text")
        is_text = isinstance(value, str)
        sheet.cell(
            sheet.row,
            sheet.VALUE_COL,
            _clean(value),
            fmt=None if is_text else fmt,
            kind="input",
            align="left" if is_text else "right",
        )
        if source is not None:
            details = (
                getattr(source, "source", None) or "n/a",
                getattr(source, "as_of", None) or "n/a",
                getattr(source, "origin", None) or "n/a",
            )
        else:
            details = ("no provenance recorded in this report", "n/a", "n/a")
        for offset, text in enumerate(details, start=1):
            sheet.cell(sheet.row, sheet.VALUE_COL + offset, str(text), kind="text", align="left")
        sheet.row += 1
    if not sources:
        sheet.note(
            "The report carries no parameter sources; values are those passed to the models."
        )
    sheet.blank()


def _is_distribution(value: Any) -> bool:
    """True for a Monte Carlo distribution spec (duck-typed: kind, mean, std, low, high)."""
    return all(hasattr(value, name) for name in ("kind", "mean", "std", "low", "high"))


def _assumptions_sheet(ws: Worksheet, report: ValuationReportLike) -> None:
    company = report.company
    sheet = _Sheet(
        ws,
        f"{company.profile.name}: assumptions",
        "All inputs of the models (blue). Rates and margins are shown as percentages.",
    )
    params = getattr(report, "params", None) or MarketParameters()
    _market_parameters_block(sheet, params, getattr(report, "parameter_sources", None))

    sheet.section("DCF assumptions")
    dcf = report.dcf
    if dcf is None:
        sheet.note(
            f"Not applicable: {report.not_applicable.get('dcf', 'no DCF result')}", bold=True
        )
    else:
        a = dcf.assumptions
        years = [f"{int(y)}E" for y in dcf.forecast.index]
        sheet.header(["Per forecast year", *years])
        sheet.line("Revenue growth", list(a.revenue_growth), FMT_PCT, kind="input")
        sheet.line("EBITDA margin", list(a.ebitda_margin), FMT_PCT, kind="input")
        sheet.blank()
        sheet.kv("D&A, % of revenue", a.da_pct_revenue, FMT_PCT_2, kind="input")
        sheet.kv("Capex, % of revenue", a.capex_pct_revenue, FMT_PCT_2, kind="input")
        sheet.kv("Operating NWC, % of revenue", a.nwc_pct_revenue, FMT_PCT_2, kind="input")
        sheet.kv("Tax rate", a.tax_rate, FMT_PCT_2, kind="input")
        sheet.kv(
            "WACC",
            dcf.wacc,
            FMT_PCT_2,
            kind="input" if dcf.wacc_result is None else "calc",
            note="given as assumption" if dcf.wacc_result is None else "computed, see DCF sheet",
        )
        sheet.kv("Terminal growth", a.terminal_growth, FMT_PCT_2, kind="input")
        sheet.kv(
            "Exit EV/EBITDA multiple",
            dcf.exit_multiple,
            FMT_MULTIPLE,
            kind="input" if dcf.exit_multiple_source == "assumption" else "calc",
            note=dcf.exit_multiple_source,
        )
        sheet.kv("Mid-year convention", a.mid_year, kind="input")
        sheet.kv("Forecast years", a.forecast_years, FMT_INT, kind="input")
    sheet.blank()

    sheet.section("LBO assumptions")
    lbo = report.lbo
    if lbo is None:
        sheet.note(
            f"Not applicable: {report.not_applicable.get('lbo', 'no LBO result')}", bold=True
        )
    else:
        a_lbo = lbo.assumptions
        years = [f"{int(y)}E" for y in lbo.forecast.index]
        sheet.header(["Per year of the holding period", *years])
        sheet.line("Revenue growth", list(a_lbo.revenue_growth), FMT_PCT, kind="input")
        sheet.line("EBITDA margin", list(a_lbo.ebitda_margin), FMT_PCT, kind="input")
        sheet.blank()
        for name, value in _dataclass_rows(a_lbo):
            if name in ("revenue_growth", "ebitda_margin"):
                continue
            note = None
            if name == "exit_multiple" and value is None:
                note = "exit multiple = entry multiple of the deal"
            if name == "entry_price_per_share" and value is None:
                note = "market price plus takeover premium"
            _assumption_kv(sheet, name, value, note)
    sheet.blank()

    sheet.section("Monte Carlo configuration")
    monte_carlo = report.monte_carlo
    if monte_carlo is None:
        sheet.note(
            f"Not applicable: {report.not_applicable.get('monte_carlo', 'no Monte Carlo result')}",
            bold=True,
        )
    else:
        sheet.kv("Simulations", int(monte_carlo.n_simulations), FMT_INT, kind="input")
        rows = _dataclass_rows(getattr(monte_carlo, "config", None))
        specs = [(name, value) for name, value in rows if _is_distribution(value)]
        for name, value in rows:
            if name in ("n_simulations",) or _is_distribution(value):
                continue
            if isinstance(value, Mapping):
                for key, inner in value.items():
                    _assumption_kv(sheet, f"{name}: {key}", inner)
            else:
                _assumption_kv(sheet, name, value)
        if not rows and getattr(monte_carlo, "config", None) is not None:
            sheet.kv("Configuration", str(monte_carlo.config), kind="input")
        if specs:
            sheet.blank()
            sheet.header(["Simulated driver", "Distribution", "Mean / mode", "Std", "Low", "High"])
            for name, spec in specs:
                sheet.cell(sheet.row, sheet.LABEL_COL, _label(name), kind="text")
                sheet.cell(
                    sheet.row, sheet.LABEL_COL + 1, str(spec.kind), kind="input", align="left"
                )
                for offset, attribute in enumerate(("mean", "std", "low", "high"), start=2):
                    number = getattr(spec, attribute, None)
                    sheet.cell(
                        sheet.row,
                        sheet.LABEL_COL + offset,
                        _clean(number) if number is not None else "n/a",
                        fmt=FMT_PCT_2,
                        kind="input",
                        align="right",
                    )
                sheet.row += 1
            absolute = any(getattr(spec, "absolute", False) for _, spec in specs)
            sheet.note(
                "Levels of the driver."
                if absolute
                else "Offsets from the base case in percentage points (0.00% = the base value); "
                "mean is the mode for a triangular distribution, std is before truncation."
            )
    sheet.finish(widths={4: 16})


def _dcf_sheet(ws: Worksheet, report: ValuationReportLike) -> None:
    company = report.company
    currency = company.market.currency
    sheet = _Sheet(
        ws,
        f"{company.profile.name}: discounted cash flow (FCFF)",
        f"{currency} millions unless stated otherwise",
    )
    dcf = report.dcf
    if dcf is None:
        _not_applicable(sheet, report.not_applicable.get("dcf"))
        return

    sheet.section("Forecast")
    rows = sheet.statement(
        dcf.forecast.drop(columns=[c for c in ("fiscal_year",) if c in dcf.forecast.columns]),
        column_suffix="E",
        input_rows=("revenue_growth", "ebitda_margin"),
        bold_rows=("fcff",),
        corner=f"{currency} m",
    )
    first = get_column_letter(_Sheet.VALUE_COL)
    last = get_column_letter(_Sheet.VALUE_COL + len(dcf.forecast.index) - 1)
    pv_row = rows.get("pv_fcff")
    sheet.blank()

    sheet.section("WACC build-up")
    w = dcf.wacc_result
    if w is None:
        sheet.kv(
            "WACC", dcf.wacc, FMT_PCT_2, kind="input", note="given as assumption; CAPM not run"
        )
    else:
        rf = sheet.kv("Risk-free rate", w.risk_free_rate, FMT_PCT_2, kind="input")
        erp = sheet.kv("Equity risk premium", w.equity_risk_premium, FMT_PCT_2, kind="input")
        sheet.kv(
            "Raw regression beta",
            w.raw_beta,
            FMT_DECIMAL,
            note=f"vs. {w.benchmark}, {w.beta_observations} weekly returns",
        )
        sheet.kv("R squared of the regression", w.raw_beta_r_squared, FMT_DECIMAL)
        sheet.kv("Blume-adjusted own beta", w.blume_beta, FMT_DECIMAL, note="2/3 x raw + 1/3")
        sheet.kv(
            "Peer unlevered beta (median)",
            w.peer_unlevered_beta,
            FMT_DECIMAL,
            note=f"{w.peer_count} peers, {w.peer_group or 'n/a'}",
        )
        sheet.kv("Peer beta relevered at target D/E", w.peer_relevered_beta, FMT_DECIMAL)
        sheet.kv("Peer beta, Blume-adjusted", w.peer_beta, FMT_DECIMAL)
        beta = sheet.kv("Beta used", w.beta, FMT_DECIMAL, note=w.beta_method, bold=True)
        size = sheet.kv(
            "Size premium", w.size_premium, FMT_PCT_2, kind="input", note=w.size_premium_basis
        )
        ke = sheet.kv(
            "Cost of equity",
            f"={rf}+{beta}*{erp}+{size}",
            FMT_PCT_2,
            note="risk-free + beta x ERP + size premium",
        )
        kd = sheet.kv(
            "Pre-tax cost of debt", w.pre_tax_cost_of_debt, FMT_PCT_2, note=w.cost_of_debt_source
        )
        tax = sheet.kv("Tax rate", w.tax_rate, FMT_PCT_2, kind="input")
        kd_net = sheet.kv("After-tax cost of debt", f"={kd}*(1-{tax})", FMT_PCT_2)
        sheet.kv("Market capitalization", w.market_cap / _MILLION, FMT_MILLIONS, kind="input")
        sheet.kv("Total debt", w.total_debt / _MILLION, FMT_MILLIONS, kind="input")
        sheet.kv("Debt / equity used", w.debt_to_equity, FMT_DECIMAL)
        we = sheet.kv("Equity weight", w.equity_weight, FMT_PCT)
        wd = sheet.kv("Debt weight", w.debt_weight, FMT_PCT)
        sheet.kv(
            "WACC",
            f"={we}*{ke}+{wd}*{kd_net}",
            FMT_PCT_2,
            bold=True,
            note=f"model value {w.wacc:.4%}",
        )
    sheet.blank()

    sheet.section("Terminal value")
    sheet.header(["", "Gordon growth", "Exit multiple"])
    last_year = int(dcf.forecast.index[-1])
    last_ebitda = float(dcf.forecast["ebitda"].iloc[-1])
    sheet.line(
        f"Normalized FCFF, year {last_year + 1}", [dcf.terminal_fcff / _MILLION, None], FMT_MILLIONS
    )
    sheet.line(
        "Terminal growth",
        [dcf.assumptions.terminal_growth, dcf.implied_perpetual_growth],
        FMT_PCT_2,
    )
    sheet.line(
        f"EBITDA, year {last_year}", [last_ebitda / _MILLION, last_ebitda / _MILLION], FMT_MILLIONS
    )
    sheet.line(
        "EV/EBITDA multiple at exit", [dcf.implied_exit_multiple, dcf.exit_multiple], FMT_MULTIPLE
    )
    sheet.line(
        f"Terminal value at end of {last_year}",
        [dcf.terminal_value_gordon / _MILLION, dcf.terminal_value_exit / _MILLION],
        FMT_MILLIONS,
    )
    sheet.line("Terminal value, share of EV", [dcf.tv_share_gordon, dcf.tv_share_exit], FMT_PCT)
    sheet.note(
        "Gordon column: growth is the assumption, the multiple is implied. Exit column: the "
        f"multiple is the assumption ({dcf.exit_multiple_source}), the growth is implied."
    )
    sheet.blank()

    sheet.section("Enterprise value to equity value")
    sheet.header(["", "Gordon growth", "Exit multiple"])
    columns = (get_column_letter(_Sheet.VALUE_COL), get_column_letter(_Sheet.VALUE_COL + 1))
    pv_formula = f"=SUM({first}{pv_row}:{last}{pv_row})" if pv_row else None
    r_pv = sheet.line(
        "PV of explicit FCFF",
        [pv_formula or dcf.pv_fcff / _MILLION] * 2,
        FMT_MILLIONS,
    )
    r_tv = sheet.line(
        "PV of terminal value",
        [dcf.pv_terminal_value_gordon / _MILLION, dcf.pv_terminal_value_exit / _MILLION],
        FMT_MILLIONS,
    )
    r_ev = sheet.line(
        "Enterprise value",
        [f"={c}{r_pv}+{c}{r_tv}" for c in columns],
        FMT_MILLIONS,
        bold=True,
        top_border=True,
    )
    r_debt = sheet.line("Less: net debt", [dcf.net_debt / _MILLION] * 2, FMT_MILLIONS, kind="input")
    r_min = sheet.line(
        "Less: minority interest",
        [dcf.minority_interest / _MILLION] * 2,
        FMT_MILLIONS,
        kind="input",
    )
    r_pref = sheet.line(
        "Less: preferred equity", [dcf.preferred_equity / _MILLION] * 2, FMT_MILLIONS, kind="input"
    )
    r_eq = sheet.line(
        "Equity value",
        [f"={c}{r_ev}-{c}{r_debt}-{c}{r_min}-{c}{r_pref}" for c in columns],
        FMT_MILLIONS,
        bold=True,
        top_border=True,
    )
    r_shares = sheet.line(
        "Shares outstanding (millions)",
        [dcf.shares_outstanding / _MILLION] * 2,
        FMT_MILLIONS_1,
        kind="input",
    )
    r_value = sheet.line(
        f"Value per share ({currency})",
        [f"={c}{r_eq}/{c}{r_shares}" for c in columns],
        FMT_PER_SHARE,
        bold=True,
    )
    r_price = sheet.line(
        f"Current share price ({currency})", [dcf.price] * 2, FMT_PER_SHARE, kind="input"
    )
    sheet.line(
        "Upside / (downside) vs. price", [f"={c}{r_value}/{c}{r_price}-1" for c in columns], FMT_PCT
    )
    sheet.note(
        f"Model values per share: Gordon {dcf.value_per_share_gordon:,.2f}, "
        f"exit multiple {dcf.value_per_share_exit:,.2f}."
    )
    sheet.blank()

    sheet.section(f"Sensitivity: value per share ({currency}), WACC x terminal growth")
    grid = dcf.sensitivity_gordon
    sheet.grid(
        grid,
        corner="WACC \\ terminal growth",
        row_fmt=FMT_PCT_2,
        column_fmt=FMT_PCT_2,
        cell_fmt=FMT_PER_SHARE,
        base=(
            _nearest(grid.index, dcf.wacc),
            _nearest(grid.columns, dcf.assumptions.terminal_growth),
        ),
    )
    sheet.blank()
    sheet.section(f"Sensitivity: value per share ({currency}), WACC x exit multiple")
    grid = dcf.sensitivity_exit
    sheet.grid(
        grid,
        corner="WACC \\ exit multiple",
        row_fmt=FMT_PCT_2,
        column_fmt=FMT_MULTIPLE,
        cell_fmt=FMT_PER_SHARE,
        base=(_nearest(grid.index, dcf.wacc), _nearest(grid.columns, dcf.exit_multiple)),
    )
    sheet.blank()
    sheet.notes("Notes and fallbacks", [str(n) for n in dcf.notes])
    sheet.finish(widths={3: 16, 4: 16})


def _comps_sheet(ws: Worksheet, report: ValuationReportLike) -> None:
    company = report.company
    currency = company.market.currency
    sheet = _Sheet(
        ws,
        f"{company.profile.name}: trading comparables",
        f"Peer amounts in SEK millions; implied values in {currency} millions and {currency} "
        "per share",
        span=12,
    )
    comps = report.comps
    if comps is None:
        _not_applicable(sheet, report.not_applicable.get("comps"))
        return

    sheet.section("Peer selection")
    sheet.kv("Candidate scope", comps.candidate_scope.replace("_", " "))
    sheet.kv("Candidates clustered", len(comps.candidates), FMT_INT)
    sheet.kv("Features", ", ".join(_label(f) for f in comps.feature_names))
    sheet.kv("Clusters chosen (k)", comps.chosen_k, FMT_INT)
    sheet.kv("Peers selected", len(comps.peers), FMT_INT)
    if comps.silhouette_scores:
        ks = sorted(comps.silhouette_scores)
        sheet.header(["Silhouette score by k", *ks], fmt="0")
        sheet.line("Score", [comps.silhouette_scores[k] for k in ks], FMT_FACTOR)
    sheet.blank()

    sheet.section("Peer group (nearest first)")
    peer_columns: list[tuple[str, str, str | None, float]] = [
        ("name", "Company", None, 1.0),
        ("industry", "Industry", None, 1.0),
        ("selection", "Selected by", None, 1.0),
        ("distance", "Distance", FMT_DECIMAL, 1.0),
        ("market_cap", "Market cap", FMT_MILLIONS, _MILLION),
        ("enterprise_value", "Enterprise value", FMT_MILLIONS, _MILLION),
        ("revenue_growth", "Revenue growth", FMT_PCT, 1.0),
        ("ebitda_margin", "EBITDA margin", FMT_PCT, 1.0),
        ("return_on_equity", "ROE", FMT_PCT, 1.0),
        ("beta", "Beta", FMT_DECIMAL, 1.0),
        ("ev_ebitda", "EV/EBITDA", FMT_MULTIPLE, 1.0),
        ("ev_sales", "EV/Sales", FMT_MULTIPLE, 1.0),
        ("pe", "P/E", FMT_MULTIPLE, 1.0),
        ("used_ev_ebitda", "Used: EV/EBITDA", None, 1.0),
        ("used_ev_sales", "Used: EV/Sales", None, 1.0),
        ("used_pe", "Used: P/E", None, 1.0),
    ]
    sheet.table(comps.peers, peer_columns, index_header="Ticker", kind="input")
    sheet.blank()

    sheet.section("Peer multiples (after outlier removal)")
    summary = comps.multiples_summary.copy()
    summary["target"] = [comps.target_multiples.get(str(m)) for m in summary.index]
    sheet.table(
        summary,
        [
            ("n", "Peers", FMT_INT, 1.0),
            ("min", "Min", FMT_MULTIPLE, 1.0),
            ("p25", "25th pct.", FMT_MULTIPLE, 1.0),
            ("median", "Median", FMT_MULTIPLE, 1.0),
            ("mean", "Mean", FMT_MULTIPLE, 1.0),
            ("p75", "75th pct.", FMT_MULTIPLE, 1.0),
            ("max", "Max", FMT_MULTIPLE, 1.0),
            ("target", f"{comps.ticker} today", FMT_MULTIPLE, 1.0),
        ],
        index_header="Multiple",
        index_format=_label,
    )
    sheet.blank()

    sheet.section("Implied values")
    if comps.implied_values.empty:
        sheet.note("No multiple could be applied to the target.", bold=True)
    else:
        implied = comps.implied_values.copy()
        implied["upside"] = implied["base_to_price"] - 1.0
        sheet.table(
            implied,
            [
                ("metric", "Target metric", None, 1.0),
                ("metric_value", "Metric value", FMT_MILLIONS, _MILLION),
                ("multiple_low", "Multiple low", FMT_MULTIPLE, 1.0),
                ("multiple_base", "Multiple base", FMT_MULTIPLE, 1.0),
                ("multiple_high", "Multiple high", FMT_MULTIPLE, 1.0),
                ("ev_base", "EV base", FMT_MILLIONS, _MILLION),
                ("equity_base", "Equity base", FMT_MILLIONS, _MILLION),
                ("per_share_low", "Per share low", FMT_PER_SHARE, 1.0),
                ("per_share_base", "Per share base", FMT_PER_SHARE, 1.0),
                ("per_share_high", "Per share high", FMT_PER_SHARE, 1.0),
                ("upside", "Base vs. price", FMT_PCT, 1.0),
                ("floored", "Floored at zero", None, 1.0),
            ],
            index_header="Multiple",
            index_format=_label,
        )
    sheet.blank()
    if comps.skipped:
        sheet.section("Multiples not applied")
        sheet.header(["Multiple", "Reason"])
        for multiple, reason in comps.skipped.items():
            sheet.cell(sheet.row, 2, _label(multiple), kind="text")
            sheet.cell(sheet.row, 3, f"Not applicable: {reason}", kind="text", align="left")
            sheet.row += 1
        sheet.blank()

    sheet.section("Equity bridge of the target")
    for name, value in comps.equity_bridge.items():
        sheet.kv(_label(name), value / _MILLION, FMT_MILLIONS, kind="input")
    sheet.kv(f"Share price ({currency})", comps.price, FMT_PER_SHARE, kind="input")
    sheet.kv(
        "Shares outstanding (millions)",
        comps.shares_outstanding / _MILLION,
        FMT_MILLIONS_1,
        kind="input",
    )
    sheet.blank()
    sheet.notes("Fallbacks that fired", [str(f) for f in comps.fallbacks])
    sheet.finish(widths={3: 30, 4: 32, 5: 18}, default_width=14.5)


def _lbo_sheet(ws: Worksheet, report: ValuationReportLike) -> None:
    company = report.company
    currency = company.market.currency
    sheet = _Sheet(
        ws,
        f"{company.profile.name}: leveraged buyout",
        f"{currency} millions unless stated otherwise",
    )
    lbo = report.lbo
    if lbo is None:
        _not_applicable(sheet, report.not_applicable.get("lbo"))
        return
    a = lbo.assumptions

    sheet.section("Transaction")
    sheet.kv(f"Market price ({currency})", lbo.market_price, FMT_PER_SHARE, kind="input")
    sheet.kv(f"Offer price per share ({currency})", lbo.entry_price_per_share, FMT_PER_SHARE)
    sheet.kv("Premium to market price", lbo.entry_price_per_share / lbo.market_price - 1.0, FMT_PCT)
    sheet.kv("Equity purchase price", lbo.entry_equity_value / _MILLION, FMT_MILLIONS)
    sheet.kv("Entry enterprise value", lbo.entry_enterprise_value / _MILLION, FMT_MILLIONS)
    sheet.kv(
        "Entry EBITDA (latest fiscal year)", lbo.entry_ebitda / _MILLION, FMT_MILLIONS, kind="input"
    )
    sheet.kv("Entry EV/EBITDA", lbo.entry_multiple, FMT_MULTIPLE)
    sheet.kv("Debt / EBITDA at closing", lbo.leverage, FMT_MULTIPLE)
    sheet.kv("Sponsor equity share of new capital", lbo.equity_share, FMT_PCT)
    sheet.blank()

    sheet.section("Sources and uses")
    totals: dict[str, str] = {}
    for side, title in (("sources", "Sources"), ("uses", "Uses")):
        block = lbo.sources_and_uses[lbo.sources_and_uses["side"] == side]
        sheet.header([title, f"{currency} m", "% of total"])
        start = sheet.row
        end = start + len(block) - 1
        total_row = end + 1
        for _, record in block.iterrows():
            sheet.cell(sheet.row, 2, str(record["item"]), kind="text")
            sheet.cell(sheet.row, 3, _scaled(record["amount"], _MILLION), fmt=FMT_MILLIONS)
            sheet.cell(
                sheet.row, 4, f"=IF($C${total_row}=0,0,C{sheet.row}/$C${total_row})", fmt=FMT_PCT
            )
            sheet.row += 1
        sheet.cell(
            sheet.row, 2, f"Total {title.lower()}", kind="text", bold=True, border=_BORDER_TOP
        )
        sheet.cell(
            sheet.row, 3, f"=SUM(C{start}:C{end})", fmt=FMT_MILLIONS, bold=True, border=_BORDER_TOP
        )
        sheet.cell(
            sheet.row, 4, f"=SUM(D{start}:D{end})", fmt=FMT_PCT, bold=True, border=_BORDER_TOP
        )
        totals[side] = f"C{sheet.row}"
        sheet.row += 2
    sheet.kv("Check: sources less uses", f"={totals['sources']}-{totals['uses']}", FMT_MILLIONS)
    sheet.blank()

    sheet.section("Operating forecast")
    sheet.statement(
        lbo.forecast,
        column_suffix="E",
        input_rows=("revenue_growth", "ebitda_margin"),
        bold_rows=("ebitda", "cash_flow_available_for_debt_service"),
        corner=f"{currency} m",
    )
    sheet.blank()
    sheet.section("Debt schedule")
    sheet.statement(
        lbo.debt_schedule,
        column_suffix="E",
        bold_rows=(
            "senior_closing",
            "subordinated_closing",
            "total_debt_closing",
            "net_debt_closing",
        ),
        corner=f"{currency} m",
    )
    sheet.blank()

    sheet.section("Exit and returns")
    sheet.kv("Exit year", lbo.exit_year, "0")
    sheet.kv(
        "Exit EV/EBITDA",
        lbo.exit_multiple,
        FMT_MULTIPLE,
        kind="input" if a.exit_multiple is not None else "calc",
        note=(
            getattr(report, "lbo_exit_multiple_source", None)
            or ("assumption" if a.exit_multiple is not None else "equal to the entry multiple")
        ),
    )
    sheet.kv("Exit EBITDA", lbo.exit_ebitda / _MILLION, FMT_MILLIONS)
    sheet.kv("Exit enterprise value", lbo.exit_enterprise_value / _MILLION, FMT_MILLIONS)
    sheet.kv("Less: net debt at exit", lbo.exit_net_debt / _MILLION, FMT_MILLIONS)
    sheet.kv("Less: minority interest", lbo.exit_minority_interest / _MILLION, FMT_MILLIONS)
    sheet.kv("Sponsor proceeds at exit", lbo.exit_equity_value / _MILLION, FMT_MILLIONS, bold=True)
    sheet.kv("Sponsor equity invested", lbo.sponsor_equity / _MILLION, FMT_MILLIONS)
    sheet.kv("IRR", lbo.irr, FMT_PCT, bold=True)
    sheet.kv("MOIC", lbo.moic, FMT_MULTIPLE_2, bold=True)
    sheet.header(["Sponsor cash flows", *[f"Year {t}" for t in range(len(lbo.equity_cash_flows))]])
    sheet.line(f"{currency} m", [flow / _MILLION for flow in lbo.equity_cash_flows], FMT_MILLIONS)
    sheet.blank()

    sheet.section("Returns attribution")
    sheet.table(
        lbo.attribution,
        [
            ("amount", f"{currency} m", FMT_MILLIONS, _MILLION),
            ("share", "Share of |total|", FMT_PCT, 1.0),
        ],
        index_header="Component",
        index_format=_label,
    )
    sheet.blank()

    sheet.section("Ability to pay (football-field bar)")
    sheet.header(["Target IRR", f"Price per share ({currency})", "Solver status"])
    for target in sorted(lbo.ability_to_pay, reverse=True):
        sheet.cell(sheet.row, 2, float(target), fmt=FMT_PCT, kind="input", align="left")
        sheet.cell(sheet.row, 3, _clean(lbo.ability_to_pay[target]), fmt=FMT_PER_SHARE)
        sheet.cell(
            sheet.row, 4, str(lbo.ability_to_pay_status.get(target, "")), kind="text", align="right"
        )
        sheet.row += 1
    sheet.blank()

    grids = (
        (
            "Sensitivity: IRR, entry x exit EV/EBITDA",
            lbo.sensitivity_irr_entry_exit,
            "Entry \\ exit multiple",
            FMT_PCT,
            lbo.entry_multiple,
        ),
        (
            "Sensitivity: MOIC, entry x exit EV/EBITDA",
            lbo.sensitivity_moic_entry_exit,
            "Entry \\ exit multiple",
            FMT_MULTIPLE_2,
            lbo.entry_multiple,
        ),
        (
            "Sensitivity: IRR, requested debt / EBITDA x exit EV/EBITDA",
            lbo.sensitivity_irr_leverage_exit,
            "Leverage \\ exit multiple",
            FMT_PCT,
            a.senior_debt_multiple + a.subordinated_debt_multiple,
        ),
    )
    for title, grid, corner, cell_fmt, row_base in grids:
        sheet.section(title)
        sheet.grid(
            grid,
            corner=corner,
            row_fmt=FMT_MULTIPLE,
            column_fmt=FMT_MULTIPLE,
            cell_fmt=cell_fmt,
            base=(_nearest(grid.index, row_base), _nearest(grid.columns, lbo.exit_multiple)),
        )
        sheet.blank()
    sheet.notes("Warnings and sizing notes", [str(w) for w in lbo.warnings])
    sheet.finish(widths={3: 16, 4: 16})


def _monte_carlo_sheet(ws: Worksheet, report: ValuationReportLike) -> None:
    company = report.company
    currency = company.market.currency
    sheet = _Sheet(
        ws,
        f"{company.profile.name}: Monte Carlo simulation of the DCF",
        f"Value per share in {currency}; summary statistics only (raw draws are not exported)",
    )
    mc = report.monte_carlo
    if mc is None:
        _not_applicable(sheet, report.not_applicable.get("monte_carlo"))
        return

    sheet.section("Statistics (Gordon growth terminal value)")
    sheet.kv("Simulations", int(mc.n_simulations), FMT_INT, kind="input")
    sheet.kv("Base case value per share", mc.base_value, FMT_PER_SHARE)
    sheet.kv("Mean", mc.mean, FMT_PER_SHARE)
    sheet.kv("Standard deviation", mc.std, FMT_PER_SHARE)
    sheet.kv("Standard error of the mean", mc.standard_error, "0.000")
    sheet.kv("Current share price", mc.price, FMT_PER_SHARE, kind="input")
    sheet.kv("Probability of value above price", mc.prob_above_price, FMT_PCT)
    sheet.blank()

    sheet.section("Percentiles")
    exit_values = np.asarray(getattr(mc, "values_exit", []), dtype=float)
    exit_values = exit_values[np.isfinite(exit_values)]
    sheet.header(["Percentile", "Gordon growth", "Exit multiple"])
    for level in (5, 10, 25, 50, 75, 90, 95):
        exit_value = float(np.percentile(exit_values, level)) if exit_values.size else None
        sheet.line(f"P{level}", [percentile(mc, level), exit_value], FMT_PER_SHARE)
    sheet.blank()

    inputs = getattr(mc, "inputs", None)
    if isinstance(inputs, pd.DataFrame) and not inputs.empty:
        sheet.section("Simulated inputs")
        sheet.header(["Driver", "Mean", "Std. dev.", "P5", "P50", "P95"])
        for name in inputs.columns:
            column = pd.to_numeric(inputs[name], errors="coerce").dropna()
            if column.empty:
                continue
            fmt = FMT_MULTIPLE if str(name) == "exit_multiple" else FMT_PCT_2
            sheet.line(
                _label(name),
                [column.mean(), column.std(), *np.percentile(column, [5, 50, 95]).tolist()],
                fmt,
            )
        sheet.blank()

    sheet.section("Tornado: one driver at a time")
    tornado = mc.tornado
    if tornado is None or len(tornado) == 0:
        sheet.note("No tornado data.")
    else:
        sheet.header(
            ["Driver", "Input low", "Input high", "Value at low", "Value at high", "Swing"]
        )
        for name, record in tornado.iterrows():
            fmt = FMT_MULTIPLE if str(name) == "exit_multiple" else FMT_PCT_2
            row = sheet.row
            sheet.cell(row, 2, _label(name), kind="text")
            sheet.cell(row, 3, _clean(record.get("input_low")), fmt=fmt, kind="input")
            sheet.cell(row, 4, _clean(record.get("input_high")), fmt=fmt, kind="input")
            sheet.cell(row, 5, _clean(record.get("value_low")), fmt=FMT_PER_SHARE)
            sheet.cell(row, 6, _clean(record.get("value_high")), fmt=FMT_PER_SHARE)
            sheet.cell(row, 7, _clean(record.get("swing")), fmt=FMT_PER_SHARE)
            sheet.row += 1
    sheet.blank()

    sheet.section("Contribution to variance")
    contribution = mc.variance_contribution
    if contribution is None or len(contribution) == 0:
        sheet.note("No variance decomposition.")
    else:
        sheet.header(["Driver", "Share of variance"])
        for name, value in contribution.items():
            sheet.line(_label(name), [value], FMT_PCT)
    sheet.blank()

    convergence = mc.convergence
    if isinstance(convergence, pd.DataFrame) and len(convergence) > 0:
        sheet.section("Convergence")
        step = max(1, math.ceil(len(convergence) / 40))
        thinned = convergence.iloc[::step]
        if thinned.index[-1] != convergence.index[-1]:
            thinned = pd.concat([thinned, convergence.iloc[[-1]]])
        sheet.table(
            thinned,
            [
                ("n", "Simulations", FMT_INT, 1.0),
                ("running_mean", "Running mean", FMT_PER_SHARE, 1.0),
                ("running_median", "Running median", FMT_PER_SHARE, 1.0),
                ("standard_error", "Standard error", "0.000", 1.0),
            ],
        )
    sheet.finish(default_width=15.0)


def _fx_number(text: Any) -> float | None:
    """Leading number of a ``source_map`` FX entry such as ``'9.20133 (2025-12)'``."""
    try:
        value = float(str(text).split()[0])
    except (ValueError, IndexError):
        return None
    return value if math.isfinite(value) else None


def _data_sheet(ws: Worksheet, report: ValuationReportLike) -> None:
    company = report.company
    statements = company.statements
    currency = statements.currency
    profile = company.profile
    sheet = _Sheet(
        ws,
        f"{profile.name}: standardized historical statements",
        f"{currency} millions, fiscal years · source: Yahoo Finance (blue = source data)",
    )
    fx_text = describe_statement_fx(company)
    if fx_text:
        sheet.note(fx_text)
        fx_rows = (
            [
                (
                    year,
                    _fx_number(source_map.get(f"fx.average.{year}")),
                    _fx_number(source_map.get(f"fx.year_end.{year}")),
                )
                for year in statements.fiscal_years
            ]
            if (source_map := getattr(statements, "source_map", None) or {})
            else []
        )
        fx_rows = [row for row in fx_rows if row[1] is not None or row[2] is not None]
        if fx_rows:
            sheet.header(
                [
                    f"FX rates applied ({profile.reporting_currency} to {currency})",
                    *[f"{year}A" for year, _, _ in fx_rows],
                ]
            )
            for label, position in (
                ("Year average (income statement, cash flow)", 1),
                ("Year-end (balance sheet)", 2),
            ):
                sheet.cell(sheet.row, sheet.LABEL_COL, label, kind="text")
                for offset, row in enumerate(fx_rows, start=1):
                    sheet.cell(
                        sheet.row,
                        sheet.LABEL_COL + offset,
                        row[position],
                        fmt="0.0000",
                        kind="input",
                    )
                sheet.row += 1
        sheet.blank()
    for title, frame in (
        ("Income statement", statements.income),
        ("Balance sheet", statements.balance),
        ("Cash flow statement", statements.cash_flow),
    ):
        sheet.section(title)
        if frame is None or frame.empty:
            sheet.note("No data.")
        else:
            sheet.statement(frame, kind="input", column_suffix="A", corner=f"{currency} m")
        sheet.blank()
    if statements.source_map:
        sheet.section("Source tags used")
        sheet.header(["Standardized item", "Source tag"])
        for item, tag in statements.source_map.items():
            sheet.cell(sheet.row, 2, str(item), kind="text")
            sheet.cell(sheet.row, 3, str(tag), kind="text", align="left")
            sheet.row += 1
    sheet.finish()


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def export_to_excel(
    report: ValuationReportLike,
    path: str | Path,
    *,
    football_field: Figure | str | Path | None = None,
    embed_football_field: bool = True,
) -> Path:
    """Write the valuation workbook for ``report`` to ``path`` and return the path.

    Parameters
    ----------
    report:
        Any object with the attributes of ``ValuationReportLike`` (``company``,
        ``dcf``, ``comps``, ``lbo``, ``monte_carlo``, ``valuations``,
        ``not_applicable``).
    path:
        Target file; ``.xlsx`` is appended when missing and parent folders are
        created.
    football_field:
        Football-field chart to embed on the Summary sheet: a matplotlib figure
        or the path of a PNG. When omitted and ``embed_football_field`` is true
        the chart is rendered from the report.
    embed_football_field:
        Set to False to write the workbook without any image.

    Sheets of models that are not applicable are still written and carry a
    "Not applicable: <reason>" note.
    """
    target = Path(path)
    if target.suffix.lower() != ".xlsx":
        target = target.with_name(target.name + ".xlsx")
    target.parent.mkdir(parents=True, exist_ok=True)

    image: Figure | str | Path | None = None
    if embed_football_field:
        image = football_field if football_field is not None else report_football_field(report)

    workbook = Workbook()
    summary = workbook.active
    summary.title = SHEET_NAMES[0]
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    _summary_sheet(summary, report, image, generated)
    _assumptions_sheet(workbook.create_sheet(SHEET_NAMES[1]), report)
    _dcf_sheet(workbook.create_sheet(SHEET_NAMES[2]), report)
    _comps_sheet(workbook.create_sheet(SHEET_NAMES[3]), report)
    _lbo_sheet(workbook.create_sheet(SHEET_NAMES[4]), report)
    _monte_carlo_sheet(workbook.create_sheet(SHEET_NAMES[5]), report)
    _data_sheet(workbook.create_sheet(SHEET_NAMES[6]), report)

    workbook.properties.title = f"{report.company.profile.name} valuation"
    workbook.properties.creator = "valuation-engine"
    workbook.save(target)
    return target
