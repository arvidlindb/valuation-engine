"""Streamlit front end of the valuation engine.

Run with::

    streamlit run app/streamlit_app.py

Pick a company in the sidebar, adjust the key assumptions and read the result
per method in the tabs. The Excel workbook of the current run can be downloaded
from the Overview tab.

All numbers come from ``valuation_engine.pipeline.run_valuation``; the sidebar
inputs are turned into DCF and LBO assumptions and passed to the pipeline.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import inspect
import io
import math
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

_SRC = Path(__file__).resolve().parents[1] / "src"
if importlib.util.find_spec("valuation_engine") is None and _SRC.is_dir():
    sys.path.insert(0, str(_SRC))

from valuation_engine import data as data_layer  # noqa: E402
from valuation_engine.core.types import (  # noqa: E402
    CompanyData,
    DCFAssumptions,
    MarketParameters,
    NotApplicableError,
    PeerSnapshot,
    ValuationMethod,
)
from valuation_engine.data import DataError  # noqa: E402
from valuation_engine.export import export_to_excel  # noqa: E402
from valuation_engine.models import dcf as dcf_model  # noqa: E402
from valuation_engine.models import lbo as lbo_model  # noqa: E402
from valuation_engine.models.lbo import LBOAssumptions  # noqa: E402
from valuation_engine.pipeline import (  # noqa: E402
    ValuationReport,
    lbo_assumptions_from_dcf,
    run_valuation,
)
from valuation_engine.viz import (  # noqa: E402
    ValuationReportLike,
    build_charts,
    display_name,
    report_valuations,
)
from valuation_engine.viz.style import (  # noqa: E402
    fmt_money,
    fmt_multiple,
    fmt_number,
    fmt_pct,
    fmt_per_share,
)

DEFAULT_TICKER = "ATCO-A.ST"
SIMULATION_CHOICES: tuple[int, ...] = (1_000, 2_500, 5_000, 10_000, 25_000, 50_000)
_MILLION = 1e6


# --------------------------------------------------------------------------
# User overrides
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Overrides:
    """Sidebar inputs. ``None`` means "use the model default"."""

    revenue_growth_start: float | None = None  # year-1 growth, fades to long-term growth
    ebitda_margin: float | None = None  # flat over the forecast
    wacc: float | None = None
    terminal_growth: float | None = None
    exit_multiple: float | None = None
    takeover_premium: float | None = None
    senior_debt_multiple: float | None = None
    subordinated_debt_multiple: float | None = None
    n_simulations: int = 10_000
    seed: int = 42


def _fade(start: float, end: float, years: int) -> tuple[float, ...]:
    """Linear path from ``start`` in year 1 to ``end`` in the last year."""
    if years == 1:
        return (start,)
    return tuple(float(start + (end - start) * t / (years - 1)) for t in range(years))


def resolve_dcf_assumptions(
    company: CompanyData, overrides: Overrides, params: MarketParameters
) -> DCFAssumptions | None:
    """Default DCF assumptions with the sidebar overrides applied.

    Returns None when nothing is overridden, so that an untouched sidebar gives
    exactly the model's default run. Raises ``NotApplicableError`` like the model.
    """
    touched = (
        overrides.revenue_growth_start,
        overrides.ebitda_margin,
        overrides.wacc,
        overrides.terminal_growth,
        overrides.exit_multiple,
    )
    if all(value is None for value in touched):
        return None
    base = dcf_model.default_assumptions(company, params)
    changes: dict[str, Any] = {}
    if overrides.revenue_growth_start is not None:
        changes["revenue_growth"] = _fade(
            overrides.revenue_growth_start, params.long_term_growth, base.forecast_years
        )
    if overrides.ebitda_margin is not None:
        changes["ebitda_margin"] = tuple(overrides.ebitda_margin for _ in base.ebitda_margin)
    if overrides.wacc is not None:
        changes["wacc"] = overrides.wacc
    if overrides.terminal_growth is not None:
        changes["terminal_growth"] = overrides.terminal_growth
    if overrides.exit_multiple is not None:
        changes["exit_multiple"] = overrides.exit_multiple
    return replace(base, **changes)


def resolve_lbo_assumptions(
    company: CompanyData,
    overrides: Overrides,
    params: MarketParameters,
    exit_multiple: float | None,
) -> LBOAssumptions:
    """Default LBO assumptions with the sidebar overrides applied.

    ``exit_multiple`` is the DCF's exit multiple (PLAN.md: the LBO exits at the
    same multiple as the DCF instead of exit = entry); None keeps exit = entry.
    """
    base = lbo_model.default_assumptions(company, params)
    changes: dict[str, Any] = {}
    years = base.holding_period
    if overrides.revenue_growth_start is not None:
        changes["revenue_growth"] = _fade(
            overrides.revenue_growth_start, params.long_term_growth, years
        )
    if overrides.ebitda_margin is not None:
        changes["ebitda_margin"] = tuple(overrides.ebitda_margin for _ in range(years))
    if overrides.takeover_premium is not None:
        changes["takeover_premium"] = overrides.takeover_premium
    if overrides.senior_debt_multiple is not None:
        changes["senior_debt_multiple"] = overrides.senior_debt_multiple
    if overrides.subordinated_debt_multiple is not None:
        changes["subordinated_debt_multiple"] = overrides.subordinated_debt_multiple
    if exit_multiple is not None and exit_multiple > 0:
        changes["exit_multiple"] = float(exit_multiple)
    return replace(base, **changes) if changes else base


# --------------------------------------------------------------------------
# Pipeline entry
# --------------------------------------------------------------------------


def run_report(ticker: str, offline: bool, overrides: Overrides) -> ValuationReport:
    """Run the valuation pipeline for ``ticker`` with the sidebar overrides.

    Untouched inputs are passed as ``None`` so that the pipeline uses its own
    defaults. The LBO exit multiple is left open: the pipeline fills it with the
    DCF's exit multiple (which is the sidebar value when that is overridden).
    LBO assumptions with sidebar changes get the DCF's operating forecast, like
    the pipeline's own LBO defaults.
    """
    params = _load_params(offline)
    company = _load_company(ticker, offline)
    dcf_assumptions: DCFAssumptions | None
    lbo_assumptions: LBOAssumptions | None
    try:
        dcf_assumptions = resolve_dcf_assumptions(company, overrides, params)
    except NotApplicableError:
        dcf_assumptions = None
    lbo_touched = (
        overrides.revenue_growth_start,
        overrides.ebitda_margin,
        overrides.takeover_premium,
        overrides.senior_debt_multiple,
        overrides.subordinated_debt_multiple,
    )
    try:
        lbo_assumptions = (
            resolve_lbo_assumptions(company, overrides, params, None)
            if any(v is not None for v in lbo_touched)
            else None
        )
    except (NotApplicableError, ValueError):
        lbo_assumptions = None
    if lbo_assumptions is not None:
        # One company, one operating forecast: the pipeline aligns the LBO with the
        # DCF only when it builds the LBO defaults itself, so do the same here for
        # LBO assumptions that carry sidebar changes (premium, leverage).
        try:
            operating = dcf_assumptions or dcf_model.default_assumptions(company, params)
            lbo_assumptions = lbo_assumptions_from_dcf(lbo_assumptions, operating)
        except NotApplicableError:
            pass
    return run_valuation(
        ticker,
        offline=offline,
        params=params,
        dcf_assumptions=dcf_assumptions,
        lbo_assumptions=lbo_assumptions,
        n_simulations=overrides.n_simulations,
        seed=overrides.seed,
        universe=_load_peers(offline),
    )


# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _load_universe(offline: bool) -> dict[str, str]:
    """Ticker -> company name for the whole universe."""
    entries, _ = data_layer.load_universe_entries(offline=offline)
    return {entry.ticker: entry.name for entry in entries}


@st.cache_resource(show_spinner=False, max_entries=64)
def _load_company(ticker: str, offline: bool) -> CompanyData:
    return data_layer.load_company(ticker, offline=offline)


@st.cache_resource(show_spinner=False)
def _load_peers(offline: bool) -> list[PeerSnapshot]:
    return data_layer.load_peer_snapshots(offline=offline)


@st.cache_resource(show_spinner=False)
def _load_params(offline: bool) -> MarketParameters:
    """Sourced market parameters (Riksbank risk-free rate, Damodaran ERP)."""
    return data_layer.load_market_parameters(offline=offline)


@st.cache_resource(show_spinner="Running the valuation models…", max_entries=32)
def _cached_report(ticker: str, offline: bool, overrides: Overrides) -> ValuationReport:
    return run_report(ticker, offline, overrides)


@st.cache_data(show_spinner="Drawing charts…", max_entries=32)
def _cached_charts(ticker: str, offline: bool, overrides: Overrides) -> dict[str, bytes]:
    """All charts of the report as PNG bytes."""
    report = _cached_report(ticker, offline, overrides)
    images: dict[str, bytes] = {}
    for name, figure in build_charts(report).items():
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=170, facecolor="white")
        images[name] = buffer.getvalue()
    return images


@st.cache_data(show_spinner="Building the Excel workbook…", max_entries=16)
def _cached_excel(ticker: str, offline: bool, overrides: Overrides) -> bytes:
    report = _cached_report(ticker, offline, overrides)
    with tempfile.TemporaryDirectory() as folder:
        path = export_to_excel(report, Path(folder) / f"{ticker}_valuation.xlsx")
        return path.read_bytes()


# --------------------------------------------------------------------------
# Small UI helpers
# --------------------------------------------------------------------------

_NEW_WIDTH_API = "width" in inspect.signature(st.dataframe).parameters and (
    inspect.signature(st.image).parameters["width"].default == "content"
)


def _stretch() -> dict[str, Any]:
    """Keyword that makes an element fill its container (API differs by version)."""
    return {"width": "stretch"} if _NEW_WIDTH_API else {"use_container_width": True}


def _chart(images: dict[str, bytes], name: str) -> None:
    if name in images:
        st.image(images[name], **_stretch())


def _table(frame: pd.DataFrame, formats: dict[str, str] | str | None = None) -> None:
    """Show a DataFrame with number formats (``{column: format}`` or one format)."""
    styler = frame.style
    styler = styler.format(formats, na_rep="-") if formats else styler.format(na_rep="-")
    st.dataframe(styler, **_stretch())


def _statement_table(frame: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
    """Statement transposed to line items x years, amounts in millions."""
    pct = {"revenue_growth", "ebitda_margin"}
    ratio = {"net_debt_to_ebitda", "interest_coverage", "discount_period", "discount_factor"}
    out: dict[str, list[str]] = {}
    for name in frame.columns:
        values = frame[name].tolist()
        if name in pct:
            cells = [fmt_pct(v, 1) for v in values]
        elif name in ratio:
            cells = [fmt_number(v, 2) for v in values]
        elif name == "liquidity_breach":
            cells = ["Yes" if bool(v) else "No" for v in values]
        else:
            cells = [fmt_number(v / _MILLION, 0) if _finite(v) else "-" for v in values]
        out[str(name).replace("_", " ").capitalize()] = cells
    table = pd.DataFrame(out, index=[f"{year}{suffix}" for year in frame.index]).T
    table.index.name = "Line item"
    return table


def _finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _not_applicable(report: ValuationReportLike, key: str, name: str) -> None:
    reason = report.not_applicable.get(key, "no result available")
    st.info(f"**{name} is not applicable to this company.** {reason}")


def _bullets(title: str, lines: list[str]) -> None:
    if lines:
        with st.expander(f"{title} ({len(lines)})"):
            st.markdown("\n".join(f"- {line}" for line in lines))


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------


def _percent_input(
    label: str, default: float, key: str, *, step: float = 0.5, **kwargs: Any
) -> float | None:
    """Number input in percent; returns a decimal, or None when left at the default."""
    shown = round(default * 100.0, 1)
    value = st.number_input(label, value=shown, step=step, format="%.1f", key=key, **kwargs)
    return None if abs(value - shown) < 1e-9 else float(value) / 100.0


def _lbo_field_default(name: str) -> float:
    return float(LBOAssumptions.__dataclass_fields__[name].default)


@dataclass(frozen=True)
class _LBODefaults:
    """Defaults of the LBO financing inputs shown in the sidebar."""

    takeover_premium: float = _lbo_field_default("takeover_premium")
    senior_debt_multiple: float = _lbo_field_default("senior_debt_multiple")
    subordinated_debt_multiple: float = _lbo_field_default("subordinated_debt_multiple")


def _sidebar_overrides(company: CompanyData, offline: bool) -> Overrides:
    """Assumption controls; defaults come from the pipeline's base case."""
    ticker = company.ticker
    params = _load_params(offline)
    values: dict[str, Any] = {}
    try:
        defaults = dcf_model.default_assumptions(company, params)
        # The untouched pipeline run: its WACC and exit multiple are the defaults
        # shown next to the override switches (and it is the report displayed
        # as long as nothing is changed, so the run is not wasted).
        baseline = _cached_report(ticker, offline, Overrides()).dcf
        if baseline is None:
            raise NotApplicableError("no DCF base case")
    except NotApplicableError:
        st.caption(
            "DCF, LBO and Monte Carlo are not applicable to this company, so there are no "
            "assumptions to adjust."
        )
        return Overrides()

    st.subheader("Operating forecast")
    values["revenue_growth_start"] = _percent_input(
        "Revenue growth, year 1 (%)",
        defaults.revenue_growth[0],
        f"growth_{ticker}",
        min_value=-50.0,
        max_value=100.0,
        help="Fades linearly to the long-term growth rate by the last forecast year. "
        "Applies to the DCF and the LBO.",
    )
    values["ebitda_margin"] = _percent_input(
        "EBITDA margin (%)",
        defaults.ebitda_margin[0],
        f"margin_{ticker}",
        min_value=0.1,
        max_value=90.0,
        help="Held flat over the forecast. Applies to the DCF and the LBO.",
    )

    st.subheader("Discounting and terminal value")
    if st.checkbox(
        "Override WACC",
        key=f"wacc_on_{ticker}",
        help=f"Computed WACC: {fmt_pct(baseline.wacc, 2)}",
    ):
        values["wacc"] = (
            st.number_input(
                "WACC (%)",
                value=round(baseline.wacc * 100.0, 1),
                min_value=1.0,
                max_value=40.0,
                step=0.25,
                format="%.2f",
                key=f"wacc_{ticker}",
            )
            / 100.0
        )
    values["terminal_growth"] = _percent_input(
        "Terminal growth (%)",
        defaults.terminal_growth,
        f"g_{ticker}",
        step=0.25,
        min_value=-2.0,
        max_value=6.0,
    )
    if st.checkbox(
        "Override exit multiple",
        key=f"exit_on_{ticker}",
        help=f"Default: {fmt_multiple(baseline.exit_multiple)} ({baseline.exit_multiple_source})",
    ):
        values["exit_multiple"] = st.number_input(
            "Exit EV/EBITDA multiple (x)",
            value=round(baseline.exit_multiple, 1),
            min_value=1.0,
            max_value=60.0,
            step=0.5,
            format="%.1f",
            key=f"exit_{ticker}",
        )

    st.subheader("LBO")
    lbo_defaults = _LBODefaults()
    values["takeover_premium"] = _percent_input(
        "Takeover premium (%)",
        lbo_defaults.takeover_premium,
        f"premium_{ticker}",
        step=5.0,
        min_value=0.0,
        max_value=200.0,
    )
    senior = st.number_input(
        "Senior debt (x EBITDA)",
        value=lbo_defaults.senior_debt_multiple,
        min_value=0.0,
        max_value=10.0,
        step=0.5,
        format="%.1f",
        key=f"senior_{ticker}",
    )
    subordinated = st.number_input(
        "Subordinated debt (x EBITDA)",
        value=lbo_defaults.subordinated_debt_multiple,
        min_value=0.0,
        max_value=6.0,
        step=0.5,
        format="%.1f",
        key=f"sub_{ticker}",
        help="Requested leverage; the model cuts it when coverage or the minimum equity "
        "share would be breached.",
    )
    if senior != lbo_defaults.senior_debt_multiple:
        values["senior_debt_multiple"] = float(senior)
    if subordinated != lbo_defaults.subordinated_debt_multiple:
        values["subordinated_debt_multiple"] = float(subordinated)

    st.subheader("Monte Carlo")
    values["n_simulations"] = st.select_slider(
        "Simulations",
        options=list(SIMULATION_CHOICES),
        value=10_000,
        format_func=lambda n: f"{n:,}",
        key="n_simulations",
    )
    values["seed"] = int(st.number_input("Random seed", value=42, min_value=0, step=1, key="seed"))
    return Overrides(**values)


# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------


def _overview_tab(
    report: ValuationReportLike, images: dict[str, bytes], excel: bytes | None
) -> None:
    market = report.company.market
    currency = market.currency
    _chart(images, "football_field")

    rows = report_valuations(report)
    if rows:
        table = pd.DataFrame(
            {
                "Method": [r.label for r in rows],
                "Low": [r.low for r in rows],
                "Base": [r.base for r in rows],
                "High": [r.high for r in rows],
                "Base vs. price": [
                    fmt_pct(r.base / market.price - 1.0, 1, sign=True)
                    if r.method != ValuationMethod.MARKET
                    else "-"
                    for r in rows
                ],
                f"Implied EV ({currency} m)": [
                    fmt_number(r.enterprise_value / _MILLION)
                    if _finite(r.enterprise_value)
                    else "-"
                    for r in rows
                ],
            }
        ).set_index("Method")
        _table(table, {"Low": "{:,.2f}", "Base": "{:,.2f}", "High": "{:,.2f}"})
        st.caption(
            f"Value per share in {currency}. Ranges: DCF from the sensitivity grids, comps from "
            "the 25th to 75th percentile peer multiple, LBO from the price giving a 25% / 20% / "
            "15% sponsor IRR, Monte Carlo from P10 to P90."
        )
    else:
        st.warning("No valuation method could be applied to this company.")

    for key, reason in report.not_applicable.items():
        name = {"dcf": "DCF", "comps": "Trading comps", "lbo": "LBO", "monte_carlo": "Monte Carlo"}
        st.info(f"**{name.get(key, key)} not applicable:** {reason}")
    for warning in getattr(report, "warnings", None) or []:
        st.warning(warning)

    if excel is not None:
        st.download_button(
            "Download Excel workbook",
            data=excel,
            file_name=f"{report.company.ticker}_valuation.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    for note in getattr(report, "notes", None) or []:
        st.caption(note)


def _dcf_tab(report: ValuationReportLike, images: dict[str, bytes]) -> None:
    dcf = report.dcf
    if dcf is None:
        _not_applicable(report, "dcf", "The DCF")
        return
    currency = dcf.currency
    columns = st.columns(5)
    columns[0].metric(
        "Value per share, Gordon",
        fmt_per_share(dcf.value_per_share_gordon, currency),
        fmt_pct(dcf.upside_gordon, 1, sign=True) + " vs. price",
    )
    columns[1].metric(
        "Value per share, exit multiple",
        fmt_per_share(dcf.value_per_share_exit, currency),
        fmt_pct(dcf.upside_exit, 1, sign=True) + " vs. price",
    )
    columns[2].metric("WACC", fmt_pct(dcf.wacc, 2))
    columns[3].metric("Exit multiple", fmt_multiple(dcf.exit_multiple))
    columns[4].metric("Terminal value / EV (Gordon)", fmt_pct(dcf.tv_share_gordon, 0))

    _chart(images, "dcf_fcff_forecast")
    left, right = st.columns(2)
    with left:
        _chart(images, "dcf_bridge_gordon")
        _chart(images, "dcf_sensitivity_wacc_growth")
    with right:
        _chart(images, "dcf_bridge_exit")
        _chart(images, "dcf_sensitivity_wacc_exit_multiple")

    st.subheader(f"Forecast ({currency} m)")
    _table(_statement_table(dcf.forecast, "E"))

    w = dcf.wacc_result
    st.subheader("WACC build-up")
    if w is None:
        st.write(f"WACC of {fmt_pct(dcf.wacc, 2)} given as an assumption; CAPM was not run.")
    else:
        build_up = pd.DataFrame(
            [
                ("Risk-free rate", fmt_pct(w.risk_free_rate, 2), ""),
                ("Equity risk premium", fmt_pct(w.equity_risk_premium, 2), ""),
                ("Beta", fmt_number(w.beta, 2), w.beta_method),
                ("Size premium", fmt_pct(w.size_premium, 2), w.size_premium_basis),
                ("Cost of equity", fmt_pct(w.cost_of_equity, 2), "risk-free + beta x ERP + size"),
                ("Pre-tax cost of debt", fmt_pct(w.pre_tax_cost_of_debt, 2), w.cost_of_debt_source),
                ("After-tax cost of debt", fmt_pct(w.after_tax_cost_of_debt, 2), ""),
                ("Equity weight", fmt_pct(w.equity_weight, 1), "market values"),
                ("Debt weight", fmt_pct(w.debt_weight, 1), ""),
                ("WACC", fmt_pct(w.wacc, 2), ""),
            ],
            columns=["Component", "Value", "Basis"],
        ).set_index("Component")
        _table(build_up)
    st.caption(
        f"Exit multiple {fmt_multiple(dcf.exit_multiple)}: {dcf.exit_multiple_source}. Implied "
        f"exit multiple of the Gordon case: {fmt_multiple(dcf.implied_exit_multiple)}; implied "
        f"perpetual growth of the exit case: {fmt_pct(dcf.implied_perpetual_growth, 2)}."
    )
    _bullets("Model notes and fallbacks", [str(n) for n in dcf.notes])


def _comps_tab(report: ValuationReportLike, images: dict[str, bytes]) -> None:
    comps = report.comps
    if comps is None:
        _not_applicable(report, "comps", "The comparable company analysis")
        return
    currency = comps.currency
    labels = {"ev_ebitda": "EV/EBITDA", "ev_sales": "EV/Sales", "pe": "P/E"}
    columns = st.columns(4)
    columns[0].metric("Peers", str(len(comps.peers)))
    for column, multiple in zip(columns[1:], labels, strict=True):
        if multiple in comps.implied_values.index:
            row = comps.implied_values.loc[multiple]
            column.metric(
                f"Implied value, {labels[multiple]}",
                fmt_per_share(row["per_share_base"], currency),
                fmt_pct(row["base_to_price"] - 1.0, 1, sign=True) + " vs. price",
            )
        else:
            column.metric(f"Implied value, {labels[multiple]}", "n/a")
    for multiple, reason in comps.skipped.items():
        st.info(f"**{labels.get(multiple, multiple)} not applicable:** {reason}")

    _chart(images, "comps_peer_multiples")

    if not comps.implied_values.empty:
        st.subheader("Implied values")
        implied = comps.implied_values[
            [
                "n_peers",
                "multiple_low",
                "multiple_base",
                "multiple_high",
                "per_share_low",
                "per_share_base",
                "per_share_high",
            ]
        ].rename(
            index=labels,
            columns={
                "n_peers": "Peers",
                "multiple_low": "Multiple, 25th pct.",
                "multiple_base": "Multiple, median",
                "multiple_high": "Multiple, 75th pct.",
                "per_share_low": f"Low ({currency})",
                "per_share_base": f"Base ({currency})",
                "per_share_high": f"High ({currency})",
            },
        )
        implied.index.name = "Multiple"
        formats = {c: "{:,.1f}x" for c in implied.columns if c.startswith("Multiple")}
        formats.update({c: "{:,.2f}" for c in implied.columns if c.endswith(")")})
        formats["Peers"] = "{:.0f}"
        _table(implied, formats)

    st.subheader("Peer group")
    peers = comps.peers[
        ["name", "industry", "selection", "market_cap", "ev_ebitda", "ev_sales", "pe"]
    ].copy()
    peers["market_cap"] = peers["market_cap"] / 1e9
    peers = peers.rename(
        columns={
            "name": "Company",
            "industry": "Industry",
            "selection": "Selected by",
            "market_cap": "Market cap (SEK bn)",
            "ev_ebitda": "EV/EBITDA",
            "ev_sales": "EV/Sales",
            "pe": "P/E",
        }
    )
    peers.index.name = "Ticker"
    _table(
        peers,
        {
            "Market cap (SEK bn)": "{:,.1f}",
            "EV/EBITDA": "{:,.1f}x",
            "EV/Sales": "{:,.1f}x",
            "P/E": "{:,.1f}x",
        },
    )
    left, right = st.columns([3, 2])
    with left:
        _chart(images, "comps_silhouette")
    with right:
        candidates = (
            f"**Peer selection.** Candidates: {len(comps.candidates)} companies "
            f"({comps.candidate_scope.replace('_', ' ')}). "
        )
        if comps.chosen_k > 1 and comps.silhouette_scores:
            st.markdown(
                candidates + f"K-means on {len(comps.feature_names)} standardized features; "
                f"k = {comps.chosen_k} has the highest silhouette score. Peers are the "
                "target's cluster, capped to the nearest companies with a preference for the "
                "same industry."
            )
        else:
            st.markdown(
                candidates + "No clustering was run (too few candidates for k-means), so there "
                f"is no silhouette score. Peers are the nearest candidates on "
                f"{len(comps.feature_names)} standardized features, with a preference for the "
                "same industry."
            )
        _bullets("Fallbacks that fired", [str(f) for f in comps.fallbacks])


def _lbo_tab(report: ValuationReportLike, images: dict[str, bytes]) -> None:
    lbo = report.lbo
    if lbo is None:
        _not_applicable(report, "lbo", "The LBO")
        return
    currency = lbo.currency
    columns = st.columns(5)
    columns[0].metric("Sponsor IRR", fmt_pct(lbo.irr, 1))
    columns[1].metric("MOIC", fmt_multiple(lbo.moic, 2))
    columns[2].metric("Entry EV/EBITDA", fmt_multiple(lbo.entry_multiple))
    columns[3].metric("Debt / EBITDA at closing", fmt_multiple(lbo.leverage))
    columns[4].metric("Sponsor equity share", fmt_pct(lbo.equity_share, 0))
    st.caption(
        f"Offer price {fmt_per_share(lbo.entry_price_per_share, currency)} per share "
        f"({fmt_pct(lbo.entry_price_per_share / lbo.market_price - 1, 0, sign=True)} to the market "
        f"price), exit in {lbo.exit_year} at {fmt_multiple(lbo.exit_multiple)} EBITDA."
    )
    for warning in lbo.warnings:
        st.warning(warning)

    left, right = st.columns(2)
    with left:
        _chart(images, "lbo_debt_paydown")
        _chart(images, "lbo_sensitivity_entry_exit")
    with right:
        _chart(images, "lbo_returns_waterfall")
        _chart(images, "lbo_sensitivity_leverage_exit")

    left, right = st.columns(2)
    with left:
        st.subheader("Sources and uses")
        table = lbo.sources_and_uses.copy()
        table["amount"] = table["amount"] / _MILLION
        table["side"] = table["side"].str.capitalize()
        table = table.rename(
            columns={"side": "Side", "item": "Item", "amount": f"{currency} m", "pct": "Share"}
        ).set_index(["Side", "Item"])
        _table(table, {f"{currency} m": "{:,.0f}", "Share": "{:.1%}"})
    with right:
        st.subheader("Ability to pay")
        ability = pd.DataFrame(
            {
                "Target IRR": [fmt_pct(k, 0) for k in sorted(lbo.ability_to_pay, reverse=True)],
                f"Price per share ({currency})": [
                    lbo.ability_to_pay[k] for k in sorted(lbo.ability_to_pay, reverse=True)
                ],
                "Status": [
                    lbo.ability_to_pay_status.get(k, "")
                    for k in sorted(lbo.ability_to_pay, reverse=True)
                ],
            }
        ).set_index("Target IRR")
        _table(ability, {f"Price per share ({currency})": "{:,.2f}"})
        st.caption(
            "Price a financial sponsor could pay today and still reach the target IRR. "
            f"Current price: {fmt_per_share(lbo.market_price, currency)}."
        )

    st.subheader(f"Debt schedule ({currency} m)")
    keep = [
        "cash_flow_available_for_debt_service",
        "total_interest",
        "senior_mandatory_amortization",
        "senior_cash_sweep",
        "senior_closing",
        "subordinated_closing",
        "revolver_closing",
        "cash_closing",
        "net_debt_closing",
        "net_debt_to_ebitda",
        "interest_coverage",
    ]
    _table(_statement_table(lbo.debt_schedule[keep], "E"))
    with st.expander(f"Operating forecast ({currency} m)"):
        _table(_statement_table(lbo.forecast, "E"))


def _monte_carlo_tab(report: ValuationReportLike, images: dict[str, bytes]) -> None:
    mc = report.monte_carlo
    if mc is None:
        _not_applicable(report, "monte_carlo", "The Monte Carlo simulation")
        return
    currency = mc.currency
    percentiles = {int(k): float(v) for k, v in mc.percentiles.items()}
    columns = st.columns(5)
    columns[0].metric("Median (P50)", fmt_per_share(percentiles.get(50), currency))
    columns[1].metric("P10", fmt_per_share(percentiles.get(10), currency))
    columns[2].metric("P90", fmt_per_share(percentiles.get(90), currency))
    columns[3].metric("Mean", fmt_per_share(mc.mean, currency))
    columns[4].metric("Probability above price", fmt_pct(mc.prob_above_price, 0))

    _chart(images, "mc_histogram")
    left, right = st.columns(2)
    with left:
        _chart(images, "mc_tornado")
    with right:
        _chart(images, "mc_variance_contribution")
    _chart(images, "mc_convergence")

    table = pd.DataFrame(
        {f"P{level}": [value] for level, value in sorted(percentiles.items())},
        index=[f"Value per share ({currency})"],
    )
    _table(table, "{:,.2f}")
    st.caption(
        f"{fmt_number(mc.n_simulations)} simulations of the DCF (Gordon growth terminal value). "
        f"Base case {fmt_per_share(mc.base_value, currency)}, standard error of the mean "
        f"{fmt_number(mc.standard_error, 3)}."
    )
    _bullets("Simulation notes", [str(n) for n in getattr(mc, "notes", [])])


def _data_tab(report: ValuationReportLike) -> None:
    company = report.company
    profile, market, statements = company.profile, company.market, company.statements
    currency = market.currency
    fx_text = data_layer.describe_statement_fx(company)
    if fx_text:
        st.caption(fx_text)
    for title, frame in (
        ("Income statement", statements.income),
        ("Balance sheet", statements.balance),
        ("Cash flow statement", statements.cash_flow),
    ):
        st.subheader(f"{title} ({currency} m)")
        if frame is None or frame.empty:
            st.write("No data.")
        else:
            _table(_statement_table(frame))

    history = company.price_history
    if history is not None and not history.prices.empty:
        st.subheader("Share price against the benchmark (indexed to 100)")
        indexed = history.prices[["stock", "benchmark"]].astype(float)
        indexed = indexed / indexed.iloc[0] * 100.0
        indexed.columns = [profile.ticker, history.benchmark]
        st.line_chart(indexed)
        st.caption(f"{history.frequency}-frequency adjusted closes; both series start at 100.")
    with st.expander("Market snapshot (raw)"):
        st.json({k: str(v) for k, v in dataclasses.asdict(market).items()})


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Valuation engine", page_icon=":bar_chart:", layout="wide")

    env_offline = data_layer.is_offline()
    with st.sidebar:
        st.title("Valuation engine")
        st.caption("Swedish listed companies · DCF, comps, LBO, Monte Carlo")
        offline = st.toggle(
            "Offline data",
            value=env_offline,
            disabled=env_offline,
            help="Read only the local fixtures and cache; no requests to Yahoo Finance."
            + (" Forced by VALUATION_ENGINE_OFFLINE." if env_offline else ""),
        )
    offline = bool(offline or env_offline)

    try:
        universe = _load_universe(offline)
    except DataError as exc:
        st.error(f"The company universe could not be loaded: {exc}")
        st.info(
            "Turn on **Offline data** in the sidebar to work from the local fixtures and cache."
        )
        st.stop()
    tickers = sorted(universe)
    if not tickers:
        st.error("The company universe is empty.")
        st.stop()

    with st.sidebar:
        ticker = st.selectbox(
            "Company",
            options=tickers,
            index=tickers.index(DEFAULT_TICKER) if DEFAULT_TICKER in tickers else 0,
            format_func=lambda t: f"{t} · {universe.get(t, '')}",
            help="Type to search by ticker or company name.",
            key="ticker",
        )

    try:
        company = _load_company(ticker, offline)
        _load_peers(offline)
    except DataError as exc:
        st.error(f"**No data for {ticker}.** {exc}")
        st.info(
            "Yahoo Finance has no annual statements for a few listed companies, and offline "
            "mode only knows the companies in the local fixtures and cache."
        )
        st.stop()

    with st.sidebar:
        overrides = _sidebar_overrides(company, offline)

    profile, market = company.profile, company.market
    currency = market.currency
    st.title(profile.name)
    st.caption(
        " · ".join(
            str(part)
            for part in (
                profile.ticker,
                profile.sector,
                profile.industry,
                f"market data as of {market.as_of}",
                "offline data" if offline else "live data",
            )
            if part
        )
    )
    columns = st.columns(5)
    columns[0].metric("Share price", fmt_per_share(market.price, currency))
    columns[1].metric("Market cap", fmt_money(market.market_cap, currency))
    columns[2].metric("Enterprise value", fmt_money(market.enterprise_value, currency))
    columns[3].metric("Net debt", fmt_money(market.net_debt, currency))
    columns[4].metric("Latest fiscal year", str(company.statements.latest_year))

    try:
        report = _cached_report(ticker, offline, overrides)
    except DataError as exc:
        st.error(f"**Data error for {ticker}.** {exc}")
        st.stop()
    except (NotApplicableError, ValueError) as exc:
        st.error(f"**The models could not be run with these assumptions.** {exc}")
        st.stop()

    try:
        images = _cached_charts(ticker, offline, overrides)
    except Exception as exc:  # noqa: BLE001 - a chart problem must not hide the numbers
        images = {}
        st.warning(f"Charts could not be drawn: {type(exc).__name__}: {exc}")
    try:
        excel: bytes | None = _cached_excel(ticker, offline, overrides)
    except Exception as exc:  # noqa: BLE001 - the export must not take the app down
        excel = None
        st.warning(f"The Excel export failed: {type(exc).__name__}: {exc}")

    st.caption(f"{display_name(report)} · all amounts in {currency}")
    tabs = st.tabs(["Overview", "DCF", "Comps", "LBO", "Monte Carlo", "Data"])
    with tabs[0]:
        _overview_tab(report, images, excel)
    with tabs[1]:
        _dcf_tab(report, images)
    with tabs[2]:
        _comps_tab(report, images)
    with tabs[3]:
        _lbo_tab(report, images)
    with tabs[4]:
        _monte_carlo_tab(report, images)
    with tabs[5]:
        _data_tab(report)


if __name__ == "__main__":
    main()
