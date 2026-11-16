"""Chart set for one valuation report (any :class:`ValuationReportLike`)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from matplotlib.figure import Figure

from valuation_engine.core.types import ValuationMethod, ValuationResult

from . import comps_charts, dcf_charts, lbo_charts, monte_carlo_charts, sensitivity
from .football_field import football_field, order_valuations, trading_range_result
from .protocols import ValuationReportLike
from .style import save_figure

__all__ = [
    "CHART_NAMES",
    "build_charts",
    "display_name",
    "report_football_field",
    "report_valuations",
    "save_charts",
]

#: Keys of :func:`build_charts`, in presentation order.
CHART_NAMES: tuple[str, ...] = (
    "football_field",
    "dcf_fcff_forecast",
    "dcf_bridge_gordon",
    "dcf_bridge_exit",
    "dcf_sensitivity_wacc_growth",
    "dcf_sensitivity_wacc_exit_multiple",
    "comps_peer_multiples",
    "comps_silhouette",
    "lbo_debt_paydown",
    "lbo_returns_waterfall",
    "lbo_sensitivity_entry_exit",
    "lbo_sensitivity_leverage_exit",
    "mc_histogram",
    "mc_tornado",
    "mc_variance_contribution",
    "mc_convergence",
)


def display_name(report: ValuationReportLike) -> str:
    """Company name without legal suffix noise, e.g. ``Atlas Copco AB``."""
    return report.company.profile.name.replace(" (publ)", "").replace(" (publ.)", "").strip()


def report_valuations(report: ValuationReportLike) -> list[ValuationResult]:
    """Football-field rows of the report in display order.

    The 52-week trading range is appended when the report does not already
    carry a ``ValuationMethod.MARKET`` row and a price history exists.
    """
    rows = list(report.valuations)
    if not any(row.method == ValuationMethod.MARKET for row in rows):
        market = report.company.market
        trading_range = trading_range_result(
            report.company.price_history, float(market.price), market.currency
        )
        if trading_range is not None:
            rows.append(trading_range)
    return order_valuations(rows)


def report_football_field(report: ValuationReportLike) -> Figure:
    """Football field of the report, titled with the company and the date."""
    market = report.company.market
    names = {"dcf": "DCF", "comps": "Comps", "lbo": "LBO", "monte_carlo": "Monte Carlo"}
    footnote = None
    if report.not_applicable:
        footnote = "\n".join(
            _shorten(f"{names.get(key, key)} not applicable: {reason}")
            for key, reason in report.not_applicable.items()
        )
    return football_field(
        report_valuations(report),
        float(market.price),
        currency=market.currency,
        title=f"{display_name(report)}: valuation summary",
        subtitle=(
            f"{report.company.ticker} · value per share, {market.currency} "
            f"· market data as of {market.as_of}"
        ),
        footnote=footnote,
    )


def _shorten(text: str, limit: int = 150) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def build_charts(report: ValuationReportLike) -> dict[str, Figure]:
    """Every chart that applies to ``report``, keyed by the names in
    ``CHART_NAMES``. Charts of models that are not applicable are left out."""
    builders: dict[str, Callable[[], Figure]] = {
        "football_field": lambda: report_football_field(report)
    }
    dcf, comps, lbo, monte_carlo = report.dcf, report.comps, report.lbo, report.monte_carlo
    if dcf is not None:
        builders.update(
            {
                "dcf_fcff_forecast": lambda: dcf_charts.dcf_fcff_forecast(dcf),
                "dcf_bridge_gordon": lambda: dcf_charts.dcf_bridge_waterfall(dcf, "gordon"),
                "dcf_bridge_exit": lambda: dcf_charts.dcf_bridge_waterfall(dcf, "exit"),
                "dcf_sensitivity_wacc_growth": lambda: sensitivity.dcf_sensitivity_heatmap(
                    dcf, "gordon"
                ),
                "dcf_sensitivity_wacc_exit_multiple": lambda: sensitivity.dcf_sensitivity_heatmap(
                    dcf, "exit"
                ),
            }
        )
    if comps is not None:
        builders.update(
            {
                "comps_peer_multiples": lambda: comps_charts.comps_peer_multiples(comps),
                "comps_silhouette": lambda: comps_charts.comps_silhouette(comps),
            }
        )
    if lbo is not None:
        builders.update(
            {
                "lbo_debt_paydown": lambda: lbo_charts.lbo_debt_paydown(lbo),
                "lbo_returns_waterfall": lambda: lbo_charts.lbo_returns_waterfall(lbo),
                "lbo_sensitivity_entry_exit": lambda: sensitivity.lbo_sensitivity_heatmap(
                    lbo, "entry_exit"
                ),
                "lbo_sensitivity_leverage_exit": lambda: sensitivity.lbo_sensitivity_heatmap(
                    lbo, "leverage_exit"
                ),
            }
        )
    if monte_carlo is not None:
        builders.update(
            {
                "mc_histogram": lambda: monte_carlo_charts.mc_histogram(monte_carlo),
                "mc_tornado": lambda: monte_carlo_charts.mc_tornado(monte_carlo),
                "mc_variance_contribution": lambda: monte_carlo_charts.mc_variance_contribution(
                    monte_carlo
                ),
                "mc_convergence": lambda: monte_carlo_charts.mc_convergence(monte_carlo),
            }
        )
    return {name: builders[name]() for name in CHART_NAMES if name in builders}


def save_charts(
    report: ValuationReportLike,
    out_dir: str | Path,
    formats: Sequence[str] = ("png", "svg"),
) -> dict[str, list[Path]]:
    """Render :func:`build_charts` and save each chart as
    ``<out_dir>/<name>.<format>``. Returns the written paths per chart."""
    folder = Path(out_dir)
    return {
        name: save_figure(figure, folder / name, formats)
        for name, figure in build_charts(report).items()
    }
