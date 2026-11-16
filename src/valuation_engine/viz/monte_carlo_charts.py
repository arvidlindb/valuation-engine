"""Monte Carlo charts: distribution, tornado, variance contribution, convergence.

The functions take any object with the ``MonteCarloResult`` fields (see
:class:`valuation_engine.viz.protocols.MonteCarloLike`); the Monte Carlo module
itself is never imported here.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from . import style
from .protocols import MonteCarloLike, percentile

__all__ = [
    "driver_label",
    "mc_convergence",
    "mc_histogram",
    "mc_tornado",
    "mc_variance_contribution",
]

_DRIVER_LABELS: dict[str, str] = {
    "revenue_growth": "Revenue growth",
    "ebitda_margin": "EBITDA margin",
    "wacc": "WACC",
    "terminal_growth": "Terminal growth",
    "exit_multiple": "Exit multiple",
    "capex_pct_revenue": "Capex / revenue",
    "nwc_pct_revenue": "NWC / revenue",
    "tax_rate": "Tax rate",
}
_MULTIPLE_DRIVERS = frozenset({"exit_multiple"})


def driver_label(name: Any) -> str:
    """Readable name of a simulated driver (``wacc`` -> ``WACC``)."""
    key = str(name)
    return _DRIVER_LABELS.get(key, key.replace("_", " ").capitalize())


def _driver_value(name: Any, value: float) -> str:
    if str(name) in _MULTIPLE_DRIVERS:
        return style.fmt_multiple(value, 1)
    return style.fmt_pct(value, 1)


def _finite(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    return array[np.isfinite(array)]


def mc_histogram(
    monte_carlo: MonteCarloLike, *, exit_values: bool = False, bins: int = 60
) -> Figure:
    """Distribution of the simulated value per share with P10 / P50 / P90, the
    base case and the current price marked.

    ``exit_values=True`` plots ``values_exit`` (exit multiple terminal value)
    instead of ``values`` (Gordon growth). The x-axis covers the central 98 %
    of the simulations so that a few extreme draws do not flatten the picture.
    """
    currency = monte_carlo.currency
    values = _finite(monte_carlo.values_exit if exit_values else monte_carlo.values)
    title = "Monte Carlo: distribution of value per share"
    if values.size < 2:
        return style.message_figure(title, "No finite simulation results to plot.")
    price = float(monte_carlo.price)
    decimals = style.per_share_decimals(price)
    if exit_values:
        p10, p50, p90 = (float(np.percentile(values, q)) for q in (10, 50, 90))
        base = math.nan
        probability = float(np.mean(values > price))
    else:
        p10, p50, p90 = (percentile(monte_carlo, q) for q in (10, 50, 90))
        base = float(monte_carlo.base_value)
        probability = float(monte_carlo.prob_above_price)

    low, high = (float(np.percentile(values, q)) for q in (1, 99))
    marks = [m for m in (price, base) if math.isfinite(m)]
    # Keep the price and the base case in view unless they are far outside.
    spread = (high - low) or max(abs(high), 1.0)
    for mark in marks:
        if low - 1.5 * spread <= mark <= high + 1.5 * spread:
            low, high = min(low, mark), max(high, mark)
    pad = 0.04 * ((high - low) or 1.0)
    low, high = low - pad, high + pad
    shown = values[(values >= low) & (values <= high)]
    outside = 1.0 - shown.size / values.size

    lines: list[tuple[float, str, dict[str, Any]]] = [
        (p10, f"P10 {style.fmt_number(p10, decimals)}", {"color": style.MUTED, "linewidth": 0.9}),
        (p50, f"P50 {style.fmt_number(p50, decimals)}", {"color": style.MUTED, "linewidth": 0.9}),
        (p90, f"P90 {style.fmt_number(p90, decimals)}", {"color": style.MUTED, "linewidth": 0.9}),
        (price, f"Price {style.fmt_number(price, 2)}", {"color": style.INK, "linewidth": 1.4}),
    ]
    if math.isfinite(base):
        lines.append(
            (
                base,
                f"Base case {style.fmt_number(base, decimals)}",
                {"color": style.ORANGE, "linewidth": 1.4},
            )
        )
    lines = [line for line in lines if low <= line[0] <= high]

    terminal = "exit multiple" if exit_values else "Gordon growth"
    footnote = (
        f"Mean {style.fmt_number(float(np.mean(values)), decimals)}, standard deviation "
        f"{style.fmt_number(float(np.std(values, ddof=1)), decimals)}. "
    )
    if outside > 0.0005:
        footnote += f"{style.fmt_pct(outside, 1)} of the simulations lie outside the axis range."
    fig, ax = style.new_figure(
        title,
        f"{monte_carlo.ticker} · {style.fmt_number(values.size)} simulations · DCF with "
        f"{terminal} terminal value, {currency} per share · "
        f"{style.fmt_pct(probability, 0)} of outcomes above the current price",
        figsize=(10.0, 5.0),
        footnote=footnote.strip(),
        reference_rows=3,
    )
    counts, edges = np.histogram(shown, bins=bins, range=(low, high))
    share = counts / values.size
    ax.bar(
        edges[:-1],
        share,
        width=np.diff(edges) * 0.9,
        align="edge",
        color=style.BLUE,
        alpha=0.85,
        linewidth=0,
        zorder=2,
    )
    ax.set_xlim(low, high)
    ax.set_ylim(0, float(share.max()) * 1.08 if share.max() > 0 else 1.0)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: style.fmt_number(v, 0)))
    ax.yaxis.set_major_formatter(style.percent_formatter(ax.yaxis))
    ax.set_xlabel(f"Value per share, {currency}")
    ax.set_ylabel("Share of simulations")
    style.add_reference_lines(ax, lines)
    return style.finalize(fig)


def mc_tornado(monte_carlo: MonteCarloLike) -> Figure:
    """Tornado chart: value per share when one driver moves to its low and to
    its high input value while the others stay at the base case."""
    title = "Monte Carlo: sensitivity to single drivers"
    tornado = monte_carlo.tornado
    if tornado is None or len(tornado) == 0:
        return style.message_figure(title, "No tornado data available.")
    table = tornado.copy()
    if "swing" not in table.columns:
        table["swing"] = (table["value_high"] - table["value_low"]).abs()
    table = table.sort_values("swing", ascending=False)
    base = float(monte_carlo.base_value)
    currency = monte_carlo.currency
    decimals = style.per_share_decimals(monte_carlo.price)

    fig, ax = style.new_figure(
        title,
        f"{monte_carlo.ticker} · value per share ({currency}) with one driver at its low or "
        "high input and all others at the base case",
        figsize=(10.0, 1.9 + 0.55 * len(table)),
        footnote="Labels show the value per share and, in brackets, the input value used.",
        reference_rows=1,
    )
    style.style_axes(ax, grid="x")
    ends = np.concatenate(
        [table["value_low"].to_numpy(float), table["value_high"].to_numpy(float), [base]]
    )
    ends = ends[np.isfinite(ends)]
    span = float(ends.max() - ends.min()) or max(abs(base), 1.0)
    ax.set_xlim(float(ends.min()) - 0.30 * span, float(ends.max()) + 0.30 * span)
    gap = 0.012 * span

    for y, (name, row) in enumerate(table.iterrows()):
        for column, input_column, colour in (
            ("value_low", "input_low", style.NEUTRAL),
            ("value_high", "input_high", style.BLUE),
        ):
            value = float(row[column])
            if not math.isfinite(value):
                continue
            ax.barh(y, value - base, left=base, height=0.5, color=colour, linewidth=0, zorder=2)
            right = value >= base
            ax.text(
                value + (gap if right else -gap),
                y,
                f"{style.fmt_number(value, decimals)} "
                f"({_driver_value(name, float(row[input_column]))})",
                ha="left" if right else "right",
                va="center",
                fontsize=style.SIZE_SMALL,
                color=style.SECONDARY,
            )
    ax.set_yticks(range(len(table)))
    ax.set_yticklabels([driver_label(name) for name in table.index])
    ax.tick_params(axis="y", length=0, colors=style.INK)
    ax.set_ylim(len(table) - 0.4, -0.6)
    ax.spines["left"].set_visible(False)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: style.fmt_number(v, 0)))
    ax.set_xlabel(f"Value per share, {currency}")
    style.style_legend(
        ax,
        handles=[
            Patch(facecolor=style.NEUTRAL, label="Low input"),
            Patch(facecolor=style.BLUE, label="High input"),
        ],
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        ncols=2,
    )
    style.add_reference_lines(
        ax, [(base, f"Base case {style.fmt_number(base, decimals)}", {"linewidth": 1.0})]
    )
    return style.finalize(fig)


def mc_variance_contribution(monte_carlo: MonteCarloLike) -> Figure:
    """Share of the variance of the simulated value explained by each driver."""
    title = "Monte Carlo: contribution to variance"
    contribution = monte_carlo.variance_contribution
    if contribution is None or len(contribution) == 0:
        return style.message_figure(title, "No variance decomposition available.")
    series = contribution.astype(float).dropna().sort_values(ascending=False)
    fig, ax = style.new_figure(
        title,
        f"{monte_carlo.ticker} · share of the variance of value per share attributed to "
        "each simulated driver",
        figsize=(8.6, 1.9 + 0.5 * len(series)),
        footnote=(
            f"Shares sum to {style.fmt_pct(float(series.sum()), 0)}; the remainder is due to "
            "interactions between drivers and non-linear effects."
            if abs(float(series.sum()) - 1.0) > 0.005
            else None
        ),
    )
    style.style_axes(ax, grid="x")
    positions = np.arange(len(series))
    ax.barh(positions, series.to_numpy(float), height=0.5, color=style.BLUE, linewidth=0, zorder=2)
    peak = float(series.abs().max()) or 1.0
    for y, value in zip(positions, series.to_numpy(float), strict=True):
        ax.text(
            value + 0.012 * peak if value >= 0 else value - 0.012 * peak,
            y,
            style.fmt_pct(value, 1),
            ha="left" if value >= 0 else "right",
            va="center",
            fontsize=style.SIZE_SMALL,
            color=style.INK,
        )
    ax.set_yticks(positions)
    ax.set_yticklabels([driver_label(name) for name in series.index])
    ax.tick_params(axis="y", length=0, colors=style.INK)
    ax.set_ylim(len(series) - 0.4, -0.6)
    ax.set_xlim(min(float(series.min()), 0.0) * 1.15, peak * 1.14)
    ax.axvline(0, color=style.AXIS, linewidth=0.8, zorder=1)
    ax.spines["left"].set_visible(False)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: style.fmt_pct(v, 0)))
    ax.set_xlabel("Share of variance")
    return style.finalize(fig)


def mc_convergence(monte_carlo: MonteCarloLike) -> Figure:
    """Running mean of the simulated value with a 95 % confidence band
    (mean +/- 1.96 standard errors) against the number of simulations."""
    title = "Monte Carlo: convergence of the mean"
    table = monte_carlo.convergence
    if table is None or len(table) < 2:
        return style.message_figure(title, "No convergence data available.")
    n = table["n"].to_numpy(float)
    mean = table["running_mean"].to_numpy(float)
    error = table["standard_error"].to_numpy(float)
    lower, upper = mean - 1.96 * error, mean + 1.96 * error
    currency = monte_carlo.currency
    decimals = style.per_share_decimals(monte_carlo.price)
    final_mean, final_error = float(mean[-1]), float(error[-1])

    fig, ax = style.new_figure(
        title,
        f"{monte_carlo.ticker} · running mean of value per share ({currency}) with a 95% "
        "confidence band (±1.96 standard errors)",
        figsize=(10.0, 4.6),
        footnote=(
            f"After {style.fmt_number(n[-1])} simulations: mean "
            f"{style.fmt_number(final_mean, 2)} ± {style.fmt_number(1.96 * final_error, 2)} "
            f"(standard error {style.fmt_number(final_error, 3)})."
        ),
    )
    ax.fill_between(
        n, lower, upper, color=style.BLUE, alpha=style.WASH_ALPHA, linewidth=0, zorder=2
    )
    ax.plot(n, mean, color=style.BLUE, linewidth=1.8, solid_joinstyle="round", zorder=3)
    ax.axhline(final_mean, color=style.AXIS, linewidth=0.8, zorder=1)
    ax.plot(
        [n[-1]],
        [final_mean],
        marker="o",
        markersize=6,
        color=style.BLUE,
        markeredgecolor=style.SURFACE,
        markeredgewidth=1.4,
        zorder=4,
    )
    ax.annotate(
        style.fmt_number(final_mean, decimals),
        (n[-1], final_mean),
        xytext=(6, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=style.SIZE_SMALL,
        fontweight="bold",
        color=style.INK,
    )
    # Zoom on the part that matters: the band after the first 5% of the draws.
    settled = n >= 0.05 * n[-1]
    view_low = float(np.nanmin(lower[settled]))
    view_high = float(np.nanmax(upper[settled]))
    margin = 0.15 * ((view_high - view_low) or 1.0)
    ax.set_ylim(view_low - margin, view_high + margin)
    ax.set_xlim(0, float(n[-1]) * 1.06)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: style.fmt_number(v, 0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: style.fmt_number(v, decimals)))
    ax.set_xlabel("Number of simulations")
    ax.set_ylabel(f"Value per share, {currency}")
    return style.finalize(fig)
