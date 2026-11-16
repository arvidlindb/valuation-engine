"""DCF charts: FCFF forecast and the enterprise-to-equity value bridge."""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from valuation_engine.models.dcf import DCFResult

from . import style
from ._waterfall import Step, draw_waterfall

__all__ = ["dcf_bridge_waterfall", "dcf_fcff_forecast"]


def dcf_fcff_forecast(dcf: DCFResult) -> Figure:
    """Free cash flow to the firm per forecast year, undiscounted and discounted."""
    forecast = dcf.forecast
    years = [int(y) for y in forecast.index]
    fcff = forecast["fcff"].to_numpy(float)
    present = forecast["pv_fcff"].to_numpy(float)
    scale = style.money_scale(np.concatenate([fcff, present]))
    unit = style.unit_label(dcf.currency, scale)
    convention = "mid-year" if dcf.assumptions.mid_year else "year-end"
    fig, ax = style.new_figure(
        "DCF: free cash flow forecast",
        f"{dcf.ticker} · FCFF, {unit} · discounted at a WACC of "
        f"{style.fmt_pct(dcf.wacc, 1)} ({convention} convention)",
        figsize=(10.0, 4.9),
        footnote=(
            f"Sum of discounted FCFF: {style.fmt_money(dcf.pv_fcff, dcf.currency)}. "
            "Margins and growth per year are listed in the forecast table."
        ),
    )
    positions = np.arange(len(years), dtype=float)
    width = 0.26
    gap = 0.03
    ax.bar(
        positions - width / 2 - gap / 2,
        fcff / scale[0],
        width=width,
        color=style.BLUE,
        linewidth=0,
        label="FCFF",
        zorder=2,
    )
    ax.bar(
        positions + width / 2 + gap / 2,
        present / scale[0],
        width=width,
        color=style.NEUTRAL_LIGHT,
        linewidth=0,
        label="Present value of FCFF",
        zorder=2,
    )
    peak = float(np.nanmax(np.abs(np.concatenate([fcff, present])))) / scale[0] or 1.0
    for x, value in zip(positions, fcff / scale[0], strict=True):
        ax.text(
            x - width / 2 - gap / 2,
            value + (0.015 * peak if value >= 0 else -0.015 * peak),
            style.fmt_number(value, 1),
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=style.SIZE_SMALL,
            color=style.INK,
        )
    ax.axhline(0, color=style.AXIS, linewidth=0.8, zorder=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [
            f"{year}E\n{style.fmt_pct(growth, 1, sign=True)} growth"
            for year, growth in zip(years, forecast["revenue_growth"], strict=True)
        ]
    )
    ax.tick_params(axis="x", length=0, colors=style.INK)
    ax.spines["bottom"].set_visible(False)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: style.fmt_number(v, 0)))
    ax.set_ylabel(unit)
    low, high = ax.get_ylim()
    ax.set_ylim(low, high + 0.08 * (high - low))
    style.style_legend(ax, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncols=2)
    return style.finalize(fig)


def dcf_bridge_waterfall(dcf: DCFResult, method: str = "gordon") -> Figure:
    """Bridge from the present values to equity value for one terminal method.

    ``method`` is ``"gordon"`` or ``"exit"``. Bridge items that are zero are
    left out.
    """
    if method not in ("gordon", "exit"):
        raise ValueError("method must be 'gordon' or 'exit'")
    if method == "gordon":
        pv_terminal, enterprise, equity = (
            dcf.pv_terminal_value_gordon,
            dcf.enterprise_value_gordon,
            dcf.equity_value_gordon,
        )
        per_share, share_terminal = dcf.value_per_share_gordon, dcf.tv_share_gordon
        name = f"Gordon growth, g = {style.fmt_pct(dcf.assumptions.terminal_growth, 1)}"
    else:
        pv_terminal, enterprise, equity = (
            dcf.pv_terminal_value_exit,
            dcf.enterprise_value_exit,
            dcf.equity_value_exit,
        )
        per_share, share_terminal = dcf.value_per_share_exit, dcf.tv_share_exit
        name = f"exit multiple {style.fmt_multiple(dcf.exit_multiple, 1)}"

    scale = style.money_scale([dcf.pv_fcff, pv_terminal, enterprise, equity, dcf.net_debt])
    unit = style.unit_label(dcf.currency, scale)
    steps = [
        Step("PV of FCFF", dcf.pv_fcff / scale[0], is_total=True),
        Step("PV of terminal\nvalue", pv_terminal / scale[0]),
        Step("Enterprise\nvalue", enterprise / scale[0], is_total=True),
    ]
    net_label = "Net debt" if dcf.net_debt >= 0 else "Net cash"
    for label, amount in (
        (net_label, dcf.net_debt),
        ("Minority\ninterest", dcf.minority_interest),
        ("Preferred\nequity", dcf.preferred_equity),
    ):
        if abs(amount) > 0:
            steps.append(Step(label, -amount / scale[0]))
    steps.append(Step("Equity value", equity / scale[0], is_total=True))

    decimals = 1 if max(abs(s.value) for s in steps) < 1000 else 0
    fig, ax = style.new_figure(
        "DCF: from enterprise value to equity value",
        f"{dcf.ticker} · {name} · {unit}",
        figsize=(10.0, 4.9),
        footnote=(
            f"Equity value per share: {style.fmt_per_share(per_share, dcf.currency, 2)} "
            f"({style.fmt_number(dcf.shares_outstanding / 1e6, 1)}m shares) vs. current price "
            f"{style.fmt_per_share(dcf.price, dcf.currency, 2)}. "
            f"Terminal value is {style.fmt_pct(share_terminal, 0)} of enterprise value."
        ),
    )
    draw_waterfall(
        ax,
        steps,
        value_format=lambda v: style.fmt_number(v, decimals),
        axis_label=unit,
    )
    return style.finalize(fig)
