"""LBO charts: debt paydown over the holding period and returns attribution."""

from __future__ import annotations

import math

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from valuation_engine.models.lbo import LBOResult

from . import style
from ._waterfall import Step, draw_waterfall

__all__ = ["lbo_debt_paydown", "lbo_returns_waterfall"]

_TRANCHES: tuple[tuple[str, str, str], ...] = (
    ("senior", "Senior term loan", style.BLUE),
    ("subordinated", "Subordinated debt", style.ORANGE),
    ("revolver", "Revolver", style.AQUA),
)


def lbo_debt_paydown(lbo: LBOResult) -> Figure:
    """Stacked debt balance by tranche at closing and at each year-end.

    The tick labels carry net debt / EBITDA so that deleveraging is readable
    without a second axis.
    """
    schedule = lbo.debt_schedule
    years = [int(y) for y in schedule.index]
    balances = {
        "senior": [lbo.senior_debt, *schedule["senior_closing"].to_numpy(float)],
        "subordinated": [lbo.subordinated_debt, *schedule["subordinated_closing"].to_numpy(float)],
        "revolver": [0.0, *schedule["revolver_closing"].to_numpy(float)],
    }
    totals = np.sum([np.asarray(v, dtype=float) for v in balances.values()], axis=0)
    scale = style.money_scale(totals)
    unit = style.unit_label(lbo.currency, scale)

    entry_cash = float(schedule["cash_opening"].iloc[0]) if len(schedule) else 0.0
    entry_leverage = (
        (lbo.senior_debt + lbo.subordinated_debt - entry_cash) / lbo.entry_ebitda
        if lbo.entry_ebitda > 0
        else math.nan
    )
    leverage = [entry_leverage, *schedule["net_debt_to_ebitda"].to_numpy(float)]
    labels = ["Closing", *[f"{year}E" for year in years]]

    footnote = "Second line of each label: net debt / EBITDA at that date."
    if lbo.liquidity_breach_years:
        footnote += (
            " Liquidity breach (cash below the minimum balance) in "
            + ", ".join(str(y) for y in lbo.liquidity_breach_years)
            + "."
        )
    fig, ax = style.new_figure(
        "LBO: debt paydown",
        f"{lbo.ticker} · debt outstanding by tranche, {unit} · entry leverage "
        f"{style.fmt_multiple(lbo.leverage, 1)} EBITDA",
        figsize=(10.0, 4.9),
        footnote=footnote,
    )
    positions = np.arange(len(labels), dtype=float)
    bottom = np.zeros(len(labels))
    for key, label, colour in _TRANCHES:
        heights = np.asarray(balances[key], dtype=float) / scale[0]
        if not np.any(heights > 0):
            continue
        ax.bar(
            positions,
            heights,
            bottom=bottom,
            width=0.5,
            color=colour,
            edgecolor=style.SURFACE,
            linewidth=1.5,
            label=label,
            zorder=2,
        )
        bottom = bottom + heights
    peak = float(np.max(bottom)) or 1.0
    for x, total in zip(positions, bottom, strict=True):
        ax.text(
            x,
            total + 0.015 * peak,
            style.fmt_number(total, 1 if peak < 1000 else 0),
            ha="center",
            va="bottom",
            fontsize=style.SIZE_SMALL,
            color=style.INK,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [
            f"{label}\n{style.fmt_multiple(lev, 1)}"
            for label, lev in zip(labels, leverage, strict=True)
        ]
    )
    ax.tick_params(axis="x", length=0, colors=style.INK)
    ax.set_ylim(0, peak * 1.12)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: style.fmt_number(v, 0)))
    ax.set_ylabel(unit)
    style.style_legend(ax, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncols=3)
    return style.finalize(fig)


def lbo_returns_waterfall(lbo: LBOResult) -> Figure:
    """Bridge from the sponsor's equity cheque to the exit equity value."""
    attribution = lbo.attribution["amount"]
    amounts = [lbo.sponsor_equity, lbo.exit_equity_value, *attribution.to_numpy(float)]
    scale = style.money_scale(amounts)
    unit = style.unit_label(lbo.currency, scale)
    labels = {
        "ebitda_growth": "EBITDA\ngrowth",
        "multiple_expansion": "Multiple\nchange",
        "debt_paydown": "Debt paydown\nand cash",
        "fees": "Fees",
        "equity_floor": "Equity floor\n(wipe-out)",
    }
    steps = [Step("Sponsor equity\nat entry", lbo.sponsor_equity / scale[0], is_total=True)]
    for key, label in labels.items():
        amount = float(attribution.get(key, 0.0))
        if key in ("ebitda_growth", "debt_paydown", "fees") or abs(amount) > 0:
            steps.append(Step(label, amount / scale[0]))
    steps.append(
        Step(f"Equity value\nat exit {lbo.exit_year}E", lbo.exit_equity_value / scale[0], True)
    )

    decimals = 1 if max(abs(s.value) for s in steps) < 1000 else 0
    fig, ax = style.new_figure(
        "LBO: sources of the sponsor's return",
        f"{lbo.ticker} · {unit} · IRR {style.fmt_pct(lbo.irr, 1)}, "
        f"MOIC {style.fmt_multiple(lbo.moic, 2)} over {lbo.assumptions.holding_period} years",
        figsize=(10.0, 4.9),
        footnote=(
            f"Entry at {style.fmt_per_share(lbo.entry_price_per_share, lbo.currency, 2)} per share "
            f"({style.fmt_multiple(lbo.entry_multiple, 1)} EBITDA), exit at "
            f"{style.fmt_multiple(lbo.exit_multiple, 1)} EBITDA. EBITDA growth is valued at the "
            "entry multiple, the multiple change on exit EBITDA."
        ),
    )
    draw_waterfall(
        ax,
        steps,
        value_format=lambda v: style.fmt_number(v, decimals),
        axis_label=unit,
    )
    return style.finalize(fig)
