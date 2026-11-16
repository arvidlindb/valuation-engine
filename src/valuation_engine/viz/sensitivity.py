"""Sensitivity heatmaps for the DCF (value per share) and the LBO (IRR)."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from valuation_engine.models.dcf import DCFResult
from valuation_engine.models.lbo import LBOResult

from . import style

__all__ = ["dcf_sensitivity_heatmap", "lbo_sensitivity_heatmap", "sensitivity_heatmap"]

Formatter = Callable[[Any], str]


def _nearest(values: Any, target: float) -> int | None:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not math.isfinite(target):
        return None
    return int(np.nanargmin(np.abs(array - target)))


def _percent_1(value: Any) -> str:
    return style.fmt_pct(value, 1)


def _multiple_1(value: Any) -> str:
    return style.fmt_multiple(value, 1)


def sensitivity_heatmap(
    grid: pd.DataFrame,
    *,
    reference: float,
    title: str,
    subtitle: str | None = None,
    row_label: str = "",
    column_label: str = "",
    row_format: Formatter = str,
    column_format: Formatter = str,
    cell_format: Formatter = str,
    base_cell: tuple[int, int] | None = None,
    footnote: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Annotated heatmap of ``grid`` with a diverging colour around ``reference``.

    Cells above ``reference`` are blue, cells below are red and cells at the
    reference are neutral grey; the scale is symmetric, so equal distances get
    equal colour. ``base_cell`` (row position, column position) is outlined.
    Missing cells (infeasible combinations) are left pale and print ``n/a``.
    """
    values = grid.to_numpy(dtype=float)
    n_rows, n_cols = values.shape
    if figsize is None:
        figsize = (max(6.4, 1.9 + 1.12 * n_cols), 2.25 + 0.5 * n_rows)
    fig, ax = style.new_figure(title, subtitle, figsize=figsize, footnote=footnote)
    style.style_axes(ax, grid=None)

    finite = values[np.isfinite(values)]
    centre = (
        reference if math.isfinite(reference) else float(np.median(finite)) if finite.size else 0
    )
    reach = float(np.max(np.abs(finite - centre))) if finite.size else 1.0
    reach = reach or 1.0
    norm = Normalize(vmin=centre - reach, vmax=centre + reach)
    cmap = style.DIVERGING_CMAP
    masked = np.ma.masked_invalid(values)
    ax.pcolormesh(
        np.arange(n_cols + 1),
        np.arange(n_rows + 1),
        masked,
        cmap=cmap,
        norm=norm,
        edgecolors=style.SURFACE,
        linewidth=2.5,
    )
    # pcolormesh leaves masked cells empty; paint them with the "bad" colour.
    for i in range(n_rows):
        for j in range(n_cols):
            value = values[i, j]
            if math.isfinite(value):
                fill = cmap(norm(value))
                text = cell_format(value)
            else:
                fill = cmap.get_bad()
                ax.add_patch(
                    Rectangle((j, i), 1, 1, facecolor=fill, edgecolor=style.SURFACE, linewidth=2.5)
                )
                text = "n/a"
            is_base = base_cell == (i, j)
            ax.text(
                j + 0.5,
                i + 0.5,
                text,
                ha="center",
                va="center",
                fontsize=style.SIZE_BASE,
                fontweight="bold" if is_base else "normal",
                color=style.text_on(fill) if math.isfinite(value) else style.MUTED,
            )
    if base_cell is not None and 0 <= base_cell[0] < n_rows and 0 <= base_cell[1] < n_cols:
        ax.add_patch(
            Rectangle(
                (base_cell[1] + 0.03, base_cell[0] + 0.05),
                0.94,
                0.90,
                fill=False,
                edgecolor=style.INK,
                linewidth=1.6,
                zorder=5,
            )
        )

    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels([column_format(c) for c in grid.columns])
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels([row_format(r) for r in grid.index])
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="both", length=0, colors=style.INK, labelsize=style.SIZE_BASE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel(column_label, labelpad=7)
    ax.set_ylabel(row_label, labelpad=7)
    return style.finalize(fig)


def dcf_sensitivity_heatmap(dcf: DCFResult, kind: str = "gordon") -> Figure:
    """DCF value per share: WACC x terminal growth (``kind="gordon"``) or WACC x
    exit multiple (``kind="exit"``), coloured around the current share price."""
    if kind not in ("gordon", "exit"):
        raise ValueError("kind must be 'gordon' or 'exit'")
    decimals = style.per_share_decimals(dcf.price)
    price_text = style.fmt_per_share(dcf.price, dcf.currency, 2)
    footnote = (
        f"Blue: above the current price ({price_text}); red: below. Outlined cell: base case."
    )
    if kind == "gordon":
        grid = dcf.sensitivity_gordon
        base = (
            _nearest(grid.index, dcf.wacc),
            _nearest(grid.columns, dcf.assumptions.terminal_growth),
        )
        title = "DCF sensitivity: WACC and terminal growth"
        column_label = "Terminal growth"
        column_format: Formatter = _percent_1
    else:
        grid = dcf.sensitivity_exit
        base = (_nearest(grid.index, dcf.wacc), _nearest(grid.columns, dcf.exit_multiple))
        title = "DCF sensitivity: WACC and exit multiple"
        column_label = "Exit EV/EBITDA multiple"
        column_format = _multiple_1
    base_cell = None if base[0] is None or base[1] is None else (base[0], base[1])
    return sensitivity_heatmap(
        grid,
        reference=dcf.price,
        title=title,
        subtitle=f"{dcf.ticker} · value per share, {dcf.currency}",
        row_label="WACC",
        column_label=column_label,
        row_format=lambda v: style.fmt_pct(v, 2),
        column_format=column_format,
        cell_format=lambda v: style.fmt_number(v, decimals),
        base_cell=base_cell,
        footnote=footnote,
    )


def lbo_sensitivity_heatmap(lbo: LBOResult, kind: str = "entry_exit") -> Figure:
    """Sponsor IRR: entry x exit multiple (``kind="entry_exit"``) or requested
    leverage x exit multiple (``kind="leverage_exit"``), coloured around the
    base target IRR."""
    if kind not in ("entry_exit", "leverage_exit"):
        raise ValueError("kind must be 'entry_exit' or 'leverage_exit'")
    target = lbo.assumptions.target_irr_base
    footnote = (
        f"Blue: above the {style.fmt_pct(target, 0)} target IRR; red: below. "
        "Outlined cell: base case. n/a: deal not feasible."
    )
    if kind == "entry_exit":
        grid = lbo.sensitivity_irr_entry_exit
        base_row = _nearest(grid.index, lbo.entry_multiple)
        title = "LBO sensitivity: IRR by entry and exit multiple"
        row_label = "Entry EV/EBITDA multiple"
    else:
        grid = lbo.sensitivity_irr_leverage_exit
        requested = (
            lbo.assumptions.senior_debt_multiple + lbo.assumptions.subordinated_debt_multiple
        )
        base_row = _nearest(grid.index, requested)
        title = "LBO sensitivity: IRR by leverage and exit multiple"
        row_label = "Requested debt / EBITDA"
    base_col = _nearest(grid.columns, lbo.exit_multiple)
    base_cell = None if base_row is None or base_col is None else (base_row, base_col)
    return sensitivity_heatmap(
        grid,
        reference=target,
        title=title,
        subtitle=f"{lbo.ticker} · sponsor IRR over {lbo.assumptions.holding_period} years",
        row_label=row_label,
        column_label="Exit EV/EBITDA multiple",
        row_format=_multiple_1,
        column_format=_multiple_1,
        cell_format=lambda v: style.fmt_pct(v, 1),
        base_cell=base_cell,
        footnote=footnote,
    )
