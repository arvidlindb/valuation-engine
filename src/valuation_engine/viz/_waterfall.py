"""Waterfall (bridge) drawing shared by the DCF and LBO charts."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from matplotlib.axes import Axes
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from . import style

__all__ = ["Step", "draw_waterfall"]


@dataclass(frozen=True)
class Step:
    """One column of a waterfall: a running ``total`` or a signed ``delta``."""

    label: str
    value: float
    is_total: bool = False


def draw_waterfall(
    ax: Axes,
    steps: Sequence[Step],
    *,
    value_format: Callable[[float], str],
    axis_label: str = "",
    bar_width: float = 0.5,
) -> None:
    """Draw ``steps`` on ``ax``.

    Totals grow from zero in slate grey; deltas float from the running level,
    blue when they add and red when they subtract. Hairline connectors link
    consecutive columns, every column is labelled with its value, and a legend
    names the three colours.
    """
    style.style_axes(ax, grid="y")
    level = 0.0
    tops: list[float] = []
    extremes: list[float] = [0.0]
    geometry: list[tuple[float, float, str]] = []  # bottom, height, colour
    for step in steps:
        if step.is_total:
            bottom, height, colour = 0.0, step.value, style.NEUTRAL
            level = step.value
        else:
            bottom, height = level, step.value
            colour = style.POSITIVE if step.value >= 0 else style.NEGATIVE
            level += step.value
        geometry.append((bottom, height, colour))
        tops.append(level)
        extremes.extend([bottom, bottom + height])

    span = (max(extremes) - min(extremes)) or 1.0
    for x, (step, (bottom, height, colour)) in enumerate(zip(steps, geometry, strict=True)):
        ax.bar(x, height, bottom=bottom, width=bar_width, color=colour, linewidth=0, zorder=2)
        upper = max(bottom, bottom + height)
        lower = min(bottom, bottom + height)
        text = value_format(step.value) if step.is_total else _signed(step.value, value_format)
        if upper < 0:
            # Bar entirely below zero: label under it.
            ax.text(
                x,
                lower - 0.018 * span,
                text,
                ha="center",
                va="top",
                fontsize=style.SIZE_SMALL,
                color=style.INK,
            )
        else:
            ax.text(
                x,
                upper + 0.018 * span,
                text,
                ha="center",
                va="bottom",
                fontsize=style.SIZE_SMALL,
                color=style.INK,
            )
        if x < len(steps) - 1:
            ax.plot(
                [x + bar_width / 2, x + 1 - bar_width / 2],
                [tops[x], tops[x]],
                color=style.AXIS,
                linewidth=0.8,
                zorder=1,
            )

    ax.axhline(0, color=style.AXIS, linewidth=0.8, zorder=1)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([s.label for s in steps])
    ax.tick_params(axis="x", length=0, colors=style.INK, labelsize=style.SIZE_SMALL)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim(-0.6, len(steps) - 0.4)
    low, high = min(extremes), max(extremes)
    ax.set_ylim(low - (0.10 * span if low < 0 else 0.0), high + 0.12 * span)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: style.fmt_number(v, 0)))
    ax.set_ylabel(axis_label)

    handles = [Patch(facecolor=style.NEUTRAL, label="Total")]
    if any(not s.is_total and s.value >= 0 for s in steps):
        handles.append(Patch(facecolor=style.POSITIVE, label="Increase"))
    if any(not s.is_total and s.value < 0 for s in steps):
        handles.append(Patch(facecolor=style.NEGATIVE, label="Decrease"))
    style.style_legend(
        ax, handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncols=len(handles)
    )


def _signed(value: float, value_format: Callable[[float], str]) -> str:
    text = value_format(value)
    return text if value <= 0 or text.startswith("+") else "+" + text
