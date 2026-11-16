"""Shared chart style: palette, fonts, number formats and figure scaffolding.

Every chart of the package is built through :func:`new_figure` and finished with
:func:`finalize`, so that all of them look like pages of the same deck: white
surface, a left-aligned title with a grey subtitle, hairline grid, thin spines,
one sans-serif face and a small, fixed colour vocabulary.

No pyplot is used anywhere. Figures are plain :class:`matplotlib.figure.Figure`
objects with an Agg canvas attached, so nothing leaks into pyplot's global state
and the module works in headless processes, notebooks and Streamlit alike.

Colour vocabulary (validated for colour-vision deficiency on a white surface):

* categorical, in fixed order: blue, orange, aqua, violet (``SERIES``);
* diverging: blue (above the reference) <-> red (below) around a neutral grey;
* totals / neutral marks: slate grey; text is always ink, never a series colour.
"""

from __future__ import annotations

import math
import textwrap
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter

__all__ = [
    "BLUE",
    "DIVERGING_CMAP",
    "FONT_FAMILY",
    "GRID",
    "INK",
    "MUTED",
    "NEGATIVE",
    "NEUTRAL",
    "ORANGE",
    "POSITIVE",
    "SECONDARY",
    "SERIES",
    "SURFACE",
    "add_reference_lines",
    "finalize",
    "fmt_money",
    "fmt_multiple",
    "fmt_number",
    "fmt_pct",
    "fmt_per_share",
    "message_figure",
    "money_scale",
    "multiple_formatter",
    "new_figure",
    "per_share_decimals",
    "percent_formatter",
    "reference_rows_needed",
    "style_legend",
    "unit_label",
    "save_figure",
    "style_axes",
    "text_on",
]

# --------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------

SURFACE = "#ffffff"
INK = "#0b0b0b"  # primary text, reference lines
SECONDARY = "#52514e"  # subtitles, value labels
MUTED = "#898781"  # axis labels, footnotes
GRID = "#e1e0d9"  # hairline grid
AXIS = "#c3c2b7"  # baseline / spines

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
VIOLET = "#6250d6"
#: Categorical colours in their fixed order (never cycled, never re-ranked).
SERIES: tuple[str, ...] = (BLUE, ORANGE, AQUA, VIOLET)

NEUTRAL = "#6f7780"  # totals in waterfalls, de-emphasised marks
NEUTRAL_LIGHT = "#c9cdd2"
POSITIVE = BLUE  # increases in waterfalls
NEGATIVE = "#d9534f"  # decreases in waterfalls
WASH_ALPHA = 0.12

#: Diverging map: red (below the reference) - neutral grey - blue (above).
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "valuation_diverging", ["#e2726e", "#f0c4c1", "#f0efec", "#bcd6f5", "#5598e7"]
).with_extremes(bad="#f4f4f2")

# --------------------------------------------------------------------------
# Fonts and sizes
# --------------------------------------------------------------------------

_FONT_CANDIDATES = ("Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans")


def _resolve_font() -> str:
    """First installed family of the candidate list (DejaVu Sans ships with
    matplotlib, so there is always a match and never a findfont warning)."""
    for family in _FONT_CANDIDATES:
        try:
            font_manager.findfont(family, fallback_to_default=False)
        except ValueError:
            continue
        return family
    return "DejaVu Sans"


FONT_FAMILY = _resolve_font()

SIZE_TITLE = 13.5
SIZE_SUBTITLE = 9.5
SIZE_BASE = 9.0
SIZE_SMALL = 8.0
SIZE_FOOTNOTE = 7.5

DEFAULT_DPI = 200

_MARGIN_IN = 0.28  # outer margin around everything, inches
_TITLE_BAND_IN = 0.36
_SUBTITLE_LINE_IN = 0.19
_FOOTNOTE_LINE_IN = 0.15
_REFERENCE_ROW_IN = 0.17

MINUS = "−"

# --------------------------------------------------------------------------
# Number formats
# --------------------------------------------------------------------------


def _is_missing(value: Any) -> bool:
    try:
        return value is None or not math.isfinite(float(value))
    except (TypeError, ValueError):
        return True


def fmt_number(value: Any, decimals: int = 0, *, sign: bool = False) -> str:
    """``1234.5`` -> ``1,235``; typographic minus; ``n/a`` for missing values."""
    if _is_missing(value):
        return "n/a"
    number = float(value)
    text = f"{abs(number):,.{decimals}f}"
    if float(text.replace(",", "")) == 0:
        return text
    if number < 0:
        return MINUS + text
    return ("+" + text) if sign else text


def per_share_decimals(reference: float) -> int:
    """Decimals for per-share values given a reference level (the price)."""
    if _is_missing(reference):
        return 1
    reference = abs(float(reference))
    if reference >= 1000:
        return 0
    if reference >= 20:
        return 1
    return 2


def fmt_per_share(value: Any, currency: str = "SEK", decimals: int | None = None) -> str:
    """Per-share amount with the currency code, e.g. ``SEK 192.25``."""
    if _is_missing(value):
        return "n/a"
    if decimals is None:
        decimals = 2
    return f"{currency} {fmt_number(value, decimals)}"


def fmt_pct(value: Any, decimals: int = 1, *, sign: bool = False) -> str:
    """``0.0812`` -> ``8.1%``."""
    if _is_missing(value):
        return "n/a"
    return fmt_number(float(value) * 100.0, decimals, sign=sign) + "%"


def fmt_multiple(value: Any, decimals: int = 1) -> str:
    """``12.34`` -> ``12.3x``."""
    if _is_missing(value):
        return "n/a"
    return fmt_number(value, decimals) + "x"


def money_scale(values: Iterable[Any]) -> tuple[float, str]:
    """Divisor and unit word for a set of absolute amounts.

    Returns ``(1e9, "bn")`` when the largest magnitude is at least 10 billion,
    ``(1e6, "m")`` from 10 million, ``(1e3, "k")`` from 10 thousand, otherwise
    ``(1.0, "")``.
    """
    finite = [abs(float(v)) for v in values if not _is_missing(v)]
    top = max(finite, default=0.0)
    if top >= 10e9:
        return 1e9, "bn"
    if top >= 10e6:
        return 1e6, "m"
    if top >= 10e3:
        return 1e3, "k"
    return 1.0, ""


def fmt_money(value: Any, currency: str = "SEK", scale: tuple[float, str] | None = None) -> str:
    """Absolute amount in a readable unit, e.g. ``SEK 938.3bn``."""
    if _is_missing(value):
        return "n/a"
    divisor, unit = scale if scale is not None else money_scale([value])
    scaled = float(value) / divisor
    decimals = 1 if abs(scaled) < 1000 and unit else 0
    return f"{currency} {fmt_number(scaled, decimals)}{unit}"


def unit_label(currency: str, scale: tuple[float, str]) -> str:
    """Axis unit such as ``SEK bn`` or ``SEK m``."""
    return f"{currency} {scale[1]}".strip()


def text_on(fill: Any) -> str:
    """Ink or white, whichever reads better on ``fill``."""
    red, green, blue = to_rgb(fill)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return INK if luminance > 0.5 else "#ffffff"


# --------------------------------------------------------------------------
# Figure scaffolding
# --------------------------------------------------------------------------


def new_figure(
    title: str,
    subtitle: str | None = None,
    *,
    figsize: tuple[float, float] = (10.0, 5.6),
    nrows: int = 1,
    ncols: int = 1,
    footnote: str | None = None,
    reference_rows: int = 0,
    **subplot_kwargs: Any,
) -> tuple[Figure, Any]:
    """Create a styled figure with a header band and return ``(fig, axes)``.

    ``axes`` is a single :class:`Axes` for a 1 x 1 grid, otherwise an array.
    ``reference_rows`` reserves room between the header and the plot for the
    labels of vertical reference lines (see :func:`add_reference_lines`).
    The layout engine keeps everything inside the figure, so labels are not
    clipped when a figure is saved or shown in Streamlit.
    """
    fig = Figure(figsize=figsize, dpi=100, facecolor=SURFACE, layout="constrained")
    FigureCanvasAgg(fig)
    width, height = figsize
    usable_pt = (width - 2.0 * _MARGIN_IN) * 72.0
    if subtitle:
        subtitle = _wrap(subtitle, int(usable_pt / (0.50 * SIZE_SUBTITLE)))
    if footnote:
        footnote = _wrap(footnote, int(usable_pt / (0.50 * SIZE_FOOTNOTE)))
    left = _MARGIN_IN / width

    subtitle_lines = subtitle.count("\n") + 1 if subtitle else 0
    footnote_lines = footnote.count("\n") + 1 if footnote else 0
    header_in = _MARGIN_IN + _TITLE_BAND_IN + subtitle_lines * _SUBTITLE_LINE_IN
    header_in += reference_rows * _REFERENCE_ROW_IN + 0.06
    footer_in = _MARGIN_IN + (footnote_lines * _FOOTNOTE_LINE_IN + 0.08 if footnote else 0.0)

    fig.text(
        left,
        1.0 - (_MARGIN_IN + 0.02) / height,
        title,
        ha="left",
        va="top",
        fontsize=SIZE_TITLE,
        fontweight="bold",
        color=INK,
    )
    if subtitle:
        fig.text(
            left,
            1.0 - (_MARGIN_IN + _TITLE_BAND_IN) / height,
            subtitle,
            ha="left",
            va="top",
            fontsize=SIZE_SUBTITLE,
            color=SECONDARY,
            linespacing=1.35,
        )
    if footnote:
        fig.text(
            left,
            _MARGIN_IN / height,
            footnote,
            ha="left",
            va="bottom",
            fontsize=SIZE_FOOTNOTE,
            color=MUTED,
            linespacing=1.35,
        )

    engine = fig.get_layout_engine()
    engine.set(
        rect=(
            _MARGIN_IN / width,
            footer_in / height,
            1.0 - 2.0 * _MARGIN_IN / width,
            1.0 - (header_in + footer_in) / height,
        ),
        w_pad=0.06,
        h_pad=0.08,
    )
    axes = fig.subplots(nrows, ncols, **subplot_kwargs)
    for ax in np.atleast_1d(axes).ravel():
        style_axes(ax)
    return fig, axes


def _wrap(text: str, width: int) -> str:
    """Wrap every line of ``text`` to at most ``width`` characters."""
    lines: list[str] = []
    for line in text.split("\n"):
        lines.extend(textwrap.wrap(line, max(width, 20)) or [""])
    return "\n".join(lines)


def style_axes(ax: Axes, grid: str | None = "y") -> None:
    """Thin spines, hairline grid on one axis (``"x"``, ``"y"`` or None)."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(
        axis="both",
        colors=SECONDARY,
        labelsize=SIZE_BASE,
        length=3,
        width=0.8,
        color=AXIS,
        labelfontfamily=FONT_FAMILY,
    )
    ax.grid(False)
    if grid in ("x", "y"):
        ax.grid(True, axis=grid, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        # The grid replaces the spine it runs parallel to the value axis of.
        ax.spines["left" if grid == "y" else "bottom"].set_visible(False)
        ax.tick_params(axis=grid, length=0)
    ax.xaxis.label.set(color=MUTED, fontsize=SIZE_BASE)
    ax.yaxis.label.set(color=MUTED, fontsize=SIZE_BASE)


def style_legend(ax: Axes, **kwargs: Any) -> None:
    """Frameless legend in the house style."""
    options: dict[str, Any] = {
        "frameon": False,
        "fontsize": SIZE_SMALL,
        "labelcolor": SECONDARY,
        "handlelength": 1.2,
        "handleheight": 0.9,
        "borderaxespad": 0.2,
        "columnspacing": 1.4,
    }
    options.update(kwargs)
    ax.legend(**options)


def finalize(fig: Figure) -> Figure:
    """Apply the house font to every text of ``fig`` and return it."""
    for text in fig.findobj(Text):
        text.set_fontfamily(FONT_FAMILY)
    return fig


def message_figure(
    title: str, message: str, *, subtitle: str | None = None, figsize: tuple[float, float] = (10, 3)
) -> Figure:
    """Placeholder chart carrying a message (e.g. a not-applicable reason)."""
    fig, ax = new_figure(title, subtitle, figsize=figsize)
    ax.set_axis_off()
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=SIZE_BASE + 1,
        color=SECONDARY,
        wrap=True,
        transform=ax.transAxes,
    )
    return finalize(fig)


# --------------------------------------------------------------------------
# Reference lines with collision-free labels
# --------------------------------------------------------------------------


def reference_rows_needed(count: int) -> int:
    """Rows to reserve in :func:`new_figure` for ``count`` reference labels."""
    return min(max(count, 0), 3)


def add_reference_lines(
    ax: Axes,
    lines: Sequence[tuple[float, str, dict[str, Any]]],
    *,
    max_rows: int = 3,
) -> None:
    """Draw vertical reference lines with labels above the axes.

    ``lines`` holds ``(x, label, line_kwargs)``. A label is centred on its line
    when there is room. Two neighbours that would collide are set back to back
    instead (the left one ends at its line, the right one starts at its line),
    and only when that is not enough does a label move up a row. Close values
    (median, base case, price) therefore never print on top of each other.
    Call it after the x-limits are final and reserve the rows in
    :func:`new_figure` (``reference_rows``).
    """
    fig = ax.get_figure()
    usable = sorted((item for item in lines if math.isfinite(item[0])), key=lambda item: item[0])
    if fig is None or not usable:
        return
    x_low, x_high = ax.get_xlim()
    span = x_high - x_low
    if span == 0:
        return
    # Run the layout engine once so that the axes position is the final one.
    fig.draw_without_rendering()
    position = ax.get_position()
    axes_width_pt = position.width * fig.get_figwidth() * 72.0
    row_height = _REFERENCE_ROW_IN / (position.height * fig.get_figheight())
    # Text width estimated from the character count (deliberately generous).
    char_pt = 0.56 * SIZE_SMALL
    pad = 4.0 / axes_width_pt

    anchors = [(x - x_low) / span for x, _, _ in usable]
    widths = [len(label) * char_pt / axes_width_pt for _, label, _ in usable]
    aligns = ["center"] * len(usable)

    def extent(index: int) -> tuple[float, float]:
        anchor, width = anchors[index], widths[index]
        if aligns[index] == "right":
            return anchor - pad - width, anchor - pad
        if aligns[index] == "left":
            return anchor + pad, anchor + pad + width
        low = min(max(anchor - width / 2, 0.0), max(1.0 - width, 0.0))
        return low, low + width

    for index in range(len(usable) - 1):
        if extent(index)[1] + pad > extent(index + 1)[0] and aligns[index] == "center":
            aligns[index], aligns[index + 1] = "right", "left"

    occupied: list[list[tuple[float, float]]] = [[] for _ in range(max_rows)]
    for index, (x, label, kwargs) in enumerate(usable):
        style = {"color": INK, "linewidth": 1.1, "zorder": 4}
        style.update(kwargs)
        ax.axvline(x, **style)
        low, high = extent(index)
        row = max_rows - 1
        for candidate in range(max_rows):
            if all(high + pad <= lo or low - pad >= hi for lo, hi in occupied[candidate]):
                row = candidate
                break
        occupied[row].append((low, high))
        text_y = 1.0 + 0.012 + row * row_height
        if row > 0:
            leader = Line2D(
                [anchors[index], anchors[index]],
                [1.0, text_y + 0.8 * row_height],
                transform=ax.transAxes,
                color=style["color"],
                linewidth=style["linewidth"],
                clip_on=False,
                zorder=4,
            )
            leader.set_in_layout(False)
            ax.add_artist(leader)
        label_artist = ax.text(
            low,
            text_y,
            label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=SIZE_SMALL,
            color=INK,
            zorder=5,
            bbox={"boxstyle": "square,pad=0.1", "facecolor": SURFACE, "edgecolor": "none"},
        )
        label_artist.set_in_layout(False)


def multiple_formatter(axis: Any) -> FuncFormatter:
    """Tick formatter for multiples (``12.5x``): one decimal unless every tick
    is a whole number."""

    def fmt(value: float, _: Any) -> str:
        ticks = np.asarray(axis.get_majorticklocs(), dtype=float)
        whole = bool(np.allclose(ticks, np.round(ticks)))
        return fmt_multiple(value, 0 if whole else 1)

    return FuncFormatter(fmt)


def percent_formatter(axis: Any) -> FuncFormatter:
    """Tick formatter for shares: as many decimals as the tick spacing needs."""

    def fmt(value: float, _: Any) -> str:
        ticks = np.asarray(axis.get_majorticklocs(), dtype=float)
        step = float(np.min(np.diff(ticks))) if ticks.size > 1 else 1.0
        decimals = 0 if step >= 0.00999 else 1 if step >= 0.000999 else 2
        return fmt_pct(value, decimals)

    return FuncFormatter(fmt)


# --------------------------------------------------------------------------
# Saving
# --------------------------------------------------------------------------


def save_figure(
    fig: Figure,
    path_without_ext: str | Path,
    formats: Sequence[str] = ("png", "svg"),
    *,
    dpi: int = DEFAULT_DPI,
) -> list[Path]:
    """Save ``fig`` once per format and return the written paths.

    ``path_without_ext`` is the target without a suffix (``out/football_field``
    -> ``out/football_field.png`` and ``.svg``). Parent folders are created.
    """
    base = Path(path_without_ext)
    base.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        target = base.parent / f"{base.name}.{fmt.lstrip('.')}"
        fig.savefig(target, format=fmt.lstrip("."), dpi=dpi, facecolor=SURFACE)
        written.append(target)
    return written
