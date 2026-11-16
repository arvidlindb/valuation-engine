"""Football field: value ranges per method against the current share price."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from valuation_engine.core.types import PriceHistory, ValuationMethod, ValuationResult

from . import style

__all__ = ["METHOD_ORDER", "football_field", "order_valuations", "trading_range_result"]

#: Row order of the football field, top to bottom.
METHOD_ORDER: tuple[ValuationMethod, ...] = (
    ValuationMethod.DCF_GORDON,
    ValuationMethod.DCF_EXIT_MULTIPLE,
    ValuationMethod.COMPS_EV_EBITDA,
    ValuationMethod.COMPS_EV_SALES,
    ValuationMethod.COMPS_PE,
    ValuationMethod.LBO,
    ValuationMethod.MONTE_CARLO,
    ValuationMethod.MARKET,
)

_GROUP_COLOURS: dict[ValuationMethod, str] = {
    ValuationMethod.DCF_GORDON: style.BLUE,
    ValuationMethod.DCF_EXIT_MULTIPLE: style.BLUE,
    ValuationMethod.COMPS_EV_EBITDA: style.ORANGE,
    ValuationMethod.COMPS_EV_SALES: style.ORANGE,
    ValuationMethod.COMPS_PE: style.ORANGE,
    ValuationMethod.LBO: style.AQUA,
    ValuationMethod.MONTE_CARLO: style.VIOLET,
    ValuationMethod.MARKET: style.NEUTRAL,
}

_BAR_HEIGHT = 0.42
#: White backing so that a value label stays readable where the price line crosses it.
_LABEL_BOX = {"boxstyle": "square,pad=0.12", "facecolor": style.SURFACE, "edgecolor": "none"}


def order_valuations(valuations: Sequence[ValuationResult]) -> list[ValuationResult]:
    """Sort by ``METHOD_ORDER`` (stable; unknown methods go last)."""
    rank = {method: position for position, method in enumerate(METHOD_ORDER)}
    return sorted(valuations, key=lambda v: rank.get(v.method, len(rank)))


def trading_range_result(
    history: PriceHistory | None,
    price: float,
    currency: str = "SEK",
    *,
    weeks: int = 52,
) -> ValuationResult | None:
    """52-week trading range as a football-field row (``ValuationMethod.MARKET``).

    low / high = minimum / maximum of the last ``weeks`` weekly closes of
    ``history.prices["stock"]``, widened to include the current price; base =
    current price. Returns None without a usable history or price. The closes
    are adjusted closes, so the low can sit slightly below the traded low of a
    stock that paid a dividend during the period.
    """
    if history is None or not math.isfinite(price) or price <= 0:
        return None
    prices = history.prices
    if "stock" not in prices.columns:
        return None
    closes = prices["stock"].astype(float).dropna().iloc[-weeks:]
    closes = closes[np.isfinite(closes.to_numpy(float))]
    if closes.empty:
        return None
    low = min(float(closes.min()), price)
    high = max(float(closes.max()), price)
    return ValuationResult(
        method=ValuationMethod.MARKET,
        label=f"{weeks}-week trading range",
        low=low,
        base=price,
        high=high,
        currency=currency,
        details={
            "definition": (
                f"minimum and maximum of the last {len(closes)} weekly adjusted closes; "
                "base = current price"
            ),
            "observations": int(len(closes)),
            "first_date": str(closes.index[0])[:10],
            "last_date": str(closes.index[-1])[:10],
        },
    )


def football_field(
    valuations: Sequence[ValuationResult],
    price: float,
    *,
    currency: str | None = None,
    title: str = "Valuation summary",
    subtitle: str | None = None,
    footnote: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Horizontal low-high bar per method, base case marked, price as a line.

    Rows follow ``METHOD_ORDER``. Values below zero are displayed at zero (see
    PLAN.md) and flagged with a dagger and a footnote. The two columns on the
    right give the base value and its premium or discount to the price.
    """
    rows = order_valuations(valuations)
    if currency is None:
        currency = rows[0].currency if rows else "SEK"
    if not rows:
        return style.message_figure(title, "No valuation method is applicable.", subtitle=subtitle)

    has_price = math.isfinite(price) and price > 0
    decimals = style.per_share_decimals(price if has_price else max(r.high for r in rows))
    clipped_any = any(r.low < 0 for r in rows)

    notes = ["Bars show the low to high range of each method; the diamond marks the base case."]
    if clipped_any:
        notes.append("† Value below zero, shown at 0.")
    if footnote:
        notes.append(footnote)
    if subtitle is None:
        subtitle = f"Value per share, {currency}"

    if figsize is None:
        note_lines = sum(note.count("\n") + 1 for note in notes)
        figsize = (10.0, 1.75 + 0.46 * len(rows) + 0.15 * note_lines)
    fig, ax = style.new_figure(
        title,
        subtitle,
        figsize=figsize,
        footnote="\n".join(notes),
        reference_rows=1 if has_price else 0,
    )
    style.style_axes(ax, grid="x")

    lows = [max(r.low, 0.0) for r in rows]
    bases = [max(r.base, 0.0) for r in rows]
    highs = [max(r.high, 0.0) for r in rows]
    points = lows + highs + ([price] if has_price else [])
    data_low, data_high = min(points), max(points)
    span = data_high - data_low or max(data_high, 1.0)
    x_low = 0.0 if data_low < 0.35 * data_high else data_low - 0.14 * span
    x_high = data_high + 0.14 * span
    if x_low == 0.0:
        x_low = -0.07 * (x_high - 0.0) if min(lows) < 0.1 * x_high else 0.0
    ax.set_xlim(x_low, x_high)
    label_gap = 0.008 * (x_high - x_low)

    positions = list(range(len(rows)))
    for y, row, low, base, high in zip(positions, rows, lows, bases, highs, strict=True):
        colour = _GROUP_COLOURS.get(row.method, style.NEUTRAL)
        width = high - low
        if width > 0:
            ax.barh(y, width, left=low, height=_BAR_HEIGHT, color=colour, linewidth=0, zorder=2)
        ax.plot(
            [base],
            [y],
            marker="D",
            markersize=6.5,
            markerfacecolor=style.SURFACE,
            markeredgecolor=style.INK,
            markeredgewidth=1.1,
            linestyle="none",
            zorder=5,
        )
        low_text = style.fmt_number(low, decimals) + ("†" if row.low < 0 else "")
        high_text = style.fmt_number(high, decimals) + ("†" if row.high < 0 else "")
        if width > 0:
            ax.text(
                low - label_gap,
                y,
                low_text,
                ha="right",
                va="center",
                fontsize=style.SIZE_SMALL,
                color=style.SECONDARY,
                zorder=6,
                bbox=_LABEL_BOX,
            )
        ax.text(
            high + label_gap + (0.0 if width > 0 else 0.012 * (x_high - x_low)),
            y,
            high_text,
            ha="left",
            va="center",
            fontsize=style.SIZE_SMALL,
            color=style.SECONDARY,
            zorder=6,
            bbox=_LABEL_BOX,
        )

    ax.set_yticks(positions)
    ax.set_yticklabels([r.label for r in rows])
    ax.tick_params(axis="y", length=0, labelsize=style.SIZE_BASE, colors=style.INK)
    ax.set_ylim(len(rows) - 0.45, -0.6)
    ax.spines["left"].set_visible(False)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: style.fmt_number(v, 0)))
    ax.set_xlabel(f"{currency} per share")

    # Right-hand columns: base value and premium / discount to the price.
    transform = ax.get_yaxis_transform()
    columns = [(1.035, "Base")]
    if has_price:
        columns.append((1.125, "vs. price"))
    for x, header in columns:
        ax.text(
            x,
            -0.6,
            header,
            transform=transform,
            ha="left",
            va="bottom",
            fontsize=style.SIZE_SMALL,
            color=style.MUTED,
        )
    for y, row, base in zip(positions, rows, bases, strict=True):
        dagger = "†" if row.base < 0 else ""
        ax.text(
            columns[0][0],
            y,
            # The market row is the share price: same precision as the price label.
            style.fmt_number(base, 2 if row.method == ValuationMethod.MARKET else decimals)
            + dagger,
            transform=transform,
            ha="left",
            va="center",
            fontsize=style.SIZE_BASE,
            fontweight="bold",
            color=style.INK,
        )
        if has_price:
            is_market = row.method == ValuationMethod.MARKET
            ax.text(
                columns[1][0],
                y,
                "-" if is_market else style.fmt_pct(base / price - 1.0, 0, sign=True),
                transform=transform,
                ha="left",
                va="center",
                fontsize=style.SIZE_BASE,
                color=style.SECONDARY,
            )

    if has_price:
        style.add_reference_lines(
            ax,
            [(price, f"Current price {style.fmt_per_share(price, currency, 2)}", {})],
        )
    return style.finalize(fig)
