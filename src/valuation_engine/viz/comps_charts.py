"""Comps charts: peer multiples and the silhouette score per number of clusters."""

from __future__ import annotations

import math

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from valuation_engine.models.comps import MULTIPLES, CompsResult

from . import style

__all__ = ["comps_peer_multiples", "comps_silhouette"]

_MULTIPLE_LABELS: dict[str, str] = {"ev_ebitda": "EV/EBITDA", "ev_sales": "EV/Sales", "pe": "P/E"}


def _spread(values: np.ndarray, x_span: float) -> np.ndarray:
    """Vertical offsets that separate dots lying close together on the x-axis
    (a small deterministic beeswarm)."""
    order = np.argsort(values)
    offsets = np.zeros(len(values))
    threshold = 0.022 * x_span
    placed: list[tuple[float, float]] = []
    levels = (0.0, 0.16, -0.16, 0.32, -0.32)
    for index in order:
        x = values[index]
        for level in levels:
            if all(abs(x - px) > threshold or level != py for px, py in placed):
                offsets[index] = level
                placed.append((x, level))
                break
        else:
            placed.append((x, 0.0))
    return offsets


def comps_peer_multiples(comps: CompsResult) -> Figure:
    """One strip per multiple: every peer as a dot, the interquartile range as a
    band, the median as a tick and the target company highlighted.

    Each multiple has its own scale (they are not comparable with each other).
    Peers whose multiple was removed as an outlier are drawn hollow.
    """
    peers = comps.peers
    short_name = comps.name.replace(" (publ)", "").replace(" AB", "")
    drawable = {
        multiple: multiple in peers.columns
        and not peers[multiple].astype(float).dropna().empty
        and float(comps.multiples_summary.loc[multiple, "n"]) > 0
        for multiple in MULTIPLES
    }
    ratios = [1.0 if drawable[multiple] else 0.4 for multiple in MULTIPLES]
    fig, axes = style.new_figure(
        "Trading comps: peer multiples",
        f"{comps.ticker} \u00b7 {len(peers)} peers \u00b7 band = interquartile range, "
        "tick = peer median",
        figsize=(10.0, 1.9 + 1.25 * sum(ratios)),
        nrows=len(MULTIPLES),
        footnote=(
            "Hollow dots: peers excluded from the statistics as outliers. Implied value range "
            "= 25th percentile, median and 75th percentile multiple applied to the target."
        ),
        gridspec_kw={"height_ratios": ratios},
    )
    for ax, multiple in zip(np.atleast_1d(axes), MULTIPLES, strict=True):
        style.style_axes(ax, grid="x")
        label = _MULTIPLE_LABELS[multiple]
        ax.set_yticks([0])
        ax.set_yticklabels([label])
        ax.tick_params(axis="y", length=0, colors=style.INK, labelsize=style.SIZE_BASE + 0.5)
        ax.spines["left"].set_visible(False)
        ax.set_ylim(-0.75, 0.75)
        ax.xaxis.set_major_formatter(style.multiple_formatter(ax.xaxis))

        summary = comps.multiples_summary.loc[multiple]
        if not drawable[multiple]:
            reason = comps.skipped.get(multiple, "no usable peer multiples")
            ax.set_xticks([])
            ax.spines["bottom"].set_visible(False)
            ax.text(
                0.5,
                0.5,
                f"Not applicable: {reason}",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=style.SIZE_BASE,
                color=style.MUTED,
            )
            continue

        used = peers[f"used_{multiple}"].astype(bool)
        values = peers[multiple].astype(float).dropna()
        target = comps.target_multiples.get(multiple)
        points = list(values.to_numpy(float))
        show_target = target is not None and math.isfinite(target)
        if show_target:
            # Keep the scale readable when the target itself is an extreme outlier.
            show_target = target <= 3.0 * max(points)
            if show_target:
                points.append(float(target))
        low, high = min(points), max(points)
        span = (high - low) or max(high, 1.0)
        ax.set_xlim(low - 0.08 * span, high + 0.08 * span)

        p25, median, p75 = (float(summary[c]) for c in ("p25", "median", "p75"))
        ax.add_patch(
            Rectangle(
                (p25, -0.42),
                p75 - p25,
                0.84,
                facecolor=style.BLUE,
                alpha=style.WASH_ALPHA,
                linewidth=0,
                zorder=1,
            )
        )
        ax.plot([median, median], [-0.42, 0.42], color=style.INK, linewidth=1.3, zorder=3)
        ax.text(
            median,
            -0.5,
            f"Median {style.fmt_multiple(median)}",
            ha="center",
            va="top",
            fontsize=style.SIZE_SMALL,
            color=style.SECONDARY,
        )

        offsets = _spread(values.to_numpy(float), span * 1.16)
        used_mask = used.reindex(values.index).to_numpy(bool)
        ax.scatter(
            values.to_numpy(float)[used_mask],
            offsets[used_mask],
            s=46,
            color=style.BLUE,
            edgecolor=style.SURFACE,
            linewidth=1.2,
            zorder=4,
        )
        ax.scatter(
            values.to_numpy(float)[~used_mask],
            offsets[~used_mask],
            s=40,
            facecolor=style.SURFACE,
            edgecolor=style.MUTED,
            linewidth=1.1,
            zorder=4,
        )
        if show_target:
            ax.scatter(
                [target],
                [0],
                s=95,
                marker="D",
                color=style.ORANGE,
                edgecolor=style.SURFACE,
                linewidth=1.4,
                zorder=6,
            )
            ax.text(
                float(target),
                0.5,
                f"{short_name} {style.fmt_multiple(target)}",
                ha="center",
                va="bottom",
                fontsize=style.SIZE_SMALL,
                fontweight="bold",
                color=style.INK,
            )
        elif target is not None and math.isfinite(target):
            ax.text(
                1.0,
                0.5,
                f"{short_name} {style.fmt_multiple(target)} (off scale, to the right)",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="bottom",
                fontsize=style.SIZE_SMALL,
                fontweight="bold",
                color=style.INK,
            )

    first = np.atleast_1d(axes)[0]
    handles = [
        Line2D(
            [], [], marker="D", linestyle="none", color=style.ORANGE, markersize=7, label=short_name
        ),
        Line2D(
            [], [], marker="o", linestyle="none", color=style.BLUE, markersize=6.5, label="Peer"
        ),
        Patch(facecolor=style.BLUE, alpha=style.WASH_ALPHA * 1.6, label="Interquartile range"),
    ]
    style.style_legend(
        first, handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.02), ncols=len(handles)
    )
    return style.finalize(fig)


def comps_silhouette(comps: CompsResult) -> Figure:
    """Silhouette score for every number of clusters tried; the chosen k is
    highlighted."""
    title = "Peer selection: silhouette score by number of clusters"
    scores = comps.silhouette_scores
    if not scores:
        return style.message_figure(
            title,
            "No clustering was performed (too few candidates); "
            "peers are the nearest companies in the candidate set.",
            subtitle=comps.ticker,
        )
    ks = sorted(scores)
    values = [scores[k] for k in ks]
    fig, ax = style.new_figure(
        title,
        f"{comps.ticker} · k-means on {len(comps.feature_names)} standardized features, "
        f"{len(comps.candidates)} candidates ({comps.candidate_scope.replace('_', ' ')}) "
        f"· chosen k = {comps.chosen_k}",
        figsize=(8.6, 4.3),
        footnote=(
            "Higher is better (range −1 to 1). The k with the highest score is used; "
            "the peer group is the target's cluster, capped to the nearest companies."
        ),
    )
    colours = [style.BLUE if k == comps.chosen_k else style.NEUTRAL_LIGHT for k in ks]
    positions = np.arange(len(ks))
    ax.bar(positions, values, width=0.5, color=colours, linewidth=0, zorder=2)
    peak = max(max(values), 0.0) or 1.0
    for x, k, value in zip(positions, ks, values, strict=True):
        ax.text(
            x,
            value + 0.02 * peak if value >= 0 else value - 0.02 * peak,
            f"{value:.2f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=style.SIZE_SMALL,
            fontweight="bold" if k == comps.chosen_k else "normal",
            color=style.INK,
        )
    ax.axhline(0, color=style.AXIS, linewidth=0.8, zorder=1)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(k) for k in ks])
    ax.tick_params(axis="x", length=0, colors=style.INK)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_ylim(min(min(values), 0.0) * 1.15, peak * 1.15)
    return style.finalize(fig)
