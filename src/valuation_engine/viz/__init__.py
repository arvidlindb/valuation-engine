"""Charts: football field, sensitivity tables, distributions and bridges.

Every function returns a :class:`matplotlib.figure.Figure` built without pyplot
(safe in headless processes); :func:`save_figure` writes PNG and SVG. All
charts share the style defined in :mod:`valuation_engine.viz.style`.

The Monte Carlo result and the report bundle are consumed through the
structural types in :mod:`valuation_engine.viz.protocols`, so this package does
not import the Monte Carlo model or the pipeline.

The public names are loaded lazily (PEP 562): importing
``valuation_engine.viz`` or ``valuation_engine.viz.protocols`` does not import
matplotlib; the first access to a chart function does. ``from
valuation_engine.viz import football_field`` works as usual.
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .comps_charts import comps_peer_multiples, comps_silhouette
    from .dcf_charts import dcf_bridge_waterfall, dcf_fcff_forecast
    from .football_field import METHOD_ORDER, football_field, order_valuations, trading_range_result
    from .lbo_charts import lbo_debt_paydown, lbo_returns_waterfall
    from .monte_carlo_charts import (
        mc_convergence,
        mc_histogram,
        mc_tornado,
        mc_variance_contribution,
    )
    from .protocols import METHOD_KEYS, MonteCarloLike, ReportBundle, ValuationReportLike
    from .report import (
        CHART_NAMES,
        build_charts,
        display_name,
        report_football_field,
        report_valuations,
        save_charts,
    )
    from .sensitivity import dcf_sensitivity_heatmap, lbo_sensitivity_heatmap, sensitivity_heatmap
    from .style import message_figure, save_figure

#: Public name -> submodule that defines it.
_EXPORTS: dict[str, str] = {
    "comps_peer_multiples": "comps_charts",
    "comps_silhouette": "comps_charts",
    "dcf_bridge_waterfall": "dcf_charts",
    "dcf_fcff_forecast": "dcf_charts",
    "METHOD_ORDER": "football_field",
    "football_field": "football_field",
    "order_valuations": "football_field",
    "trading_range_result": "football_field",
    "lbo_debt_paydown": "lbo_charts",
    "lbo_returns_waterfall": "lbo_charts",
    "mc_convergence": "monte_carlo_charts",
    "mc_histogram": "monte_carlo_charts",
    "mc_tornado": "monte_carlo_charts",
    "mc_variance_contribution": "monte_carlo_charts",
    "METHOD_KEYS": "protocols",
    "MonteCarloLike": "protocols",
    "ReportBundle": "protocols",
    "ValuationReportLike": "protocols",
    "CHART_NAMES": "report",
    "build_charts": "report",
    "display_name": "report",
    "report_football_field": "report",
    "report_valuations": "report",
    "save_charts": "report",
    "dcf_sensitivity_heatmap": "sensitivity",
    "lbo_sensitivity_heatmap": "sensitivity",
    "sensitivity_heatmap": "sensitivity",
    "message_figure": "style",
    "save_figure": "style",
}

__all__ = [
    "CHART_NAMES",
    "METHOD_KEYS",
    "METHOD_ORDER",
    "MonteCarloLike",
    "ReportBundle",
    "ValuationReportLike",
    "build_charts",
    "comps_peer_multiples",
    "comps_silhouette",
    "dcf_bridge_waterfall",
    "dcf_fcff_forecast",
    "dcf_sensitivity_heatmap",
    "display_name",
    "football_field",
    "lbo_debt_paydown",
    "lbo_returns_waterfall",
    "lbo_sensitivity_heatmap",
    "mc_convergence",
    "mc_histogram",
    "mc_tornado",
    "mc_variance_contribution",
    "message_figure",
    "order_valuations",
    "report_football_field",
    "report_valuations",
    "save_charts",
    "save_figure",
    "sensitivity_heatmap",
    "trading_range_result",
]


def __getattr__(name: str) -> Any:
    """Load a public name from its submodule on first access (PEP 562)."""
    submodule = _EXPORTS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{submodule}"), name)
    globals()[name] = value
    return value


class _VizPackage(types.ModuleType):
    """Keeps ``viz.football_field`` bound to the chart function.

    ``football_field`` names both a submodule and the function in it. The import
    system binds a submodule as an attribute of its package when it is first
    imported, which would make ``from valuation_engine.viz import
    football_field`` return the module or the function depending on import
    order. Ignoring that one binding keeps the public name stable (the function,
    as before the lazy loading); the submodule stays reachable through
    ``sys.modules`` and ``importlib.import_module``.
    """

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "football_field" and isinstance(value, types.ModuleType):
            return
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _VizPackage


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
