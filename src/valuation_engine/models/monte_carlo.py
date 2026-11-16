"""Monte Carlo simulation on top of the FCFF DCF.

Public API
----------
* :class:`DistributionSpec` - marginal distribution of one driver (normal, truncated
  normal or triangular).
* :class:`MonteCarloConfig` - number of simulations, seed, the four distributions and
  their correlation.
* :func:`default_driver_stds` - the company-informed default standard deviations.
* :func:`run_monte_carlo` - runs (or reuses) the base DCF and simulates it
  (:class:`MonteCarloResult`).

Model in short
--------------
The base case is :func:`valuation_engine.models.dcf.run_dcf`. Its exact keyword
arguments (``DCFResult.core_inputs``) are re-used and four of them are replaced by
arrays with one entry per simulation; everything is then valued in ONE call of
:func:`valuation_engine.models.dcf.dcf_core`. There is no Python loop over
simulations and no DCF arithmetic in this module.

Stochastic drivers and how a draw maps onto the forecast
-------------------------------------------------------
Every driver is simulated as an OFFSET from its base-case value, so a distribution
centred on 0 is centred on the base case.

``revenue_growth``
    One draw per simulation is added to the growth of EVERY forecast year (a
    parallel shift of the whole base growth path). The draw is therefore a
    persistent surprise in the growth rate over the explicit period, not
    year-by-year noise. Terminal growth is NOT shifted by it.
``ebitda_margin``
    One draw per simulation is added to the margin of every forecast year (parallel
    shift). Because the terminal year inherits the year-N margin, the shift also
    applies in perpetuity.
``wacc``
    The draw is added to the base WACC (used for the explicit years and the
    terminal value).
``terminal_growth``
    The draw is added to the base perpetual growth ``g`` (Gordon case only; the
    exit-multiple value does not depend on it).

All other inputs (D&A, capex, NWC, tax, exit multiple, net debt, shares) stay at
their base-case values.

``MonteCarloResult.inputs`` reports the sampled driver LEVELS: ``revenue_growth`` is
the year-1 growth (base year-1 growth + shift), ``ebitda_margin`` the year-1 margin
(base year-1 margin + shift), ``wacc`` and ``terminal_growth`` the sampled rates.

Default uncertainty (all can be overridden per driver)
------------------------------------------------------
Default kind is a truncated normal centred on the base case, truncated at
+/- ``TRUNCATION_SIGMAS`` (2) standard deviations. Default standard deviations
(:func:`default_driver_stds`):

* Growth shift: ``GROWTH_VOL_SCALE`` (0.5) x the sample standard deviation of the
  company's historical year-over-year revenue growth, clipped to ``GROWTH_STD_BAND``
  (1 pp .. 5 pp); ``GROWTH_STD_FALLBACK`` (3 pp) with fewer than two growth
  observations. The factor 0.5 is roughly ``1 / sqrt(5)``: the shift is an error in
  the AVERAGE growth over five years, which is less volatile than a single year.
  The band is needed because the history is only 3 growth observations.
* Margin shift: sample standard deviation of the historical EBITDA margin, clipped
  to ``MARGIN_STD_BAND`` (1 pp .. 5 pp); ``MARGIN_STD_FALLBACK`` (2 pp) with fewer
  than two observations. It is additionally capped at half of the lowest base
  margin, so the default 2-sigma truncation never produces a negative margin.
* WACC: ``WACC_STD`` (1 pp), about the dispersion one gets from reasonable
  alternative choices of beta, equity risk premium and size premium.
* Terminal growth: ``TERMINAL_GROWTH_STD`` (0.5 pp), i.e. 1 % .. 3 % at 2 sigma
  around the 2 % inflation anchor.

Note that a truncated normal has a smaller standard deviation than its ``std``
parameter (factor 0.88 at +/- 2 sigma).

Keeping WACC - g positive
-------------------------
The Gordon value has a pole at ``WACC = g``. Where the bounds of the WACC and
terminal growth distributions are chosen by this module (truncated normal or
triangular without explicit ``low``/``high``), they are narrowed symmetrically, if
necessary, so that the smallest possible sampled spread is
``min(config.min_wacc_growth_spread, half of the base spread)``. Invalid draws are
then impossible. With an unbounded normal or user-given bounds the guarantee does
not hold; a simulation is then INVALID when ``WACC - g`` is below the DCF's own
minimum spread (``dcf.MIN_WACC_GROWTH_SPREAD``), ``WACC <= 0`` or any growth
``<= -100 %``. Invalid simulations are NaN in ``values`` / ``values_exit``, counted
in ``n_non_finite`` and excluded from every statistic; they are never silently
dropped or replaced.

Correlation
-----------
Correlated standard normals are produced with the Cholesky factor of the 4x4
correlation matrix (``Z = E L^T``) and each column is mapped to its marginal by
``F^-1(Phi(z))`` (Gaussian copula), so the dependence works for all three kinds.
The correlation is specified on the normal scale; the Pearson correlation of the
sampled drivers is marginally lower for non-normal marginals. Default: growth and
margin correlate with +0.3 (operating leverage: faster growth spreads fixed costs
over more revenue), everything else is independent.

Interpretation: the mean is above the base case
-----------------------------------------------
The Gordon value is convex in WACC and in g (it behaves like ``1 / (WACC - g)``),
so by Jensen's inequality symmetric uncertainty around the base case raises the
MEAN above the base value, while the MEDIAN stays close to it (compounding makes
the value mildly convex in the growth shift as well). For a typical large
industrial company (Atlas Copco, ATCO-A.ST, 10 000 simulations, default
distributions) the median lands about 1 % and the mean about 5 % above the base
value, and across seeds the mean varies by about its reported standard error.
The gap is a property of the valuation formula, not evidence of upside.
``to_valuation_results`` therefore
reports the median as the central value. When every standard deviation is exactly
zero each simulation reproduces the base value exactly.

Limitations
-----------
* Parallel shifts only: no year-by-year noise, no mean reversion.
* Only four drivers are stochastic; capex, NWC, tax and the exit multiple are fixed.
* The default volatilities rest on at most four fiscal years.
* The variance contributions are a rank-correlation heuristic (see
  :func:`_variance_contribution`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri
from scipy.stats import rankdata

from valuation_engine.core.types import (
    CompanyData,
    DCFAssumptions,
    MarketParameters,
    NotApplicableError,
    PeerSnapshot,
    ValuationMethod,
    ValuationResult,
)
from valuation_engine.models import dcf

__all__ = [
    "DRIVERS",
    "DistributionSpec",
    "MonteCarloConfig",
    "MonteCarloResult",
    "default_driver_stds",
    "run_monte_carlo",
]

# --------------------------------------------------------------------------
# Documented default choices
# --------------------------------------------------------------------------

_DEFAULT_PARAMS = MarketParameters()

#: The stochastic drivers, in the order used by the correlation matrix.
DRIVERS: tuple[str, ...] = ("revenue_growth", "ebitda_margin", "wacc", "terminal_growth")

DistributionKind = Literal["normal", "truncnormal", "triangular"]
_KINDS: tuple[str, ...] = ("normal", "truncnormal", "triangular")

#: Kind used for a driver without an explicit spec.
DEFAULT_KIND: DistributionKind = "truncnormal"
#: Default truncation of a truncated normal, in standard deviations either side.
TRUNCATION_SIGMAS = 2.0
#: Half-width of a symmetric triangular distribution with standard deviation 1.
TRIANGULAR_HALF_WIDTH = math.sqrt(6.0)

#: Share of the historical year-over-year growth volatility applied to the shift.
GROWTH_VOL_SCALE = 0.5
GROWTH_STD_BAND = (0.01, 0.05)
GROWTH_STD_FALLBACK = 0.03
MARGIN_STD_BAND = (0.01, 0.05)
MARGIN_STD_FALLBACK = 0.02
#: The default margin std is at most this share of the lowest base margin.
MARGIN_STD_MAX_SHARE = 0.5
WACC_STD = 0.01
TERMINAL_GROWTH_STD = 0.005

#: Default correlation between the growth shift and the margin shift.
GROWTH_MARGIN_CORRELATION = 0.3
#: Smallest sampled WACC - g the automatic bounds allow (see module docstring).
MIN_WACC_GROWTH_SPREAD = 0.015

#: Percentiles reported in ``MonteCarloResult.percentiles``.
PERCENTILES: tuple[int, ...] = (5, 10, 25, 50, 75, 90, 95)
#: Number of log-spaced checkpoints in the convergence table.
CONVERGENCE_POINTS = 50
#: Uniforms are kept inside (eps, 1 - eps) before an inverse CDF is applied.
_UNIFORM_EPS = 1e-12


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DistributionSpec:
    """Marginal distribution of one driver.

    By default the numbers are OFFSETS from the base case (``mean=0`` is centred on
    the base value; ``low=-0.01`` means "one percentage point below base"). With
    ``absolute=True`` ``mean``, ``low`` and ``high`` are levels of the driver
    instead (year-1 growth, year-1 margin, WACC, terminal growth), e.g.
    ``DistributionSpec("triangular", mean=0.08, low=0.07, high=0.10, absolute=True)``
    for the WACC.

    kind
        ``"normal"``: ``mean``, ``std``; unbounded (``low``/``high`` not allowed).
        ``"truncnormal"``: normal(``mean``, ``std``) restricted to [``low``,
        ``high``]; ``std`` is the standard deviation BEFORE truncation. Missing
        bounds default to ``mean +/- TRUNCATION_SIGMAS * std`` when both are
        missing; a single missing bound means unbounded on that side.
        ``"triangular"``: ``mean`` is the MODE, on [``low``, ``high``]. Without
        bounds a symmetric triangle with standard deviation ``std`` is used
        (half-width ``sqrt(6) * std``); with bounds ``std`` is ignored.
    mean
        Location (mode for the triangular). ``None`` -> the base case.
    std
        ``None`` -> the company-informed default of :func:`default_driver_stds`.
        Zero is legal and makes the driver deterministic.
    """

    kind: DistributionKind = DEFAULT_KIND
    mean: float | None = None
    std: float | None = None
    low: float | None = None
    high: float | None = None
    absolute: bool = False

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"unknown distribution kind {self.kind!r}; expected one of {_KINDS}")
        for name in ("mean", "std", "low", "high"):
            value = getattr(self, name)
            if value is not None and math.isnan(float(value)):
                raise ValueError(f"DistributionSpec.{name} must not be NaN")
        if self.mean is not None and not math.isfinite(self.mean):
            raise ValueError("DistributionSpec.mean must be finite")
        if self.std is not None and not (math.isfinite(self.std) and self.std >= 0):
            raise ValueError("DistributionSpec.std must be a finite number >= 0")
        if self.kind == "normal" and any(
            bound is not None and math.isfinite(bound) for bound in (self.low, self.high)
        ):
            raise ValueError("a normal distribution has no bounds; use kind='truncnormal'")
        if self.kind == "triangular" and (self.low is None) != (self.high is None):
            raise ValueError("a triangular distribution needs both low and high, or neither")
        if self.low is not None and self.high is not None and self.low > self.high:
            raise ValueError("DistributionSpec.low must not exceed DistributionSpec.high")


@dataclass(frozen=True)
class MonteCarloConfig:
    """Settings of one Monte Carlo run.

    n_simulations
        Number of simulations (default 10 000: the standard error of the mean is
        then about 0.3 % of the base value for a typical company).
    seed
        Seed of ``numpy.random.default_rng``. The default 42 makes repeated runs
        identical; ``None`` draws fresh entropy.
    revenue_growth, ebitda_margin, wacc, terminal_growth
        :class:`DistributionSpec` per driver; ``None`` -> truncated normal centred
        on the base case with the default standard deviation.
    growth_margin_correlation
        Correlation between the growth and the margin shift (default +0.3). Ignored
        when ``correlation_matrix`` is given.
    correlation_matrix
        Optional full 4x4 correlation matrix in the order of ``DRIVERS``. Must be
        symmetric with unit diagonal and positive definite.
    min_wacc_growth_spread
        Smallest sampled ``WACC - g`` the automatically chosen bounds allow (capped
        at half of the base spread).
    tornado_percentiles
        The two percentiles of each driver's own distribution used for the tornado.
    convergence_points
        Approximate number of log-spaced checkpoints in the convergence table.

    ``MonteCarloResult.config`` holds the RESOLVED configuration: all four specs
    filled in as offsets from the base case and the full correlation matrix.
    """

    n_simulations: int = 10_000
    seed: int | None = 42
    revenue_growth: DistributionSpec | None = None
    ebitda_margin: DistributionSpec | None = None
    wacc: DistributionSpec | None = None
    terminal_growth: DistributionSpec | None = None
    growth_margin_correlation: float = GROWTH_MARGIN_CORRELATION
    correlation_matrix: Sequence[Sequence[float]] | np.ndarray | None = None
    min_wacc_growth_spread: float = MIN_WACC_GROWTH_SPREAD
    tornado_percentiles: tuple[float, float] = (10.0, 90.0)
    convergence_points: int = CONVERGENCE_POINTS

    def __post_init__(self) -> None:
        if isinstance(self.n_simulations, bool) or not isinstance(
            self.n_simulations, (int, np.integer)
        ):
            raise ValueError("n_simulations must be an integer")
        if self.n_simulations < 1:
            raise ValueError("n_simulations must be at least 1")
        if not (
            math.isfinite(self.min_wacc_growth_spread)
            and self.min_wacc_growth_spread >= dcf.MIN_WACC_GROWTH_SPREAD
        ):
            raise ValueError(
                f"min_wacc_growth_spread must be at least {dcf.MIN_WACC_GROWTH_SPREAD} "
                "(the minimum spread of the base DCF)"
            )
        low, high = self.tornado_percentiles
        if not 0.0 < low < high < 100.0:
            raise ValueError("tornado_percentiles must satisfy 0 < low < high < 100")
        if self.convergence_points < 2:
            raise ValueError("convergence_points must be at least 2")

    def spec(self, driver: str) -> DistributionSpec | None:
        """The spec of ``driver`` (one of ``DRIVERS``)."""
        if driver not in DRIVERS:
            raise KeyError(driver)
        return getattr(self, driver)


# --------------------------------------------------------------------------
# Default volatilities
# --------------------------------------------------------------------------


def _clip(value: float, band: tuple[float, float]) -> float:
    return min(max(value, band[0]), band[1])


def _sample_std(values: np.ndarray) -> float | None:
    """Sample standard deviation of the finite entries; None with fewer than two."""
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return None
    result = float(np.std(finite, ddof=1))
    return result if math.isfinite(result) else None


def default_driver_stds(company: CompanyData, base: dcf.DCFResult) -> dict[str, float]:
    """Default standard deviation of each driver's offset from the base case.

    See "Default uncertainty" in the module docstring. Growth and margin use the
    company's own history (all available fiscal years); WACC and terminal growth
    are constants.
    """
    income = company.statements.income

    def series(name: str) -> np.ndarray:
        if name not in income.columns:
            return np.full(len(income.index), np.nan)
        return pd.to_numeric(income[name], errors="coerce").to_numpy(float)

    revenue = series("revenue")
    ebitda = series("ebitda")
    usable = np.isfinite(revenue) & (revenue > 0)

    with np.errstate(divide="ignore", invalid="ignore"):
        # Year-over-year growth between consecutive rows where both are usable.
        growth = np.where(usable[1:] & usable[:-1], revenue[1:] / revenue[:-1] - 1.0, np.nan)
        margin = np.where(usable, ebitda / revenue, np.nan)

    growth_vol = _sample_std(growth)
    growth_std = (
        GROWTH_STD_FALLBACK
        if growth_vol is None
        else _clip(GROWTH_VOL_SCALE * growth_vol, GROWTH_STD_BAND)
    )

    margin_vol = _sample_std(margin)
    margin_std = MARGIN_STD_FALLBACK if margin_vol is None else _clip(margin_vol, MARGIN_STD_BAND)
    lowest_margin = float(np.min(np.asarray(base.core_inputs["ebitda_margin"], dtype=float)))
    if lowest_margin > 0:
        margin_std = min(margin_std, MARGIN_STD_MAX_SHARE * lowest_margin)

    return {
        "revenue_growth": float(growth_std),
        "ebitda_margin": float(margin_std),
        "wacc": WACC_STD,
        "terminal_growth": TERMINAL_GROWTH_STD,
    }


# --------------------------------------------------------------------------
# Resolving specs and sampling
# --------------------------------------------------------------------------


def _triangular_std(low: float, mode: float, high: float) -> float:
    variance = (
        low * low + mode * mode + high * high - low * mode - low * high - mode * high
    ) / 18.0
    return math.sqrt(max(variance, 0.0))


def _resolve_spec(
    driver: str, spec: DistributionSpec | None, base_level: float, default_std: float
) -> tuple[DistributionSpec, bool]:
    """Return ``spec`` as a complete offset spec and whether its bounds were chosen
    here (and may therefore be narrowed to protect the WACC - g spread)."""
    spec = spec if spec is not None else DistributionSpec()
    shift = base_level if spec.absolute else 0.0
    mean = 0.0 if spec.mean is None else float(spec.mean) - shift
    low = None if spec.low is None else float(spec.low) - shift
    high = None if spec.high is None else float(spec.high) - shift
    std = default_std if spec.std is None else float(spec.std)
    auto_bounds = False

    if spec.kind == "normal":
        low, high = -math.inf, math.inf
    elif spec.kind == "truncnormal":
        if low is None and high is None:
            low, high = mean - TRUNCATION_SIGMAS * std, mean + TRUNCATION_SIGMAS * std
            auto_bounds = True
        else:
            low = -math.inf if low is None else low
            high = math.inf if high is None else high
        if std > 0:
            mass = float(ndtr((high - mean) / std) - ndtr((low - mean) / std))
            if not mass > 1e-9:
                raise ValueError(
                    f"{driver}: the truncation interval [{low}, {high}] carries no "
                    f"probability mass of normal(mean={mean}, std={std})"
                )
    else:  # triangular
        if low is None or high is None:
            half = TRIANGULAR_HALF_WIDTH * std
            low, high = mean - half, mean + half
            auto_bounds = True
        elif not low <= mean <= high:
            raise ValueError(
                f"{driver}: the mode of a triangular distribution must lie within "
                f"[low, high]; got mode offset {mean} and bounds [{low}, {high}]"
            )
        else:
            std = _triangular_std(low, mean, high)

    resolved = DistributionSpec(
        kind=spec.kind, mean=mean, std=std, low=low, high=high, absolute=False
    )
    return resolved, auto_bounds


def _narrow(spec: DistributionSpec, scale: float) -> DistributionSpec:
    """Shrink automatically chosen bounds symmetrically around the mean."""
    assert spec.mean is not None and spec.low is not None and spec.high is not None
    half = 0.5 * (spec.high - spec.low) * scale
    std = spec.std
    if spec.kind == "triangular":
        std = half / TRIANGULAR_HALF_WIDTH
    return replace(spec, low=spec.mean - half, high=spec.mean + half, std=std)


def _protect_spread(
    specs: dict[str, DistributionSpec],
    auto: dict[str, bool],
    base_spread: float,
    min_spread: float,
    notes: list[str],
) -> None:
    """Narrow automatic WACC / terminal growth bounds so that the smallest sampled
    spread is ``min(min_spread, base_spread / 2)``. Modifies ``specs`` in place."""
    wacc, growth = specs["wacc"], specs["terminal_growth"]
    assert wacc.low is not None and growth.high is not None
    floor = min(min_spread, 0.5 * base_spread)
    # Smallest possible sampled spread = base_spread + wacc.low - growth.high.
    shortfall = floor - (base_spread + wacc.low - growth.high)
    if not shortfall > 0:
        return
    adjustable = [name for name in ("wacc", "terminal_growth") if auto[name]]
    widths = {
        name: 0.5 * (float(specs[name].high) - float(specs[name].low))  # type: ignore[arg-type]
        for name in adjustable
    }
    total = sum(widths.values())
    if not adjustable or not math.isfinite(shortfall) or total <= 0:
        notes.append(
            "WACC - g: the given distributions allow a spread below "
            f"{floor:.2%}; simulations below the DCF minimum spread are reported as invalid"
        )
        return
    scale = max(1.0 - shortfall / total, 0.0)
    for name in adjustable:
        specs[name] = _narrow(specs[name], scale)
    notes.append(
        f"WACC - g: bounds of {' and '.join(adjustable)} narrowed to {scale:.0%} of their "
        f"default width so that the sampled spread stays at or above {floor:.2%} "
        f"(base spread {base_spread:.2%})"
    )
    if scale == 0.0:
        notes.append(
            "WACC - g: even fully narrowed automatic bounds cannot guarantee the minimum "
            "spread because of the other, user-given distribution"
        )


def _quantile(spec: DistributionSpec, uniform: np.ndarray) -> np.ndarray:
    """Inverse CDF of a resolved spec at ``uniform`` (values in (0, 1))."""
    assert spec.mean is not None and spec.std is not None
    assert spec.low is not None and spec.high is not None
    mean, std, low, high = spec.mean, spec.std, spec.low, spec.high
    u = np.asarray(uniform, dtype=float)

    if spec.kind == "normal":
        if std == 0:
            return np.full(u.shape, mean)
        return mean + std * ndtri(u)

    if spec.kind == "truncnormal":
        if std == 0 or low == high:
            return np.full(u.shape, min(max(mean, low), high))
        cdf_low = ndtr((low - mean) / std)
        cdf_high = ndtr((high - mean) / std)
        return np.clip(mean + std * ndtri(cdf_low + u * (cdf_high - cdf_low)), low, high)

    width = high - low
    if width == 0:
        return np.full(u.shape, mean)
    split = (mean - low) / width
    lower = low + np.sqrt(u * width * (mean - low))
    upper = high - np.sqrt((1.0 - u) * width * (high - mean))
    return np.where(u < split, lower, upper)


def _transform(spec: DistributionSpec, normals: np.ndarray) -> np.ndarray:
    """Map standard normal draws to the marginal of ``spec`` (Gaussian copula)."""
    assert spec.mean is not None and spec.std is not None
    if spec.kind == "normal":
        # Phi^-1(Phi(z)) = z: skip the round trip, it only loses precision.
        if spec.std == 0:
            return np.full(normals.shape, spec.mean)
        return spec.mean + spec.std * normals
    uniform = np.clip(ndtr(normals), _UNIFORM_EPS, 1.0 - _UNIFORM_EPS)
    return _quantile(spec, uniform)


def _correlation_matrix(config: MonteCarloConfig) -> np.ndarray:
    """Validated 4x4 correlation matrix in the order of ``DRIVERS``."""
    k = len(DRIVERS)
    if config.correlation_matrix is None:
        rho = float(config.growth_margin_correlation)
        if not (math.isfinite(rho) and -1.0 < rho < 1.0):
            raise ValueError(
                f"growth_margin_correlation must lie strictly between -1 and 1, got {rho}"
            )
        matrix = np.eye(k)
        matrix[0, 1] = matrix[1, 0] = rho
        return matrix
    try:
        matrix = np.array(config.correlation_matrix, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("correlation_matrix must be a numeric 4x4 matrix") from error
    if matrix.shape != (k, k):
        raise ValueError(
            f"correlation_matrix must have shape ({k}, {k}) in the order {DRIVERS}, "
            f"got {matrix.shape}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("correlation_matrix contains non-finite entries")
    if not np.allclose(matrix, matrix.T, atol=1e-10):
        raise ValueError("correlation_matrix must be symmetric")
    if not np.allclose(np.diag(matrix), 1.0, atol=1e-10):
        raise ValueError("correlation_matrix must have ones on the diagonal")
    if np.any(np.abs(matrix) > 1.0 + 1e-10):
        raise ValueError("correlation_matrix entries must lie in [-1, 1]")
    matrix = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(matrix, 1.0)
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        smallest = float(np.min(np.linalg.eigvalsh(matrix)))
        raise ValueError(
            "correlation_matrix is not positive definite (smallest eigenvalue "
            f"{smallest:.3g}): the pairwise correlations are mutually inconsistent, so no "
            "joint distribution has them"
        ) from error
    return matrix


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def _percentiles(values: np.ndarray) -> dict[int, float]:
    if values.size == 0:
        return {p: math.nan for p in PERCENTILES}
    return {
        p: float(q) for p, q in zip(PERCENTILES, np.percentile(values, PERCENTILES), strict=True)
    }


def _mean(values: np.ndarray) -> float:
    """Mean; exactly the common value when all observations are identical."""
    return float(values[0]) if np.ptp(values) == 0 else float(np.mean(values))


def _std(values: np.ndarray) -> float:
    """Sample standard deviation; exactly 0 for one or for identical observations."""
    if values.size < 2 or np.ptp(values) == 0:
        return 0.0
    return float(np.std(values, ddof=1))


def _variance_contribution(inputs: pd.DataFrame, values: np.ndarray) -> pd.Series:
    """Share of the output variance attributed to each driver.

    Method: the squared Spearman rank correlation between each sampled input and
    the simulated value, normalized to sum to one. Rank correlation is used because
    the value is monotone but strongly non-linear in WACC and g.

    Limitations: (1) with correlated inputs the shares overlap: the rank
    correlation of growth also picks up the part of the margin effect that moves
    with growth, so the shared part is attributed to both and the split between
    correlated drivers is not unique; (2) interaction effects (e.g. WACC x g) are
    not separated; (3) the normalization hides how much of the variance the four
    monotone effects explain together. It is a ranking device, not a variance
    decomposition in the Sobol sense. With zero total variance all shares are 0.
    """
    result = pd.Series(0.0, index=pd.Index(DRIVERS, name="driver"), name="variance_contribution")
    if values.size < 3 or np.ptp(values) == 0:
        return result
    value_ranks = rankdata(values)
    squared: dict[str, float] = {}
    for name in DRIVERS:
        column = inputs[name].to_numpy(float)
        if np.ptp(column) == 0:
            squared[name] = 0.0
            continue
        rho = float(np.corrcoef(rankdata(column), value_ranks)[0, 1])
        squared[name] = rho * rho if math.isfinite(rho) else 0.0
    total = sum(squared.values())
    if total > 0:
        for name in DRIVERS:
            result[name] = squared[name] / total
    return result


def _convergence(values: np.ndarray, points: int) -> pd.DataFrame:
    """Running mean, median and standard error at log-spaced checkpoints.

    ``values`` are the valid simulations in simulation order; ``n`` counts them.
    ``standard_error`` is ``std(first n, ddof=1) / sqrt(n)`` and falls like
    ``1 / sqrt(n)``.
    """
    columns = ["n", "running_mean", "running_median", "standard_error"]
    total = values.size
    if total == 0:
        return pd.DataFrame(columns=columns)
    start = min(10, total)
    checkpoints = np.unique(np.round(np.geomspace(start, total, points)).astype(int))
    cumulative = np.cumsum(values)
    rows = []
    for n in checkpoints:
        head = values[:n]
        rows.append(
            (
                int(n),
                float(cumulative[n - 1] / n) if np.ptp(head) else float(head[0]),
                float(np.median(head)),
                _std(head) / math.sqrt(n) if n > 1 else math.nan,
            )
        )
    return pd.DataFrame(rows, columns=columns)


def _input_correlation(inputs: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation of the sampled inputs (NaN for a constant driver)."""
    data = inputs.to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        matrix = np.corrcoef(data, rowvar=False) if len(data) > 1 else np.full((4, 4), np.nan)
    return pd.DataFrame(matrix, index=list(DRIVERS), columns=list(DRIVERS))


# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------


@dataclass
class MonteCarloResult:
    """Output of :func:`run_monte_carlo`. Values per share in the trading currency.

    ``values``, ``values_exit`` and the rows of ``inputs`` are aligned and have
    length ``n_simulations``. Invalid simulations (see module docstring) are NaN in
    both value arrays; with the default configuration there are none. All
    statistics use the valid simulations only (``valid_values``). The headline
    statistics (``percentiles``, ``mean``, ``std``, ``standard_error``,
    ``prob_above_price``, ``tornado``, ``variance_contribution``, ``convergence``)
    refer to the Gordon growth value.
    """

    ticker: str
    currency: str
    price: float
    #: Base-case DCF value per share (Gordon growth).
    base_value: float
    n_simulations: int
    #: Gordon growth value per share, one entry per simulation.
    values: np.ndarray
    #: Exit-multiple value per share, one entry per simulation.
    values_exit: np.ndarray
    #: Sampled driver levels: year-1 growth, year-1 margin, WACC, terminal growth.
    inputs: pd.DataFrame
    #: Percentile (5, 10, 25, 50, 75, 90, 95) -> value per share.
    percentiles: dict[int, float]
    mean: float
    std: float
    #: Standard error of the mean, ``std / sqrt(n_valid)``.
    standard_error: float
    #: Share of valid simulations with a value above the current price.
    prob_above_price: float
    #: One-at-a-time sensitivity; index = driver, sorted by ``swing`` descending.
    #: ``value_low`` / ``value_high`` are the values with the driver at ``input_low``
    #: / ``input_high`` (its own P10 / P90), the others at base. For WACC
    #: ``value_low > value_high``. ``swing = |value_high - value_low|``.
    tornado: pd.DataFrame
    #: Normalized squared Spearman rank correlations, summing to 1.
    variance_contribution: pd.Series
    #: Columns ``n``, ``running_mean``, ``running_median``, ``standard_error``.
    convergence: pd.DataFrame
    #: Resolved configuration (specs as offsets from base, full correlation matrix).
    config: MonteCarloConfig
    # ---- beyond the PLAN.md contract ----
    #: Base-case DCF value per share (exit multiple).
    base_value_exit: float = math.nan
    percentiles_exit: dict[int, float] = field(default_factory=dict)
    mean_exit: float = math.nan
    std_exit: float = math.nan
    prob_above_price_exit: float = math.nan
    #: Number of invalid (NaN) simulations and of valid ones.
    n_non_finite: int = 0
    n_valid: int = 0
    #: Base-case level of each driver (same definition as the columns of ``inputs``).
    base_inputs: dict[str, float] = field(default_factory=dict)
    #: Empirical Pearson correlation of the sampled inputs.
    input_correlation: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: Adjustments made while resolving the configuration, and warnings.
    notes: list[str] = field(default_factory=list)

    @property
    def median(self) -> float:
        """Median Gordon value per share (the 50th percentile)."""
        return self.percentiles[50]

    @property
    def valid_values(self) -> np.ndarray:
        """Gordon values of the valid simulations (no NaN)."""
        return self.values[np.isfinite(self.values)]

    @property
    def valid_values_exit(self) -> np.ndarray:
        """Exit-multiple values of the valid simulations (no NaN)."""
        return self.values_exit[np.isfinite(self.values_exit)]

    @property
    def share_non_finite(self) -> float:
        """Share of simulations that are invalid."""
        return self.n_non_finite / self.n_simulations

    @property
    def mean_vs_base(self) -> float:
        """Mean relative to the base value, minus one (the convexity effect)."""
        return self.mean / self.base_value - 1.0 if self.base_value else math.nan

    @property
    def median_vs_base(self) -> float:
        """Median relative to the base value, minus one."""
        return self.median / self.base_value - 1.0 if self.base_value else math.nan

    def summary(self) -> pd.Series:
        """Headline statistics as one Series (for tables and export)."""
        rows: dict[str, float] = {
            "base_value": self.base_value,
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "standard_error": self.standard_error,
            **{f"p{p}": v for p, v in self.percentiles.items()},
            "price": self.price,
            "prob_above_price": self.prob_above_price,
            "n_simulations": float(self.n_simulations),
            "n_non_finite": float(self.n_non_finite),
        }
        return pd.Series(rows, name=self.ticker)

    def to_valuation_results(self) -> list[ValuationResult]:
        """One per-share range: low = P10, base = median, high = P90 (Gordon).

        The median, not the mean, is the central value: the mean is pulled up by
        the convexity of the Gordon formula (module docstring). The base-case DCF
        value, the mean and the remaining statistics are in ``details``.
        """
        result = ValuationResult(
            method=ValuationMethod.MONTE_CARLO,
            label="Monte Carlo DCF (P10 - P90)",
            low=self.percentiles[10],
            base=self.percentiles[50],
            high=self.percentiles[90],
            currency=self.currency,
            details={
                "range_definition": (
                    "low: 10th percentile, base: median, high: 90th percentile of the "
                    "simulated Gordon growth value per share"
                ),
                "n_simulations": self.n_simulations,
                "n_non_finite": self.n_non_finite,
                "seed": self.config.seed,
                "base_dcf_value": self.base_value,
                "mean": self.mean,
                "std": self.std,
                "standard_error": self.standard_error,
                "percentiles": dict(self.percentiles),
                "price": self.price,
                "prob_above_price": self.prob_above_price,
                "mean_vs_base": self.mean_vs_base,
                "median_vs_base": self.median_vs_base,
                "variance_contribution": self.variance_contribution.to_dict(),
                "exit_multiple_percentiles": dict(self.percentiles_exit),
                "exit_multiple_mean": self.mean_exit,
                "distributions": {
                    name: {
                        "kind": spec.kind,
                        "mean_offset": spec.mean,
                        "std": spec.std,
                        "low_offset": spec.low,
                        "high_offset": spec.high,
                    }
                    for name in DRIVERS
                    if (spec := self.config.spec(name)) is not None
                },
                "notes": list(self.notes),
            },
        )
        return [result]


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def _scenario_inputs(core_inputs: dict[str, Any], offsets: dict[str, np.ndarray]) -> dict[str, Any]:
    """Base-case core inputs with the four drivers replaced by scenario arrays.

    ``offsets[name]`` has shape ``(n,)``; growth and margin offsets are added to
    every forecast year (parallel shift).
    """
    growth_path = np.asarray(core_inputs["revenue_growth"], dtype=float)
    margin_path = np.asarray(core_inputs["ebitda_margin"], dtype=float)
    return {
        **core_inputs,
        "revenue_growth": growth_path[None, :] + offsets["revenue_growth"][:, None],
        "ebitda_margin": margin_path[None, :] + offsets["ebitda_margin"][:, None],
        "wacc": float(core_inputs["wacc"]) + offsets["wacc"],
        "terminal_growth": float(core_inputs["terminal_growth"]) + offsets["terminal_growth"],
    }


def _evaluate(scenario: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Value the scenarios in one core call; invalid scenarios become NaN.

    Returns (Gordon value per share, exit-multiple value per share).
    """
    output = dcf.dcf_core(**scenario)
    wacc = np.asarray(scenario["wacc"], dtype=float)
    spread = wacc - np.asarray(scenario["terminal_growth"], dtype=float)
    gordon = output.value_per_share_gordon
    exit_value = output.value_per_share_exit
    invalid = (
        ~(spread >= dcf.MIN_WACC_GROWTH_SPREAD)
        | ~(wacc > 0)
        | ~np.all(np.asarray(scenario["revenue_growth"], dtype=float) > -1.0, axis=1)
        | ~np.isfinite(gordon)
        | ~np.isfinite(exit_value)
    )
    return np.where(invalid, np.nan, gordon), np.where(invalid, np.nan, exit_value)


def _tornado(
    core_inputs: dict[str, Any],
    specs: dict[str, DistributionSpec],
    base_levels: dict[str, float],
    percentiles: tuple[float, float],
) -> pd.DataFrame:
    """One-at-a-time sensitivity of the Gordon value, in one core call.

    Each driver is set to the two ``percentiles`` (default P10 and P90) of its own
    marginal distribution (exact quantiles, not sample quantiles) while the other
    three stay at base.
    """
    probabilities = np.asarray(percentiles, dtype=float) / 100.0
    k = len(DRIVERS)
    offsets = {name: np.zeros(2 * k) for name in DRIVERS}
    quantiles: dict[str, np.ndarray] = {}
    for i, name in enumerate(DRIVERS):
        quantiles[name] = _quantile(specs[name], probabilities)
        offsets[name][2 * i : 2 * i + 2] = quantiles[name]
    values, _ = _evaluate(_scenario_inputs(core_inputs, offsets))
    frame = pd.DataFrame(
        {
            "input_low": [base_levels[name] + quantiles[name][0] for name in DRIVERS],
            "input_high": [base_levels[name] + quantiles[name][1] for name in DRIVERS],
            "value_low": values[0::2],
            "value_high": values[1::2],
        },
        index=pd.Index(DRIVERS, name="driver"),
    )
    frame["swing"] = (frame["value_high"] - frame["value_low"]).abs()
    return frame.sort_values("swing", ascending=False, kind="stable", na_position="last")


def run_monte_carlo(
    company: CompanyData,
    assumptions: DCFAssumptions | None = None,
    config: MonteCarloConfig | None = None,
    *,
    params: MarketParameters = _DEFAULT_PARAMS,
    peers: list[PeerSnapshot] | None = None,
    dcf_result: dcf.DCFResult | None = None,
) -> MonteCarloResult:
    """Monte Carlo simulation of the DCF value per share.

    * ``dcf_result`` given -> it is the base case (``assumptions``, ``params`` and
      ``peers`` are then ignored); otherwise ``dcf.run_dcf(company, assumptions,
      params, peers)`` is run first.
    * ``config=None`` -> :class:`MonteCarloConfig` defaults: 10 000 simulations,
      seed 42, truncated normals with company-informed standard deviations and a
      growth-margin correlation of +0.3.

    All simulations are valued in one call of ``dcf.dcf_core``. A fixed seed gives
    identical results. With all standard deviations exactly zero every simulation
    equals the base value exactly.

    Raises :class:`NotApplicableError` when the base DCF is not applicable (the
    error of ``run_dcf`` propagates) or when no simulation is valid, and
    ``ValueError`` for an inconsistent configuration (e.g. a correlation matrix
    that is not positive definite).
    """
    config = config if config is not None else MonteCarloConfig()
    if dcf_result is None:
        base = dcf.run_dcf(company, assumptions, params, peers)
    else:
        if dcf_result.ticker != company.ticker:
            raise ValueError(f"dcf_result belongs to {dcf_result.ticker}, not to {company.ticker}")
        base = dcf_result
    core_inputs = base.core_inputs
    notes: list[str] = []

    base_levels = {
        "revenue_growth": float(np.asarray(core_inputs["revenue_growth"], dtype=float)[0]),
        "ebitda_margin": float(np.asarray(core_inputs["ebitda_margin"], dtype=float)[0]),
        "wacc": float(core_inputs["wacc"]),
        "terminal_growth": float(core_inputs["terminal_growth"]),
    }

    # --- distributions -----------------------------------------------------------
    default_stds = default_driver_stds(company, base)
    specs: dict[str, DistributionSpec] = {}
    auto_bounds: dict[str, bool] = {}
    for name in DRIVERS:
        specs[name], auto_bounds[name] = _resolve_spec(
            name, config.spec(name), base_levels[name], default_stds[name]
        )
    _protect_spread(
        specs,
        auto_bounds,
        base_levels["wacc"] - base_levels["terminal_growth"],
        config.min_wacc_growth_spread,
        notes,
    )
    correlation = _correlation_matrix(config)
    resolved_config = replace(
        config,
        **specs,
        growth_margin_correlation=float(correlation[0, 1]),
        correlation_matrix=tuple(tuple(float(x) for x in row) for row in correlation),
    )

    # --- sampling (Gaussian copula) ------------------------------------------------
    n = int(config.n_simulations)
    rng = np.random.default_rng(config.seed)
    cholesky = np.linalg.cholesky(correlation)
    normals = rng.standard_normal((n, len(DRIVERS))) @ cholesky.T
    offsets = {name: _transform(specs[name], normals[:, i]) for i, name in enumerate(DRIVERS)}

    # --- valuation: one vectorized core call ----------------------------------------
    values, values_exit = _evaluate(_scenario_inputs(core_inputs, offsets))
    inputs = pd.DataFrame(
        {name: base_levels[name] + offsets[name] for name in DRIVERS},
        index=pd.RangeIndex(n, name="simulation"),
    )

    valid_mask = np.isfinite(values)
    n_valid = int(valid_mask.sum())
    n_non_finite = n - n_valid
    if n_valid == 0:
        raise NotApplicableError(
            "non-finite result: no simulation produced a finite value under the given "
            "distributions (WACC not sufficiently above terminal growth in every draw)"
        )
    if n_non_finite:
        notes.append(
            f"{n_non_finite} of {n} simulations ({n_non_finite / n:.2%}) are invalid "
            "(WACC - g below the minimum spread, non-positive WACC or growth <= -100 %) "
            "and excluded from the statistics"
        )
    valid = values[valid_mask]
    valid_exit = values_exit[valid_mask]
    price = float(base.price)
    std = _std(valid)

    return MonteCarloResult(
        ticker=base.ticker,
        currency=base.currency,
        price=price,
        base_value=float(base.value_per_share_gordon),
        n_simulations=n,
        values=values,
        values_exit=values_exit,
        inputs=inputs,
        percentiles=_percentiles(valid),
        mean=_mean(valid),
        std=std,
        standard_error=std / math.sqrt(n_valid),
        prob_above_price=float(np.mean(valid > price)),
        tornado=_tornado(core_inputs, specs, base_levels, config.tornado_percentiles),
        variance_contribution=_variance_contribution(inputs[valid_mask], valid),
        convergence=_convergence(valid, config.convergence_points),
        config=resolved_config,
        base_value_exit=float(base.value_per_share_exit),
        percentiles_exit=_percentiles(valid_exit),
        mean_exit=_mean(valid_exit),
        std_exit=_std(valid_exit),
        prob_above_price_exit=float(np.mean(valid_exit > price)),
        n_non_finite=n_non_finite,
        n_valid=n_valid,
        base_inputs=base_levels,
        input_correlation=_input_correlation(inputs),
        notes=notes,
    )
