"""Monte Carlo on top of the DCF: sampling, statistics and consistency with the DCF."""

from __future__ import annotations

import math
import time
from dataclasses import replace

import numpy as np
import pytest
from scipy import stats

from invariants import assert_monte_carlo_invariants
from support import make_company
from valuation_engine.core.types import DCFAssumptions, NotApplicableError
from valuation_engine.models import dcf
from valuation_engine.models import monte_carlo as mc
from valuation_engine.models.monte_carlo import DistributionSpec as Spec
from valuation_engine.models.monte_carlo import MonteCarloConfig as Config

KINDS = ("normal", "truncnormal", "triangular")

ASSUMPTIONS = DCFAssumptions(
    revenue_growth=(0.08, 0.06, 0.04, 0.03, 0.02),
    ebitda_margin=(0.20, 0.21, 0.22, 0.22, 0.22),
    da_pct_revenue=0.04,
    capex_pct_revenue=0.05,
    nwc_pct_revenue=0.10,
    tax_rate=0.22,
    wacc=0.09,
    terminal_growth=0.02,
    exit_multiple=9.0,
)


@pytest.fixture(scope="module")
def company():
    return make_company()


@pytest.fixture(scope="module")
def base(company) -> dcf.DCFResult:
    return dcf.run_dcf(company, ASSUMPTIONS)


def simulate(company, base, **config: object) -> mc.MonteCarloResult:
    return mc.run_monte_carlo(company, config=Config(**config), dcf_result=base)  # type: ignore[arg-type]


def specs(kind: str, growth: float, margin: float, wacc: float, g: float) -> dict[str, Spec]:
    return {
        "revenue_growth": Spec(kind, std=growth),
        "ebitda_margin": Spec(kind, std=margin),
        "wacc": Spec(kind, std=wacc),
        "terminal_growth": Spec(kind, std=g),
    }


def shifted(growth: float = 0.0, margin: float = 0.0, wacc: float = 0.0, g: float = 0.0):
    """The base assumptions with a parallel shift of the four drivers."""
    return replace(
        ASSUMPTIONS,
        revenue_growth=tuple(x + growth for x in ASSUMPTIONS.revenue_growth),
        ebitda_margin=tuple(x + margin for x in ASSUMPTIONS.ebitda_margin),
        wacc=ASSUMPTIONS.wacc + wacc,
        terminal_growth=ASSUMPTIONS.terminal_growth + g,
    )


# ------------------------------------------------------------ zero variance


@pytest.mark.parametrize("kind", KINDS)
def test_zero_variance_reproduces_the_base_dcf_exactly(company, base, kind: str) -> None:
    result = simulate(company, base, n_simulations=500, **specs(kind, 0.0, 0.0, 0.0, 0.0))
    assert np.all(result.values == base.value_per_share_gordon)  # exact, not approximate
    assert np.all(result.values_exit == base.value_per_share_exit)
    assert result.mean == base.value_per_share_gordon
    assert result.std == 0.0 and result.standard_error == 0.0
    assert set(result.percentiles.values()) == {base.value_per_share_gordon}
    assert (result.variance_contribution == 0.0).all()
    assert (result.tornado["swing"] == 0.0).all()
    assert result.n_non_finite == 0
    (row,) = result.to_valuation_results()
    assert row.low == row.base == row.high == base.value_per_share_gordon


def test_median_converges_to_the_base_value_as_the_variance_shrinks(company, base) -> None:
    gaps, spreads = [], []
    for scale in (1.0, 0.3, 0.1, 0.03, 0.01):
        result = simulate(
            company,
            base,
            n_simulations=20_000,
            **specs("truncnormal", 0.02 * scale, 0.02 * scale, 0.01 * scale, 0.005 * scale),
        )
        gaps.append(abs(result.median / base.value_per_share_gordon - 1.0))
        spreads.append(result.std)
    assert gaps == sorted(gaps, reverse=True)
    assert spreads == sorted(spreads, reverse=True)
    assert gaps[-1] < 1e-4
    assert spreads[-1] < 0.02 * spreads[0]


def test_mean_exceeds_median_because_the_gordon_value_is_convex(company, base) -> None:
    result = simulate(company, base, n_simulations=50_000)
    assert result.mean > result.median
    assert result.mean > base.value_per_share_gordon
    assert result.mean_vs_base == pytest.approx(result.mean / result.base_value - 1)
    assert result.median_vs_base == pytest.approx(result.median / result.base_value - 1)
    assert abs(result.median_vs_base) < 0.02


# ---------------------------------------------------------------------- seeds


def test_fixed_seed_is_reproducible_and_seeds_differ(company, base) -> None:
    first = simulate(company, base, n_simulations=2_000, seed=7)
    second = simulate(company, base, n_simulations=2_000, seed=7)
    other = simulate(company, base, n_simulations=2_000, seed=8)
    assert np.array_equal(first.values, second.values)
    assert np.array_equal(first.values_exit, second.values_exit)
    assert first.inputs.equals(second.inputs)
    assert first.percentiles == second.percentiles
    assert not np.array_equal(first.values, other.values)
    assert first.mean != other.mean
    fresh = [simulate(company, base, n_simulations=200, seed=None).mean for _ in range(2)]
    assert fresh[0] != fresh[1]
    assert Config().seed == 42 and Config().n_simulations == 10_000


def test_larger_run_with_the_same_seed_extends_the_smaller_one(company, base) -> None:
    small = simulate(company, base, n_simulations=1_000, seed=3)
    large = simulate(company, base, n_simulations=4_000, seed=3)
    assert np.array_equal(small.values, large.values[:1_000])


# -------------------------------------------------------------- standard error


def test_standard_error_is_std_over_sqrt_n_and_scales_with_one_over_sqrt_n(company, base) -> None:
    small = simulate(company, base, n_simulations=2_000)
    large = simulate(company, base, n_simulations=32_000)
    for result in (small, large):
        assert result.standard_error == pytest.approx(
            np.std(result.values, ddof=1) / math.sqrt(result.n_simulations), rel=1e-12
        )
    assert small.standard_error / large.standard_error == pytest.approx(4.0, rel=0.10)


def test_standard_error_predicts_the_spread_of_the_mean_across_seeds(company, base) -> None:
    runs = [simulate(company, base, n_simulations=2_000, seed=seed) for seed in range(120)]
    spread_of_means = float(np.std([r.mean for r in runs], ddof=1))
    reported = float(np.mean([r.standard_error for r in runs]))
    assert spread_of_means == pytest.approx(reported, rel=0.25)


# ----------------------------------------------------------------- correlation


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("rho", [0.6, -0.4])
def test_growth_margin_correlation_is_near_the_target(company, base, kind: str, rho: float) -> None:
    result = simulate(
        company,
        base,
        n_simulations=100_000,
        growth_margin_correlation=rho,
        **specs(kind, 0.02, 0.02, 0.005, 0.002),
    )
    sampled = np.corrcoef(result.inputs["revenue_growth"], result.inputs["ebitda_margin"])[0, 1]
    # exact for normal marginals, marginally weaker for bounded ones (Gaussian copula)
    assert sampled == pytest.approx(rho, abs=0.01 if kind == "normal" else 0.02)
    assert abs(sampled) <= abs(rho) + 0.01
    assert result.input_correlation.loc["revenue_growth", "ebitda_margin"] == pytest.approx(
        sampled, abs=1e-12
    )
    others = np.corrcoef(result.inputs.to_numpy(), rowvar=False)
    assert abs(others[0, 2]) < 0.015 and abs(others[1, 3]) < 0.015 and abs(others[2, 3]) < 0.015
    assert result.config.growth_margin_correlation == rho


def test_default_correlation_and_full_correlation_matrix(company, base) -> None:
    default = simulate(company, base, n_simulations=100_000)
    assert default.input_correlation.loc["revenue_growth", "ebitda_margin"] == pytest.approx(
        0.3, abs=0.02
    )
    matrix = [
        [1.0, 0.2, 0.0, 0.0],
        [0.2, 1.0, -0.3, 0.0],
        [0.0, -0.3, 1.0, 0.5],
        [0.0, 0.0, 0.5, 1.0],
    ]
    result = simulate(
        company,
        base,
        n_simulations=100_000,
        correlation_matrix=matrix,
        growth_margin_correlation=0.9,  # ignored when the matrix is given
        **specs("normal", 0.02, 0.02, 0.005, 0.002),
    )
    sampled = np.corrcoef(result.inputs.to_numpy(), rowvar=False)
    assert sampled == pytest.approx(np.array(matrix), abs=0.012)
    assert result.config.growth_margin_correlation == pytest.approx(0.2)


@pytest.mark.parametrize(
    "matrix",
    [
        [[1, 0.9, 0.9, 0], [0.9, 1, -0.9, 0], [0.9, -0.9, 1, 0], [0, 0, 0, 1]],  # not PSD
        [[1, 0.5, 0, 0], [0.4, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],  # not symmetric
        [[1, 2, 0, 0], [2, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],  # |rho| > 1
        [[2, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],  # diagonal not 1
        [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],  # singular
        [[1, math.nan, 0, 0], [math.nan, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # wrong shape
        "not a matrix",
    ],
)
def test_invalid_correlation_matrix_raises(company, base, matrix: object) -> None:
    with pytest.raises(ValueError):
        simulate(company, base, n_simulations=10, correlation_matrix=matrix)


@pytest.mark.parametrize("rho", [1.0, -1.0, 1.5, math.nan])
def test_invalid_growth_margin_correlation_raises(company, base, rho: float) -> None:
    with pytest.raises(ValueError):
        simulate(company, base, n_simulations=10, growth_margin_correlation=rho)


# ------------------------------------------------------------------- marginals


def test_normal_marginal_matches_its_mean_and_std(company, base) -> None:
    result = simulate(
        company, base, n_simulations=200_000, wacc=Spec("normal", mean=0.002, std=0.004)
    )
    offsets = result.inputs["wacc"] - 0.09
    assert offsets.mean() == pytest.approx(0.002, abs=4 * 0.004 / math.sqrt(200_000))
    assert offsets.std() == pytest.approx(0.004, rel=0.01)
    assert stats.skew(offsets) == pytest.approx(0.0, abs=0.03)


def test_truncated_normal_respects_its_bounds_and_has_the_textbook_std(company, base) -> None:
    result = simulate(company, base, n_simulations=200_000, ebitda_margin=Spec(std=0.02))
    offsets = result.inputs["ebitda_margin"] - 0.20
    assert offsets.min() >= -0.04 and offsets.max() <= 0.04  # +/- 2 sigma
    assert offsets.min() < -0.039 and offsets.max() > 0.039
    assert offsets.mean() == pytest.approx(0.0, abs=3e-4)
    # variance of a standard normal truncated at +/- a: 1 - 2 a phi(a) / (2 Phi(a) - 1)
    a = 2.0
    phi = math.exp(-a * a / 2) / math.sqrt(2 * math.pi)
    big_phi = 0.5 * (1 + math.erf(a / math.sqrt(2)))
    factor = math.sqrt(1 - 2 * a * phi / (2 * big_phi - 1))
    assert factor == pytest.approx(0.8796, abs=1e-4)
    assert offsets.std() == pytest.approx(0.02 * factor, rel=0.01)
    spec = result.config.ebitda_margin
    assert (spec.kind, spec.low, spec.high, spec.std) == ("truncnormal", -0.04, 0.04, 0.02)


def test_explicit_and_one_sided_truncation_bounds(company, base) -> None:
    result = simulate(
        company,
        base,
        n_simulations=50_000,
        revenue_growth=Spec("truncnormal", std=0.03, low=-0.01, high=0.05),
        ebitda_margin=Spec("truncnormal", std=0.02, low=0.0),  # only a floor
    )
    growth = result.inputs["revenue_growth"] - 0.08
    assert growth.min() >= -0.01 and growth.max() <= 0.05
    assert growth.mean() > 0.005  # the asymmetric window shifts the mean up
    margin = result.inputs["ebitda_margin"] - 0.20
    assert margin.min() >= 0.0 and margin.max() > 0.06
    # the mean of a half-normal is sigma * sqrt(2 / pi)
    assert margin.mean() == pytest.approx(0.02 * math.sqrt(2 / math.pi), rel=0.02)


def test_triangular_marginals(company, base) -> None:
    result = simulate(
        company,
        base,
        n_simulations=200_000,
        revenue_growth=Spec("triangular", std=0.02),  # symmetric, half-width sqrt(6) sigma
        wacc=Spec("triangular", mean=0.09, low=0.08, high=0.11, absolute=True),
    )
    growth = result.inputs["revenue_growth"] - 0.08
    half = math.sqrt(6) * 0.02
    assert growth.min() >= -half and growth.max() <= half
    assert growth.mean() == pytest.approx(0.0, abs=3e-4)
    assert growth.std() == pytest.approx(0.02, rel=0.01)
    wacc = result.inputs["wacc"]
    low, mode, high = 0.08, 0.09, 0.11
    assert wacc.min() >= low and wacc.max() <= high
    assert wacc.mean() == pytest.approx((low + mode + high) / 3, abs=1e-4)
    variance = (low**2 + mode**2 + high**2 - low * mode - low * high - mode * high) / 18
    assert wacc.std() == pytest.approx(math.sqrt(variance), rel=0.01)
    # the mode: the density peaks at 9 %
    histogram, edges = np.histogram(wacc, bins=30, range=(low, high))
    assert edges[np.argmax(histogram)] == pytest.approx(mode, abs=0.002)
    assert result.config.wacc.absolute is False  # resolved to offsets from the base
    assert result.config.wacc.low == pytest.approx(-0.01)


def test_absolute_levels_are_translated_to_offsets(company, base) -> None:
    result = simulate(
        company,
        base,
        n_simulations=50_000,
        terminal_growth=Spec("normal", mean=0.025, std=0.001, absolute=True),
        wacc=Spec("normal", std=0.0),
    )
    assert result.inputs["terminal_growth"].mean() == pytest.approx(0.025, abs=2e-5)
    assert result.config.terminal_growth.mean == pytest.approx(0.005)
    assert (result.inputs["wacc"] == 0.09).all()
    assert result.median > base.value_per_share_gordon  # centred above the base growth


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "lognormal"},
        {"std": -0.01},
        {"std": math.nan},
        {"mean": math.inf},
        {"kind": "normal", "low": -0.01},
        {"kind": "triangular", "low": -0.01},
        {"low": 0.02, "high": 0.01},
    ],
)
def test_distribution_spec_validation(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        Spec(**kwargs)


def test_inconsistent_distributions_are_rejected_when_resolved(company, base) -> None:
    with pytest.raises(ValueError, match="mode"):
        simulate(company, base, wacc=Spec("triangular", mean=0.05, low=-0.01, high=0.01))
    with pytest.raises(ValueError, match="no probability mass"):
        simulate(company, base, wacc=Spec("truncnormal", std=0.001, low=0.05, high=0.06))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_simulations": 0},
        {"n_simulations": -5},
        {"n_simulations": 1000.0},
        {"n_simulations": True},
        {"min_wacc_growth_spread": 0.001},
        {"tornado_percentiles": (90.0, 10.0)},
        {"tornado_percentiles": (0.0, 100.0)},
        {"convergence_points": 1},
    ],
)
def test_config_validation(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        Config(**kwargs)


# ------------------------------------------------- consistency with the DCF


def test_every_simulation_is_an_ordinary_dcf_with_shifted_drivers(company, base) -> None:
    result = simulate(company, base, n_simulations=300, seed=5)
    assert result.base_inputs == {
        "revenue_growth": 0.08,
        "ebitda_margin": 0.20,
        "wacc": 0.09,
        "terminal_growth": 0.02,
    }
    for i in (0, 1, 17, 150, 299):
        draw = result.inputs.iloc[i]
        direct = dcf.run_dcf(
            company,
            shifted(
                growth=draw["revenue_growth"] - 0.08,
                margin=draw["ebitda_margin"] - 0.20,
                wacc=draw["wacc"] - 0.09,
                g=draw["terminal_growth"] - 0.02,
            ),
        )
        assert result.values[i] == pytest.approx(direct.value_per_share_gordon, rel=1e-10)
        assert result.values_exit[i] == pytest.approx(direct.value_per_share_exit, rel=1e-10)


@pytest.mark.parametrize("kind", ["normal", "truncnormal"])
def test_tornado_matches_direct_revaluation_at_the_p10_and_p90_of_each_driver(
    company, base, kind: str
) -> None:
    sigma = {
        "revenue_growth": 0.02,
        "ebitda_margin": 0.015,
        "wacc": 0.008,
        "terminal_growth": 0.004,
    }
    result = simulate(
        company,
        base,
        n_simulations=1_000,
        **{name: Spec(kind, std=std) for name, std in sigma.items()},
    )
    keyword = {
        "revenue_growth": "growth",
        "ebitda_margin": "margin",
        "wacc": "wacc",
        "terminal_growth": "g",
    }
    swings = {}
    for name, std in sigma.items():
        if kind == "normal":
            low, high = stats.norm.ppf([0.10, 0.90], scale=std)
        else:
            low, high = stats.truncnorm.ppf([0.10, 0.90], -2.0, 2.0, scale=std)
        row = result.tornado.loc[name]
        assert row["input_low"] == pytest.approx(result.base_inputs[name] + low, rel=1e-9)
        assert row["input_high"] == pytest.approx(result.base_inputs[name] + high, rel=1e-9)
        value_low = dcf.run_dcf(company, shifted(**{keyword[name]: low})).value_per_share_gordon
        value_high = dcf.run_dcf(company, shifted(**{keyword[name]: high})).value_per_share_gordon
        assert row["value_low"] == pytest.approx(value_low, rel=1e-9)
        assert row["value_high"] == pytest.approx(value_high, rel=1e-9)
        assert row["swing"] == pytest.approx(abs(value_high - value_low), rel=1e-9)
        swings[name] = abs(value_high - value_low)
    assert list(result.tornado.index) == sorted(swings, key=swings.get, reverse=True)
    # a higher WACC lowers the value, the other three raise it
    assert result.tornado.loc["wacc", "value_low"] > result.tornado.loc["wacc", "value_high"]
    for name in ("revenue_growth", "ebitda_margin", "terminal_growth"):
        assert result.tornado.loc[name, "value_low"] < result.tornado.loc[name, "value_high"]


def test_custom_tornado_percentiles(company, base) -> None:
    result = simulate(
        company,
        base,
        n_simulations=100,
        tornado_percentiles=(25.0, 75.0),
        **specs("normal", 0.02, 0.02, 0.01, 0.005),
    )
    low, high = stats.norm.ppf([0.25, 0.75], scale=0.01)
    assert result.tornado.loc["wacc", "input_low"] == pytest.approx(0.09 + low, rel=1e-9)
    assert result.tornado.loc["wacc", "input_high"] == pytest.approx(0.09 + high, rel=1e-9)


def test_exit_value_does_not_depend_on_terminal_growth(company, base) -> None:
    result = simulate(
        company,
        base,
        n_simulations=2_000,
        revenue_growth=Spec(std=0.0),
        ebitda_margin=Spec(std=0.0),
        wacc=Spec(std=0.0),
        terminal_growth=Spec(std=0.004),
    )
    assert np.ptp(result.values_exit) == 0.0
    assert result.std_exit == 0.0
    assert result.std > 0
    assert result.variance_contribution["terminal_growth"] == pytest.approx(1.0)
    assert result.variance_contribution.sum() == pytest.approx(1.0)


def test_base_case_is_run_when_no_dcf_result_is_given(company, base) -> None:
    config = Config(n_simulations=500, seed=1)
    via_assumptions = mc.run_monte_carlo(company, ASSUMPTIONS, config)
    via_result = mc.run_monte_carlo(company, config=config, dcf_result=base)
    assert np.array_equal(via_assumptions.values, via_result.values)
    assert via_assumptions.base_value == base.value_per_share_gordon
    with pytest.raises(ValueError, match="belongs to"):
        mc.run_monte_carlo(make_company("OTHER.ST"), config=config, dcf_result=base)
    with pytest.raises(NotApplicableError, match="financial sector"):
        mc.run_monte_carlo(make_company(sector="Financial Services"))


# ------------------------------------------------------------------ statistics


def test_statistics_are_recomputed_from_the_raw_values(company, base) -> None:
    result = simulate(company, base, n_simulations=5_000)
    assert_monte_carlo_invariants(result, base)
    values = result.values
    assert result.percentiles[10] == pytest.approx(np.percentile(values, 10), rel=1e-12)
    assert result.median == result.percentiles[50]
    assert result.prob_above_price == float((values > 10.0).mean())
    assert result.prob_above_price_exit == float((result.values_exit > 10.0).mean())
    assert result.percentiles_exit[50] == pytest.approx(np.median(result.values_exit), rel=1e-12)
    assert result.mean_exit == pytest.approx(result.values_exit.mean(), rel=1e-12)
    summary = result.summary()
    assert summary["mean"] == result.mean and summary["p90"] == result.percentiles[90]
    assert summary["n_simulations"] == 5_000
    assert result.share_non_finite == 0.0
    assert np.array_equal(result.valid_values, values)
    (row,) = result.to_valuation_results()
    assert row.details["base_dcf_value"] == base.value_per_share_gordon
    assert row.details["n_simulations"] == 5_000
    assert set(row.details["distributions"]) == set(mc.DRIVERS)


def test_variance_contributions_sum_to_one_and_rank_the_drivers(company, base) -> None:
    result = simulate(company, base, n_simulations=20_000)
    shares = result.variance_contribution
    assert shares.sum() == pytest.approx(1.0)
    assert (shares > 0).all()
    wacc_only = simulate(
        company, base, n_simulations=5_000, **specs("truncnormal", 0.0, 0.0, 0.01, 0.0)
    )
    assert wacc_only.variance_contribution["wacc"] == pytest.approx(1.0)
    assert wacc_only.variance_contribution.drop("wacc").sum() == 0.0


def test_convergence_table_is_the_running_statistics_of_the_simulations(company, base) -> None:
    result = simulate(company, base, n_simulations=3_000)
    table = result.convergence
    assert table["n"].is_monotonic_increasing and table["n"].is_unique
    assert table["n"].iloc[0] == 10 and table["n"].iloc[-1] == 3_000
    for _, row in table.iloc[[0, len(table) // 2, -1]].iterrows():
        head = result.values[: int(row["n"])]
        assert row["running_mean"] == pytest.approx(head.mean(), rel=1e-12)
        assert row["running_median"] == pytest.approx(np.median(head), rel=1e-12)
        assert row["standard_error"] == pytest.approx(
            head.std(ddof=1) / math.sqrt(len(head)), rel=1e-12
        )


@pytest.mark.parametrize("n", [1, 2, 3, 9])
def test_tiny_runs_do_not_break(company, base, n: int) -> None:
    result = simulate(company, base, n_simulations=n)
    assert result.values.shape == (n,)
    assert math.isfinite(result.mean) and math.isfinite(result.standard_error)
    assert (n > 1) is (result.std > 0)
    (row,) = result.to_valuation_results()
    assert row.low <= row.base <= row.high


# ----------------------------------------------------- keeping WACC above g


def test_automatic_bounds_are_narrowed_to_protect_the_spread(company) -> None:
    tight = dcf.run_dcf(company, replace(ASSUMPTIONS, wacc=0.045, terminal_growth=0.02))
    result = mc.run_monte_carlo(company, config=Config(n_simulations=20_000), dcf_result=tight)
    spread = result.inputs["wacc"] - result.inputs["terminal_growth"]
    floor = min(0.015, 0.025 / 2)  # min(config floor, half of the base spread)
    assert spread.min() >= floor - 1e-12
    assert result.n_non_finite == 0
    assert any("narrowed" in note for note in result.notes)
    # default widths 2 pp and 1 pp, both scaled by the same factor
    scale = 1 - (floor - (0.025 - 0.02 - 0.01)) / 0.03
    assert result.config.wacc.low == pytest.approx(-0.02 * scale)
    assert result.config.terminal_growth.high == pytest.approx(0.01 * scale)
    assert_monte_carlo_invariants(result, tight)


def test_wide_base_spread_needs_no_narrowing(company, base) -> None:
    result = simulate(company, base, n_simulations=1_000)
    assert result.notes == []
    assert (result.config.wacc.low, result.config.wacc.high) == (-0.02, 0.02)
    assert (result.config.terminal_growth.low, result.config.terminal_growth.high) == (-0.01, 0.01)


def test_invalid_draws_are_nan_counted_and_excluded(company, base) -> None:
    result = simulate(company, base, n_simulations=20_000, wacc=Spec("normal", std=0.05))
    spread = result.inputs["wacc"] - result.inputs["terminal_growth"]
    expected_invalid = (spread < dcf.MIN_WACC_GROWTH_SPREAD) | (result.inputs["wacc"] <= 0)
    assert expected_invalid.sum() > 500
    assert np.array_equal(np.isnan(result.values), expected_invalid.to_numpy())
    assert np.array_equal(np.isnan(result.values_exit), expected_invalid.to_numpy())
    assert result.n_non_finite == int(expected_invalid.sum())
    assert result.n_valid == 20_000 - result.n_non_finite
    assert result.share_non_finite == pytest.approx(result.n_non_finite / 20_000)
    valid = result.values[~expected_invalid.to_numpy()]
    assert result.mean == pytest.approx(valid.mean(), rel=1e-12)
    assert result.standard_error == pytest.approx(
        valid.std(ddof=1) / math.sqrt(len(valid)), rel=1e-12
    )
    assert any("invalid" in note for note in result.notes)
    assert_monte_carlo_invariants(result, base)


def test_no_valid_simulation_is_not_applicable(company, base) -> None:
    with pytest.raises(NotApplicableError, match="no simulation produced a finite value"):
        simulate(
            company,
            base,
            n_simulations=100,
            wacc=Spec("normal", mean=0.01, std=0.0, absolute=True),
        )


def test_growth_at_or_below_minus_100_percent_is_invalid(company, base) -> None:
    result = simulate(
        company, base, n_simulations=5_000, revenue_growth=Spec("normal", mean=-1.0, std=0.05)
    )
    hit = (result.inputs["revenue_growth"] + (0.02 - 0.08)) <= -1.0  # the lowest year of the path
    assert 0 < hit.sum() < 5_000
    assert np.isnan(result.values[hit.to_numpy()]).all()
    assert np.isfinite(result.values[~hit.to_numpy()]).all()


# ------------------------------------------------------------ default stds


def test_default_standard_deviations_follow_the_documented_recipe() -> None:
    target = make_company(
        years=(2021, 2022, 2023, 2024, 2025),
        revenue=(1000.0, 1100.0, 1155.0, 1270.5, 1334.025),  # +10 %, +5 %, +10 %, +5 %
        margin=(0.20, 0.24, 0.20, 0.24, 0.20),
    )
    result = dcf.run_dcf(target)
    stds = mc.default_driver_stds(target, result)
    growth_vol = float(np.std([0.10, 0.05, 0.10, 0.05], ddof=1))
    margin_vol = float(np.std([0.20, 0.24, 0.20, 0.24, 0.20], ddof=1))
    assert stds["revenue_growth"] == pytest.approx(0.5 * growth_vol, rel=1e-9)
    assert stds["ebitda_margin"] == pytest.approx(margin_vol, rel=1e-9)
    assert stds["wacc"] == 0.01
    assert stds["terminal_growth"] == 0.005


def test_default_standard_deviations_are_clipped_and_capped() -> None:
    steady = make_company(revenue=(1000.0, 1050.0, 1102.5, 1157.625), margin=0.2)
    stds = mc.default_driver_stds(steady, dcf.run_dcf(steady))
    assert stds["revenue_growth"] == 0.01 and stds["ebitda_margin"] == 0.01  # the floor
    wild = make_company(revenue=(100.0, 400.0, 200.0, 900.0), margin=(0.05, 0.60, 0.10, 0.50))
    stds = mc.default_driver_stds(wild, dcf.run_dcf(wild))
    assert stds["revenue_growth"] == 0.05 and stds["ebitda_margin"] == 0.05  # the cap
    thin = make_company(margin=(0.01, 0.12, 0.02, 0.03), da=0.001, capex=0.001, nwc=0.0)
    base_case = dcf.run_dcf(thin)
    stds = mc.default_driver_stds(thin, base_case)
    lowest = min(base_case.assumptions.ebitda_margin)
    assert stds["ebitda_margin"] == pytest.approx(0.5 * lowest)  # margin stays positive at 2 sigma
    two_years = make_company(years=(2024, 2025), revenue=(900.0, 1000.0))
    stds = mc.default_driver_stds(two_years, dcf.run_dcf(two_years))
    assert stds["revenue_growth"] == 0.03  # one growth observation: the fallback


# -------------------------------------------------------------------- speed


def test_two_hundred_thousand_simulations_are_vectorised(company, base) -> None:
    simulate(company, base, n_simulations=1_000)  # warm-up
    started = time.perf_counter()
    result = simulate(company, base, n_simulations=200_000)
    elapsed = time.perf_counter() - started
    assert result.n_valid == 200_000
    # a Python-level loop over 200,000 DCFs would take minutes; this takes a fraction of a second
    assert elapsed < 10.0


def test_result_carries_the_fields_of_the_plan_contract(company, base) -> None:
    """PLAN.md, section 'MonteCarloResult'."""
    result = simulate(company, base, n_simulations=1_000)
    assert (result.ticker, result.currency, result.price) == ("TEST.ST", "SEK", 10.0)
    assert result.n_simulations == 1_000
    assert isinstance(result.values, np.ndarray) and isinstance(result.values_exit, np.ndarray)
    assert list(result.inputs.columns) == [
        "revenue_growth",
        "ebitda_margin",
        "wacc",
        "terminal_growth",
    ]
    assert list(result.percentiles) == [5, 10, 25, 50, 75, 90, 95]
    for name in ("mean", "std", "standard_error", "prob_above_price", "base_value"):
        assert isinstance(getattr(result, name), float)
    assert list(result.tornado.columns) == [
        "input_low",
        "input_high",
        "value_low",
        "value_high",
        "swing",
    ]
    assert result.tornado.index.name == "driver"
    assert list(result.variance_contribution.index) == list(mc.DRIVERS)
    assert list(result.convergence.columns) == [
        "n",
        "running_mean",
        "running_median",
        "standard_error",
    ]
    assert isinstance(result.config, Config)
    assert mc.DRIVERS == ("revenue_growth", "ebitda_margin", "wacc", "terminal_growth")
