"""Trading comps: multiple statistics, the equity bridge, peer selection and fallbacks."""

from __future__ import annotations

import math
import statistics

import numpy as np
import pandas as pd
import pytest

from support import FIXTURE_TICKERS, linear_percentile, make_company, make_peer
from valuation_engine.core.types import NotApplicableError, ValuationMethod
from valuation_engine.models import comps

REL = 1e-10
EV_MULTIPLES = [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]


def target(**kwargs: object):
    kwargs.setdefault("name", "Target AB")
    return make_company("TGT-A.ST", **kwargs)  # type: ignore[arg-type]


def small_universe() -> list:
    """Seven same-sector peers with EV/EBITDA 6..12, EV/Sales 1.0..1.6, P/E 10..16."""
    peers = []
    for i, multiple in enumerate(EV_MULTIPLES):
        market_cap = 1000.0 + 10.0 * i
        peers.append(
            make_peer(
                f"P{i}.ST",
                market_cap=market_cap,
                enterprise_value=multiple * 100.0,
                ebitda=100.0,
                revenue=multiple * 100.0 / (1.0 + 0.1 * i),
                net_income=market_cap / (10.0 + i),
                revenue_growth=0.01 * i,
                return_on_equity=0.10 + 0.01 * i,
                beta=0.8 + 0.05 * i,
                total_debt=100.0 + 20.0 * i,
            )
        )
    return peers


def ruler_universe(n: int = 16) -> list:
    """Peers that differ only in size: P00 is nearest to a 1,000 market cap target."""
    return [
        make_peer(
            f"P{i:02d}.ST",
            market_cap=1000.0 * 1.3 ** (i + 1),
            enterprise_value=(8.0 + 0.25 * i) * 100.0,
            ebitda=100.0,
        )
        for i in range(n)
    ]


# ------------------------------------------------------------ multiple statistics


def test_quartiles_of_a_known_universe() -> None:
    result = comps.run_comps(target(), small_universe())
    summary = result.multiples_summary
    expected_columns = ["n", "min", "p25", "median", "mean", "p75", "max"]
    assert list(summary.columns) == expected_columns
    assert list(summary.index) == ["ev_ebitda", "ev_sales", "pe"]

    known = {
        "ev_ebitda": EV_MULTIPLES,
        "ev_sales": [1.0 + 0.1 * i for i in range(7)],
        "pe": [10.0 + i for i in range(7)],
    }
    for name, multiples in known.items():
        row = summary.loc[name]
        assert row["n"] == 7
        assert row["min"] == pytest.approx(min(multiples), rel=REL)
        assert row["max"] == pytest.approx(max(multiples), rel=REL)
        assert row["mean"] == pytest.approx(sum(multiples) / 7, rel=REL)
        assert row["p25"] == pytest.approx(linear_percentile(multiples, 25), rel=REL)
        assert row["median"] == pytest.approx(linear_percentile(multiples, 50), rel=REL)
        assert row["p75"] == pytest.approx(linear_percentile(multiples, 75), rel=REL)
    # and the plain hand numbers for EV/EBITDA 6..12
    assert summary.loc["ev_ebitda", ["p25", "median", "p75"]].tolist() == pytest.approx(
        [7.5, 9.0, 10.5]
    )


def test_implied_values_and_the_equity_bridge_by_hand() -> None:
    company = target(minority=30.0, preferred=20.0)  # debt 300, cash 50, 100 shares
    result = comps.run_comps(company, small_universe())
    claims = (300.0 - 50.0) + 30.0 + 20.0
    assert result.equity_bridge == {
        "total_debt": 300.0,
        "cash": 50.0,
        "minority_interest": 30.0,
        "preferred_equity": 20.0,
        "net_debt": 250.0,
    }
    implied = result.implied_values

    ebitda, revenue = 0.2 * 1100.0, 1100.0
    net_income = (ebitda - 0.04 * 1100.0 - 15.0) * 0.8
    cases = {
        "ev_ebitda": (ebitda, [7.5, 9.0, 10.5]),
        "ev_sales": (revenue, [1.15, 1.3, 1.45]),
    }
    for name, (metric, quartiles) in cases.items():
        row = implied.loc[name]
        assert row["metric_value"] == pytest.approx(metric, rel=REL)
        assert [row["multiple_low"], row["multiple_base"], row["multiple_high"]] == pytest.approx(
            quartiles, rel=REL
        )
        for suffix, q in zip(("low", "base", "high"), quartiles, strict=True):
            assert row[f"ev_{suffix}"] == pytest.approx(q * metric, rel=REL)
            assert row[f"equity_{suffix}"] == pytest.approx(q * metric - claims, rel=REL)
            assert row[f"per_share_{suffix}"] == pytest.approx(
                (q * metric - claims) / 100.0, rel=REL
            )
        assert row["base_to_price"] == pytest.approx(row["per_share_base"] / 10.0, rel=REL)
        assert not row["floored"]

    # P/E gives equity directly; the bridge is ADDED to get to enterprise value
    row = implied.loc["pe"]
    assert row["metric_value"] == pytest.approx(net_income, rel=REL)
    for suffix, q in zip(("low", "base", "high"), (11.5, 13.0, 14.5), strict=True):
        assert row[f"equity_{suffix}"] == pytest.approx(q * net_income, rel=REL)
        assert row[f"ev_{suffix}"] == pytest.approx(q * net_income + claims, rel=REL)
        assert row[f"per_share_{suffix}"] == pytest.approx(q * net_income / 100.0, rel=REL)


def test_target_multiples_are_the_companys_own_current_multiples() -> None:
    result = comps.run_comps(target(), small_universe())
    ev = 1000.0 + 300.0 - 50.0
    net_income = (220.0 - 44.0 - 15.0) * 0.8
    assert result.target_multiples["ev_ebitda"] == pytest.approx(ev / 220.0, rel=REL)
    assert result.target_multiples["ev_sales"] == pytest.approx(ev / 1100.0, rel=REL)
    assert result.target_multiples["pe"] == pytest.approx(1000.0 / net_income, rel=REL)


def test_valuation_results_mirror_the_implied_values() -> None:
    result = comps.run_comps(target(), small_universe())
    rows = result.to_valuation_results()
    assert [row.method for row in rows] == [
        ValuationMethod.COMPS_EV_EBITDA,
        ValuationMethod.COMPS_EV_SALES,
        ValuationMethod.COMPS_PE,
    ]
    for row in rows:
        implied = result.implied_values.loc[row.details["multiple"]]
        assert row.low <= row.base <= row.high
        assert (row.low, row.base, row.high) == (
            implied["per_share_low"],
            implied["per_share_base"],
            implied["per_share_high"],
        )
        assert row.enterprise_value == implied["ev_base"]
        assert row.equity_value == implied["equity_base"]
        assert row.details["n_peers"] == 7
        assert sorted(row.details["peers"]) == [f"P{i}.ST" for i in range(7)]
    assert result.applicable_multiples == ["ev_ebitda", "ev_sales", "pe"]


def test_tukey_fence_on_log_multiples_removes_the_outlier() -> None:
    universe = small_universe()
    universe.append(make_peer("WILD.ST", enterprise_value=5500.0, ebitda=100.0))  # 55x
    result = comps.run_comps(target(), universe)
    multiples = [*EV_MULTIPLES, 55.0]
    logs = [math.log(m) for m in multiples]
    q1, q3 = linear_percentile(logs, 25), linear_percentile(logs, 75)
    upper = math.exp(q3 + 1.5 * (q3 - q1))
    lower = math.exp(q1 - 1.5 * (q3 - q1))
    assert lower < 6.0 and 12.0 < upper < 55.0  # only the 55x peer is outside
    assert "multiple_outliers_removed:ev_ebitda" in result.fallbacks
    assert not result.peers.loc["WILD.ST", "used_ev_ebitda"]
    assert result.peers.loc["WILD.ST", "ev_ebitda"] == pytest.approx(55.0)
    assert result.multiples_summary.loc["ev_ebitda", "n"] == 7
    assert result.multiples_summary.loc["ev_ebitda", "max"] == pytest.approx(12.0)
    assert "WILD.ST" not in result.to_valuation_results()[0].details["peers"]


def test_multiples_outside_the_hard_bounds_are_not_meaningful() -> None:
    universe = small_universe()
    universe.append(make_peer("HUGE.ST", enterprise_value=7000.0, ebitda=100.0))  # 70x > 60x
    universe.append(make_peer("TINY.ST", enterprise_value=50.0, ebitda=100.0))  # 0.5x < 1x
    universe.append(make_peer("LOSS.ST", ebitda=-50.0, net_income=-10.0))
    result = comps.run_comps(target(), universe)
    for ticker in ("HUGE.ST", "TINY.ST", "LOSS.ST"):
        assert math.isnan(result.peers.loc[ticker, "ev_ebitda"])
        assert not result.peers.loc[ticker, "used_ev_ebitda"]
    assert math.isnan(result.peers.loc["LOSS.ST", "pe"])
    assert result.multiples_summary.loc["ev_ebitda", "n"] == 7


# ------------------------------------------------------- the target is no peer


def test_target_and_its_sister_share_class_are_never_peers() -> None:
    universe = small_universe()
    universe += [
        make_peer("TGT-A.ST", name="Target AB", enterprise_value=9900.0, ebitda=100.0),
        make_peer("TGT-B.ST", name="Target AB", enterprise_value=9900.0, ebitda=100.0),
        make_peer("TGT-PREF.ST", name="Something else", enterprise_value=9900.0, ebitda=100.0),
        make_peer("OTHER.ST", name="  target ab ", enterprise_value=9900.0, ebitda=100.0),
    ]
    result = comps.run_comps(target(), universe)
    assert sorted(result.peer_tickers) == [f"P{i}.ST" for i in range(7)]
    assert not {"TGT-A.ST", "TGT-B.ST", "TGT-PREF.ST", "OTHER.ST"} & set(result.candidates.index)
    assert result.multiples_summary.loc["ev_ebitda", "max"] == pytest.approx(12.0)
    assert "target_not_in_universe" not in result.fallbacks


def test_target_outside_the_universe_is_handled() -> None:
    result = comps.run_comps(target(), small_universe())
    assert "target_not_in_universe" in result.fallbacks
    assert result.target_features.loc["log_market_cap", "raw"] == pytest.approx(3.0)
    assert result.target_features.loc["ebitda_margin", "raw"] == pytest.approx(0.2)
    assert result.target_features.loc["revenue_growth", "raw"] == pytest.approx(0.1)
    assert result.target_features.loc["leverage", "raw"] == pytest.approx(300.0 / 1300.0)


@pytest.mark.parametrize("ticker", FIXTURE_TICKERS)
def test_fixture_companies_never_meet_themselves(ticker: str, companies, universe) -> None:
    company = companies[ticker]
    result = comps.run_comps(company, universe)
    root = ticker.split(".")[0].split("-")[0]
    name = company.profile.name.strip().casefold()
    for frame in (result.peers, result.candidates):
        assert ticker not in frame.index
        assert all(t.split(".")[0].split("-")[0] != root for t in frame.index)
        assert all(str(n).strip().casefold() != name for n in frame["name"])
    assert len(result.peers) <= 12 + 3 * 5  # max_peers plus per-multiple top-ups
    assert len(set(result.peer_tickers)) == len(result.peer_tickers)


# ------------------------------------------------------------------ clustering


def two_cluster_universe() -> list:
    """Ten large and ten small companies; nothing else differs systematically."""
    rng = np.random.default_rng(11)
    peers = []
    for i in range(20):
        large = i < 10
        peers.append(
            make_peer(
                f"{'L' if large else 'S'}{i:02d}.ST",
                market_cap=(1e11 if large else 1e8) * float(rng.uniform(0.9, 1.1)),
                enterprise_value=(10.0 if large else 6.0) * 100.0 + i,
                ebitda=100.0,
                revenue=float(500.0 if large else 1000.0),
                revenue_growth=float((0.10 if large else -0.05) + rng.normal(0, 0.002)),
                return_on_equity=float((0.25 if large else 0.02) + rng.normal(0, 0.002)),
                beta=float((1.3 if large else 0.6) + rng.normal(0, 0.01)),
                total_debt=200.0,
            )
        )
    return peers


def silhouette_by_hand(points: np.ndarray, labels: np.ndarray) -> float:
    distances = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2))
    scores = []
    for i in range(len(points)):
        own = labels == labels[i]
        if own.sum() == 1:
            scores.append(0.0)
            continue
        a = distances[i, own].sum() / (own.sum() - 1)
        b = min(distances[i, labels == other].mean() for other in set(labels) - {labels[i]})
        scores.append((b - a) / max(a, b))
    return float(np.mean(scores))


def clustered_points(result: comps.CompsResult) -> tuple[np.ndarray, np.ndarray]:
    """z-scores and cluster labels of all clustered points (candidates + target)."""
    features = list(result.feature_names)
    z_candidates = result.candidates[[f"z_{f}" for f in features]].to_numpy(float)
    z_target = result.target_features.loc[features, "z"].to_numpy(float)
    points = np.vstack([z_candidates, z_target])
    labels = np.append(result.candidates["cluster"].to_numpy(int), result.target_cluster)
    return points, labels


def test_peers_are_the_members_of_the_targets_cluster() -> None:
    company = target(shares=1e10, debt=200.0, cash=0.0)  # market cap 1e11: a "large" one
    result = comps.run_comps(company, two_cluster_universe())
    assert result.chosen_k == 2
    assert sorted(result.peer_tickers) == [f"L{i:02d}.ST" for i in range(10)]
    assert set(result.peers["selection"]) == {"cluster"}
    assert (result.peers["cluster"] == result.target_cluster).all()
    assert result.candidates["is_peer"].sum() == 10
    assert result.multiples_summary.loc["ev_ebitda", "min"] >= 10.0


def test_silhouette_is_reported_for_every_k_and_the_best_k_is_chosen() -> None:
    result = comps.run_comps(target(shares=1e10), two_cluster_universe())
    # 21 points, at least 4 per cluster -> k = 2..5
    assert sorted(result.silhouette_scores) == [2, 3, 4, 5]
    best = max(result.silhouette_scores.values())
    assert result.silhouette_scores[result.chosen_k] == best
    assert all(-1.0 <= s <= 1.0 for s in result.silhouette_scores.values())
    points, labels = clustered_points(result)
    assert len(set(labels)) == result.chosen_k
    assert silhouette_by_hand(points, labels) == pytest.approx(best, rel=1e-9)


def test_silhouette_on_the_real_universe(atco, universe) -> None:
    result = comps.run_comps(atco, universe)
    assert sorted(result.silhouette_scores) == list(range(2, 9))
    best_k = max(result.silhouette_scores, key=lambda k: (result.silhouette_scores[k], -k))
    assert result.chosen_k == best_k
    points, labels = clustered_points(result)
    assert silhouette_by_hand(points, labels) == pytest.approx(
        result.silhouette_scores[result.chosen_k], rel=1e-9
    )
    # reported distances are Euclidean distances to the target in z-space
    by_hand = np.sqrt(((points[:-1] - points[-1]) ** 2).sum(axis=1))
    assert result.candidates["distance"].to_numpy() == pytest.approx(by_hand, rel=1e-9)
    # every z-score column is standardized over candidates + target
    assert points.mean(axis=0) == pytest.approx(np.zeros(points.shape[1]), abs=1e-9)
    assert points.std(axis=0) == pytest.approx(np.ones(points.shape[1]), rel=1e-9)


def test_large_cluster_is_cut_to_the_nearest_peers(atco, universe) -> None:
    result = comps.run_comps(atco, universe)
    assert "cluster_capped_to_nearest" in result.fallbacks
    in_cluster = result.candidates[result.candidates["cluster"] == result.target_cluster]
    assert len(in_cluster) > 12
    nearest = in_cluster.sort_values("ranking_distance").index[:12]
    core = result.peers[result.peers["selection"] == "cluster"]
    assert list(core.index) == list(nearest)
    assert result.peers["ranking_distance"].is_monotonic_increasing
    # the industry penalty: 0 for the same industry, +0.5 inside the sector
    penalty = result.peers["ranking_distance"] - result.peers["distance"]
    assert penalty[result.peers["same_industry"]].abs().max() < 1e-12
    assert penalty[~result.peers["same_industry"]].tolist() == pytest.approx(
        [0.5] * int((~result.peers["same_industry"]).sum())
    )


def test_run_is_deterministic(atco, universe) -> None:
    first = comps.run_comps(atco, universe)
    second = comps.run_comps(atco, list(universe))
    assert first.peer_tickers == second.peer_tickers
    assert first.silhouette_scores == second.silhouette_scores
    assert first.chosen_k == second.chosen_k
    pd.testing.assert_frame_equal(first.peers, second.peers)
    pd.testing.assert_frame_equal(first.multiples_summary, second.multiples_summary)
    pd.testing.assert_frame_equal(first.implied_values, second.implied_values)


def test_too_few_points_means_no_clustering() -> None:
    result = comps.run_comps(target(), small_universe())
    assert result.chosen_k == 1
    assert result.silhouette_scores == {}
    assert "no_clustering_too_few_candidates" in result.fallbacks
    assert set(result.peers["selection"]) == {"all_candidates"}


# -------------------------------------------------------------------- fallbacks


def test_tiny_sector_is_widened_to_the_non_financial_universe() -> None:
    company = target(sector="Utilities", industry="Utilities - Renewable")
    universe = [
        make_peer("U1.ST", sector="Utilities", industry="Utilities - Renewable"),
        make_peer("U2.ST", sector="Utilities", industry="Utilities - Regulated Electric"),
        *small_universe(),
        make_peer("BANK.ST", sector="Financial Services", industry="Banks - Regional"),
    ]
    result = comps.run_comps(company, universe)
    assert result.candidate_scope == "universe"
    assert "sector_too_small_widened_to_universe" in result.fallbacks
    assert "BANK.ST" not in result.candidates.index
    assert len(result.candidates) == 9
    # same industry first, then the same industry family, then the rest
    assert result.peer_tickers[0] == "U1.ST"
    penalty = (result.candidates["ranking_distance"] - result.candidates["distance"]).round(12)
    assert penalty["U1.ST"] == 0.0
    assert penalty["U2.ST"] == 0.25
    assert penalty["P0.ST"] == 1.0


def test_enough_sector_peers_keeps_the_scope_at_the_sector() -> None:
    universe = small_universe() + [make_peer("TECH.ST", sector="Technology", industry="Software")]
    result = comps.run_comps(target(), universe)
    assert result.candidate_scope == "sector"
    assert "TECH.ST" not in result.candidates.index


def test_missing_or_negative_target_metrics_skip_only_that_multiple() -> None:
    company = target(margin=-0.10, net_income=-5.0)
    result = comps.run_comps(company, small_universe())
    assert result.applicable_multiples == ["ev_sales"]
    assert set(result.skipped) == {"ev_ebitda", "pe"}
    assert "missing or non-positive" in result.skipped["ev_ebitda"]
    assert [r.method for r in result.to_valuation_results()] == [ValuationMethod.COMPS_EV_SALES]
    assert result.target_multiples["ev_ebitda"] is None
    assert result.target_multiples["pe"] is None


def test_negative_equity_is_floored_at_zero_and_flagged() -> None:
    # EV/EBITDA quartiles 7.5 / 9 / 10.5 on EBITDA 220 -> EV 1650 / 1980 / 2310
    company = target(debt=1900.0, cash=0.0)
    result = comps.run_comps(company, small_universe())
    row = result.implied_values.loc["ev_ebitda"]
    assert row["equity_low_unfloored"] == pytest.approx(1650.0 - 1900.0)
    assert row["equity_low"] == 0.0 and row["per_share_low"] == 0.0
    assert row["equity_base"] == pytest.approx(80.0)
    assert row["per_share_high"] == pytest.approx(4.1)
    assert bool(row["floored"]) is True
    assert "negative_equity_floored_at_zero:ev_ebitda" in result.fallbacks
    bar = next(r for r in result.to_valuation_results() if r.details["multiple"] == "ev_ebitda")
    assert (bar.low, bar.details["floored_at_zero"]) == (0.0, True)
    # EV/Sales: 1.45 x 1100 = 1595 < 1900 -> no positive value anywhere in the range
    assert "ev_sales" in result.skipped
    assert "non-positive across the whole range" in result.skipped["ev_sales"]
    # P/E does not go through the bridge at all
    assert not result.implied_values.loc["pe", "floored"]


def test_too_few_observations_skip_the_multiple() -> None:
    universe = small_universe()[:2]
    result = comps.run_comps(target(), universe)
    assert result.implied_values.empty
    assert result.to_valuation_results() == []
    assert set(result.skipped) == {"ev_ebitda", "ev_sales", "pe"}
    assert all("3 required" in reason for reason in result.skipped.values())
    assert "fewer_than_min_peers_available" in result.fallbacks
    assert comps.run_comps(target(), small_universe()[:3]).applicable_multiples == [
        "ev_ebitda",
        "ev_sales",
        "pe",
    ]


def test_single_peer_universe_gives_a_result_without_values() -> None:
    result = comps.run_comps(target(), small_universe()[:1])
    assert result.peer_tickers == ["P0.ST"]
    assert result.implied_values.empty
    assert result.chosen_k == 1


def test_no_candidates_at_all_is_not_applicable() -> None:
    with pytest.raises(NotApplicableError, match="empty"):
        comps.run_comps(target(), [])
    with pytest.raises(NotApplicableError, match="no usable peer candidates"):
        comps.run_comps(target(), [make_peer("TGT-A.ST"), make_peer("TGT-B.ST")])
    with pytest.raises(NotApplicableError, match="no usable peer candidates"):
        comps.run_comps(target(), [make_peer("X.ST", ebitda=None, revenue=None, net_income=None)])


def test_invalid_market_data_is_not_applicable() -> None:
    with pytest.raises(NotApplicableError, match="shares outstanding"):
        comps.run_comps(target(shares=0.0), small_universe())
    company = target()
    company.market.market_cap = math.nan
    with pytest.raises(NotApplicableError, match="market cap"):
        comps.run_comps(company, small_universe())


def test_multiple_is_topped_up_with_the_nearest_peers_that_have_it() -> None:
    universe = ruler_universe(16)
    # only two of the twelve nearest have a P/E; the four farthest all have one
    universe = [
        make_peer(
            peer.ticker,
            market_cap=peer.market_cap,
            enterprise_value=peer.enterprise_value,
            ebitda=100.0,
            net_income=peer.market_cap / 20.0 if i in (0, 1) or i >= 12 else -5.0,
        )
        for i, peer in enumerate(universe)
    ]
    config = comps.CompsConfig(min_cluster_points=1000)  # no clustering: a pure ranking
    result = comps.run_comps(target(), universe, config=config)
    assert result.chosen_k == 1
    assert "cluster_capped_to_nearest" in result.fallbacks
    core = [t for t, s in result.peers["selection"].items() if s == "all_candidates"]
    assert core == [f"P{i:02d}.ST" for i in range(12)]
    filled = [t for t, s in result.peers["selection"].items() if s == "fill:pe"]
    assert filled == ["P12.ST", "P13.ST", "P14.ST"]  # up to min_peers = 5 observations
    assert "multiple_topped_up_with_nearest:pe" in result.fallbacks
    assert result.multiples_summary.loc["pe", "n"] == 5
    # the fill peers serve the P/E only
    assert not result.peers.loc[filled, "used_ev_ebitda"].any()
    assert result.multiples_summary.loc["ev_ebitda", "n"] == 12


def test_small_cluster_is_topped_up_with_nearest_neighbours() -> None:
    universe = two_cluster_universe()
    large = [p for p in universe if p.ticker.startswith("L")][:3]
    small = [p for p in universe if p.ticker.startswith("S")]
    extra = [
        make_peer(
            f"S{i:02d}.ST",
            market_cap=1e8 * (1 + 0.01 * i),
            enterprise_value=600.0 + i,
            ebitda=100.0,
            revenue=1000.0,
            revenue_growth=-0.05 + 0.0001 * i,
            return_on_equity=0.02 + 0.0001 * i,
            beta=0.6 + 0.001 * i,
        )
        for i in range(20, 26)
    ]
    result = comps.run_comps(target(shares=1e10, debt=200.0, cash=0.0), large + small + extra)
    selections = result.peers["selection"].value_counts().to_dict()
    assert selections.get("cluster") == 3
    assert selections.get("nearest_neighbour") == 2
    assert "cluster_too_small_added_nearest_neighbours" in result.fallbacks
    assert len(result.peers) == 5


# --------------------------------------------------------------------- financials


def bank_universe() -> list:
    banks = [
        make_peer(
            f"B{i}.ST",
            sector="Financial Services",
            industry="Banks - Regional" if i % 2 else "Banks - Diversified",
            ebitda=None,
            revenue=300.0,
            net_income=100.0,
            market_cap=800.0 + 100.0 * i,  # P/E 8..15
            beta=0.9 + 0.02 * i,
            return_on_equity=0.10 + 0.01 * i,
        )
        for i in range(8)
    ]
    others = [
        make_peer(
            f"INV{i}.ST",
            sector="Financial Services",
            industry="Asset Management",
            ebitda=None,
            net_income=10.0,
            market_cap=900.0,  # P/E 90
        )
        for i in range(6)
    ]
    return banks + others + small_universe()


def test_financial_company_gets_pe_only_from_its_industry_family() -> None:
    bank = make_company(
        "BK.ST", sector="Financial Services", industry="Banks - Regional", name="Bank AB"
    )
    result = comps.run_comps(bank, bank_universe())
    assert result.is_financial is True
    assert result.candidate_scope == "industry_family"
    assert sorted(result.peer_tickers) == [f"B{i}.ST" for i in range(8)]
    assert result.applicable_multiples == ["pe"]
    assert set(result.skipped) == {"ev_ebitda", "ev_sales"}
    assert all("financial" in reason for reason in result.skipped.values())
    assert set(result.feature_names) <= set(comps.FEATURES_FINANCIAL)
    rows = result.to_valuation_results()
    assert [row.method for row in rows] == [ValuationMethod.COMPS_PE]
    pes = [8.0 + i for i in range(8)]
    net_income = (220.0 - 44.0 - 15.0) * 0.8
    assert rows[0].base == pytest.approx(linear_percentile(pes, 50) * net_income / 100.0, rel=REL)
    assert rows[0].low == pytest.approx(linear_percentile(pes, 25) * net_income / 100.0, rel=REL)
    assert rows[0].high == pytest.approx(linear_percentile(pes, 75) * net_income / 100.0, rel=REL)
    assert result.multiples_summary.loc[["ev_ebitda", "ev_sales"], "n"].tolist() == [0.0, 0.0]


def test_financial_company_without_a_family_uses_the_sector() -> None:
    insurer = make_company(
        "INS.ST", sector="Financial Services", industry="Insurance - Life", name="Liv AB"
    )
    result = comps.run_comps(insurer, bank_universe())
    assert result.candidate_scope == "sector"
    assert all(t.startswith(("B", "INV")) for t in result.candidates.index)


def test_seb_is_valued_on_bank_pe_multiples_only(companies, universe) -> None:
    result = comps.run_comps(companies["SEB-A.ST"], universe)
    assert result.applicable_multiples == ["pe"]
    assert result.candidate_scope == "industry_family"
    assert all(str(i).startswith("Banks") for i in result.peers["industry"])
    assert set(result.skipped) == {"ev_ebitda", "ev_sales"}


# ------------------------------------------------------------------ configuration


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_peers": 0},
        {"min_peers": 6, "max_peers": 5},
        {"k_min": 1},
        {"k_min": 5, "k_max": 4},
        {"winsor_lower": 0.6, "winsor_upper": 0.4},
        {"min_observations": 0},
    ],
)
def test_config_validation(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        comps.CompsConfig(**kwargs)


def test_config_changes_the_peer_group_size() -> None:
    universe = ruler_universe(16)
    config = comps.CompsConfig(min_cluster_points=1000, max_peers=6)
    result = comps.run_comps(target(), universe, config=config)
    assert result.peer_tickers == [f"P{i:02d}.ST" for i in range(6)]
    multiples = [8.0 + 0.25 * i for i in range(6)]
    assert result.multiples_summary.loc["ev_ebitda", "median"] == pytest.approx(
        linear_percentile(multiples, 50), rel=REL
    )


# ----------------------------------------------------------- feature cleaning


def test_feature_cleaning_by_hand() -> None:
    """Clip, winsorize at the 5th / 95th percentile, impute the median, z-score."""
    caps = [1e8, 3e8, 1e9, 3e9, 1e10, 3e10, 1e11]
    betas = [0.5, 0.7, None, 1.1, 1.3, 1.5, 8.0]  # one missing, one beyond the hard bound of 3
    universe = [
        make_peer(f"P{i}.ST", market_cap=cap, beta=beta)
        for i, (cap, beta) in enumerate(zip(caps, betas, strict=True))
    ]
    universe.append(make_peer("TGT-A.ST", name="Target AB", market_cap=5e9, beta=1.0))
    result = comps.run_comps(target(), universe)
    order = [f"P{i}.ST" for i in range(7)]

    def clean(values: list[float | None], bounds: tuple[float, float] | None) -> list[float]:
        bounded = [
            v if v is None or bounds is None else min(max(v, bounds[0]), bounds[1]) for v in values
        ]
        known = [v for v in bounded if v is not None]
        low, high = linear_percentile(known, 5), linear_percentile(known, 95)
        clipped = [None if v is None else min(max(v, low), high) for v in bounded]
        median = statistics.median(v for v in clipped if v is not None)
        filled = [median if v is None else v for v in clipped]
        mean = sum(filled) / len(filled)
        std = math.sqrt(sum((v - mean) ** 2 for v in filled) / len(filled))
        return [(v - mean) / std for v in filled]

    # the target is standardized together with the candidates (last element)
    z_size = clean([math.log10(c) for c in [*caps, 5e9]], None)
    z_beta = clean([*betas, 1.0], (-0.5, 3.0))
    assert result.candidates.loc[order, "z_log_market_cap"].tolist() == pytest.approx(z_size[:7])
    assert result.target_features.loc["log_market_cap", "z"] == pytest.approx(z_size[7])
    assert result.candidates.loc[order, "z_beta"].tolist() == pytest.approx(z_beta[:7])
    assert result.target_features.loc["beta", "z"] == pytest.approx(z_beta[7])
    # the raw feature values are reported untouched
    assert result.candidates.loc["P6.ST", "beta"] == 8.0
    assert math.isnan(result.candidates.loc["P2.ST", "beta"])
    assert result.target_features.loc["beta", "raw"] == 1.0


def test_leverage_feature_is_debt_over_debt_plus_market_cap() -> None:
    universe = small_universe()
    result = comps.run_comps(target(), universe)
    for i in range(7):
        debt, market_cap = 100.0 + 20.0 * i, 1000.0 + 10.0 * i
        assert result.candidates.loc[f"P{i}.ST", "leverage"] == pytest.approx(
            debt / (debt + market_cap)
        )
    assert result.candidates.loc["P3.ST", "log_market_cap"] == pytest.approx(math.log10(1030.0))


def test_constant_and_empty_features_are_dropped() -> None:
    universe = [
        make_peer(
            f"P{i}.ST",
            market_cap=1000.0,
            enterprise_value=(8.0 + i) * 100.0,
            revenue_growth=None,
            return_on_equity=0.1,
            beta=None,
            total_debt=None,
        )
        for i in range(6)
    ]
    universe.append(
        make_peer(
            "TGT-A.ST",
            name="Target AB",
            market_cap=1000.0,
            revenue_growth=None,
            return_on_equity=0.1,
            beta=None,
            total_debt=None,
        )
    )
    result = comps.run_comps(target(), universe)
    assert result.feature_names == ()
    assert "no_usable_features" in result.fallbacks
    assert "feature_dropped_no_data:beta" in result.fallbacks
    assert "feature_dropped_constant:log_market_cap" in result.fallbacks
    assert (result.peers["distance"] == 0.0).all()
    assert result.peer_tickers == [f"P{i}.ST" for i in range(6)]  # ties broken by ticker
    assert result.multiples_summary.loc["ev_ebitda", "median"] == pytest.approx(10.5)
    assert result.target_features["z"].isna().all()


def test_feature_weights_stretch_the_distance() -> None:
    universe = ruler_universe(10)
    plain = comps.run_comps(target(), universe)
    weighted = comps.run_comps(
        target(), universe, config=comps.CompsConfig(feature_weights={"log_market_cap": 3.0})
    )
    assert weighted.candidates["z_log_market_cap"].tolist() == pytest.approx(
        (3.0 * plain.candidates["z_log_market_cap"]).tolist()
    )


def test_target_outside_the_universe_uses_its_own_price_history_for_beta() -> None:
    from support import make_history

    company = target(history=make_history(1.4, ticker="TGT-A.ST"))
    result = comps.run_comps(company, small_universe())
    assert result.target_features.loc["beta", "raw"] == pytest.approx(1.4, rel=1e-9)
    net_income = (220.0 - 44.0 - 15.0) * 0.8
    equity = (0.5 * 1000.0 + 0.5 * 1100.0) / 2
    assert result.target_features.loc["return_on_equity", "raw"] == pytest.approx(
        net_income / equity
    )
    no_history = comps.run_comps(target(), small_universe())
    assert math.isnan(no_history.target_features.loc["beta", "raw"])
    assert "target_feature_imputed:beta" in no_history.fallbacks
