"""Comparable company analysis (trading comps) with k-means peer selection.

Public API::

    run_comps(company, universe, *, config=None) -> CompsResult
    CompsResult.to_valuation_results() -> list[ValuationResult]

Method
------
1. **Candidates.** Peers are drawn from the target's Yahoo *sector* only (for
   financial and real estate targets from the *industry family*, e.g. all
   "Banks - ..." industries, when it holds at least ``min_peers`` candidates:
   P/E is the only multiple there and an investment company's P/E says nothing
   about a bank). Pure
   clustering across the whole market would put a bank next to a biotech of the
   same size, which is useless for valuation: multiples are driven first by the
   business model and only then by size, growth, profitability and risk. The
   target itself (matched on ticker, ticker root and company name, so that a
   second share class can never slip in) is excluded. A candidate must have a
   positive market cap and at least one usable multiple, otherwise it cannot
   contribute anything. If the sector holds fewer than ``min_peers`` candidates
   (Energy and Utilities are tiny in Sweden) the set is widened to every
   company on the same side of the financial / non-financial divide.
2. **Features.** Log market cap (size), revenue growth, EBITDA margin, return on
   equity, beta and leverage (debt / (debt + market cap)). These are the classic
   drivers of a trading multiple: scale and liquidity, growth, profitability,
   return on capital, systematic risk and financial risk. For financial and real
   estate companies only size, ROE and beta are used (EBITDA is undefined for a
   bank, and balance sheet leverage is the business itself, not a risk add-on).
3. **Cleaning.** Every feature is first clipped to hard sanity bounds (the data
   contains betas of 94 and margins of -3 000 %), then winsorized at the 5th and
   95th percentile of the candidate set, missing values are imputed with the
   candidate median, and the result is standardized to z-scores. A feature with
   no variation in the candidate set is dropped.
4. **Clustering.** K-means (fixed ``random_state``) for every k in a range; the k
   with the highest silhouette score wins. All scores are returned. The peers
   are the members of the target's cluster.
5. **Sizing the peer group.** Distances are Euclidean in z-space plus a penalty
   for a different industry (half for the same industry family, double for a
   different sector), so that same-industry companies are preferred. A cluster
   above ``max_peers`` is cut to the nearest ones; a cluster below ``min_peers``
   is topped up with the nearest neighbours from outside the cluster. A multiple
   that the target can use (positive metric) but that has fewer than
   ``min_peers`` usable observations in the peer group is topped up, for that
   multiple only, with the nearest candidates that do have the multiple.
6. **Multiples.** EV/EBITDA, EV/Sales and P/E. Non-positive values and values
   outside hard bounds are not meaningful and dropped; within the peer group a
   Tukey fence (1.5 x IQR on the log multiple) removes remaining outliers.
7. **Implied value.** 25th percentile / median / 75th percentile of the peer
   multiple times the target's own metric, bridged from enterprise to equity
   value (net debt, minority interest, preferred equity) and divided by
   ``MarketData.shares_outstanding``. Negative equity is floored at zero and
   flagged.

Every deviation from the plain path is recorded in ``CompsResult.fallbacks`` and
every multiple that could not be applied in ``CompsResult.skipped``.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from valuation_engine.core.types import (
    CompanyData,
    NotApplicableError,
    PeerSnapshot,
    ValuationMethod,
    ValuationResult,
)

__all__ = [
    "FEATURES_FINANCIAL",
    "FEATURES_NON_FINANCIAL",
    "MULTIPLES",
    "CompsConfig",
    "CompsResult",
    "run_comps",
]

#: Multiples in presentation order.
MULTIPLES: tuple[str, ...] = ("ev_ebitda", "ev_sales", "pe")

FEATURES_NON_FINANCIAL: tuple[str, ...] = (
    "log_market_cap",
    "revenue_growth",
    "ebitda_margin",
    "return_on_equity",
    "beta",
    "leverage",
)
FEATURES_FINANCIAL: tuple[str, ...] = ("log_market_cap", "return_on_equity", "beta")

_METHODS: dict[str, ValuationMethod] = {
    "ev_ebitda": ValuationMethod.COMPS_EV_EBITDA,
    "ev_sales": ValuationMethod.COMPS_EV_SALES,
    "pe": ValuationMethod.COMPS_PE,
}
_LABELS: dict[str, str] = {
    "ev_ebitda": "Comps - EV/EBITDA",
    "ev_sales": "Comps - EV/Sales",
    "pe": "Comps - P/E",
}
#: Target metric (income statement item) each multiple is applied to.
_METRICS: dict[str, str] = {"ev_ebitda": "ebitda", "ev_sales": "revenue", "pe": "net_income"}

_SUMMARY_COLUMNS: tuple[str, ...] = ("n", "min", "p25", "median", "mean", "p75", "max")


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CompsConfig:
    """Tuning knobs of the comps model. The defaults are used in batch runs."""

    min_peers: int = 5  # smaller peer groups are topped up with nearest neighbours
    max_peers: int = 12  # larger clusters are cut to the nearest ones
    min_observations: int = 3  # usable peer multiples needed to apply a multiple
    k_min: int = 2
    k_max: int = 8
    min_cluster_points: int = 12  # fewer points (candidates + target): no clustering
    min_points_per_cluster: int = 4  # caps k at n_points // this
    random_state: int = 42
    n_init: int = 10
    winsor_lower: float = 0.05
    winsor_upper: float = 0.95
    #: Added to the z-space distance for a different industry in the same
    #: sector. Half of it for the same industry family ("Banks - Regional" vs
    #: "Banks - Diversified"), twice for a different sector. 0 disables it.
    industry_penalty: float = 0.5
    #: Optional weight per feature (multiplies the z-score); default 1.0.
    feature_weights: Mapping[str, float] = field(default_factory=dict)
    #: Hard sanity bounds per feature, applied before winsorization.
    feature_bounds: Mapping[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "revenue_growth": (-1.0, 3.0),
            "ebitda_margin": (-1.0, 1.0),
            "return_on_equity": (-1.0, 1.0),
            "beta": (-0.5, 3.0),
            "leverage": (0.0, 1.0),
        }
    )
    #: Range within which a peer multiple is considered meaningful.
    multiple_bounds: Mapping[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "ev_ebitda": (1.0, 60.0),
            "ev_sales": (0.05, 30.0),
            "pe": (1.0, 100.0),
        }
    )
    iqr_factor: float = 1.5  # Tukey fence on log multiples within the peer group
    iqr_min_observations: int = 5  # no fence below this many observations

    def __post_init__(self) -> None:
        if self.min_peers < 1 or self.max_peers < self.min_peers:
            raise ValueError("require 1 <= min_peers <= max_peers")
        if self.k_min < 2 or self.k_max < self.k_min:
            raise ValueError("require 2 <= k_min <= k_max")
        if not 0.0 <= self.winsor_lower < self.winsor_upper <= 1.0:
            raise ValueError("require 0 <= winsor_lower < winsor_upper <= 1")
        if self.min_observations < 1:
            raise ValueError("min_observations must be at least 1")


# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------


@dataclass
class CompsResult:
    """Outcome of :func:`run_comps`.

    Attributes
    ----------
    peers:
        One row per selected peer, indexed by ticker and sorted by distance.
        Columns: ``name``, ``sector``, ``industry``, ``same_industry``,
        ``selection`` (``cluster``, ``nearest_neighbour``, ``all_candidates`` or
        ``fill:<multiple>``), ``cluster``, ``distance`` (Euclidean in z-space),
        ``ranking_distance`` (distance + industry penalty), ``market_cap``,
        ``enterprise_value``, the raw features, the cleaned multiples
        (``ev_ebitda``, ``ev_sales``, ``pe``; NaN when not meaningful) and
        ``used_<multiple>`` flags telling which observations entered the summary.
    candidates:
        Every candidate that was clustered (same columns minus the ``used_*``
        flags and ``selection``, plus ``z_<feature>`` columns and ``is_peer``).
    target_features:
        Raw (pre-cleaning) and standardized feature values of the target.
    multiples_summary:
        Index ``ev_ebitda``/``ev_sales``/``pe``; columns ``n``, ``min``, ``p25``,
        ``median``, ``mean``, ``p75``, ``max``. Rows without observations are NaN.
    implied_values:
        One row per applicable multiple with the target metric, the low / base /
        high multiple, enterprise value, equity value and value per share, the
        base value relative to the current price and the ``floored`` flag.
    candidate_scope:
        ``sector`` (default), ``industry_family`` (financial targets with enough
        companies in the same family, e.g. banks) or ``universe`` (sector too
        small).
    silhouette_scores:
        Silhouette score for every k tried (empty when no clustering was done).
    chosen_k:
        Number of clusters used (1 when no clustering was done).
    fallbacks:
        Codes of every fallback that fired, in order.
    skipped:
        Multiple -> reason, for each multiple that was not applied.
    """

    ticker: str
    name: str
    currency: str
    price: float
    shares_outstanding: float
    is_financial: bool
    candidate_scope: str
    peers: pd.DataFrame
    candidates: pd.DataFrame
    target_features: pd.DataFrame
    feature_names: tuple[str, ...]
    chosen_k: int
    target_cluster: int
    silhouette_scores: dict[int, float]
    multiples_summary: pd.DataFrame
    target_multiples: dict[str, float | None]
    implied_values: pd.DataFrame
    fallbacks: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    equity_bridge: dict[str, float] = field(default_factory=dict)

    @property
    def peer_tickers(self) -> list[str]:
        """Tickers of the selected peers, nearest first."""
        return [str(t) for t in self.peers.index]

    @property
    def applicable_multiples(self) -> list[str]:
        """Multiples for which an implied value exists."""
        return [str(m) for m in self.implied_values.index]

    def to_valuation_results(self) -> list[ValuationResult]:
        """One :class:`ValuationResult` per applicable multiple (per share:
        low = 25th percentile, base = median, high = 75th percentile)."""
        results: list[ValuationResult] = []
        for multiple, row in self.implied_values.iterrows():
            key = str(multiple)
            used = f"used_{key}"
            used_peers = (
                [str(t) for t in self.peers.index[self.peers[used].astype(bool)]]
                if used in self.peers.columns
                else []
            )
            details: dict[str, Any] = {
                "multiple": key,
                "metric": _METRICS[key],
                "metric_value": float(row["metric_value"]),
                "multiple_low": float(row["multiple_low"]),
                "multiple_base": float(row["multiple_base"]),
                "multiple_high": float(row["multiple_high"]),
                "n_peers": int(row["n_peers"]),
                "peers": used_peers,
                "target_multiple": self.target_multiples.get(key),
                "chosen_k": self.chosen_k,
                "floored_at_zero": bool(row["floored"]),
                "equity_low_unfloored": float(row["equity_low_unfloored"]),
                "equity_bridge": dict(self.equity_bridge),
                "fallbacks": list(self.fallbacks),
            }
            results.append(
                ValuationResult(
                    method=_METHODS[key],
                    label=_LABELS[key],
                    low=float(row["per_share_low"]),
                    base=float(row["per_share_base"]),
                    high=float(row["per_share_high"]),
                    currency=self.currency,
                    enterprise_value=float(row["ev_base"]),
                    equity_value=float(row["equity_base"]),
                    details=details,
                )
            )
        return results


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _num(value: Any) -> float:
    """Return ``value`` as a finite float, or NaN."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _num0(value: Any) -> float:
    """Return ``value`` as a finite float, or 0.0 when missing."""
    number = _num(value)
    return 0.0 if math.isnan(number) else number


def _ticker_root(ticker: str) -> str:
    """``ATCO-A.ST`` -> ``ATCO`` (same rule as the data layer's share-class grouping)."""
    return ticker.upper().split(".")[0].split("-")[0]


def _industry_family(industry: Any) -> str:
    """``Banks - Regional`` -> ``Banks``; an industry without `` - `` is its own
    family. Empty string when the industry is unknown."""
    if not isinstance(industry, str) or not industry.strip():
        return ""
    return industry.split(" - ")[0].strip()


def _latest(frame: pd.DataFrame, item: str) -> float:
    """Latest fiscal-year value of ``item`` (NaN when missing)."""
    if frame is None or item not in frame.columns or frame.empty:
        return math.nan
    return _num(frame[item].iloc[-1])


def _universe_frame(universe: Sequence[PeerSnapshot]) -> pd.DataFrame:
    """Peer snapshots as a DataFrame indexed by ticker, with raw features and
    raw (not yet bounded) multiples. Duplicate tickers keep the first row."""
    rows: list[dict[str, Any]] = []
    for snap in universe:
        market_cap = _num(snap.market_cap)
        enterprise_value = _num(snap.enterprise_value)
        revenue, ebitda, net_income = _num(snap.revenue), _num(snap.ebitda), _num(snap.net_income)
        debt = _num(snap.total_debt)
        rows.append(
            {
                "ticker": snap.ticker,
                "name": snap.name,
                "sector": snap.sector,
                "industry": snap.industry,
                "is_financial": snap.is_financial,
                "market_cap": market_cap,
                "enterprise_value": enterprise_value,
                "log_market_cap": math.log10(market_cap) if market_cap > 0 else math.nan,
                "revenue_growth": _num(snap.revenue_growth),
                "ebitda_margin": _num(snap.ebitda_margin),
                "return_on_equity": _num(snap.return_on_equity),
                "beta": _num(snap.beta),
                "leverage": _leverage(debt, market_cap),
                "ev_ebitda": _raw_multiple(enterprise_value, ebitda),
                "ev_sales": _raw_multiple(enterprise_value, revenue),
                "pe": _raw_multiple(market_cap, net_income),
            }
        )
    columns = [
        "ticker",
        "name",
        "sector",
        "industry",
        "is_financial",
        "market_cap",
        "enterprise_value",
        *FEATURES_NON_FINANCIAL,
        *MULTIPLES,
    ]
    frame = pd.DataFrame(rows, columns=columns)
    frame = frame.drop_duplicates(subset="ticker", keep="first").set_index("ticker")
    return frame


def _leverage(debt: float, market_cap: float) -> float:
    """Debt / (debt + market cap), in [0, 1]; NaN when not computable."""
    if math.isnan(debt) or math.isnan(market_cap) or market_cap <= 0:
        return math.nan
    debt = max(debt, 0.0)
    return debt / (debt + market_cap)


def _raw_multiple(numerator: float, denominator: float) -> float:
    """numerator / denominator when both are positive, else NaN."""
    if math.isnan(numerator) or math.isnan(denominator) or numerator <= 0 or denominator <= 0:
        return math.nan
    return numerator / denominator


def _bounded(values: pd.Series, bounds: tuple[float, float]) -> pd.Series:
    """Set values outside ``bounds`` (inclusive) to NaN."""
    low, high = bounds
    return values.where((values >= low) & (values <= high))


#: Minimum weekly returns for the fallback beta (one year), the same rule as
#: the data layer (``data.loader.MIN_BETA_RETURNS``) and the DCF.
_MIN_BETA_RETURNS = 52


def _regression_beta(company: CompanyData) -> float:
    """OLS beta of the stock against the benchmark; provider beta as fallback."""
    history = company.price_history
    if history is not None:
        try:
            returns = history.prices[["stock", "benchmark"]].pct_change().dropna()
            variance = float(returns["benchmark"].var())
            if len(returns) >= _MIN_BETA_RETURNS and variance > 0:
                beta = float(returns["stock"].cov(returns["benchmark"])) / variance
                if math.isfinite(beta):
                    return beta
        except (KeyError, TypeError, ValueError):
            pass
    return _num(company.market.beta)


def _target_row_from_company(company: CompanyData) -> dict[str, Any]:
    """Feature row for a target that is not in the universe table.

    Amounts are in the trading currency rather than SEK. This only matters for
    ``log_market_cap`` of the (one) company that trades in another currency.
    """
    income, balance = company.statements.income, company.statements.balance
    market = company.market
    market_cap = _num(market.market_cap)
    revenue, ebitda, net_income = (
        _latest(income, "revenue"),
        _latest(income, "ebitda"),
        _latest(income, "net_income"),
    )
    growth = math.nan
    try:
        if len(income) >= 2:
            previous = _num(income["revenue"].iloc[-2])
            consecutive = int(income.index[-1]) - int(income.index[-2]) == 1
            if consecutive and previous > 0 and not math.isnan(revenue):
                growth = revenue / previous - 1.0
    except (KeyError, TypeError, ValueError):
        growth = math.nan
    margin = ebitda / revenue if revenue > 0 and not math.isnan(ebitda) else math.nan
    roe = math.nan
    try:
        equity = balance["total_equity"].dropna()
        if not equity.empty and not math.isnan(net_income):
            base = float(equity.iloc[-2:].mean())
            if base > 0:
                roe = net_income / base
    except (KeyError, TypeError, ValueError):
        roe = math.nan
    enterprise_value = _num(market.enterprise_value)
    return {
        "name": company.profile.name,
        "sector": company.profile.sector,
        "industry": company.profile.industry,
        "is_financial": company.profile.is_financial,
        "market_cap": market_cap,
        "enterprise_value": enterprise_value,
        "log_market_cap": math.log10(market_cap) if market_cap > 0 else math.nan,
        "revenue_growth": growth,
        "ebitda_margin": margin,
        "return_on_equity": roe,
        "beta": _regression_beta(company),
        "leverage": _leverage(_num(market.total_debt), market_cap),
        "ev_ebitda": _raw_multiple(enterprise_value, ebitda),
        "ev_sales": _raw_multiple(enterprise_value, revenue),
        "pe": _raw_multiple(market_cap, net_income),
    }


# --------------------------------------------------------------------------
# Feature matrix
# --------------------------------------------------------------------------


def _feature_matrix(
    frame: pd.DataFrame,
    features: Sequence[str],
    config: CompsConfig,
    fallbacks: list[str],
    target: str,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Clip, winsorize, impute and standardize ``features`` of ``frame``.

    ``frame`` holds the candidates plus the target. Returns the z-score matrix
    (weighted) and the features that survived (constant features are dropped).
    """
    cleaned: dict[str, pd.Series] = {}
    for name in features:
        column = frame[name].astype(float)
        bounds = config.feature_bounds.get(name)
        if bounds is not None:
            column = column.clip(lower=bounds[0], upper=bounds[1])
        observed = column.dropna()
        if observed.empty:
            fallbacks.append(f"feature_dropped_no_data:{name}")
            continue
        low = float(observed.quantile(config.winsor_lower))
        high = float(observed.quantile(config.winsor_upper))
        column = column.clip(lower=low, upper=high)
        if math.isnan(column.loc[target]):
            fallbacks.append(f"target_feature_imputed:{name}")
        column = column.fillna(float(column.median()))
        std = float(column.std(ddof=0))
        if not math.isfinite(std) or std < 1e-12:
            fallbacks.append(f"feature_dropped_constant:{name}")
            continue
        weight = float(config.feature_weights.get(name, 1.0))
        cleaned[name] = (column - float(column.mean())) / std * weight
    kept = tuple(cleaned)
    return pd.DataFrame(cleaned, index=frame.index), kept


# --------------------------------------------------------------------------
# Clustering
# --------------------------------------------------------------------------


def _cluster(z: np.ndarray, config: CompsConfig) -> tuple[np.ndarray, int, dict[int, float]]:
    """K-means for every admissible k; pick the best silhouette score.

    Returns labels (all zero when clustering is not possible), the chosen k
    (1 when no clustering was done) and the silhouette score per k tried.
    """
    n_points = z.shape[0]
    no_clusters = np.zeros(n_points, dtype=int)
    n_distinct = np.unique(z.round(10), axis=0).shape[0]
    upper = min(config.k_max, n_points // max(config.min_points_per_cluster, 1), n_distinct - 1)
    if n_points < config.min_cluster_points or upper < config.k_min:
        return no_clusters, 1, {}

    scores: dict[int, float] = {}
    labels_by_k: dict[int, np.ndarray] = {}
    for k in range(config.k_min, upper + 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = KMeans(n_clusters=k, n_init=config.n_init, random_state=config.random_state)
            labels = model.fit_predict(z)
        if np.unique(labels).shape[0] < 2:
            continue
        scores[k] = float(silhouette_score(z, labels))
        labels_by_k[k] = labels.astype(int)
    if not scores:
        return no_clusters, 1, {}
    # Highest score wins; ties go to the smaller k (deterministic).
    best_k = min(scores, key=lambda k: (-scores[k], k))
    return labels_by_k[best_k], best_k, scores


# --------------------------------------------------------------------------
# Multiples
# --------------------------------------------------------------------------


def _tukey_mask(values: pd.Series, config: CompsConfig) -> pd.Series:
    """True for observations inside the Tukey fence on the log multiple."""
    observed = values.dropna()
    keep = values.notna()
    if len(observed) < config.iqr_min_observations:
        return keep
    logs = np.log(observed)
    q1, q3 = float(logs.quantile(0.25)), float(logs.quantile(0.75))
    spread = q3 - q1
    low, high = q1 - config.iqr_factor * spread, q3 + config.iqr_factor * spread
    inside = (logs >= low - 1e-12) & (logs <= high + 1e-12)
    keep.loc[observed.index] = inside
    return keep


def _summary_row(values: pd.Series) -> dict[str, float]:
    """min / quartiles / mean / max of a non-empty series of multiples."""
    array = values.to_numpy(dtype=float)
    return {
        "n": float(len(array)),
        "min": float(array.min()),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.percentile(array, 50)),
        "mean": float(array.mean()),
        "p75": float(np.percentile(array, 75)),
        "max": float(array.max()),
    }


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------


def run_comps(
    company: CompanyData,
    universe: Sequence[PeerSnapshot],
    *,
    config: CompsConfig | None = None,
) -> CompsResult:
    """Run the comparable company analysis for ``company``.

    Parameters
    ----------
    company:
        The target. Its metrics (revenue, EBITDA, net income) and the equity
        bridge come from here, in the trading currency.
    universe:
        Peer snapshots (one share class per company). The target may or may
        not be part of it; it is never used as its own peer.
    config:
        Optional :class:`CompsConfig`; defaults are used when omitted.

    Raises
    ------
    NotApplicableError
        When no comps can be run at all: invalid share count / market cap, or
        no usable candidate in the universe. A company for which individual
        multiples are not applicable still gets a result (see ``skipped``).
    """
    cfg = config or CompsConfig()
    ticker = company.ticker
    market = company.market
    fallbacks: list[str] = []
    skipped: dict[str, str] = {}

    shares = _num(market.shares_outstanding)
    price = _num(market.price)
    if not shares > 0:
        raise NotApplicableError(f"{ticker}: shares outstanding missing or non-positive")
    if not _num(market.market_cap) > 0:
        raise NotApplicableError(f"{ticker}: market cap missing or non-positive")

    frame = _universe_frame(universe)
    if frame.empty:
        raise NotApplicableError(f"{ticker}: the peer universe is empty")

    # ---- target row -------------------------------------------------------
    is_financial = company.profile.is_financial
    if ticker in frame.index:
        target_row = frame.loc[ticker].to_dict()
    else:
        target_row = _target_row_from_company(company)
        fallbacks.append("target_not_in_universe")
    # Sector/industry of the target always follow CompanyData when available.
    sector = company.profile.sector or target_row.get("sector")
    industry = company.profile.industry or target_row.get("industry")
    target_row["sector"], target_row["industry"] = sector, industry
    target_row["is_financial"] = is_financial

    applicable = ("pe",) if is_financial else MULTIPLES
    if is_financial:
        reason = "financial / real estate company: enterprise value multiples are not meaningful"
        skipped["ev_ebitda"] = reason
        skipped["ev_sales"] = reason

    # ---- candidates -------------------------------------------------------
    for multiple in MULTIPLES:
        frame[multiple] = _bounded(frame[multiple], cfg.multiple_bounds[multiple])
    root, name_key = _ticker_root(ticker), company.profile.name.strip().casefold()
    is_self = (
        (frame.index == ticker)
        | (frame.index.map(_ticker_root) == root)
        | (frame["name"].astype(str).str.strip().str.casefold() == name_key)
    )
    eligible = frame.loc[
        ~is_self & (frame["market_cap"] > 0) & frame[list(applicable)].notna().any(axis=1)
    ]
    same_sector = eligible.loc[eligible["sector"] == sector] if sector else eligible.iloc[0:0]
    family = _industry_family(industry)
    same_family = same_sector.loc[same_sector["industry"].map(_industry_family) == family]
    if is_financial and family and len(same_family) >= cfg.min_peers:
        candidates, candidate_scope = same_family, "industry_family"
    elif len(same_sector) >= cfg.min_peers:
        candidates, candidate_scope = same_sector, "sector"
    else:
        candidates, candidate_scope = (
            eligible.loc[eligible["is_financial"] == is_financial],
            ("universe"),
        )
        fallbacks.append("sector_too_small_widened_to_universe")
        if candidates.empty:
            candidates = eligible
    if candidates.empty:
        raise NotApplicableError(f"{ticker}: no usable peer candidates in the universe")

    # ---- features, clustering, distances ----------------------------------
    target_frame = pd.DataFrame([target_row], index=pd.Index([ticker], name=frame.index.name))
    target_frame = target_frame.reindex(columns=frame.columns)
    work = pd.concat([candidates, target_frame])
    features = FEATURES_FINANCIAL if is_financial else FEATURES_NON_FINANCIAL
    z, kept = _feature_matrix(work, features, cfg, fallbacks, ticker)

    if kept:
        labels, chosen_k, scores = _cluster(z.to_numpy(dtype=float), cfg)
        deltas = z.to_numpy(dtype=float) - z.loc[ticker].to_numpy(dtype=float)
        distance = np.sqrt((deltas**2).sum(axis=1))
    else:
        labels, chosen_k, scores = np.zeros(len(work), dtype=int), 1, {}
        distance = np.zeros(len(work))
        fallbacks.append("no_usable_features")
    if chosen_k == 1:
        fallbacks.append("no_clustering_too_few_candidates")

    work = work.assign(cluster=labels, distance=distance)
    target_cluster = int(work.loc[ticker, "cluster"])
    cand = work.drop(index=ticker).copy()
    cand["same_industry"] = (cand["industry"] == industry) if industry else False
    same_sector_flag = (cand["sector"] == sector) if sector else False
    same_family_flag = same_sector_flag & (cand["industry"].map(_industry_family) == family)
    penalty = np.where(
        cand["same_industry"],
        0.0,
        np.where(same_family_flag, 0.5, np.where(same_sector_flag, 1.0, 2.0)),
    )
    cand["ranking_distance"] = cand["distance"] + cfg.industry_penalty * penalty
    cand["_ticker"] = cand.index
    cand = cand.sort_values(["ranking_distance", "_ticker"], kind="mergesort").drop(
        columns="_ticker"
    )

    # ---- peer group size --------------------------------------------------
    in_cluster = cand["cluster"] == target_cluster
    selection = pd.Series("", index=cand.index, dtype=object)
    cluster_members = cand.index[in_cluster]
    if len(cluster_members) > cfg.max_peers:
        fallbacks.append("cluster_capped_to_nearest")
        cluster_members = cluster_members[: cfg.max_peers]
    selection.loc[cluster_members] = "cluster" if chosen_k > 1 else "all_candidates"
    if len(cluster_members) < cfg.min_peers:
        outside = cand.index[~in_cluster][: cfg.min_peers - len(cluster_members)]
        if len(outside) > 0:
            fallbacks.append("cluster_too_small_added_nearest_neighbours")
            selection.loc[outside] = "nearest_neighbour"
    core = selection != ""
    if int(core.sum()) < cfg.min_peers:
        fallbacks.append("fewer_than_min_peers_available")

    # ---- target metrics -----------------------------------------------------
    income = company.statements.income
    metrics = {m: _latest(income, _METRICS[m]) for m in MULTIPLES}
    enterprise_value = _num(market.enterprise_value)
    target_multiples: dict[str, float | None] = {}
    for multiple in MULTIPLES:
        numerator = _num(market.market_cap) if multiple == "pe" else enterprise_value
        value = _raw_multiple(numerator, metrics[multiple])
        target_multiples[multiple] = None if math.isnan(value) else value

    # ---- multiples --------------------------------------------------------
    used: dict[str, pd.Series] = {}
    summary = pd.DataFrame(
        np.nan, index=pd.Index(MULTIPLES, name="multiple"), columns=list(_SUMMARY_COLUMNS)
    )
    summary["n"] = 0.0
    for multiple in applicable:
        has_value = cand[multiple].notna()
        chosen = core & has_value
        if int(chosen.sum()) < cfg.min_peers and metrics[multiple] > 0:
            extra = cand.index[~core & has_value][: cfg.min_peers - int(chosen.sum())]
            if len(extra) > 0:
                fallbacks.append(f"multiple_topped_up_with_nearest:{multiple}")
                for peer in extra:
                    if selection.loc[peer] == "":
                        selection.loc[peer] = f"fill:{multiple}"
                    else:
                        selection.loc[peer] = f"{selection.loc[peer]},{multiple}"
                chosen = chosen.copy()
                chosen.loc[extra] = True
        values = cand.loc[chosen, multiple]
        inside = _tukey_mask(values, cfg)
        if not bool(inside.all()):
            fallbacks.append(f"multiple_outliers_removed:{multiple}")
        values = values.loc[inside]
        flag = pd.Series(False, index=cand.index)
        flag.loc[values.index] = True
        used[multiple] = flag
        if len(values) > 0:
            summary.loc[multiple] = _summary_row(values)
    for multiple in MULTIPLES:
        if multiple not in used:
            used[multiple] = pd.Series(False, index=cand.index)

    # ---- implied values ----------------------------------------------------
    bridge = {
        "total_debt": _num0(market.total_debt),
        "cash": _num0(market.cash),
        "minority_interest": _num0(market.minority_interest),
        "preferred_equity": _num0(market.preferred_equity),
    }
    bridge["net_debt"] = bridge["total_debt"] - bridge["cash"]
    ev_to_equity = bridge["net_debt"] + bridge["minority_interest"] + bridge["preferred_equity"]

    implied_rows: dict[str, dict[str, Any]] = {}
    for multiple in applicable:
        metric = metrics[multiple]
        n_obs = int(summary.loc[multiple, "n"])
        if math.isnan(metric) or metric <= 0:
            skipped[multiple] = f"target {_METRICS[multiple]} is missing or non-positive"
            continue
        if n_obs < cfg.min_observations:
            skipped[multiple] = (
                f"only {n_obs} usable peer multiple(s), {cfg.min_observations} required"
            )
            continue
        quartiles = [float(summary.loc[multiple, c]) for c in ("p25", "median", "p75")]
        if multiple == "pe":
            equity = [q * metric for q in quartiles]
            ev = [e + ev_to_equity for e in equity]
        else:
            ev = [q * metric for q in quartiles]
            equity = [e - ev_to_equity for e in ev]
        if equity[2] <= 0:
            skipped[multiple] = (
                "implied equity value is non-positive across the whole range "
                "(net debt exceeds the implied enterprise value)"
            )
            continue
        floored = equity[0] < 0 or equity[1] < 0
        if floored:
            fallbacks.append(f"negative_equity_floored_at_zero:{multiple}")
        per_share = [max(e, 0.0) / shares for e in equity]
        implied_rows[multiple] = {
            "metric": _METRICS[multiple],
            "metric_value": metric,
            "n_peers": n_obs,
            "multiple_low": quartiles[0],
            "multiple_base": quartiles[1],
            "multiple_high": quartiles[2],
            "ev_low": ev[0],
            "ev_base": ev[1],
            "ev_high": ev[2],
            "equity_low": max(equity[0], 0.0),
            "equity_base": max(equity[1], 0.0),
            "equity_high": equity[2],
            "equity_low_unfloored": equity[0],
            "per_share_low": per_share[0],
            "per_share_base": per_share[1],
            "per_share_high": per_share[2],
            "base_to_price": per_share[1] / price if price > 0 else math.nan,
            "floored": floored,
        }
    implied_columns = [
        "metric",
        "metric_value",
        "n_peers",
        "multiple_low",
        "multiple_base",
        "multiple_high",
        "ev_low",
        "ev_base",
        "ev_high",
        "equity_low",
        "equity_base",
        "equity_high",
        "equity_low_unfloored",
        "per_share_low",
        "per_share_base",
        "per_share_high",
        "base_to_price",
        "floored",
    ]
    implied = pd.DataFrame.from_dict(implied_rows, orient="index", columns=implied_columns)
    implied.index.name = "multiple"

    # ---- output tables ----------------------------------------------------
    base_columns = [
        "name",
        "sector",
        "industry",
        "same_industry",
        "cluster",
        "distance",
        "ranking_distance",
        "market_cap",
        "enterprise_value",
        *features,
        *MULTIPLES,
    ]
    cand["selection"] = selection
    is_peer = cand["selection"] != ""
    peers = cand.loc[is_peer, [*base_columns[:3], "same_industry", "selection", *base_columns[4:]]]
    peers = peers.copy()
    for multiple in MULTIPLES:
        peers[f"used_{multiple}"] = used[multiple].loc[peers.index].astype(bool)
    peers.index.name = "ticker"

    z_columns = z.drop(index=ticker).add_prefix("z_").reindex(cand.index)
    candidates_out = pd.concat([cand[base_columns], z_columns], axis=1)
    candidates_out["is_peer"] = is_peer
    candidates_out.index.name = "ticker"

    raw_target = pd.Series({f: _num(target_row.get(f)) for f in features}, name="raw")
    z_target = z.loc[ticker].reindex(list(features)).rename("z") if kept else raw_target * np.nan
    target_features = pd.concat([raw_target, z_target.rename("z")], axis=1)
    target_features.index.name = "feature"

    return CompsResult(
        ticker=ticker,
        name=company.profile.name,
        currency=company.profile.currency,
        price=price,
        shares_outstanding=shares,
        is_financial=is_financial,
        candidate_scope=candidate_scope,
        peers=peers,
        candidates=candidates_out,
        target_features=target_features,
        feature_names=kept,
        chosen_k=chosen_k,
        target_cluster=target_cluster,
        silhouette_scores=scores,
        multiples_summary=summary,
        target_multiples=target_multiples,
        implied_values=implied,
        fallbacks=list(dict.fromkeys(fallbacks)),
        skipped=skipped,
        equity_bridge=bridge,
    )
