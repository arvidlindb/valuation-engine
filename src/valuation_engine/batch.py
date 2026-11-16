"""Batch valuation of the whole universe.

Public API::

    run_universe(tickers=None, *, output_dir="output", ...) -> BatchResult
    report_row(report) -> dict            # one CSV row from a ValuationReport

``run_universe`` values every company of ``data.list_universe()`` with the
pipeline and writes

* ``<output_dir>/universe_valuation.csv``: one row per company (columns in
  ``CSV_COLUMNS``), and
* ``<output_dir>/universe_summary.md``: coverage per method, the reasons why
  models were not applicable and the distribution of base value / price.

The universe table (peer snapshots) and the market parameters are loaded once
and shared by all companies; only the company files are read one by one. A
company that cannot be loaded becomes a ``no_data`` row, and no failure of a
single company stops the run. Charts and Excel workbooks are only written with
``with_reports=True``.

Speed: about 0.1 seconds per company with 2,000 simulations (most of it
k-means in comps), i.e. roughly 70 seconds for 690 companies in one
process. That is why there is no process pool.
"""

from __future__ import annotations

import logging
import math
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

import pandas as pd

from valuation_engine import data as data_layer
from valuation_engine.core.types import MarketParameters, PeerSnapshot, ValuationMethod
from valuation_engine.data import DataError, ParameterSource
from valuation_engine.pipeline import (
    METHOD_NAMES,
    ValuationReport,
    save_report,
    value_company,
)
from valuation_engine.viz.protocols import METHOD_KEYS

__all__ = [
    "CSV_COLUMNS",
    "CSV_NAME",
    "DEFAULT_BATCH_SIMULATIONS",
    "STATUS_ERROR",
    "STATUS_NO_DATA",
    "STATUS_OK",
    "STATUS_PARTIAL",
    "SUMMARY_NAME",
    "BatchResult",
    "build_summary_markdown",
    "report_row",
    "run_universe",
]

logger = logging.getLogger(__name__)

CSV_NAME = "universe_valuation.csv"
SUMMARY_NAME = "universe_summary.md"
#: Monte Carlo simulations per company in batch mode (single runs use 10,000).
DEFAULT_BATCH_SIMULATIONS = 2_000

#: Every model produced a result.
STATUS_OK = "ok"
#: The company was loaded but at least one model is not applicable or failed
#: (``methods_count`` tells how many football-field methods remain, possibly 0).
STATUS_PARTIAL = "partial"
#: The company could not be loaded (no statements at the data provider etc.).
STATUS_NO_DATA = "no_data"
#: Unexpected failure outside the models (should not occur).
STATUS_ERROR = "error"

#: Football-field methods with low / base / high / base_to_price columns.
_RANGE_METHODS: tuple[ValuationMethod, ...] = (
    ValuationMethod.DCF_GORDON,
    ValuationMethod.DCF_EXIT_MULTIPLE,
    ValuationMethod.COMPS_EV_EBITDA,
    ValuationMethod.COMPS_EV_SALES,
    ValuationMethod.COMPS_PE,
    ValuationMethod.LBO,
)
_METHOD_LABELS: dict[str, str] = {
    "dcf_gordon": "DCF, Gordon growth",
    "dcf_exit_multiple": "DCF, exit multiple",
    "comps_ev_ebitda": "Comps, EV/EBITDA",
    "comps_ev_sales": "Comps, EV/Sales",
    "comps_pe": "Comps, P/E",
    "lbo": "LBO, ability to pay",
    "mc": "Monte Carlo, median",
}

CSV_COLUMNS: tuple[str, ...] = (
    "ticker",
    "name",
    "sector",
    "industry",
    "status",
    "methods_count",
    "currency",
    "as_of",
    "market_cap",
    "market_cap_sek",
    "price",
    *(
        f"{method.value}_{suffix}"
        for method in _RANGE_METHODS
        for suffix in ("low", "base", "high", "base_to_price")
    ),
    "mc_p10",
    "mc_p50",
    "mc_p90",
    "mc_p50_to_price",
    "mc_prob_above_price",
    "wacc",
    "beta",
    "size_premium",
    "exit_multiple",
    "dcf_peer_source",
    "peers_count",
    "lbo_irr",
    "lbo_moic",
    "lbo_entry_multiple",
    *(f"na_{key}" for key in METHOD_KEYS),
    "errors",
    "warnings_count",
)


@dataclass
class BatchResult:
    """Outcome of :func:`run_universe`."""

    #: One row per company, columns ``CSV_COLUMNS``.
    frame: pd.DataFrame
    csv_path: Path
    summary_path: Path
    #: Wall-clock seconds of the whole run.
    seconds: float
    #: Ticker -> message for companies that could not be loaded or crashed.
    failures: dict[str, str] = field(default_factory=dict)

    @property
    def status_counts(self) -> dict[str, int]:
        return {str(k): int(v) for k, v in self.frame["status"].value_counts().items()}


# --------------------------------------------------------------------------
# Rows
# --------------------------------------------------------------------------


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def report_row(report: ValuationReport, snapshot: PeerSnapshot | None = None) -> dict[str, Any]:
    """Flatten one report into a CSV row (keys ``CSV_COLUMNS``)."""
    company = report.company
    profile, market = company.profile, company.market
    price = float(market.price)
    row: dict[str, Any] = dict.fromkeys(CSV_COLUMNS)
    row.update(
        ticker=company.ticker,
        name=profile.name,
        sector=profile.sector,
        industry=profile.industry,
        status=STATUS_OK if report.status == "ok" else STATUS_PARTIAL,
        methods_count=len(report.valuations),
        currency=market.currency,
        as_of=str(market.as_of),
        market_cap=_finite(market.market_cap),
        market_cap_sek=_finite(snapshot.market_cap) if snapshot is not None else None,
        price=price,
    )
    by_method = {result.method: result for result in report.valuations}
    for method in _RANGE_METHODS:
        result = by_method.get(method)
        if result is None:
            continue
        row[f"{method.value}_low"] = _finite(result.low)
        row[f"{method.value}_base"] = _finite(result.base)
        row[f"{method.value}_high"] = _finite(result.high)
        if price > 0 and _finite(result.base) is not None:
            row[f"{method.value}_base_to_price"] = result.base / price

    mc = report.monte_carlo
    if mc is not None:
        row["mc_p10"] = _finite(mc.percentiles.get(10))
        row["mc_p50"] = _finite(mc.percentiles.get(50))
        row["mc_p90"] = _finite(mc.percentiles.get(90))
        if price > 0 and row["mc_p50"] is not None:
            row["mc_p50_to_price"] = row["mc_p50"] / price
        row["mc_prob_above_price"] = _finite(mc.prob_above_price)

    dcf = report.dcf
    if dcf is not None:
        row["wacc"] = _finite(dcf.wacc)
        row["exit_multiple"] = _finite(dcf.exit_multiple)
        row["dcf_peer_source"] = report.dcf_peer_source
        if dcf.wacc_result is not None:
            row["beta"] = _finite(dcf.wacc_result.beta)
            row["size_premium"] = _finite(dcf.wacc_result.size_premium)
    if report.comps is not None:
        row["peers_count"] = len(report.comps.peers)
    lbo = report.lbo
    if lbo is not None:
        row["lbo_irr"] = _finite(lbo.irr)
        row["lbo_moic"] = _finite(lbo.moic)
        row["lbo_entry_multiple"] = _finite(lbo.entry_multiple)

    for key in METHOD_KEYS:
        row[f"na_{key}"] = report.not_applicable.get(key)
    if report.errors:
        row["errors"] = "; ".join(f"{key}: {message}" for key, message in report.errors.items())
    row["warnings_count"] = len(report.warnings)
    return row


def _failure_row(
    ticker: str, name: str | None, snapshot: PeerSnapshot | None, status: str, message: str
) -> dict[str, Any]:
    row: dict[str, Any] = dict.fromkeys(CSV_COLUMNS)
    row.update(
        ticker=ticker,
        name=name or (snapshot.name if snapshot is not None else None),
        sector=snapshot.sector if snapshot is not None else None,
        industry=snapshot.industry if snapshot is not None else None,
        status=status,
        methods_count=0,
        market_cap_sek=_finite(snapshot.market_cap) if snapshot is not None else None,
        errors=message,
        warnings_count=0,
    )
    return row


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------


def _reason_category(reason: str, ticker: str) -> str:
    """Short, ticker-free category of a not-applicable reason."""
    text = reason.strip()
    if text.startswith(f"{ticker}:"):
        text = text[len(ticker) + 1 :].strip()
    for separator in (":", " ("):
        head = text.split(separator, 1)[0].strip()
        if head:
            text = head
    return text[:90]


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(" --- " for _ in headers) + "|"]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return lines


def _share(count: int, total: int) -> str:
    return f"{count / total:.0%}" if total else "n/a"


def build_summary_markdown(
    frame: pd.DataFrame,
    *,
    seconds: float,
    n_simulations: int,
    seed: int | None,
    size_premium: bool,
    use_comps_peers: bool,
    params: MarketParameters,
    parameter_sources: dict[str, ParameterSource] | None = None,
    offline: bool = False,
) -> str:
    """Markdown summary of a batch run (plain ASCII punctuation)."""
    total = len(frame)
    loaded = frame[frame["status"].isin([STATUS_OK, STATUS_PARTIAL])]
    n_loaded = len(loaded)
    lines: list[str] = ["# Universe valuation summary", ""]
    lines.append(
        f"Generated {datetime.now():%Y-%m-%d %H:%M}. {total} companies, {seconds:,.0f} seconds "
        f"({seconds / total if total else 0.0:.2f} s per company). Monte Carlo: "
        f"{n_simulations:,} simulations per company, seed {seed}. Size premium in the cost of "
        f"equity: {'on' if size_premium else 'off'}. DCF peers: "
        f"{'comps peer group' if use_comps_peers else 'full universe'}. Data: "
        f"{'offline (fixtures and cache)' if offline else 'cache first, network for the rest'}."
    )
    lines.append("")
    lines.append("## Market parameters")
    lines.append("")
    rows = []
    for name in ("risk_free_rate", "equity_risk_premium", "tax_rate", "long_term_growth"):
        source = (parameter_sources or {}).get(name)
        rows.append(
            (
                name,
                f"{getattr(params, name):.3%}",
                source.source if source is not None else "n/a",
                (source.as_of or "n/a") if source is not None else "n/a",
            )
        )
    rows.append(("benchmark", params.benchmark, "project default", "n/a"))
    lines.extend(_md_table(("Parameter", "Value", "Source", "As of"), rows))

    lines.extend(["", "## Status", ""])
    counts = Counter(frame["status"])
    meanings = {
        STATUS_OK: "all four models produced a result",
        STATUS_PARTIAL: "loaded, at least one model not applicable or failed",
        STATUS_NO_DATA: "the company could not be loaded",
        STATUS_ERROR: "unexpected failure outside the models",
    }
    lines.extend(
        _md_table(
            ("Status", "Companies", "Share", "Meaning"),
            [
                (status, counts.get(status, 0), _share(counts.get(status, 0), total), meaning)
                for status, meaning in meanings.items()
            ],
        )
    )
    without_any = int((loaded["methods_count"] == 0).sum()) if n_loaded else 0
    lines.append("")
    lines.append(
        f"{without_any} of the {n_loaded} loaded companies have no valuation method at all."
    )

    lines.extend(["", "## Coverage per method", ""])
    coverage_rows = []
    base_columns = {
        **{method.value: f"{method.value}_base" for method in _RANGE_METHODS},
        "mc": "mc_p50",
    }
    for key, column in base_columns.items():
        count = int(frame[column].notna().sum())
        coverage_rows.append(
            (_METHOD_LABELS[key], count, _share(count, total), _share(count, n_loaded))
        )
    lines.extend(
        _md_table(
            ("Method", "Companies with a value", "Share of universe", "Share of loaded"),
            coverage_rows,
        )
    )

    lines.extend(["", "## Model outcomes", ""])
    result_columns = {
        "dcf": "dcf_gordon_base",
        "comps": "peers_count",
        "lbo": "lbo_irr",
        "monte_carlo": "mc_p50",
    }
    errors = frame["errors"].fillna("").astype(str)
    outcome_rows = []
    for key in METHOD_KEYS:
        ran = int(loaded[result_columns[key]].notna().sum()) if n_loaded else 0
        not_applicable = int(loaded[f"na_{key}"].notna().sum()) if n_loaded else 0
        crashed = int(errors[loaded.index].str.contains(f"{key}: ", regex=False).sum())
        outcome_rows.append((METHOD_NAMES[key], ran, not_applicable, crashed))
    lines.extend(
        _md_table(("Model", "Ran", "Not applicable", "Failed with an error"), outcome_rows)
    )
    lines.append("")
    lines.append(
        "A model that ran can still give no football field bar: comps without any applicable "
        "multiple, or an LBO in which no positive price reaches the base target IRR."
    )

    lines.extend(["", "## Why models were not applicable", ""])
    for key in METHOD_KEYS:
        column = f"na_{key}"
        reasons = Counter(
            _reason_category(str(reason), str(ticker))
            for ticker, reason in zip(loaded["ticker"], loaded[column], strict=True)
            if isinstance(reason, str) and reason
        )
        lines.append(f"### {METHOD_NAMES[key]}")
        lines.append("")
        if reasons:
            lines.extend(_md_table(("Reason", "Companies"), reasons.most_common()))
        else:
            lines.append("Applicable to every loaded company.")
        lines.append("")

    lines.extend(["## Base value relative to the share price", ""])
    lines.append(
        "Ratio of the base value per share to the current price (1.00 means the method agrees "
        "with the market). Percentiles across the companies that have the method."
    )
    lines.append("")
    ratio_columns = {
        **{method.value: f"{method.value}_base_to_price" for method in _RANGE_METHODS},
        "mc": "mc_p50_to_price",
    }
    ratio_rows = []
    for key, column in ratio_columns.items():
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            ratio_rows.append((_METHOD_LABELS[key], 0, *("n/a",) * 6))
            continue
        quantiles = values.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
        ratio_rows.append(
            (
                _METHOD_LABELS[key],
                len(values),
                *(f"{quantiles.loc[q]:.2f}" for q in (0.10, 0.25, 0.50, 0.75, 0.90)),
                f"{(values > 1.0).mean():.0%}",
            )
        )
    lines.extend(
        _md_table(
            ("Method", "Companies", "P10", "P25", "Median", "P75", "P90", "Above price"),
            ratio_rows,
        )
    )
    lines.append("")
    lines.append(
        "Negative ratios are possible: a DCF value per share is negative when net debt exceeds "
        "the enterprise value. Values are reported as computed."
    )

    no_data = frame[frame["status"].isin([STATUS_NO_DATA, STATUS_ERROR])]
    if len(no_data):
        lines.extend(["", "## Companies without a valuation run", ""])
        lines.extend(
            _md_table(
                ("Ticker", "Status", "Reason"),
                [
                    (row.ticker, row.status, str(row.errors).replace("|", "/"))
                    for row in no_data.itertuples()
                ],
            )
        )
    lines.append("")
    text = "\n".join(lines)
    return text.replace("\u2013", "-").replace("\u2014", "-")


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------


def run_universe(
    tickers: Sequence[str] | None = None,
    *,
    offline: bool = False,
    refresh: bool = False,
    output_dir: str | Path = "output",
    n_simulations: int = DEFAULT_BATCH_SIMULATIONS,
    seed: int | None = 42,
    size_premium: bool = True,
    use_comps_peers: bool = True,
    limit: int | None = None,
    with_reports: bool = False,
    charts: bool = True,
    excel: bool = True,
    progress: bool = True,
    stream: TextIO | None = None,
) -> BatchResult:
    """Value every company of the universe and write the CSV and the summary.

    Parameters
    ----------
    tickers:
        Companies to value; ``None`` means ``data.list_universe()``.
    offline, refresh:
        Passed to the data layer. Online mode reads the local cache first.
    output_dir:
        Folder for ``universe_valuation.csv`` and ``universe_summary.md``.
    n_simulations, seed:
        Monte Carlo settings per company.
    size_premium, use_comps_peers:
        As in ``pipeline.run_valuation``.
    limit:
        Value only the first ``limit`` companies (for testing). The files are
        then named ``universe_valuation_limit<N>.csv`` and
        ``universe_summary_limit<N>.md`` so that a full run is not overwritten.
    with_reports:
        Also write charts (``charts``) and the Excel workbook (``excel``) per
        company into ``<output_dir>/<TICKER>/``. Slow: about a second each.
    progress, stream:
        Print a progress line every 25 companies to ``stream`` (default stderr).
    """
    started = time.perf_counter()
    out = stream if stream is not None else sys.stderr
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)

    names: dict[str, str] = {}
    if tickers is None:
        entries, _ = data_layer.load_universe_entries(offline=offline, refresh=refresh)
        names = {entry.ticker: entry.name for entry in entries}
        tickers = [entry.ticker for entry in entries]
    tickers = list(tickers)
    if limit is not None:
        tickers = tickers[: max(int(limit), 0)]

    try:
        universe = data_layer.load_peer_snapshots(offline=offline)
    except DataError as exc:
        universe = []
        logger.warning("peer universe unavailable (%s); comps cannot be run", exc)
    snapshots = {peer.ticker: peer for peer in universe}
    sources = data_layer.market_parameter_sources(offline=offline, refresh=refresh)
    params = MarketParameters(
        **{
            name: source.value
            for name, source in sources.items()
            if name in MarketParameters.__dataclass_fields__
        }
    )

    rows: list[dict[str, Any]] = []
    failures: dict[str, str] = {}
    counts: Counter[str] = Counter()
    total = len(tickers)
    for index, ticker in enumerate(tickers, start=1):
        snapshot = snapshots.get(ticker)
        try:
            company = data_layer.load_company(ticker, offline=offline, refresh=refresh)
        except DataError as exc:
            failures[ticker] = f"{type(exc).__name__}: {exc}"
            rows.append(
                _failure_row(ticker, names.get(ticker), snapshot, STATUS_NO_DATA, failures[ticker])
            )
        except Exception as exc:  # noqa: BLE001 - one company must never stop the run
            failures[ticker] = f"{type(exc).__name__}: {exc}"
            logger.warning("%s: loading failed: %s", ticker, failures[ticker], exc_info=True)
            rows.append(
                _failure_row(ticker, names.get(ticker), snapshot, STATUS_ERROR, failures[ticker])
            )
        else:
            try:
                report = value_company(
                    company,
                    universe,
                    params,
                    parameter_sources=sources,
                    n_simulations=n_simulations,
                    seed=seed,
                    size_premium=size_premium,
                    use_comps_peers=use_comps_peers,
                )
                if with_reports:
                    save_report(report, folder / ticker, charts=charts, excel=excel)
                rows.append(report_row(report, snapshot))
            except Exception as exc:  # noqa: BLE001 - one company must never stop the run
                failures[ticker] = f"{type(exc).__name__}: {exc}"
                logger.warning("%s: valuation failed: %s", ticker, failures[ticker], exc_info=True)
                rows.append(
                    _failure_row(
                        ticker, names.get(ticker), snapshot, STATUS_ERROR, failures[ticker]
                    )
                )
        counts[rows[-1]["status"]] += 1
        if progress and (index % 25 == 0 or index == total):
            elapsed = time.perf_counter() - started
            print(
                f"[{index:>4}/{total}] {elapsed:6.1f} s  ok {counts[STATUS_OK]}, "
                f"partial {counts[STATUS_PARTIAL]}, no_data {counts[STATUS_NO_DATA]}, "
                f"error {counts[STATUS_ERROR]}  (last: {ticker})",
                file=out,
                flush=True,
            )

    frame = pd.DataFrame(rows, columns=list(CSV_COLUMNS))
    seconds = time.perf_counter() - started
    # A limited (test) run never overwrites the files of a full run.
    suffix = f"_limit{int(limit)}" if limit is not None else ""
    csv_path = folder / f"{Path(CSV_NAME).stem}{suffix}.csv"
    frame.round(6).to_csv(csv_path, index=False)
    summary_path = folder / f"{Path(SUMMARY_NAME).stem}{suffix}.md"
    summary_path.write_text(
        build_summary_markdown(
            frame,
            seconds=seconds,
            n_simulations=n_simulations,
            seed=seed,
            size_premium=size_premium,
            use_comps_peers=use_comps_peers,
            params=params,
            parameter_sources=sources,
            offline=data_layer.is_offline(offline),
        ),
        encoding="utf-8",
    )
    return BatchResult(
        frame=frame,
        csv_path=csv_path,
        summary_path=summary_path,
        seconds=time.perf_counter() - started,
        failures=failures,
    )
