"""Command line interface and the universe batch, offline on the fixtures."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import openpyxl
import pandas as pd
import pytest

from support import REPO_ROOT
from valuation_engine import batch, pipeline
from valuation_engine.__main__ import build_parser, main
from valuation_engine.core.types import ValuationMethod
from valuation_engine.export import SHEET_NAMES
from valuation_engine.viz import CHART_NAMES

SITECUSTOMIZE = """
import socket

def _blocked(*args, **kwargs):
    raise RuntimeError("network access attempted in an offline test subprocess")

socket.socket.connect = _blocked
socket.socket.connect_ex = _blocked
socket.create_connection = _blocked
socket.getaddrinfo = _blocked
"""


def run_cli(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """``python -m valuation_engine`` in a fresh interpreter with sockets disabled."""
    guard = tmp_path / "guard"
    guard.mkdir(exist_ok=True)
    (guard / "sitecustomize.py").write_text(SITECUSTOMIZE, encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(guard), str(REPO_ROOT / "src"), env.get("PYTHONPATH", "")]
    )
    env["MPLBACKEND"] = "Agg"
    return subprocess.run(
        [sys.executable, "-m", "valuation_engine", *args],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
        timeout=180,
    )


# ------------------------------------------------------------------------- CLI


def test_cli_values_atlas_copco_and_writes_all_deliverables(tmp_path: Path, atco_report) -> None:
    out = tmp_path / "tmp"
    done = run_cli(tmp_path, "ATCO-A.ST", "--offline", "--output-dir", str(out))
    assert done.returncode == 0, done.stderr[-2000:]
    assert "network access attempted" not in done.stderr
    assert "Atlas Copco AB (publ) (ATCO-A.ST)" in done.stdout
    assert "DCF - Gordon growth" in done.stdout and "LBO - ability to pay" in done.stdout

    folder = out / "ATCO-A.ST"
    workbook = folder / "ATCO-A.ST_valuation.xlsx"
    assert workbook.is_file()
    assert tuple(openpyxl.load_workbook(workbook).sheetnames) == SHEET_NAMES
    for name in CHART_NAMES:
        for suffix in ("png", "svg"):
            chart = folder / "charts" / f"{name}.{suffix}"
            assert chart.is_file() and chart.stat().st_size > 1_000, chart
    assert (folder / "charts" / "football_field.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    assert summary["ticker"] == "ATCO-A.ST" and summary["status"] == "ok"
    assert summary["price"] == 192.25
    assert len(summary["valuations"]) == 7
    assert summary["settings"]["n_simulations"] == 10_000 and summary["settings"]["seed"] == 42
    assert summary["not_applicable"] == {} and summary["errors"] == {}
    assert "ATCO-A.ST_valuation.xlsx" in summary["files"]
    assert len(summary["files"]) == 1 + 2 * len(CHART_NAMES)
    # a separate process with the same seed reproduces the in-process report exactly
    in_process = atco_report.summary()
    assert summary["valuations"] == in_process["valuations"]
    assert summary["dcf"] == in_process["dcf"]
    assert summary["monte_carlo"] == in_process["monte_carlo"]
    # nothing was written outside the requested folder
    assert sorted(p.name for p in tmp_path.iterdir()) == ["guard", "tmp"]


def test_cli_unknown_ticker_exits_non_zero(tmp_path: Path) -> None:
    done = run_cli(tmp_path, "NOPE.ST", "--offline", "--output-dir", str(tmp_path / "out"))
    assert done.returncode == 1
    assert "NOPE.ST: no data" in done.stderr
    assert "Traceback" not in done.stderr
    assert not (tmp_path / "out" / "NOPE.ST").exists()


def test_main_in_process_for_a_bank_and_a_missing_ticker(tmp_path: Path, capsys) -> None:
    code = main(
        ["seb-a.st", "NOPE.ST", "--offline", "--output-dir", str(tmp_path), "--simulations", "500"]
    )
    captured = capsys.readouterr()
    assert code == 1  # one of the two tickers could not be loaded
    assert "Skandinaviska Enskilda Banken" in captured.out
    assert "Could not be loaded: NOPE.ST" in captured.err
    folder = tmp_path / "SEB-A.ST"
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "partial"
    assert [row["method"] for row in summary["valuations"]] == ["comps_pe"]
    assert set(summary["not_applicable"]) == {"dcf", "lbo", "monte_carlo"}
    charts = sorted(p.stem for p in (folder / "charts").glob("*.png"))
    assert charts == ["comps_peer_multiples", "comps_silhouette", "football_field"]
    assert (folder / "SEB-A.ST_valuation.xlsx").is_file()


def test_main_flags(tmp_path: Path, capsys) -> None:
    code = main(
        [
            "ATCO-A.ST",
            "--offline",
            "--output-dir",
            str(tmp_path),
            "--simulations",
            "300",
            "--seed",
            "7",
            "--no-charts",
            "--no-excel",
            "--no-size-premium",
            "--universe-peers",
        ]
    )
    assert code == 0
    capsys.readouterr()
    folder = tmp_path / "ATCO-A.ST"
    assert sorted(p.name for p in folder.iterdir()) == ["summary.json"]
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    assert summary["files"] == []
    assert summary["settings"] == {
        "n_simulations": 300,
        "seed": 7,
        "size_premium": False,
        "dcf_peer_source": "universe",
    }
    assert summary["monte_carlo"]["seed"] == 7


@pytest.mark.parametrize("ticker", [".ST", "", "  "])
def test_malformed_ticker_is_reported_without_a_traceback(
    ticker: str, tmp_path: Path, capsys
) -> None:
    assert main([ticker, "ATCO-A.ST", "--offline", "--output-dir", str(tmp_path),
                 "--simulations", "100", "--no-charts", "--no-excel"]) == 1  # fmt: skip
    captured = capsys.readouterr()
    assert "no data, the company could not be loaded" in captured.err
    assert "not a valid ticker" in captured.err and "Could not be loaded:" in captured.err
    assert (tmp_path / "ATCO-A.ST" / "summary.json").is_file()  # the valid one still ran
    batch_result = batch.run_universe(
        [ticker], offline=True, output_dir=tmp_path / "b", n_simulations=100, progress=False
    )
    assert batch_result.frame["status"].tolist() == ["no_data"]


def test_negative_seed_draws_fresh_entropy(tmp_path: Path, capsys) -> None:
    argv = ["ATCO-A.ST", "--offline", "--output-dir", str(tmp_path), "--simulations", "200"]
    argv += ["--no-charts", "--no-excel", "--seed", "-1"]
    medians = []
    for _ in range(2):
        assert main(argv) == 0
        summary = json.loads((tmp_path / "ATCO-A.ST" / "summary.json").read_text("utf-8"))
        assert summary["settings"]["seed"] is None
        medians.append(summary["monte_carlo"]["percentiles"]["p50"])
    capsys.readouterr()
    assert medians[0] != medians[1]


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["ATCO-A.ST", "--all"],
        ["ATCO-A.ST", "--simulations", "0"],
        ["ATCO-A.ST", "--limit", "3"],
        ["ATCO-A.ST", "--with-reports"],
        ["--no-such-flag"],
    ],
)
def test_invalid_arguments_exit_with_code_two(argv: list[str], capsys) -> None:
    with pytest.raises(SystemExit) as error:
        main(argv)
    assert error.value.code == 2
    capsys.readouterr()


def test_parser_defaults() -> None:
    args = build_parser().parse_args(["ATCO-A.ST"])
    assert args.output_dir == Path("output")
    assert (args.offline, args.refresh, args.all_companies) == (False, False, False)
    assert args.seed == 42 and args.simulations is None
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(["--version"])
    assert error.value.code == 0


def test_cli_all_with_limit_writes_the_csv_and_reports_missing_data(tmp_path: Path, capsys) -> None:
    code = main(
        [
            "--all",
            "--offline",
            "--limit",
            "5",
            "--output-dir",
            str(tmp_path),
            "--quiet",
            "--simulations",
            "200",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0  # companies without data are not an error
    # a limited run gets its own file names, so that it never overwrites a full run
    assert not (tmp_path / batch.CSV_NAME).exists()
    frame = pd.read_csv(tmp_path / "universe_valuation_limit5.csv")
    # the first five universe tickers; only ABB is in the fixtures
    assert frame["ticker"].tolist() == ["4C.ST", "8TRA.ST", "AAC.ST", "AAK.ST", "ABB.ST"]
    assert frame["status"].tolist() == ["no_data"] * 4 + ["ok"]
    assert frame.loc[0, "name"] == "4C Group AB (publ)"
    assert frame.loc[:3, "methods_count"].tolist() == [0, 0, 0, 0]
    assert "5 companies" in out and "no_data 4" in out and "ok 1" in out
    assert (tmp_path / "universe_summary_limit5.md").is_file()


# ----------------------------------------------------------------------- batch

BATCH_TICKERS = ["ATCO-A.ST", "SEB-A.ST", "ABB.ST", "VSURE.ST", "NIBE-B.ST", "NOPE.ST"]


@pytest.fixture(scope="module")
def batch_result(tmp_path_factory: pytest.TempPathFactory) -> batch.BatchResult:
    folder = tmp_path_factory.mktemp("batch")
    return batch.run_universe(
        BATCH_TICKERS, offline=True, output_dir=folder, n_simulations=500, seed=42, progress=False
    )


def test_batch_writes_the_csv_with_the_documented_columns(batch_result) -> None:
    assert batch_result.csv_path.name == "universe_valuation.csv"
    assert batch_result.summary_path.name == "universe_summary.md"
    frame = pd.read_csv(batch_result.csv_path)
    assert tuple(frame.columns) == batch.CSV_COLUMNS
    assert frame["ticker"].tolist() == BATCH_TICKERS
    assert list(batch_result.frame.columns) == list(batch.CSV_COLUMNS)
    expected_columns = {
        "ticker", "name", "sector", "industry", "status", "methods_count", "currency", "as_of",
        "market_cap", "market_cap_sek", "price", "mc_p10", "mc_p50", "mc_p90", "mc_p50_to_price",
        "mc_prob_above_price", "wacc", "beta", "size_premium", "exit_multiple", "dcf_peer_source",
        "peers_count", "lbo_irr", "lbo_moic", "lbo_entry_multiple", "na_dcf", "na_comps", "na_lbo",
        "na_monte_carlo", "errors", "warnings_count",
    }  # fmt: skip
    for method in ("dcf_gordon", "dcf_exit_multiple", "comps_ev_ebitda", "comps_ev_sales",
                   "comps_pe", "lbo"):  # fmt: skip
        expected_columns |= {f"{method}_{s}" for s in ("low", "base", "high", "base_to_price")}
    assert set(batch.CSV_COLUMNS) == expected_columns
    assert len(batch.CSV_COLUMNS) == len(set(batch.CSV_COLUMNS))


def test_batch_statuses(batch_result) -> None:
    frame = batch_result.frame.set_index("ticker")
    assert frame["status"].to_dict() == {
        "ATCO-A.ST": "ok",
        "SEB-A.ST": "partial",  # a bank: P/E comps only
        "ABB.ST": "ok",
        "VSURE.ST": "ok",
        "NIBE-B.ST": "ok",
        "NOPE.ST": "no_data",
    }
    assert batch_result.status_counts == {"ok": 4, "partial": 1, "no_data": 1}
    assert set(batch_result.failures) == {"NOPE.ST"}
    assert "DataUnavailableError" in batch_result.failures["NOPE.ST"]
    assert frame.loc["NOPE.ST", "methods_count"] == 0
    assert "not available offline" in frame.loc["NOPE.ST", "errors"]
    bank = frame.loc["SEB-A.ST"]
    assert bank["methods_count"] == 1
    assert bank["na_dcf"].startswith("financial sector") and bank["na_comps"] is None
    assert math.isfinite(bank["comps_pe_base"])
    assert bank["dcf_gordon_base"] is None or pd.isna(bank["dcf_gordon_base"])
    assert frame.loc["VSURE.ST", "currency"] == "EUR"
    # market cap in SEK from the peer table: 11.2 bn EUR is 124 bn SEK
    assert frame.loc["VSURE.ST", "market_cap_sek"] / frame.loc["VSURE.ST", "market_cap"] == (
        pytest.approx(11.1075, rel=1e-6)
    )


def test_batch_rows_equal_the_values_of_the_reports(
    batch_result, companies, universe, params
) -> None:
    csv = pd.read_csv(batch_result.csv_path).set_index("ticker")
    for ticker in ("ATCO-A.ST", "ABB.ST", "VSURE.ST", "NIBE-B.ST", "SEB-A.ST"):
        report = pipeline.value_company(
            companies[ticker], universe, params, n_simulations=500, seed=42
        )
        row = csv.loc[ticker]
        price = report.price
        assert row["price"] == price
        assert row["methods_count"] == len(report.valuations)
        assert row["market_cap"] == pytest.approx(report.company.market.market_cap)
        assert row["name"] == report.company.profile.name
        by_method = {result.method: result for result in report.valuations}
        for method in ValuationMethod:
            if method in (ValuationMethod.MONTE_CARLO, ValuationMethod.MARKET):
                continue
            result = by_method.get(method)
            columns = [f"{method.value}_{s}" for s in ("low", "base", "high", "base_to_price")]
            if result is None:
                assert row[columns].isna().all(), (ticker, method)
                continue
            expected = [result.low, result.base, result.high, result.base / price]
            assert row[columns].tolist() == pytest.approx(expected, abs=1e-6), (ticker, method)
        if report.monte_carlo is not None:
            mc = report.monte_carlo
            assert row[["mc_p10", "mc_p50", "mc_p90"]].tolist() == pytest.approx(
                [mc.percentiles[10], mc.percentiles[50], mc.percentiles[90]], abs=1e-6
            )
            assert row["mc_p50_to_price"] == pytest.approx(mc.percentiles[50] / price, abs=1e-6)
            assert row["mc_prob_above_price"] == pytest.approx(mc.prob_above_price, abs=1e-6)
        if report.dcf is not None:
            assert row["wacc"] == pytest.approx(report.dcf.wacc, abs=1e-6)
            assert row["beta"] == pytest.approx(report.dcf.wacc_result.beta, abs=1e-6)
            assert row["exit_multiple"] == pytest.approx(report.dcf.exit_multiple, abs=1e-6)
            assert row["size_premium"] == pytest.approx(report.dcf.wacc_result.size_premium)
            assert row["dcf_peer_source"] == report.dcf_peer_source
        if report.lbo is not None:
            assert row["lbo_irr"] == pytest.approx(report.lbo.irr, abs=1e-6)
            assert row["lbo_moic"] == pytest.approx(report.lbo.moic, abs=1e-6)
            assert row["lbo_entry_multiple"] == pytest.approx(report.lbo.entry_multiple, abs=1e-6)
        assert row["peers_count"] == len(report.comps.peers)
        assert row["warnings_count"] == len(report.warnings)


def test_report_row_is_the_flattened_report(atco_report, universe) -> None:
    snapshot = next(peer for peer in universe if peer.ticker == "ATCO-A.ST")
    row = batch.report_row(atco_report, snapshot)
    assert tuple(row) == batch.CSV_COLUMNS
    assert row["status"] == "ok" and row["methods_count"] == 7
    assert row["dcf_gordon_base"] == atco_report.dcf.value_per_share_gordon
    assert row["lbo_base"] == atco_report.lbo.ability_to_pay[0.20]
    assert row["mc_p50"] == atco_report.monte_carlo.percentiles[50]
    assert row["market_cap_sek"] == snapshot.market_cap == row["market_cap"]
    assert row["errors"] is None and row["na_dcf"] is None
    assert batch.report_row(atco_report)["market_cap_sek"] is None


def test_batch_summary_markdown(batch_result) -> None:
    text = batch_result.summary_path.read_text(encoding="utf-8")
    assert text.startswith("# Universe valuation summary")
    assert "6 companies" in text and "500 simulations per company, seed 42" in text
    assert "offline (fixtures and cache)" in text
    assert "| risk_free_rate | 3.075% |" in text
    assert "| ok | 4 | 67% |" in text and "| no_data | 1 | 17% |" in text
    assert "| DCF, Gordon growth | 4 | 67% | 80% |" in text
    assert "| Comps, P/E | 4 |" in text  # VSURE made a loss
    assert "financial sector" in text
    assert "| NOPE.ST | no_data |" in text
    assert text.isascii()


def test_batch_with_reports_and_limit(tmp_path: Path) -> None:
    import io

    stream = io.StringIO()
    result = batch.run_universe(
        ["SEB-A.ST", "ATCO-A.ST", "ABB.ST"],
        offline=True,
        output_dir=tmp_path,
        n_simulations=200,
        limit=2,
        with_reports=True,
        charts=False,
        stream=stream,
    )
    assert result.frame["ticker"].tolist() == ["SEB-A.ST", "ATCO-A.ST"]
    assert result.csv_path == tmp_path / "universe_valuation_limit2.csv"
    assert result.summary_path == tmp_path / "universe_summary_limit2.md"
    assert result.csv_path.is_file() and result.summary_path.is_file()
    assert "[   2/2]" in stream.getvalue() and "last: ATCO-A.ST" in stream.getvalue()
    for ticker in ("SEB-A.ST", "ATCO-A.ST"):
        folder = tmp_path / ticker
        assert (folder / f"{ticker}_valuation.xlsx").is_file()
        assert (folder / "summary.json").is_file()
        assert not (folder / "charts").exists()
    assert not (tmp_path / "ABB.ST").exists()
    assert result.seconds > 0


def test_batch_survives_a_company_that_crashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = batch.value_company

    def flaky(company, *args, **kwargs):
        if company.ticker == "ABB.ST":
            raise RuntimeError("synthetic crash")
        return real(company, *args, **kwargs)

    monkeypatch.setattr(batch, "value_company", flaky)
    result = batch.run_universe(
        ["ABB.ST", "ATCO-A.ST"],
        offline=True,
        output_dir=tmp_path,
        n_simulations=200,
        progress=False,
    )
    frame = result.frame.set_index("ticker")
    assert frame["status"].to_dict() == {"ABB.ST": "error", "ATCO-A.ST": "ok"}
    assert frame.loc["ABB.ST", "errors"] == "RuntimeError: synthetic crash"
    assert result.failures == {"ABB.ST": "RuntimeError: synthetic crash"}
    assert "unexpected failure" in result.summary_path.read_text(encoding="utf-8")


def test_batch_writes_one_football_field_per_company_with_enough_methods(tmp_path: Path) -> None:
    result = batch.run_universe(
        ["ATCO-A.ST", "SEB-A.ST"],
        offline=True,
        output_dir=tmp_path,
        n_simulations=200,
        progress=False,
        football_fields=True,
        min_methods=4,
    )
    assert len(result.frame) == 2
    charts = sorted(p.name for p in (tmp_path / "football_fields").iterdir())
    # Atlas Copco has seven methods; the bank only has P/E and gets no chart.
    assert charts == ["ATCO-A.ST.png"]
    assert (tmp_path / "football_fields" / "ATCO-A.ST.png").stat().st_size > 10_000
