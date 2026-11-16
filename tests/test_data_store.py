"""Data layer plumbing: the peer table, the on-disk stores, the resolution order
(fixtures / cache / network) with a faked network, and the universe filter."""

from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import fields
from pathlib import Path

import pytest

from support import FIXTURE_TICKERS, FIXTURES
from valuation_engine import data
from valuation_engine.core.types import PeerSnapshot
from valuation_engine.data import download, loader, store, universe, yahoo
from valuation_engine.data.errors import FetchError

NUMERIC_FIELDS = [
    f.name for f in fields(PeerSnapshot) if f.name not in ("ticker", "name", "sector", "industry")
]


def fixture_payload(ticker: str) -> dict:
    return json.loads((FIXTURES / "companies" / f"{ticker}.json").read_text(encoding="utf-8"))


# ----------------------------------------------------------------- peer table


@pytest.mark.parametrize("ticker", FIXTURE_TICKERS)
def test_committed_peer_table_equals_the_row_rebuilt_from_the_company_file(
    ticker: str, universe
) -> None:
    """``peer_snapshots.csv`` is derived data: it must agree with the loaders."""
    committed = next(peer for peer in universe if peer.ticker == ticker)
    bundle = loader.load_company_bundle(ticker, offline=True)
    rebuilt, extras = loader.snapshot_from_bundle(bundle)
    assert bundle.origin == "fixtures"
    assert (rebuilt.name, rebuilt.sector, rebuilt.industry) == (
        committed.name,
        committed.sector,
        committed.industry,
    )
    for name in NUMERIC_FIELDS:
        old, new = getattr(committed, name), getattr(rebuilt, name)
        if old is None or new is None:
            assert old is None and new is None, name
        else:
            assert new == pytest.approx(old, rel=1e-12, abs=1e-12), name
    assert extras["currency"] == bundle.company.profile.currency
    assert extras["fiscal_year"] == bundle.company.statements.latest_year
    assert extras["beta_source"] == "regression vs ^OMXSPI"


def test_peer_table_csv_round_trip_is_lossless(tmp_path: Path, universe) -> None:
    target = store.Store(tmp_path, "tmp")
    rows = [(peer, {}) for peer in reversed(universe[:50])]
    assert loader.write_peer_table(target, rows) == 50
    assert loader.read_peer_table(target) == universe[:50]  # sorted by ticker, bit for bit
    assert loader.read_peer_table(store.Store(tmp_path / "missing", "x")) is None
    with target.peers_path.open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    assert header[: len(fields(PeerSnapshot))] == [f.name for f in fields(PeerSnapshot)]
    assert header[-8:] == list(loader.PEER_EXTRA_COLUMNS)


def test_peer_snapshots_are_built_from_company_files_when_no_table_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, universe
) -> None:
    fixtures = tmp_path / "fixtures"
    (fixtures / "companies").mkdir(parents=True)
    for ticker in ("ATCO-A.ST", "SEB-A.ST", "ABB.ST"):
        shutil.copy(FIXTURES / "companies" / f"{ticker}.json", fixtures / "companies")
    for name in ("universe.json", "benchmark.json", "riksbank.json"):
        shutil.copy(FIXTURES / name, fixtures / name)
    monkeypatch.setenv("VALUATION_ENGINE_FIXTURES_DIR", str(fixtures))
    monkeypatch.setenv("VALUATION_ENGINE_CACHE_DIR", str(tmp_path / "cache"))

    built = data.load_peer_snapshots(offline=True)
    assert [peer.ticker for peer in built] == ["ABB.ST", "ATCO-A.ST", "SEB-A.ST"]
    by_ticker = {peer.ticker: peer for peer in universe}
    for peer in built:
        assert peer.market_cap == pytest.approx(by_ticker[peer.ticker].market_cap, rel=1e-12)
        assert peer.beta == pytest.approx(by_ticker[peer.ticker].beta, rel=1e-12)
    rows, failures = loader.build_peer_snapshots(["ATCO-A.ST", "NOPE.ST"], offline=True)
    assert [snapshot.ticker for snapshot, _ in rows] == ["ATCO-A.ST"]
    assert set(failures) == {"NOPE.ST"} and "DataUnavailableError" in failures["NOPE.ST"]

    (fixtures / "universe.json").unlink()
    shutil.rmtree(fixtures / "companies")
    with pytest.raises(data.DataUnavailableError, match="peer snapshots: not available offline"):
        data.load_peer_snapshots(offline=True)
    with pytest.raises(data.DataUnavailableError, match="universe: not available offline"):
        data.list_universe(offline=True)


def test_rebuild_and_freeze_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, universe):
    """The maintenance commands: cache -> peer table -> frozen fixtures -> same company."""
    cache, frozen = tmp_path / "cache", tmp_path / "frozen"
    (cache / "companies").mkdir(parents=True)
    tickers = ["ATCO-A.ST", "HEXA-B.ST", "VSURE.ST"]
    for ticker in tickers:
        shutil.copy(FIXTURES / "companies" / f"{ticker}.json", cache / "companies")
    for name in ("universe.json", "benchmark.json", "riksbank.json"):
        shutil.copy(FIXTURES / name, cache / name)
    monkeypatch.setenv("VALUATION_ENGINE_CACHE_DIR", str(cache))

    summary = download.rebuild_peer_table([*tickers, "SAND.ST"])
    assert summary["loaded"] == 3 and summary["not_downloaded"] == ["SAND.ST"]
    assert summary["non_sek_reporting"] == 2 and summary["non_sek_trading"] == 1
    assert summary["fx_riksbank_per_year"] == 1 and summary["fx_spot_fallback"] == 0
    assert json.loads((cache / "download_summary.json").read_text())["loaded"] == 3
    rebuilt = loader.read_peer_table(store.Store(cache, "cache"))
    by_ticker = {peer.ticker: peer for peer in universe}
    assert rebuilt == [by_ticker[t] for t in sorted(tickers)]  # identical to the committed rows

    monkeypatch.setenv("VALUATION_ENGINE_FIXTURES_DIR", str(frozen))
    assert download.main(["--freeze", "atco-a.st", "VSURE.ST"]) == 0
    assert sorted(p.name for p in (frozen / "companies").iterdir()) == [
        "ATCO-A.ST.json",
        "VSURE.ST.json",
    ]
    assert {p.name for p in frozen.iterdir()} == {
        "companies",
        "universe.json",
        "benchmark.json",
        "peer_snapshots.csv",
        "riksbank.json",
    }
    monkeypatch.setenv("VALUATION_ENGINE_CACHE_DIR", str(tmp_path / "empty"))
    thawed = data.load_company("ATCO-A.ST", offline=True)
    assert (
        thawed.market
        == data.build_company(
            fixture_payload("ATCO-A.ST"), json.loads((FIXTURES / "benchmark.json").read_text())
        ).market
    )
    with pytest.raises(data.DataUnavailableError, match="cannot freeze"):
        download.freeze_fixtures(["ATCO-A.ST"])  # the cache is empty now


# ------------------------------------------------------------------------ store


def test_store_reads_writes_and_sanitizes(tmp_path: Path) -> None:
    target = store.Store(tmp_path, "tmp")
    assert target.read_company("X.ST") is None and not target.has_company("X.ST")
    assert target.cached_tickers() == []
    target.write_company("x.st", {"ticker": "X.ST", "value": math.nan, "nested": [1.0, math.inf]})
    assert target.has_company("X.ST") and target.cached_tickers() == ["X.ST"]
    assert target.read_company("X.ST") == {"ticker": "X.ST", "value": None, "nested": [1.0, None]}
    assert not list(tmp_path.rglob("*.tmp"))  # written atomically
    target.company_path("BAD.ST").write_text("{not json", encoding="utf-8")
    assert target.read_company("BAD.ST") is None
    target.company_path("LIST.ST").write_text("[1, 2]", encoding="utf-8")
    assert target.read_company("LIST.ST") is None
    assert store.normalize({"a": (1, math.nan), 2: "x"}) == {"a": [1, None], "2": "x"}
    assert target.company_path("a/b.st").name == "A_B.ST.json"
    for bad in ("", "  ", ".hidden"):
        with pytest.raises(ValueError, match="invalid ticker"):
            target.company_path(bad)


def test_directories_follow_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VALUATION_ENGINE_CACHE_DIR", str(tmp_path / "c"))
    monkeypatch.setenv("VALUATION_ENGINE_FIXTURES_DIR", str(tmp_path / "f"))
    assert data.cache_dir() == tmp_path / "c" and data.fixtures_dir() == tmp_path / "f"
    monkeypatch.delenv("VALUATION_ENGINE_CACHE_DIR")
    monkeypatch.delenv("VALUATION_ENGINE_FIXTURES_DIR")
    assert data.cache_dir().parts[-2:] == (".cache", "valuation_engine")
    assert data.fixtures_dir() == FIXTURES


# ------------------------------------------------- resolution order, fake network


@pytest.fixture
def online(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Online mode with an empty temporary cache and a recording fake of Yahoo."""
    calls: dict = {"company": [], "benchmark": 0, "screen": 0, "fail": False}

    def fetch_company(ticker: str, **kwargs: object) -> dict:
        calls["company"].append(ticker)
        if calls["fail"]:
            raise FetchError(f"{ticker}: simulated outage")
        return fixture_payload("ATCO-A.ST") | {"ticker": ticker}

    def fetch_benchmark() -> dict:
        calls["benchmark"] += 1
        if calls["fail"]:
            raise FetchError("benchmark: simulated outage")
        return json.loads((FIXTURES / "benchmark.json").read_text(encoding="utf-8"))

    def fetch_screen(*args: object, **kwargs: object) -> dict:
        calls["screen"] += 1
        if calls["fail"]:
            raise FetchError("screen: simulated outage")
        return json.loads((FIXTURES / "universe.json").read_text(encoding="utf-8"))

    monkeypatch.setenv("VALUATION_ENGINE_OFFLINE", "0")
    monkeypatch.setenv("VALUATION_ENGINE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(yahoo, "fetch_company", fetch_company)
    monkeypatch.setattr(yahoo, "fetch_benchmark", fetch_benchmark)
    monkeypatch.setattr(yahoo, "fetch_screen", fetch_screen)
    calls["cache"] = tmp_path / "cache"
    return calls


def test_online_load_goes_cache_then_network_and_fills_the_cache(online: dict) -> None:
    assert data.is_offline() is False
    first = loader.load_company_bundle("FAKE.ST")
    assert first.origin == "network" and online["company"] == ["FAKE.ST"]
    assert (online["cache"] / "companies" / "FAKE.ST.json").is_file()
    assert (online["cache"] / "benchmark.json").is_file() and online["benchmark"] == 1
    assert first.company.market.price == 192.25 and first.company.price_history is not None

    second = loader.load_company_bundle("FAKE.ST")
    assert second.origin == "cache" and online["company"] == ["FAKE.ST"]  # no second download
    assert second.company.market == first.company.market
    third = loader.load_company_bundle("FAKE.ST", refresh=True)
    assert third.origin == "network" and online["company"] == ["FAKE.ST", "FAKE.ST"]
    # the explicit offline switch reads the cache that was just filled
    assert loader.load_company_bundle("FAKE.ST", offline=True).origin == "cache"
    assert len(online["company"]) == 2


def test_network_failure_falls_back_to_local_copies(online: dict) -> None:
    online["fail"] = True
    bundle = loader.load_company_bundle("ATCO-A.ST")  # not cached, but in the fixtures
    assert bundle.origin == "fixtures" and online["company"] == ["ATCO-A.ST"]
    assert bundle.company.market.price == 192.25
    with pytest.raises(data.DataUnavailableError, match="network failed and no local copy"):
        data.load_company("FAKE.ST")
    assert len(data.list_universe()) == 687 and online["screen"] == 1  # fixtures again


def test_online_universe_is_cached(online: dict) -> None:
    assert len(data.list_universe()) == 687
    assert len(data.list_universe()) == 687
    assert online["screen"] == 1
    assert (online["cache"] / "universe.json").is_file()
    data.list_universe(refresh=True)
    assert online["screen"] == 2


def test_download_command_fills_the_cache_and_logs_failures(online: dict, capsys) -> None:
    real = yahoo.fetch_company

    def flaky(ticker: str, **kwargs: object) -> dict:
        if ticker == "BROKEN.ST":
            raise FetchError("simulated outage")
        return real(ticker, **kwargs)

    yahoo.fetch_company = flaky  # the fixture's monkeypatch restores the attribute
    failures = download.download_tickers(["GOOD.ST", "BROKEN.ST", "GOOD.ST"])
    assert set(failures) == {"BROKEN.ST"} and "FetchError" in failures["BROKEN.ST"]
    assert online["company"] == ["GOOD.ST"]  # the second GOOD.ST was already cached
    log = (online["cache"] / "download_failures.csv").read_text(encoding="utf-8").splitlines()
    assert log[0] == "timestamp,ticker,stage,error,message"
    assert len(log) == 2 and ",BROKEN.ST,download,FetchError,simulated outage" in log[1]

    assert download.main(["good.st", "other.st"]) == 0
    out = capsys.readouterr().out
    assert "screener symbols: 984" in out and "companies: 687" in out
    assert "download failures this run: 0" in out
    assert (online["cache"] / "companies" / "OTHER.ST.json").is_file()
    with pytest.raises(SystemExit) as error:
        download.main([])
    assert error.value.code == 2
    capsys.readouterr()


# ---------------------------------------------------------------------- universe


def screener(*rows: dict) -> dict:
    columns = [
        "symbol", "quoteType", "longName", "currency", "financialCurrency",
        "regularMarketPrice", "marketCap", "averageDailyVolume3Month",
    ]  # fmt: skip
    defaults = {
        "quoteType": "EQUITY", "currency": "SEK", "financialCurrency": "SEK",
        "regularMarketPrice": 10.0, "marketCap": 1e9, "averageDailyVolume3Month": 1000.0,
    }  # fmt: skip
    return {"columns": columns, "rows": [[(defaults | r).get(c) for c in columns] for r in rows]}


def test_universe_filter_and_share_class_selection() -> None:
    payload = screener(
        {"symbol": "AAA-A.ST", "longName": "Alpha AB", "averageDailyVolume3Month": 100.0},
        {"symbol": "AAA-B.ST", "longName": "Alpha AB", "averageDailyVolume3Month": 5000.0},
        {"symbol": "AAA-PREF.ST", "longName": "Alpha AB", "averageDailyVolume3Month": 9e9},
        {"symbol": "BBB.ST", "longName": "Beta AB"},
        {"symbol": "BETA2.ST", "longName": "Beta AB", "averageDailyVolume3Month": 1.0},  # same name
        {"symbol": "CCC-PREF.ST", "longName": "Gamma AB"},  # only the preference share is listed
        {"symbol": "DDD-D.ST", "longName": "Delta AB"},  # only D shares
        {"symbol": "ETF.ST", "longName": "Some ETF", "quoteType": "ETF"},
        {"symbol": "NONAME.ST", "longName": None},  # warrants and certificates have no name
        {"symbol": "ZERO.ST", "longName": "Zero AB", "regularMarketPrice": 0.0},
        {"symbol": "EEE-TO1.ST", "longName": "Epsilon AB"},  # a warrant by symbol pattern
        {"symbol": "EEE.ST", "longName": "Epsilon AB", "financialCurrency": "EUR"},
    )
    entries, stats = universe.build_universe(payload)
    assert [entry.ticker for entry in entries] == ["AAA-B.ST", "BBB.ST", "EEE.ST"]
    alpha, beta, epsilon = entries
    assert alpha.share_classes == ("AAA-A.ST", "AAA-B.ST", "AAA-PREF.ST")
    assert beta.share_classes == ("BBB.ST", "BETA2.ST")  # merged by company name
    assert (epsilon.currency, epsilon.reporting_currency) == ("SEK", "EUR")
    assert stats.screener_symbols == 12 and stats.equity_quote_type == 11
    assert stats.companies == 3
    assert stats.preference_only_companies == ("CCC-PREF.ST", "DDD-D.ST")
    assert universe.build_universe({"columns": [], "rows": []})[0] == []


# ------------------------------------------------- documented coverage figures


def test_peer_table_matches_the_coverage_figures_in_the_documentation(universe) -> None:
    """docs/data_mapping.md, section 9 (run of 2026-11-16)."""
    import numpy as np

    non_financial = [peer for peer in universe if not peer.is_financial]
    assert (len(universe), len(non_financial)) == (680, 577)
    assert sum(peer.ebitda is None for peer in universe) == 26
    assert sum(peer.ebitda is not None and peer.ebitda <= 0 for peer in universe) == 208
    positive = [p for p in non_financial if p.ebitda is not None and p.ebitda > 0]
    assert len(positive) == 387
    multiples = [p.enterprise_value / p.ebitda for p in positive]
    assert np.median(multiples) == pytest.approx(10.5, abs=0.05)
    assert np.percentile(multiples, [10, 90]) == pytest.approx([4.8, 30.5], abs=0.05)
    betas = [peer.beta for peer in universe if peer.beta is not None]
    assert len(universe) - len(betas) == 17
    assert sum(beta < 0 or beta > 3 for beta in betas) == 21
    assert np.median(betas) == pytest.approx(0.63, abs=0.005)
    assert sum(peer.enterprise_value < 0 for peer in universe) == 6
    assert sum(peer.market_cap < 100e6 for peer in universe) == 119
    sectors: dict[str, int] = {}
    for peer in universe:
        sectors[peer.sector] = sectors.get(peer.sector, 0) + 1
    assert sectors == {
        "Industrials": 151,
        "Healthcare": 126,
        "Technology": 121,
        "Consumer Cyclical": 68,
        "Financial Services": 53,
        "Real Estate": 50,
        "Communication Services": 42,
        "Basic Materials": 34,
        "Consumer Defensive": 25,
        "Utilities": 6,
        "Energy": 4,
    }
    with (FIXTURES / "peer_snapshots.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert sum(row["reporting_currency"] != "SEK" for row in rows) == 63
    assert [row["ticker"] for row in rows if row["currency"] != "SEK"] == ["VSURE.ST"]
    assert sum(row["fx_source"].startswith("Riksbank") for row in rows) == 62
    assert sorted(row["ticker"] for row in rows if int(row["fiscal_year"]) < 2025) == [
        "NOTEK.ST",
        "OCTV-SDB.ST",
        "PADEL.ST",
        "SONE.ST",
    ]
