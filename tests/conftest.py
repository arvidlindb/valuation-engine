"""Suite-wide setup: offline data, empty temporary cache, no network, Agg backend.

* ``VALUATION_ENGINE_OFFLINE=1`` and the fixtures directory are set before the
  package is imported; the cache directory is an empty temporary folder.
* Every attempt to open a network connection is recorded and makes the test that
  caused it fail, even when the code under test swallows the exception.
* Every test runs with a temporary working directory, so a default relative
  output folder can never land in the repository.
"""

from __future__ import annotations

import os
import shutil
import socket
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

_TESTS_DIR = Path(__file__).resolve().parent
_CACHE_DIR = Path(tempfile.mkdtemp(prefix="valuation-engine-test-cache-"))
os.environ["VALUATION_ENGINE_OFFLINE"] = "1"
os.environ["VALUATION_ENGINE_FIXTURES_DIR"] = str(_TESTS_DIR / "fixtures")
os.environ["VALUATION_ENGINE_CACHE_DIR"] = str(_CACHE_DIR)
os.environ["MPLBACKEND"] = "Agg"
sys.path.insert(0, str(_TESTS_DIR))

import pytest  # noqa: E402

from support import FIXTURE_TICKERS  # noqa: E402
from valuation_engine import data, pipeline  # noqa: E402
from valuation_engine.core.types import CompanyData, MarketParameters, PeerSnapshot  # noqa: E402

NETWORK_ATTEMPTS: list[str] = []
#: Switched on only by the ``live_network`` fixture of tests marked ``network``.
_NETWORK = {"allowed": False}


class NetworkAccessError(AssertionError):
    """A test tried to open a network connection."""


def _is_local(address: object) -> bool:
    if _NETWORK["allowed"]:
        return True
    if isinstance(address, (str, bytes)):  # AF_UNIX path
        return True
    if isinstance(address, tuple) and address:
        return str(address[0]) in {"127.0.0.1", "::1", "localhost"}
    return False


@pytest.fixture(scope="session", autouse=True)
def _block_network() -> Iterator[None]:
    patcher = pytest.MonkeyPatch()
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo

    def connect(self: socket.socket, address: object) -> None:
        if _is_local(address):
            return real_connect(self, address)
        NETWORK_ATTEMPTS.append(f"connect {address!r}")
        raise NetworkAccessError(f"network access attempted: connect {address!r}")

    def connect_ex(self: socket.socket, address: object) -> int:
        if _is_local(address):
            return real_connect_ex(self, address)
        NETWORK_ATTEMPTS.append(f"connect_ex {address!r}")
        raise NetworkAccessError(f"network access attempted: connect_ex {address!r}")

    def getaddrinfo(host: object, *args: object, **kwargs: object) -> object:
        if _NETWORK["allowed"] or host in (None, "", "localhost", "127.0.0.1", "::1"):
            return real_getaddrinfo(host, *args, **kwargs)
        NETWORK_ATTEMPTS.append(f"getaddrinfo {host!r}")
        raise NetworkAccessError(f"network access attempted: getaddrinfo {host!r}")

    patcher.setattr(socket.socket, "connect", connect)
    patcher.setattr(socket.socket, "connect_ex", connect_ex)
    patcher.setattr(socket, "getaddrinfo", getaddrinfo)
    yield
    patcher.undo()


@pytest.fixture(autouse=True)
def _isolated(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory):
    """Temporary working directory, and a loud failure on any network attempt."""
    before = len(NETWORK_ATTEMPTS)
    previous = Path.cwd()
    os.chdir(tmp_path_factory.mktemp("cwd"))
    try:
        yield
    finally:
        os.chdir(previous)
    if "network" not in request.keywords:
        attempts = NETWORK_ATTEMPTS[before:]
        assert not attempts, f"the test touched the network: {attempts}"


@pytest.fixture
def live_network(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """Real network and an empty temporary cache, for tests marked ``network`` only
    (deselected by default; run them with ``pytest -m network``)."""
    assert "network" in request.keywords, "live_network is reserved for @pytest.mark.network"
    monkeypatch.setenv("VALUATION_ENGINE_OFFLINE", "0")
    monkeypatch.setenv("VALUATION_ENGINE_CACHE_DIR", str(tmp_path / "cache"))
    _NETWORK["allowed"] = True
    try:
        yield tmp_path / "cache"
    finally:
        _NETWORK["allowed"] = False


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    shutil.rmtree(_CACHE_DIR, ignore_errors=True)


@pytest.fixture(scope="session")
def cache_dir() -> Path:
    """The (empty) temporary cache directory used by the whole session."""
    return _CACHE_DIR


@pytest.fixture(scope="session")
def universe() -> list[PeerSnapshot]:
    return data.load_peer_snapshots(offline=True)


@pytest.fixture(scope="session")
def params() -> MarketParameters:
    return data.load_market_parameters(offline=True)


@pytest.fixture(scope="session")
def companies() -> dict[str, CompanyData]:
    return {ticker: data.load_company(ticker, offline=True) for ticker in FIXTURE_TICKERS}


@pytest.fixture(scope="session")
def atco(companies: dict[str, CompanyData]) -> CompanyData:
    return companies["ATCO-A.ST"]


@pytest.fixture(scope="session")
def reports(
    companies: dict[str, CompanyData], universe: list[PeerSnapshot], params: MarketParameters
) -> dict[str, pipeline.ValuationReport]:
    """One pipeline report per fixture company (2,000 simulations, seed 42)."""
    return {
        ticker: pipeline.value_company(company, universe, params, n_simulations=2_000, seed=42)
        for ticker, company in companies.items()
    }


@pytest.fixture(scope="session")
def atco_report() -> pipeline.ValuationReport:
    """The documented entry point with its defaults (10,000 simulations)."""
    return pipeline.run_valuation("ATCO-A.ST", offline=True)


@pytest.fixture(scope="session")
def seb_report() -> pipeline.ValuationReport:
    return pipeline.run_valuation("SEB-A.ST", offline=True)
