"""On-disk storage shared by the local cache and the test fixtures.

A *store* is a plain directory with human-inspectable JSON/CSV files::

    <root>/
        universe.json            raw screener rows (all Swedish symbols)
        benchmark.json           weekly closes of the benchmark index
        peer_snapshots.csv       derived peer table (one row per company, SEK)
        companies/<TICKER>.json  raw Yahoo payload for one ticker

The fixtures in ``tests/fixtures`` use exactly the same layout, i.e. fixtures
are a frozen (and trimmed) copy of the cache. All loaders build their objects
from these raw files, never directly from live responses, which guarantees
that an offline load equals the online load that produced the files.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

ENV_OFFLINE = "VALUATION_ENGINE_OFFLINE"
ENV_CACHE_DIR = "VALUATION_ENGINE_CACHE_DIR"
ENV_FIXTURES_DIR = "VALUATION_ENGINE_FIXTURES_DIR"

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})

#: Repository root when running from a source checkout (src layout).
_REPO_ROOT = Path(__file__).resolve().parents[3]


def is_offline(offline: bool = False) -> bool:
    """Return True if offline mode is requested by argument or environment."""
    if offline:
        return True
    return os.environ.get(ENV_OFFLINE, "").strip().lower() in _TRUE_VALUES


def cache_dir() -> Path:
    """Directory of the local cache (``VALUATION_ENGINE_CACHE_DIR`` overrides)."""
    override = os.environ.get(ENV_CACHE_DIR, "").strip()
    if override:
        return Path(override).expanduser()
    return _REPO_ROOT / ".cache" / "valuation_engine"


def fixtures_dir() -> Path:
    """Directory of the committed fixtures (``VALUATION_ENGINE_FIXTURES_DIR`` overrides)."""
    override = os.environ.get(ENV_FIXTURES_DIR, "").strip()
    if override:
        return Path(override).expanduser()
    return _REPO_ROOT / "tests" / "fixtures"


def _sanitize(value: Any) -> Any:
    """Make ``value`` strictly JSON-compatible (NaN/inf -> None)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    return value


def normalize(payload: Any) -> Any:
    """Return ``payload`` exactly as it would look after a save/load cycle."""
    return json.loads(json.dumps(_sanitize(payload), allow_nan=False))


class Store:
    """A directory of raw data files (the cache or the fixtures)."""

    def __init__(self, root: Path, name: str) -> None:
        self.root = Path(root)
        self.name = name

    def __repr__(self) -> str:
        return f"Store({self.name!r}, {str(self.root)!r})"

    # -- paths ------------------------------------------------------------

    def company_path(self, ticker: str) -> Path:
        return self.root / "companies" / f"{_safe_name(ticker)}.json"

    @property
    def universe_path(self) -> Path:
        return self.root / "universe.json"

    @property
    def benchmark_path(self) -> Path:
        return self.root / "benchmark.json"

    @property
    def peers_path(self) -> Path:
        return self.root / "peer_snapshots.csv"

    @property
    def fx_path(self) -> Path:
        return self.root / "fx.json"

    # -- JSON documents ---------------------------------------------------

    def read_json(self, path: Path) -> dict[str, Any] | None:
        """Read a JSON document; None if it does not exist or is corrupt."""
        if not path.is_file():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def write_json(self, path: Path, payload: dict[str, Any]) -> None:
        """Atomically write a JSON document (compact but line-oriented)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(_sanitize(payload), handle, ensure_ascii=False, indent=1, allow_nan=False)
            handle.write("\n")
        tmp.replace(path)

    # -- typed accessors --------------------------------------------------

    def read_company(self, ticker: str) -> dict[str, Any] | None:
        return self.read_json(self.company_path(ticker))

    def write_company(self, ticker: str, payload: dict[str, Any]) -> None:
        self.write_json(self.company_path(ticker), payload)

    def has_company(self, ticker: str) -> bool:
        return self.company_path(ticker).is_file()

    def read_universe(self) -> dict[str, Any] | None:
        return self.read_json(self.universe_path)

    def write_universe(self, payload: dict[str, Any]) -> None:
        self.write_json(self.universe_path, payload)

    def read_benchmark(self) -> dict[str, Any] | None:
        return self.read_json(self.benchmark_path)

    def write_benchmark(self, payload: dict[str, Any]) -> None:
        self.write_json(self.benchmark_path, payload)

    def cached_tickers(self) -> list[str]:
        """Tickers that have a raw company file in this store."""
        folder = self.root / "companies"
        if not folder.is_dir():
            return []
        return sorted(p.stem for p in folder.glob("*.json"))


def _safe_name(ticker: str) -> str:
    """File-system safe version of a ticker (tickers are already tame)."""
    cleaned = ticker.strip().upper().replace("/", "_").replace("\\", "_")
    if not cleaned or cleaned.startswith("."):
        raise ValueError(f"invalid ticker: {ticker!r}")
    return cleaned


def cache_store() -> Store:
    """The local, writable cache."""
    return Store(cache_dir(), "cache")


def fixture_store() -> Store:
    """The committed, read-only fixtures."""
    return Store(fixtures_dir(), "fixtures")
