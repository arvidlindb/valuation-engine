"""Turn the raw Yahoo screener table into a clean company universe.

Rules (documented in ``docs/data_mapping.md``):

1. Keep ``quoteType == "EQUITY"`` only (drops ETFs/ETPs/funds).
2. Drop rows without a company name (``longName``): Yahoo labels warrants,
   mini futures, bull/bear certificates and option-like symbols as EQUITY but
   gives them no ``longName``.
3. Drop subscription rights/paid subscribed units/warrants/trackers by symbol
   pattern (``-TO*``, ``-TR``, ``-UR``, ``-BTU``, ``-BTA``, ``-XBT*``).
4. Drop symbols without a positive price.
5. Group share classes of the same company (same symbol root or same
   ``longName``) and keep ONE primary class: preference and D shares are
   never primary (a company with only such classes listed is dropped); among
   the remaining classes the one with the highest average traded value
   (3-month average volume x price) wins.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

#: Symbol patterns that are not ordinary shares.
_NON_SHARE_PATTERN = re.compile(
    r"-(TO|TR|UR|BTU|BTA|IL|XBT|XBTE)(-?\d+)?(-[A-Z])?$|^(BULL|BEAR|MINI|T-LONG|T-SHORT|LONG|SHORT)-"
)
_NON_SHARE_NAME = re.compile(r"\b(ETN|ETP|ETF|Tracker)\b", re.IGNORECASE)
#: Class tokens of preference-like shares (never the primary class).
_PREFERENCE_TOKENS = frozenset({"PREF", "D"})


@dataclass(frozen=True)
class UniverseEntry:
    """One company of the universe with its chosen primary share class."""

    ticker: str  # primary class
    name: str
    share_classes: tuple[str, ...]  # all listed ordinary/preference classes
    currency: str | None
    reporting_currency: str | None


@dataclass(frozen=True)
class UniverseStats:
    """Counts for the coverage report."""

    screener_symbols: int
    equity_quote_type: int
    ordinary_share_symbols: int
    companies: int
    preference_only_companies: tuple[str, ...]


def _records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    columns = list(payload.get("columns") or [])
    return [dict(zip(columns, row, strict=False)) for row in payload.get("rows") or []]


def _root_and_class(symbol: str) -> tuple[str, tuple[str, ...]]:
    """Split ``ATCO-A.ST`` into ``("ATCO", ("A",))``."""
    base = symbol.rsplit(".", 1)[0] if symbol.upper().endswith(".ST") else symbol
    parts = base.split("-")
    return parts[0], tuple(parts[1:])


def _is_preference(symbol: str) -> bool:
    _, tokens = _root_and_class(symbol)
    return any(token in _PREFERENCE_TOKENS for token in tokens)


def _is_ordinary_candidate(row: dict[str, Any]) -> bool:
    symbol = str(row.get("symbol") or "")
    name = row.get("longName")
    price = row.get("regularMarketPrice")
    if row.get("quoteType") != "EQUITY" or not symbol or not name:
        return False
    if _NON_SHARE_PATTERN.search(symbol.rsplit(".", 1)[0]) or _NON_SHARE_NAME.search(str(name)):
        return False
    return isinstance(price, (int, float)) and price > 0


def _traded_value(row: dict[str, Any]) -> float:
    volume = row.get("averageDailyVolume3Month") or 0.0
    price = row.get("regularMarketPrice") or 0.0
    return float(volume) * float(price)


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def build_universe(payload: dict[str, Any]) -> tuple[list[UniverseEntry], UniverseStats]:
    """Filter and deduplicate the raw screener payload."""
    records = _records(payload)
    equities = [r for r in records if r.get("quoteType") == "EQUITY"]
    candidates = [r for r in equities if _is_ordinary_candidate(r)]

    # Union share classes by symbol root and by company name.
    group_of: dict[str, int] = {}
    groups: list[list[dict[str, Any]]] = []
    for row in sorted(candidates, key=lambda r: str(r["symbol"])):
        root, _ = _root_and_class(str(row["symbol"]))
        keys = (f"root:{root}", f"name:{_normalize_name(str(row['longName']))}")
        existing = sorted({group_of[k] for k in keys if k in group_of})
        if existing:
            target = existing[0]
            for other in existing[1:]:
                groups[target].extend(groups[other])
                groups[other] = []
                for key, index in list(group_of.items()):
                    if index == other:
                        group_of[key] = target
        else:
            target = len(groups)
            groups.append([])
        groups[target].append(row)
        for key in keys:
            group_of[key] = target

    entries: list[UniverseEntry] = []
    preference_only: list[str] = []
    for group in groups:
        if not group:
            continue
        ordinary = [r for r in group if not _is_preference(str(r["symbol"]))]
        if not ordinary:
            preference_only.extend(str(r["symbol"]) for r in group)
            continue
        primary = max(
            ordinary,
            key=lambda r: (
                _traded_value(r),
                1 if r.get("marketCap") else 0,
                # deterministic tie-break: alphabetically first symbol
                tuple(-ord(c) for c in str(r["symbol"])),
            ),
        )
        entries.append(
            UniverseEntry(
                ticker=str(primary["symbol"]),
                name=str(primary["longName"]),
                share_classes=tuple(sorted(str(r["symbol"]) for r in group)),
                currency=primary.get("currency"),
                reporting_currency=primary.get("financialCurrency"),
            )
        )
    entries.sort(key=lambda e: e.ticker)
    stats = UniverseStats(
        screener_symbols=len(records),
        equity_quote_type=len(equities),
        ordinary_share_symbols=len(candidates),
        companies=len(entries),
        preference_only_companies=tuple(sorted(preference_only)),
    )
    return entries, stats
