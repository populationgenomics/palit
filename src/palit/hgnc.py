#!/usr/bin/env python3
"""HGNC gene symbol resolver using the complete HGNC dataset."""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

HGNC_DATA_PATH = Path("data/hgnc_complete_set.json")

# Known LLM misspelling prefixes: wrong prefix → correct prefix.
# A symbol starting with a wrong prefix is corrected before HGNC lookup.
# Add entries as real cases are encountered.
KNOWN_MISSPELLING_PREFIXES: dict[str, str] = {
    "ERRC": "ERCC",
    "SERBIN": "SERPIN",  # SERBINB7 → SERPINB7
    "TBCID": "TBC1D",  # TBCID24 → TBC1D24
    "ZYFVE": "ZFYVE",  # ZYFVE26 → ZFYVE26
}

# Known LLM exact misspellings: wrong symbol → correct symbol.
# Character transpositions, missing characters, O/0 confusion, etc.
KNOWN_EXACT_MISSPELLINGS: dict[str, str] = {
    "VSP33B": "VPS33B",
    "WSF1": "WFS1",
    "PI3KR1": "PIK3R1",
    "GPIBB": "GP1BB",
    "SR5A2": "SRD5A2",
    "PDCH15": "PCDH15",
    "SSPL2C": "SPPL2C",
    "NROB1": "NR0B1",  # letter O vs digit 0
    "SMARCAL": "SMARCAL1",
}


@dataclass(frozen=True)
class HgncEntry:
    """A single HGNC gene entry."""

    hgnc_id: int  # Integer ID (e.g. 8607), "HGNC:" prefix stripped
    symbol: str  # Current approved symbol (e.g. "PRKN")
    prev_symbols: tuple[str, ...]
    alias_symbols: tuple[str, ...]
    locus_group: str
    chromosome: str | None  # Parsed from HGNC `location` (e.g. "17q21.31" → "17")


@dataclass
class HgncResolver:
    """Resolve arbitrary gene symbols to HGNC entries.

    Resolution order:
    1. Exact match on current symbol
    2. Exact match on prev_symbol (unambiguous only)
    3. Exact match on alias_symbol (unambiguous only)
    4. Known misspelling correction, then retry steps 1-3
    5. Unresolved → return None
    """

    _by_symbol: dict[str, HgncEntry] = field(default_factory=dict)
    _by_prev: dict[str, list[HgncEntry]] = field(default_factory=dict)
    _by_alias: dict[str, list[HgncEntry]] = field(default_factory=dict)
    _by_hgnc_id: dict[int, HgncEntry] = field(default_factory=dict)

    @staticmethod
    def from_file(path: Path = HGNC_DATA_PATH) -> "HgncResolver":
        """Load HGNC data from JSON file and build lookup tables."""
        logger.info(f"Loading HGNC data from {path}...")

        with open(path) as f:
            data: dict = json.load(f)

        docs = data["response"]["docs"]

        by_symbol: dict[str, HgncEntry] = {}
        by_prev: dict[str, list[HgncEntry]] = {}
        by_alias: dict[str, list[HgncEntry]] = {}
        by_hgnc_id: dict[int, HgncEntry] = {}

        for doc in docs:
            hgnc_id_str: str = doc["hgnc_id"]  # e.g. "HGNC:8607"
            hgnc_id = int(hgnc_id_str.removeprefix("HGNC:"))
            symbol: str = doc["symbol"]
            prev_symbols = tuple(doc.get("prev_symbol", []))
            alias_symbols = tuple(doc.get("alias_symbol", []))
            locus_group: str = doc.get("locus_group", "")
            chromosome = _parse_chromosome(doc.get("location"))

            entry = HgncEntry(
                hgnc_id=hgnc_id,
                symbol=symbol,
                prev_symbols=prev_symbols,
                alias_symbols=alias_symbols,
                locus_group=locus_group,
                chromosome=chromosome,
            )

            by_symbol[symbol.upper()] = entry
            by_hgnc_id[hgnc_id] = entry

            for prev in prev_symbols:
                by_prev.setdefault(prev.upper(), []).append(entry)

            for alias in alias_symbols:
                by_alias.setdefault(alias.upper(), []).append(entry)

        resolver = HgncResolver(
            _by_symbol=by_symbol,
            _by_prev=by_prev,
            _by_alias=by_alias,
            _by_hgnc_id=by_hgnc_id,
        )
        logger.info(
            f"Loaded {len(by_symbol)} HGNC entries "
            f"({len(by_prev)} prev_symbols, {len(by_alias)} alias_symbols)"
        )
        return resolver

    def get_symbol(self, hgnc_id: int) -> str:
        """Look up current symbol by HGNC ID. Raises KeyError if not found."""
        return self._by_hgnc_id[hgnc_id].symbol

    def get_chromosome(self, hgnc_id: int) -> str | None:
        """Look up gene's chromosome by HGNC ID. None when HGNC has no
        parseable location (e.g. pseudogenes, withdrawn entries).
        Raises KeyError if hgnc_id is unknown."""
        return self._by_hgnc_id[hgnc_id].chromosome

    def resolve(self, symbol: str) -> HgncEntry | None:
        """Resolve arbitrary gene symbol to HGNC entry. Returns None if unresolved."""
        normalized = _normalize_unicode(symbol)
        upper = normalized.upper()

        # Steps 1-3: exact lookups
        entry = self._resolve_exact(upper)
        if entry is not None:
            return entry

        # Step 4: try known exact misspellings, then prefix corrections
        corrected = KNOWN_EXACT_MISSPELLINGS.get(upper)
        if corrected is not None:
            entry = self._resolve_exact(corrected)
            if entry is not None:
                logger.info(f"Resolved '{symbol}' via exact misspelling → '{entry.symbol}'")
                return entry

        corrected_prefix = _apply_known_misspelling_prefixes(upper)
        if corrected_prefix != upper:
            entry = self._resolve_exact(corrected_prefix)
            if entry is not None:
                logger.info(f"Resolved '{symbol}' via prefix misspelling → '{entry.symbol}'")
                return entry

        return None

    def _resolve_exact(self, upper: str) -> HgncEntry | None:
        """Try exact resolution: current symbol, then prev_symbol, then alias."""
        # 1. Exact match on current symbol
        entry = self._by_symbol.get(upper)
        if entry is not None:
            return entry

        # 2. Exact match on prev_symbol (unambiguous only)
        prev_matches = self._by_prev.get(upper)
        if prev_matches is not None:
            if len(prev_matches) == 1:
                return prev_matches[0]
            logger.debug(
                f"Ambiguous prev_symbol '{upper}' matches {len(prev_matches)} entries, skipping"
            )
            return None

        # 3. Exact match on alias_symbol (unambiguous only)
        alias_matches = self._by_alias.get(upper)
        if alias_matches is not None:
            if len(alias_matches) == 1:
                return alias_matches[0]
            logger.debug(
                f"Ambiguous alias '{upper}' matches {len(alias_matches)} entries, skipping"
            )
            return None

        return None


# Unicode characters that LLMs sometimes substitute for ASCII equivalents.
_UNICODE_REPLACEMENTS: dict[str, str] = {
    "\u2011": "-",  # non-breaking hyphen
    "\u2013": "-",  # en dash
    "\u2014": "-",  # em dash
    "\u200b": "",  # zero-width space
    "\u200c": "",  # zero-width non-joiner
    "\u200d": "",  # zero-width joiner
    "\ufeff": "",  # BOM / zero-width no-break space
    ".": "-",  # dots in gene symbols (NKX2.1 → NKX2-1)
}

# Greek letters (alpha, beta, gamma, delta, epsilon) that LLMs sometimes
# use instead of Latin equivalents.
_GREEK_TO_LATIN: dict[str, str] = {
    "\u03b1": "A",
    "\u03b2": "B",
    "\u03b3": "G",
    "\u03b4": "D",
    "\u03b5": "E",
}


_CHROMOSOME_PATTERN = re.compile(r"^(\d+|X|Y)")


def _parse_chromosome(location: str | None) -> str | None:
    """Extract the chromosome token from an HGNC `location` string.

    HGNC encodes cytogenetic location as e.g. "17q21.31", "Xq28", "Yp11.2",
    or "mitochondria"/"mitochondrially encoded" for chrM. Anything that
    doesn't start with a chromosome number/letter (e.g. "reserved",
    "not on reference assembly") returns None.
    """
    if not location:
        return None
    if location.lower().startswith("mitochond"):
        return "MT"
    match = _CHROMOSOME_PATTERN.match(location)
    return match.group(1) if match else None


def _normalize_unicode(symbol: str) -> str:
    """Clean unicode artifacts from LLM-extracted gene symbols."""
    for char, replacement in _UNICODE_REPLACEMENTS.items():
        if char in symbol:
            symbol = symbol.replace(char, replacement)
    for greek, latin in _GREEK_TO_LATIN.items():
        if greek in symbol:
            symbol = symbol.replace(greek, latin)
    return symbol


def _apply_known_misspelling_prefixes(symbol: str) -> str:
    """Apply known misspelling corrections to a symbol."""
    for wrong, correct in KNOWN_MISSPELLING_PREFIXES.items():
        if symbol.startswith(wrong):
            return correct + symbol[len(wrong) :]
    return symbol
