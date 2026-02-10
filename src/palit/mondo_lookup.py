#!/usr/bin/env python3
"""MONDO disease ID candidate lookup using GenCC gene-disease mappings."""

import csv
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import httpx
import pronto

from palit.panelapp_integration import MONDO_CATEGORIES

logger = logging.getLogger(__name__)

GENCC_URL = "https://search.thegencc.org/download/action/submissions-export-tsv"
MONDO_OBO_URL = "https://github.com/monarch-initiative/mondo/releases/latest/download/mondo.obo"

# GenCC classifications to include (exclude Disputed, Refuted, Animal Model Only, etc.)
INCLUDED_CLASSIFICATIONS = {"Definitive", "Strong", "Moderate", "Limited"}


_MAX_AGE_SECONDS = 7 * 24 * 3600  # 1 week


def _download_if_stale(url: str, path: Path) -> None:
    """Download a file if it doesn't exist or is older than 1 week."""
    if path.exists():
        age = time.time() - path.stat().st_mtime
        if age < _MAX_AGE_SECONDS:
            logger.info(f"Using cached {path} (age: {age / 3600:.0f}h)")
            return
        logger.info(f"Re-downloading stale {path} (age: {age / 3600:.0f}h)")
    else:
        logger.info(f"Downloading {url}")

    with httpx.Client(follow_redirects=True, timeout=120) as client:
        response = client.get(url)
        response.raise_for_status()
        path.write_bytes(response.content)
    logger.info(f"Downloaded {len(response.content):,} bytes → {path}")


@dataclass(frozen=True)
class MondoCandidate:
    """A MONDO disease term candidate for a gene."""

    mondo_id: str
    title: str
    definition: str


class MondoLookup:
    """Lookup service for gene → MONDO disease candidates.

    Downloads GenCC gene-disease submissions and MONDO ontology on init,
    caching them in the specified directory. Keyed by HGNC gene symbol.
    """

    def __init__(self, cache_dir: Path = Path("data")) -> None:
        self._gene_to_candidates: dict[str, list[MondoCandidate]] = {}
        self._mondo_labels: dict[str, str] = {}

        gencc_path = cache_dir / "gencc_submissions.tsv"
        mondo_path = cache_dir / "mondo.obo"

        # Download data (re-downloads if older than 7 days)
        _download_if_stale(GENCC_URL, gencc_path)
        _download_if_stale(MONDO_OBO_URL, mondo_path)

        # Load GenCC → gene→MONDO ID mapping
        gencc_entries = self._load_gencc(gencc_path)

        # Load MONDO ontology for definitions and labels
        logger.info("Loading MONDO ontology...")
        mondo = pronto.Ontology(str(mondo_path))
        logger.info(f"Loaded MONDO: {len(mondo):,} terms")

        # Build candidates with definitions from MONDO
        all_mondo_ids: set[str] = set()
        for entries in gencc_entries.values():
            all_mondo_ids.update(entries.keys())

        # Cache labels and definitions for all referenced MONDO terms
        for mondo_id in all_mondo_ids:
            term = mondo.get(mondo_id)
            if term is None:
                continue
            if term.name is not None:
                self._mondo_labels[mondo_id] = term.name

        # Build per-gene candidate lists
        for gene_symbol, disease_map in gencc_entries.items():
            candidates = []
            for mondo_id, title in disease_map.items():
                term = mondo.get(mondo_id)
                definition = term.definition.strip() if term and term.definition else ""
                candidates.append(
                    MondoCandidate(
                        mondo_id=mondo_id,
                        title=title,
                        definition=definition,
                    )
                )
            self._gene_to_candidates[gene_symbol] = candidates

        # Also cache fallback category labels
        for mondo_id, info in MONDO_CATEGORIES.items():
            self._mondo_labels[mondo_id] = info["label"]

        logger.info(
            f"MondoLookup ready: {len(self._gene_to_candidates):,} genes, "
            f"{len(self._mondo_labels):,} MONDO labels cached"
        )

    @staticmethod
    def _load_gencc(path: Path) -> dict[str, dict[str, str]]:
        """Load GenCC submissions, grouped by gene symbol.

        Returns:
            Dict mapping gene_symbol → {mondo_id: disease_title}.
            Deduplicates by (gene, mondo_id), keeping the first occurrence.
        """
        gene_to_diseases: dict[str, dict[str, str]] = defaultdict(dict)

        with path.open() as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if row["classification_title"] not in INCLUDED_CLASSIFICATIONS:
                    continue

                gene_symbol = row["gene_symbol"]
                disease_curie = row["disease_curie"]

                # Keep first occurrence per (gene, disease) — GenCC file is ordered
                # by classification strength within submitter groups
                if disease_curie not in gene_to_diseases[gene_symbol]:
                    gene_to_diseases[gene_symbol][disease_curie] = row["disease_title"]

        logger.info(
            f"Loaded GenCC: {len(gene_to_diseases):,} genes, "
            f"{sum(len(v) for v in gene_to_diseases.values()):,} gene-disease pairs"
        )
        return dict(gene_to_diseases)

    def get_candidates(self, hgnc_symbol: str) -> list[MondoCandidate]:
        """Get MONDO disease candidates for a gene by HGNC symbol.

        Args:
            hgnc_symbol: Current HGNC gene symbol.

        Returns:
            List of MondoCandidate, empty if gene not found in GenCC.
        """
        return self._gene_to_candidates.get(hgnc_symbol, [])

    def get_label(self, mondo_id: str) -> str:
        """Get human-readable label for a MONDO ID.

        Checks the cached MONDO term names first, then falls back to
        MONDO_CATEGORIES for the 4 high-level fallback IDs.

        Args:
            mondo_id: MONDO ID (e.g., "MONDO:0007947")

        Returns:
            Human-readable disease name, or the MONDO ID itself if unknown.
        """
        return self._mondo_labels.get(mondo_id, mondo_id)
