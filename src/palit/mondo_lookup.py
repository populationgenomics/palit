#!/usr/bin/env python3
"""MONDO disease ID candidate lookup using GenCC gene-disease mappings."""

import csv
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx2
import pronto

from palit.panelapp_integration import MONDO_CATEGORIES

logger = logging.getLogger(__name__)

GENCC_URL = "https://search.thegencc.org/download/action/submissions-export-tsv"
MONDO_OBO_URL = "https://github.com/monarch-initiative/mondo/releases/latest/download/mondo.obo"

DisputeStatus = Literal["None", "Disputed", "Refuted"]

# GenCC classification_title values that flag the gene-disease pair as
# disputed/refuted by an expert curation panel. Refuted is the stronger
# negative call (active rejection) vs Disputed (contested) — both gate
# Criteria A/B/C identically downstream.
_REFUTED_TITLES = {"Refuted Evidence"}
_DISPUTED_TITLES = {"Disputed Evidence"}


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

    with httpx2.Client(follow_redirects=True, timeout=120) as client:
        response = client.get(url)
        response.raise_for_status()
        path.write_bytes(response.content)
    logger.info(f"Downloaded {len(response.content):,} bytes → {path}")


@dataclass(frozen=True)
class _RawSubmission:
    """One GenCC submission row, retained while aggregating per (gene, disease)."""

    classification_title: str
    submitter: str
    date: str
    pmids: tuple[int, ...]
    rationale: str


@dataclass(frozen=True)
class DisputeRecord:
    """One curation panel's documented basis for disputing or refuting a
    gene-disease association.

    The aggregate-assessment prompt uses this to constrain overrule logic:
    contributing papers whose PMIDs appear in `pmids` cannot ground an
    overrule, and each concern in `rationale` must be resolved by the new
    evidence at the bar the panel applied.
    """

    submitter: str
    date: str
    pmids: tuple[int, ...]
    rationale: str


@dataclass(frozen=True)
class MondoCandidate:
    """A MONDO disease term candidate for a gene.

    `dispute_status` aggregates GenCC submissions for this (gene, disease)
    pair: "Refuted" if any submission is Refuted Evidence, "Disputed" if any
    is Disputed Evidence (and none Refuted), otherwise "None". When
    `dispute_status` is not "None", `dispute_records` contains every
    submission whose classification matched the winning status (so if
    multiple panels disputed the same pair, each appears).
    """

    mondo_id: str
    title: str
    definition: str
    dispute_status: DisputeStatus
    dispute_records: tuple[DisputeRecord, ...]


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
        mondo = pronto.Ontology(str(mondo_path), encoding="utf-8")
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
            for mondo_id, (title, dispute_status, dispute_records) in disease_map.items():
                term = mondo.get(mondo_id)
                definition = term.definition.strip() if term and term.definition else ""
                candidates.append(
                    MondoCandidate(
                        mondo_id=mondo_id,
                        title=title,
                        definition=definition,
                        dispute_status=dispute_status,
                        dispute_records=dispute_records,
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
    def _load_gencc(
        path: Path,
    ) -> dict[str, dict[str, tuple[str, DisputeStatus, tuple[DisputeRecord, ...]]]]:
        """Load GenCC submissions, grouped by gene symbol.

        Aggregates across submitters: for each (gene, disease) pair, computes
        a single dispute_status from all submissions ("Refuted" wins over
        "Disputed", which wins over "None") and collects every submission
        whose classification matched the winning status into dispute_records.

        Returns:
            Dict mapping gene_symbol → {mondo_id: (disease_title,
            dispute_status, dispute_records)}.
        """
        gene_to_submissions: dict[str, dict[str, list[_RawSubmission]]] = defaultdict(
            lambda: defaultdict(list)
        )
        # gene_symbol → mondo_id → first-seen disease title
        gene_to_titles: dict[str, dict[str, str]] = defaultdict(dict)

        with path.open() as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                gene_symbol = row["gene_symbol"]
                disease_curie = row["disease_curie"]

                pmids_raw = (row.get("submitted_as_pmids") or "").strip()
                pmids = tuple(int(p.strip()) for p in pmids_raw.split(",") if p.strip().isdigit())
                date_raw = (row.get("submitted_as_date") or "").strip()
                date = date_raw.split("T", 1)[0] if date_raw else ""

                gene_to_submissions[gene_symbol][disease_curie].append(
                    _RawSubmission(
                        classification_title=row["classification_title"],
                        submitter=row["submitter_title"],
                        date=date,
                        pmids=pmids,
                        rationale=(row.get("submitted_as_notes") or "").strip(),
                    )
                )
                if disease_curie not in gene_to_titles[gene_symbol]:
                    gene_to_titles[gene_symbol][disease_curie] = row["disease_title"]

        result: dict[str, dict[str, tuple[str, DisputeStatus, tuple[DisputeRecord, ...]]]] = {}
        disputed_pairs = 0
        refuted_pairs = 0
        for gene_symbol, diseases in gene_to_submissions.items():
            per_disease: dict[str, tuple[str, DisputeStatus, tuple[DisputeRecord, ...]]] = {}
            for disease_curie, submissions in diseases.items():
                titles = [s.classification_title for s in submissions]
                winning: set[str]
                if any(t in _REFUTED_TITLES for t in titles):
                    status: DisputeStatus = "Refuted"
                    refuted_pairs += 1
                    winning = _REFUTED_TITLES
                elif any(t in _DISPUTED_TITLES for t in titles):
                    status = "Disputed"
                    disputed_pairs += 1
                    winning = _DISPUTED_TITLES
                else:
                    status = "None"
                    winning = set()

                records: tuple[DisputeRecord, ...] = (
                    tuple(
                        DisputeRecord(
                            submitter=s.submitter,
                            date=s.date,
                            pmids=s.pmids,
                            rationale=s.rationale,
                        )
                        for s in submissions
                        if s.classification_title in winning
                    )
                    if status != "None"
                    else ()
                )

                per_disease[disease_curie] = (
                    gene_to_titles[gene_symbol][disease_curie],
                    status,
                    records,
                )
            result[gene_symbol] = per_disease

        total_pairs = sum(len(v) for v in result.values())
        logger.info(
            f"Loaded GenCC: {len(result):,} genes, "
            f"{total_pairs:,} gene-disease pairs "
            f"({disputed_pairs} disputed, {refuted_pairs} refuted)"
        )
        return result

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
