#!/usr/bin/env python3
"""Seed the publications PanelApp already cites for a gene into the run corpus.

Tournament selection optimises for a minimal, non-redundant evidence set, so it
routinely discards papers a PanelApp curator has already cited — a single-family
report loses to a comprehensive cohort by design. That is the right call for
discovering new evidence, but it makes the report's "new evidence" framing rest
on papers we never read. Seeding runs after the tournament and is unconditional,
so a PanelApp-cited paper enters the corpus whether or not the tournament kept it,
and whether or not it was ever in the screened baseline.
"""

import logging
import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from palit.discover_citations import (
    fetch_paper_metadata_by_doi,
    fetch_papers_by_pmids,
    store_referenced_paper,
)
from palit.hgnc import HgncResolver
from palit.panelapp_client import (
    PanelGeneData,
    collect_panelapp_gene_publications,
    find_gene_panel,
)
from palit.progress import LoggingProgress as Progress

logger = logging.getLogger(__name__)


class PanelPublicationSource(Protocol):
    """The PanelApp lookups seeding needs. `PanelAppClient` satisfies it."""

    def get_panel_data(self, panel_id: int) -> dict[str, Any]: ...

    def get_gene_evaluations(self, panel_id: int, hgnc_id: int) -> list[dict[str, Any]]: ...


# PMIDs per efetch request. Keeps the fetch to a handful of round trips rather
# than one per paper, which matters at ~1.5k publications per run.
EFETCH_BATCH_SIZE = 200

# source_details prefix marking a paper as present because PanelApp cites it.
# Mirrors the "referenced:{hgnc_id}:{citing_doi}" convention used for papers
# discovered from evidence citations.
SOURCE_DETAILS_PREFIX = "panelapp"


@dataclass
class SeedStats:
    """Outcome of a seeding pass."""

    genes_considered: int
    genes_on_target_panel: int
    publications_cited: int
    already_present: int
    added: int
    unresolved: int


def _genes_under_assessment(db_path: Path) -> list[int]:
    """Genes this run is assessing, i.e. those with papers from the recent window."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT DISTINCT hgnc_id
            FROM gene_mentions
            WHERE source = 'recent_evidence'
            ORDER BY hgnc_id
            """
        )
        return [row[0] for row in cursor.fetchall()]


def _existing_identifiers(db_path: Path) -> tuple[set[int], set[str]]:
    """PMIDs and lowercased DOIs already in the corpus.

    DOIs are lowercased because PanelApp publication fields are free text and
    are not case-normalised.
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT pmid FROM papers WHERE pmid IS NOT NULL")
        pmids = {row[0] for row in cursor.fetchall()}
        cursor.execute("SELECT doi FROM papers")
        dois = {row[0].lower() for row in cursor.fetchall()}
    return pmids, dois


@dataclass
class _MissingPublications:
    """PanelApp identifiers absent from the corpus, each owned by one gene.

    A publication cited for several genes is fetched once and attributed to the
    lowest HGNC ID citing it; gene linkage comes from evidence extraction, so
    the owner only determines the recorded provenance.
    """

    pmid_owner: dict[int, int]
    doi_owner: dict[str, int]
    genes_on_target_panel: int
    publications_cited: int
    already_present: int


def collect_missing_publications(
    db_path: Path,
    panelapp_client: PanelPublicationSource,
    panel_data: PanelGeneData,
    hgnc_ids: list[int],
) -> _MissingPublications:
    """Find PanelApp-cited publications for the given genes that we do not hold.

    Uses the same gene-entity + per-review union that `assess-genes` compares
    evidence against, so the seeded set matches the set the skip check consults.
    """
    known_pmids, known_dois = _existing_identifiers(db_path)

    pmid_owner: dict[int, int] = {}
    doi_owner: dict[str, int] = {}
    genes_on_target_panel = 0
    publications_cited = 0
    already_present = 0

    with Progress() as progress:
        task = progress.add_task("Collecting PanelApp publications", total=len(hgnc_ids))
        for hgnc_id in hgnc_ids:
            progress.update(task, advance=1)

            panel_id = find_gene_panel(hgnc_id, panel_data.panel_ids, panel_data)
            if panel_id is None:
                # Novel gene: not on any target panel, so PanelApp cites nothing for it.
                continue
            genes_on_target_panel += 1

            reviews = panelapp_client.get_gene_evaluations(panel_id, hgnc_id)
            pubs = collect_panelapp_gene_publications(
                panelapp_client.get_panel_data(panel_id), hgnc_id, reviews
            )
            publications_cited += len(pubs.pmids) + len(pubs.dois)

            for pmid in sorted(pubs.pmids):
                if pmid in known_pmids:
                    already_present += 1
                elif pmid not in pmid_owner:
                    pmid_owner[pmid] = hgnc_id

            for doi in sorted(pubs.dois):
                if doi.lower() in known_dois:
                    already_present += 1
                elif doi.lower() not in doi_owner:
                    doi_owner[doi.lower()] = hgnc_id

    return _MissingPublications(
        pmid_owner=pmid_owner,
        doi_owner=doi_owner,
        genes_on_target_panel=genes_on_target_panel,
        publications_cited=publications_cited,
        already_present=already_present,
    )


def _store_missing_publications(db_path: Path, missing: _MissingPublications) -> tuple[int, int]:
    """Fetch and store the missing publications. Returns (added, unresolved)."""
    added = 0
    resolved_pmids: set[int] = set()

    pmids = sorted(missing.pmid_owner)
    batches = [pmids[i : i + EFETCH_BATCH_SIZE] for i in range(0, len(pmids), EFETCH_BATCH_SIZE)]

    with Progress() as progress:
        task = progress.add_task("Fetching PanelApp publications", total=len(pmids))
        for batch in batches:
            # One efetch per batch, so provenance is stamped per paper afterwards
            # rather than per request.
            papers = fetch_papers_by_pmids(batch, SOURCE_DETAILS_PREFIX)
            progress.update(task, advance=len(batch))

            for paper in papers:
                # PubMed XML records always carry a PMID; the field is optional
                # only because CrossRef-sourced papers share the dataclass.
                assert paper.pmid is not None
                owner = missing.pmid_owner[paper.pmid]
                resolved_pmids.add(paper.pmid)
                stored = replace(paper, source_details=f"{SOURCE_DETAILS_PREFIX}:{owner}")
                if store_referenced_paper(db_path, stored):
                    added += 1

    unresolved = len(set(pmids) - resolved_pmids)
    for pmid in sorted(set(pmids) - resolved_pmids):
        logger.warning(f"PanelApp PMID {pmid} did not resolve to a DOI; not seeded")

    for doi, owner in sorted(missing.doi_owner.items()):
        doi_paper = fetch_paper_metadata_by_doi(doi, f"{SOURCE_DETAILS_PREFIX}:{owner}")
        if doi_paper is None:
            logger.warning(f"PanelApp DOI {doi} could not be resolved; not seeded")
            unresolved += 1
            continue
        if store_referenced_paper(db_path, doi_paper):
            added += 1

    return added, unresolved


def seed_panelapp_publications(
    db_path: Path,
    panelapp_client: PanelPublicationSource,
    panel_data: PanelGeneData,
    hgnc_resolver: HgncResolver,
) -> SeedStats:
    """Add every PanelApp-cited publication for this run's genes to the corpus.

    Seeded papers are stored as expansion papers scheduled for download, so they
    flow through the existing download and extraction steps unchanged.
    """
    hgnc_ids = _genes_under_assessment(db_path)
    logger.info(f"Seeding PanelApp publications for {len(hgnc_ids)} gene(s)")

    missing = collect_missing_publications(db_path, panelapp_client, panel_data, hgnc_ids)
    to_fetch = len(missing.pmid_owner) + len(missing.doi_owner)
    logger.info(
        f"PanelApp cites {missing.publications_cited} publication(s) for "
        f"{missing.genes_on_target_panel} gene(s) on a target panel: "
        f"{missing.already_present} already held, {to_fetch} to fetch"
    )

    added, unresolved = _store_missing_publications(db_path, missing)

    stats = SeedStats(
        genes_considered=len(hgnc_ids),
        genes_on_target_panel=missing.genes_on_target_panel,
        publications_cited=missing.publications_cited,
        already_present=missing.already_present,
        added=added,
        unresolved=unresolved,
    )
    logger.info(
        f"PanelApp seeding complete: {stats.added} paper(s) added, {stats.unresolved} unresolved"
    )

    _log_gene_coverage(missing, hgnc_resolver)
    return stats


def _log_gene_coverage(missing: _MissingPublications, hgnc_resolver: HgncResolver) -> None:
    """Log which genes needed seeding, so a partial corpus is visible in run logs."""
    by_gene: dict[int, int] = {}
    for owner in missing.pmid_owner.values():
        by_gene[owner] = by_gene.get(owner, 0) + 1
    for owner in missing.doi_owner.values():
        by_gene[owner] = by_gene.get(owner, 0) + 1

    for hgnc_id, count in sorted(by_gene.items(), key=lambda kv: (-kv[1], kv[0])):
        symbol = hgnc_resolver.get_symbol(hgnc_id)
        logger.debug(f"  {symbol} (HGNC:{hgnc_id}): {count} PanelApp publication(s) seeded")
