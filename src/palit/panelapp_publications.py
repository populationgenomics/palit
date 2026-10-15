#!/usr/bin/env python3
"""Add the publications PanelApp cites to a database: seeding a run's genes, and
taking every citation of a snapshot into the baseline.

Tournament selection optimises for a minimal, non-redundant evidence set, so it
routinely discards papers a PanelApp curator has already cited — a single-family
report loses to a comprehensive cohort by design. That is the right call for
discovering new evidence, but it makes the report's "new evidence" framing rest
on papers we never read. Seeding runs after the tournament and is unconditional,
so a PanelApp-cited paper enters the corpus whether or not the tournament kept it,
and whether or not it was ever in the screened baseline.

The baseline (``data/pubmed_baseline_screening.sqlite``) takes in every
publication PanelApp Australia cites in a snapshot, on any panel, so that a
panel-scoped run, which copies the baseline's papers, starts from PanelApp's
citations as well as the screened literature. The baseline holds these papers'
metadata only; which genes cite them is a run's concern.
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
    PanelPublications,
    clean_panel_publications,
    collect_panelapp_gene_publications,
    find_gene_panel,
)
from palit.papers import Paper
from palit.progress import LoggingProgress as Progress
from palit.run_corpus import run_genes

logger = logging.getLogger(__name__)


class PanelPublicationSource(Protocol):
    """The PanelApp lookups seeding needs. `PanelAppClient` satisfies it."""

    def get_panel_data(self, panel_id: int) -> dict[str, Any]: ...

    def holds_gene(self, panel_id: int, hgnc_id: int) -> bool: ...

    def gene_reviews(self, panel_id: int, hgnc_id: int) -> list[dict[str, Any]] | None: ...


# PMIDs per efetch request. Keeps the fetch to a handful of round trips rather
# than one per paper, which matters at ~1.5k publications per run.
EFETCH_BATCH_SIZE = 200

# source_details prefix of a paper seeded because PanelApp cites it for a run's
# gene: "panelapp:{hgnc_id}". Mirrors the "referenced:{hgnc_id}:{citing_doi}"
# convention used for papers discovered from evidence citations.
SEED_SOURCE_DETAILS_PREFIX = "panelapp"

# source_type of a seeded paper: an expansion paper, which is in the run's corpus
# once downloaded (see run_corpus.py).
SEED_SOURCE_TYPE = "expansion"

# source_details prefix of a paper the baseline took in because a snapshot cites
# it: "panelapp-snapshot:{panel_date}".
CITATION_SOURCE_DETAILS_PREFIX = "panelapp-snapshot"

# source_type of a paper the baseline took in because PanelApp cites it. A
# baseline paper's source_type names how it entered the baseline, such as
# 'baseline_screening' for the classifier's screen of the PubMed baseline. A
# panel-scoped run copies every baseline paper as 'initial', whatever its
# source_type here.
CITATION_SOURCE_TYPE = "panelapp_citation"


@dataclass
class SeedStats:
    """Outcome of a seeding pass."""

    genes_considered: int
    genes_on_target_panel: int
    publications_cited: int
    already_present: int
    added: int
    already_held: int
    pmids_linked: int
    unresolved: int


@dataclass
class HeldIdentifiers:
    """The PMIDs and DOIs a database holds.

    DOIs are keyed lowercased, because PanelApp publication fields are free text
    and are not case-normalised, and map to the DOI as the database stores it.
    """

    pmids: set[int]
    dois: dict[str, str]


def held_identifiers(db_path: Path) -> HeldIdentifiers:
    """The PMIDs and DOIs of the papers in *db_path*."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT pmid FROM papers WHERE pmid IS NOT NULL")
        pmids = {row[0] for row in cursor.fetchall()}
        cursor.execute("SELECT doi FROM papers")
        dois = {row[0].lower(): row[0] for row in cursor.fetchall()}
    return HeldIdentifiers(pmids=pmids, dois=dois)


def _by_lowercase(dois: set[str]) -> dict[str, str]:
    """Distinct DOIs keyed lowercased, each mapped to one of its cited spellings: the
    first in sort order, so the spelling stored does not depend on set order."""
    spellings: dict[str, str] = {}
    for doi in sorted(dois):
        spellings.setdefault(doi.lower(), doi)
    return spellings


@dataclass
class _MissingPublications:
    """PanelApp identifiers absent from the corpus, each owned by one gene.

    A publication cited for several genes is fetched once and attributed to the
    lowest HGNC ID citing it; gene linkage comes from evidence extraction, so
    the owner only determines the recorded provenance. `doi_owner` holds one
    spelling of each DOI, as PanelApp cites it: the first met, in gene order and
    then sort order.
    """

    pmid_owner: dict[int, int]
    doi_owner: dict[str, int]
    genes_on_target_panel: int
    publications_cited: int
    already_present: int


def collect_missing_publications(
    held: HeldIdentifiers,
    panelapp_client: PanelPublicationSource,
    panel_data: PanelGeneData,
    hgnc_ids: list[int],
) -> _MissingPublications:
    """Find PanelApp-cited publications for the given genes that are not *held*.

    Uses the same gene-entity + per-review union that `assess-genes` compares
    evidence against, so the seeded set matches the set the skip check consults.
    """
    pmid_owner: dict[int, int] = {}
    doi_owner: dict[str, int] = {}
    owned_dois: set[str] = set()  # doi_owner's DOIs, lowercased
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

            # A gene held only as a repeat expansion has no reviews in the snapshot, and
            # those PanelApp could not serve are unknown; the entry's own publications
            # count either way.
            reviews = (
                panelapp_client.gene_reviews(panel_id, hgnc_id)
                if panelapp_client.holds_gene(panel_id, hgnc_id)
                else None
            )
            pubs = collect_panelapp_gene_publications(
                panelapp_client.get_panel_data(panel_id), hgnc_id, reviews or []
            )
            publications_cited += len(pubs.pmids) + len(pubs.dois)

            for pmid in sorted(pubs.pmids):
                if pmid in held.pmids:
                    already_present += 1
                elif pmid not in pmid_owner:
                    pmid_owner[pmid] = hgnc_id

            for doi in sorted(pubs.dois):
                if doi.lower() in held.dois:
                    already_present += 1
                elif doi.lower() not in owned_dois:
                    owned_dois.add(doi.lower())
                    doi_owner[doi] = hgnc_id

    return _MissingPublications(
        pmid_owner=pmid_owner,
        doi_owner=doi_owner,
        genes_on_target_panel=genes_on_target_panel,
        publications_cited=publications_cited,
        already_present=already_present,
    )


@dataclass
class StoreOutcome:
    """Outcome of fetching and storing publications."""

    added: int
    # Resolved to a paper whose DOI the database already holds: a paper cited by
    # both its PMID and its DOI, or held under another identifier or DOI case.
    already_held: int
    # Of `already_held`, papers resolved from a PMID whose held row lacked a PMID
    # and now has it, so a rerun matches them by PMID without fetching them again.
    pmids_linked: int
    unresolved: int


def _link_pmid(db_path: Path, doi: str, pmid: int) -> bool:
    """Give the paper stored under *doi* the PMID *pmid* if it has none."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            "UPDATE papers SET pmid = ? WHERE doi = ? AND pmid IS NULL", (pmid, doi)
        )
        return cursor.rowcount > 0


def store_publications(
    db_path: Path,
    held_dois: dict[str, str],
    pmid_source_details: dict[int, str],
    doi_source_details: dict[str, str],
    source_type: str,
) -> StoreOutcome:
    """Fetch the given publications and store them with their `source_details`.

    PMIDs are fetched from PubMed in batches, then DOIs from CrossRef one at a
    time; a DOI is fetched and stored as spelled in *doi_source_details*. DOIs are
    matched case-insensitively against *held_dois* (see `HeldIdentifiers.dois`)
    and those stored earlier in the call, so a paper cited by both its PMID and
    its DOI is stored once. A paper a PMID resolves to that is held without a
    PMID gets it. Raises `PubmedFetchError` when PubMed cannot be reached; the
    papers stored until then stay stored.
    """
    added = 0
    already_held = 0
    pmids_linked = 0
    held_dois = dict(held_dois)
    resolved_pmids: set[int] = set()

    def store(paper: Paper, source_details: str) -> None:
        nonlocal added, already_held, pmids_linked
        held_doi = held_dois.get(paper.doi.lower())
        if held_doi is not None:
            already_held += 1
            if paper.pmid is not None and _link_pmid(db_path, held_doi, paper.pmid):
                pmids_linked += 1
            return
        held_dois[paper.doi.lower()] = paper.doi
        stored = replace(paper, source_type=source_type, source_details=source_details)
        inserted = store_referenced_paper(db_path, stored)
        # held_dois covers every DOI in the database, so the insert cannot conflict.
        assert inserted
        added += 1

    pmids = sorted(pmid_source_details)
    batches = [pmids[i : i + EFETCH_BATCH_SIZE] for i in range(0, len(pmids), EFETCH_BATCH_SIZE)]
    dois = sorted(doi_source_details)

    with Progress() as progress:
        task = progress.add_task("Fetching PanelApp PMIDs", total=len(pmids))
        for batch in batches:
            # One efetch per batch, so provenance is stamped per paper afterwards
            # rather than per request.
            papers = fetch_papers_by_pmids(batch, source_details="PanelApp citations")
            progress.update(task, advance=len(batch))

            for paper in papers:
                # PubMed XML records always carry a PMID; the field is optional
                # only because CrossRef-sourced papers share the dataclass.
                assert paper.pmid is not None
                resolved_pmids.add(paper.pmid)
                store(paper, pmid_source_details[paper.pmid])

        task = progress.add_task("Fetching PanelApp DOIs", total=len(dois))
        unresolved_dois: list[str] = []
        for doi in dois:
            progress.update(task, advance=1)
            if doi.lower() in held_dois:
                already_held += 1
                continue
            doi_paper = fetch_paper_metadata_by_doi(doi, doi_source_details[doi])
            if doi_paper is None:
                unresolved_dois.append(doi)
                continue
            store(doi_paper, doi_source_details[doi])

    # A PMID PubMed gives no usable record for (no DOI, or a record the extractor
    # skips) is routine for older papers and junk in PanelApp's free text, so the
    # identifiers are listed at DEBUG only. A DOI CrossRef could not resolve was
    # logged with the reason as it failed.
    unresolved_pmids = sorted(set(pmids) - resolved_pmids)
    if unresolved_pmids:
        logger.info(f"PubMed gave no usable record for {len(unresolved_pmids)} PanelApp PMID(s)")
        logger.debug(f"PMIDs without a usable record: {', '.join(map(str, unresolved_pmids))}")
    if pmids_linked:
        logger.info(f"{pmids_linked} held paper(s) given the PMID PanelApp cites them by")
    if unresolved_dois:
        logger.info(f"{len(unresolved_dois)} PanelApp DOI(s) could not be resolved")
        logger.debug(f"Unresolved DOIs: {', '.join(unresolved_dois)}")

    return StoreOutcome(
        added=added,
        already_held=already_held,
        pmids_linked=pmids_linked,
        unresolved=len(unresolved_pmids) + len(unresolved_dois),
    )


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
    with sqlite3.connect(db_path) as conn:
        hgnc_ids = run_genes(conn)
    logger.info(f"Seeding PanelApp publications for {len(hgnc_ids)} gene(s)")

    held = held_identifiers(db_path)
    missing = collect_missing_publications(held, panelapp_client, panel_data, hgnc_ids)
    to_fetch = len(missing.pmid_owner) + len(missing.doi_owner)
    logger.info(
        f"PanelApp cites {missing.publications_cited} publication(s) for "
        f"{missing.genes_on_target_panel} gene(s) on a target panel: "
        f"{missing.already_present} already held, {to_fetch} to fetch"
    )

    outcome = store_publications(
        db_path,
        held.dois,
        {
            pmid: f"{SEED_SOURCE_DETAILS_PREFIX}:{owner}"
            for pmid, owner in missing.pmid_owner.items()
        },
        {doi: f"{SEED_SOURCE_DETAILS_PREFIX}:{owner}" for doi, owner in missing.doi_owner.items()},
        SEED_SOURCE_TYPE,
    )

    stats = SeedStats(
        genes_considered=len(hgnc_ids),
        genes_on_target_panel=missing.genes_on_target_panel,
        publications_cited=missing.publications_cited,
        already_present=missing.already_present,
        added=outcome.added,
        already_held=outcome.already_held,
        pmids_linked=outcome.pmids_linked,
        unresolved=outcome.unresolved,
    )
    logger.info(
        f"PanelApp seeding complete: {stats.added} paper(s) added, {stats.already_held} "
        f"resolved to a paper already held ({stats.pmids_linked} given their PMID), "
        f"{stats.unresolved} unresolved"
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


def snapshot_publications(panels: dict[int, dict[str, Any]]) -> PanelPublications:
    """Every publication the snapshot's gene and repeat-expansion entries cite, on any
    panel: each entry's publication list and, for gene entries, each review's.

    Region entries (copy-number variants) are left out. Repeat-expansion entries
    have no reviews in a snapshot, and a gene entry whose reviews PanelApp could
    not serve has none either; their entries' own publications count all the same.
    """
    pmids: set[int] = set()
    dois: set[str] = set()

    def add(publications: list[str]) -> None:
        cited = clean_panel_publications(publications)
        pmids.update(cited.pmids)
        dois.update(cited.dois)

    for panel in panels.values():
        for entry in panel["genes"] + panel["strs"]:
            add(entry["publications"])
        for entry in panel["genes"]:
            for review in entry["evaluations"] or []:
                # PanelApp gives some reviews' publications as null.
                add(review["publications"] or [])
    return PanelPublications(pmids=pmids, dois=dois)


@dataclass
class CitationIntakeStats:
    """Outcome of taking a snapshot's citations into a database.

    `publications_cited` counts distinct PMIDs plus distinct DOIs (case-insensitive),
    so a paper PanelApp cites by its PMID in one place and its DOI in another counts
    twice; the second identifier then ends up in `already_present` or `already_held`.
    """

    publications_cited: int
    already_present: int
    added: int
    already_held: int
    pmids_linked: int
    unresolved: int


def ingest_snapshot_citations(
    db_path: Path, panels: dict[int, dict[str, Any]], panel_date: str
) -> CitationIntakeStats:
    """Add every publication the snapshot of *panel_date* cites that *db_path* does not
    hold, as metadata only, with source_details "panelapp-snapshot:{panel_date}".

    A paper is held when the database has its PMID or, case-insensitively, its DOI,
    so a rerun adds only what is new or failed to resolve before. A DOI PanelApp
    cites in several letter cases is fetched once, under one of its cited spellings.
    """
    cited = snapshot_publications(panels)
    cited_dois = _by_lowercase(cited.dois)
    held = held_identifiers(db_path)
    missing_pmids = cited.pmids - held.pmids
    missing_dois = [doi for lowered, doi in cited_dois.items() if lowered not in held.dois]

    publications_cited = len(cited.pmids) + len(cited_dois)
    already_present = publications_cited - len(missing_pmids) - len(missing_dois)
    logger.info(
        f"PanelApp cites {publications_cited} publication(s) in the {panel_date} snapshot "
        f"({len(cited.pmids)} PMIDs, {len(cited_dois)} DOIs): {already_present} already "
        f"held, {len(missing_pmids) + len(missing_dois)} to fetch"
    )

    source_details = f"{CITATION_SOURCE_DETAILS_PREFIX}:{panel_date}"
    outcome = store_publications(
        db_path,
        held.dois,
        dict.fromkeys(missing_pmids, source_details),
        dict.fromkeys(missing_dois, source_details),
        CITATION_SOURCE_TYPE,
    )

    stats = CitationIntakeStats(
        publications_cited=publications_cited,
        already_present=already_present,
        added=outcome.added,
        already_held=outcome.already_held,
        pmids_linked=outcome.pmids_linked,
        unresolved=outcome.unresolved,
    )
    logger.info(
        f"PanelApp citations: {stats.publications_cited} cited, {stats.already_present} "
        f"already present, {stats.added} added, {stats.already_held} resolved to a paper "
        f"already held ({stats.pmids_linked} given their PMID), {stats.unresolved} unresolved"
    )
    return stats
