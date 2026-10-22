#!/usr/bin/env python3
"""Add the publications PanelApp cites to a database: seeding a run's genes, and
taking every citation of a snapshot into the baseline.

Seeding gives each of the run's genes every publication PanelApp Australia cites
for it, on any panel: in its gene and repeat-expansion entries and in its gene
entries' reviews. Tournament selection optimises for a minimal, non-redundant
evidence set, so it routinely discards papers a PanelApp curator has already
cited, and the relevance screen may reject them; a run that never reads them
cannot say what is new. Seeding is unconditional:

- A cited paper the run does not hold is copied from the baseline when the
  baseline holds it, and fetched from PubMed or CrossRef otherwise. It is stored
  as an expansion paper with source_details "panelapp:{hgnc_id}", scheduled for
  download.
- Table ``panelapp_citations`` records which run genes cite each held paper, and
  a linked paper without a download status is scheduled. A cited paper of a run
  gene is in the corpus whatever its relevance (see :mod:`palit.run_corpus`).

The baseline (``data/pubmed_baseline_screening.sqlite``) takes in every
publication PanelApp Australia cites in a snapshot, on any panel, so that a
panel-scoped run, which copies the baseline's papers, starts from PanelApp's
citations as well as the screened literature, and so that seeding seldom needs
to fetch. The baseline holds these papers' metadata only; which genes cite them
is a run's concern.
"""

import logging
import sqlite3
from collections.abc import Collection, Iterator
from contextlib import closing
from dataclasses import dataclass, replace
from itertools import chain
from pathlib import Path
from typing import Any

from palit.discover_citations import (
    fetch_paper_metadata_by_doi,
    fetch_papers_by_pmids,
    store_referenced_paper,
)
from palit.hgnc import HgncResolver
from palit.panelapp_client import PanelPublications, clean_panel_publications
from palit.papers import Paper, deserialize_source_metadata
from palit.progress import LoggingProgress as Progress
from palit.run_corpus import run_genes

logger = logging.getLogger(__name__)

# The baseline: the screened PubMed literature and every publication PanelApp cites.
BASELINE_DB_PATH = Path("data/pubmed_baseline_screening.sqlite")

# PMIDs per efetch request. Keeps the fetch to a handful of round trips rather
# than one per paper, which matters at ~1.5k publications per run.
EFETCH_BATCH_SIZE = 200

# DOIs per query when reading papers from the baseline, below SQLite's limit on
# parameters.
QUERY_CHUNK_SIZE = 900

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


def _entry_publications(panel: dict[str, Any]) -> Iterator[tuple[dict[str, Any], list[str]]]:
    """Each gene and repeat-expansion entry of a snapshot panel, with the publication
    strings it cites: its own list and, for a gene entry, its reviews' lists.

    Region entries (copy-number variants) are left out. Repeat-expansion entries
    have no reviews in a snapshot, and a gene entry whose reviews PanelApp could
    not serve has none either; their entries' own publications count all the same.
    """
    for entry in panel["strs"]:
        yield entry, entry["publications"]
    for entry in panel["genes"]:
        # PanelApp gives some reviews' publications as null.
        reviews = [
            pub for review in entry["evaluations"] or [] for pub in review["publications"] or []
        ]
        yield entry, entry["publications"] + reviews


def snapshot_publications(panels: dict[int, dict[str, Any]]) -> PanelPublications:
    """Every publication the snapshot's gene and repeat-expansion entries cite, on any
    panel, in the entries and in the gene entries' reviews."""
    pmids: set[int] = set()
    dois: set[str] = set()
    for panel in panels.values():
        for _, publications in _entry_publications(panel):
            cited = clean_panel_publications(publications)
            pmids.update(cited.pmids)
            dois.update(cited.dois)
    return PanelPublications(pmids=pmids, dois=dois)


def gene_citations(
    panels: dict[int, dict[str, Any]], hgnc_ids: Collection[int]
) -> dict[int, PanelPublications]:
    """The publications the snapshot cites for each of *hgnc_ids*, on any panel, in
    the gene's entries and in its gene entries' reviews.

    Genes no panel holds are left out; a gene a panel holds without citing anything
    for it has empty publications.
    """
    wanted = set(hgnc_ids)
    cited: dict[int, PanelPublications] = {}
    for panel in panels.values():
        for entry, publications in _entry_publications(panel):
            hgnc_id = int(entry["gene_data"]["hgnc_id"].removeprefix("HGNC:"))
            if hgnc_id not in wanted:
                continue
            gene = cited.setdefault(hgnc_id, PanelPublications(pmids=set(), dois=set()))
            entry_cited = clean_panel_publications(publications)
            gene.pmids.update(entry_cited.pmids)
            gene.dois.update(entry_cited.dois)
    return cited


@dataclass
class _MissingPublications:
    """PanelApp identifiers absent from the database, each owned by one gene.

    A publication cited for several genes is fetched once and attributed to the
    lowest HGNC ID citing it; `panelapp_citations` links it to every gene citing
    it, so the owner only determines the recorded provenance. `doi_owner` holds
    one spelling of each DOI, as PanelApp cites it: the first met, in gene order
    and then sort order.
    """

    pmid_owner: dict[int, int]
    doi_owner: dict[str, int]
    publications_cited: int
    already_present: int


def collect_missing_publications(
    held: HeldIdentifiers, cited: dict[int, PanelPublications]
) -> _MissingPublications:
    """The publications in *cited* (by gene) that are not *held*."""
    pmid_owner: dict[int, int] = {}
    doi_owner: dict[str, int] = {}
    owned_dois: set[str] = set()  # doi_owner's DOIs, lowercased
    publications_cited = 0
    already_present = 0

    for hgnc_id in sorted(cited):
        pubs = cited[hgnc_id]
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
        publications_cited=publications_cited,
        already_present=already_present,
    )


def _link_pmid(db_path: Path, doi: str, pmid: int) -> bool:
    """Give the paper stored under *doi* the PMID *pmid* if it has none."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            "UPDATE papers SET pmid = ? WHERE doi = ? AND pmid IS NULL", (pmid, doi)
        )
        return cursor.rowcount > 0


class PaperWriter:
    """Stores papers in a database, each once, scheduled for download.

    A paper whose DOI the database holds, in any letter case, or that this writer
    stored before, is not stored again; if it has a PMID and the held row has
    none, the held row gets it, so a rerun matches the paper by PMID.
    """

    def __init__(self, db_path: Path, held_dois: dict[str, str], source_type: str) -> None:
        """*held_dois* are the database's DOIs as `HeldIdentifiers.dois` gives them."""
        self.db_path = db_path
        self.source_type = source_type
        self._held_dois = dict(held_dois)
        self.added = 0
        # Resolved to a paper whose DOI the database already holds: a paper cited by
        # both its PMID and its DOI, or held under another identifier or DOI case.
        self.already_held = 0
        # Of `already_held`, papers whose held row lacked a PMID and now has it.
        self.pmids_linked = 0

    def holds(self, doi: str) -> bool:
        return doi.lower() in self._held_dois

    def store(self, paper: Paper, source_details: str) -> None:
        held_doi = self._held_dois.get(paper.doi.lower())
        if held_doi is not None:
            self.already_held += 1
            if paper.pmid is not None and _link_pmid(self.db_path, held_doi, paper.pmid):
                self.pmids_linked += 1
            return
        self._held_dois[paper.doi.lower()] = paper.doi
        stored = replace(paper, source_type=self.source_type, source_details=source_details)
        inserted = store_referenced_paper(self.db_path, stored)
        # _held_dois covers every DOI in the database, so the insert cannot conflict.
        assert inserted
        self.added += 1


def fetch_publications(
    writer: PaperWriter,
    pmid_source_details: dict[int, str],
    doi_source_details: dict[str, str],
) -> int:
    """Fetch the given publications and store them with their `source_details`; the
    number of identifiers that resolved to no paper.

    PMIDs are fetched from PubMed in batches, then DOIs from CrossRef one at a
    time; a DOI is fetched and stored as spelled in *doi_source_details*, unless
    *writer* holds it by then. Raises `PubmedFetchError` when PubMed cannot be
    reached; the papers stored until then stay stored.
    """
    resolved_pmids: set[int] = set()
    pmids = sorted(pmid_source_details)
    batches = [pmids[i : i + EFETCH_BATCH_SIZE] for i in range(0, len(pmids), EFETCH_BATCH_SIZE)]
    dois = sorted(doi_source_details)
    pmids_linked_before = writer.pmids_linked

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
                writer.store(paper, pmid_source_details[paper.pmid])

        task = progress.add_task("Fetching PanelApp DOIs", total=len(dois))
        unresolved_dois: list[str] = []
        for doi in dois:
            progress.update(task, advance=1)
            if writer.holds(doi):
                writer.already_held += 1
                continue
            doi_paper = fetch_paper_metadata_by_doi(doi, doi_source_details[doi])
            if doi_paper is None:
                unresolved_dois.append(doi)
                continue
            writer.store(doi_paper, doi_source_details[doi])

    # A PMID PubMed gives no usable record for (no DOI, or a record the extractor
    # skips) is routine for older papers and junk in PanelApp's free text, so the
    # identifiers are listed at DEBUG only. A DOI CrossRef could not resolve was
    # logged with the reason as it failed.
    unresolved_pmids = sorted(set(pmids) - resolved_pmids)
    if unresolved_pmids:
        logger.info(f"PubMed gave no usable record for {len(unresolved_pmids)} PanelApp PMID(s)")
        logger.debug(f"PMIDs without a usable record: {', '.join(map(str, unresolved_pmids))}")
    pmids_linked = writer.pmids_linked - pmids_linked_before
    if pmids_linked:
        logger.info(f"{pmids_linked} held paper(s) given the PMID PanelApp cites them by")
    if unresolved_dois:
        logger.info(f"{len(unresolved_dois)} PanelApp DOI(s) could not be resolved")
        logger.debug(f"Unresolved DOIs: {', '.join(unresolved_dois)}")

    return len(unresolved_pmids) + len(unresolved_dois)


@dataclass
class BaselinePapers:
    """The papers a baseline holds for cited identifiers, with its metadata."""

    by_pmid: dict[int, Paper]
    # Keyed by the DOI as cited; the paper carries the DOI as the baseline stores it.
    by_doi: dict[str, Paper]


def _baseline_paper(row: sqlite3.Row) -> Paper:
    return Paper(
        doi=row["doi"],
        pmid=row["pmid"],
        title=row["title"],
        abstract=row["abstract"],
        authors=row["authors"],
        journal=row["journal"],
        journal_issn=row["journal_issn"],
        source=row["source"],
        source_date=row["source_date"],
        source_metadata=deserialize_source_metadata(row["source"], row["source_metadata"]),
        source_type=row["source_type"],
        source_details=row["source_details"],
    )


def baseline_papers(
    baseline_path: Path, pmids: Collection[int], dois: Collection[str]
) -> BaselinePapers:
    """The papers *baseline_path* holds for *pmids* and *dois*, matching DOIs in any
    letter case. Of several papers matching one identifier, the first by DOI is taken.

    The baseline is opened read-only, since other commands may be writing to it."""
    wanted_pmids = set(pmids)
    wanted_dois = {doi.lower(): doi for doi in dois}
    pmid_doi: dict[int, str] = {}
    cited_doi: dict[str, str] = {}
    with closing(sqlite3.connect(f"file:{baseline_path}?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        for doi, pmid in conn.execute("SELECT doi, pmid FROM papers ORDER BY doi"):
            if pmid in wanted_pmids:
                pmid_doi.setdefault(pmid, doi)
            cited = wanted_dois.get(doi.lower())
            if cited is not None:
                cited_doi.setdefault(cited, doi)

        found = sorted(set(pmid_doi.values()) | set(cited_doi.values()))
        papers: dict[str, Paper] = {}
        for i in range(0, len(found), QUERY_CHUNK_SIZE):
            chunk = found[i : i + QUERY_CHUNK_SIZE]
            rows = conn.execute(
                f"""
                SELECT doi, pmid, title, abstract, authors, journal, journal_issn, source,
                       source_date, source_metadata, source_type, source_details
                FROM papers WHERE doi IN ({",".join("?" * len(chunk))})
                """,
                chunk,
            ).fetchall()
            papers.update((row["doi"], _baseline_paper(row)) for row in rows)

    return BaselinePapers(
        by_pmid={pmid: papers[doi] for pmid, doi in pmid_doi.items()},
        by_doi={cited: papers[doi] for cited, doi in cited_doi.items()},
    )


@dataclass
class CitationLinks:
    """Outcome of recording which genes cite which held papers."""

    links: int  # rows of panelapp_citations
    papers: int  # distinct papers linked
    scheduled: int  # linked papers that had no download status and were scheduled


def link_citations(db_path: Path, cited: dict[int, PanelPublications]) -> CitationLinks:
    """Replace `panelapp_citations` with a row per gene of *cited* and held paper it
    cites, by PMID or by DOI in any letter case, and schedule for download the linked
    papers without a download status."""
    with sqlite3.connect(db_path) as conn:
        by_pmid: dict[int, list[str]] = {}
        by_doi: dict[str, list[str]] = {}
        for doi, pmid in conn.execute("SELECT doi, pmid FROM papers"):
            by_doi.setdefault(doi.lower(), []).append(doi)
            if pmid is not None:
                by_pmid.setdefault(pmid, []).append(doi)

        links: set[tuple[int, str]] = set()
        for hgnc_id, pubs in cited.items():
            for pmid in pubs.pmids:
                links.update((hgnc_id, doi) for doi in by_pmid.get(pmid, []))
            for cited_doi in pubs.dois:
                links.update((hgnc_id, doi) for doi in by_doi.get(cited_doi.lower(), []))

        conn.execute("DELETE FROM panelapp_citations")
        conn.executemany(
            "INSERT INTO panelapp_citations (hgnc_id, paper_doi) VALUES (?, ?)", sorted(links)
        )
        scheduled = conn.execute(
            """
            UPDATE papers SET download_status = 'scheduled'
            WHERE download_status IS NULL
              AND doi IN (SELECT paper_doi FROM panelapp_citations)
            """
        ).rowcount
    return CitationLinks(
        links=len(links), papers=len({doi for _, doi in links}), scheduled=scheduled
    )


@dataclass
class SeedStats:
    """Outcome of a seeding pass."""

    genes_considered: int
    genes_on_panelapp: int  # of genes_considered, those some panel holds
    publications_cited: int
    already_present: int
    copied_from_baseline: int
    fetched: int
    already_held: int
    pmids_linked: int
    unresolved: int
    links: CitationLinks


def seed_panelapp_publications(
    db_path: Path,
    panels: dict[int, dict[str, Any]],
    baseline_path: Path,
    hgnc_resolver: HgncResolver,
) -> SeedStats:
    """Add every publication *panels* cite for this run's genes, on any panel, to the
    run's corpus, and record which genes cite each held paper.

    A paper the run lacks is copied from *baseline_path* when the baseline holds it
    and fetched otherwise. Added papers are expansion papers scheduled for download;
    a held paper a run gene cites is scheduled if it has no download status. The
    links are recorded before fetching as well as after, so the papers held and
    those copied from the baseline are linked and scheduled even when a fetch fails. Safe to rerun: held
    papers are not added again and the links are replaced.
    """
    with sqlite3.connect(db_path) as conn:
        hgnc_ids = run_genes(conn)
    logger.info(f"Seeding PanelApp publications for {len(hgnc_ids)} run gene(s)")

    cited = gene_citations(panels, hgnc_ids)
    held = held_identifiers(db_path)
    missing = collect_missing_publications(held, cited)
    pmid_source_details = {
        pmid: f"{SEED_SOURCE_DETAILS_PREFIX}:{owner}" for pmid, owner in missing.pmid_owner.items()
    }
    doi_source_details = {
        doi: f"{SEED_SOURCE_DETAILS_PREFIX}:{owner}" for doi, owner in missing.doi_owner.items()
    }
    in_baseline = baseline_papers(baseline_path, pmid_source_details, doi_source_details)
    in_baseline_papers = {
        paper.doi for paper in chain(in_baseline.by_pmid.values(), in_baseline.by_doi.values())
    }
    to_fetch = (len(pmid_source_details) - len(in_baseline.by_pmid)) + (
        len(doi_source_details) - len(in_baseline.by_doi)
    )
    logger.info(
        f"PanelApp cites {missing.publications_cited} publication(s) for {len(cited)} of "
        f"the run's genes: {missing.already_present} already held, "
        f"{len(in_baseline_papers)} paper(s) to copy from the baseline, "
        f"{to_fetch} identifier(s) to fetch"
    )

    writer = PaperWriter(db_path, held.dois, SEED_SOURCE_TYPE)
    for pmid, paper in sorted(in_baseline.by_pmid.items()):
        writer.store(paper, pmid_source_details[pmid])
    for doi, paper in sorted(in_baseline.by_doi.items()):
        writer.store(paper, doi_source_details[doi])
    copied_from_baseline = writer.added
    before_fetching = link_citations(db_path, cited)

    unresolved = fetch_publications(
        writer,
        {
            pmid: details
            for pmid, details in pmid_source_details.items()
            if pmid not in in_baseline.by_pmid
        },
        {
            doi: details
            for doi, details in doi_source_details.items()
            if doi not in in_baseline.by_doi
        },
    )
    after_fetching = link_citations(db_path, cited)
    links = replace(after_fetching, scheduled=before_fetching.scheduled + after_fetching.scheduled)

    stats = SeedStats(
        genes_considered=len(hgnc_ids),
        genes_on_panelapp=len(cited),
        publications_cited=missing.publications_cited,
        already_present=missing.already_present,
        copied_from_baseline=copied_from_baseline,
        fetched=writer.added - copied_from_baseline,
        already_held=writer.already_held,
        pmids_linked=writer.pmids_linked,
        unresolved=unresolved,
        links=links,
    )
    logger.info(
        f"PanelApp seeding for {stats.genes_considered} run gene(s) complete: "
        f"{stats.copied_from_baseline} paper(s) copied from the "
        f"baseline, {stats.fetched} fetched, {stats.already_held} resolved to a paper already "
        f"held ({stats.pmids_linked} given their PMID), {stats.unresolved} unresolved; "
        f"{links.papers} held paper(s) cited for the run's genes ({links.links} link(s)), "
        f"{links.scheduled} newly scheduled for download"
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
    writer = PaperWriter(db_path, held.dois, CITATION_SOURCE_TYPE)
    unresolved = fetch_publications(
        writer,
        dict.fromkeys(missing_pmids, source_details),
        dict.fromkeys(missing_dois, source_details),
    )

    stats = CitationIntakeStats(
        publications_cited=publications_cited,
        already_present=already_present,
        added=writer.added,
        already_held=writer.already_held,
        pmids_linked=writer.pmids_linked,
        unresolved=unresolved,
    )
    logger.info(
        f"PanelApp citations: {stats.publications_cited} cited, {stats.already_present} "
        f"already present, {stats.added} added, {stats.already_held} resolved to a paper "
        f"already held ({stats.pmids_linked} given their PMID), {stats.unresolved} unresolved"
    )
    return stats
