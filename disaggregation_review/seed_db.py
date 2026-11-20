#!/usr/bin/env python3
"""Seed the run database of the disaggregation review.

The review reads eleven Mendeliome genes in full, to show what a curator would
check if PanelApp Australia split its gene entries into gene-disease-MoI
associations. Its corpus is the union of three sets, for these genes only:

- the papers of the earlier pilot database;
- every publication PanelApp Australia cites for the gene, on any panel;
- every publication ClinGen cites for the gene: the PMIDs of its curations in the
  GenCC export, and those its evidence summaries give after "PMID".

Paper metadata comes from the screening baseline where it holds the paper, and
from PubMed or CrossRef otherwise. Every paper is stored as a relevant initial
paper, so the run reads all of them and none counts as older literature;
``source_details`` keeps where each came from.
"""

import csv
import json
import logging
import re
import sqlite3
from contextlib import closing
from pathlib import Path

import typer

from palit.clingen import fetch_clingen
from palit.hgnc import HgncResolver
from palit.panelapp_client import PanelAppClient, PanelPublications
from palit.panelapp_publications import (
    PaperWriter,
    baseline_papers,
    fetch_publications,
    gene_citations,
    held_identifiers,
    link_citations,
)
from palit.papers import Paper, deserialize_source_metadata

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROSTER_HGNC_IDS = [20, 5959, 4922, 6021, 29882, 6898, 9498, 6772, 11273, 12407, 16700]

RELEVANCE_JSON = json.dumps(
    {
        "relevant": True,
        "screen": {
            "rationale": "Seeded for the disaggregation review, without a relevance screen.",
            "relevant": True,
            "confidence": "HIGH",
            "associations": [],
        },
    }
)

# "PMID: 12145747", "PMIDs: 34953066, 32499606 and 27606499", "PMID 19608031"
_PMID_LIST = re.compile(r"PMIDs?:?\s*(\d{5,8}(?:(?:\s*[,;]\s*|\s+and\s+)\d{5,8})*)", re.IGNORECASE)


def pilot_papers(pilot_path: Path) -> dict[int, list[Paper]]:
    """The pilot's papers by the roster gene its relevance mentions link them to."""
    by_gene: dict[int, list[Paper]] = {}
    with closing(sqlite3.connect(f"file:{pilot_path}?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT DISTINCT gm.hgnc_id, p.doi, p.pmid, p.title, p.abstract, p.authors, p.journal,
                   p.source, p.source_date, p.source_metadata
            FROM gene_mentions gm JOIN papers p ON p.doi = gm.paper_doi
            WHERE gm.source = 'relevance_assessment'
            ORDER BY gm.hgnc_id, p.doi
            """
        ).fetchall()
    for row in rows:
        by_gene.setdefault(row["hgnc_id"], []).append(
            Paper(
                doi=row["doi"],
                pmid=row["pmid"],
                title=row["title"],
                abstract=row["abstract"],
                authors=row["authors"],
                journal=row["journal"],
                journal_issn=None,
                source=row["source"],
                source_date=row["source_date"],
                source_metadata=deserialize_source_metadata(row["source"], row["source_metadata"]),
                source_type="initial",
                source_details="pilot",
            )
        )
    return by_gene


def clingen_citations(data_dir: Path, gencc_path: Path) -> dict[int, PanelPublications]:
    """The PMIDs ClinGen cites for each roster gene that it has curated."""
    cited: dict[int, set[int]] = {}
    clingen = fetch_clingen(data_dir)
    for hgnc_id in ROSTER_HGNC_IDS:
        for curation in clingen.for_gene(hgnc_id):
            for match in _PMID_LIST.finditer(curation.evidence_summary):
                cited.setdefault(hgnc_id, set()).update(
                    int(pmid) for pmid in re.findall(r"\d+", match.group(1))
                )
    with gencc_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            hgnc_id = int(row["gene_curie"].removeprefix("HGNC:"))
            if row["submitter_title"] == "ClinGen" and hgnc_id in ROSTER_HGNC_IDS:
                cited.setdefault(hgnc_id, set()).update(
                    int(pmid) for pmid in row["submitted_as_pmids"].split(",") if pmid.strip()
                )
    return {hgnc_id: PanelPublications(pmids=pmids, dois=set()) for hgnc_id, pmids in cited.items()}


def add_cited(
    db_path: Path, baseline_path: Path, cited: dict[int, PanelPublications], prefix: str
) -> None:
    """Store the publications of *cited* that the database lacks, with the
    ``source_details`` ``{prefix}:{lowest HGNC ID citing the paper}``."""
    held = held_identifiers(db_path)
    pmid_details: dict[int, str] = {}
    doi_details: dict[str, str] = {}
    for hgnc_id in sorted(cited):
        for pmid in cited[hgnc_id].pmids - held.pmids:
            pmid_details.setdefault(pmid, f"{prefix}:{hgnc_id}")
        for doi in cited[hgnc_id].dois:
            if doi.lower() not in held.dois:
                doi_details.setdefault(doi, f"{prefix}:{hgnc_id}")
    in_baseline = baseline_papers(baseline_path, pmid_details, doi_details)
    writer = PaperWriter(db_path, held.dois, "initial")
    for pmid, paper in sorted(in_baseline.by_pmid.items()):
        writer.store(paper, pmid_details[pmid])
    for doi, paper in sorted(in_baseline.by_doi.items()):
        writer.store(paper, doi_details[doi])
    copied = writer.added
    unresolved = fetch_publications(
        writer,
        {p: d for p, d in pmid_details.items() if p not in in_baseline.by_pmid},
        {doi: d for doi, d in doi_details.items() if doi not in in_baseline.by_doi},
    )
    logger.info(
        "%s: %d identifiers not held, %d papers copied from the baseline, %d fetched, "
        "%d resolved to a held paper, %d unresolved",
        prefix,
        len(pmid_details) + len(doi_details),
        copied,
        writer.added - copied,
        writer.already_held,
        unresolved,
    )


def link_genes(
    db_path: Path, cited: dict[int, PanelPublications], hgnc_resolver: HgncResolver
) -> None:
    """Record each gene's held papers of *cited* as relevance mentions of the gene."""
    with sqlite3.connect(db_path) as conn:
        by_pmid: dict[int, list[str]] = {}
        by_doi: dict[str, str] = {}
        for doi, pmid in conn.execute("SELECT doi, pmid FROM papers"):
            by_doi[doi.lower()] = doi
            if pmid is not None:
                by_pmid.setdefault(pmid, []).append(doi)
        for hgnc_id, publications in cited.items():
            dois = {doi for pmid in publications.pmids for doi in by_pmid.get(pmid, [])}
            dois |= {by_doi[d.lower()] for d in publications.dois if d.lower() in by_doi}
            conn.executemany(
                "INSERT OR IGNORE INTO gene_mentions (hgnc_id, paper_gene_symbol, paper_doi, source) "
                "VALUES (?, ?, ?, 'relevance_assessment')",
                [(hgnc_id, hgnc_resolver.get_symbol(hgnc_id).upper(), doi) for doi in sorted(dois)],
            )


def main(
    db_path: Path = typer.Option(Path("data/db_disaggregation_review.sqlite")),
    pilot_path: Path = typer.Option(Path("data/db_disaggregated_pilot2.sqlite")),
    baseline_path: Path = typer.Option(Path("data/pubmed_baseline_screening.sqlite")),
    gencc_path: Path = typer.Option(Path("data/gencc_submissions.tsv")),
    schema_path: Path = typer.Option(Path("schema.sql")),
    panel_date: str = typer.Option(..., help="Date of the PanelApp snapshot (YYYY-MM-DD)"),
) -> None:
    """Create *db_path* and seed it with the review's corpus."""
    if db_path.exists():
        raise typer.BadParameter(f"{db_path} exists; move it away to seed afresh")
    with sqlite3.connect(db_path) as conn:
        conn.executescript(schema_path.read_text())
    hgnc_resolver = HgncResolver.from_file()

    pilot = pilot_papers(pilot_path)
    unknown = set(pilot) - set(ROSTER_HGNC_IDS)
    if unknown:
        raise ValueError(f"The pilot links papers to genes outside the roster: {sorted(unknown)}")
    pilot_cited = {
        hgnc_id: PanelPublications(pmids=set(), dois={p.doi for p in papers})
        for hgnc_id, papers in pilot.items()
    }
    in_baseline = baseline_papers(
        baseline_path, [], {doi for cited in pilot_cited.values() for doi in cited.dois}
    )
    writer = PaperWriter(db_path, {}, "initial")
    for hgnc_id, papers in sorted(pilot.items()):
        for paper in papers:
            writer.store(in_baseline.by_doi.get(paper.doi, paper), f"pilot:{hgnc_id}")
    logger.info(
        "pilot: %d papers stored, %d of them with the baseline's metadata",
        writer.added,
        len(in_baseline.by_doi),
    )

    panels = PanelAppClient(panel_date).get_all_panel_data()
    panelapp_cited = gene_citations(panels, ROSTER_HGNC_IDS)
    add_cited(db_path, baseline_path, panelapp_cited, "panelapp")
    clingen_cited = clingen_citations(db_path.parent, gencc_path)
    add_cited(db_path, baseline_path, clingen_cited, "clingen")

    for cited in (pilot_cited, panelapp_cited, clingen_cited):
        link_genes(db_path, cited, hgnc_resolver)
    links = link_citations(db_path, panelapp_cited)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE papers SET source_type = 'initial', relevance_assessment_json = ?",
            (RELEVANCE_JSON,),
        )
        papers, genes = conn.execute(
            "SELECT (SELECT COUNT(*) FROM papers), "
            "(SELECT COUNT(DISTINCT hgnc_id) FROM gene_mentions)"
        ).fetchone()
        per_gene = conn.execute(
            "SELECT hgnc_id, COUNT(*) FROM gene_mentions GROUP BY hgnc_id ORDER BY hgnc_id"
        ).fetchall()
    logger.info(
        "Seeded %d papers for %d genes; %d papers PanelApp cites (%d links)",
        papers,
        genes,
        links.papers,
        links.links,
    )
    for hgnc_id, count in per_gene:
        logger.info("  %s: %d papers", hgnc_resolver.get_symbol(hgnc_id), count)


if __name__ == "__main__":
    typer.run(main)
