#!/usr/bin/env python3
"""Stage A of the disaggregated prototype: seed its working database.

The prototype curates ten genes against the disaggregated-UI mockup, so its
corpus is the union of two sets: every paper the screening baseline already
attributes to one of those genes, and every paper the mockup ships a full text
for. The first set arrives complete with relevance assessments and is copied
wholesale; the second is topped up from PubMed for the handful of PMIDs the
baseline never saw, and its PDFs are imported into `data/papers` so extraction
has local text to work from.

Papers are relabelled `source_type='initial'` and `download_status='scheduled'`,
which is what the download and extraction stages expect of a working database
forked from the baseline. Gene mentions are restricted to the ten target genes:
a baseline paper can mention many more, and carrying those along would pull
off-target genes into every downstream per-gene query.
"""

import argparse
import dataclasses
import json
import logging
import shutil
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from palit.disaggregated_corpus import (
    ImportAction,
    MockupCorpus,
    MockupGeneLink,
    MockupPaper,
    load_mockup_corpus,
    plan_pdf_imports,
    reconcile_fetched_papers,
)
from palit.discover_citations import fetch_papers_by_pmids, store_referenced_paper
from palit.download_papers import concatenate_pdfs
from palit.hgnc import HgncResolver
from palit.papers import doi_to_path
from palit.progress import LoggingProgress as Progress

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Provenance stamped on the papers this script pulls from PubMed itself.
SOURCE_DETAILS = "disaggregated_prototype:mockup_fulltext"

# The mockup is a fixed artefact; a different shape means the wrong directory
# was passed or the mockup changed under us, either of which invalidates the
# expectations the rest of the prototype is calibrated against.
EXPECTED_MOCKUP_PAPERS = 67
EXPECTED_MOCKUP_GENES = 10

# Copy the baseline's paper rows verbatim apart from the two columns a working
# database redefines. `evidence_extraction_*` and `bbox_mapping` are left unset:
# the baseline only ever screened for relevance, so they are NULL there anyway,
# and the prototype re-extracts everything against its own entity list.
COPY_PAPERS_SQL = """
INSERT OR IGNORE INTO papers
    (doi, pmid, title, abstract, authors, journal, source, source_date, source_metadata,
     source_type, source_details, download_status,
     relevance_assessment_raw, relevance_assessment_json)
SELECT p.doi, p.pmid, p.title, p.abstract, p.authors, p.journal, p.source, p.source_date,
       p.source_metadata, 'initial', p.source_details, 'scheduled',
       p.relevance_assessment_raw, p.relevance_assessment_json
FROM baseline.papers p
WHERE p.doi IN (
        SELECT paper_doi FROM baseline.gene_mentions
        WHERE hgnc_id IN (SELECT hgnc_id FROM target_genes)
      )
   OR p.pmid IN (SELECT pmid FROM mockup_pmids)
"""

# Joining against the new papers table rather than the baseline's keeps mentions
# in step with the papers actually copied, and the hgnc_id filter drops the
# off-target genes a copied paper also mentions.
COPY_MENTIONS_SQL = """
INSERT OR IGNORE INTO gene_mentions (hgnc_id, paper_gene_symbol, paper_doi, source)
SELECT gm.hgnc_id, gm.paper_gene_symbol, gm.paper_doi, gm.source
FROM baseline.gene_mentions gm
JOIN papers p ON p.doi = gm.paper_doi
WHERE gm.hgnc_id IN (SELECT hgnc_id FROM target_genes)
"""


@dataclass(frozen=True)
class BaselineCopy:
    """Row counts written by the baseline copy stage."""

    papers_inserted: int
    mentions_inserted: int


def init_target_db(db_path: Path, schema_path: Path, force: bool) -> None:
    """Create the working database from the schema, refusing to clobber silently."""
    if db_path.exists():
        if not force:
            raise FileExistsError(f"{db_path} already exists. Pass --force to overwrite.")
        logger.warning(f"Removing existing {db_path}")
        for suffix in ("", "-wal", "-shm"):
            db_path.with_name(db_path.name + suffix).unlink(missing_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.executescript(schema_path.read_text())
    logger.info(f"Initialized {db_path} from {schema_path}")


def copy_baseline_rows(
    db_path: Path,
    baseline_db: Path,
    hgnc_ids: Sequence[int],
    mockup_pmids: Sequence[int],
) -> BaselineCopy:
    """Copy the target genes' baseline papers plus their on-target gene mentions.

    A mockup paper whose baseline mentions are all off-target would otherwise be
    missed, so the paper selection also admits anything carrying a mockup PMID.
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute("ATTACH DATABASE ? AS baseline", (str(baseline_db),))

        conn.execute("CREATE TEMP TABLE target_genes (hgnc_id INTEGER PRIMARY KEY)")
        conn.executemany(
            "INSERT INTO target_genes (hgnc_id) VALUES (?)",
            [(hgnc_id,) for hgnc_id in hgnc_ids],
        )
        conn.execute("CREATE TEMP TABLE mockup_pmids (pmid INTEGER PRIMARY KEY)")
        conn.executemany(
            "INSERT INTO mockup_pmids (pmid) VALUES (?)",
            [(pmid,) for pmid in mockup_pmids],
        )

        papers_inserted = conn.execute(COPY_PAPERS_SQL).rowcount
        mentions_inserted = conn.execute(COPY_MENTIONS_SQL).rowcount

        conn.commit()
        conn.execute("DETACH DATABASE baseline")

    logger.info(
        f"Copied {papers_inserted} baseline papers and {mentions_inserted} gene mentions "
        f"for {len(hgnc_ids)} target genes"
    )
    return BaselineCopy(papers_inserted=papers_inserted, mentions_inserted=mentions_inserted)


def _doi_by_pmid(db_path: Path, pmids: Sequence[int]) -> dict[int, str]:
    """Look up the stored DOI of every requested PMID present in the database."""
    placeholders = ",".join("?" * len(pmids))
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            f"SELECT pmid, doi FROM papers WHERE pmid IN ({placeholders})", list(pmids)
        ).fetchall()
    return dict(rows)


def fetch_absent_mockup_papers(
    db_path: Path,
    absent_pmids: Sequence[int],
    pmid_to_doi: Mapping[int, str],
) -> int:
    """Fetch and store the mockup papers the baseline never screened.

    A PMID that comes back empty or under an unexpected DOI leaves the mockup's
    PDF unattached to any paper row, so both abort the seed rather than produce a
    database that quietly misses full texts.
    """
    papers = fetch_papers_by_pmids(absent_pmids, SOURCE_DETAILS)
    reconciliation = reconcile_fetched_papers(absent_pmids, papers, pmid_to_doi)

    if reconciliation.dropped:
        raise ValueError(f"PubMed returned no usable record for PMIDs: {reconciliation.dropped}")
    if reconciliation.doi_conflicts:
        raise ValueError(
            "Fetched DOIs disagree with the mockup's PMID→DOI map: "
            + ", ".join(
                f"PMID {pmid}: mockup {expected!r} vs PubMed {fetched!r}"
                for pmid, expected, fetched in reconciliation.doi_conflicts
            )
        )

    inserted = 0
    for paper in reconciliation.fetched:
        # fetch_papers_by_pmids stamps every paper as expansion literature; these
        # are primary papers of the prototype's corpus.
        seeded = dataclasses.replace(paper, source_type="initial", source_details=SOURCE_DETAILS)
        if store_referenced_paper(db_path, seeded):
            inserted += 1

    logger.info(f"Fetched {len(papers)} papers from PubMed, inserted {inserted}")
    return inserted


def insert_mockup_mentions(db_path: Path, links: Sequence[MockupGeneLink]) -> int:
    """Record each mockup gene-paper link as a relevance-assessment mention.

    The mockup's citations are curator-asserted evidence for the gene, which is
    the same claim a relevance assessment makes, so downstream stages that read
    `source = 'relevance_assessment'` pick them up without special-casing.
    """
    doi_by_pmid = _doi_by_pmid(db_path, [link.pmid for link in links])

    inserted = 0
    with sqlite3.connect(db_path) as conn:
        for link in links:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO gene_mentions
                    (hgnc_id, paper_gene_symbol, paper_doi, source)
                VALUES (?, ?, ?, 'relevance_assessment')
                """,
                (link.hgnc_id, link.gene_symbol, doi_by_pmid[link.pmid]),
            )
            inserted += cursor.rowcount
        conn.commit()

    logger.info(f"Inserted {inserted} new mentions from {len(links)} mockup gene links")
    return inserted


def import_mockup_pdfs(
    papers: Sequence[MockupPaper],
    pmid_to_doi: Mapping[int, str],
    papers_dir: Path,
) -> dict[ImportAction, int]:
    """Place every mockup full text under its DOI-derived name in `papers_dir`.

    A paper with supplements is written as one concatenated PDF, matching how
    downloaded papers with supplements are stored, and its docling JSON is
    dropped so the next conversion re-derives it from the new file.
    """
    plans = plan_pdf_imports(papers, pmid_to_doi, papers_dir)
    tally: dict[ImportAction, int] = dict.fromkeys(ImportAction, 0)

    with Progress() as progress:
        task = progress.add_task("Importing mockup PDFs...", total=len(plans))
        for plan in plans:
            match plan.action:
                case ImportAction.COPY:
                    if len(plan.sources) == 1:
                        shutil.copy2(plan.sources[0], plan.destination)
                    else:
                        concatenate_pdfs(list(plan.sources), plan.destination)
                case ImportAction.CONCATENATE:
                    concatenate_pdfs(list(plan.sources), plan.destination)
                    assert plan.stale_json is not None
                    plan.stale_json.unlink(missing_ok=True)
                case ImportAction.SKIP_MISMATCH:
                    logger.warning(
                        f"PMID {plan.pmid} ({plan.doi}): existing PDF differs from the mockup's, "
                        f"keeping {plan.destination}"
                    )
                case ImportAction.SKIP_IDENTICAL:
                    pass
            tally[plan.action] += 1
            progress.update(task, advance=1)

    return tally


def log_verification(
    db_path: Path,
    corpus: MockupCorpus,
    papers_dir: Path,
    tally: Mapping[ImportAction, int],
    resolver: HgncResolver,
) -> None:
    """Log the shape of the seeded database, with the invariants it must satisfy."""
    hgnc_ids = sorted({link.hgnc_id for link in corpus.links})
    placeholders = ",".join("?" * len(hgnc_ids))

    with sqlite3.connect(db_path) as conn:
        papers_total = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        by_source = conn.execute(
            "SELECT source, COUNT(*) FROM papers GROUP BY source ORDER BY COUNT(*) DESC"
        ).fetchall()
        initial = conn.execute(
            "SELECT COUNT(*) FROM papers WHERE source_type = 'initial'"
        ).fetchone()[0]
        scheduled = conn.execute(
            "SELECT COUNT(*) FROM papers WHERE download_status = 'scheduled'"
        ).fetchone()[0]
        with_relevance = conn.execute(
            "SELECT COUNT(*) FROM papers WHERE relevance_assessment_json IS NOT NULL"
        ).fetchone()[0]
        without_pmid = conn.execute("SELECT COUNT(*) FROM papers WHERE pmid IS NULL").fetchone()[0]
        empty_authors = conn.execute(
            "SELECT COUNT(*) FROM papers WHERE authors IS NULL OR authors = ''"
        ).fetchone()[0]
        relevance_mentions = conn.execute(
            "SELECT COUNT(*) FROM gene_mentions WHERE source = 'relevance_assessment'"
        ).fetchone()[0]
        distinct_genes = conn.execute(
            "SELECT COUNT(DISTINCT hgnc_id) FROM gene_mentions"
        ).fetchone()[0]
        off_target = conn.execute(
            f"SELECT COUNT(*) FROM gene_mentions WHERE hgnc_id NOT IN ({placeholders})",
            hgnc_ids,
        ).fetchone()[0]
        per_gene = conn.execute(
            """
            SELECT hgnc_id, COUNT(DISTINCT paper_doi)
            FROM gene_mentions
            GROUP BY hgnc_id
            ORDER BY COUNT(DISTINCT paper_doi) DESC, hgnc_id
            """
        ).fetchall()
        dois = [row[0] for row in conn.execute("SELECT doi FROM papers").fetchall()]

    pdfs_present = sum(1 for doi in dois if doi_to_path(doi, papers_dir, ".pdf").exists())
    json_present = sum(1 for doi in dois if doi_to_path(doi, papers_dir, ".json").exists())

    logger.info("=== seed verification ===")
    logger.info("papers total:                        %d", papers_total)
    for source, count in by_source:
        logger.info("  source=%-16s %d", source, count)
    logger.info("source_type='initial':               %d", initial)
    logger.info("download_status='scheduled':         %d", scheduled)
    logger.info("relevance_assessment_json set:       %d", with_relevance)
    logger.info("papers without a PMID:               %d", without_pmid)
    logger.info("papers with empty authors:           %d (must be 0)", empty_authors)
    logger.info("relevance_assessment mentions:       %d", relevance_mentions)
    logger.info("distinct genes mentioned:            %d", distinct_genes)
    logger.info("off-target mentions:                 %d (must be 0)", off_target)
    logger.info("papers per gene:")
    for hgnc_id, count in per_gene:
        logger.info("  %-10s HGNC:%-7d %d", resolver.get_symbol(hgnc_id), hgnc_id, count)
    logger.info("mockup PDF import actions:")
    for action in ImportAction:
        logger.info("  %-16s %d", action.value, tally[action])
    logger.info("local readiness over %d papers:", papers_total)
    logger.info("  PDF on disk:                       %d", pdfs_present)
    logger.info("  docling JSON on disk:              %d", json_present)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, required=True, help="Working database to create")
    parser.add_argument(
        "--baseline-db", type=Path, required=True, help="Screening baseline to copy papers from"
    )
    parser.add_argument("--schema-path", type=Path, default=Path("schema.sql"))
    parser.add_argument("--papers-dir", type=Path, default=Path("data/papers"))
    parser.add_argument(
        "--mockup-fulltext-dir", type=Path, required=True, help="Mockup PDFs named by PMID"
    )
    parser.add_argument(
        "--mockup-content-dir", type=Path, required=True, help="Mockup per-gene content JSON files"
    )
    parser.add_argument(
        "--mockup-pmid-doi",
        type=Path,
        required=True,
        help="Mockup PMID→DOI map, used to validate the DOIs PubMed reports",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing database")
    args = parser.parse_args()

    resolver = HgncResolver.from_file()
    corpus = load_mockup_corpus(args.mockup_fulltext_dir, args.mockup_content_dir, resolver)
    if len(corpus.papers) != EXPECTED_MOCKUP_PAPERS:
        raise ValueError(
            f"Expected {EXPECTED_MOCKUP_PAPERS} mockup papers, found {len(corpus.papers)}"
        )
    hgnc_ids = sorted({link.hgnc_id for link in corpus.links})
    if len(hgnc_ids) != EXPECTED_MOCKUP_GENES:
        raise ValueError(f"Expected {EXPECTED_MOCKUP_GENES} mockup genes, found {len(hgnc_ids)}")

    mockup_pmid_doi: dict[int, str] = {
        int(pmid): doi for pmid, doi in json.loads(args.mockup_pmid_doi.read_text()).items()
    }
    mockup_pmids = [paper.pmid for paper in corpus.papers]

    init_target_db(args.db_path, args.schema_path, args.force)
    copy_baseline_rows(args.db_path, args.baseline_db, hgnc_ids, mockup_pmids)

    present = _doi_by_pmid(args.db_path, mockup_pmids)
    absent_pmids = [pmid for pmid in mockup_pmids if pmid not in present]
    logger.info(f"{len(absent_pmids)} mockup PMIDs absent from the baseline: {absent_pmids}")
    if absent_pmids:
        fetch_absent_mockup_papers(args.db_path, absent_pmids, mockup_pmid_doi)

    insert_mockup_mentions(args.db_path, corpus.links)

    pmid_to_doi = _doi_by_pmid(args.db_path, mockup_pmids)
    missing = [pmid for pmid in mockup_pmids if pmid not in pmid_to_doi]
    if missing:
        raise ValueError(f"Mockup PMIDs still have no paper row: {missing}")
    tally = import_mockup_pdfs(corpus.papers, pmid_to_doi, args.papers_dir)

    log_verification(args.db_path, corpus, args.papers_dir, tally, resolver)


if __name__ == "__main__":
    main()
