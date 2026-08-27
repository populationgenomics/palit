"""Reading the disaggregated-UI mockup corpus and planning its import.

The mockup ships a directory of PDFs named by PMID plus one JSON file per gene
describing that gene's disease associations. Together they define the paper set
the disaggregated prototype is seeded from. Everything here is a transform over
those inputs plus the filesystem state of `data/papers`; the database work lives
in the seeding script that drives these functions.
"""

import enum
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from palit.hgnc import HgncResolver
from palit.papers import Paper, doi_to_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MockupPaper:
    """One mockup full text: its main PDF plus any supplement PDFs."""

    pmid: int
    pdf: Path
    supplements: tuple[Path, ...]


@dataclass(frozen=True)
class MockupGeneLink:
    """A paper the mockup cites as evidence for one gene."""

    pmid: int
    hgnc_id: int
    gene_symbol: str


@dataclass(frozen=True)
class MockupCorpus:
    """The mockup's papers and the gene links that justify each one."""

    papers: tuple[MockupPaper, ...]
    links: tuple[MockupGeneLink, ...]


def _load_mockup_papers(fulltext_dir: Path) -> list[MockupPaper]:
    """Group the full-text directory into main PDFs and their supplements.

    A main PDF is named for its PMID alone; supplements carry a suffix after an
    underscore, so a purely numeric stem identifies the main file unambiguously.
    """
    papers: list[MockupPaper] = []
    for pdf in sorted(fulltext_dir.glob("*.pdf")):
        if not pdf.stem.isdigit():
            continue
        pmid = int(pdf.stem)
        supplements = tuple(sorted(fulltext_dir.glob(f"{pmid}_*.pdf")))
        papers.append(MockupPaper(pmid=pmid, pdf=pdf, supplements=supplements))
    return papers


def _load_mockup_links(
    content_dir: Path,
    pmids_with_pdf: set[int],
    resolver: HgncResolver,
) -> list[MockupGeneLink]:
    """Read the gene-paper links the mockup's content files assert.

    Only `associations[].publications` counts: the content files cite PMIDs in
    other places (audit trails, unattributed leftovers) that carry no claim that
    the paper is evidence for the gene. Papers without a mockup PDF are dropped
    because the corpus is defined by the full texts on disk.
    """
    links: list[MockupGeneLink] = []
    seen: set[tuple[int, int]] = set()

    for content_path in sorted(content_dir.glob("*.json")):
        data = json.loads(content_path.read_text())
        symbol: str = data["gene"]
        if symbol != content_path.stem:
            raise ValueError(
                f"{content_path} names gene {symbol!r}, which disagrees with its filename"
            )

        entry = resolver.resolve(symbol)
        if entry is None:
            raise ValueError(f"Mockup gene symbol does not resolve to HGNC: {symbol}")

        for association in data["associations"]:
            for publication in association["publications"]:
                pmid = int(publication["pmid"])
                if pmid not in pmids_with_pdf:
                    continue
                key = (entry.hgnc_id, pmid)
                if key in seen:
                    continue
                seen.add(key)
                links.append(
                    MockupGeneLink(pmid=pmid, hgnc_id=entry.hgnc_id, gene_symbol=entry.symbol)
                )

    return links


def load_mockup_corpus(
    fulltext_dir: Path,
    content_dir: Path,
    resolver: HgncResolver,
) -> MockupCorpus:
    """Read the mockup's full texts and the gene links that justify them.

    Every PDF must be cited by at least one gene: an unlinked full text would be
    seeded as a paper no gene ever looks at, which silently inflates the corpus.
    """
    papers = _load_mockup_papers(fulltext_dir)
    pmids_with_pdf = {paper.pmid for paper in papers}
    links = _load_mockup_links(content_dir, pmids_with_pdf, resolver)

    unlinked = sorted(pmids_with_pdf - {link.pmid for link in links})
    if unlinked:
        raise ValueError(f"Mockup PDFs cited by no gene: {unlinked}")

    logger.info(
        f"Loaded {len(papers)} mockup papers and {len(links)} gene links "
        f"over {len({link.hgnc_id for link in links})} genes"
    )
    return MockupCorpus(papers=tuple(papers), links=tuple(links))


class ImportAction(enum.Enum):
    """What importing one mockup full text into `data/papers` requires."""

    COPY = "copy"
    SKIP_IDENTICAL = "skip_identical"
    SKIP_MISMATCH = "skip_mismatch"
    CONCATENATE = "concatenate"


@dataclass(frozen=True)
class PdfImportPlan:
    """The decided import for one mockup full text."""

    pmid: int
    doi: str
    sources: tuple[Path, ...]
    destination: Path
    stale_json: Path | None
    action: ImportAction


def classify_import(
    *,
    dest_exists: bool,
    dest_bytes_match: bool,
    dest_pages: int | None,
    source_pages: int,
    has_supplements: bool,
) -> ImportAction:
    """Decide how one mockup full text reconciles with what `data/papers` holds.

    A paper without supplements is compared byte for byte, so a re-run of a copy
    is recognised exactly and a differing file is reported rather than
    overwritten — the existing PDF may be a better rendition than the mockup's.

    A paper with supplements is written by concatenation, whose output is not
    byte-stable across pypdf runs, so identity is judged on the page count of the
    destination against the summed page count of main plus supplements. That is
    what makes re-running the import idempotent instead of rewriting the file
    every time.
    """
    if not dest_exists:
        return ImportAction.COPY
    if not has_supplements:
        return ImportAction.SKIP_IDENTICAL if dest_bytes_match else ImportAction.SKIP_MISMATCH
    if dest_pages == source_pages:
        return ImportAction.SKIP_IDENTICAL
    return ImportAction.CONCATENATE


def _page_count(pdf: Path) -> int:
    return len(PdfReader(pdf).pages)


def plan_pdf_imports(
    papers: Sequence[MockupPaper],
    pmid_to_doi: Mapping[int, str],
    papers_dir: Path,
) -> list[PdfImportPlan]:
    """Decide the import action for every mockup paper against `papers_dir`."""
    plans: list[PdfImportPlan] = []

    for paper in papers:
        doi = pmid_to_doi[paper.pmid]
        destination = doi_to_path(doi, papers_dir, ".pdf")
        sources = (paper.pdf, *paper.supplements)
        has_supplements = bool(paper.supplements)

        dest_exists = destination.exists()
        dest_bytes_match = False
        dest_pages: int | None = None
        source_pages = 0
        if dest_exists:
            if has_supplements:
                dest_pages = _page_count(destination)
                source_pages = sum(_page_count(source) for source in sources)
            else:
                dest_bytes_match = destination.read_bytes() == paper.pdf.read_bytes()

        action = classify_import(
            dest_exists=dest_exists,
            dest_bytes_match=dest_bytes_match,
            dest_pages=dest_pages,
            source_pages=source_pages,
            has_supplements=has_supplements,
        )

        # Concatenation replaces the PDF the docling JSON was parsed from, so
        # that JSON no longer describes the file and has to be re-derived.
        stale_json = (
            doi_to_path(doi, papers_dir, ".json") if action is ImportAction.CONCATENATE else None
        )

        plans.append(
            PdfImportPlan(
                pmid=paper.pmid,
                doi=doi,
                sources=sources,
                destination=destination,
                stale_json=stale_json,
                action=action,
            )
        )

    return plans


@dataclass(frozen=True)
class FetchReconciliation:
    """What a PMID batch fetch returned, measured against what was asked for."""

    fetched: tuple[Paper, ...]
    dropped: tuple[int, ...]
    # (pmid, expected DOI, fetched DOI) for each disagreement.
    doi_conflicts: tuple[tuple[int, str, str], ...]


def reconcile_fetched_papers(
    requested: Sequence[int],
    papers: Sequence[Paper],
    pmid_to_doi: Mapping[int, str],
) -> FetchReconciliation:
    """Check a PMID batch fetch against the PMIDs and DOIs that were expected.

    PubMed drops records without a DOI, so the fetch can come back short; and the
    DOI it reports can disagree with the mockup's own PMID→DOI map, which would
    file the mockup's PDF under a DOI the mockup does not use. Both are reported
    for the caller to act on. DOIs compare case-insensitively because registrars
    treat DOI case as insignificant.
    """
    fetched_dois: dict[int, str] = {}
    for paper in papers:
        if paper.pmid is None:
            raise ValueError(f"Paper fetched by PMID came back without one: {paper.doi}")
        fetched_dois[paper.pmid] = paper.doi

    dropped = tuple(pmid for pmid in requested if pmid not in fetched_dois)

    conflicts: list[tuple[int, str, str]] = []
    for pmid, fetched_doi in fetched_dois.items():
        expected_doi = pmid_to_doi.get(pmid)
        if expected_doi is not None and expected_doi.lower() != fetched_doi.lower():
            conflicts.append((pmid, expected_doi, fetched_doi))

    return FetchReconciliation(
        fetched=tuple(papers),
        dropped=dropped,
        doi_conflicts=tuple(conflicts),
    )
