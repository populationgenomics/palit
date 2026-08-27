"""Tests for the disaggregated prototype's corpus loading, import planning and seeding."""

import importlib.util
import json
import sqlite3
from pathlib import Path
from types import ModuleType

import pytest
from pypdf import PdfWriter

from palit.disaggregated_corpus import (
    ImportAction,
    classify_import,
    load_mockup_corpus,
    plan_pdf_imports,
    reconcile_fetched_papers,
)
from palit.hgnc import HgncResolver
from palit.papers import Paper, PubmedMetadata

SEED_DB_SCRIPT = Path("scripts/disaggregated_prototype/seed_db.py")

# (hgnc_id, symbol) for the genes used below.
_FIXTURE_GENES = [(20, "AARS1"), (4922, "HK1")]

# --- helpers ----------------------------------------------------------------


@pytest.fixture
def resolver(tmp_path: Path) -> HgncResolver:
    """An HgncResolver over a handful of genes, loaded through its real loader."""
    docs = [
        {
            "hgnc_id": f"HGNC:{hgnc_id}",
            "symbol": symbol,
            "prev_symbol": [],
            "alias_symbol": [],
            "locus_group": "protein-coding gene",
            "location": "16q22.1",
        }
        for hgnc_id, symbol in _FIXTURE_GENES
    ]
    path = tmp_path / "hgnc.json"
    path.write_text(json.dumps({"response": {"docs": docs}}))
    return HgncResolver.from_file(path)


def _write_pdf(path: Path, pages: int = 1) -> None:
    """Write a minimal valid PDF with the requested number of blank pages."""
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=72, height=72)
    with open(path, "wb") as f:
        writer.write(f)


def _write_content(
    path: Path,
    gene: str,
    associations: list[list[int]],
    *,
    unattributed: list[int] | None = None,
) -> None:
    """Write a mockup content file citing `associations` as per-association PMID groups."""
    path.write_text(
        json.dumps(
            {
                "gene": gene,
                "associations": [
                    {"publications": [{"pmid": str(pmid)} for pmid in group]}
                    for group in associations
                ],
                "unattributed": [{"pmid": str(pmid)} for pmid in (unattributed or [])],
            }
        )
    )


@pytest.fixture
def mockup(tmp_path: Path) -> tuple[Path, Path]:
    """A miniature mockup: three papers, one supplement, one paper cited by both genes."""
    fulltext_dir = tmp_path / "fulltext"
    content_dir = tmp_path / "content"
    fulltext_dir.mkdir()
    content_dir.mkdir()

    _write_pdf(fulltext_dir / "111.pdf", pages=2)
    _write_pdf(fulltext_dir / "111_supplement.pdf", pages=3)
    _write_pdf(fulltext_dir / "222.pdf")
    _write_pdf(fulltext_dir / "333.pdf")

    _write_content(content_dir / "AARS1.json", "AARS1", [[111], [222]])
    _write_content(content_dir / "HK1.json", "HK1", [[222, 333]])

    return fulltext_dir, content_dir


def _new_db(path: Path) -> Path:
    conn = sqlite3.connect(path)
    try:
        conn.executescript(Path("schema.sql").read_text())
    finally:
        conn.close()
    return path


def _seed_db_module() -> ModuleType:
    """Load the seeding script by path; scripts/ is not an importable package."""
    spec = importlib.util.spec_from_file_location("disaggregated_prototype_seed_db", SEED_DB_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _paper(pmid: int, doi: str) -> Paper:
    return Paper(
        doi=doi,
        pmid=pmid,
        title="A title",
        abstract="",
        authors="Doe, Jane",
        journal="J Test",
        source="pubmed",
        source_date="2024-01-01",
        source_metadata=PubmedMetadata(),
        source_type="expansion",
        source_details="test",
    )


# --- import classification --------------------------------------------------


@pytest.mark.parametrize(
    (
        "dest_exists",
        "dest_bytes_match",
        "dest_pages",
        "source_pages",
        "has_supplements",
        "expected",
    ),
    [
        (False, False, None, 0, False, ImportAction.COPY),
        (True, True, None, 0, False, ImportAction.SKIP_IDENTICAL),
        (True, False, None, 0, False, ImportAction.SKIP_MISMATCH),
        # Re-running an already-concatenated import must not rewrite the file:
        # pypdf output is not byte-stable, so identity is judged on page count.
        (True, False, 36, 36, True, ImportAction.SKIP_IDENTICAL),
        (True, False, 9, 36, True, ImportAction.CONCATENATE),
    ],
)
def test_classify_import_truth_table(
    dest_exists: bool,
    dest_bytes_match: bool,
    dest_pages: int | None,
    source_pages: int,
    has_supplements: bool,
    expected: ImportAction,
) -> None:
    assert (
        classify_import(
            dest_exists=dest_exists,
            dest_bytes_match=dest_bytes_match,
            dest_pages=dest_pages,
            source_pages=source_pages,
            has_supplements=has_supplements,
        )
        is expected
    )


# --- corpus loading ---------------------------------------------------------


def test_load_mockup_corpus_groups_supplements(
    mockup: tuple[Path, Path], resolver: HgncResolver
) -> None:
    fulltext_dir, content_dir = mockup
    corpus = load_mockup_corpus(fulltext_dir, content_dir, resolver)

    assert [paper.pmid for paper in corpus.papers] == [111, 222, 333]
    by_pmid = {paper.pmid: paper for paper in corpus.papers}
    assert by_pmid[111].supplements == (fulltext_dir / "111_supplement.pdf",)
    assert by_pmid[222].supplements == ()


def test_load_mockup_corpus_links_one_paper_to_both_genes(
    mockup: tuple[Path, Path], resolver: HgncResolver
) -> None:
    fulltext_dir, content_dir = mockup
    corpus = load_mockup_corpus(fulltext_dir, content_dir, resolver)

    assert sorted((link.hgnc_id, link.pmid) for link in corpus.links) == [
        (20, 111),
        (20, 222),
        (4922, 222),
        (4922, 333),
    ]
    assert {link.gene_symbol for link in corpus.links} == {"AARS1", "HK1"}


def test_load_mockup_corpus_dedupes_repeated_citations(
    mockup: tuple[Path, Path], resolver: HgncResolver
) -> None:
    fulltext_dir, content_dir = mockup
    # The same paper supporting two of a gene's associations is still one link.
    _write_content(content_dir / "AARS1.json", "AARS1", [[111, 222], [222]])

    corpus = load_mockup_corpus(fulltext_dir, content_dir, resolver)
    assert len([link for link in corpus.links if link.hgnc_id == 20]) == 2


def test_load_mockup_corpus_ignores_pmids_outside_associations(
    mockup: tuple[Path, Path], resolver: HgncResolver
) -> None:
    fulltext_dir, content_dir = mockup
    # 333 is only cited outside the associations, so nothing justifies its PDF.
    _write_content(content_dir / "HK1.json", "HK1", [[222]], unattributed=[333])

    with pytest.raises(ValueError, match=r"cited by no gene: \[333\]"):
        load_mockup_corpus(fulltext_dir, content_dir, resolver)


def test_load_mockup_corpus_rejects_pdf_without_gene_link(
    mockup: tuple[Path, Path], resolver: HgncResolver
) -> None:
    fulltext_dir, content_dir = mockup
    _write_pdf(fulltext_dir / "444.pdf")

    with pytest.raises(ValueError, match=r"cited by no gene: \[444\]"):
        load_mockup_corpus(fulltext_dir, content_dir, resolver)


def test_load_mockup_corpus_rejects_unresolvable_gene(
    mockup: tuple[Path, Path], resolver: HgncResolver
) -> None:
    fulltext_dir, content_dir = mockup
    _write_content(content_dir / "NOTAGENE.json", "NOTAGENE", [[222]])

    with pytest.raises(ValueError, match="NOTAGENE"):
        load_mockup_corpus(fulltext_dir, content_dir, resolver)


def test_load_mockup_corpus_rejects_gene_field_filename_mismatch(
    mockup: tuple[Path, Path], resolver: HgncResolver
) -> None:
    fulltext_dir, content_dir = mockup
    _write_content(content_dir / "AARS1.json", "HK1", [[111], [222]])

    with pytest.raises(ValueError, match="disagrees with its filename"):
        load_mockup_corpus(fulltext_dir, content_dir, resolver)


# --- import planning --------------------------------------------------------


def test_plan_pdf_imports_flags_stale_json_only_for_concatenation(
    mockup: tuple[Path, Path], resolver: HgncResolver, tmp_path: Path
) -> None:
    fulltext_dir, content_dir = mockup
    corpus = load_mockup_corpus(fulltext_dir, content_dir, resolver)
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()

    pmid_to_doi = {111: "10.1/a", 222: "10.1/b", 333: "10.1/c"}
    # A stale two-page rendition of the supplemented paper, plus a byte-identical
    # copy of an unsupplemented one.
    _write_pdf(papers_dir / "10.1%2Fa.pdf", pages=2)
    (papers_dir / "10.1%2Fa.json").write_text("{}")
    (papers_dir / "10.1%2Fb.pdf").write_bytes((fulltext_dir / "222.pdf").read_bytes())

    plans = {plan.pmid: plan for plan in plan_pdf_imports(corpus.papers, pmid_to_doi, papers_dir)}

    assert plans[111].action is ImportAction.CONCATENATE
    assert plans[111].stale_json == papers_dir / "10.1%2Fa.json"
    assert plans[111].sources == (fulltext_dir / "111.pdf", fulltext_dir / "111_supplement.pdf")
    assert plans[222].action is ImportAction.SKIP_IDENTICAL
    assert plans[222].stale_json is None
    assert plans[333].action is ImportAction.COPY
    assert plans[333].destination == papers_dir / "10.1%2Fc.pdf"


# --- fetch reconciliation ---------------------------------------------------


def test_reconcile_fetched_papers_happy_path() -> None:
    papers = [_paper(111, "10.1/a"), _paper(222, "10.1/b")]
    result = reconcile_fetched_papers([111, 222], papers, {111: "10.1/a"})

    assert result.dropped == ()
    assert result.doi_conflicts == ()
    assert result.fetched == tuple(papers)


def test_reconcile_fetched_papers_reports_dropped_pmids() -> None:
    result = reconcile_fetched_papers([111, 222, 333], [_paper(222, "10.1/b")], {})
    assert result.dropped == (111, 333)


def test_reconcile_fetched_papers_compares_dois_case_insensitively() -> None:
    papers = [_paper(111, "10.1/ABC"), _paper(222, "10.1/b")]
    result = reconcile_fetched_papers([111, 222], papers, {111: "10.1/abc", 222: "10.1/other"})

    assert result.doi_conflicts == ((222, "10.1/other", "10.1/b"),)


def test_reconcile_fetched_papers_rejects_paper_without_pmid() -> None:
    paper = _paper(111, "10.1/a")
    paper.pmid = None
    with pytest.raises(ValueError, match="without one"):
        reconcile_fetched_papers([111], [paper], {})


# --- baseline copy ----------------------------------------------------------


def test_copy_baseline_rows_selects_by_gene_or_mockup_pmid(tmp_path: Path) -> None:
    seed_db = _seed_db_module()
    baseline = _new_db(tmp_path / "baseline.sqlite")
    target = _new_db(tmp_path / "target.sqlite")

    with sqlite3.connect(baseline) as conn:
        conn.executemany(
            """
            INSERT INTO papers (doi, pmid, title, source, source_type, download_status)
            VALUES (?, ?, ?, 'pubmed', 'expansion', 'downloaded')
            """,
            [
                ("10.1/a", 111, "target gene only"),
                ("10.1/b", 222, "target plus off-target"),
                ("10.1/c", 333, "mockup pmid, off-target mentions only"),
                ("10.1/d", 444, "neither"),
            ],
        )
        conn.executemany(
            """
            INSERT INTO gene_mentions (hgnc_id, paper_gene_symbol, paper_doi, source)
            VALUES (?, ?, ?, 'relevance_assessment')
            """,
            [
                (20, "AARS1", "10.1/a"),
                (20, "AARS1", "10.1/b"),
                (9999, "OFFTARGET", "10.1/b"),
                (9999, "OFFTARGET", "10.1/c"),
                (9999, "OFFTARGET", "10.1/d"),
            ],
        )

    stats = seed_db.copy_baseline_rows(target, baseline, [20], [111, 222, 333])
    assert stats.papers_inserted == 3
    assert stats.mentions_inserted == 2

    with sqlite3.connect(target) as conn:
        assert [row[0] for row in conn.execute("SELECT doi FROM papers ORDER BY doi")] == [
            "10.1/a",
            "10.1/b",
            "10.1/c",
        ]
        assert conn.execute(
            "SELECT COUNT(*) FROM papers WHERE source_type = 'initial'"
        ).fetchone() == (3,)
        assert conn.execute(
            "SELECT COUNT(*) FROM papers WHERE download_status = 'scheduled'"
        ).fetchone() == (3,)
        assert conn.execute(
            "SELECT COUNT(*) FROM gene_mentions WHERE hgnc_id != 20"
        ).fetchone() == (0,)
