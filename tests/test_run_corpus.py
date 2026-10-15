"""Tests for the run's genes and corpus, as each stage after relevance derives them."""

import json
import sqlite3
from pathlib import Path

import pytest

from palit.discover_citations import extract_referenced_sources_from_db
from palit.download_papers import read_dois_from_db
from palit.expand_literature import genes_to_expand
from palit.extract_evidence import STAGE, Remaining, count_remaining, select_papers
from palit.llm import stage_history
from palit.reduce_literature import genes_to_reduce, papers_of_small_genes
from palit.run_corpus import CORPUS_PAPER, run_genes

ROOT = Path(__file__).resolve().parents[1]


def _relevance(relevant: bool) -> str:
    screen = {"relevant": relevant, "confidence": "HIGH", "rationale": "r", "associations": []}
    return json.dumps({"relevant": relevant, "screen": screen})


def _extraction(*hgnc_ids: int) -> str:
    """An extraction citing one earlier source for each gene."""
    return json.dumps(
        {
            "gene_evaluations": [
                {
                    "hgnc_id": hgnc_id,
                    "disease_entities": [
                        {
                            "previously_reported_sources": [
                                {"title": f"Earlier report for gene {hgnc_id}", "context": "c"}
                            ]
                        }
                    ],
                }
                for hgnc_id in hgnc_ids
            ]
        }
    )


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Gene 1 has recent evidence from a relevant and a not-relevant paper; gene 2 only
    from the not-relevant one; gene 3 only from an expansion paper; gene 5 only a
    relevance mention of the not-relevant paper."""
    path = tmp_path / "run.sqlite"
    with sqlite3.connect(path) as conn:
        conn.executescript((ROOT / "schema.sql").read_text())
        conn.executemany(
            "INSERT INTO papers (doi, title, source, source_type, relevance_assessment_json, "
            "evidence_extraction_json) VALUES (?, 't', 'pubmed', ?, ?, ?)",
            [
                ("10.1/relevant", "initial", _relevance(True), _extraction(1)),
                ("10.1/not-relevant", "initial", _relevance(False), _extraction(1, 2)),
                ("10.1/expansion", "expansion", None, _extraction(3)),
            ],
        )
        conn.executemany(
            "INSERT INTO gene_mentions (hgnc_id, paper_gene_symbol, paper_doi, source) "
            "VALUES (?, 'G', ?, ?)",
            [
                (1, "10.1/relevant", "relevance_assessment"),
                (1, "10.1/relevant", "recent_evidence"),
                (1, "10.1/not-relevant", "relevance_assessment"),
                (1, "10.1/not-relevant", "recent_evidence"),
                (2, "10.1/not-relevant", "relevance_assessment"),
                (2, "10.1/not-relevant", "recent_evidence"),
                (5, "10.1/not-relevant", "relevance_assessment"),
                (3, "10.1/expansion", "expansion_evidence"),
            ],
        )
    return path


def test_run_genes_have_recent_evidence_from_a_relevant_paper(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        assert run_genes(conn) == [1]


def test_expansion_covers_the_run_genes_without_a_tournament(db_path: Path) -> None:
    assert genes_to_expand(db_path) == [1]
    with sqlite3.connect(db_path) as conn:
        conn.execute("INSERT INTO tournament_results (hgnc_id) VALUES (1)")
    assert genes_to_expand(db_path) == []


def test_referenced_sources_are_attributed_to_run_genes_only(db_path: Path) -> None:
    """Gene 1's sources count from both of its papers; genes 2 and 3 are not run genes."""
    sources = extract_referenced_sources_from_db(db_path)
    assert sorted((s.hgnc_id, s.citing_doi) for s in sources) == [
        (1, "10.1/not-relevant"),
        (1, "10.1/relevant"),
    ]


def test_reduction_counts_only_relevant_papers(db_path: Path) -> None:
    assert genes_to_reduce(db_path, max_papers=0) == [(1, 1)]
    assert genes_to_reduce(db_path, max_papers=1) == []
    assert papers_of_small_genes(db_path, max_papers=1) == {"10.1/relevant"}


@pytest.fixture
def cited_db_path(tmp_path: Path) -> Path:
    """Gene 1 is the run's gene. PanelApp cites papers the screen rejected for gene 1
    and for gene 2, which is not a run gene. No paper has an extraction yet: the
    downloaded ones wait for one, the others for a manual download. A recent-evidence
    mention of gene 7 on the paper cited for gene 1 stands for what its extraction
    would record."""
    path = tmp_path / "run.sqlite"
    with sqlite3.connect(path) as conn:
        conn.executescript((ROOT / "schema.sql").read_text())
        conn.executemany(
            "INSERT INTO papers (doi, title, source, source_type, download_status, "
            "relevance_assessment_json) VALUES (?, 't', 'pubmed', ?, ?, ?)",
            [
                ("10.1/relevant", "initial", "downloaded", _relevance(True)),
                ("10.1/relevant-manual", "initial", "manual_required", _relevance(True)),
                ("10.1/cited", "initial", "downloaded", _relevance(False)),
                ("10.1/cited-manual", "initial", "manual_required", _relevance(False)),
                ("10.1/cited-other", "initial", "downloaded", _relevance(False)),
                ("10.1/cited-other-manual", "initial", "manual_required", _relevance(False)),
                ("10.1/not-relevant", "initial", "downloaded", _relevance(False)),
                ("10.1/expansion-manual", "expansion", "manual_required", None),
            ],
        )
        conn.executemany(
            "INSERT INTO gene_mentions (hgnc_id, paper_gene_symbol, paper_doi, source) "
            "VALUES (?, 'G', ?, 'recent_evidence')",
            [(1, "10.1/relevant"), (7, "10.1/cited")],
        )
        conn.executemany(
            "INSERT INTO panelapp_citations (hgnc_id, paper_doi) VALUES (?, ?)",
            [
                (1, "10.1/cited"),
                (1, "10.1/cited-manual"),
                (2, "10.1/cited-other"),
                (2, "10.1/cited-other-manual"),
            ],
        )
    return path


def test_the_corpus_admits_papers_cited_for_a_run_gene_whatever_relevance_said(
    cited_db_path: Path,
) -> None:
    with sqlite3.connect(cited_db_path) as conn:
        corpus = {row[0] for row in conn.execute(f"SELECT doi FROM papers p WHERE {CORPUS_PAPER}")}
        # A cited paper's extraction does not add its genes to the run.
        assert run_genes(conn) == [1]
    assert corpus == {
        "10.1/relevant",
        "10.1/relevant-manual",
        "10.1/cited",
        "10.1/cited-manual",
        "10.1/expansion-manual",
    }


def test_extraction_selects_the_corpus(cited_db_path: Path) -> None:
    with sqlite3.connect(cited_db_path) as conn:
        history = stage_history(conn, STAGE)
    selected = select_papers(cited_db_path, None, None, history, set())
    assert [p["doi"] for p in selected] == ["10.1/cited", "10.1/relevant"]
    assert count_remaining(cited_db_path) == Remaining(due=2, outside_corpus=2)


def test_expansion_downloads_cover_papers_cited_for_a_run_gene(cited_db_path: Path) -> None:
    assert read_dois_from_db(cited_db_path, expansion_only=True) == [
        "10.1/cited-manual",
        "10.1/expansion-manual",
    ]
    assert read_dois_from_db(cited_db_path) == [
        "10.1/cited-manual",
        "10.1/cited-other-manual",
        "10.1/expansion-manual",
        "10.1/relevant-manual",
    ]
