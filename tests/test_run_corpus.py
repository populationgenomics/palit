"""Tests for the run's genes, as each stage after extraction derives them."""

import json
import sqlite3
from pathlib import Path

import pytest

from palit.discover_citations import extract_referenced_sources_from_db
from palit.expand_literature import genes_to_expand
from palit.reduce_literature import genes_to_reduce, papers_of_small_genes
from palit.run_corpus import run_genes

ROOT = Path(__file__).resolve().parents[1]


def _relevance(relevant: bool) -> str:
    return json.dumps({"relevant": relevant, "screen": None, "panelapp_check": None})


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
