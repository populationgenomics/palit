"""Tests for aggregate evidence loading and citation handling."""

import json
import sqlite3
from pathlib import Path
from typing import Any

from palit.assess_genes import PaperBatchProcessor, prune_invalid_citations

ROOT = Path(__file__).resolve().parents[1]
AARS1 = 20


def _extraction(hgnc_id: int, paper_gene_symbol: str) -> str:
    return json.dumps(
        {
            "genome_build": "unknown",
            "gene_evaluations": [
                {
                    "hgnc_id": hgnc_id,
                    "paper_gene_symbol": paper_gene_symbol,
                    "summary": "s",
                    "variants": [],
                    "disease_entities": [{"description": "d"}],
                    "quality_concerns": [],
                }
            ],
        }
    )


def test_evidence_for_gene_lists_each_paper_once_with_extraction_symbol(tmp_path: Path) -> None:
    """A relevance mention under an old symbol neither duplicates the paper nor adds an alias."""
    db_path = tmp_path / "run.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript((ROOT / "schema.sql").read_text())
        conn.execute(
            "INSERT INTO papers (doi, title, source, source_type, evidence_extraction_json) "
            "VALUES ('10.1/a', 't', 'pubmed', 'initial', ?)",
            (_extraction(AARS1, "AARS1"),),
        )
        conn.executemany(
            "INSERT INTO gene_mentions (hgnc_id, paper_gene_symbol, paper_doi, source) "
            "VALUES (?, ?, '10.1/a', ?)",
            [(AARS1, "AARS", "relevance_assessment"), (AARS1, "AARS1", "recent_evidence")],
        )

    evidence = PaperBatchProcessor(db_path).get_evidence_for_gene(AARS1)

    assert [(e["doi"], e["paper_gene_symbol"]) for e in evidence] == [("10.1/a", "AARS1")]


def test_prune_invalid_citations_keeps_extraction_quotes() -> None:
    assessment: dict[str, Any] = {
        "disease_entities": [
            {
                "citations": [{"doi": "10.1/a", "quote": "exact quote", "commentary": "c"}],
                "evidence_assessments": [
                    {
                        "name": "criterion_A",
                        "citations": [{"doi": "10.1/a", "quote": "paraphrase", "commentary": "c"}],
                    }
                ],
            }
        ],
        "quality_concerns": [
            {
                "concern": "x",
                "citations": [{"doi": "10.1/b", "quote": "exact quote", "commentary": "c"}],
            }
        ],
    }
    total, removed = prune_invalid_citations(
        assessment, {"10.1/a": {"exact quote"}, "10.1/b": set()}
    )
    assert total == 3
    assert removed == [("10.1/a", "paraphrase"), ("10.1/b", "exact quote")]
    assert assessment["disease_entities"][0]["citations"][0]["quote"] == "exact quote"
    assert assessment["disease_entities"][0]["evidence_assessments"][0]["citations"] == []
    assert assessment["quality_concerns"][0]["citations"] == []
