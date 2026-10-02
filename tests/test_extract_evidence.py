"""Tests for extraction helpers that need no network or PDFs."""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from palit.extract_evidence import (
    EVERY_REFUSAL,
    STAGE,
    extraction_quotes,
    normalize_extraction_genes,
    prune_citations,
    select_papers,
    structural_problems,
    variant_lookup_results,
)
from palit.hgnc import HgncEntry, HgncResolver
from palit.lookup_tools import frequency_row, summarise_variant_response
from palit.panelapp_integration import criteria_object_to_list

CRITERIA = ["criterion_A", "criterion_B", "criterion_C", "criterion_D", "criterion_E"]
SCHEMA_SQL = Path(__file__).resolve().parents[1] / "schema.sql"


def _citation(quote: str) -> dict[str, str]:
    return {"quote": quote, "commentary": "c"}


def _entity(family_count: int | None, independent: int | None) -> dict[str, Any]:
    return {
        "family_count": family_count,
        "independent_family_count": independent,
        "citations": [_citation("entity quote")],
        "evidence_assessments": [
            {"name": name, "result": False, "rationale": "r", "confidence": "LOW", "citations": []}
            for name in CRITERIA
        ],
    }


def _gene(symbol: str, entities: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "gene_symbol": symbol,
        "summary": "s",
        "variants": [{"variant": "c.1A>G", "quote": "variant quote"}],
        "disease_entities": entities,
        "quality_concerns": [{"concern": "x", "citations": [_citation("concern quote")]}],
    }


def test_extraction_quotes_covers_every_citation_site() -> None:
    extraction = {"gene_evaluations": [_gene("ABC1", [_entity(2, 2)])]}
    extraction["gene_evaluations"][0]["disease_entities"][0]["evidence_assessments"][0][
        "citations"
    ] = [_citation("criterion quote")]
    assert extraction_quotes(extraction) == [
        "variant quote",
        "entity quote",
        "criterion quote",
        "concern quote",
    ]


def test_structural_problems() -> None:
    assert structural_problems({"gene_evaluations": [_gene("ABC1", [_entity(2, 2)])]}) == []
    problems = structural_problems({"gene_evaluations": [_gene("PLACEHOLDER", [_entity(2, 3)])]})
    assert any("placeholder" in p for p in problems)
    assert any("independent_family_count" in p for p in problems)


def test_variant_lookup_results_reads_only_variant_tool_results() -> None:
    variant_result = {"gene_symbol": "ABC1", "variant": "c.1A>G", "status": "ok", "candidates": []}
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "paper"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t1", "name": "lookup_variants", "input": {}},
                {"type": "tool_use", "id": "t2", "name": "lookup_genes", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": json.dumps({"results": [variant_result]}),
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "t2",
                    "content": json.dumps({"results": [{"symbol": "ABC1"}]}),
                },
                {"type": "text", "text": "finalise"},
            ],
        },
    ]
    assert variant_lookup_results(messages) == [variant_result]


def test_frequency_row_picks_highest_ac_candidate() -> None:
    item = {"gene_symbol": "ABC1", "variant": "p.Arg1Cys", "genome_build": "unknown"}
    response = {
        "normalized": [
            {
                "hgvs_c": "c.1C>T",
                "hgvs_p": "p.Arg1Cys",
                "pseudo_vcf": "1-100-C-T",
                "frequency": {"ac": 3, "an": 10},
            },
            {
                "hgvs_c": "c.1C>A",
                "hgvs_p": "p.Arg1Cys",
                "pseudo_vcf": "1-100-C-A",
                "frequency": None,
            },
            {
                "hgvs_c": "c.1C>G",
                "hgvs_p": "p.Arg1Cys",
                "pseudo_vcf": "1-100-C-G",
                "frequency": {"ac": 7, "an": 10},
            },
        ]
    }
    row = frequency_row(summarise_variant_response(item, response))
    assert row.variant_id == "1-100-C-G"
    assert row.gnomad == {"ac": 7, "an": 10}
    assert row.normalization["total_normalizations"] == 3
    assert row.normalization["selected_for_max_ac"] == 7


def test_frequency_row_for_absent_and_failed_variants() -> None:
    item = {"gene_symbol": "ABC1", "variant": "c.1A>G", "genome_build": "GRCh38"}
    absent = frequency_row(
        summarise_variant_response(
            item,
            {
                "normalized": [
                    {"hgvs_c": "c.1A>G", "hgvs_p": None, "pseudo_vcf": "1-5-A-G", "frequency": None}
                ]
            },
        )
    )
    assert absent.gnomad == {"variant_not_found": True}
    assert absent.normalization["selected_for_max_ac"] is None

    failed = frequency_row(
        summarise_variant_response(
            item, {"error": {"code": "PARSE_ERROR", "message": "bad notation"}}
        )
    )
    assert failed.variant_id == "c.1A>G"
    assert failed.gnomad == {"normalization_error": True}
    assert failed.normalization["error_code"] == "PARSE_ERROR"


def test_normalize_extraction_genes_adds_hgnc_id() -> None:
    entry = HgncEntry(
        hgnc_id=8607,
        symbol="PRKN",
        name="parkin RBR E3 ubiquitin protein ligase",
        prev_symbols=("PARK2",),
        alias_symbols=(),
        locus_group="protein-coding gene",
        location="6q26",
        chromosome="6",
    )
    resolver = HgncResolver(
        _by_symbol={"PRKN": entry},
        _by_prev={"PARK2": [entry]},
        _by_alias={},
        _by_hgnc_id={8607: entry},
    )
    normalized, unresolved = normalize_extraction_genes(
        {"gene_evaluations": [{"gene_symbol": "PARK2", "summary": "PARK2 variants"}]}, resolver
    )
    gene = normalized["gene_evaluations"][0]
    assert gene["hgnc_id"] == 8607
    assert gene["summary"] == "PRKN variants"
    assert unresolved == []


def test_criteria_object_to_list_orders_and_names() -> None:
    entity: dict[str, Any] = {
        "evidence_assessments": {
            name: {"result": False, "rationale": name, "confidence": "LOW", "citations": []}
            for name in reversed(CRITERIA)
        }
    }
    criteria_object_to_list([entity])
    assert [c["name"] for c in entity["evidence_assessments"]] == CRITERIA
    assert entity["evidence_assessments"][0]["rationale"] == "criterion_A"


def test_prune_citations_keeps_variant_quotes() -> None:
    extraction = {"gene_evaluations": [_gene("ABC1", [_entity(2, 2)])]}
    removed = prune_citations(extraction, {"entity quote", "concern quote", "variant quote"})
    gene = extraction["gene_evaluations"][0]
    assert removed == 2
    assert gene["disease_entities"][0]["citations"] == []
    assert gene["quality_concerns"][0]["citations"] == []
    assert gene["variants"][0]["quote"] == "variant quote"


def test_select_papers_retries_only_refusals_before_the_cutoff(tmp_path: Path) -> None:
    """Refusals before the cutoff no longer exclude a paper; in-flight and later ones do."""
    db_path = tmp_path / "run.sqlite"
    before = datetime(2026, 9, 1, tzinfo=UTC)
    cutoff = datetime(2026, 9, 2, tzinfo=UTC)
    after = datetime(2026, 9, 3, tzinfo=UTC)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA_SQL.read_text())
        conn.executemany(
            "INSERT INTO papers (doi, title, source, source_type, download_status, "
            "evidence_extraction_json) VALUES (?, 't', 'pubmed', 'initial', 'downloaded', ?)",
            [
                ("10.1/new", None),
                ("10.1/refused-before", None),
                ("10.1/refused-after", None),
                ("10.1/pending", None),
                ("10.1/extracted", "{}"),
            ],
        )
        conn.executemany(
            "INSERT INTO llm_requests (custom_id, stage, subject, round, model, status, "
            "completed_at) VALUES (?, ?, ?, 1, 'm', ?, ?)",
            [
                ("a", STAGE, "10.1/refused-before", "refused", before.isoformat()),
                ("b", STAGE, "10.1/refused-after", "refused", after.isoformat()),
                ("c", STAGE, "10.1/pending", "pending", None),
                ("d", "assess_relevance", "10.1/new", "refused", before.isoformat()),
            ],
        )

    def selected(refusals_since: datetime) -> list[str]:
        return [p["doi"] for p in select_papers(db_path, None, refusals_since)]

    assert selected(EVERY_REFUSAL) == ["10.1/new"]
    assert selected(cutoff) == ["10.1/new", "10.1/refused-before"]
